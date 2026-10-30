//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/log_merge.h"

#include <algorithm>
#include <limits>

#include "test_util/sync_point.h"

namespace ROCKSDB_NAMESPACE {
namespace log {

WALIndexMerger::ReaderMember::ReaderMember(ReaderMemberOptions&& options)
    : wal_number_(options.wal_number),
      wal_recovery_mode_(options.wal_recovery_mode),
      stop_on_report_(options.stop_on_report),
      reporter_(options.outer_reporter, &reader_) {
  if (options.fragment_buffered) {
    reader_.reset(new FragmentBufferedReader(
        options.info_log, std::move(options.file), &reporter_, options.checksum,
        options.wal_number));
  } else {
    reader_.reset(new Reader(
        options.info_log, std::move(options.file), &reporter_, options.checksum,
        options.wal_number, options.track_and_verify_wals,
        options.stop_replay_for_corruption, options.min_wal_number_to_keep,
        options.observed_predecessor_wal_info));
  }
  terminal_.kind = Member::Pull::kFinishedClean;
}

WALIndexMerger::Member::PulledRecord WALIndexMerger::ReaderMember::PullNext() {
  if (done_) {
    return terminal_;
  }
  Slice record;
  checksum_ = 0;
  if (reader_->ReadRecord(&record, &scratch_, wal_recovery_mode_, &checksum_)) {
    if (stop_on_report_ && reporter_.reported_since_record()) {
      // The read reported corruption: recovery would break before processing
      // this record, so the member stops here and drops it. The swallowed
      // tail accounts for the gap it leaves.
      done_ = true;
      terminal_.kind = Member::Pull::kFinishedStop;
      return terminal_;
    }
    const bool swallowed_before = reporter_.reported_since_record();
    reporter_.NoteRecord();
    if (!seen_first_) {
      seen_first_ = true;
      corrupt_before_first_ = reporter_.saw_corruption();
      head_floor_ = reader_->GetWALIndexProgressBound();
    }
    if (swallowed_before && !swallow_pending_) {
      // Reports since the previous read: corruption was observed and skipped
      // on the way to this pull. The next positioned pull closes the span.
      swallow_pending_ = true;
      swallow_pending_lo_ = last_positioned_index_;
    }
    NoteProgress();
    const uint64_t idx = reader_->GetLastReadWALIndex();
    PulledRecord pulled;
    if (idx == 0) {
      pulled.kind = Member::Pull::kUnindexed;
      pulled.payload = record;
      unindexed_pulled_++;
      return pulled;
    }
    if (swallow_pending_) {
      swallowed_.emplace_back(swallow_pending_lo_, idx);
      swallow_pending_ = false;
    }
    last_positioned_index_ = idx;
    if (first_indexed_ == 0) {
      first_indexed_ = idx;
    }
    indexed_pulled_++;
    pulled.kind = Member::Pull::kRecord;
    pulled.wal_index = idx;
    pulled.payload = record;
    return pulled;
  }
  NoteProgress();
  PulledRecord terminal;
  if (reader_->hasReadError()) {
    terminal.kind = Member::Pull::kFailedIO;
  } else if (reader_->WALRecoveryStoppedBeforeEndOfFile()) {
    terminal.kind = Member::Pull::kFinishedStop;
  } else if (reader_->IsEOF()) {
    terminal.kind = Member::Pull::kFinishedClean;
  } else {
    // FragmentBufferedReader no-progress pull: retryable, not terminal.
    terminal.kind = Member::Pull::kIdle;
    return terminal;
  }
  done_ = true;
  terminal_ = terminal;
  return terminal;
}

bool WALIndexMerger::ReaderMember::HeadScanned() const {
  // A progress record closes the head, so reading one proves the head
  // metadata is complete even before any data arrives.
  return seen_first_ || done_ || progress_noted_;
}

bool WALIndexMerger::ReaderMember::HeadIntact() const {
  if (seen_first_) {
    return !corrupt_before_first_;
  }
  if (progress_noted_) {
    return !corrupt_before_progress_;
  }
  if (done_) {
    // No data record survived. A stop at a truncated end of file, or at a
    // record from a recycled file's previous use, leaves nothing above it in
    // this file: a lost tail, whose reports still reach the recovery-mode
    // mapping. Corruption read past before the stop may have dropped data,
    // and so may a recycled file's silent stop, which the reader also takes
    // at a checksum mismatch: both are a damaged head.
    if (reader_->IsRecycled() && reader_->StoppedAtOldRecord() &&
        !reader_->MayHaveDroppedOwnRecord()) {
      // Stopped at the previous use's records, and every report on the way
      // dropped only bytes of that use: nothing of this file's was lost.
      return true;
    }
    const bool silent_recycled_stop =
        reader_->WALRecoveryStoppedBeforeEndOfFile() && reader_->IsRecycled() &&
        !reporter_.saw_corruption() && !reporter_.saw_old();
    return !reporter_.saw_corruption_before_stop() && !silent_recycled_stop;
  }
  // Unscanned: provisional, the merge never asks yet.
  return true;
}

uint64_t WALIndexMerger::ReaderMember::ProgressBound() const {
  return reader_->GetWALIndexProgressBound();
}

uint64_t WALIndexMerger::ReaderMember::HeadFloor() const {
  return seen_first_ ? head_floor_ : reader_->GetWALIndexProgressBound();
}

bool WALIndexMerger::ReaderMember::VoidCovers(uint64_t index) const {
  return reader_->IsWALIndexVoidCovered(index);
}

bool WALIndexMerger::ReaderMember::Supersedes(uint64_t bearer,
                                              uint64_t index) const {
  return reader_->IsWALIndexSuperseded(bearer, index);
}

bool WALIndexMerger::ReaderMember::AttestedSpan(PredecessorSpan* span) const {
  return reader_->GetWALIndexPredecessorSpan(&span->wal_number, &span->first,
                                             &span->last);
}

void WALIndexMerger::ReaderMember::AttestedSpans(
    std::vector<PredecessorSpan>* spans) const {
  for (const auto& read : reader_->GetWALIndexPredecessorSpans()) {
    PredecessorSpan span;
    span.wal_number = read.wal_number;
    span.first = read.first;
    span.last = read.last;
    spans->push_back(span);
  }
}

bool WALIndexMerger::ReaderMember::SwallowedCovers(uint64_t index) const {
  for (const auto& [lo, hi] : swallowed_) {
    if (index > lo && index < hi) {
      return true;
    }
  }
  return false;
}

bool WALIndexMerger::ReaderMember::TailImpure() const {
  if (!done_ || first_indexed_ == 0) {
    // A member that never held an index has no tail to lose: corruption
    // before any indexed pull is head damage, which the lower-bound gate
    // already failed closed on or the frontier stops at.
    return false;
  }
  return terminal_.kind == Member::Pull::kFinishedStop ||
         reporter_.reported_since_record() || swallow_pending_;
}

bool WALIndexMerger::ReaderMember::TailReported() const {
  return done_ && (reporter_.reported_since_record() || swallow_pending_);
}

void WALIndexMerger::ReaderMember::NoteProgress() {
  if (!progress_noted_ && reader_->GetWALIndexProgressBound() != 0) {
    progress_noted_ = true;
    corrupt_before_progress_ = reporter_.saw_corruption();
  }
}

WALIndexMerger::WALIndexMerger(
    std::vector<std::unique_ptr<Member>> members,
    std::vector<uint64_t> expected_wals,
    std::function<Status(Slice, SequenceNumber*)> sequence_of,
    bool continue_past_discarded_holes, bool continue_past_unexplained_holes)
    : members_(std::move(members)),
      expected_wals_(expected_wals.begin(), expected_wals.end()),
      sequence_of_(std::move(sequence_of)),
      continue_past_discarded_holes_(continue_past_discarded_holes),
      continue_past_unexplained_holes_(continue_past_unexplained_holes) {
  for (auto& m : members_) {
    states_.emplace_back(m.get());
  }
}

WALIndexMerger::MemberEnd WALIndexMerger::GetMemberEnd(
    uint64_t wal_number) const {
  for (const auto& s : states_) {
    if (s.member->wal_number() == wal_number) {
      if (!s.finished) {
        return MemberEnd::kUnfinished;
      }
      return s.finished_clean ? MemberEnd::kClean : MemberEnd::kStop;
    }
  }
  return MemberEnd::kUnfinished;
}

WALIndexMerger::MemberState* WALIndexMerger::FindState(uint64_t wal_number) {
  for (auto& s : states_) {
    if (s.member->wal_number() == wal_number) {
      return &s;
    }
  }
  return nullptr;
}

namespace {
// Indexed data from a file whose intact head shows no marker: the writer
// emits the marker first, so it would have survived. Shared by the
// lower-bound check and the late-data check so both fail identically.
Status IndexedWithoutMarker(uint64_t wal_number) {
  return Status::Corruption("WAL index merge: WAL " +
                            std::to_string(wal_number) +
                            " bears indexed data without a WAL index marker; "
                            "the marker is the file's first record, so an "
                            "intact head would show it");
}
}  // namespace

Status WALIndexMerger::CheckMemberCoherence(const MemberState& state) {
  const Member* member = state.member;
  const uint64_t indexed = member->IndexedDataPulled();
  if (indexed == 0 && member->UnindexedDataPulled() == 0) {
    return Status::OK();
  }
  const uint64_t wal_number = member->wal_number();
  if (!member->SawWALIndexMarker()) {
    if (indexed == 0) {
      // No marker and no index anywhere: the file has no position in the
      // merged order. Head integrity tells a file that predates the option
      // apart from one whose marker was lost to damage.
      return Status::Corruption(
          "WAL index merge: WAL " + std::to_string(wal_number) +
          " carries no WAL index (no marker; " +
          (member->HeadIntact()
               ? "head intact, predates partition_wal_usage"
               : "head damaged, a lost marker cannot be ruled out") +
          "); an unindexed file cannot be ordered in the merge");
    }
    if (lower_bound_set_ && member->HeadIntact()) {
      return IndexedWithoutMarker(wal_number);
    }
    // Before the lower bound, indexed data without a marker waits for the
    // attestation checks in ComputeLowerBound, which own the
    // marker-lost-to-damage cases.
    return Status::OK();
  }
  if (state.finished && indexed == 0 && member->UnindexedDataPulled() > 0) {
    // The marker promised indices and every data record broke the promise.
    // A marker-only file stays legal: it promised nothing about data.
    return Status::Corruption("WAL index merge: WAL " +
                              std::to_string(wal_number) +
                              " carries a WAL index marker but its data "
                              "records carry no wal_index");
  }
  return Status::OK();
}

Status WALIndexMerger::PullMembers(uint64_t threshold, PullOutcome* outcome) {
  for (auto& s : states_) {
    if (s.finished) {
      continue;
    }
    for (;;) {
      const uint64_t bound = s.member->ProgressBound();
      if (bound > s.progress) {
        s.progress = bound;
        outcome->advanced = true;
      }
      if (s.has_lookahead && s.lookahead.wal_index >= threshold) {
        break;
      }
      // A bound past the threshold clears the member: nothing below it is
      // still coming. A bound AT the threshold does not: the threshold
      // itself may still arrive.
      if (s.progress > threshold) {
        break;
      }
      Member::PulledRecord pulled = s.member->PullNext();
      if (pulled.kind == Member::Pull::kRecord) {
        if (pulled.wal_index < s.member->ProgressBound()) {
          return Status::Corruption(
              "WAL index merge: WAL " + std::to_string(s.member->wal_number()) +
              " produced index " + std::to_string(pulled.wal_index) +
              " below its asserted progress bound");
        }
        if (pulled.wal_index < threshold) {
          return Status::Corruption(
              "WAL index merge: WAL " + std::to_string(s.member->wal_number()) +
              " produced index " + std::to_string(pulled.wal_index) +
              " below expectation " + std::to_string(threshold));
        }
        s.max_seen = std::max(s.max_seen, pulled.wal_index);
        s.lookahead = pulled;
        s.has_lookahead = true;
        outcome->advanced = true;
        if (Status coherence = CheckMemberCoherence(s); !coherence.ok()) {
          return coherence;
        }
      } else if (pulled.kind == Member::Pull::kUnindexed) {
        unindexed_queue_.push_back({s.member, pulled.payload.ToString(),
                                    s.member->LastRecordChecksum()});
        outcome->advanced = true;
        if (Status coherence = CheckMemberCoherence(s); !coherence.ok()) {
          return coherence;
        }
      } else if (pulled.kind == Member::Pull::kFailedIO) {
        return Status::IOError("WAL index merge: WAL " +
                               std::to_string(s.member->wal_number()) +
                               " unreadable");
      } else if (pulled.kind == Member::Pull::kIdle) {
        // The idle pull may have scanned a progress record on the way out;
        // re-check the bound before blocking on this member.
        const uint64_t idle_bound = s.member->ProgressBound();
        if (idle_bound > s.progress) {
          s.progress = idle_bound;
          outcome->advanced = true;
        }
        if (s.progress > threshold) {
          break;
        }
        outcome->blocked = true;
        break;
      } else {
        s.finished = true;
        s.finished_clean = (pulled.kind == Member::Pull::kFinishedClean);
        const uint64_t end_bound = s.member->ProgressBound();
        if (end_bound > s.progress) {
          s.progress = end_bound;
        }
        outcome->advanced = true;
        if (Status coherence = CheckMemberCoherence(s); !coherence.ok()) {
          return coherence;
        }
        break;
      }
    }
  }
  return Status::OK();
}

Status WALIndexMerger::ComputeLowerBound() {
  // Fold attestations, taking the minimum on conflict: a lower bound
  // examines more indices, which is the fail-closed direction.
  std::map<uint64_t, std::pair<uint64_t, uint64_t>> attested;
  for (auto& s : states_) {
    std::vector<Member::PredecessorSpan> spans;
    s.member->AttestedSpans(&spans);
    for (const auto& span : spans) {
      const auto it = attested.find(span.wal_number);
      if (it == attested.end()) {
        attested.emplace(span.wal_number,
                         std::make_pair(span.first, span.last));
      } else {
        it->second.first = std::min(it->second.first, span.first);
        it->second.second = std::min(it->second.second, span.last);
      }
    }
  }
  for (const auto& [pred, span] : attested) {
    attested_wals_.insert(pred);
  }
  // A span anchors the bound unless it attests a held or lost predecessor
  // that bears no index.
  for (auto& s : states_) {
    std::vector<Member::PredecessorSpan> spans;
    s.member->AttestedSpans(&spans);
    for (const auto& span : spans) {
      if (span.first != 0 || (FindState(span.wal_number) == nullptr &&
                              span.wal_number < retirement_bound_)) {
        min_anchoring_wal_ =
            std::min(min_anchoring_wal_, s.member->wal_number());
        break;
      }
    }
  }
  for (auto& s : states_) {
    if (s.member->HeadIntact()) {
      continue;
    }
    // A damaged head can only be bounded by an attestation from elsewhere.
    if (!HeadBounded(s.member->wal_number())) {
      return Status::Corruption("WAL index merge: WAL " +
                                std::to_string(s.member->wal_number()) +
                                " head damaged and unattested; cannot bound");
    }
  }
  for (const auto& [pred, span] : attested) {
    MemberState* ps = FindState(pred);
    if (ps == nullptr) {
      continue;  // retired file: anchors the bound below, checked nowhere.
    }
    const uint64_t observed = ps->member->FirstIndexed();
    if (observed != 0 && observed < span.first) {
      return Status::Corruption("WAL index merge: WAL " + std::to_string(pred) +
                                " starts below its attested span");
    }
    if (observed != 0 && observed != span.first && ps->member->HeadIntact()) {
      // Two intact heads contradict each other: fail closed.
      return Status::Corruption("WAL index merge: WAL " + std::to_string(pred) +
                                " attested start mismatch");
    }
  }
  for (auto& s : states_) {
    // Indexed data without the marker that promises it, from a head intact
    // enough that the marker would have survived. Runs after the attestation
    // checks so damaged heads keep their existing verdicts: the marker may
    // have been lost to the damage, and the attestation owns that question.
    if (s.member->IndexedDataPulled() > 0 && !s.member->SawWALIndexMarker() &&
        s.member->HeadIntact()) {
      return IndexedWithoutMarker(s.member->wal_number());
    }
  }
  // Max doubles as "nothing anchors the bound" below, so track found-ness
  // explicitly: Max itself is a reachable (if only physically crafted) index.
  uint64_t lower = std::numeric_limits<uint64_t>::max();
  bool have_lower = false;
  for (auto& s : states_) {
    if (s.member->FirstIndexed() != 0) {
      lower = std::min(lower, s.member->FirstIndexed());
      have_lower = true;
    }
  }
  for (const auto& [pred, span] : attested) {
    if (FindState(pred) != nullptr || pred >= retirement_bound_) {
      // A zero start attests an empty predecessor and anchors nothing.
      if (span.first != 0) {
        lower = std::min(lower, span.first);
        have_lower = true;
      }
      if (FindState(pred) == nullptr && span.first != 0) {
        // Not retired and not held: the file is lost, and so is every index
        // it bore.
        if (span.first < lost_floor_) {
          lost_floor_ = span.first;
          lost_predecessor_ = pred;
        }
      }
    } else if (span.second != std::numeric_limits<uint64_t>::max()) {
      lower = std::min(lower, span.second + 1);
      have_lower = true;
      if (span.first != 0) {
        retired_spans_.emplace_back(span.first, span.second);
      }
    }
  }
  if (have_lower && history_floor_ != 0 && attested.empty()) {
    lower = std::min(lower, history_floor_);
  }
  // A head floor bounds the indices allocated since its WAL was created,
  // whatever lost WAL bore them: a partition's current WAL is attested by no
  // successor, so a loss below every surviving record surfaces as a hole only
  // through the floor.
  bool use_head_floors = true;
  // Lets the prefix oracle's G-20 self-test drop every lower-bound witness.
  TEST_SYNC_POINT_CALLBACK("WALIndexMerger::ComputeLowerBound:UseHeadFloors",
                           &use_head_floors);
  for (auto& s : states_) {
    const uint64_t floor = s.member->HeadFloor();
    if (have_lower && use_head_floors && floor != 0) {
      lower = std::min(lower, floor);
    }
  }
  if (!have_lower) {
    frontier_ = 0;
    terminal_ = Result::kCleanEnd;
    return Status::OK();
  }
  lower_ = lower;
  expected_ = lower;
  lower_bound_set_ = true;
  return Status::OK();
}

WALIndexMerger::Result WALIndexMerger::Next(Slice* record) {
  if (terminal_ != Result::kWaiting) {
    return terminal_;
  }
  if (queue_outstanding_) {
    unindexed_queue_.pop_front();
    queue_outstanding_ = false;
  }
  for (;;) {
    // Buffered unindexed records drain before any indexed decision so
    // encounter order is preserved.
    if (!unindexed_queue_.empty()) {
      *record = Slice(unindexed_queue_.front().payload);
      last_emitted_index_ = 0;
      last_emitted_member_ = unindexed_queue_.front().member;
      last_emitted_checksum_ = unindexed_queue_.front().checksum;
      queue_outstanding_ = true;
      return Result::kRecord;
    }
    if (lower_bound_set_ && expected_ == 0) {
      // The expectation wrapped past the type limit: only physically crafted
      // indices reach here, and recovery fails on the Max-valued emission
      // first. Fail closed rather than re-deciding index zero.
      return Fail(Status::Corruption(
          "WAL index merge: wal_index expectation overflowed"));
    }
    if (!membership_checked_) {
      membership_checked_ = true;
      std::set<uint64_t> held;
      for (auto& s : states_) {
        held.insert(s.member->wal_number());
      }
      for (uint64_t wal : expected_wals_) {
        if (held.count(wal) == 0) {
          return Fail(Status::Corruption("WAL index merge: WAL " +
                                         std::to_string(wal) +
                                         " named but not held; cannot bound"));
        }
      }
    }
    if (!lower_bound_set_) {
      PullOutcome outcome;
      Status s = PullMembers(1, &outcome);
      if (!s.ok()) {
        return Fail(s);
      }
      if (!unindexed_queue_.empty()) {
        continue;
      }
      if (outcome.blocked) {
        return Result::kWaiting;
      }
      s = ComputeLowerBound();
      if (!s.ok()) {
        return Fail(s);
      }
      if (terminal_ != Result::kWaiting) {
        return terminal_;  // kCleanEnd when nothing is indexed anywhere.
      }
      continue;
    }
    PullOutcome outcome;
    Status s = PullMembers(expected_, &outcome);
    if (!s.ok()) {
      return Fail(s);
    }
    if (!unindexed_queue_.empty()) {
      continue;
    }
    if (outcome.blocked && !outcome.advanced) {
      return Result::kWaiting;
    }
    if (outcome.blocked) {
      continue;  // others advanced; re-pull the blocked member.
    }
    // Decide expected_. Every unfinished member is past it or bounded past
    // it, and every member is head-scanned, so the void and supersession
    // unions are complete with respect to it.
    std::vector<MemberState*> holders;
    for (auto& state : states_) {
      if (!state.finished && state.has_lookahead &&
          state.lookahead.wal_index == expected_) {
        holders.push_back(&state);
      }
    }
    bool void_covered = false;
    for (auto& state : states_) {
      if (state.member->VoidCovers(expected_)) {
        void_covered = true;
        break;
      }
    }
    if (!holders.empty() && void_covered) {
      return Fail(Status::Corruption("WAL index merge: index " +
                                     std::to_string(expected_) +
                                     " bears records and a void cover"));
    }
    std::vector<MemberState*> live;
    for (auto* h : holders) {
      bool superseded = false;
      for (auto& state : states_) {
        if (state.member->Supersedes(h->member->wal_number(), expected_)) {
          superseded = true;
          break;
        }
      }
      if (!superseded) {
        live.push_back(h);
      }
    }
    if (live.size() > 1) {
      return Fail(
          Status::Corruption("WAL index merge: two live records share "
                             "index " +
                             std::to_string(expected_)));
    }
    if (live.size() == 1 && pending_discarded_ != 0) {
      // A live record above a skipped superseded-only index: writes after the
      // cut reissue indices in order, so the reissued copy was lost.
      return StopAtFrontier();
    }
    if (live.size() == 1) {
      MemberState* winner = live[0];
      SequenceNumber seq = 0;
      s = sequence_of_(winner->lookahead.payload, &seq);
      if (!s.ok()) {
        return Fail(Status::Corruption("WAL index merge: index " +
                                       std::to_string(expected_) + ": " +
                                       s.ToString()));
      }
      if (seq == 0) {
        // An empty record occupies its index but carries no witness: emit
        // it without advancing the sequence check. Recovery drops it.
      } else if (seq < last_sequence_) {
        // Non-decreasing, not strictly increasing: an empty batch consumes
        // an index but no sequence, so the recovery dummy and a user-written
        // empty batch both repeat the sequence the next record reuses.
        return Fail(Status::Corruption("WAL index merge: index " +
                                       std::to_string(expected_) +
                                       " does not advance the sequence"));
      } else {
        last_sequence_ = seq;
      }
      *record = winner->lookahead.payload;
      last_emitted_index_ = expected_;
      last_emitted_member_ = winner->member;
      last_emitted_checksum_ = winner->member->LastRecordChecksum();
      for (auto* h : holders) {
        h->has_lookahead = false;  // superseded copies die silently.
      }
      expected_++;
      return Result::kRecord;
    }
    const bool had_holders = !holders.empty();
    for (auto* h : holders) {
      h->has_lookahead = false;  // superseded copies explain themselves.
    }
    bool swallowed_covered = false;
    for (auto& state : states_) {
      if (state.member->SwallowedCovers(expected_)) {
        swallowed_covered = true;
        break;
      }
    }
    if (void_covered || swallowed_covered) {
      // A void in a WAL older than the declaration predates the cut, and an
      // open at N=1 seeds past such voids: the pending indices below it were
      // never reissued.
      for (auto& state : states_) {
        if (pending_discarded_ != 0 &&
            state.member->wal_number() < pending_declarer_ &&
            state.member->VoidCovers(expected_)) {
          pending_discarded_ = 0;
        }
      }
      expected_++;
      continue;
    }
    // No live holder, no void cover, and no swallowed span. When every
    // member finished, the trailing loss ends the stream, even past the
    // lost file's floor: nothing survives above to bound it. The frontier
    // and the member end states carry it to the recovery-mode mapping.
    bool any_unfinished = false;
    for (auto& state : states_) {
      if (!state.finished) {
        any_unfinished = true;
        break;
      }
    }
    if (expected_ >= lost_floor_ && any_unfinished) {
      // Possibly borne by the lost file: no torn tail or supersession
      // explains it.
      if (continue_past_unexplained_holes_) {
        ++skipped_unexplained_holes_;
        expected_++;
        continue;
      }
      return StopAtFrontier();
    }
    if (!any_unfinished) {
      if (expected_ >= lost_floor_ && !continue_past_unexplained_holes_) {
        // As before: outside skip, a trailing attested loss stops the
        // stream rather than ending it.
        return StopAtFrontier();
      }
      if (expected_ >= lost_floor_) {
        skipped_trailing_lost_wal_ = true;
      }
      frontier_ = expected_;
      terminal_ = Result::kCleanEnd;
      return terminal_;
    }
    // Members hold records above the index. A finished member whose torn or
    // swallowed tail covers it explains it as lost tail.
    TailGap gap;
    gap.index = expected_;
    for (auto& state : states_) {
      if (state.finished && state.member->TailImpure() &&
          state.max_seen < expected_) {
        gap.members.push_back(state.member->wal_number());
        gap.reported = gap.reported || state.member->TailReported();
      }
    }
    if (had_holders) {
      // The holders were all superseded, so the copies are explained and the
      // question is the absent live copy. With no torn tail covering the
      // index, no member lost anything and the index was never reissued
      // after the cut that discarded it: re-trust that declaration and move
      // on, unless a live record turns up above it first (see
      // pending_discarded_). A torn tail over the same index may still hold
      // its live copy, which stays a frontier. (Under skip, recovery opts out
      // of even that via continue_past_discarded_holes.)
      if (continue_past_discarded_holes_) {
        skipped_discard_ = true;
        expected_++;
        continue;
      }
      if (gap.members.empty()) {
        if (pending_discarded_ == 0) {
          pending_discarded_ = expected_;
          pending_declarer_ = std::numeric_limits<uint64_t>::max();
        }
        // The declaration is written to the lowest-numbered WAL its open
        // creates, so every WAL below the declarer predates the cut.
        for (auto* h : holders) {
          for (auto& state : states_) {
            if (state.member->Supersedes(h->member->wal_number(), expected_)) {
              pending_declarer_ =
                  std::min(pending_declarer_, state.member->wal_number());
            }
          }
        }
        expected_++;
        continue;
      }
      return StopAtFrontier();
    }
    // No holder at all: surface the lost tail for the recovery-mode mapping
    // and continue past it, or stop at the pristine hole.
    if (!gap.members.empty()) {
      tail_gap_ = gap;
      expected_++;
      return Result::kTailGap;
    }
    // A retired predecessor bore indices in its span, and a held
    // generation-mate (a WAL a 2PC prepare kept alive) interleaves with
    // them: the retired file's records left with it, once recovery no longer
    // needed them.
    bool retired = false;
    for (const auto& [first, last] : retired_spans_) {
      if (first <= expected_ && expected_ <= last) {
        retired = true;
        break;
      }
    }
    if (retired) {
      expected_++;
      continue;
    }
#ifndef NDEBUG
    // Lets the prefix oracle's self-test move the stop one index later.
    bool explained = false;
    TEST_SYNC_POINT_CALLBACK("WALIndexMerger::Next:PristineHole", &explained);
    if (explained) {
      expected_++;
      continue;
    }
#endif  // NDEBUG
    if (continue_past_unexplained_holes_) {
      ++skipped_unexplained_holes_;
      expected_++;
      continue;
    }
    return StopAtFrontier();
  }
}

}  // namespace log
}  // namespace ROCKSDB_NAMESPACE
