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
      reporter_(options.outer_reporter) {
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
  if (reader_->ReadRecord(&record, &scratch_, wal_recovery_mode_)) {
    if (!seen_first_) {
      seen_first_ = true;
      corrupt_before_first_ = reporter_.saw_corruption();
    }
    NoteProgress();
    const uint64_t idx = reader_->GetLastReadWALIndex();
    PulledRecord pulled;
    if (idx == 0) {
      pulled.kind = Member::Pull::kUnindexed;
      pulled.payload = record;
      return pulled;
    }
    if (first_indexed_ == 0) {
      first_indexed_ = idx;
    }
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
    // A torn stop before any data is a damaged head even when the reader
    // reported nothing: recycled torn tails stop silently by design.
    return !reporter_.saw_corruption() &&
           !reader_->WALRecoveryStoppedBeforeEndOfFile();
  }
  // Unscanned: provisional, the merge never asks yet.
  return true;
}

uint64_t WALIndexMerger::ReaderMember::ProgressBound() const {
  return reader_->GetWALIndexProgressBound();
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

void WALIndexMerger::ReaderMember::NoteProgress() {
  if (!progress_noted_ && reader_->GetWALIndexProgressBound() != 0) {
    progress_noted_ = true;
    corrupt_before_progress_ = reporter_.saw_corruption();
  }
}

WALIndexMerger::WALIndexMerger(
    std::vector<std::unique_ptr<Member>> members,
    std::vector<uint64_t> expected_wals,
    std::function<Status(Slice, SequenceNumber*)> sequence_of)
    : members_(std::move(members)),
      expected_wals_(expected_wals.begin(), expected_wals.end()),
      sequence_of_(std::move(sequence_of)) {
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
        s.lookahead = pulled;
        s.has_lookahead = true;
        outcome->advanced = true;
      } else if (pulled.kind == Member::Pull::kUnindexed) {
        unindexed_queue_.push_back(pulled.payload.ToString());
        outcome->advanced = true;
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
    Member::PredecessorSpan span;
    if (!s.member->AttestedSpan(&span)) {
      continue;
    }
    const auto it = attested.find(span.wal_number);
    if (it == attested.end()) {
      attested.emplace(span.wal_number, std::make_pair(span.first, span.last));
    } else {
      it->second.first = std::min(it->second.first, span.first);
      it->second.second = std::min(it->second.second, span.last);
    }
  }
  for (const auto& [pred, span] : attested) {
    attested_wals_.insert(pred);
  }
  // A span anchors the bound unless it attests a held or lost predecessor
  // that bears no index.
  for (auto& s : states_) {
    Member::PredecessorSpan span;
    if (s.member->AttestedSpan(&span) &&
        (span.first != 0 || (FindState(span.wal_number) == nullptr &&
                             span.wal_number < retirement_bound_))) {
      min_anchoring_wal_ = std::min(min_anchoring_wal_, s.member->wal_number());
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
  uint64_t lower = std::numeric_limits<uint64_t>::max();
  for (auto& s : states_) {
    if (s.member->FirstIndexed() != 0) {
      lower = std::min(lower, s.member->FirstIndexed());
    }
  }
  for (const auto& [pred, span] : attested) {
    if (FindState(pred) != nullptr || pred >= retirement_bound_) {
      // A zero start attests an empty predecessor and anchors nothing.
      if (span.first != 0) {
        lower = std::min(lower, span.first);
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
    }
  }
  if (lower == std::numeric_limits<uint64_t>::max()) {
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
      *record = Slice(unindexed_queue_.front());
      last_emitted_index_ = 0;
      queue_outstanding_ = true;
      return Result::kRecord;
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
    if (live.size() == 1) {
      MemberState* winner = live[0];
      SequenceNumber seq = 0;
      s = sequence_of_(winner->lookahead.payload, &seq);
      if (!s.ok()) {
        return Fail(Status::Corruption("WAL index merge: index " +
                                       std::to_string(expected_) + ": " +
                                       s.ToString()));
      }
      if (seq <= last_sequence_) {
        return Fail(Status::Corruption("WAL index merge: index " +
                                       std::to_string(expected_) +
                                       " does not advance the sequence"));
      }
      last_sequence_ = seq;
      *record = winner->lookahead.payload;
      last_emitted_index_ = expected_;
      for (auto* h : holders) {
        h->has_lookahead = false;  // superseded copies die silently.
      }
      expected_++;
      return Result::kRecord;
    }
    for (auto* h : holders) {
      h->has_lookahead = false;  // superseded copies explain themselves.
    }
    if (void_covered) {
      expected_++;
      continue;
    }
    for (auto& state : states_) {
      if (!state.finished) {
        frontier_ = expected_;
        terminal_ = Result::kFrontier;
        return terminal_;
      }
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
    frontier_ = expected_;
    terminal_ = Result::kCleanEnd;
    return terminal_;
  }
}

}  // namespace log
}  // namespace ROCKSDB_NAMESPACE
