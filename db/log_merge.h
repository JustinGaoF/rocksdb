//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "db/dbformat.h"
#include "db/log_reader.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"

namespace ROCKSDB_NAMESPACE {
namespace log {

// Merges the record streams of N WAL files into the single ordered history
// the database wrote: the same scan a single WAL would have produced. The
// wal_index is the merge key; it does not appear in the emitted records.
//
// The merge pulls every member forward to each expected index in turn. At an
// absent index it decides: covered by a burn cover (skip), below nothing
// (the merge derives its lower bound, so below-bound cannot occur), swallowed
// between two positioned pulls of one member (skip), superseded-only with no
// torn tail covering it and no live record above it before a pre-cut void
// (skip: never reissued after the cut), covered by a
// finished member's torn or swallowed tail (a tail gap, surfaced for the
// recovery-mode mapping, then skipped), or the frontier (stop). A record
// covered by a supersession for its bearer is consumed silently and never
// emitted; two live records at one index fail closed rather than being chosen
// between.
//
// Emitted records stay valid until the next call into the merger, mirroring
// Reader's lifetime contract.
class WALIndexMerger {
 public:
  // One merge member: a pull-based record source plus the obsolescence and
  // head state observed so far.
  class Member {
   public:
    virtual ~Member() = default;
    virtual uint64_t wal_number() const = 0;

    enum class Pull {
      kRecord,         // indexed data record; `wal_index` set
      kUnindexed,      // data record without an index: passed through in order
      kFinishedClean,  // clean EOF: nothing more is coming, nothing was lost
      kFinishedStop,   // torn tail / old record / corruption stop: done, lossy
      kFailedIO,       // unreadable
      kIdle,           // no data yet; the merge waits and pulls again later
    };
    struct PulledRecord {
      Pull kind = Pull::kIdle;
      uint64_t wal_index = 0;  // kRecord only
      Slice payload;           // kRecord/kUnindexed: stripped record bytes,
                               // valid until the next pull on this member
    };
    virtual PulledRecord PullNext() = 0;

    // True once the member's first data record has been pulled, it finished,
    // or a progress bound was read (a progress record closes the head, so
    // reading one proves the head metadata is complete). Only then is its
    // supersession set complete.
    virtual bool HeadScanned() const = 0;
    // True when no corruption was reported before the member's first data
    // record. A member that holds no data is intact when it read no
    // corruption before ending: one that stopped at a truncated end of file or
    // at a recycled file's old record is a lost tail. Only meaningful once
    // HeadScanned(). A damaged head can only be bounded by an attestation from
    // elsewhere.
    virtual bool HeadIntact() const = 0;
    // Index of the first indexed record pulled, or zero when none.
    virtual uint64_t FirstIndexed() const = 0;
    // Highest progress bound read so far, or zero when none.
    virtual uint64_t ProgressBound() const = 0;
    // Progress bound read before the member's first data record, or zero
    // when none. A progress bound is the next index the writer would allocate
    // when it wrote the record, so every index allocated later is at or above
    // it, in this WAL's generation or a later one.
    virtual uint64_t HeadFloor() const { return 0; }
    // Whether any void range read so far covers `index`.
    virtual bool VoidCovers(uint64_t index) const = 0;
    // Whether a supersession read so far obsoletes (`bearer`, `index`).
    virtual bool Supersedes(uint64_t bearer, uint64_t index) const = 0;
    // Predecessor span attested by this member's head, if one was read.
    struct PredecessorSpan {
      uint64_t wal_number = 0;
      uint64_t first = 0;
      uint64_t last = 0;
    };
    virtual bool AttestedSpan(PredecessorSpan* span) const = 0;
    // Every span this member's head attests: AttestedSpan's first, then any
    // further ones, which a WAL opened at fewer partitions than wrote its
    // predecessors attests for the chains it folds.
    virtual void AttestedSpans(std::vector<PredecessorSpan>* spans) const {
      PredecessorSpan span;
      if (AttestedSpan(&span)) {
        spans->push_back(span);
      }
    }
    // Whether the member observed and skipped corruption strictly between
    // two positioned pulls bounding `index`: it pulled an index below
    // `index`, then reported corruption, then pulled an index above `index`.
    // Only members that read past reports attest these; members that stop
    // at the first report never do. A swallowed index skips silently like
    // a void cover, except it never contradicts a holder: the destroyed
    // record may have been unindexed.
    virtual bool SwallowedCovers(uint64_t index) const = 0;
    // True once the member finished with a torn or swallowed tail: it pulled
    // indexed data, then stopped mid-file or observed corruption after its
    // last positioned pull. A finished, tail-impure member explains the
    // indices above its last pull as lost tail rather than a pristine hole.
    // Only meaningful once finished; unfinished members never account for
    // a gap, and neither do members that never held an index.
    virtual bool TailImpure() const = 0;
    // True when the tail impurity was reported through the member's reporter
    // (as opposed to a silent torn stop). Only meaningful when TailImpure().
    virtual bool TailReported() const = 0;
    // Whether the member's file claims to carry indices: its reader consumed
    // the WAL index marker. A file may legally mix indexed and unindexed
    // data records, but only a real file can contradict its own marker, so
    // scripted members satisfy this by default.
    virtual bool SawWALIndexMarker() const { return true; }
    // Data records pulled so far, by kind. Zero both until the first pull.
    virtual uint64_t IndexedDataPulled() const { return 0; }
    virtual uint64_t UnindexedDataPulled() const { return 0; }
    // True once the member's reader saw a record from a previous use of the
    // log (a recycled log number). Recovery attributes point-in-time cuts
    // to the latest WAL that saw one.
    virtual bool SawOldLogRecord() const { return false; }
    // Checksum of the most recently pulled record, or zero when the pull
    // produced no checksum (a non-record pull, or a fragment-buffered member,
    // whose reader does not compute record checksums).
    virtual uint64_t LastRecordChecksum() const { return 0; }
    // Timestamp-size map recorded by the member's reader so far.
    virtual const UnorderedMap<uint32_t, size_t>& LastTimestampSizes() const {
      static const UnorderedMap<uint32_t, size_t> empty;
      return empty;
    }
  };

  // A Member that pulls records from a log::Reader it owns. Corruption is
  // counted internally for head-integrity tracking and optionally forwarded
  // to an outer reporter.
  struct ReaderMemberOptions {
    uint64_t wal_number = 0;
    std::unique_ptr<SequentialFileReader> file;
    bool checksum = true;
    bool fragment_buffered = false;
    WALRecoveryMode wal_recovery_mode =
        WALRecoveryMode::kTolerateCorruptedTailRecords;
    bool track_and_verify_wals = false;
    bool stop_replay_for_corruption = false;
    uint64_t min_wal_number_to_keep = std::numeric_limits<uint64_t>::max();
    PredecessorWALInfo observed_predecessor_wal_info;
    std::shared_ptr<Logger> info_log;
    Reader::Reporter* outer_reporter = nullptr;
    // Stop the member at the first observed report, dropping the in-flight
    // record, mirroring recovery's break-before-processing the record whose
    // read reported corruption. Set when the outer reporter wires reports
    // into recovery state (paranoid, non-skip modes); when false, reports
    // are swallowed and the member reads on, as skip-mode recovery does.
    bool stop_on_report = false;
  };

  class ReaderMember : public Member {
   public:
    explicit ReaderMember(ReaderMemberOptions&& options);
    uint64_t wal_number() const override { return wal_number_; }
    PulledRecord PullNext() override;
    bool HeadScanned() const override;
    bool HeadIntact() const override;
    uint64_t FirstIndexed() const override { return first_indexed_; }
    // Index of the last indexed record pulled, or zero when none.
    uint64_t LastIndexed() const { return last_positioned_index_; }
    uint64_t ProgressBound() const override;
    uint64_t HeadFloor() const override;
    bool VoidCovers(uint64_t index) const override;
    bool Supersedes(uint64_t bearer, uint64_t index) const override;
    bool AttestedSpan(PredecessorSpan* span) const override;
    void AttestedSpans(std::vector<PredecessorSpan>* spans) const override;
    bool SwallowedCovers(uint64_t index) const override;
    bool TailImpure() const override;
    bool TailReported() const override;
    bool SawWALIndexMarker() const override {
      return reader_->SawWALIndexMarker();
    }
    uint64_t IndexedDataPulled() const override { return indexed_pulled_; }
    uint64_t UnindexedDataPulled() const override { return unindexed_pulled_; }
    bool SawOldLogRecord() const override { return reporter_.saw_old(); }
    uint64_t LastRecordChecksum() const override { return checksum_; }
    const UnorderedMap<uint32_t, size_t>& LastTimestampSizes() const override {
      return reader_->GetRecordedTimestampSize();
    }
    const std::unique_ptr<Reader>& reader() const { return reader_; }

   private:
    class CountingReporter : public Reader::Reporter {
     public:
      CountingReporter(Reader::Reporter* outer,
                       const std::unique_ptr<Reader>* reader)
          : outer_(outer), reader_(reader) {}
      void Corruption(size_t bytes, const Status& status,
                      uint64_t log_number = kMaxSequenceNumber) override {
        saw_corruption_ = true;
        // The reader marks itself stopped before raising a truncated-tail
        // report, so an earlier report is corruption it read past.
        if (!(*reader_)->WALRecoveryStoppedBeforeEndOfFile()) {
          saw_corruption_before_stop_ = true;
        }
        reported_since_record_ = true;
        if (outer_ != nullptr) {
          outer_->Corruption(bytes, status, log_number);
        }
      }
      void OldLogRecord(size_t bytes) override {
        saw_old_ = true;
        if (outer_ != nullptr) {
          outer_->OldLogRecord(bytes);
        }
      }
      void NoteRecord() { reported_since_record_ = false; }
      bool saw_corruption() const { return saw_corruption_; }
      bool saw_corruption_before_stop() const {
        return saw_corruption_before_stop_;
      }
      bool reported_since_record() const { return reported_since_record_; }
      bool saw_old() const { return saw_old_; }

     private:
      Reader::Reporter* outer_;
      const std::unique_ptr<Reader>* reader_;
      bool saw_corruption_ = false;
      bool saw_corruption_before_stop_ = false;
      bool reported_since_record_ = false;
      bool saw_old_ = false;
    };

    uint64_t wal_number_;
    WALRecoveryMode wal_recovery_mode_;
    bool stop_on_report_;
    CountingReporter reporter_;
    std::unique_ptr<Reader> reader_;
    std::string scratch_;
    uint64_t checksum_ = 0;
    bool done_ = false;
    PulledRecord terminal_;
    bool seen_first_ = false;
    bool corrupt_before_first_ = false;
    uint64_t head_floor_ = 0;
    uint64_t indexed_pulled_ = 0;
    uint64_t unindexed_pulled_ = 0;
    bool progress_noted_ = false;
    bool corrupt_before_progress_ = false;
    uint64_t first_indexed_ = 0;
    // Last positioned (indexed) pull, bounding swallowed spans below. Reports
    // observed since arm `swallow_pending_`; the next positioned pull closes
    // the span into `swallowed_`. Pending at finish means corruption after
    // the last positioned pull, which is tail impurity even when an
    // intervening unindexed pull cleared the per-read report flag.
    uint64_t last_positioned_index_ = 0;
    bool swallow_pending_ = false;
    uint64_t swallow_pending_lo_ = 0;
    std::vector<std::pair<uint64_t, uint64_t>> swallowed_;

    void NoteProgress();
  };

  // Merges `members` into one ordered stream.
  //
  // `expected_wals` names every WAL the merge must hold (the MANIFEST's
  // recovery set in recovery): a named file the merge does not hold cannot
  // be bounded and fails the open rather than being skipped.
  //
  // `sequence_of` projects the witness sequence number from an emitted
  // record's payload. The merge checks it never decreases across indexed
  // emissions (unindexed records are skipped) and fails closed otherwise; it
  // returns non-OK when the payload holds no readable sequence. Equal is
  // allowed: empty batches consume an index but no sequence.
  //
  // `continue_past_discarded_holes` skips superseded-only holes even when a
  // torn tail covers them, instead of stopping at the frontier. Recovery
  // sets it only under skip: the discard was declared by an earlier
  // recovery, so re-trusting it extracts the surviving stream. Strict by
  // default.
  WALIndexMerger(std::vector<std::unique_ptr<Member>> members,
                 std::vector<uint64_t> expected_wals,
                 std::function<Status(Slice, SequenceNumber*)> sequence_of,
                 bool continue_past_discarded_holes = false);

  enum class Result {
    kRecord,  // *record set (indexed or unindexed; see GetLastEmittedWALIndex)
    kCleanEnd,  // every member finished and every index was explained
    kFrontier,  // first unexplained hole, with members holding records above it
    kError,     // contradiction or unreadable member; see GetError
    kWaiting,   // an idle member needs re-pulling later; call Next() again
    // An index with no live holder and no void cover, explained as lost tail
    // by at least one finished member; see GetTailGap. Non-terminal: the
    // merge continues past it. *record is untouched.
    kTailGap,
  };

  // A skipped tail-gap index: which members' torn or swallowed tails cover
  // it, and whether any of them reported corruption (as opposed to stopping
  // silently). Valid after kTailGap, until the next call into the merger.
  struct TailGap {
    uint64_t index = 0;
    std::vector<uint64_t> members;
    bool reported = false;
  };

  // Reads the next merged record into *record. Returns kRecord with *record
  // set, valid until the next call. Terminal results (kCleanEnd, kFrontier,
  // kError) are sticky; kWaiting and kTailGap resume on the next call.
  Result Next(Slice* record);

  // First unexplained index. Valid after kFrontier; after kCleanEnd it is one
  // past the highest explained index, or zero when no index was explained.
  uint64_t GetFrontier() const { return frontier_; }
  // Failure details. Valid after kError.
  const Status& GetError() const { return error_; }
  // wal_index of the most recently emitted record, or zero when it carried
  // none (an unindexed record) or nothing was emitted. Recovery seeds
  // next_wal_index_ one past the highest index it actually replayed.
  uint64_t GetLastEmittedWALIndex() const { return last_emitted_index_; }
  // Member the most recently emitted record was pulled from, or nullptr when
  // nothing was emitted. Recovery attributes each record to its source WAL
  // for filtering, memtable log references, and per-file cuts.
  const Member* GetLastEmittedMember() const { return last_emitted_member_; }
  // Checksum of the most recently emitted record, captured when it was pulled
  // (ported from D126834369): an unindexed record drains after its member
  // read past it, so the member's current checksum is a later record's.
  uint64_t GetLastEmittedChecksum() const { return last_emitted_checksum_; }
  // Details of the most recently skipped tail gap. Valid after kTailGap.
  const TailGap& GetTailGap() const { return tail_gap_; }

  // WALs numbered below `bound` are legitimately retired. An attested
  // predecessor below it anchors the lower bound one past its span; one at or
  // above it that the merge does not hold is lost, and the merge stops at
  // the frontier at or after the first index it bore. Call before the first
  // Next(). Without a call every absent predecessor counts as retired.
  void SetRetirementBound(uint64_t bound) { retirement_bound_ = bound; }
  // The lowest index any member could hold when no earlier index was ever
  // retired, or zero (the default) for no floor. It bounds the lower bound
  // only when no member attests a predecessor, as in a brand-new DB's first
  // WALs: a loss at the bottom of those then surfaces as a hole instead of
  // raising the bound past it. Call before the first Next().
  void SetHistoryFloor(uint64_t index) { history_floor_ = index; }
  // Skips every unexplained hole, pristine or below a lost predecessor's
  // first index, instead of stopping at the frontier, so the records above
  // it are emitted. Recovery sets it only under kSkipAnyCorruptedRecords,
  // which skips what it cannot read, as it does with one WAL. Superseded
  // records are still never emitted. Call before the first Next().
  void SetSkipUnexplainedHoles(bool skip) { skip_unexplained_holes_ = skip; }
  // Holes skipped so far under SetSkipUnexplainedHoles, and the first one.
  uint64_t GetSkippedHoles() const { return skipped_holes_; }
  uint64_t GetFirstSkippedHole() const { return first_skipped_hole_; }
  // An attested predecessor the merge stopped for, or zero. Valid after
  // kFrontier.
  uint64_t GetLostPredecessor() const { return lost_predecessor_; }
  // Whether the frontier is a pending superseded-only index (one no void
  // older than its declaration explained) rather than a plain hole. Valid
  // after kFrontier.
  bool FrontierIsPendingDiscard() const { return frontier_pending_discard_; }
  // Whether the drive skipped a superseded-only index that no void older
  // than its declaration explained: one re-trusted under
  // continue_past_discarded_holes, or one still pending at the end. A record
  // written above such an index later reads to the next merge as a live
  // record above a lost reissue. Valid after kCleanEnd.
  bool SkippedUnexplainedDiscard() const {
    return skipped_discard_ || pending_discarded_ != 0;
  }
  // Whether the lower bound sits at or below every index `wal_number` could
  // have lost from its head: a successor attests it, or a member numbered no
  // higher anchors the bound through its span (every index such a span
  // anchors was allocated before `wal_number` was created). A loss in a
  // bounded head surfaces as a hole instead of raising the bound. Valid once
  // the lower bound is set.
  // With no attestation anywhere, the history floor bounds every head: the
  // lower bound sits at the floor, below which no index was ever allocated.
  bool HeadBounded(uint64_t wal_number) const {
    return attested_wals_.count(wal_number) != 0 ||
           wal_number >= min_anchoring_wal_ ||
           (history_floor_ != 0 && attested_wals_.empty());
  }

  // How a member finished, for the recovery-mode mapping: after kCleanEnd,
  // recovery asks this to tell a torn tail (mode-dependent) from a clean one.
  enum class MemberEnd { kUnfinished, kClean, kStop };
  MemberEnd GetMemberEnd(uint64_t wal_number) const;

 private:
  struct MemberState {
    explicit MemberState(Member* m) : member(m) {}
    Member* member;
    bool finished = false;
    bool finished_clean = false;
    bool has_lookahead = false;
    Member::PulledRecord lookahead;
    uint64_t progress = 0;
    // Highest index pulled from the member so far. A finished member
    // accounts for a gap only above this: its torn or swallowed tail covers
    // (max_seen, +inf).
    uint64_t max_seen = 0;
  };

  struct PullOutcome {
    bool blocked = false;   // some member idled or its bound is too low
    bool advanced = false;  // some member finished, buffered, or progressed
  };

  Result Fail(const Status& status) {
    error_ = status;
    terminal_ = Result::kError;
    return terminal_;
  }

  // Under SetSkipUnexplainedHoles, steps past expected_ and returns true;
  // otherwise returns false and leaves the stop to the caller.
  bool SkipUnexplainedHole() {
    if (!skip_unexplained_holes_) {
      return false;
    }
    if (skipped_holes_++ == 0) {
      first_skipped_hole_ = expected_;
    }
    expected_++;
    return true;
  }

  // Stops at expected_ with members holding records above it, or at the
  // pending superseded-only index below it, which those records leave
  // unexplained.
  Result StopAtFrontier() {
    frontier_ = pending_discarded_ != 0 ? pending_discarded_ : expected_;
    frontier_pending_discard_ = pending_discarded_ != 0;
    terminal_ = Result::kFrontier;
    return terminal_;
  }

  // Pulls every unfinished member without a sufficient lookahead to at least
  // `threshold`, buffering unindexed records. Records below `threshold` or
  // below their member's progress bound fail the merge.
  Status PullMembers(uint64_t threshold, PullOutcome* outcome);

  // Fails a member that contradicts its own index marker, or that carries no
  // index at all. Runs after every pull, so a fully unindexed file fails
  // before its records are emitted; indexed data without a marker waits for
  // the attestation checks when the lower bound is not set yet.
  Status CheckMemberCoherence(const MemberState& state);

  // Derives the lower bound from first indices and attested spans, enforcing
  // the head-integrity gates. Sets kCleanEnd when nothing is indexed
  // anywhere.
  Status ComputeLowerBound();

  MemberState* FindState(uint64_t wal_number);

  std::vector<std::unique_ptr<Member>> members_;
  std::set<uint64_t> expected_wals_;
  std::function<Status(Slice, SequenceNumber*)> sequence_of_;
  bool continue_past_discarded_holes_;
  std::vector<MemberState> states_;
  // Buffered unindexed payloads with their source member and record checksum
  // (payloads copied: member buffers do not outlive the pulls that produced
  // them; checksums captured: the member reads past them before they drain).
  // Drained before any indexed decision so encounter order is preserved.
  struct UnindexedQueued {
    Member* member;
    std::string payload;
    uint64_t checksum;
  };
  std::deque<UnindexedQueued> unindexed_queue_;
  bool queue_outstanding_ = false;
  Result terminal_ = Result::kWaiting;
  bool membership_checked_ = false;
  bool lower_bound_set_ = false;
  bool skipped_discard_ = false;
  uint64_t lower_ = 0;
  uint64_t expected_ = 0;
  uint64_t last_sequence_ = 0;
  uint64_t last_emitted_index_ = 0;
  const Member* last_emitted_member_ = nullptr;
  uint64_t last_emitted_checksum_ = 0;
  TailGap tail_gap_;
  // First superseded-only index skipped since the last void that a WAL older
  // than its declaration holds, or zero. A live record above it makes it the
  // frontier. `pending_declarer_` is the lowest WAL declaring it.
  uint64_t pending_discarded_ = 0;
  uint64_t pending_declarer_ = 0;
  uint64_t frontier_ = 0;
  bool frontier_pending_discard_ = false;
  Status error_ = Status::OK();
  uint64_t retirement_bound_ = std::numeric_limits<uint64_t>::max();
  uint64_t history_floor_ = 0;
  bool skip_unexplained_holes_ = false;
  uint64_t skipped_holes_ = 0;
  uint64_t first_skipped_hole_ = 0;
  // First index a lost predecessor bore: no hole at or above it is explained
  // as a tail gap.
  uint64_t lost_floor_ = std::numeric_limits<uint64_t>::max();
  uint64_t lost_predecessor_ = 0;
  std::set<uint64_t> attested_wals_;
  // [first, last] spans attested for predecessors below the retirement
  // bound. An index in one that no member holds or explains was the retired
  // file's.
  std::vector<std::pair<uint64_t, uint64_t>> retired_spans_;
  // Lowest-numbered member whose span anchors the lower bound.
  uint64_t min_anchoring_wal_ = std::numeric_limits<uint64_t>::max();
};

}  // namespace log
}  // namespace ROCKSDB_NAMESPACE
