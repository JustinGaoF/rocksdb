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
// (the merge derives its lower bound, so below-bound cannot occur), or the
// frontier (stop). A record covered by a supersession for its bearer is
// consumed silently and never emitted; two live records at one index fail
// closed rather than being chosen between.
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
    // record (or before finishing, when it holds no data). Only meaningful
    // once HeadScanned(). A damaged head can only be bounded by an
    // attestation from elsewhere.
    virtual bool HeadIntact() const = 0;
    // Index of the first indexed record pulled, or zero when none.
    virtual uint64_t FirstIndexed() const = 0;
    // Highest progress bound read so far, or zero when none.
    virtual uint64_t ProgressBound() const = 0;
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
  };

  class ReaderMember : public Member {
   public:
    explicit ReaderMember(ReaderMemberOptions&& options);
    uint64_t wal_number() const override { return wal_number_; }
    PulledRecord PullNext() override;
    bool HeadScanned() const override;
    bool HeadIntact() const override;
    uint64_t FirstIndexed() const override { return first_indexed_; }
    uint64_t ProgressBound() const override;
    bool VoidCovers(uint64_t index) const override;
    bool Supersedes(uint64_t bearer, uint64_t index) const override;
    bool AttestedSpan(PredecessorSpan* span) const override;
    void AttestedSpans(std::vector<PredecessorSpan>* spans) const override;

   private:
    class CountingReporter : public Reader::Reporter {
     public:
      explicit CountingReporter(Reader::Reporter* outer) : outer_(outer) {}
      void Corruption(size_t bytes, const Status& status,
                      uint64_t log_number = kMaxSequenceNumber) override {
        saw_corruption_ = true;
        if (outer_ != nullptr) {
          outer_->Corruption(bytes, status, log_number);
        }
      }
      void OldLogRecord(size_t bytes) override {
        if (outer_ != nullptr) {
          outer_->OldLogRecord(bytes);
        }
      }
      bool saw_corruption() const { return saw_corruption_; }

     private:
      Reader::Reporter* outer_;
      bool saw_corruption_ = false;
    };

    uint64_t wal_number_;
    WALRecoveryMode wal_recovery_mode_;
    CountingReporter reporter_;
    std::unique_ptr<Reader> reader_;
    std::string scratch_;
    bool done_ = false;
    PulledRecord terminal_;
    bool seen_first_ = false;
    bool corrupt_before_first_ = false;
    bool progress_noted_ = false;
    bool corrupt_before_progress_ = false;
    uint64_t first_indexed_ = 0;

    void NoteProgress();
  };

  // Merges `members` into one ordered stream.
  //
  // `expected_wals` names every WAL the merge must hold (the MANIFEST's
  // recovery set in recovery): a named file the merge does not hold cannot
  // be bounded and fails the open rather than being skipped.
  //
  // `sequence_of` projects the witness sequence number from an emitted
  // record's payload. The merge checks it strictly increases across indexed
  // emissions (unindexed records are skipped) and fails closed otherwise; it
  // returns non-OK when the payload holds no readable sequence.
  WALIndexMerger(std::vector<std::unique_ptr<Member>> members,
                 std::vector<uint64_t> expected_wals,
                 std::function<Status(Slice, SequenceNumber*)> sequence_of);

  enum class Result {
    kRecord,  // *record set (indexed or unindexed; see GetLastEmittedWALIndex)
    kCleanEnd,  // every member finished and every index was explained
    kFrontier,  // first unexplained hole, with members holding records above it
    kError,     // contradiction or unreadable member; see GetError
    kWaiting,   // an idle member needs re-pulling later; call Next() again
  };

  // Reads the next merged record into *record. Returns kRecord with *record
  // set, valid until the next call. Terminal results (kCleanEnd, kFrontier,
  // kError) are sticky; kWaiting resumes on the next call.
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

  // WALs numbered below `bound` are legitimately retired. An attested
  // predecessor below it anchors the lower bound one past its span; one at or
  // above it that the merge does not hold is lost, and the merge stops at
  // the frontier at or after the first index it bore. Call before the first
  // Next(). Without a call every absent predecessor counts as retired.
  void SetRetirementBound(uint64_t bound) { retirement_bound_ = bound; }
  // An attested predecessor the merge stopped for, or zero. Valid after
  // kFrontier.
  uint64_t GetLostPredecessor() const { return lost_predecessor_; }
  // Whether the lower bound sits at or below every index `wal_number` could
  // have lost from its head: a successor attests it, or a member numbered no
  // higher anchors the bound through its span (every index such a span
  // anchors was allocated before `wal_number` was created). A loss in a
  // bounded head surfaces as a hole instead of raising the bound. Valid once
  // the lower bound is set.
  bool HeadBounded(uint64_t wal_number) const {
    return attested_wals_.count(wal_number) != 0 ||
           wal_number >= min_anchoring_wal_;
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

  // Pulls every unfinished member without a sufficient lookahead to at least
  // `threshold`, buffering unindexed records. Records below `threshold` or
  // below their member's progress bound fail the merge.
  Status PullMembers(uint64_t threshold, PullOutcome* outcome);

  // Derives the lower bound from first indices and attested spans, enforcing
  // the head-integrity gates. Sets kCleanEnd when nothing is indexed
  // anywhere.
  Status ComputeLowerBound();

  MemberState* FindState(uint64_t wal_number);

  std::vector<std::unique_ptr<Member>> members_;
  std::set<uint64_t> expected_wals_;
  std::function<Status(Slice, SequenceNumber*)> sequence_of_;
  std::vector<MemberState> states_;
  // Buffered unindexed payloads (copied: member buffers do not outlive the
  // pulls that produced them). Drained before any indexed decision so
  // encounter order is preserved.
  std::deque<std::string> unindexed_queue_;
  bool queue_outstanding_ = false;
  Result terminal_ = Result::kWaiting;
  bool membership_checked_ = false;
  bool lower_bound_set_ = false;
  uint64_t lower_ = 0;
  uint64_t expected_ = 0;
  uint64_t last_sequence_ = 0;
  uint64_t last_emitted_index_ = 0;
  uint64_t frontier_ = 0;
  Status error_ = Status::OK();
  uint64_t retirement_bound_ = std::numeric_limits<uint64_t>::max();
  // First index a lost predecessor bore: no hole at or above it is explained
  // as a tail gap.
  uint64_t lost_floor_ = std::numeric_limits<uint64_t>::max();
  uint64_t lost_predecessor_ = 0;
  std::set<uint64_t> attested_wals_;
  // Lowest-numbered member whose span anchors the lower bound.
  uint64_t min_anchoring_wal_ = std::numeric_limits<uint64_t>::max();
};

}  // namespace log
}  // namespace ROCKSDB_NAMESPACE
