//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/log_merge.h"

#include <string>
#include <vector>

#include "db/log_reader.h"
#include "db/log_writer.h"
#include "file/sequence_file_reader.h"
#include "file/writable_file_writer.h"
#include "port/port.h"
#include "rocksdb/env.h"
#include "test_util/testharness.h"
#include "test_util/testutil.h"
#include "util/coding.h"
#include "util/crc32c.h"

namespace ROCKSDB_NAMESPACE::log {

// Param type is tuple<int, bool>:
// get<0>(tuple): non-zero if recycling log, zero if regular log
// get<1>(tuple): true for FragmentBufferedReader members, false for Reader
class LogMergeTest : public ::testing::TestWithParam<std::tuple<int, bool>> {
 protected:
  bool recyclable() const { return std::get<0>(GetParam()) != 0; }
  bool fragment_buffered() const { return std::get<1>(GetParam()); }
  int header_size() const {
    return recyclable() ? kRecyclableHeaderSize : kHeaderSize;
  }

  // In-memory sequential file with graceful EOF and optional read-error
  // injection (fails the Nth Read call when `fail_on_read` is positive).
  class InMemorySource : public FSSequentialFile {
   public:
    InMemorySource(std::string contents, int fail_on_read)
        : contents_(std::move(contents)), fail_on_read_(fail_on_read) {}
    IOStatus Read(size_t n, const IOOptions& /*opts*/, Slice* result,
                  char* scratch, IODebugContext* /*dbg*/) override {
      ++reads_;
      if (fail_on_read_ > 0 && reads_ == fail_on_read_) {
        return IOStatus::IOError("injected read error");
      }
      n = std::min(n, contents_.size() - pos_);
      memcpy(scratch, contents_.data() + pos_, n);
      *result = Slice(scratch, n);
      pos_ += n;
      return IOStatus::OK();
    }
    IOStatus Skip(uint64_t n) override {
      pos_ = std::min(pos_ + static_cast<size_t>(n), contents_.size());
      return IOStatus::OK();
    }

   private:
    std::string contents_;
    size_t pos_ = 0;
    int fail_on_read_;
    int reads_ = 0;
  };

  // One indexed WAL image under construction. Data payloads carry an 8-byte
  // witness sequence ahead of the body.
  class MemberFile {
   public:
    MemberFile(uint64_t log_number, bool recyclable) : log_number_(log_number) {
      std::unique_ptr<FSWritableFile> sink(new test::StringSink(&contents_));
      std::unique_ptr<WritableFileWriter> file_writer(
          new WritableFileWriter(std::move(sink), "", FileOptions()));
      writer_.reset(new Writer(std::move(file_writer), log_number, recyclable));
      writer_->SetPartitionWALUsage(
          PartitionWALUsage::kPartitionByColumnFamily);
    }
    Status Marker() {
      return writer_->MaybeAddWALIndexMarkerRecord(WriteOptions());
    }
    Status Data(const std::string& body, uint64_t index, uint64_t seq) {
      std::string payload;
      PutFixed64(&payload, seq);
      payload += body;
      return writer_->AddRecord(WriteOptions(), Slice(payload), /*seqno=*/0,
                                index);
    }
    Status Void(uint64_t lo, uint64_t hi) {
      return writer_->AddWALIndexVoidRecord(WriteOptions(), lo, hi);
    }
    Status Sup(uint64_t wal, uint64_t first) {
      return writer_->AddWALIndexSupersessionRecord(WriteOptions(), wal, first);
    }
    Status Progress(uint64_t bound) {
      return writer_->AddWALIndexProgressRecord(WriteOptions(), bound);
    }
    Status Span(uint64_t pred, uint64_t first, uint64_t last) {
      return writer_->AddWALIndexPredecessorSpanRecord(WriteOptions(), pred,
                                                       first, last);
    }
    std::string Bytes() const { return contents_.ToString(); }
    uint64_t log_number() const { return log_number_; }

   private:
    uint64_t log_number_;
    Slice contents_;
    std::unique_ptr<Writer> writer_;
  };

  // Scripted merge member for cases real files cannot produce on demand: a
  // persistently idle member, or head damage with synthetic attestation.
  class ScriptMember : public WALIndexMerger::Member {
   public:
    struct Step {
      Pull kind = Pull::kIdle;
      uint64_t wal_index = 0;
      std::string payload;
    };
    ScriptMember(uint64_t wal_number, std::vector<Step> script)
        : wal_number_(wal_number), script_(std::move(script)) {}
    uint64_t wal_number() const override { return wal_number_; }
    PulledRecord PullNext() override {
      ++pulls_;
      if (pos_ >= script_.size()) {
        done_ = true;
        PulledRecord fin;
        fin.kind = Pull::kFinishedClean;
        return fin;
      }
      const Step& step = script_[pos_++];
      if (step.kind == Pull::kIdle) {
        ++idle_pulls_;
      } else if (step.kind == Pull::kRecord) {
        seen_first_ = true;
        if (first_indexed_ == 0) {
          first_indexed_ = step.wal_index;
        }
      } else if (step.kind == Pull::kUnindexed) {
        seen_first_ = true;
      } else {
        done_ = true;
      }
      PulledRecord pulled;
      pulled.kind = step.kind;
      pulled.wal_index = step.wal_index;
      pulled.payload = Slice(step.payload);
      return pulled;
    }
    bool HeadScanned() const override {
      return seen_first_ || done_ || ProgressBound() != 0;
    }
    bool HeadIntact() const override { return head_intact; }
    uint64_t FirstIndexed() const override { return first_indexed_; }
    uint64_t ProgressBound() const override {
      // Active strictly after `progress_after_pulls` pulls: the bound is
      // scanned during that pull and constrains what comes after, mirroring
      // a real reader that decodes progress mid-scan.
      return (progress_after_pulls > 0 && pulls_ > progress_after_pulls)
                 ? progress_bound
                 : 0;
    }
    bool VoidCovers(uint64_t index) const override {
      for (const auto& [lo, hi] : voids) {
        if (index >= lo && index <= hi) {
          return true;
        }
      }
      return false;
    }
    bool Supersedes(uint64_t bearer, uint64_t index) const override {
      const auto it = sups.find(bearer);
      return it != sups.end() && index >= it->second;
    }
    bool SwallowedCovers(uint64_t index) const override {
      for (const auto& [lo, hi] : swallowed) {
        if (index > lo && index < hi) {
          return true;
        }
      }
      return false;
    }
    bool AttestedSpan(PredecessorSpan* out) const override {
      if (!has_span) {
        return false;
      }
      *out = span;
      return true;
    }
    bool TailImpure() const override {
      return done_ && first_indexed_ != 0 && tail_impure;
    }
    bool TailReported() const override { return done_ && tail_reported; }

    bool head_intact = true;
    bool tail_impure = false;
    bool tail_reported = false;
    uint64_t progress_bound = 0;
    int progress_after_pulls = 0;
    std::vector<std::pair<uint64_t, uint64_t>> voids;
    std::vector<std::pair<uint64_t, uint64_t>> swallowed;
    std::map<uint64_t, uint64_t> sups;
    bool has_span = false;
    PredecessorSpan span;
    int pulls_ = 0;
    int idle_pulls_ = 0;

   private:
    uint64_t wal_number_;
    std::vector<Step> script_;
    size_t pos_ = 0;
    bool done_ = false;
    bool seen_first_ = false;
    uint64_t first_indexed_ = 0;
  };

  static ScriptMember::Step Rec(uint64_t idx, uint64_t seq) {
    ScriptMember::Step step;
    step.kind = WALIndexMerger::Member::Pull::kRecord;
    step.wal_index = idx;
    PutFixed64(&step.payload, seq);
    step.payload += "s";
    return step;
  }
  static ScriptMember::Step Idle() {
    ScriptMember::Step step;
    step.kind = WALIndexMerger::Member::Pull::kIdle;
    return step;
  }
  static ScriptMember::Step Stop() {
    ScriptMember::Step step;
    step.kind = WALIndexMerger::Member::Pull::kFinishedStop;
    return step;
  }

  std::unique_ptr<WALIndexMerger::Member> MakeMember(
      const MemberFile& file, const std::string& bytes, int fail_on_read = 0,
      bool stop_on_report = false,
      WALRecoveryMode wal_recovery_mode =
          WALRecoveryMode::kTolerateCorruptedTailRecords) {
    std::unique_ptr<FSSequentialFile> source(
        new InMemorySource(bytes, fail_on_read));
    std::unique_ptr<SequentialFileReader> file_reader(
        new SequentialFileReader(std::move(source), ""));
    WALIndexMerger::ReaderMemberOptions options;
    options.wal_number = file.log_number();
    options.file = std::move(file_reader);
    options.fragment_buffered = fragment_buffered();
    options.wal_recovery_mode = wal_recovery_mode;
    options.stop_on_report = stop_on_report;
    return std::unique_ptr<WALIndexMerger::Member>(
        new WALIndexMerger::ReaderMember(std::move(options)));
  }

  static Status TestSequenceOf(Slice payload, SequenceNumber* seq) {
    if (payload.size() < kWALIndexSize) {
      return Status::Corruption("test payload too short for a sequence");
    }
    *seq = DecodeFixed64(payload.data());
    return Status::OK();
  }

  std::unique_ptr<WALIndexMerger> MakeMerger(
      std::vector<std::unique_ptr<WALIndexMerger::Member>> members,
      std::vector<uint64_t> expected_wals,
      bool continue_past_discarded_holes = false) {
    return std::unique_ptr<WALIndexMerger>(
        new WALIndexMerger(std::move(members), std::move(expected_wals),
                           TestSequenceOf, continue_past_discarded_holes));
  }

  struct DrivenRecord {
    uint64_t index = 0;  // zero for unindexed records
    std::string payload;
  };
  struct DriveResult {
    WALIndexMerger::Result outcome = WALIndexMerger::Result::kWaiting;
    std::vector<DrivenRecord> records;
    std::vector<WALIndexMerger::TailGap> gaps;
    int waiting_rounds = 0;
  };

  DriveResult Drive(WALIndexMerger* merger, int max_waits = 1000) {
    DriveResult out;
    Slice record;
    for (;;) {
      const auto r = merger->Next(&record);
      if (r == WALIndexMerger::Result::kRecord) {
        out.records.push_back(
            {merger->GetLastEmittedWALIndex(), record.ToString()});
      } else if (r == WALIndexMerger::Result::kTailGap) {
        out.gaps.push_back(merger->GetTailGap());
      } else if (r == WALIndexMerger::Result::kWaiting) {
        if (++out.waiting_rounds > max_waits) {
          ADD_FAILURE() << "merge stuck waiting";
          out.outcome = r;
          return out;
        }
      } else {
        out.outcome = r;
        return out;
      }
    }
  }

  std::vector<uint64_t> DrivenIndices(const DriveResult& driven) {
    std::vector<uint64_t> indices;
    for (const auto& r : driven.records) {
      if (r.index != 0) {
        indices.push_back(r.index);
      }
    }
    return indices;
  }

  static void SetByte(std::string* data, size_t off, char value) {
    (*data)[off] = value;
  }
  static void SetFixed64At(std::string* data, size_t off, uint64_t value) {
    EncodeFixed64(&(*data)[off], value);
  }
  static void FixChecksumAt(std::string* data, size_t header_off, size_t len,
                            bool recyclable) {
    const int hs = recyclable ? kRecyclableHeaderSize : kHeaderSize;
    uint32_t crc = crc32c::Value(&(*data)[header_off + 6], hs - 6 + len);
    crc = crc32c::Mask(crc);
    EncodeFixed32(&(*data)[header_off], crc);
  }
};

TEST_P(LogMergeTest, InterleaveOrdersAcrossFiles) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(b.Data("b2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(b.Data("b4", 4, 4));
  ASSERT_OK(a.Data("a5", 5, 5));
  ASSERT_OK(b.Data("b6", 6, 6));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 6}), DrivenIndices(driven));
  ASSERT_EQ("a1", driven.records[0].payload.substr(kWALIndexSize));
  ASSERT_EQ("b6", driven.records[5].payload.substr(kWALIndexSize));
  ASSERT_EQ(7U, merger->GetFrontier());
}

// An older and a newer file both carry records at the same indices; only the
// newer set is emitted and no hole is reported.
TEST_P(LogMergeTest, SupersessionRedirectsToNewerFile) {
  MemberFile old(120, recyclable());
  MemberFile young(121, recyclable());
  ASSERT_OK(old.Marker());
  for (uint64_t i = 1; i <= 6; ++i) {
    ASSERT_OK(old.Data("o" + std::to_string(i), i, i));
  }
  ASSERT_OK(young.Marker());
  ASSERT_OK(young.Sup(120, 4));
  ASSERT_OK(young.Data("n4", 4, 7));
  ASSERT_OK(young.Data("n5", 5, 8));
  ASSERT_OK(young.Data("n6", 6, 9));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(old, old.Bytes()));
  members.push_back(MakeMember(young, young.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 6}), DrivenIndices(driven));
  ASSERT_EQ("o1", driven.records[0].payload.substr(kWALIndexSize));
  ASSERT_EQ("o3", driven.records[2].payload.substr(kWALIndexSize));
  ASSERT_EQ("n4", driven.records[3].payload.substr(kWALIndexSize));
  ASSERT_EQ("n6", driven.records[5].payload.substr(kWALIndexSize));
}

TEST_P(LogMergeTest, BurnCoverExplainsHole) {
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Void(3, 4));
  ASSERT_OK(a.Data("a5", 5, 5));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 5}), DrivenIndices(driven));
}

TEST_P(LogMergeTest, RaggedTailMergesToCleanEnd) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(b.Data("b2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(b.Data("b4", 4, 4));
  ASSERT_OK(a.Data("a5", 5, 5));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5}), DrivenIndices(driven));
}

TEST_P(LogMergeTest, UnexplainedHoleStopsAtFrontier) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(b.Data("b2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(a.Data("a5", 5, 5));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
  ASSERT_EQ(4U, merger->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
}

TEST_P(LogMergeTest, ExhaustedMembersEndClean) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(b.Data("b3", 3, 3));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(4U, merger->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
}

// Two live records at one index means bearer scoping failed: fail closed
// rather than choose.
TEST_P(LogMergeTest, DuplicateLiveIndexFails) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(b.Data("b1", 1, 2));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_TRUE(merger->GetError().IsCorruption());
  ASSERT_NE(std::string::npos, merger->GetError().ToString().find("two live"));
}

TEST_P(LogMergeTest, VoidCoverPlusDataFails) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Void(2, 3));
  ASSERT_OK(a.Data("a4", 4, 4));
  ASSERT_OK(b.Data("b2", 2, 2));
  ASSERT_OK(b.Data("b3", 3, 3));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("void cover"));
  ASSERT_EQ(std::vector<uint64_t>({1}), DrivenIndices(driven));
}

TEST_P(LogMergeTest, VoidCoverPlusSupersededDataFails) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(a.Void(4, 6));
  ASSERT_OK(a.Data("a4", 4, 4));
  ASSERT_OK(a.Data("a5", 5, 5));
  ASSERT_OK(a.Data("a6", 6, 6));
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Sup(120, 4));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("void cover"));
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
}

// A merge over a set including one idle member advances past that member's
// last written index on the strength of its progress assertion alone. Real
// readers always resolve a transient idle on re-pull, so the persistent idle
// here is scripted; that is what makes the progress load-bearing.
TEST_P(LogMergeTest, IdleMemberWithProgressDoesNotPinStream) {
  std::vector<ScriptMember::Step> idle_script;
  idle_script.push_back(Rec(1, 1));
  for (int i = 0; i < 10; ++i) {
    idle_script.push_back(Idle());
  }
  idle_script.push_back(Rec(50, 50));
  auto idle = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, std::move(idle_script)));
  auto* idle_ptr = static_cast<ScriptMember*>(idle.get());
  idle_ptr->progress_bound = 50;
  idle_ptr->progress_after_pulls = 1;
  std::vector<ScriptMember::Step> busy_script;
  for (uint64_t i = 2; i <= 49; ++i) {
    busy_script.push_back(Rec(i, i));
  }
  auto busy = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, std::move(busy_script)));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(idle));
  members.push_back(std::move(busy));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  // The member idled and the merge waited for its data at the end, but the
  // middle of the stream advanced past it on progress alone.
  ASSERT_GT(idle_ptr->idle_pulls_, 0);
  ASSERT_GT(driven.waiting_rounds, 0);
  std::vector<uint64_t> expected;
  for (uint64_t i = 1; i <= 50; ++i) {
    expected.push_back(i);
  }
  ASSERT_EQ(expected, DrivenIndices(driven));
}

// Idle is the one member state that keeps waiting: without a progress bound
// the merge holds the stream, then completes once the member produces.
TEST_P(LogMergeTest, IdleMemberWithoutProgressWaits) {
  auto idle = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, {Rec(1, 1), Idle(), Idle(), Rec(2, 2)}));
  auto busy = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, {Rec(3, 3)}));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(idle));
  members.push_back(std::move(busy));
  auto merger = MakeMerger(std::move(members), {120, 121});
  Slice record;
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  ASSERT_EQ(1U, merger->GetLastEmittedWALIndex());
  ASSERT_EQ(WALIndexMerger::Result::kWaiting, merger->Next(&record));
  ASSERT_EQ(WALIndexMerger::Result::kWaiting, merger->Next(&record));
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  ASSERT_EQ(2U, merger->GetLastEmittedWALIndex());
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  ASSERT_EQ(3U, merger->GetLastEmittedWALIndex());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, merger->Next(&record));
}

// A file the MANIFEST names and the merge does not hold cannot be bounded
// and must fail the open, not be skipped.
TEST_P(LogMergeTest, MissingMemberFailsOpen) {
  MemberFile b(121, recyclable());
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Data("b2", 2, 2));

  {
    std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
    members.push_back(MakeMember(b, b.Bytes()));
    auto merger = MakeMerger(std::move(members), {120, 121});
    DriveResult driven = Drive(merger.get());
    ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
    ASSERT_NE(std::string::npos,
              merger->GetError().ToString().find("cannot bound"));
  }
  {
    std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
    auto merger = MakeMerger(std::move(members), {120});
    DriveResult driven = Drive(merger.get());
    ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
    ASSERT_NE(std::string::npos,
              merger->GetError().ToString().find("cannot bound"));
  }
}

// Clean head truncation of the oldest file, with a successor-side span: two
// intact heads contradict each other, so the merge fails closed.
TEST_P(LogMergeTest, TruncatedHeadWithSpanFailsClosed) {
  const int hs = header_size();
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 0; i <= 4; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), 30 + i, 30 + i));
  }
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Span(120, 30, 34));
  for (uint64_t i = 5; i <= 9; ++i) {
    ASSERT_OK(b.Data("b" + std::to_string(i), 30 + i, 30 + i));
  }
  // Drop the marker and the first two records at a record boundary: the
  // survivor parses cleanly from index 32.
  const size_t record_size = static_cast<size_t>(hs) + 2 * kWALIndexSize + 2;
  std::string damaged = a.Bytes().substr(hs + 2 * record_size);

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, damaged));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("attested start mismatch"));
  ASSERT_TRUE(driven.records.empty());
}

// Corrupt head of the oldest file, with a successor-side span: the resulting
// gap is reported as the frontier, not absorbed into the lower bound.
TEST_P(LogMergeTest, DamagedHeadWithSpanReportsGap) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 30; i <= 34; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Span(120, 30, 34));
  for (uint64_t i = 35; i <= 39; ++i) {
    ASSERT_OK(b.Data("b" + std::to_string(i), i, i));
  }
  // Break the head's checksum without repairing it: the reader reports
  // corruption before any data record.
  std::string damaged = a.Bytes();
  SetByte(&damaged, 0, static_cast<char>(damaged[0] + 1));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, damaged));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
  ASSERT_EQ(30U, merger->GetFrontier());
  ASSERT_TRUE(driven.records.empty());
}

// Corrupt head with no attestation anywhere: fail closed, cannot bound.
TEST_P(LogMergeTest, DamagedHeadWithoutSpanFailsClosed) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 30; i <= 34; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }
  ASSERT_OK(b.Marker());
  for (uint64_t i = 35; i <= 39; ++i) {
    ASSERT_OK(b.Data("b" + std::to_string(i), i, i));
  }
  std::string damaged = a.Bytes();
  SetByte(&damaged, 0, static_cast<char>(damaged[0] + 1));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, damaged));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("cannot bound"));
}

// Control for the head-damage suite: an intact, consistently attested set
// merges cleanly.
TEST_P(LogMergeTest, AttestedSetMergesCleanly) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 30; i <= 34; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Span(120, 30, 34));
  for (uint64_t i = 35; i <= 39; ++i) {
    ASSERT_OK(b.Data("b" + std::to_string(i), i, i));
  }

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({30, 31, 32, 33, 34, 35, 36, 37, 38, 39}),
            DrivenIndices(driven));
}

// A void trailing behind higher-index data is corrupt input (D1 rejects it),
// so the merge stops at the hole it leaves rather than absorbing it.
TEST_P(LogMergeTest, TrailingVoidStopsAtHole) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a7", 7, 7));
  ASSERT_OK(a.Void(5, 6));
  ASSERT_OK(b.Data("b3", 3, 3));
  ASSERT_OK(b.Data("b4", 4, 4));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
  ASSERT_EQ(5U, merger->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4}), DrivenIndices(driven));
}

// Neuter check for the sequence witness: perturb the order and the merge
// fails rather than emitting.
TEST_P(LogMergeTest, SequenceWitnessRejectsReorder) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 100));
  ASSERT_OK(b.Data("b2", 2, 50));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos, merger->GetError().ToString().find("sequence"));
  ASSERT_EQ(std::vector<uint64_t>({1}), DrivenIndices(driven));
}

// A supersession explains the copies it names, not the absence of a live
// copy: when the live copy of 14 is lost and the member stops instead of
// reading past the loss, the merge stops rather than skipping. (Without the
// stop the loss is a bounded swallowed span and the merge skips it; see
// SwallowedHoleSkipsSilently.)
TEST_P(LogMergeTest, SupersededOnlyWithLiveAboveStops) {
  const int hs = header_size();
  MemberFile old(120, recyclable());
  MemberFile young(121, recyclable());
  ASSERT_OK(old.Marker());
  for (uint64_t i = 1; i <= 20; ++i) {
    ASSERT_OK(
        old.Data("o" + std::to_string(i % 10) + std::to_string(i / 10), i, i));
  }
  ASSERT_OK(young.Marker());
  ASSERT_OK(young.Sup(120, 11));
  ASSERT_OK(young.Data("n1", 11, 21));
  ASSERT_OK(young.Data("n2", 12, 22));
  ASSERT_OK(young.Data("n3", 13, 23));
  ASSERT_OK(young.Data("n4", 14, 24));
  ASSERT_OK(young.Data("n5", 15, 25));
  // Corrupt record 14's type (with a fixed checksum) so the reader skips
  // exactly that record and continues.
  std::string young_bytes = young.Bytes();
  const size_t payload = 2 * kWALIndexSize + 2;
  const size_t rec14 = static_cast<size_t>(hs) + (hs + 16) + 3 * (hs + payload);
  SetByte(&young_bytes, rec14 + 6, static_cast<char>(50));
  FixChecksumAt(&young_bytes, rec14, payload, recyclable());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(old, old.Bytes()));
  members.push_back(MakeMember(young, young_bytes, /*fail_on_read=*/0,
                               /*stop_on_report=*/true));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
  ASSERT_EQ(14U, merger->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
            DrivenIndices(driven));
  ASSERT_EQ("n1", driven.records[10].payload.substr(kWALIndexSize));
}

// A torn tail with nothing above it ends the stream; the lossy member is
// reported through its end state for the recovery-mode mapping.
TEST_P(LogMergeTest, TornTailWithoutRewriteEndsClean) {
  const int hs = header_size();
  MemberFile old(120, recyclable());
  MemberFile young(121, recyclable());
  ASSERT_OK(old.Marker());
  for (uint64_t i = 1; i <= 20; ++i) {
    ASSERT_OK(
        old.Data("o" + std::to_string(i % 10) + std::to_string(i / 10), i, i));
  }
  ASSERT_OK(young.Marker());
  ASSERT_OK(young.Sup(120, 11));
  std::string old_bytes = old.Bytes();
  const size_t rec_size = static_cast<size_t>(hs) + 2 * kWALIndexSize + 3;
  old_bytes.resize(hs + 10 * rec_size + 9);

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(old, old_bytes));
  members.push_back(MakeMember(young, young.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(11U, merger->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
            DrivenIndices(driven));
  // The fragment-buffered reader treats a torn tail as growable (its tailing
  // semantic) rather than a stop, so it reports a clean end where the plain
  // reader reports a stop. The merged stream is identical either way.
  ASSERT_EQ(fragment_buffered() ? WALIndexMerger::MemberEnd::kClean
                                : WALIndexMerger::MemberEnd::kStop,
            merger->GetMemberEnd(120));
  ASSERT_EQ(WALIndexMerger::MemberEnd::kClean, merger->GetMemberEnd(121));
}

// Unindexed records pass through in encounter order and skip the witness.
TEST_P(LogMergeTest, UnindexedRecordsPassThrough) {
  const int hs = header_size();
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("u1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(b.Data("b4", 4, 4));
  ASSERT_OK(b.Data("b5", 5, 5));
  std::string a_bytes = a.Bytes();
  SetByte(&a_bytes, hs + 6,
          static_cast<char>(recyclable() ? kRecyclableFullType : kFullType));
  FixChecksumAt(&a_bytes, hs, 2 * kWALIndexSize + 2, recyclable());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a_bytes));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(5U, driven.records.size());
  ASSERT_EQ(0U, driven.records[0].index);
  ASSERT_EQ("u1", driven.records[0].payload.substr(2 * kWALIndexSize));
  ASSERT_EQ(std::vector<uint64_t>({2, 3, 4, 5}), DrivenIndices(driven));
}

// Data below the member's own asserted progress bound contradicts the
// assertion and fails the merge.
TEST_P(LogMergeTest, DataBelowProgressBoundFails) {
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(a.Data("a4", 4, 4));
  ASSERT_OK(a.Progress(6));
  ASSERT_OK(a.Data("a5", 5, 5));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("progress bound"));
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4}), DrivenIndices(driven));
}

// A span attesting a retired predecessor anchors the bound without tripping
// it: the live set starts one past the retired end.
TEST_P(LogMergeTest, RetiredPredecessorSpanAnchorsBound) {
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Span(119, 1, 29));
  for (uint64_t i = 30; i <= 35; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({30, 31, 32, 33, 34, 35}),
            DrivenIndices(driven));
}

// R2 with a member that never receives data: its progress record alone lets
// the merge emit every other member's records past it while it stays open.
TEST_P(LogMergeTest, DataLessIdleMemberWithProgressDoesNotPinStream) {
  auto idle = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, std::vector<ScriptMember::Step>(100, Idle())));
  auto* idle_ptr = static_cast<ScriptMember*>(idle.get());
  idle_ptr->progress_bound = 50;
  idle_ptr->progress_after_pulls = 1;
  std::vector<ScriptMember::Step> busy_script;
  for (uint64_t i = 1; i <= 49; ++i) {
    busy_script.push_back(Rec(i, i));
  }
  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(idle));
  members.push_back(std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, std::move(busy_script))));
  auto merger = MakeMerger(std::move(members), {120, 121});
  Slice record;
  uint64_t emitted = 0;
  WALIndexMerger::Result r;
  while ((r = merger->Next(&record)) == WALIndexMerger::Result::kRecord ||
         (r == WALIndexMerger::Result::kWaiting && emitted == 0)) {
    emitted += r == WALIndexMerger::Result::kRecord ? 1 : 0;
  }
  // Everything below the bound came out while the member was still open.
  ASSERT_EQ(WALIndexMerger::Result::kWaiting, r);
  ASSERT_EQ(49U, emitted);
  ASSERT_LT(idle_ptr->idle_pulls_, 100);
}

// An attested predecessor at or above the retirement bound that the merge
// does not hold is lost: the merge stops at the frontier at its first index,
// even where another member's torn tail would explain the hole as a tail gap.
// Below the bound the same span is a retirement: it anchors the bound past
// its end, and the torn tail explains the gap as usual (recyclable and
// fragment-buffered readers end a torn record differently, so only the plain
// reader shows that).
TEST_P(LogMergeTest, LostPredecessorStopsAtItsFirstIndex) {
  for (const bool lost : {false, true}) {
    SCOPED_TRACE(lost ? "lost" : "retired");
    MemberFile b(121, recyclable());
    MemberFile c(122, recyclable());
    ASSERT_OK(b.Marker());
    ASSERT_OK(b.Span(120, 5, 6));
    for (uint64_t i = 7; i <= 9; ++i) {
      ASSERT_OK(b.Data("b" + std::to_string(i), i, i));
    }
    ASSERT_OK(c.Marker());
    ASSERT_OK(c.Span(119, 1, 3));
    ASSERT_OK(c.Data("c4", 4, 4));
    ASSERT_OK(c.Data("c6", 6, 6));
    // Tear c's last record: its tail covers everything above 4.
    std::string torn = c.Bytes();
    torn.resize(torn.size() - 3);
    std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
    members.push_back(MakeMember(b, b.Bytes()));
    members.push_back(MakeMember(c, torn));
    auto merger = MakeMerger(std::move(members), {121, 122});
    merger->SetRetirementBound(lost ? 120 : 121);
    DriveResult driven = Drive(merger.get());
    if (lost) {
      ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
      ASSERT_EQ(5U, merger->GetFrontier());
      ASSERT_EQ(120U, merger->GetLostPredecessor());
      ASSERT_EQ(std::vector<uint64_t>({4}), DrivenIndices(driven));
      ASSERT_TRUE(driven.gaps.empty());
    } else {
      ASSERT_EQ(0U, merger->GetLostPredecessor());
      if (!recyclable() && !fragment_buffered()) {
        ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
        ASSERT_EQ(std::vector<uint64_t>({4, 7, 8, 9}), DrivenIndices(driven));
        ASSERT_EQ(2U, driven.gaps.size());
      }
    }
  }
}

// A damaged head on a file numbered above a member whose span anchors the
// bound (here a retired predecessor's) is bounded: the indices it lost
// surface as a hole instead of failing the merge as unattested. Without the
// anchoring span the same set still fails closed.
TEST_P(LogMergeTest, DamagedHeadBoundedByLowerMemberSpan) {
  for (const bool span : {true, false}) {
    SCOPED_TRACE(span ? "span" : "no span");
    MemberFile a(120, recyclable());
    MemberFile b(121, recyclable());
    ASSERT_OK(a.Marker());
    if (span) {
      ASSERT_OK(a.Span(110, 3, 10));
    }
    ASSERT_OK(a.Data("a11", 11, 11));
    ASSERT_OK(a.Data("a13", 13, 13));
    ASSERT_OK(b.Marker());
    ASSERT_OK(b.Data("b12", 12, 12));
    ASSERT_OK(b.Data("b14", 14, 14));
    std::string damaged = b.Bytes();
    SetByte(&damaged, 0, static_cast<char>(damaged[0] + 1));

    std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
    members.push_back(MakeMember(a, a.Bytes()));
    members.push_back(MakeMember(b, damaged));
    auto merger = MakeMerger(std::move(members), {120, 121});
    merger->SetRetirementBound(120);
    DriveResult driven = Drive(merger.get());
    if (span) {
      ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
      ASSERT_EQ(12U, merger->GetFrontier());
      ASSERT_EQ(std::vector<uint64_t>({11}), DrivenIndices(driven));
    } else {
      ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
      ASSERT_NE(std::string::npos,
                merger->GetError().ToString().find("head damaged"));
    }
  }
}

// A member yielding indices out of order fails the merge.
TEST_P(LogMergeTest, OutOfOrderMemberFails) {
  const int hs = header_size();
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  std::string bytes = a.Bytes();
  const size_t third =
      static_cast<size_t>(hs) + 2 * (hs + 2 * kWALIndexSize + 2);
  SetFixed64At(&bytes, third + hs, 2);
  FixChecksumAt(&bytes, third, 2 * kWALIndexSize + 2, recyclable());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, bytes));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("below expectation"));
  ASSERT_EQ(std::vector<uint64_t>({1, 2}), DrivenIndices(driven));
}

// Degenerate input: no members and nothing expected ends cleanly with no
// frontier. Recovery with no WALs must still open.
TEST_P(LogMergeTest, EmptySetEndsClean) {
  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  auto merger = MakeMerger(std::move(members), {});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(0U, merger->GetFrontier());
  ASSERT_EQ(0U, merger->GetLastEmittedWALIndex());
  ASSERT_TRUE(driven.records.empty());
}

// Progress assertions in otherwise ordinary files disturb nothing: members
// that finish need no bounds.
TEST_P(LogMergeTest, ProgressInFinishedFilesIsHarmless) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(a.Progress(100));
  ASSERT_OK(b.Data("b4", 4, 4));
  ASSERT_OK(b.Data("b5", 5, 5));
  ASSERT_OK(b.Data("b6", 6, 6));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 6}), DrivenIndices(driven));
}

// An unreadable member fails the merge rather than truncating the stream:
// records already emitted stand, nothing above is silently dropped. The
// fourth record spans the block boundary so the failure lands mid-stream,
// after the first block was consumed.
TEST_P(LogMergeTest, UnreadableMemberFails) {
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(a.Data(std::string(40000, 'x'), 4, 4));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes(), /*fail_on_read=*/2));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_TRUE(merger->GetError().IsIOError());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
}

// A finished member's reported torn tail explains a mid-stream gap: the merge
// surfaces it for the recovery-mode mapping and continues past it.
TEST_P(LogMergeTest, ReportedTailGapSkipsAndContinues) {
  std::vector<ScriptMember::Step> torn_script;
  for (uint64_t i = 1; i <= 5; ++i) {
    torn_script.push_back(Rec(i, i));
  }
  torn_script.push_back(Stop());
  auto torn = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, std::move(torn_script)));
  static_cast<ScriptMember*>(torn.get())->tail_impure = true;
  static_cast<ScriptMember*>(torn.get())->tail_reported = true;
  std::vector<ScriptMember::Step> live_script;
  for (uint64_t i = 7; i <= 10; ++i) {
    live_script.push_back(Rec(i, i));
  }
  auto live = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, std::move(live_script)));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(torn));
  members.push_back(std::move(live));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 7, 8, 9, 10}),
            DrivenIndices(driven));
  ASSERT_EQ(1U, driven.gaps.size());
  EXPECT_EQ(6U, driven.gaps[0].index);
  EXPECT_EQ(std::vector<uint64_t>({120}), driven.gaps[0].members);
  EXPECT_TRUE(driven.gaps[0].reported);
  EXPECT_EQ(11U, merger->GetFrontier());
}

// A real torn tail mid-stream: the plain reader stops and the gap surfaces
// silently. The fragment-buffered reader treats the torn tail as growable,
// so it idles, resolves clean, and accounts for nothing: the same hole is a
// pristine frontier there. Recovery uses plain readers, so only the plain
// shape reaches it.
TEST_P(LogMergeTest, TornTailMidStreamSurfacesGapOrFrontier) {
  MemberFile old(120, recyclable());
  MemberFile young(121, recyclable());
  ASSERT_OK(old.Marker());
  for (uint64_t i = 1; i <= 10; ++i) {
    ASSERT_OK(old.Data("o" + std::to_string(i), i, i));
  }
  ASSERT_OK(old.Data("torn", 11, 11));
  ASSERT_OK(young.Marker());
  for (uint64_t i = 12; i <= 15; ++i) {
    ASSERT_OK(young.Data("n" + std::to_string(i), i, i));
  }
  std::string old_bytes = old.Bytes();
  old_bytes.resize(old_bytes.size() - 1);

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(old, old_bytes));
  members.push_back(MakeMember(young, young.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  if (fragment_buffered()) {
    ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
    ASSERT_EQ(11U, merger->GetFrontier());
    ASSERT_TRUE(driven.gaps.empty());
    ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
              DrivenIndices(driven));
    return;
  }
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(
      std::vector<uint64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15}),
      DrivenIndices(driven));
  ASSERT_EQ(1U, driven.gaps.size());
  EXPECT_EQ(11U, driven.gaps[0].index);
  EXPECT_EQ(std::vector<uint64_t>({120}), driven.gaps[0].members);
  EXPECT_FALSE(driven.gaps[0].reported);
  EXPECT_EQ(16U, merger->GetFrontier());
}

// A member that stops early leaves its whole unread tail to the gap: every
// index between its last pull and the next live record surfaces.
TEST_P(LogMergeTest, StoppedMemberSkipsUnreadTail) {
  auto torn = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, {Rec(1, 1), Rec(2, 2), Rec(3, 3), Stop()}));
  static_cast<ScriptMember*>(torn.get())->tail_impure = true;
  static_cast<ScriptMember*>(torn.get())->tail_reported = true;
  auto live = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, {Rec(6, 6), Rec(7, 7)}));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(torn));
  members.push_back(std::move(live));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 6, 7}), DrivenIndices(driven));
  ASSERT_EQ(2U, driven.gaps.size());
  EXPECT_EQ(4U, driven.gaps[0].index);
  EXPECT_EQ(5U, driven.gaps[1].index);
  EXPECT_TRUE(driven.gaps[0].reported);
}

// An empty record occupies its index but carries no witness: it emits
// without advancing the sequence check, wherever it sits.
TEST_P(LogMergeTest, EmptySequenceSkipsWitness) {
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("e1", 1, 0));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("e3", 3, 0));
  ASSERT_OK(a.Data("a4", 4, 4));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4}), DrivenIndices(driven));
}

// Every emission names its source member, indexed or unindexed, and plain
// members attach the record checksum the protection-info update consumes.
TEST_P(LogMergeTest, EmittedRecordsCarrySourceAndChecksum) {
  const int hs = header_size();
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("u0", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(b.Data("b1", 1, 1));
  std::string a_bytes = a.Bytes();
  SetByte(&a_bytes, hs + 6,
          static_cast<char>(recyclable() ? kRecyclableFullType : kFullType));
  FixChecksumAt(&a_bytes, hs, 2 * kWALIndexSize + 2, recyclable());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a_bytes));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  Slice record;
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  EXPECT_EQ(0U, merger->GetLastEmittedWALIndex());
  ASSERT_NE(nullptr, merger->GetLastEmittedMember());
  EXPECT_EQ(120U, merger->GetLastEmittedMember()->wal_number());
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  EXPECT_EQ(1U, merger->GetLastEmittedWALIndex());
  ASSERT_NE(nullptr, merger->GetLastEmittedMember());
  EXPECT_EQ(121U, merger->GetLastEmittedMember()->wal_number());
  if (fragment_buffered()) {
    // Fragment-buffered readers do not compute record checksums.
    EXPECT_EQ(0U, merger->GetLastEmittedMember()->LastRecordChecksum());
  } else {
    EXPECT_NE(0U, merger->GetLastEmittedMember()->LastRecordChecksum());
  }
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  EXPECT_EQ(2U, merger->GetLastEmittedWALIndex());
  ASSERT_NE(nullptr, merger->GetLastEmittedMember());
  EXPECT_EQ(120U, merger->GetLastEmittedMember()->wal_number());
  ASSERT_EQ(WALIndexMerger::Result::kRecord, merger->Next(&record));
  EXPECT_EQ(3U, merger->GetLastEmittedWALIndex());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, merger->Next(&record));
}

// Driving past the type limit fails closed instead of re-deciding index
// zero. Only physically crafted indices reach here.
TEST_P(LogMergeTest, IndexOverflowFailsClosed) {
  const int hs = header_size();
  MemberFile a(120, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("m", 1, 1));
  std::string bytes = a.Bytes();
  SetFixed64At(&bytes, hs + hs, std::numeric_limits<uint64_t>::max());
  FixChecksumAt(&bytes, hs, 2 * kWALIndexSize + 1, recyclable());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, bytes));
  auto merger = MakeMerger(std::move(members), {120});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
  ASSERT_NE(std::string::npos,
            merger->GetError().ToString().find("overflowed"));
  ASSERT_EQ(std::vector<uint64_t>({std::numeric_limits<uint64_t>::max()}),
            DrivenIndices(driven));
}

// A member that finished clean with an intact tail explains nothing above
// it: the hole above stays a pristine frontier. (The scripted equivalent of
// mid-file corruption swallowed before clean records, which likewise leaves
// the tail intact.)
TEST_P(LogMergeTest, CleanFinishedMemberDoesNotExplainGap) {
  std::vector<ScriptMember::Step> a_script;
  for (uint64_t i = 1; i <= 5; ++i) {
    a_script.push_back(Rec(i, i));
  }
  auto a = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, std::move(a_script)));
  auto b = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, {Rec(7, 7)}));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(a));
  members.push_back(std::move(b));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kFrontier, driven.outcome);
  ASSERT_EQ(6U, merger->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 4, 5}), DrivenIndices(driven));
  ASSERT_TRUE(driven.gaps.empty());
}

// The neuter pair of SupersededOnlyWithLiveAboveStops: the same destroyed
// record, but the member reads past it instead of stopping. The loss is a
// bounded swallowed span, so the merge skips it silently and continues.
TEST_P(LogMergeTest, SwallowedHoleSkipsSilently) {
  const int hs = header_size();
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(a.Data("a1", 1, 1));
  ASSERT_OK(a.Data("a2", 2, 2));
  ASSERT_OK(a.Data("a3", 3, 3));
  ASSERT_OK(a.Data("a4", 4, 4));
  ASSERT_OK(a.Data("a5", 5, 5));
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Data("b6", 6, 6));
  ASSERT_OK(b.Data("b7", 7, 7));
  // Corrupt record 3's type (with a fixed checksum) so the reader reports
  // exactly that record and continues.
  std::string a_bytes = a.Bytes();
  const size_t payload = 2 * kWALIndexSize + 2;
  const size_t rec3 = static_cast<size_t>(hs) + 2 * (hs + payload);
  SetByte(&a_bytes, rec3 + 6, static_cast<char>(50));
  // The reader verifies an unknown type with the non-recyclable framing, so
  // the checksum is fixed that way even in a recyclable file.
  FixChecksumAt(&a_bytes, rec3, payload, /*recyclable=*/false);

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a_bytes));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  if (recyclable()) {
    // In a recyclable file the unknown type's shorter framing desyncs the
    // stream (the skip comes up four bytes short of the next header), so the
    // reader stops its tail there instead of skipping past it, and each lost
    // index surfaces as a tail gap. Recovery WALs are not recycled, so only
    // the plain shape below reaches it.
    ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
    ASSERT_EQ(std::vector<uint64_t>({1, 2, 6, 7}), DrivenIndices(driven));
    ASSERT_EQ(3U, driven.gaps.size());
    EXPECT_EQ(3U, driven.gaps[0].index);
    EXPECT_EQ(4U, driven.gaps[1].index);
    EXPECT_EQ(5U, driven.gaps[2].index);
    EXPECT_EQ(std::vector<uint64_t>({120}), driven.gaps[0].members);
    ASSERT_EQ(8U, merger->GetFrontier());
    return;
  }
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 4, 5, 6, 7}), DrivenIndices(driven));
  ASSERT_TRUE(driven.gaps.empty());
  ASSERT_EQ(8U, merger->GetFrontier());
}

// The sequence witness is non-decreasing: an empty batch consumes an index
// but no sequence, so the next record legitimately repeats it.
TEST_P(LogMergeTest, EqualSequenceEmitsInOrder) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  ASSERT_OK(b.Marker());
  ASSERT_OK(a.Data("a1", 1, 5));
  ASSERT_OK(a.Data("a2", 2, 6));
  ASSERT_OK(b.Data("b3", 3, 6));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
}

// Superseded-only holes with no torn tail covering them were never reissued
// after the cut: the merge re-trusts the declaration and moves on.
TEST_P(LogMergeTest, DiscardedHoleWithoutTornTailSkips) {
  MemberFile old(120, recyclable());
  MemberFile young(121, recyclable());
  ASSERT_OK(old.Marker());
  for (uint64_t i = 1; i <= 5; ++i) {
    ASSERT_OK(old.Data("o" + std::to_string(i), i, i));
  }
  ASSERT_OK(young.Marker());
  ASSERT_OK(young.Sup(120, 4));
  ASSERT_OK(young.Data("n6", 6, 6));
  ASSERT_OK(young.Data("n7", 7, 7));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(old, old.Bytes()));
  members.push_back(MakeMember(young, young.Bytes()));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 6, 7}), DrivenIndices(driven));
  ASSERT_TRUE(driven.gaps.empty());
  ASSERT_EQ(8U, merger->GetFrontier());
}

// A torn tail over a superseded-only hole keeps the strict merge at the
// frontier; the forgiving merge (skip-mode recovery) skips it.
TEST_P(LogMergeTest, DiscardedHoleSkipsWhenForgiving) {
  auto torn = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, {Rec(1, 1), Rec(2, 2), Rec(3, 3), Stop()}));
  static_cast<ScriptMember*>(torn.get())->tail_impure = true;
  auto live = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, {Rec(6, 6), Rec(7, 7)}));
  static_cast<ScriptMember*>(live.get())->sups[122] = 4;
  auto discarded = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(122, {Rec(4, 4), Rec(5, 5)}));

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(std::move(torn));
  members.push_back(std::move(live));
  members.push_back(std::move(discarded));
  auto strict = MakeMerger(std::move(members), {120, 121, 122});
  DriveResult strict_driven = Drive(strict.get());
  ASSERT_EQ(WALIndexMerger::Result::kFrontier, strict_driven.outcome);
  ASSERT_EQ(4U, strict->GetFrontier());
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(strict_driven));

  auto torn2 = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(120, {Rec(1, 1), Rec(2, 2), Rec(3, 3), Stop()}));
  static_cast<ScriptMember*>(torn2.get())->tail_impure = true;
  auto live2 = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(121, {Rec(6, 6), Rec(7, 7)}));
  static_cast<ScriptMember*>(live2.get())->sups[122] = 4;
  auto discarded2 = std::unique_ptr<WALIndexMerger::Member>(
      new ScriptMember(122, {Rec(4, 4), Rec(5, 5)}));
  std::vector<std::unique_ptr<WALIndexMerger::Member>> members2;
  members2.push_back(std::move(torn2));
  members2.push_back(std::move(live2));
  members2.push_back(std::move(discarded2));
  auto forgiving = MakeMerger(std::move(members2), {120, 121, 122},
                              /*continue_past_discarded_holes=*/true);
  DriveResult forgiving_driven = Drive(forgiving.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, forgiving_driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3, 6, 7}),
            DrivenIndices(forgiving_driven));
  ASSERT_EQ(8U, forgiving->GetFrontier());
}

// A member whose file holds its marker and then a header cut short at end of
// file holds no data record, so nothing in it sits above where it stopped:
// the merge treats it as a lost tail rather than a damaged head. Point-in-time
// mode, as recovery reads by default, reports the truncation after the stop.
TEST_P(LogMergeTest, DataLessTornMemberIsATail) {
  if (fragment_buffered()) {
    return;  // Recovery reads with plain readers; this one waits out the tail.
  }
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 1; i <= 3; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }
  ASSERT_OK(b.Marker());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b.Bytes() + std::string(3, '\x01'),
                               /*fail_on_read=*/0, /*stop_on_report=*/true,
                               WALRecoveryMode::kPointInTimeRecovery));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
  ASSERT_TRUE(driven.gaps.empty());
}

// The normal crash state under recycling: the file's new marker, then a record
// from its previous use. The reader stops at the old record before any new
// data, which is a lost tail too.
TEST_P(LogMergeTest, RecycledDataLessMemberIsATail) {
  if (!recyclable() || fragment_buffered()) {
    // Only recyclable records name the log they were written for, and only
    // the plain reader, which recovery uses, reports an old record.
    return;
  }
  MemberFile a(120, recyclable());
  MemberFile previous(100, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 1; i <= 3; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }
  ASSERT_OK(previous.Marker());
  ASSERT_OK(previous.Data("old", 7, 7));
  ASSERT_OK(b.Marker());
  // The new marker overwrites the old one in place.
  const std::string b_bytes =
      b.Bytes() + previous.Bytes().substr(b.Bytes().size());

  std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
  members.push_back(MakeMember(a, a.Bytes()));
  members.push_back(MakeMember(b, b_bytes, /*fail_on_read=*/0,
                               /*stop_on_report=*/true,
                               WALRecoveryMode::kPointInTimeRecovery));
  auto merger = MakeMerger(std::move(members), {120, 121});
  DriveResult driven = Drive(merger.get());
  ASSERT_EQ(WALIndexMerger::Result::kCleanEnd, driven.outcome);
  ASSERT_EQ(std::vector<uint64_t>({1, 2, 3}), DrivenIndices(driven));
}

// Control for the two above: a checksum failure the reader reports and reads
// past, or in a recycled file stops at silently, may have dropped data, so a
// member left with no data record that way is still a damaged head. That
// holds when a torn end of file follows the failure, too: the stop comes only
// after the reader read past the corruption.
TEST_P(LogMergeTest, DataLessMemberWithSkippedCorruptionFailsClosed) {
  MemberFile a(120, recyclable());
  MemberFile b(121, recyclable());
  ASSERT_OK(a.Marker());
  for (uint64_t i = 1; i <= 3; ++i) {
    ASSERT_OK(a.Data("a" + std::to_string(i), i, i));
  }
  ASSERT_OK(b.Marker());
  ASSERT_OK(b.Data("b4", 4, 4));
  ASSERT_OK(b.Data("b5", 5, 5));
  // A checksum failure in b's first record drops the rest of its block.
  std::string b_bytes = b.Bytes();
  const size_t first_payload = static_cast<size_t>(2 * header_size()) + 2;
  SetByte(&b_bytes, first_payload,
          static_cast<char>(b_bytes[first_payload] + 1));
  std::string torn_end = b_bytes;
  torn_end.resize(kBlockSize, '\0');
  torn_end += std::string(3, '\x01');

  for (const bool with_torn_end : {false, true}) {
    SCOPED_TRACE(with_torn_end ? "torn end" : "clean end");
    if (with_torn_end && fragment_buffered()) {
      continue;  // Recovery reads with plain readers; this one waits it out.
    }
    std::vector<std::unique_ptr<WALIndexMerger::Member>> members;
    members.push_back(MakeMember(a, a.Bytes()));
    if (with_torn_end) {
      members.push_back(MakeMember(b, torn_end, /*fail_on_read=*/0,
                                   /*stop_on_report=*/true,
                                   WALRecoveryMode::kPointInTimeRecovery));
    } else {
      members.push_back(MakeMember(b, b_bytes));
    }
    auto merger = MakeMerger(std::move(members), {120, 121});
    DriveResult driven = Drive(merger.get());
    ASSERT_EQ(WALIndexMerger::Result::kError, driven.outcome);
    ASSERT_NE(std::string::npos,
              merger->GetError().ToString().find("121 head damaged"));
  }
}

INSTANTIATE_TEST_CASE_P(LogMerge, LogMergeTest,
                        ::testing::Combine(::testing::Values(0, 1),
                                           ::testing::Bool()));

}  // namespace ROCKSDB_NAMESPACE::log

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
