//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "utilities/latency_injection_fs.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "db/db_impl/db_impl.h"
#include "db/db_test_util.h"
#include "db/write_batch_internal.h"
#include "file/filename.h"
#include "port/port.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

namespace {

using Latency = LatencyInjectionFileSystem::Latency;
using Counters = LatencyInjectionFileSystem::Counters;

Latency MakeLatency(uint64_t append_micros, uint64_t sync_micros,
                    uint64_t jitter_micros, uint64_t append_bytes_per_sec = 0,
                    uint64_t sync_bytes_per_sec = 0) {
  Latency latency;
  latency.append_micros = append_micros;
  latency.sync_micros = sync_micros;
  latency.jitter_micros = jitter_micros;
  latency.append_bytes_per_sec = append_bytes_per_sec;
  latency.sync_bytes_per_sec = sync_bytes_per_sec;
  return latency;
}

// Counts which overload reached the file below the wrapper, so a test can
// tell that the wrapper forwarded each call through the same overload.
struct OverloadCounts {
  std::atomic<int> append{0};
  std::atomic<int> append_with_info{0};
  std::atomic<int> positioned_append{0};
  std::atomic<int> positioned_append_with_info{0};
};

class OverloadCountingFile : public FSWritableFileOwnerWrapper {
 public:
  OverloadCountingFile(std::unique_ptr<FSWritableFile>&& file,
                       OverloadCounts* counts)
      : FSWritableFileOwnerWrapper(std::move(file)), counts_(counts) {}

  IOStatus Append(const Slice& data, const IOOptions& options,
                  IODebugContext* dbg) override {
    counts_->append++;
    return target()->Append(data, options, dbg);
  }
  IOStatus Append(const Slice& data, const IOOptions& options,
                  const DataVerificationInfo& info,
                  IODebugContext* dbg) override {
    counts_->append_with_info++;
    return target()->Append(data, options, info, dbg);
  }
  IOStatus PositionedAppend(const Slice& data, uint64_t offset,
                            const IOOptions& options,
                            IODebugContext* dbg) override {
    counts_->positioned_append++;
    return target()->PositionedAppend(data, offset, options, dbg);
  }
  IOStatus PositionedAppend(const Slice& data, uint64_t offset,
                            const IOOptions& options,
                            const DataVerificationInfo& info,
                            IODebugContext* dbg) override {
    counts_->positioned_append_with_info++;
    return target()->PositionedAppend(data, offset, options, info, dbg);
  }

 private:
  OverloadCounts* counts_;
};

class OverloadCountingFS : public FileSystemWrapper {
 public:
  explicit OverloadCountingFS(const std::shared_ptr<FileSystem>& base)
      : FileSystemWrapper(base) {}
  const char* Name() const override { return "OverloadCountingFS"; }

  IOStatus NewWritableFile(const std::string& fname,
                           const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, file_opts, result, dbg);
    if (s.ok()) {
      result->reset(new OverloadCountingFile(std::move(*result), &counts));
    }
    return s;
  }

  OverloadCounts counts;
};

// Fails the first `failures` Sync/Fsync/RangeSync calls on any file it
// creates, then passes through. Appends always pass.
class SyncFailingFile : public FSWritableFileOwnerWrapper {
 public:
  SyncFailingFile(std::unique_ptr<FSWritableFile>&& file,
                  std::atomic<int>* remaining)
      : FSWritableFileOwnerWrapper(std::move(file)), remaining_(remaining) {}

  IOStatus Sync(const IOOptions& options, IODebugContext* dbg) override {
    if (TakeFailure()) {
      return IOStatus::IOError("injected sync failure");
    }
    return target()->Sync(options, dbg);
  }
  IOStatus Fsync(const IOOptions& options, IODebugContext* dbg) override {
    if (TakeFailure()) {
      return IOStatus::IOError("injected sync failure");
    }
    return target()->Fsync(options, dbg);
  }
  IOStatus RangeSync(uint64_t offset, uint64_t nbytes, const IOOptions& options,
                     IODebugContext* dbg) override {
    if (TakeFailure()) {
      return IOStatus::IOError("injected sync failure");
    }
    return target()->RangeSync(offset, nbytes, options, dbg);
  }

 private:
  bool TakeFailure() {
    int left = remaining_->load();
    while (left > 0 && !remaining_->compare_exchange_weak(left, left - 1)) {
    }
    return left > 0;
  }

  std::atomic<int>* remaining_;
};

class SyncFailingFS : public FileSystemWrapper {
 public:
  SyncFailingFS(const std::shared_ptr<FileSystem>& base, int failures)
      : FileSystemWrapper(base), remaining_(failures) {}
  const char* Name() const override { return "SyncFailingFS"; }

  IOStatus NewWritableFile(const std::string& fname,
                           const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, file_opts, result, dbg);
    if (s.ok()) {
      result->reset(new SyncFailingFile(std::move(*result), &remaining_));
    }
    return s;
  }

 private:
  std::atomic<int> remaining_;
};

}  // namespace

class LatencyInjectionFSTest : public DBTestBase {
 public:
  LatencyInjectionFSTest()
      : DBTestBase("latency_injection_fs_test", /*env_do_fsync=*/false) {}
  // Closes the DB while the wrapper it may have opened through is alive.
  ~LatencyInjectionFSTest() override { Close(); }

  // Installs a fresh wrapper over `base` (the default FileSystem when null)
  // with `latency` as the default, for the test's DB to open through.
  void InstallWrapper(const Latency& latency, uint64_t seed,
                      std::shared_ptr<FileSystem> base) {
    if (base == nullptr) {
      base = FileSystem::Default();
    }
    latency_fs_ =
        std::make_shared<LatencyInjectionFileSystem>(base, latency, seed);
    latency_env_ = NewCompositeEnv(latency_fs_);
  }

  Options WrappedOptions() {
    Options options = CurrentOptions();
    options.env = latency_env_.get();
    return options;
  }

  // Opens `name` in the DB directory through the wrapper, creating it.
  std::unique_ptr<FSWritableFile> OpenThroughWrapper(const std::string& name) {
    EXPECT_OK(latency_fs_->CreateDirIfMissing(dbname_, IOOptions(), nullptr));
    std::unique_ptr<FSWritableFile> file;
    EXPECT_OK(latency_fs_->NewWritableFile(dbname_ + "/" + name, FileOptions(),
                                           &file, nullptr));
    return file;
  }

  // Runs `op` and returns its wall time in microseconds.
  static uint64_t ElapsedMicros(const std::function<void()>& op) {
    const uint64_t start = SystemClock::Default()->NowMicros();
    op();
    return SystemClock::Default()->NowMicros() - start;
  }

  static void ExpectCountersEqual(const Counters& expected,
                                  const Counters& actual) {
    EXPECT_EQ(expected.appends, actual.appends);
    EXPECT_EQ(expected.append_bytes, actual.append_bytes);
    EXPECT_EQ(expected.syncs, actual.syncs);
    EXPECT_EQ(expected.injected_sleep_micros, actual.injected_sleep_micros);
    EXPECT_EQ(expected.size_sleep_micros, actual.size_sleep_micros);
  }

  // Counters accumulated by `op`, from the wrapper's aggregate.
  Counters AggregateDelta(const std::function<void()>& op) {
    const Counters before = latency_fs_->GetAggregateCounters();
    op();
    const Counters after = latency_fs_->GetAggregateCounters();
    Counters delta;
    delta.appends = after.appends - before.appends;
    delta.append_bytes = after.append_bytes - before.append_bytes;
    delta.syncs = after.syncs - before.syncs;
    delta.injected_sleep_micros =
        after.injected_sleep_micros - before.injected_sleep_micros;
    delta.size_sleep_micros =
        after.size_sleep_micros - before.size_sleep_micros;
    delta.wall_micros = after.wall_micros - before.wall_micros;
    return delta;
  }

  std::shared_ptr<LatencyInjectionFileSystem> latency_fs_;
  std::unique_ptr<Env> latency_env_;
};

TEST_F(LatencyInjectionFSTest, ElapsedTimeScalesWithAppendCount) {
  constexpr uint64_t kAppendMicros = 2000;
  InstallWrapper(MakeLatency(kAppendMicros, 0, 0), /*seed=*/0, nullptr);
  std::unique_ptr<FSWritableFile> wal = OpenThroughWrapper("000005.log");
  for (int appends : {5, 10}) {
    latency_fs_->ResetCounters();
    const uint64_t elapsed = ElapsedMicros([&] {
      for (int i = 0; i < appends; ++i) {
        ASSERT_OK(wal->Append("record", IOOptions(), nullptr));
      }
    });
    EXPECT_GE(elapsed, appends * kAppendMicros);
    const Counters counters = latency_fs_->GetWalCounters(5);
    EXPECT_EQ(static_cast<uint64_t>(appends), counters.appends);
    EXPECT_EQ(appends * kAppendMicros, counters.injected_sleep_micros);
    EXPECT_GE(counters.wall_micros, counters.injected_sleep_micros);
  }
  ASSERT_OK(wal->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, SyncAndNonSyncWritesDivergeAsConfigured) {
  // R1: a sync=false write pays only the append knob; a sync=true write pays
  // the append knob and the sync knob.
  constexpr uint64_t kAppendMicros = 1000;
  constexpr uint64_t kSyncMicros = 4000;
  constexpr int kWrites = 10;
  InstallWrapper(MakeLatency(kAppendMicros, kSyncMicros, 0), /*seed=*/0,
                 nullptr);
  Options options = WrappedOptions();
  options.manual_wal_flush = false;
  Reopen(options);

  WriteOptions async_wo;
  const Counters async_delta = AggregateDelta([&] {
    for (int i = 0; i < kWrites; ++i) {
      ASSERT_OK(Put("async" + std::to_string(i), "v", async_wo));
    }
  });
  EXPECT_EQ(static_cast<uint64_t>(kWrites), async_delta.appends);
  EXPECT_EQ(0u, async_delta.syncs);
  EXPECT_EQ(kWrites * kAppendMicros, async_delta.injected_sleep_micros);

  WriteOptions sync_wo;
  sync_wo.sync = true;
  const uint64_t sync_elapsed = ElapsedMicros([&] {
    const Counters sync_delta = AggregateDelta([&] {
      for (int i = 0; i < kWrites; ++i) {
        ASSERT_OK(Put("sync" + std::to_string(i), "v", sync_wo));
      }
    });
    EXPECT_EQ(static_cast<uint64_t>(kWrites), sync_delta.appends);
    EXPECT_EQ(static_cast<uint64_t>(kWrites), sync_delta.syncs);
    EXPECT_EQ(kWrites * (kAppendMicros + kSyncMicros),
              sync_delta.injected_sleep_micros);
  });
  EXPECT_GE(sync_elapsed, kWrites * (kAppendMicros + kSyncMicros));
  Close();
}

TEST_F(LatencyInjectionFSTest, NonWalFilesAreUntouched) {
  // R2: only files ParseFileName classifies as kWalFile are delayed or
  // counted. The latency is large so a stray delay would also be visible as
  // time, but the assertions are on the counters.
  InstallWrapper(MakeLatency(1000000, 1000000, 0), /*seed=*/0, nullptr);
  for (const std::string& name :
       {std::string("000007.sst"), std::string("MANIFEST-000001"),
        std::string("OPTIONS-000003"), std::string("000009.blob"),
        std::string("CURRENT"), std::string("LOG")}) {
    std::unique_ptr<FSWritableFile> file = OpenThroughWrapper(name);
    ASSERT_OK(file->Append("data", IOOptions(), nullptr));
    ASSERT_OK(file->Sync(IOOptions(), nullptr));
    ASSERT_OK(file->Fsync(IOOptions(), nullptr));
    ASSERT_OK(file->Close(IOOptions(), nullptr));
  }
  ExpectCountersEqual(Counters(), latency_fs_->GetAggregateCounters());
  EXPECT_TRUE(latency_fs_->GetWalNumbers().empty());

  // A whole DB life (WAL, flushed SST, MANIFEST, OPTIONS, CURRENT): only the
  // WALs the DB created are known to the wrapper.
  latency_fs_->SetDefaultLatency(Latency());
  DestroyAndReopen(WrappedOptions());
  ASSERT_OK(Put("key", "value"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("key2", "value2"));
  Close();
  std::vector<std::string> children;
  ASSERT_OK(env_->GetChildren(dbname_, &children));
  std::set<uint64_t> non_wal_numbers;
  for (const std::string& child : children) {
    uint64_t number = 0;
    FileType type;
    if (ParseFileName(child, &number, &type) && type != kWalFile &&
        number != 0) {
      non_wal_numbers.insert(number);
    }
  }
  EXPECT_FALSE(non_wal_numbers.empty());
  const std::vector<uint64_t> wal_numbers = latency_fs_->GetWalNumbers();
  EXPECT_FALSE(wal_numbers.empty());
  for (uint64_t number : wal_numbers) {
    EXPECT_EQ(0u, non_wal_numbers.count(number)) << number;
  }
}

TEST_F(LatencyInjectionFSTest, PerWalOverridesBehaveAsConfigured) {
  // R3: two WAL numbers configured differently each get their own latency,
  // and an override set while the WAL is open takes effect at once.
  constexpr uint64_t kSlowMicros = 3000;
  constexpr int kAppends = 5;
  InstallWrapper(Latency(), /*seed=*/0, nullptr);
  latency_fs_->SetWalLatency(10, MakeLatency(kSlowMicros, 0, 0));
  std::unique_ptr<FSWritableFile> slow = OpenThroughWrapper("000010.log");
  std::unique_ptr<FSWritableFile> fast = OpenThroughWrapper("000011.log");
  for (int i = 0; i < kAppends; ++i) {
    ASSERT_OK(slow->Append("x", IOOptions(), nullptr));
    ASSERT_OK(fast->Append("x", IOOptions(), nullptr));
  }
  EXPECT_EQ(kAppends * kSlowMicros,
            latency_fs_->GetWalCounters(10).injected_sleep_micros);
  EXPECT_EQ(0u, latency_fs_->GetWalCounters(11).injected_sleep_micros);

  latency_fs_->SetWalLatency(11, MakeLatency(0, kSlowMicros, 0));
  latency_fs_->ClearWalLatency(10);
  ASSERT_OK(slow->Sync(IOOptions(), nullptr));
  ASSERT_OK(fast->Sync(IOOptions(), nullptr));
  EXPECT_EQ(kAppends * kSlowMicros,
            latency_fs_->GetWalCounters(10).injected_sleep_micros);
  EXPECT_EQ(kSlowMicros, latency_fs_->GetWalCounters(11).injected_sleep_micros);
  ASSERT_OK(slow->Close(IOOptions(), nullptr));
  ASSERT_OK(fast->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, OneSlowPartitionAtTwoPartitions) {
  // R3 end to end: with two WAL partitions, only the WAL configured slow is
  // delayed, so one partition straggles while the other does not.
  constexpr uint64_t kSlowMicros = 2000;
  InstallWrapper(Latency(), /*seed=*/0, nullptr);
  Options options = WrappedOptions();
  options.wal_partitions = 2;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  options.create_missing_column_families = true;
  // Column families 1 and 2 route to different partitions or the search
  // below creates more until both partitions are covered.
  std::vector<int> cf_for_partition(2, -1);
  std::vector<std::string> cf_names;
  for (uint32_t cf_id = 1; cf_for_partition[0] < 0 || cf_for_partition[1] < 0;
       ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = 2;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(batch, 2, &partition));
    cf_names.push_back("cf" + std::to_string(cf_id));
    if (cf_for_partition[partition] < 0) {
      cf_for_partition[partition] = static_cast<int>(cf_id);
    }
  }
  CreateAndReopenWithCF(cf_names, options);
  const uint64_t slow_wal = dbfull()->TEST_CurrentWalNumber(1);
  const uint64_t fast_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_NE(slow_wal, fast_wal);
  latency_fs_->SetWalLatency(slow_wal, MakeLatency(kSlowMicros, 0, 0));
  latency_fs_->ResetCounters();
  constexpr int kWrites = 4;
  for (int i = 0; i < kWrites; ++i) {
    ASSERT_OK(Put(cf_for_partition[0], "k" + std::to_string(i), "v"));
    ASSERT_OK(Put(cf_for_partition[1], "k" + std::to_string(i), "v"));
  }
  EXPECT_EQ(kWrites * kSlowMicros,
            latency_fs_->GetWalCounters(slow_wal).injected_sleep_micros);
  EXPECT_EQ(static_cast<uint64_t>(kWrites),
            latency_fs_->GetWalCounters(slow_wal).appends);
  EXPECT_EQ(0u, latency_fs_->GetWalCounters(fast_wal).injected_sleep_micros);
  EXPECT_EQ(static_cast<uint64_t>(kWrites),
            latency_fs_->GetWalCounters(fast_wal).appends);
  Close();
}

TEST_F(LatencyInjectionFSTest, CountersMatchKnownWorkload) {
  // R4: per-WAL and aggregate counters for a workload of known shape, with
  // the injected sleep reported apart from (and bounded by) wall time.
  constexpr uint64_t kAppendMicros = 300;
  constexpr uint64_t kSyncMicros = 700;
  InstallWrapper(MakeLatency(kAppendMicros, kSyncMicros, 0), /*seed=*/0,
                 nullptr);
  std::unique_ptr<FSWritableFile> a = OpenThroughWrapper("000020.log");
  std::unique_ptr<FSWritableFile> b = OpenThroughWrapper("000021.log");
  for (size_t size : {1, 10, 100}) {
    ASSERT_OK(a->Append(std::string(size, 'a'), IOOptions(), nullptr));
  }
  ASSERT_OK(a->Sync(IOOptions(), nullptr));
  ASSERT_OK(b->Append(std::string(1000, 'b'), IOOptions(), nullptr));
  ASSERT_OK(b->Fsync(IOOptions(), nullptr));
  ASSERT_OK(b->RangeSync(0, 1000, IOOptions(), nullptr));

  Counters expected_a;
  expected_a.appends = 3;
  expected_a.append_bytes = 111;
  expected_a.syncs = 1;
  expected_a.injected_sleep_micros = 3 * kAppendMicros + kSyncMicros;
  ExpectCountersEqual(expected_a, latency_fs_->GetWalCounters(20));
  Counters expected_b;
  expected_b.appends = 1;
  expected_b.append_bytes = 1000;
  expected_b.syncs = 2;
  expected_b.injected_sleep_micros = kAppendMicros + 2 * kSyncMicros;
  ExpectCountersEqual(expected_b, latency_fs_->GetWalCounters(21));
  Counters expected_all;
  expected_all.appends = 4;
  expected_all.append_bytes = 1111;
  expected_all.syncs = 3;
  expected_all.injected_sleep_micros =
      expected_a.injected_sleep_micros + expected_b.injected_sleep_micros;
  const Counters all = latency_fs_->GetAggregateCounters();
  ExpectCountersEqual(expected_all, all);
  EXPECT_GE(all.wall_micros, all.injected_sleep_micros);
  EXPECT_EQ(std::vector<uint64_t>({20, 21}), latency_fs_->GetWalNumbers());
  EXPECT_NE(std::string::npos, latency_fs_->PrintCounters().find("WAL 21: "));

  latency_fs_->ResetCounters();
  ExpectCountersEqual(Counters(), latency_fs_->GetAggregateCounters());
  ExpectCountersEqual(Counters(), latency_fs_->GetWalCounters(20));
  ASSERT_OK(a->Close(IOOptions(), nullptr));
  ASSERT_OK(b->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, SameSeedGivesSameSchedule) {
  // R5: with jitter, the per-operation delays are a function of the seed,
  // the WAL number and the operation's index on that WAL, so two runs with
  // one seed match, interleaving across WALs does not matter, and another
  // seed gives another schedule.
  constexpr size_t kOps = 20;
  auto schedule = [&](uint64_t seed, bool interleave) {
    InstallWrapper(MakeLatency(10, 20, 200), seed, nullptr);
    std::unique_ptr<FSWritableFile> a = OpenThroughWrapper("000030.log");
    std::unique_ptr<FSWritableFile> b = OpenThroughWrapper("000031.log");
    std::vector<uint64_t> delays;
    auto step = [&](FSWritableFile* file, uint64_t number, size_t i) {
      const uint64_t before =
          latency_fs_->GetWalCounters(number).injected_sleep_micros;
      if (i % 3 == 2) {
        EXPECT_OK(file->Sync(IOOptions(), nullptr));
      } else {
        EXPECT_OK(file->Append("x", IOOptions(), nullptr));
      }
      delays.push_back(
          latency_fs_->GetWalCounters(number).injected_sleep_micros - before);
    };
    if (interleave) {
      for (size_t i = 0; i < kOps; ++i) {
        step(a.get(), 30, i);
        step(b.get(), 31, i);
      }
    } else {
      for (size_t i = 0; i < kOps; ++i) {
        step(a.get(), 30, i);
      }
      for (size_t i = 0; i < kOps; ++i) {
        step(b.get(), 31, i);
      }
    }
    EXPECT_OK(a->Close(IOOptions(), nullptr));
    EXPECT_OK(b->Close(IOOptions(), nullptr));
    // Per-WAL order: a's delays, then b's.
    std::vector<uint64_t> per_wal;
    for (size_t i = 0; i < delays.size(); ++i) {
      if (!interleave ? i < kOps : i % 2 == 0) {
        per_wal.push_back(delays[i]);
      }
    }
    for (size_t i = 0; i < delays.size(); ++i) {
      if (!interleave ? i >= kOps : i % 2 == 1) {
        per_wal.push_back(delays[i]);
      }
    }
    return per_wal;
  };
  const std::vector<uint64_t> first = schedule(42, /*interleave=*/false);
  EXPECT_EQ(first, schedule(42, /*interleave=*/false));
  EXPECT_EQ(first, schedule(42, /*interleave=*/true));
  EXPECT_NE(first, schedule(43, /*interleave=*/false));
  for (size_t i = 0; i < first.size(); ++i) {
    const bool is_sync = (i % kOps) % 3 == 2;
    EXPECT_GE(first[i], is_sync ? 20u : 10u) << i;
    EXPECT_LE(first[i], is_sync ? 220u : 210u) << i;
  }
}

TEST_F(LatencyInjectionFSTest, ZeroLatencyPassthroughMatchesUnwrapped) {
  // R6 and the passthrough requirement: with every knob at zero the wrapper
  // sleeps nowhere, and a DB opened through it writes, flushes, reopens and
  // reads exactly what one opened on the unwrapped FileSystem does.
  auto run = [&](bool wrapped) {
    Options options = CurrentOptions();
    if (wrapped) {
      InstallWrapper(Latency(), /*seed=*/0, nullptr);
      options.env = latency_env_.get();
    }
    DestroyAndReopen(options);
    for (int i = 0; i < 100; ++i) {
      EXPECT_OK(Put("key" + std::to_string(i), "value" + std::to_string(i)));
      if (i == 50) {
        EXPECT_OK(Flush());
      }
    }
    WriteOptions sync_wo;
    sync_wo.sync = true;
    EXPECT_OK(Put("synced", "yes", sync_wo));
    Reopen(options);
    std::vector<std::pair<std::string, std::string>> contents;
    std::unique_ptr<Iterator> it(db_->NewIterator(ReadOptions()));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      contents.emplace_back(it->key().ToString(), it->value().ToString());
    }
    EXPECT_OK(it->status());
    it.reset();
    Close();
    return contents;
  };
  const std::vector<std::pair<std::string, std::string>> unwrapped = run(false);
  const std::vector<std::pair<std::string, std::string>> wrapped = run(true);
  EXPECT_EQ(unwrapped, wrapped);
  EXPECT_EQ(101u, wrapped.size());
  const Counters counters = latency_fs_->GetAggregateCounters();
  EXPECT_GT(counters.appends, 0u);
  EXPECT_GT(counters.syncs, 0u);
  EXPECT_EQ(0u, counters.injected_sleep_micros);
}

TEST_F(LatencyInjectionFSTest, EveryEntryPointDelays) {
  // Q2: all seven entry points WritableFileWriter reaches the file through
  // are delayed by their knob, and each forwards through the same overload.
  // No WAL takes the direct-write (PositionedAppend) path at this tree, so
  // the PositionedAppend overloads are called directly on a WAL-named file.
  constexpr uint64_t kAppendMicros = 5000;
  constexpr uint64_t kSyncMicros = 7000;
  struct EntryPoint {
    const char* name;
    bool is_sync;
    std::function<IOStatus(FSWritableFile*)> call;
    std::atomic<int> OverloadCounts::* reached;
  };
  DataVerificationInfo info;
  const std::vector<EntryPoint> entry_points = {
      {"Append", false,
       [](FSWritableFile* f) { return f->Append("abc", IOOptions(), nullptr); },
       &OverloadCounts::append},
      {"Append(DataVerificationInfo)", false,
       [&](FSWritableFile* f) {
         return f->Append("abc", IOOptions(), info, nullptr);
       },
       &OverloadCounts::append_with_info},
      {"PositionedAppend", false,
       [](FSWritableFile* f) {
         return f->PositionedAppend("abc", 0, IOOptions(), nullptr);
       },
       &OverloadCounts::positioned_append},
      {"PositionedAppend(DataVerificationInfo)", false,
       [&](FSWritableFile* f) {
         return f->PositionedAppend("abc", 0, IOOptions(), info, nullptr);
       },
       &OverloadCounts::positioned_append_with_info},
      {"Sync", true,
       [](FSWritableFile* f) { return f->Sync(IOOptions(), nullptr); },
       nullptr},
      {"Fsync", true,
       [](FSWritableFile* f) { return f->Fsync(IOOptions(), nullptr); },
       nullptr},
      {"RangeSync", true,
       [](FSWritableFile* f) {
         return f->RangeSync(0, 3, IOOptions(), nullptr);
       },
       nullptr},
  };
  uint64_t wal_number = 40;
  for (const EntryPoint& entry : entry_points) {
    SCOPED_TRACE(entry.name);
    auto counting_fs =
        std::make_shared<OverloadCountingFS>(FileSystem::Default());
    InstallWrapper(MakeLatency(kAppendMicros, kSyncMicros, 0), /*seed=*/0,
                   counting_fs);
    std::unique_ptr<FSWritableFile> wal =
        OpenThroughWrapper(LogFileName("", wal_number).substr(1));
    const uint64_t delay = entry.is_sync ? kSyncMicros : kAppendMicros;
    const uint64_t elapsed =
        ElapsedMicros([&] { ASSERT_OK(entry.call(wal.get())); });
    EXPECT_GE(elapsed, delay);
    const Counters counters = latency_fs_->GetWalCounters(wal_number);
    EXPECT_EQ(delay, counters.injected_sleep_micros);
    EXPECT_EQ(entry.is_sync ? 0u : 1u, counters.appends);
    EXPECT_EQ(entry.is_sync ? 1u : 0u, counters.syncs);
    if (entry.reached != nullptr) {
      EXPECT_EQ(1, (counting_fs->counts.*entry.reached).load());
    }
    ASSERT_OK(wal->Close(IOOptions(), nullptr));
    ++wal_number;
  }
}

TEST_F(LatencyInjectionFSTest, ChecksumHandoffReachesVerificationOverload) {
  // Q2 through a real DB: with WAL checksum handoff on, WritableFileWriter
  // appends through the DataVerificationInfo overload, which the wrapper
  // delays and forwards as that overload.
  constexpr uint64_t kAppendMicros = 500;
  auto counting_fs =
      std::make_shared<OverloadCountingFS>(FileSystem::Default());
  InstallWrapper(MakeLatency(kAppendMicros, 0, 0), /*seed=*/0, counting_fs);
  Options options = WrappedOptions();
  options.checksum_handoff_file_types.Add(FileType::kWalFile);
  Reopen(options);
  const int with_info_before = counting_fs->counts.append_with_info.load();
  const Counters delta = AggregateDelta([&] {
    for (int i = 0; i < 5; ++i) {
      ASSERT_OK(Put("key" + std::to_string(i), "value"));
    }
  });
  EXPECT_EQ(5u, delta.appends);
  EXPECT_EQ(5 * kAppendMicros, delta.injected_sleep_micros);
  EXPECT_EQ(5, counting_fs->counts.append_with_info.load() - with_info_before);
  Close();
}

TEST_F(LatencyInjectionFSTest, SizeDelayMicrosArithmetic) {
  // Q1: the size term's formula and its edges: exact multiples, rounding up,
  // zero bytes, a zero rate, and a 4 GiB append at 1 byte/s (no overflow).
  using LIFS = LatencyInjectionFileSystem;
  EXPECT_EQ(0u, LIFS::SizeDelayMicros(0, 1000000));
  EXPECT_EQ(0u, LIFS::SizeDelayMicros(100, 0));
  EXPECT_EQ(100u, LIFS::SizeDelayMicros(100, 1000000));
  EXPECT_EQ(1u, LIFS::SizeDelayMicros(1, 1000000));
  EXPECT_EQ(1500u, LIFS::SizeDelayMicros(1500, 1000000));
  EXPECT_EQ(1u, LIFS::SizeDelayMicros(1, 3000000));
  EXPECT_EQ(1000000u, LIFS::SizeDelayMicros(1000000, 1000000));
  EXPECT_EQ(79u, LIFS::SizeDelayMicros(4096, 50u << 20));
  EXPECT_EQ(33334u, LIFS::SizeDelayMicros(100000, 3000000));
  EXPECT_EQ(4294967296ULL * 1000000ULL,
            LIFS::SizeDelayMicros(4294967296ULL, 1));
}

TEST_F(LatencyInjectionFSTest, AppendDelayScalesWithBytes) {
  // Q1: with base 0 and a known append bandwidth, each append sleeps
  // ceil(bytes * 1e6 / rate), reported as the size term inside the injected
  // sleep. A new default bandwidth applies to the already-open WAL at once.
  using LIFS = LatencyInjectionFileSystem;
  constexpr uint64_t kBps = 1000000;  // 1 byte/us: the delay equals the size.
  InstallWrapper(MakeLatency(0, 0, 0, kBps, 0), /*seed=*/0, nullptr);
  std::unique_ptr<FSWritableFile> wal = OpenThroughWrapper("000050.log");
  const uint64_t elapsed = ElapsedMicros([&] {
    for (size_t size : {0, 1, 100, 1500, 100000}) {
      ASSERT_OK(wal->Append(std::string(size, 'a'), IOOptions(), nullptr));
    }
  });
  constexpr uint64_t kExpected = 0 + 1 + 100 + 1500 + 100000;
  Counters counters = latency_fs_->GetWalCounters(50);
  EXPECT_EQ(kExpected, counters.size_sleep_micros);
  EXPECT_EQ(kExpected, counters.injected_sleep_micros);
  EXPECT_GE(elapsed, kExpected);
  EXPECT_EQ(5u, counters.appends);
  EXPECT_EQ(kExpected, counters.append_bytes);

  constexpr uint64_t kOddBps = 3000000;
  latency_fs_->SetDefaultLatency(MakeLatency(0, 0, 0, kOddBps, 0));
  ASSERT_OK(wal->Append(std::string(100000, 'b'), IOOptions(), nullptr));
  constexpr uint64_t kOddExpected = 33334;  // ceil(1e5 * 1e6 / 3e6).
  EXPECT_EQ(kOddExpected, LIFS::SizeDelayMicros(100000, kOddBps));
  counters = latency_fs_->GetWalCounters(50);
  EXPECT_EQ(kExpected + kOddExpected, counters.size_sleep_micros);
  EXPECT_EQ(kExpected + kOddExpected, counters.injected_sleep_micros);
  ASSERT_OK(wal->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, AppendBandwidthPerWalOverride) {
  // Q1: each WAL is charged by its own bandwidth: an override on one WAL,
  // the default on another, and the default again once cleared.
  InstallWrapper(MakeLatency(0, 0, 0, 1000000, 0), /*seed=*/0, nullptr);
  latency_fs_->SetWalLatency(51, MakeLatency(0, 0, 0, 2000000, 0));
  std::unique_ptr<FSWritableFile> slow = OpenThroughWrapper("000051.log");
  std::unique_ptr<FSWritableFile> fast = OpenThroughWrapper("000052.log");
  ASSERT_OK(slow->Append(std::string(10000, 'a'), IOOptions(), nullptr));
  ASSERT_OK(fast->Append(std::string(10000, 'b'), IOOptions(), nullptr));
  EXPECT_EQ(5000u, latency_fs_->GetWalCounters(51).size_sleep_micros);
  EXPECT_EQ(10000u, latency_fs_->GetWalCounters(52).size_sleep_micros);
  latency_fs_->ClearWalLatency(51);
  ASSERT_OK(slow->Append(std::string(10000, 'c'), IOOptions(), nullptr));
  EXPECT_EQ(15000u, latency_fs_->GetWalCounters(51).size_sleep_micros);
  ASSERT_OK(slow->Close(IOOptions(), nullptr));
  ASSERT_OK(fast->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, AppendSizeTermOnEveryAppendEntryPoint) {
  // Q1: all four append overloads charge the size term (the EveryEntryPoint
  // pattern, for the size term).
  InstallWrapper(MakeLatency(0, 0, 0, 1000000, 0), /*seed=*/0, nullptr);
  DataVerificationInfo info;
  const std::vector<std::function<IOStatus(FSWritableFile*)>> entry_points = {
      [](FSWritableFile* f) { return f->Append("abc", IOOptions(), nullptr); },
      [&](FSWritableFile* f) {
        return f->Append("abc", IOOptions(), info, nullptr);
      },
      [](FSWritableFile* f) {
        return f->PositionedAppend("abc", 0, IOOptions(), nullptr);
      },
      [&](FSWritableFile* f) {
        return f->PositionedAppend("abc", 0, IOOptions(), info, nullptr);
      },
  };
  uint64_t wal_number = 60;
  for (const auto& call : entry_points) {
    std::unique_ptr<FSWritableFile> wal =
        OpenThroughWrapper("0000" + std::to_string(wal_number) + ".log");
    ASSERT_OK(call(wal.get()));
    const Counters counters = latency_fs_->GetWalCounters(wal_number);
    EXPECT_EQ(3u, counters.size_sleep_micros) << wal_number;
    EXPECT_EQ(3u, counters.injected_sleep_micros) << wal_number;
    EXPECT_EQ(1u, counters.appends) << wal_number;
    ASSERT_OK(wal->Close(IOOptions(), nullptr));
    ++wal_number;
  }
}

TEST_F(LatencyInjectionFSTest, SyncChargesBytesSinceLastSync) {
  // Q2: each Sync/Fsync sleeps for the bytes appended since the previous
  // successful sync at the sync bandwidth and resets the count; a sync with
  // nothing new sleeps only its base.
  constexpr uint64_t kSyncBase = 100;
  InstallWrapper(MakeLatency(0, kSyncBase, 0, 0, 1000000), /*seed=*/0, nullptr);
  std::unique_ptr<FSWritableFile> wal = OpenThroughWrapper("000070.log");
  ASSERT_OK(wal->Append(std::string(1000, 'a'), IOOptions(), nullptr));
  ASSERT_OK(wal->Sync(IOOptions(), nullptr));
  ASSERT_OK(wal->Append(std::string(2000, 'b'), IOOptions(), nullptr));
  ASSERT_OK(wal->Fsync(IOOptions(), nullptr));
  ASSERT_OK(wal->Sync(IOOptions(), nullptr));
  const Counters counters = latency_fs_->GetWalCounters(70);
  EXPECT_EQ(3u, counters.syncs);
  EXPECT_EQ(3000u, counters.append_bytes);
  EXPECT_EQ(3000u, counters.size_sleep_micros);
  EXPECT_EQ(3 * kSyncBase + 3000, counters.injected_sleep_micros);
  ASSERT_OK(wal->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, RangeSyncChargesUnsyncedPartOfRange) {
  // Q2: a RangeSync of n bytes is charged for min(n, unsynced) and lowers
  // the count by that amount.
  InstallWrapper(MakeLatency(0, 0, 0, 0, 1000000), /*seed=*/0, nullptr);
  std::unique_ptr<FSWritableFile> wal = OpenThroughWrapper("000071.log");
  ASSERT_OK(wal->Append(std::string(5000, 'a'), IOOptions(), nullptr));
  ASSERT_OK(wal->RangeSync(0, 2000, IOOptions(), nullptr));
  EXPECT_EQ(2000u, latency_fs_->GetWalCounters(71).size_sleep_micros);
  ASSERT_OK(wal->RangeSync(0, 10000, IOOptions(), nullptr));
  EXPECT_EQ(5000u, latency_fs_->GetWalCounters(71).size_sleep_micros);
  ASSERT_OK(wal->RangeSync(0, 100, IOOptions(), nullptr));
  ASSERT_OK(wal->Sync(IOOptions(), nullptr));
  const Counters counters = latency_fs_->GetWalCounters(71);
  EXPECT_EQ(4u, counters.syncs);
  EXPECT_EQ(5000u, counters.size_sleep_micros);
  EXPECT_EQ(5000u, counters.injected_sleep_micros);
  ASSERT_OK(wal->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, FailedSyncKeepsUnsyncedBytes) {
  // Q2: a failed sync sleeps for its claim but makes nothing durable: the
  // bytes stay unsynced and the next successful sync charges them again.
  auto failing_fs =
      std::make_shared<SyncFailingFS>(FileSystem::Default(), /*failures=*/1);
  InstallWrapper(MakeLatency(0, 0, 0, 0, 1000000), /*seed=*/0, failing_fs);
  std::unique_ptr<FSWritableFile> wal = OpenThroughWrapper("000072.log");
  ASSERT_OK(wal->Append(std::string(1000, 'a'), IOOptions(), nullptr));
  EXPECT_FALSE(wal->Sync(IOOptions(), nullptr).ok());
  ASSERT_OK(wal->Append(std::string(2000, 'b'), IOOptions(), nullptr));
  ASSERT_OK(wal->Sync(IOOptions(), nullptr));
  const Counters counters = latency_fs_->GetWalCounters(72);
  EXPECT_EQ(2u, counters.syncs);
  EXPECT_EQ(3000u, counters.append_bytes);
  EXPECT_EQ(4000u, counters.size_sleep_micros);
  EXPECT_EQ(4000u, counters.injected_sleep_micros);
  ASSERT_OK(wal->Close(IOOptions(), nullptr));
}

TEST_F(LatencyInjectionFSTest, SyncSizeTermThroughDbWrites) {
  // Q2 through a real DB: with a sync bandwidth set, sync=true Puts at N=1
  // record a size term of one ceiling per sync over the append_bytes the
  // counters show: between ceil(total) and ceil(total) + syncs.
  using LIFS = LatencyInjectionFileSystem;
  constexpr uint64_t kSyncBps = 50u << 20;  // 50 MiB/s.
  InstallWrapper(MakeLatency(0, 0, 0, 0, kSyncBps), /*seed=*/0, nullptr);
  Options options = WrappedOptions();
  options.manual_wal_flush = false;
  Reopen(options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("warmup", "v", sync_wo));  // Drains any unsynced bytes.
  constexpr int kWrites = 10;
  const Counters delta = AggregateDelta([&] {
    for (int i = 0; i < kWrites; ++i) {
      ASSERT_OK(Put("k" + std::to_string(i), std::string(1000, 'v'), sync_wo));
    }
  });
  EXPECT_EQ(static_cast<uint64_t>(kWrites), delta.appends);
  EXPECT_EQ(static_cast<uint64_t>(kWrites), delta.syncs);
  const uint64_t floor = LIFS::SizeDelayMicros(delta.append_bytes, kSyncBps);
  EXPECT_GE(delta.size_sleep_micros, floor);
  EXPECT_LE(delta.size_sleep_micros, floor + delta.syncs);
  EXPECT_EQ(delta.size_sleep_micros, delta.injected_sleep_micros);
  Close();
}

TEST_F(LatencyInjectionFSTest, ConcurrentAppendsAndSyncsChargeEveryByteOnce) {
  // Q2: concurrent appends and syncs on one WAL charge every appended byte
  // exactly once: at 1 byte/us the size term equals append_bytes. Appends
  // are serialized (the file below makes no thread-safety promise); syncs
  // race with everything, which is what the claim has to survive.
  constexpr int kThreads = 4;
  constexpr int kAppendsPerThread = 50;
  constexpr size_t kSize = 100;
  InstallWrapper(MakeLatency(0, 0, 0, 0, 1000000), /*seed=*/0, nullptr);
  std::unique_ptr<FSWritableFile> wal = OpenThroughWrapper("000073.log");
  std::mutex append_mu;
  std::atomic<int> failures{0};
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kAppendsPerThread; ++i) {
        {
          std::lock_guard<std::mutex> lock(append_mu);
          if (!wal->Append(std::string(kSize, 'x'), IOOptions(), nullptr)
                   .ok()) {
            failures.fetch_add(1);
            return;
          }
        }
        if (!wal->Sync(IOOptions(), nullptr).ok()) {
          failures.fetch_add(1);
          return;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(0, failures.load());
  ASSERT_OK(wal->Sync(IOOptions(), nullptr));
  const Counters counters = latency_fs_->GetWalCounters(73);
  EXPECT_EQ(kThreads * kAppendsPerThread * kSize, counters.append_bytes);
  EXPECT_EQ(counters.append_bytes, counters.size_sleep_micros);
  EXPECT_EQ(counters.size_sleep_micros, counters.injected_sleep_micros);
  ASSERT_OK(wal->Close(IOOptions(), nullptr));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
