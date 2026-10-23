//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// Prefix-property crash oracle for partitioned WALs.
//
// Each cycle opens a DB with N WAL partitions and WAL indexing, runs several
// writer threads over column families spread across every partition (mixed
// sync=true and sync=false writes, coupled rotations, flushes and SyncWAL
// calls), crashes at a point chosen by the cycle's seed, recovers with
// point-in-time recovery and checks the recovered state against the
// history the writers recorded:
//  - No hole: the recovered writes are exactly the writes with sequence 1..S,
//    S the highest recovered sequence. Writes that reported failure count
//    too: the crash is the only fault, so no index is burned and covered, and
//    a failed write is recovered only if its record landed, below S.
//  - Nothing recovered that was not written, and every recovered value and
//    sequence is the one written.
//  - Process crash (nothing unsynced is dropped): every acknowledged write is
//    recovered, and S is at least the runtime durability frontier
//    (WalDurabilitySafePoint) at the crash, so the runtime frontier and the
//    recovery cut agree.
//  - Power loss (unsynced data dropped): every acknowledged sync=true write is
//    recovered, and with it everything acknowledged below it.
//  - Published but lost (off by default, see kCheckPublished below).
// A hole entirely below the lowest sequence recovery replayed from the WALs
// fails like any other: strict is the only mode (see Verify).
//
// The cycle's seed fixes its configuration: N, the crash kind, the crash
// point (the n-th acknowledgement, WAL append, leadership sync or gate park),
// the writers' schedules and the slow partition. Thread interleaving is not
// controlled, so a failure reproduces from its seed with the same configuration
// and crash point, but may need reruns to hit the same interleaving.
//
// Environment variables:
//   ROCKSDB_WAL_ORACLE_CYCLES  cycles to run (default kDefaultCycles)
//   ROCKSDB_WAL_ORACLE_SEED    seed of the first cycle; cycle i uses seed + i
//   ROCKSDB_WAL_ORACLE_CHECK_PUBLISHED  1 turns the published-but-lost check
//                              on (see kCheckPublished)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "db/column_family.h"
#include "db/db_impl/db_impl.h"
#include "db/db_test_util.h"
#include "db/write_batch_internal.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/utilities/debug.h"
#include "test_util/sync_point.h"
#include "test_util/testharness.h"
#include "util/coding.h"
#include "util/random.h"
#include "utilities/fault_injection_fs.h"
#include "utilities/latency_injection_fs.h"

namespace ROCKSDB_NAMESPACE {

namespace {

constexpr uint64_t kDefaultCycles = 40;
// Far above any cycle's run time, sanitizer builds included.
constexpr int kCycleTimeoutSeconds = 300;
constexpr uint64_t kDefaultSeed = 20261019;
// Cycle budget of the two neuter self-tests below. Thread interleaving is not
// controlled, so the cycle that first exposes an injected off-by-one varies
// from run to run (cycles 8 to 73 observed for the merge stop, and one run of
// 100 missed it). They stop at the first violation, so a large budget costs
// time only when the oracle has really gone blind.
constexpr uint64_t kSelfTestMaxCycles = 400;

// Published but lost (I2): a sequence a reader could observe through
// GetLatestSequenceNumber() before a process crash is missing after it.
// Off by default under D-G54 (a): visibility is not yet bounded by the
// durability safe point (G-10, N-2-1), so publication can run ahead of
// durability until GC07 lands. GC07 turns it on. Cycles where it would fail
// are counted and printed either way.
constexpr bool kCheckPublished = false;

uint64_t EnvU64(const char* name, uint64_t default_value) {
  const char* value = getenv(name);
  return value == nullptr ? default_value : std::strtoull(value, nullptr, 10);
}

enum class CrashKind {
  // The process dies: everything written to the file system stays.
  kProcessCrash,
  // Power loss: every WAL loses what was not synced.
  kPowerLoss,
};

const char* CrashKindName(CrashKind kind) {
  return kind == CrashKind::kProcessCrash ? "process_crash" : "power_loss";
}

// The event whose crash_at-th occurrence crashes the cycle.
enum class CrashTrigger {
  // A write is acknowledged (returns OK).
  kAck,
  // A WAL append starts (before the record is written).
  kAppend,
  // A sync write's leadership sync completed, before its completion gate.
  kLeadershipSync,
  // A completion gate parks, waiting for a lower record on another partition.
  kPark,
};

const char* CrashTriggerName(CrashTrigger trigger) {
  switch (trigger) {
    case CrashTrigger::kAck:
      return "ack";
    case CrashTrigger::kAppend:
      return "append";
    case CrashTrigger::kLeadershipSync:
      return "leadership_sync";
    case CrashTrigger::kPark:
      return "park";
  }
  return "";
}

constexpr const char* kAppendSyncPoint = "DBImpl::WriteToWAL:log_entry";

// The sync point marking the trigger's event, or nullptr for kAck, which the
// writers count themselves.
const char* CrashTriggerSyncPoint(CrashTrigger trigger) {
  switch (trigger) {
    case CrashTrigger::kAck:
      return nullptr;
    case CrashTrigger::kAppend:
      return kAppendSyncPoint;
    case CrashTrigger::kLeadershipSync:
      return "DBImpl::WriteGroupToWAL:AfterLeadershipSync";
    case CrashTrigger::kPark:
      return "DBImpl::WaitForWalDurability:BeforePark";
  }
  return nullptr;
}

// Everything a cycle does, derived from its seed.
struct CycleConfig {
  uint64_t seed = 0;
  size_t partitions = 2;
  CrashKind crash = CrashKind::kProcessCrash;
  int threads = 4;
  int writes_per_thread = 100;
  // The crash fires at the crash_at-th occurrence of the trigger event, or
  // when the writers finish if it never occurs that often.
  CrashTrigger trigger = CrashTrigger::kAck;
  uint64_t crash_at = 1;
  // A write is sync=true one time in this many.
  uint64_t sync_one_in = 8;
  // Thread 0 rotates (switches every partition's WAL) or flushes one time in
  // this many writes, and calls SyncWAL one time in this many.
  uint64_t rotate_one_in = 40;
  uint64_t sync_wal_one_in = 60;
  // When set, partition slow_partition's current WAL is delayed by
  // slow_latency, re-applied after each rotation thread 0 makes.
  bool slow = false;
  size_t slow_partition = 0;
  LatencyInjectionFileSystem::Latency slow_latency;
  // When non-zero, a write whose first sequence is a multiple of
  // delay_append_every waits delay_append_micros after its sequence and index
  // are allocated and before it takes its WAL's mutex to append: the window
  // in which a later write on another partition can append and be synced
  // above it. The WAL file system cannot open it, because it delays appends
  // under the mutex.
  uint64_t delay_append_every = 0;
  uint64_t delay_append_micros = 0;

  std::string ToString() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "seed=%" PRIu64 " N=%zu crash=%s trigger=%s crash_at=%" PRIu64
             " threads=%d writes_per_thread=%d sync_one_in=%" PRIu64
             " slow=%d slow_partition=%zu slow_append_us=%" PRIu64
             " slow_sync_us=%" PRIu64 " delay_append_every=%" PRIu64
             " delay_append_us=%" PRIu64,
             seed, partitions, CrashKindName(crash), CrashTriggerName(trigger),
             crash_at, threads, writes_per_thread, sync_one_in, slow ? 1 : 0,
             slow_partition, slow_latency.append_micros,
             slow_latency.sync_micros, delay_append_every, delay_append_micros);
    return buf;
  }
};

CycleConfig ConfigForSeed(uint64_t seed) {
  Random64 rnd(seed);
  CycleConfig config;
  config.seed = seed;
  config.partitions = rnd.OneIn(2) ? 2 : 8;
  config.crash =
      rnd.OneIn(2) ? CrashKind::kProcessCrash : CrashKind::kPowerLoss;
  config.threads = 2 + static_cast<int>(rnd.Uniform(4));
  config.writes_per_thread = 40 + static_cast<int>(rnd.Uniform(100));
  config.sync_one_in = 1 + rnd.Uniform(12);
  const uint64_t writes = static_cast<uint64_t>(config.threads) *
                          static_cast<uint64_t>(config.writes_per_thread);
  config.trigger = static_cast<CrashTrigger>(rnd.Uniform(4));
  switch (config.trigger) {
    case CrashTrigger::kAck:
    case CrashTrigger::kAppend:
      config.crash_at = 1 + rnd.Uniform(writes);
      break;
    case CrashTrigger::kLeadershipSync:
      config.crash_at = 1 + rnd.Uniform(writes / config.sync_one_in + 1);
      break;
    case CrashTrigger::kPark:
      config.crash_at = 1 + rnd.Uniform(16);
      break;
  }
  config.rotate_one_in = 10 + rnd.Uniform(50);
  config.sync_wal_one_in = 10 + rnd.Uniform(80);
  config.slow = rnd.OneIn(2);
  config.slow_partition = static_cast<size_t>(rnd.Uniform(config.partitions));
  if (rnd.OneIn(2)) {
    config.slow_latency.append_micros = 200 + rnd.Uniform(1500);
  } else {
    config.slow_latency.sync_micros = 500 + rnd.Uniform(4000);
  }
  if (rnd.OneIn(2)) {
    config.delay_append_every = 2 + rnd.Uniform(8);
    config.delay_append_micros = 100 + rnd.Uniform(1000);
  }
  return config;
}

// One write a writer issued: a single Put of a unique key.
struct WriteRecord {
  int cf = 0;
  bool sync = false;
  bool issued = false;
  bool acked = false;
  SequenceNumber seq = 0;  // valid when acked
};

std::string KeyFor(int thread, int index) {
  char buf[32];
  snprintf(buf, sizeof(buf), "t%02d_%06d", thread, index);
  return buf;
}

std::string ValueFor(uint64_t seed, int thread, int index) {
  Random rnd(static_cast<uint32_t>(seed * 1000003 + thread * 7919 + index));
  return KeyFor(thread, index) + ":" +
         rnd.RandomString(16 + static_cast<int>(rnd.Uniform(200)));
}

struct CycleResult {
  std::vector<std::string> violations;
  bool published_lost = false;
  // A power-loss cycle lost an acknowledged write (see Verify): the proof
  // the crash actually dropped unsynced data (G-54).
  bool power_loss_lost_write = false;
  // Unexplained holes the recovery's merge met (each stops replay there).
  uint64_t holes = 0;
  uint64_t acked = 0;
  uint64_t recovered = 0;
};

}  // namespace

class DBWalOracleTest : public DBTestBase {
 public:
  DBWalOracleTest()
      : DBTestBase("db_wal_oracle_test", /*env_do_fsync=*/false) {}
  // Closes the DB while the environment it opened through is alive.
  ~DBWalOracleTest() override { Close(); }

  // Runs one crash/recover cycle and checks the recovered state.
  CycleResult RunCycle(const CycleConfig& config);

  // Runs `cycles` cycles from `seed` (cycle i uses seed + i), each configured
  // by ConfigForSeed and then `adjust`, and returns the 1-based index of the
  // first cycle with a violation, or 0 when every cycle passed. With
  // `stop_at_violation` false every cycle runs and each violation is reported
  // as a test failure. The out-parameters report how many power-loss cycles
  // ran and how many of them lost an acknowledged write.
  uint64_t RunCycles(uint64_t seed, uint64_t cycles,
                     const std::function<void(CycleConfig*)>& adjust,
                     bool stop_at_violation,
                     uint64_t* power_loss_cycles_out = nullptr,
                     uint64_t* power_loss_lost_out = nullptr);

  // Self-test only: the recovery's merge treats the first unexplained hole
  // as explained, stopping one index late.
  bool inject_merge_stop_off_by_one_ = false;

 private:
  // Creates column families until every partition has one, and returns the
  // handle index of one column family per partition.
  std::vector<int> OpenWithEveryPartitionCovered(
      const Options& options, std::vector<std::string>* names);
  void Verify(const CycleConfig& config,
              const std::vector<std::vector<WriteRecord>>& history,
              SequenceNumber safe_point, SequenceNumber published,
              CycleResult* result);

  std::unique_ptr<Env> cycle_env_;
};

std::vector<int> DBWalOracleTest::OpenWithEveryPartitionCovered(
    const Options& options, std::vector<std::string>* names) {
  const size_t partitions = options.wal_partitions;
  std::vector<int> cf_for_partition(partitions, -1);
  size_t covered = 0;
  std::vector<std::string> cfs;
  for (uint32_t cf_id = 1; covered < partitions; ++cf_id) {
    WriteBatch batch;
    EXPECT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = partitions;
    EXPECT_OK(DBImpl::TEST_WalPartitionForWrite(batch, partitions, &partition));
    cfs.push_back("cf" + std::to_string(cf_id));
    if (cf_for_partition[partition] < 0) {
      // Handle indexes follow creation order after "default".
      cf_for_partition[partition] = static_cast<int>(cf_id);
      ++covered;
    }
  }
  DestroyAndReopen(options);
  CreateAndReopenWithCF(cfs, options);
  names->assign(1, kDefaultColumnFamilyName);
  names->insert(names->end(), cfs.begin(), cfs.end());
  return cf_for_partition;
}

CycleResult DBWalOracleTest::RunCycle(const CycleConfig& config) {
  CycleResult result;
  // A cycle that never finishes (a completion gate parked for good) fails
  // with its configuration instead of hanging the test.
  std::mutex watchdog_mu;
  std::condition_variable watchdog_cv;
  bool cycle_done = false;
  port::Thread watchdog([&] {
    std::unique_lock<std::mutex> lock(watchdog_mu);
    if (!watchdog_cv.wait_for(lock, std::chrono::seconds(kCycleTimeoutSeconds),
                              [&] { return cycle_done; })) {
      fprintf(stderr, "prefix oracle: cycle hung: %s\n",
              config.ToString().c_str());
      std::abort();
    }
  });
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(FileSystem::Default());
  // Unsynced data lives in memory until synced, so a power-loss crash loses
  // exactly what no sync covered.
  fault_fs->SetInjectUnsyncedDataLoss(config.crash == CrashKind::kPowerLoss);
  auto latency_fs = std::make_shared<LatencyInjectionFileSystem>(
      fault_fs, LatencyInjectionFileSystem::Latency(), config.seed);
  // The previous cycle's environment stays alive until this cycle's open has
  // destroyed the previous DB through it.
  std::unique_ptr<Env> env = NewCompositeEnv(latency_fs);

  Options options = CurrentOptions();
  options.env = env.get();
  options.wal_partitions = static_cast<uint32_t>(config.partitions);
  options.experimental_allow_wal_partitions_for_testing = true;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  // The WALs stay behind for recovery rather than being flushed away, and
  // sequence numbers stay as written so recovered writes can be matched.
  options.avoid_flush_during_shutdown = true;
  options.avoid_flush_during_recovery = true;
  options.disable_auto_compactions = true;
  options.max_write_buffer_number = 16;
  options.write_buffer_size = 4 << 20;

  std::vector<std::string> cf_names;
  const std::vector<int> cf_for_partition =
      OpenWithEveryPartitionCovered(options, &cf_names);
  cycle_env_ = std::move(env);

  auto apply_slow_latency = [&] {
    if (config.slow) {
      latency_fs->SetWalLatency(
          dbfull()->TEST_CurrentWalNumber(config.slow_partition),
          config.slow_latency);
    }
  };
  apply_slow_latency();

  std::vector<std::vector<WriteRecord>> history(
      config.threads, std::vector<WriteRecord>(config.writes_per_thread));
  std::atomic<uint64_t> events{0};
  std::atomic<bool> crashed{false};
  std::mutex crash_mu;
  auto crash = [&] {
    std::lock_guard<std::mutex> lock(crash_mu);
    if (!crashed.load()) {
      fault_fs->SetFilesystemActive(false);
      crashed.store(true);
    }
  };
  auto count_event = [&] {
    if (events.fetch_add(1) + 1 == config.crash_at) {
      crash();
    }
  };
  const char* trigger_point = CrashTriggerSyncPoint(config.trigger);
  if (trigger_point != nullptr && trigger_point != kAppendSyncPoint) {
    SyncPoint::GetInstance()->SetCallBack(trigger_point,
                                          [&](void*) { count_event(); });
  }
  const bool hook_append =
      trigger_point == kAppendSyncPoint || config.delay_append_every != 0;
  if (hook_append) {
    SyncPoint::GetInstance()->SetCallBack(kAppendSyncPoint, [&](void* arg) {
      if (config.delay_append_every != 0) {
        const Slice* entry = static_cast<const Slice*>(arg);
        if (DecodeFixed64(entry->data()) % config.delay_append_every == 0) {
          SystemClock::Default()->SleepForMicroseconds(
              static_cast<int>(config.delay_append_micros));
        }
      }
      if (trigger_point == kAppendSyncPoint) {
        count_event();
      }
    });
  }
  uint64_t holes = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "WALIndexMerger::Next:PristineHole", [&](void* arg) {
        ++holes;
        if (inject_merge_stop_off_by_one_ && holes == 1) {
          *static_cast<bool*>(arg) = true;
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();

  auto writer = [&](int thread) {
    Random64 rnd(config.seed * 31 + static_cast<uint64_t>(thread) + 1);
    for (int i = 0; i < config.writes_per_thread && !crashed.load(); ++i) {
      if (thread == 0 && i > 0 && rnd.OneIn(config.rotate_one_in)) {
        const int cf = cf_for_partition[rnd.Uniform(config.partitions)];
        if (rnd.OneIn(2)) {
          ColumnFamilyData* cfd =
              static_cast_with_check<ColumnFamilyHandleImpl>(handles_[cf])
                  ->cfd();
          dbfull()->TEST_SwitchMemtable(cfd).PermitUncheckedError();
        } else {
          FlushOptions flush_options;
          flush_options.wait = false;
          db_->Flush(flush_options, handles_[cf]).PermitUncheckedError();
        }
        apply_slow_latency();
      }
      if (thread == 0 && rnd.OneIn(config.sync_wal_one_in)) {
        db_->SyncWAL().PermitUncheckedError();
      }
      WriteRecord& record = history[thread][i];
      record.cf = cf_for_partition[rnd.Uniform(config.partitions)];
      record.sync = rnd.OneIn(config.sync_one_in);
      record.issued = true;
      WriteBatch batch;
      EXPECT_OK(batch.Put(handles_[record.cf], KeyFor(thread, i),
                          ValueFor(config.seed, thread, i)));
      WriteOptions write_options;
      write_options.sync = record.sync;
      const Status s = db_->Write(write_options, &batch);
      if (s.ok()) {
        record.acked = true;
        record.seq = WriteBatchInternal::Sequence(&batch);
        if (config.trigger == CrashTrigger::kAck) {
          count_event();
        }
      }
    }
  };

  std::vector<port::Thread> writers;
  for (int t = 0; t < config.threads; ++t) {
    writers.emplace_back(writer, t);
  }
  for (port::Thread& t : writers) {
    t.join();
  }
  // Every writer finished before the crash point (or writes failed first):
  // crash at quiescence.
  crash();
  if (trigger_point != nullptr) {
    SyncPoint::GetInstance()->ClearCallBack(trigger_point);
  }
  if (hook_append) {
    SyncPoint::GetInstance()->ClearCallBack(kAppendSyncPoint);
  }

  // Whatever the frontier covers was written before the crash: an append the
  // crash failed never completes its range, so the frontier never passes it.
  const SequenceNumber safe_point = dbfull()->WalDurabilitySafePoint();
  const SequenceNumber published = db_->GetLatestSequenceNumber();
  Close();
  if (config.crash == CrashKind::kPowerLoss) {
    // Forget the unsynced data the file system kept in memory to serve reads
    // of files open for write, so recovery reads only what was synced.
    EXPECT_OK(fault_fs->DropUnsyncedFileData());
  }
  fault_fs->SetFilesystemActive(true);

  holes = 0;
  const Status open_status = TryReopenWithColumnFamilies(cf_names, options);
  SyncPoint::GetInstance()->ClearCallBack("WALIndexMerger::Next:PristineHole");
  result.holes = holes;
  if (!open_status.ok()) {
    result.violations.push_back("recovery failed: " + open_status.ToString());
  } else {
    Verify(config, history, safe_point, published, &result);
  }
  Close();
  {
    std::lock_guard<std::mutex> lock(watchdog_mu);
    cycle_done = true;
  }
  watchdog_cv.notify_all();
  watchdog.join();
  return result;
}

void DBWalOracleTest::Verify(
    const CycleConfig& config,
    const std::vector<std::vector<WriteRecord>>& history,
    SequenceNumber safe_point, SequenceNumber published, CycleResult* result) {
  // Recovered key -> (sequence, value), from every column family.
  std::map<std::string, std::pair<SequenceNumber, std::string>> recovered;
  for (ColumnFamilyHandle* handle : handles_) {
    std::vector<KeyVersion> versions;
    const Status s = GetAllKeyVersions(db_.get(), handle, OptSlice(),
                                       OptSlice(), 1 << 20, &versions);
    if (!s.ok()) {
      result->violations.push_back("GetAllKeyVersions: " + s.ToString());
      return;
    }
    for (const KeyVersion& version : versions) {
      recovered[version.user_key] = {version.sequence, version.value};
    }
  }
  result->recovered = recovered.size();

  auto violation = [&](const std::string& what) {
    if (result->violations.size() < 8) {
      result->violations.push_back(what);
    }
  };
  SequenceNumber max_recovered = 0;
  size_t matched = 0;
  for (int t = 0; t < config.threads; ++t) {
    for (int i = 0; i < config.writes_per_thread; ++i) {
      const WriteRecord& record = history[t][i];
      const std::string key = KeyFor(t, i);
      const std::map<std::string,
                     std::pair<SequenceNumber, std::string>>::const_iterator
          it = recovered.find(key);
      if (record.acked) {
        ++result->acked;
      }
      if (it == recovered.end()) {
        continue;
      }
      ++matched;
      if (!record.issued) {
        violation("recovered a write never issued: " + key);
        continue;
      }
      if (it->second.second != ValueFor(config.seed, t, i)) {
        violation("recovered a wrong value for " + key);
      }
      if (record.acked && it->second.first != record.seq) {
        violation("recovered " + key + " at sequence " +
                  std::to_string(it->second.first) + ", written at " +
                  std::to_string(record.seq));
      }
      max_recovered = std::max(max_recovered, it->second.first);
    }
  }
  if (matched != recovered.size()) {
    violation("recovered " + std::to_string(recovered.size() - matched) +
              " keys no writer wrote");
  }

  // Every write is one Put, so it consumes one sequence, and the only fault is
  // the crash, so no index is burned and covered before it: the recovered
  // sequences must be exactly 1..S. This also covers writes that reported
  // failure, whose sequence the history does not know.
  // Strict is the only mode: a hole entirely below the lowest replayed
  // sequence fails like any other hole (predecessor spans anchor the merge's
  // lower bound, G-20).
  auto hole = [&](SequenceNumber /*missing*/, const std::string& what) {
    violation(what);
  };
  std::set<SequenceNumber> sequences;
  for (const auto& entry : recovered) {
    sequences.insert(entry.second.first);
  }
  for (SequenceNumber seq = 1; seq <= max_recovered; ++seq) {
    if (sequences.count(seq) == 0) {
      hole(seq, "hole: sequence " + std::to_string(seq) +
                    " is missing below recovered sequence " +
                    std::to_string(max_recovered));
    }
  }

  for (int t = 0; t < config.threads; ++t) {
    for (int i = 0; i < config.writes_per_thread; ++i) {
      const WriteRecord& record = history[t][i];
      if (!record.acked || recovered.count(KeyFor(t, i)) != 0) {
        continue;
      }
      if (config.crash == CrashKind::kPowerLoss) {
        // Crash fidelity (G-54, the drop check ported from D126193958):
        // every acknowledged write reached the file system, so one missing
        // after a power-loss recovery proves the crash dropped unsynced
        // data. If the fault FileSystem stopped dropping, every cycle
        // becomes a clean restart and this never sets.
        result->power_loss_lost_write = true;
      }
      const std::string what = "acknowledged " +
                               std::string(record.sync ? "sync" : "async") +
                               " write " + KeyFor(t, i) + " at sequence " +
                               std::to_string(record.seq) + " lost";
      if (record.seq <= max_recovered) {
        hole(record.seq, "hole: " + what + " below recovered sequence " +
                             std::to_string(max_recovered));
      } else if (config.crash == CrashKind::kProcessCrash) {
        violation("process crash: " + what);
      } else if (record.sync) {
        violation("power loss: " + what);
      }
    }
  }
  if (config.crash == CrashKind::kProcessCrash) {
    if (safe_point > max_recovered) {
      violation("runtime durability safe point " + std::to_string(safe_point) +
                " is above the recovered prefix " +
                std::to_string(max_recovered));
    }
    if (published > max_recovered) {
      result->published_lost = true;
      if (kCheckPublished ||
          EnvU64("ROCKSDB_WAL_ORACLE_CHECK_PUBLISHED", 0) != 0) {
        violation("published sequence " + std::to_string(published) +
                  " is above the recovered prefix " +
                  std::to_string(max_recovered));
      }
    }
  }
}

uint64_t DBWalOracleTest::RunCycles(
    uint64_t seed, uint64_t cycles,
    const std::function<void(CycleConfig*)>& adjust, bool stop_at_violation,
    uint64_t* power_loss_cycles_out, uint64_t* power_loss_lost_out) {
  uint64_t first_violation = 0;
  uint64_t published_lost = 0;
  uint64_t cycles_with_holes = 0;
  uint64_t two_partition_cycles = 0;
  uint64_t power_loss_cycles = 0;
  uint64_t power_loss_lost = 0;
  uint64_t slow_cycles = 0;
  uint64_t ran = 0;
  uint64_t acked = 0;
  uint64_t recovered = 0;
  const uint64_t start = SystemClock::Default()->NowMicros();
  for (uint64_t i = 0; i < cycles; ++i) {
    CycleConfig config = ConfigForSeed(seed + i);
    if (adjust) {
      adjust(&config);
    }
    const CycleResult result = RunCycle(config);
    ++ran;
    two_partition_cycles += config.partitions == 2 ? 1 : 0;
    power_loss_cycles += config.crash == CrashKind::kPowerLoss ? 1 : 0;
    power_loss_lost +=
        (config.crash == CrashKind::kPowerLoss && result.power_loss_lost_write)
            ? 1
            : 0;
    slow_cycles += config.slow ? 1 : 0;
    acked += result.acked;
    recovered += result.recovered;
    published_lost += result.published_lost ? 1 : 0;
    cycles_with_holes += result.holes > 0 ? 1 : 0;
    if (result.violations.empty()) {
      continue;
    }
    std::string report =
        "prefix oracle violation in cycle " + std::to_string(i + 1) + ": " +
        config.ToString() +
        "\n  rerun: ROCKSDB_WAL_ORACLE_SEED=" + std::to_string(config.seed) +
        " ROCKSDB_WAL_ORACLE_CYCLES=1";
    for (const std::string& v : result.violations) {
      report += "\n  " + v;
    }
    fprintf(stderr, "%s\n", report.c_str());
    if (first_violation == 0) {
      first_violation = i + 1;
    }
    if (stop_at_violation) {
      break;
    }
    ADD_FAILURE() << report;
  }
  fprintf(stderr,
          "prefix oracle: %" PRIu64 " cycles from seed %" PRIu64
          " in %.1f s; %" PRIu64 " writes acknowledged, %" PRIu64
          " recovered; %" PRIu64
          " recoveries stopped at a cross-partition hole; published-but-lost"
          " in %" PRIu64 " process-crash cycles (check %s); N=2 in %" PRIu64
          ", N=8 in %" PRIu64 ", power loss in %" PRIu64
          ", a slow partition in %" PRIu64
          "; power loss lost an acknowledged write in %" PRIu64 "\n",
          ran, seed, (SystemClock::Default()->NowMicros() - start) / 1e6, acked,
          recovered, cycles_with_holes, published_lost,
          kCheckPublished || EnvU64("ROCKSDB_WAL_ORACLE_CHECK_PUBLISHED", 0)
              ? "on"
              : "off",
          two_partition_cycles, ran - two_partition_cycles, power_loss_cycles,
          slow_cycles, power_loss_lost);
  if (power_loss_cycles_out != nullptr) {
    *power_loss_cycles_out = power_loss_cycles;
  }
  if (power_loss_lost_out != nullptr) {
    *power_loss_lost_out = power_loss_lost;
  }
  return first_violation;
}

TEST_F(DBWalOracleTest, RecoveredStateIsPrefixOfHistory) {
  uint64_t power_loss_cycles = 0;
  uint64_t power_loss_lost = 0;
  RunCycles(EnvU64("ROCKSDB_WAL_ORACLE_SEED", kDefaultSeed),
            EnvU64("ROCKSDB_WAL_ORACLE_CYCLES", kDefaultCycles),
            /*adjust=*/nullptr, /*stop_at_violation=*/false, &power_loss_cycles,
            &power_loss_lost);
  // The crashes prove they lose data (G-54): when power-loss cycles ran, at
  // least one must have lost an acknowledged write, or every cycle was a
  // clean restart. Skipped when no power-loss cycle ran (single-cycle
  // reruns of a process-crash seed), where there is nothing to prove.
  if (power_loss_cycles > 0) {
    EXPECT_GT(power_loss_lost, 0u)
        << "no power-loss cycle lost an acknowledged write";
  }
}

// The two tests below are the oracle's neuter: each injects an off-by-one
// through a test-only sync point and requires the oracle to report it, then
// requires the same cycles to pass with the injection off.

TEST_F(DBWalOracleTest, CatchesSafePointOffByOne) {
  // Runtime site: the tracker's safe point computation reports min_pending
  // instead of min_pending - 1, claiming the lowest in-flight sequence
  // written. The completion gate cannot show it (a waiter's target is the
  // high end of its own completed range, and live ranges are disjoint, so a
  // safe point one too high never releases a waiter early); the oracle's
  // cross-check of the runtime frontier against the recovered prefix does.
  const uint64_t seed = EnvU64("ROCKSDB_WAL_ORACLE_SEED", kDefaultSeed);
  // Crashing as an append starts strands its range in flight. The safe point
  // is recomputed past it only when a lower range on another partition
  // completes afterwards, so writes are held between allocation and append
  // to keep ranges overlapping.
  auto process_crash = [](CycleConfig* config) {
    config->crash = CrashKind::kProcessCrash;
    config->trigger = CrashTrigger::kAppend;
    config->threads = std::max(config->threads, 3);
    config->delay_append_every = 2;
    config->delay_append_micros = 500;
  };
  SyncPoint::GetInstance()->SetCallBack(
      "WalDurabilityTracker::ForceRecalculate:Pending",
      [](void* arg) { ++*static_cast<uint64_t*>(arg); });
  const uint64_t caught_at = RunCycles(seed, kSelfTestMaxCycles, process_crash,
                                       /*stop_at_violation=*/true);
  SyncPoint::GetInstance()->ClearCallBack(
      "WalDurabilityTracker::ForceRecalculate:Pending");
  fprintf(stderr, "safe point off-by-one caught at cycle %" PRIu64 "\n",
          caught_at);
  ASSERT_NE(0u, caught_at);
  EXPECT_EQ(0u, RunCycles(seed, caught_at, process_crash,
                          /*stop_at_violation=*/false));
}

TEST_F(DBWalOracleTest, CatchesBottomLossWithoutSpans) {
  // G-20: without predecessor spans the merge derives its lower bound from
  // the lowest index any live WAL still holds, so a bottom loss replays a
  // non-prefix. Seed 20261809 crashes on cross-partition skew (a completion
  // gate park) with a slow partition and delayed appends; rerunning its
  // single cycle 60 times violates the prefix property without bottom-loss
  // protections (43 of 60 here, 58 of 60 under TSan; GC11 measured 44 of 60
  // for spans alone before the Q6 floor existed) and passes with them. The
  // neuter disables span emission and the first-session floor together: the
  // floor masks first-session spanless loss by design, so proving the oracle
  // sees it needs both off. Post-flush spanless loss, where the floor
  // honestly does not apply, is pinned deterministically with spans alone by
  // DBWritePartitionTest.PostFlushBottomLossNeedsSpans. The slow partition
  // stalls 50 ms per append so the skew forms under instrumentation too.
  constexpr uint64_t kReproSeed = 20261809;
  constexpr uint64_t kReruns = 60;
  CycleConfig config = ConfigForSeed(kReproSeed);
  config.partitions = 2;
  config.slow_partition %= config.partitions;
  config.crash = CrashKind::kProcessCrash;
  config.trigger = CrashTrigger::kPark;
  // Vary the race window across reruns: different builds interleave
  // differently (instrumentation slows the operations the delays race), and
  // the skew must be in flight when the crash lands.
  config.threads = 6;
  config.delay_append_every = 2;
  config.delay_append_micros = 2000;
  config.slow = true;
  config.slow_partition = 0;
  config.slow_latency.append_micros = 2000;
  config.slow_latency.sync_micros = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::StartWALFile:SkipPredecessorSpan",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ProcessLogFilesMerged:SkipHistoryFloor",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  uint64_t caught_at = 0;
  uint64_t hits = 0;
  for (uint64_t i = 0; i < kReruns; ++i) {
    config.crash_at = 1;
    config.delay_append_micros = 2000 * (1 + (i % 5));
    config.slow_latency.append_micros = 50000;
    const CycleResult result = RunCycle(config);
    if (!result.violations.empty()) {
      ++hits;
      if (caught_at == 0) {
        caught_at = i + 1;
        fprintf(
            stderr,
            "bottom loss without spans first caught at rerun %llu/%llu: %s\n",
            (unsigned long long)caught_at, (unsigned long long)kReruns,
            config.ToString().c_str());
        for (const std::string& v : result.violations) {
          fprintf(stderr, "  %s\n", v.c_str());
        }
      }
    }
  }
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::StartWALFile:SkipPredecessorSpan");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ProcessLogFilesMerged:SkipHistoryFloor");
  fprintf(stderr,
          "bottom loss without spans: %" PRIu64 " of %" PRIu64
          " reruns violated\n",
          hits, kReruns);
  ASSERT_NE(0u, caught_at);
  for (uint64_t i = 0; i < caught_at; ++i) {
    config.crash_at = 1;
    config.delay_append_micros = 2000 * (1 + (i % 5));
    config.slow_latency.append_micros = 50000;
    const CycleResult result = RunCycle(config);
    EXPECT_TRUE(result.violations.empty())
        << "rerun " << i + 1 << " with spans on";
  }
}

TEST_F(DBWalOracleTest, CatchesMergeStopOffByOne) {
  // Recovery site: the merge stops one index past the first unexplained
  // hole, treating the hole as explained and replaying the record above it.
  const uint64_t seed = EnvU64("ROCKSDB_WAL_ORACLE_SEED", kDefaultSeed);
  // A hole needs a record synced above an unsynced lower one on another
  // partition: crash right after a sync write's leadership sync while a
  // lower write is held between its allocation and its append.
  auto power_loss = [](CycleConfig* config) {
    config->crash = CrashKind::kPowerLoss;
    config->trigger = CrashTrigger::kLeadershipSync;
    config->crash_at = 1 + config->seed % 8;
    config->sync_one_in = 2;
    config->threads = std::max(config->threads, 4);
    config->delay_append_every = 3;
    config->delay_append_micros = 1000;
  };
  inject_merge_stop_off_by_one_ = true;
  const uint64_t caught_at = RunCycles(seed, kSelfTestMaxCycles, power_loss,
                                       /*stop_at_violation=*/true);
  inject_merge_stop_off_by_one_ = false;
  fprintf(stderr, "merge stop off-by-one caught at cycle %" PRIu64 "\n",
          caught_at);
  ASSERT_NE(0u, caught_at);
  EXPECT_EQ(
      0u, RunCycles(seed, caught_at, power_loss, /*stop_at_violation=*/false));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
