//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <algorithm>
#include <atomic>
#include <cassert>
#include <mutex>
#include <string>
#include <vector>

#include "db/write_batch_internal.h"
#include "util/cast_util.h"
#include "utilities/transactions/transaction_test.h"

namespace ROCKSDB_NAMESPACE {

constexpr std::array TimestampedSnapshotWithTsSanityCheck_Params = {
    std::make_tuple(false, false, WRITE_PREPARED, kOrderedWrite),
    std::make_tuple(false, true, WRITE_PREPARED, kUnorderedWrite),
    std::make_tuple(false, false, WRITE_UNPREPARED, kOrderedWrite)};

INSTANTIATE_TEST_CASE_P(
    Unsupported, TimestampedSnapshotWithTsSanityCheck,
    ::testing::ValuesIn(WRAP_PARAM_WITH_PER_KEY_POINT_LOCK_MANAGER_PARAMS(
        WRAP_PARAM(bool, bool, TxnDBWritePolicy, WriteOrdering),
        TimestampedSnapshotWithTsSanityCheck_Params)));

INSTANTIATE_TEST_CASE_P(
    WriteCommitted, TransactionTest,
    ::testing::Combine(/*use_stackable_db=*/::testing::Bool(),
                       /*two_write_queue=*/::testing::Bool(),
                       ::testing::Values(WRITE_COMMITTED),
                       ::testing::Values(kOrderedWrite),
                       /*use_per_key_point_lock_mgr=*/::testing::Bool(),
                       /*deadlock_timeout_us=*/::testing::Values(0, 1000)));

namespace {
// Not thread-safe. Caller needs to provide external synchronization.
class TsCheckingTxnNotifier : public TransactionNotifier {
 public:
  explicit TsCheckingTxnNotifier() = default;

  ~TsCheckingTxnNotifier() override = default;

  void SnapshotCreated(const Snapshot* new_snapshot) override {
    assert(new_snapshot);
    if (prev_snapshot_seq_ != kMaxSequenceNumber) {
      assert(prev_snapshot_seq_ <= new_snapshot->GetSequenceNumber());
    }
    prev_snapshot_seq_ = new_snapshot->GetSequenceNumber();
    if (prev_snapshot_ts_ != kMaxTxnTimestamp) {
      assert(prev_snapshot_ts_ <= new_snapshot->GetTimestamp());
    }
    prev_snapshot_ts_ = new_snapshot->GetTimestamp();
  }

  TxnTimestamp prev_snapshot_ts() const { return prev_snapshot_ts_; }

 private:
  SequenceNumber prev_snapshot_seq_ = kMaxSequenceNumber;
  TxnTimestamp prev_snapshot_ts_ = kMaxTxnTimestamp;
};
}  // anonymous namespace

TEST_P(TimestampedSnapshotWithTsSanityCheck, WithoutCommitTs) {
  std::unique_ptr<Transaction> txn(
      db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v"));
  ASSERT_OK(txn->Prepare());
  Status s = txn->CommitAndTryCreateSnapshot();
  ASSERT_TRUE(s.IsInvalidArgument());
  ASSERT_OK(txn->Rollback());

  txn.reset(db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v"));
  s = txn->CommitAndTryCreateSnapshot();
  ASSERT_TRUE(s.IsInvalidArgument());
}

TEST_P(TimestampedSnapshotWithTsSanityCheck, SetCommitTs) {
  std::unique_ptr<Transaction> txn(
      db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v"));
  ASSERT_OK(txn->Prepare());
  std::shared_ptr<const Snapshot> snapshot;
  Status s = txn->CommitAndTryCreateSnapshot(nullptr, 10, &snapshot);
  ASSERT_TRUE(s.IsNotSupported());
  ASSERT_OK(txn->Rollback());

  txn.reset(db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v"));
  s = txn->CommitAndTryCreateSnapshot(nullptr, 10, &snapshot);
  ASSERT_TRUE(s.IsNotSupported());
}

TEST_P(TransactionTest, WithoutCommitTs) {
  std::unique_ptr<Transaction> txn(
      db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v"));
  ASSERT_OK(txn->Prepare());
  Status s = txn->CommitAndTryCreateSnapshot();
  ASSERT_TRUE(s.IsInvalidArgument());
  ASSERT_OK(txn->Rollback());

  txn.reset(db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v"));
  s = txn->CommitAndTryCreateSnapshot();
  ASSERT_TRUE(s.IsInvalidArgument());
}

TEST_P(TransactionTest, ReuseExistingTxn) {
  Transaction* txn = db->BeginTransaction(WriteOptions(), TransactionOptions());
  assert(txn);
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("a", "v1"));
  ASSERT_OK(txn->Prepare());

  auto notifier = std::make_shared<TsCheckingTxnNotifier>();
  std::shared_ptr<const Snapshot> snapshot1;
  Status s =
      txn->CommitAndTryCreateSnapshot(notifier, /*commit_ts=*/100, &snapshot1);
  ASSERT_OK(s);
  ASSERT_EQ(100, snapshot1->GetTimestamp());

  Transaction* txn1 =
      db->BeginTransaction(WriteOptions(), TransactionOptions(), txn);
  assert(txn1 == txn);
  ASSERT_OK(txn1->SetName("txn1"));
  ASSERT_OK(txn->Put("a", "v2"));
  ASSERT_OK(txn->Prepare());
  std::shared_ptr<const Snapshot> snapshot2;
  s = txn->CommitAndTryCreateSnapshot(notifier, /*commit_ts=*/110, &snapshot2);
  ASSERT_OK(s);
  ASSERT_EQ(110, snapshot2->GetTimestamp());
  delete txn;

  {
    std::string value;
    ReadOptions read_opts;
    read_opts.snapshot = snapshot1.get();
    ASSERT_OK(db->Get(read_opts, "a", &value));
    ASSERT_EQ("v1", value);

    read_opts.snapshot = snapshot2.get();
    ASSERT_OK(db->Get(read_opts, "a", &value));
    ASSERT_EQ("v2", value);
  }
}

TEST_P(TransactionTest, CreateSnapshotWhenCommit) {
  std::unique_ptr<Transaction> txn(
      db->BeginTransaction(WriteOptions(), TransactionOptions()));
  assert(txn);

  constexpr int batch_size = 10;
  for (int i = 0; i < batch_size; ++i) {
    ASSERT_OK(db->Put(WriteOptions(), "k" + std::to_string(i), "v0"));
  }
  const SequenceNumber seq0 = db->GetLatestSequenceNumber();
  ASSERT_EQ(static_cast<SequenceNumber>(batch_size), seq0);

  txn->SetSnapshot();
  {
    const Snapshot* const snapshot = txn->GetSnapshot();
    assert(snapshot);
    ASSERT_EQ(seq0, snapshot->GetSequenceNumber());
  }

  for (int i = 0; i < batch_size; ++i) {
    ASSERT_OK(txn->Put("k" + std::to_string(i), "v1"));
  }
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Prepare());

  std::shared_ptr<const Snapshot> snapshot;
  constexpr TxnTimestamp timestamp = 1;
  auto notifier = std::make_shared<TsCheckingTxnNotifier>();
  Status s = txn->CommitAndTryCreateSnapshot(notifier, timestamp, &snapshot);
  ASSERT_OK(s);
  ASSERT_LT(notifier->prev_snapshot_ts(), kMaxTxnTimestamp);
  assert(snapshot);
  ASSERT_EQ(timestamp, snapshot->GetTimestamp());
  ASSERT_EQ(seq0 + batch_size, snapshot->GetSequenceNumber());
  const Snapshot* const raw_snapshot_ptr = txn->GetSnapshot();
  ASSERT_EQ(raw_snapshot_ptr, snapshot.get());
  ASSERT_EQ(snapshot, txn->GetTimestampedSnapshot());

  {
    std::shared_ptr<const Snapshot> snapshot1 =
        db->GetLatestTimestampedSnapshot();
    ASSERT_EQ(snapshot, snapshot1);
  }
  {
    std::shared_ptr<const Snapshot> snapshot1 =
        db->GetTimestampedSnapshot(timestamp);
    ASSERT_EQ(snapshot, snapshot1);
  }
  {
    std::vector<std::shared_ptr<const Snapshot> > snapshots;
    s = db->GetAllTimestampedSnapshots(snapshots);
    ASSERT_OK(s);
    ASSERT_EQ(std::vector<std::shared_ptr<const Snapshot> >{snapshot},
              snapshots);
  }
}

TEST_P(TransactionTest, CreateSnapshot) {
  // First create a non-timestamped snapshot
  ManagedSnapshot snapshot_guard(db);
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(db->Put(WriteOptions(), "k" + std::to_string(i),
                      "v0_" + std::to_string(i)));
  }
  {
    auto ret = db->CreateTimestampedSnapshot(kMaxTxnTimestamp);
    ASSERT_TRUE(ret.first.IsInvalidArgument());
    auto snapshot = ret.second;
    ASSERT_EQ(nullptr, snapshot.get());
  }
  constexpr TxnTimestamp timestamp = 100;
  Status s;
  std::shared_ptr<const Snapshot> ts_snap0;
  std::tie(s, ts_snap0) = db->CreateTimestampedSnapshot(timestamp);
  ASSERT_OK(s);
  assert(ts_snap0);
  ASSERT_EQ(timestamp, ts_snap0->GetTimestamp());
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(db->Delete(WriteOptions(), "k" + std::to_string(i)));
  }
  {
    ReadOptions read_opts;
    read_opts.snapshot = ts_snap0.get();
    for (int i = 0; i < 10; ++i) {
      std::string value;
      s = db->Get(read_opts, "k" + std::to_string(i), &value);
      ASSERT_OK(s);
      ASSERT_EQ("v0_" + std::to_string(i), value);
    }
  }
  {
    std::shared_ptr<const Snapshot> snapshot =
        db->GetLatestTimestampedSnapshot();
    ASSERT_EQ(ts_snap0, snapshot);
  }
  {
    std::shared_ptr<const Snapshot> snapshot =
        db->GetTimestampedSnapshot(timestamp);
    ASSERT_OK(s);
    ASSERT_EQ(ts_snap0, snapshot);
  }
  {
    std::vector<std::shared_ptr<const Snapshot> > snapshots;
    s = db->GetAllTimestampedSnapshots(snapshots);
    ASSERT_OK(s);
    ASSERT_EQ(std::vector<std::shared_ptr<const Snapshot> >{ts_snap0},
              snapshots);
  }
}

TEST_P(TransactionTest, SequenceAndTsOrder) {
  Status s;
  std::shared_ptr<const Snapshot> snapshot;
  std::tie(s, snapshot) = db->CreateTimestampedSnapshot(100);
  ASSERT_OK(s);
  assert(snapshot);
  {
    // Cannot request smaller timestamp for the new timestamped snapshot.
    std::shared_ptr<const Snapshot> tmp_snapshot;
    std::tie(s, tmp_snapshot) = db->CreateTimestampedSnapshot(50);
    ASSERT_TRUE(s.IsInvalidArgument());
    ASSERT_EQ(nullptr, tmp_snapshot.get());
  }

  // If requesting a new timestamped snapshot with the same timestamp and
  // sequence number, we avoid creating new snapshot object but reuse
  // exisisting one.
  std::shared_ptr<const Snapshot> snapshot1;
  std::tie(s, snapshot1) = db->CreateTimestampedSnapshot(100);
  ASSERT_OK(s);
  ASSERT_EQ(snapshot.get(), snapshot1.get());

  // If there is no write, but we request a larger timestamp, we still create
  // a new snapshot object.
  std::shared_ptr<const Snapshot> snapshot2;
  std::tie(s, snapshot2) = db->CreateTimestampedSnapshot(200);
  ASSERT_OK(s);
  assert(snapshot2);
  ASSERT_NE(snapshot.get(), snapshot2.get());
  ASSERT_EQ(snapshot2->GetSequenceNumber(), snapshot->GetSequenceNumber());
  ASSERT_EQ(200, snapshot2->GetTimestamp());

  // Increase sequence number.
  ASSERT_OK(db->Put(WriteOptions(), "foo", "v0"));
  {
    // We are requesting the same timestamp for a larger sequence number, thus
    // we cannot create timestamped snapshot.
    std::shared_ptr<const Snapshot> tmp_snapshot;
    std::tie(s, tmp_snapshot) = db->CreateTimestampedSnapshot(200);
    ASSERT_TRUE(s.IsInvalidArgument());
    ASSERT_EQ(nullptr, tmp_snapshot.get());
  }
  {
    std::unique_ptr<Transaction> txn1(
        db->BeginTransaction(WriteOptions(), TransactionOptions()));
    ASSERT_OK(txn1->Put("bar", "v0"));
    std::shared_ptr<const Snapshot> ss;
    ASSERT_OK(txn1->CommitAndTryCreateSnapshot(nullptr, 200, &ss));
    // Cannot create snapshot because requested timestamp is the same as the
    // latest timestamped snapshot while sequence number is strictly higher.
    ASSERT_EQ(nullptr, ss);
  }
  {
    std::unique_ptr<Transaction> txn2(
        db->BeginTransaction(WriteOptions(), TransactionOptions()));
    ASSERT_OK(txn2->Put("bar", "v0"));
    std::shared_ptr<const Snapshot> ss;
    // Application should never do this. This is just to demonstrate error
    // handling.
    ASSERT_OK(txn2->CommitAndTryCreateSnapshot(nullptr, 100, &ss));
    // Cannot create snapshot because requested timestamp is smaller than
    // latest timestamped snapshot.
    ASSERT_EQ(nullptr, ss);
  }
}

TEST_P(TransactionTest, CloseDbWithSnapshots) {
  std::unique_ptr<Transaction> txn(
      db->BeginTransaction(WriteOptions(), TransactionOptions()));
  ASSERT_OK(txn->SetName("txn0"));
  ASSERT_OK(txn->Put("foo", "v"));
  ASSERT_OK(txn->Prepare());
  std::shared_ptr<const Snapshot> snapshot;
  constexpr TxnTimestamp timestamp = 121;
  auto notifier = std::make_shared<TsCheckingTxnNotifier>();
  ASSERT_OK(txn->CommitAndTryCreateSnapshot(notifier, timestamp, &snapshot));
  assert(snapshot);
  ASSERT_LT(notifier->prev_snapshot_ts(), kMaxTxnTimestamp);
  ASSERT_EQ(timestamp, snapshot->GetTimestamp());
  ASSERT_TRUE(db->Close().IsAborted());
}

TEST_P(TransactionTest, MultipleTimestampedSnapshots) {
  auto* dbimpl = static_cast_with_check<DBImpl>(db->GetRootDB());
  assert(dbimpl);
  const bool seq_per_batch = dbimpl->seq_per_batch();
  // Used only by the assert below; mark used for opt builds (assert is
  // empty under NDEBUG, which -Werror would flag as unused).
  (void)seq_per_batch;
  // TODO: remove the following assert(!seq_per_batch) once timestamped snapshot
  // is supported in write-prepared/write-unprepared transactions.
  assert(!seq_per_batch);
  constexpr size_t txn_size = 10;
  constexpr TxnTimestamp ts_delta = 10;
  constexpr size_t num_txns = 100;
  std::vector<std::shared_ptr<const Snapshot> > snapshots(num_txns);
  constexpr TxnTimestamp start_ts = 10000;
  auto notifier = std::make_shared<TsCheckingTxnNotifier>();
  for (size_t i = 0; i < num_txns; ++i) {
    std::unique_ptr<Transaction> txn(
        db->BeginTransaction(WriteOptions(), TransactionOptions()));
    ASSERT_OK(txn->SetName("txn" + std::to_string(i)));
    for (size_t j = 0; j < txn_size; ++j) {
      ASSERT_OK(txn->Put("k" + std::to_string(j),
                         "v" + std::to_string(j) + "_" + std::to_string(i)));
    }
    if (0 == (i % 2)) {
      ASSERT_OK(txn->Prepare());
    }
    ASSERT_OK(txn->CommitAndTryCreateSnapshot(notifier, start_ts + i * ts_delta,
                                              &snapshots[i]));
    assert(snapshots[i]);
    ASSERT_LT(notifier->prev_snapshot_ts(), kMaxTxnTimestamp);
    ASSERT_EQ(start_ts + i * ts_delta, snapshots[i]->GetTimestamp());
  }

  {
    auto snapshot = db->GetTimestampedSnapshot(start_ts + 1);
    ASSERT_EQ(nullptr, snapshot);
  }

  constexpr TxnTimestamp max_ts = start_ts + num_txns * ts_delta;
  for (size_t i = 0; i < num_txns; ++i) {
    auto snapshot = db->GetTimestampedSnapshot(start_ts + i * ts_delta);
    ASSERT_EQ(snapshots[i], snapshot);

    std::vector<std::shared_ptr<const Snapshot> > tmp_snapshots;
    Status s = db->GetTimestampedSnapshots(max_ts, start_ts + i * ts_delta,
                                           tmp_snapshots);
    ASSERT_TRUE(s.IsInvalidArgument());
    ASSERT_TRUE(tmp_snapshots.empty());

    for (size_t j = i; j < num_txns; ++j) {
      std::vector<std::shared_ptr<const Snapshot> > expected_snapshots(
          snapshots.begin() + i, snapshots.begin() + j);
      tmp_snapshots.clear();
      s = db->GetTimestampedSnapshots(start_ts + i * ts_delta,
                                      start_ts + j * ts_delta, tmp_snapshots);
      if (i < j) {
        ASSERT_OK(s);
      } else {
        ASSERT_TRUE(s.IsInvalidArgument());
      }
      ASSERT_EQ(expected_snapshots, tmp_snapshots);
    }
  }

  {
    std::vector<std::shared_ptr<const Snapshot> > tmp_snapshots;
    const Status s = db->GetAllTimestampedSnapshots(tmp_snapshots);
    ASSERT_OK(s);
    ASSERT_EQ(snapshots, tmp_snapshots);

    const std::shared_ptr<const Snapshot> latest_snapshot =
        db->GetLatestTimestampedSnapshot();
    ASSERT_EQ(snapshots.back(), latest_snapshot);
  }

  for (size_t i = 0; i <= num_txns; ++i) {
    std::vector<std::shared_ptr<const Snapshot> > snapshots1(
        snapshots.begin() + i, snapshots.end());
    if (i > 0) {
      auto snapshot1 =
          db->GetTimestampedSnapshot(start_ts + (i - 1) * ts_delta);
      assert(snapshot1);
      ASSERT_EQ(start_ts + (i - 1) * ts_delta, snapshot1->GetTimestamp());
    }

    db->ReleaseTimestampedSnapshotsOlderThan(start_ts + i * ts_delta);

    if (i > 0) {
      auto snapshot1 =
          db->GetTimestampedSnapshot(start_ts + (i - 1) * ts_delta);
      ASSERT_EQ(nullptr, snapshot1);
    }

    std::vector<std::shared_ptr<const Snapshot> > tmp_snapshots;
    const Status s = db->GetAllTimestampedSnapshots(tmp_snapshots);
    ASSERT_OK(s);
    ASSERT_EQ(snapshots1, tmp_snapshots);
  }

  // Even after released by db, the applications still hold reference to shared
  // snapshots.
  for (size_t i = 0; i < num_txns; ++i) {
    assert(snapshots[i]);
    ASSERT_EQ(start_ts + i * ts_delta, snapshots[i]->GetTimestamp());
  }

  snapshots.clear();
  ASSERT_OK(db->Close());
  delete db;
  db = nullptr;
}

// GC24 Q10: opt-capable twin of TransactionPartitionedWalTest.
// GC23TimestampedSnapshotsSurviveFlushStress in transaction_test.cc (kept
// there as the debug-canonical check). transaction_test does not build with
// DEBUG_LEVEL=0 (SyncPoint and most TEST_ helpers are debug-only); this copy
// uses none of those -- only the unguarded TEST_WalPartitionForWrite routing
// query -- so the snapshot value check also runs in release builds. The body
// is verbatim; keep the two in sync.
//
// GC23 Q9: GC22's compaction guarantee without SyncPoint, so nothing here
// depends on a debug-only hook. At N in {2, 8}, synced committers on every
// partition but 1 call CommitAndTryCreateSnapshot concurrently while a writer
// keeps overwriting key "k" on partition 1 and a flusher keeps flushing and
// compacting partition 1's column family. Before any committer starts, "k"
// holds a version, so every timestamped snapshot must find "k". Each
// snapshot reads "k" right after it is created and again after a final flush
// and compaction, and both reads must agree; ordered by sequence, the
// writer's counters they read never go backwards. Neuter: drop GC22's
// reservation; a flush that starts between a commit's publication and its
// snapshot drops the version the snapshot needs, and the snapshot's reads
// change after the final compaction.
TEST(TransactionPartitionedWalTest,
     GC23TimestampedSnapshotsSurviveFlushStress) {
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions));
    const std::string dbname =
        test::PerThreadDBPath("gc23_ts_snapshot_flush_stress");
    Options options;
    options.create_if_missing = true;
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    options.disable_auto_compactions = true;
    options.track_and_verify_wals_in_manifest = true;
    options.max_write_buffer_number = 8;
    ASSERT_OK(DestroyDB(dbname, options));
    TransactionDB* db = nullptr;
    ASSERT_OK(
        TransactionDB::Open(options, TransactionDBOptions(), dbname, &db));
    std::vector<ColumnFamilyHandle*> handles;
    std::vector<ColumnFamilyHandle*> cf_for_partition(num_partitions, nullptr);
    size_t covered = 0;
    while (covered < num_partitions) {
      ColumnFamilyHandle* handle = nullptr;
      ASSERT_OK(db->CreateColumnFamily(
          ColumnFamilyOptions(options),
          "cf" + std::to_string(handles.size() + 1), &handle));
      handles.push_back(handle);
      WriteBatch batch;
      ASSERT_OK(
          WriteBatchInternal::Put(&batch, handle->GetID(), "key", "value"));
      size_t partition = num_partitions;
      ASSERT_OK(
          DBImpl::TEST_WalPartitionForWrite(batch, num_partitions, &partition));
      if (cf_for_partition[partition] == nullptr) {
        cf_for_partition[partition] = handle;
        ++covered;
      }
    }
    ColumnFamilyHandle* const hot = cf_for_partition[1];
    // Counters are fixed width, so they order as strings.
    auto counter_value = [](uint64_t counter) {
      std::string value = std::to_string(counter);
      return std::string(20 - value.size(), '0') + value;
    };
    ASSERT_OK(db->Put(WriteOptions(), hot, "k", counter_value(0)));

    struct Taken {
      std::shared_ptr<const Snapshot> snapshot;
      std::string first_read;
    };
    std::mutex mu;
    std::vector<Taken> taken;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> next_ts{1};
    std::atomic<uint64_t> missing{0};
    // Synced commits: a commit whose predecessors on other partitions land
    // after its fsync syncs again in its gate, between publishing and its
    // snapshot, which widens the window the reservation covers.
    WriteOptions synced;
    synced.sync = true;
    constexpr int kCommitters = 8;
    constexpr int kCommitsPerThread = 100;
    std::vector<port::Thread> committers;
    for (int t = 0; t < kCommitters; ++t) {
      size_t p = static_cast<size_t>(t) % num_partitions;
      if (p == 1) {
        p = 0;
      }
      committers.emplace_back([&, p, t] {
        for (int i = 0; i < kCommitsPerThread; ++i) {
          std::unique_ptr<Transaction> txn(db->BeginTransaction(synced));
          ASSERT_OK(txn->Put(cf_for_partition[p],
                             std::to_string(t) + "_" + std::to_string(i), "v"));
          std::shared_ptr<const Snapshot> snapshot;
          ASSERT_OK(txn->CommitAndTryCreateSnapshot(
              nullptr, next_ts.fetch_add(1), &snapshot));
          if (snapshot == nullptr) {
            continue;  // Timestamp order disagreed with sequence order.
          }
          ReadOptions ro;
          ro.snapshot = snapshot.get();
          std::string value;
          const Status s = db->Get(ro, hot, "k", &value);
          if (!s.ok()) {
            missing.fetch_add(1);
            continue;
          }
          std::lock_guard<std::mutex> lock(mu);
          taken.push_back({std::move(snapshot), value});
        }
      });
    }
    port::Thread writer([&] {
      for (uint64_t counter = 1; !stop.load(); ++counter) {
        ASSERT_OK(db->Put(WriteOptions(), hot, "k", counter_value(counter)));
      }
    });
    port::Thread flusher([&] {
      for (int round = 0; !stop.load(); ++round) {
        ASSERT_OK(db->Flush(FlushOptions(), hot));
        if (round % 8 == 0) {
          ASSERT_OK(
              db->CompactRange(CompactRangeOptions(), hot, nullptr, nullptr));
        }
      }
    });
    for (port::Thread& committer : committers) {
      committer.join();
    }
    stop.store(true);
    writer.join();
    flusher.join();
    ASSERT_OK(db->Flush(FlushOptions(), hot));
    ASSERT_OK(db->CompactRange(CompactRangeOptions(), hot, nullptr, nullptr));

    EXPECT_EQ(0u, missing.load()) << "snapshots that found no version of k";
    EXPECT_FALSE(taken.empty());
    std::sort(taken.begin(), taken.end(), [](const Taken& a, const Taken& b) {
      return a.snapshot->GetSequenceNumber() < b.snapshot->GetSequenceNumber();
    });
    std::string previous;
    for (const Taken& t : taken) {
      ReadOptions ro;
      ro.snapshot = t.snapshot.get();
      std::string value;
      ASSERT_OK(db->Get(ro, hot, "k", &value))
          << "snapshot " << t.snapshot->GetSequenceNumber();
      EXPECT_EQ(t.first_read, value)
          << "snapshot " << t.snapshot->GetSequenceNumber();
      EXPECT_LE(previous, value)
          << "snapshot " << t.snapshot->GetSequenceNumber();
      previous = value;
    }
    taken.clear();
    db->ReleaseTimestampedSnapshotsOlderThan(next_ts.load());
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    delete db;
    ASSERT_OK(DestroyDB(dbname, options));
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
