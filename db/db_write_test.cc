//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "db/db_impl/compacted_db_impl.h"
#include "db/db_impl/wal_durability_tracker.h"
#include "db/db_test_util.h"
#include "db/log_writer.h"
#include "db/write_batch_internal.h"
#include "db/write_callback.h"
#include "db/write_thread.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/sst_file_reader.h"
#include "rocksdb/utilities/backup_engine.h"
#include "rocksdb/utilities/checkpoint.h"
#include "rocksdb/utilities/debug.h"
#include "rocksdb/utilities/options_util.h"
#include "rocksdb/utilities/transaction_db.h"
#include "rocksdb/wal_filter.h"
#include "test_util/sync_point.h"
#include "util/coding.h"
#include "util/hash.h"
#include "util/random.h"
#include "util/string_util.h"
#include "util/write_batch_util.h"
#include "utilities/fault_injection_env.h"
#include "utilities/fault_injection_fs.h"
#include "utilities/latency_injection_fs.h"

namespace ROCKSDB_NAMESPACE {

// Test variations of WriteImpl.
class DBWriteTest : public DBTestBase, public testing::WithParamInterface<int> {
 public:
  DBWriteTest() : DBTestBase("db_write_test", /*env_do_fsync=*/true) {}

  Options GetOptions() { return DBTestBase::GetOptions(GetParam()); }

  void Open() { DBTestBase::Reopen(GetOptions()); }
};

class DBWriteTestUnparameterized : public DBTestBase {
 public:
  explicit DBWriteTestUnparameterized()
      : DBTestBase("pipelined_write_test", /*env_do_fsync=*/false) {}
};

TEST(WalRoutingTest, RoutesOnlyByColumnFamily) {
  constexpr size_t kNumPartitions = 4;
  constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> cf_for_partition(kNumPartitions, kUnassigned);

  for (uint32_t cf_id = 0; cf_id < 10000; ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = kNumPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kNumPartitions, &partition));
    ASSERT_LT(partition, kNumPartitions);
    cf_for_partition[partition] = cf_id;
  }
  for (size_t partition = 0; partition < kNumPartitions; ++partition) {
    ASSERT_NE(cf_for_partition[partition], kUnassigned)
        << "no CF id routed to partition " << partition;
  }

  const uint32_t first_cf = std::min(cf_for_partition[0], cf_for_partition[1]);
  const uint32_t second_cf = std::max(cf_for_partition[0], cf_for_partition[1]);
  WriteBatch first_cf_batch;
  ASSERT_OK(WriteBatchInternal::Put(&first_cf_batch, first_cf, "a", "first"));
  size_t expected_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(first_cf_batch, kNumPartitions,
                                              &expected_partition));

  WriteBatch forward;
  ASSERT_OK(WriteBatchInternal::Put(&forward, first_cf, "a", "first"));
  ASSERT_OK(WriteBatchInternal::Put(&forward, second_cf, "z", "second"));
  WriteBatch reverse;
  ASSERT_OK(WriteBatchInternal::Put(&reverse, second_cf, "different-key",
                                    "different-value"));
  ASSERT_OK(WriteBatchInternal::Put(&reverse, first_cf, "a", "first"));
  size_t forward_partition = kNumPartitions;
  size_t reverse_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(forward, kNumPartitions,
                                              &forward_partition));
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(reverse, kNumPartitions,
                                              &reverse_partition));
  EXPECT_EQ(forward_partition, expected_partition);
  EXPECT_EQ(reverse_partition, expected_partition);

  WriteBatchInternal::SetSequence(&first_cf_batch, 1);
  size_t low_sequence_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(first_cf_batch, kNumPartitions,
                                              &low_sequence_partition));
  WriteBatchInternal::SetSequence(&first_cf_batch, 1000000);
  size_t high_sequence_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(first_cf_batch, kNumPartitions,
                                              &high_sequence_partition));
  EXPECT_EQ(low_sequence_partition, high_sequence_partition);
}

TEST(WalRoutingTest, HashesXidForZeroColumnFamilyBatches) {
  constexpr size_t kNumPartitions = 4;
  std::vector<bool> covered(kNumPartitions, false);
  for (uint32_t i = 0; i < 10000; ++i) {
    WriteBatch marker;
    ASSERT_OK(WriteBatchInternal::MarkCommit(
        &marker, "transaction-" + std::to_string(i)));
    size_t partition = kNumPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(marker, kNumPartitions, &partition));
    ASSERT_LT(partition, kNumPartitions);
    covered[partition] = true;
  }
  for (size_t partition = 0; partition < kNumPartitions; ++partition) {
    EXPECT_TRUE(covered[partition])
        << "no XID routed to partition " << partition;
  }

  WriteBatch smallest_xid;
  ASSERT_OK(WriteBatchInternal::MarkRollback(&smallest_xid, "a"));
  size_t expected_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(smallest_xid, kNumPartitions,
                                              &expected_partition));
  WriteBatch two_xids;
  ASSERT_OK(WriteBatchInternal::MarkCommit(&two_xids, "z"));
  ASSERT_OK(WriteBatchInternal::MarkRollback(&two_xids, "a"));
  size_t two_xid_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(two_xids, kNumPartitions,
                                              &two_xid_partition));
  EXPECT_EQ(two_xid_partition, expected_partition);

  WriteBatch no_xid;
  ASSERT_OK(WriteBatchInternal::InsertNoop(&no_xid));
  size_t no_xid_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(no_xid, kNumPartitions,
                                              &no_xid_partition));
  EXPECT_EQ(no_xid_partition, 0);
}

TEST(WalRoutingTest, OnePartitionIsNeutered) {
  WriteBatch cf_batch;
  ASSERT_OK(WriteBatchInternal::Put(&cf_batch, 123, "key", "value"));
  WriteBatch xid_batch;
  ASSERT_OK(WriteBatchInternal::MarkCommit(&xid_batch, "transaction"));

  for (const WriteBatch* batch : {&cf_batch, &xid_batch}) {
    size_t partition = 1;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(*batch, /*num_partitions=*/1,
                                                &partition));
    EXPECT_EQ(partition, 0);
  }
}

TEST(WalRoutingTest, RoutesDeferredPutEntitiesByColumnFamily) {
  // Q9 ([05-3]): deferred PutEntity ops contribute their column families to
  // the route exactly like batch entries: the route hashes the minimum CF
  // id across both. An XID-free batch with no CFs routes 0 on its own;
  // with a deferred op it routes like a direct Put to that CF. Dropping
  // the deferred contribution routes 0 and fails the nonzero assertions.
  constexpr size_t kNumPartitions = 4;

  // A CF id whose direct route is nonzero, so the equality below is not
  // vacuous (a dropped contribution routes 0).
  uint32_t routed_cf = 0;
  size_t direct_partition = 0;
  bool found = false;
  for (uint32_t cf_id = 1; cf_id < 10000 && !found; ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = kNumPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kNumPartitions, &partition));
    if (partition != 0) {
      routed_cf = cf_id;
      direct_partition = partition;
      found = true;
    }
  }
  ASSERT_TRUE(found);

  const std::vector<uint32_t> deferred_cf_ids{routed_cf};

  WriteBatch empty;
  ASSERT_OK(WriteBatchInternal::InsertNoop(&empty));
  size_t deferred_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(
      empty, kNumPartitions, &deferred_partition, &deferred_cf_ids));
  EXPECT_EQ(deferred_partition, direct_partition);
  EXPECT_NE(deferred_partition, 0);

  // The deferred CF joins the minimum: a batch touching only a larger CF
  // still routes by the smaller deferred CF.
  WriteBatch larger_cf_batch;
  ASSERT_OK(WriteBatchInternal::Put(&larger_cf_batch, routed_cf + 1000, "key",
                                    "value"));
  size_t joined_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(
      larger_cf_batch, kNumPartitions, &joined_partition, &deferred_cf_ids));
  EXPECT_EQ(joined_partition, direct_partition);
}

TEST(WalRoutingTest, HashesPrepareAndTimestampedCommitXids) {
  // Q9 ([05-3]): MarkEndPrepare and MarkCommitWithTimestamp feed the same
  // XID routing input as the other markers: the same XID routes the same
  // through any of them. Dropping either marker's contribution routes 0
  // and fails the nonzero assertion.
  constexpr size_t kNumPartitions = 4;

  std::string routed_xid;
  size_t direct_partition = 0;
  bool found = false;
  for (uint32_t i = 0; i < 10000 && !found; ++i) {
    WriteBatch marker;
    std::string xid = "prepare-" + std::to_string(i);
    // EndPrepare replaces the batch's leading noop.
    ASSERT_OK(WriteBatchInternal::InsertNoop(&marker));
    ASSERT_OK(WriteBatchInternal::MarkEndPrepare(&marker, xid));
    size_t partition = kNumPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(marker, kNumPartitions, &partition));
    if (partition != 0) {
      routed_xid = xid;
      direct_partition = partition;
      found = true;
    }
  }
  ASSERT_TRUE(found);

  WriteBatch commit_ts;
  ASSERT_OK(WriteBatchInternal::MarkCommitWithTimestamp(&commit_ts, routed_xid,
                                                        "commit-ts"));
  size_t commit_ts_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(commit_ts, kNumPartitions,
                                              &commit_ts_partition));
  EXPECT_EQ(commit_ts_partition, direct_partition);

  WriteBatch commit;
  ASSERT_OK(WriteBatchInternal::MarkCommit(&commit, routed_xid));
  size_t commit_partition = kNumPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(commit, kNumPartitions,
                                              &commit_partition));
  EXPECT_EQ(commit_partition, direct_partition);
  EXPECT_NE(commit_partition, 0);
}

namespace {

// The router as D122869562 wrote it: a full WriteBatch::Iterate into a
// ColumnFamilyCollector that also keeps the smallest XID, then the same seeds
// and tie-breaks. Kept as the reference the header-scan router must match.
class ReferenceRoutingCollector final : public ColumnFamilyCollector {
 public:
  Status MarkEndPrepare(const Slice& xid) override { return AddXid(xid); }
  Status MarkRollback(const Slice& xid) override { return AddXid(xid); }
  Status MarkCommit(const Slice& xid) override { return AddXid(xid); }
  Status MarkCommitWithTimestamp(const Slice& xid, const Slice&) override {
    return AddXid(xid);
  }
  Status AddXid(const Slice& xid) {
    if (!has_xid_ || xid.compare(xid_) < 0) {
      xid_ = xid;
      has_xid_ = true;
    }
    return Status::OK();
  }
  bool has_xid_ = false;
  Slice xid_;
};

size_t ReferenceWalPartition(const WriteBatch& batch,
                             const std::vector<uint32_t>& deferred_cf_ids,
                             size_t num_partitions) {
  if (num_partitions == 1) {
    return 0;
  }
  ReferenceRoutingCollector collector;
  EXPECT_OK(batch.Iterate(&collector));
  std::optional<uint32_t> cf;
  for (uint32_t id : collector.column_families()) {
    cf = std::min(cf.value_or(id), id);
  }
  for (uint32_t id : deferred_cf_ids) {
    cf = std::min(cf.value_or(id), id);
  }
  uint64_t hash = 0;
  if (cf.has_value()) {
    char encoded_id[sizeof(uint32_t)];
    EncodeFixed32(encoded_id, cf.value());
    hash = Hash64(encoded_id, sizeof(encoded_id),
                  /*kWalColumnFamilyRoutingSeed=*/0x6a09e667f3bcc909ULL);
  } else if (collector.has_xid_) {
    hash = Hash64(collector.xid_.data(), collector.xid_.size(),
                  /*kWalXidRoutingSeed=*/0xbb67ae8584caa73bULL);
  } else {
    return 0;
  }
  return static_cast<size_t>(hash % num_partitions);
}

// Reports every writable file as not sync-thread-safe, exercising the gate
// sync's fallback (hold the syncing partition's mutex across its fsync).
// Forwards everything else to the target file.
class NonSyncThreadSafeWritableFile : public FSWritableFileWrapper {
 public:
  explicit NonSyncThreadSafeWritableFile(std::unique_ptr<FSWritableFile> target)
      : FSWritableFileWrapper(target.get()), owned_(std::move(target)) {}
  bool IsSyncThreadSafe() const override { return false; }

 private:
  std::unique_ptr<FSWritableFile> owned_;
};

class NonSyncThreadSafeFS : public FileSystemWrapper {
 public:
  using FileSystemWrapper::FileSystemWrapper;
  const char* Name() const override { return "NonSyncThreadSafeFS"; }
  IOStatus NewWritableFile(const std::string& f, const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* r,
                           IODebugContext* dbg) override {
    std::unique_ptr<FSWritableFile> target;
    IOStatus s = FileSystemWrapper::NewWritableFile(f, file_opts, &target, dbg);
    if (s.ok()) {
      r->reset(new NonSyncThreadSafeWritableFile(std::move(target)));
    }
    return s;
  }
};

}  // namespace

TEST(WalRoutingTest, HeaderScanMatchesIteratingRouter) {
  // G-19b: the router reads record headers instead of iterating the batch
  // through a handler, and must route every batch exactly as the iterating
  // router did: same seeds, smallest CF id first, else smallest XID, else 0,
  // deferred PutEntity ops counted. The corpus mixes every record kind the
  // router reads (and LogData and Noop, which it skips) across single-CF,
  // multi-CF, XID-only, LogData-only and deferred-entity batches. Changing
  // the tie-break to the largest CF id fails the multi-CF batches.
  const uint32_t seed = static_cast<uint32_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  SCOPED_TRACE("seed=" + std::to_string(seed));
  Random rnd(seed);
  auto random_cf = [&rnd]() -> uint32_t {
    switch (rnd.Uniform(4)) {
      case 0:
        return 0;
      case 1:
        return rnd.Uniform(8);
      case 2:
        return rnd.Uniform(1000);
      default:
        return std::numeric_limits<uint32_t>::max() - rnd.Uniform(4);
    }
  };
  auto random_xid = [&rnd]() {
    return "xid-" + std::to_string(rnd.Uniform(100000));
  };
  int multi_cf = 0;
  int xid_only = 0;
  int log_data_only = 0;
  int deferred = 0;
  for (int i = 0; i < 20000; ++i) {
    WriteBatch batch;
    std::vector<uint32_t> deferred_cf_ids;
    std::set<uint32_t> cfs;
    bool has_xid = false;
    bool has_log_data = false;
    if (rnd.OneIn(8)) {
      // A prepared batch: data records, then the end-prepare marker that
      // replaces the leading noop.
      ASSERT_OK(WriteBatchInternal::InsertNoop(&batch));
      if (rnd.OneIn(2)) {
        const uint32_t cf = random_cf();
        cfs.insert(cf);
        ASSERT_OK(WriteBatchInternal::Put(&batch, cf, "k", "v"));
      }
      ASSERT_OK(WriteBatchInternal::MarkEndPrepare(&batch, random_xid()));
      has_xid = true;
    } else {
      const int records = static_cast<int>(rnd.Uniform(6));
      for (int r = 0; r < records; ++r) {
        const uint32_t cf = random_cf();
        switch (rnd.Uniform(13)) {
          case 0:
            ASSERT_OK(WriteBatchInternal::Put(&batch, cf, "k", "v"));
            cfs.insert(cf);
            break;
          case 1:
            ASSERT_OK(WriteBatchInternal::Delete(&batch, cf, Slice("k")));
            cfs.insert(cf);
            break;
          case 2:
            ASSERT_OK(WriteBatchInternal::SingleDelete(&batch, cf, Slice("k")));
            cfs.insert(cf);
            break;
          case 3:
            ASSERT_OK(WriteBatchInternal::DeleteRange(&batch, cf, Slice("a"),
                                                      Slice("b")));
            cfs.insert(cf);
            break;
          case 4:
            ASSERT_OK(
                WriteBatchInternal::Merge(&batch, cf, Slice("k"), Slice("v")));
            cfs.insert(cf);
            break;
          case 5:
            ASSERT_OK(WriteBatchInternal::PutEntity(&batch, cf, "k",
                                                    WideColumns{{"c", "v"}}));
            cfs.insert(cf);
            break;
          case 6:
            ASSERT_OK(WriteBatchInternal::TimedPut(&batch, cf, "k", "v", 1));
            cfs.insert(cf);
            break;
          case 7:
            ASSERT_OK(WriteBatchInternal::PutBlobIndex(&batch, cf, "k", "v"));
            cfs.insert(cf);
            break;
          case 8:
            ASSERT_OK(batch.PutLogData("blob"));
            has_log_data = true;
            break;
          case 9:
            ASSERT_OK(WriteBatchInternal::MarkCommit(&batch, random_xid()));
            has_xid = true;
            break;
          case 10:
            ASSERT_OK(WriteBatchInternal::MarkRollback(&batch, random_xid()));
            has_xid = true;
            break;
          case 11:
            ASSERT_OK(WriteBatchInternal::MarkCommitWithTimestamp(
                &batch, random_xid(), "ts"));
            has_xid = true;
            break;
          default:
            ASSERT_OK(WriteBatchInternal::InsertNoop(&batch));
            break;
        }
      }
    }
    if (rnd.OneIn(6)) {
      const int ops = 1 + static_cast<int>(rnd.Uniform(3));
      for (int d = 0; d < ops; ++d) {
        deferred_cf_ids.push_back(random_cf());
      }
      ++deferred;
    }
    multi_cf += cfs.size() > 1 ? 1 : 0;
    xid_only += cfs.empty() && has_xid ? 1 : 0;
    log_data_only += cfs.empty() && !has_xid && has_log_data ? 1 : 0;
    for (size_t n :
         {size_t{1}, size_t{2}, size_t{4}, size_t{8}, size_t{16}, size_t{64}}) {
      size_t partition = n;
      ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(
          batch, n, &partition,
          deferred_cf_ids.empty() ? nullptr : &deferred_cf_ids));
      ASSERT_EQ(ReferenceWalPartition(batch, deferred_cf_ids, n), partition)
          << "batch " << i << " N=" << n;
    }
  }
  // The corpus reached every category it exists for.
  EXPECT_GT(multi_cf, 1000);
  EXPECT_GT(xid_only, 100);
  EXPECT_GT(log_data_only, 10);
  EXPECT_GT(deferred, 1000);
}

TEST(WalRoutingTest, MalformedBatchRoutesDeterministically) {
  // G-19b: the router never fails. A batch whose records stop parsing routes
  // on what was read before the damage (else partition 0), the same every
  // time; the write path reports the corruption (MalformedBatchFailsAsAtN1).
  WriteBatch valid;
  ASSERT_OK(WriteBatchInternal::Put(&valid, 7, "k", "v"));
  size_t valid_partition = 8;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(valid, 8, &valid_partition));
  for (const char garbage : {'\x7f', '\x01'}) {
    WriteBatch damaged;
    std::string rep = valid.Data();
    rep.push_back(garbage);
    ASSERT_OK(WriteBatchInternal::SetContents(&damaged, rep));
    ReferenceRoutingCollector collector;
    ASSERT_NOK(damaged.Iterate(&collector));
    size_t partition = 8;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(damaged, 8, &partition));
    EXPECT_EQ(valid_partition, partition);
  }
}

TEST(WriteThreadPartitionTest, DifferentPartitionsFormSeparateGroups) {
  // Q8 ([05-1]): a write group becomes one WAL record and cannot span
  // partitions, so group formation excludes a writer whose wal_partition
  // differs from the leader's. Three writers join one WriteThread with
  // partitions {0, 0, 1}; the leader's formation must group the two
  // partition-0 writers and leave the partition-1 writer for the next
  // group. Removing the clause groups all three and fails this test.
  // (Through the DB this is unreachable: each partition has its own write
  // thread, so cross-partition writers never meet in one formation.)
  WriteThread write_thread{ImmutableDBOptions{}};

  WriteBatch batch_a, batch_b, batch_c;
  ASSERT_OK(batch_a.Put("a", "v"));
  ASSERT_OK(batch_b.Put("b", "v"));
  ASSERT_OK(batch_c.Put("c", "v"));
  WriteThread::Writer wa(WriteOptions(), &batch_a, nullptr, nullptr, 0, false);
  WriteThread::Writer wb(WriteOptions(), &batch_b, nullptr, nullptr, 0, false);
  WriteThread::Writer wc(WriteOptions(), &batch_c, nullptr, nullptr, 0, false);
  wa.wal_partition = 0;
  wb.wal_partition = 0;
  wc.wal_partition = 1;

  // A links as leader and returns at once; B and C queue behind it and
  // wait. Formation runs only once both followers are queued.
  std::mutex mu;
  std::condition_variable cv;
  int waiting = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:BeganWaiting", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        ++waiting;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->EnableProcessing();

  write_thread.JoinBatchGroup(&wa);
  port::Thread thread_b([&] { write_thread.JoinBatchGroup(&wb); });
  port::Thread thread_c([&] { write_thread.JoinBatchGroup(&wc); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(10),
                            [&] { return waiting == 2; }));
  }

  WriteThread::WriteGroup group;
  write_thread.EnterAsBatchGroupLeader(&wa, &group);
  ASSERT_EQ(2u, group.size);
  ASSERT_EQ(0u, group.num_cap_excluded);
  std::vector<WriteThread::Writer*> members;
  for (auto* w : group) {
    members.push_back(w);
  }
  ASSERT_EQ(2u, members.size());
  EXPECT_EQ(&wa, members[0]);
  EXPECT_EQ(&wb, members[1]);

  // Exiting completes B and promotes C to the next group leader.
  Status status;
  write_thread.ExitAsBatchGroupLeader(group, status);
  thread_b.join();
  thread_c.join();
  EXPECT_EQ(WriteThread::STATE_COMPLETED, wb.state.load());
  EXPECT_EQ(WriteThread::STATE_GROUP_LEADER, wc.state.load());

  // The excluded writer forms a group of its own.
  WriteThread::WriteGroup group2;
  write_thread.EnterAsBatchGroupLeader(&wc, &group2);
  EXPECT_EQ(1u, group2.size);
  write_thread.ExitAsBatchGroupLeader(group2, status);

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST(WalDurabilityTrackerTest, WorkedExampleGivesSafePointOne) {
  // 1:[1,4,5,6] 2:[2,7] 3:[3,8,9] with 1 durable and 2 pending.
  WalDurabilityTracker tracker(3);
  tracker.Register(0, 1, 1);
  tracker.Register(0, 4, 4);
  tracker.Register(0, 5, 5);
  tracker.Register(0, 6, 6);
  tracker.Register(1, 2, 2);
  tracker.Register(1, 7, 7);
  tracker.Register(2, 3, 3);
  tracker.Register(2, 8, 8);
  tracker.Register(2, 9, 9);
  EXPECT_EQ(0, tracker.SafePoint());
  tracker.Complete(0, 1, 1);
  EXPECT_EQ(4, tracker.PendingLow(0));
  EXPECT_EQ(2, tracker.PendingLow(1));
  EXPECT_EQ(3, tracker.PendingLow(2));
  EXPECT_EQ(1, tracker.SafePoint());
}

TEST(WalDurabilityTrackerTest, AllIdleGivesMaxDurable) {
  WalDurabilityTracker tracker(3);
  tracker.Register(0, 1, 1);
  tracker.Register(1, 2, 2);
  tracker.Register(2, 3, 3);
  tracker.Complete(1, 2, 2);
  tracker.Complete(0, 1, 1);
  EXPECT_EQ(2, tracker.SafePoint());
  tracker.Complete(2, 3, 3);
  EXPECT_EQ(3, tracker.SafePoint());
  EXPECT_EQ(WalDurabilityTracker::kIdle, tracker.PendingLow(0));
  EXPECT_EQ(WalDurabilityTracker::kIdle, tracker.PendingLow(1));
  EXPECT_EQ(WalDurabilityTracker::kIdle, tracker.PendingLow(2));
}

TEST(WalDurabilityTrackerTest, OutOfOrderCompletionPinsAtLowestPending) {
  WalDurabilityTracker tracker(2);
  tracker.Register(0, 1, 1);
  tracker.Register(1, 2, 2);
  // Higher completes first: the safe point must not jump past the hole.
  tracker.Complete(1, 2, 2);
  EXPECT_EQ(0, tracker.SafePoint());
  tracker.Complete(0, 1, 1);
  EXPECT_EQ(2, tracker.SafePoint());
}

TEST(WalDurabilityTrackerTest, DuplicateSequenceStaysUntilBothDurable) {
  WalDurabilityTracker tracker(2);
  tracker.Register(0, 1, 1);
  tracker.Register(1, 1, 1);
  tracker.Complete(0, 1, 1);
  EXPECT_EQ(0, tracker.SafePoint());
  EXPECT_EQ(1, tracker.PendingLow(1));
  tracker.Complete(1, 1, 1);
  EXPECT_EQ(1, tracker.SafePoint());
}

TEST(WalDurabilityTrackerTest, DuplicateReverseOrder) {
  WalDurabilityTracker tracker(2);
  tracker.Register(0, 1, 1);
  tracker.Register(1, 1, 1);
  tracker.Complete(1, 1, 1);
  EXPECT_EQ(0, tracker.SafePoint());
  tracker.Complete(0, 1, 1);
  EXPECT_EQ(1, tracker.SafePoint());
}

TEST(WalDurabilityTrackerTest, ZeroCountEntryStaysBelowItsLow) {
  // GC20 Q3 (K1): a record that consumes no sequence registers the empty
  // range [S, S - 1], S = last allocated + 1. While outstanding it pins S;
  // completed while every other partition is idle, it leaves the safe point
  // at S - 1, since S is not allocated yet. The next real write takes S and
  // holds the safe point below it until it completes.
  // Neuter: treat [S, S - 1] as [S, S] (high = max(high, low) in Register
  // and Complete); the idle safe point becomes S.
  WalDurabilityTracker tracker(2);
  tracker.Register(0, 1, 4);
  tracker.Complete(0, 1, 4);
  EXPECT_EQ(4, tracker.SafePoint());
  tracker.Register(1, 5, 4);
  EXPECT_EQ(5, tracker.PendingLow(1));
  EXPECT_EQ(4, tracker.SafePoint());
  tracker.Complete(1, 5, 4);
  EXPECT_EQ(WalDurabilityTracker::kIdle, tracker.PendingLow(1));
  EXPECT_EQ(4, tracker.DurableUpto(1));
  EXPECT_EQ(4, tracker.SafePoint());
  tracker.Register(0, 5, 6);
  tracker.Register(1, 7, 7);
  tracker.Complete(1, 7, 7);
  EXPECT_EQ(4, tracker.SafePoint());
  tracker.Complete(0, 5, 6);
  EXPECT_EQ(7, tracker.SafePoint());
}

TEST(WalDurabilityTrackerTest, RecomputeCostAt16Partitions) {
  // Loose bound for dev+asan; the real evidence is the -O2 standalone
  // measurement reported in the test plan. This guards against algorithmic
  // regressions (e.g. accidentally quadratic rescans), not nanoseconds.
  constexpr size_t kPartitions = 16;
  constexpr int kIters = 20000;
  WalDurabilityTracker tracker(kPartitions);
  for (size_t p = 0; p < kPartitions; ++p) {
    tracker.Register(p, 100 + p, 100 + p);
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    tracker.ForceRecalculate();
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const double ns_per_op =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
              .count()) /
      kIters;
  fprintf(stderr, "WalDurabilityTracker rescan at 16 partitions: %.1f ns/op\n",
          ns_per_op);
  EXPECT_LT(ns_per_op, 5000);
}

TEST(WalDurabilityTrackerTest, ThreadedReleaseAcquireIsClean) {
  // One thread per partition: cross-partition visibility runs entirely
  // through the tracker's release/acquire pairing, which TSan validates.
  constexpr size_t kPartitions = 8;
  constexpr int kRounds = 200;
  WalDurabilityTracker tracker(kPartitions);
  std::vector<port::Thread> threads;
  for (size_t p = 0; p < kPartitions; ++p) {
    threads.emplace_back([&tracker, p] {
      for (int i = 0; i < kRounds; ++i) {
        const uint64_t seq = 1 + p * kRounds + i;
        tracker.Register(p, seq, seq);
        tracker.Complete(p, seq, seq);
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  tracker.ForceRecalculate();
  EXPECT_EQ(kPartitions * kRounds, tracker.SafePoint());
}

// It is invalid to do sync write while disabling WAL.
TEST_P(DBWriteTest, SyncAndDisableWAL) {
  WriteOptions write_options;
  write_options.sync = true;
  write_options.disableWAL = true;
  ASSERT_TRUE(dbfull()->Put(write_options, "foo", "bar").IsInvalidArgument());
  WriteBatch batch;
  ASSERT_OK(batch.Put("foo", "bar"));
  ASSERT_TRUE(dbfull()->Write(write_options, &batch).IsInvalidArgument());
}

TEST_P(DBWriteTest, WriteStallRemoveNoSlowdownWrite) {
  Options options = GetOptions();
  options.level0_stop_writes_trigger = options.level0_slowdown_writes_trigger =
      4;
  std::vector<port::Thread> threads;
  std::atomic<int> thread_num(0);
  port::Mutex mutex;
  port::CondVar cv(&mutex);
  // Guarded by mutex
  int writers = 0;

  Reopen(options);

  std::function<void()> write_slowdown_func = [&]() {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions wo;
    wo.no_slowdown = false;
    ASSERT_OK(dbfull()->Put(wo, key, "bar"));
  };
  std::function<void()> write_no_slowdown_func = [&]() {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions wo;
    wo.no_slowdown = true;
    Status s = dbfull()->Put(wo, key, "bar");
    ASSERT_TRUE(s.ok() || s.IsIncomplete());
  };
  std::function<void(void*)> unblock_main_thread_func = [&](void*) {
    mutex.Lock();
    ++writers;
    cv.SignalAll();
    mutex.Unlock();
  };

  // Create 3 L0 files and schedule 4th without waiting
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Start", unblock_main_thread_func);
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteTest::WriteStallRemoveNoSlowdownWrite:1",
        "DBImpl::BackgroundCallFlush:start"},
       {"DBWriteTest::WriteStallRemoveNoSlowdownWrite:2",
        "DBImplWrite::PipelinedWriteImpl:AfterJoinBatchGroup"},
       // Make compaction start wait for the write stall to be detected and
       // implemented by a write group leader
       {"DBWriteTest::WriteStallRemoveNoSlowdownWrite:3",
        "BackgroundCallCompaction:0"}});
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  // Schedule creation of 4th L0 file without waiting. This will seal the
  // memtable and then wait for a sync point before writing the file. We need
  // to do it this way because SwitchMemtable() needs to enter the
  // write_thread
  FlushOptions fopt;
  fopt.wait = false;
  ASSERT_OK(dbfull()->Flush(fopt));

  // Create a mix of slowdown/no_slowdown write threads
  mutex.Lock();
  // First leader
  threads.emplace_back(write_slowdown_func);
  while (writers != 1) {
    cv.Wait();
  }

  // Second leader. Will stall writes
  // Build a writers list with no slowdown in the middle:
  //  +-------------+
  //  | slowdown    +<----+ newest
  //  +--+----------+
  //     |
  //     v
  //  +--+----------+
  //  | no slowdown |
  //  +--+----------+
  //     |
  //     v
  //  +--+----------+
  //  | slowdown    +
  //  +-------------+
  threads.emplace_back(write_slowdown_func);
  while (writers != 2) {
    cv.Wait();
  }
  threads.emplace_back(write_no_slowdown_func);
  while (writers != 3) {
    cv.Wait();
  }
  threads.emplace_back(write_slowdown_func);
  while (writers != 4) {
    cv.Wait();
  }

  mutex.Unlock();

  TEST_SYNC_POINT("DBWriteTest::WriteStallRemoveNoSlowdownWrite:1");
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(nullptr));
  // This would have triggered a write stall. Unblock the write group leader
  TEST_SYNC_POINT("DBWriteTest::WriteStallRemoveNoSlowdownWrite:2");
  // The leader is going to create missing newer links. When the leader
  // finishes, the next leader is going to delay writes and fail writers with
  // no_slowdown

  TEST_SYNC_POINT("DBWriteTest::WriteStallRemoveNoSlowdownWrite:3");
  for (auto& t : threads) {
    t.join();
  }

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST_P(DBWriteTest, WriteThreadHangOnWriteStall) {
  Options options = GetOptions();
  options.level0_stop_writes_trigger = options.level0_slowdown_writes_trigger =
      4;
  std::vector<port::Thread> threads;
  std::atomic<int> thread_num(0);
  port::Mutex mutex;
  port::CondVar cv(&mutex);
  // Guarded by mutex
  int writers = 0;

  Reopen(options);

  std::function<void()> write_slowdown_func = [&]() {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions wo;
    wo.no_slowdown = false;
    ASSERT_OK(dbfull()->Put(wo, key, "bar"));
  };
  std::function<void()> write_no_slowdown_func = [&]() {
    int a = thread_num.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions wo;
    wo.no_slowdown = true;
    Status s = dbfull()->Put(wo, key, "bar");
    ASSERT_TRUE(s.ok() || s.IsIncomplete());
  };
  std::function<void(void*)> unblock_main_thread_func = [&](void*) {
    mutex.Lock();
    ++writers;
    cv.SignalAll();
    mutex.Unlock();
  };

  // Create 3 L0 files and schedule 4th without waiting
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo" + std::to_string(thread_num.fetch_add(1)), "bar"));

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Start", unblock_main_thread_func);
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteTest::WriteThreadHangOnWriteStall:1",
        "DBImpl::BackgroundCallFlush:start"},
       {"DBWriteTest::WriteThreadHangOnWriteStall:2",
        "DBImpl::WriteImpl:BeforeLeaderEnters"},
       // Make compaction start wait for the write stall to be detected and
       // implemented by a write group leader
       {"DBWriteTest::WriteThreadHangOnWriteStall:3",
        "BackgroundCallCompaction:0"}});
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  // Schedule creation of 4th L0 file without waiting. This will seal the
  // memtable and then wait for a sync point before writing the file. We need
  // to do it this way because SwitchMemtable() needs to enter the
  // write_thread
  FlushOptions fopt;
  fopt.wait = false;
  ASSERT_OK(dbfull()->Flush(fopt));

  // Create a mix of slowdown/no_slowdown write threads
  mutex.Lock();
  // First leader
  threads.emplace_back(write_slowdown_func);
  while (writers != 1) {
    cv.Wait();
  }
  // Second leader. Will stall writes
  threads.emplace_back(write_slowdown_func);
  threads.emplace_back(write_no_slowdown_func);
  threads.emplace_back(write_slowdown_func);
  threads.emplace_back(write_no_slowdown_func);
  threads.emplace_back(write_slowdown_func);
  while (writers != 6) {
    cv.Wait();
  }
  mutex.Unlock();

  TEST_SYNC_POINT("DBWriteTest::WriteThreadHangOnWriteStall:1");
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(nullptr));
  // This would have triggered a write stall. Unblock the write group leader
  TEST_SYNC_POINT("DBWriteTest::WriteThreadHangOnWriteStall:2");
  // The leader is going to create missing newer links. When the leader
  // finishes, the next leader is going to delay writes and fail writers with
  // no_slowdown

  TEST_SYNC_POINT("DBWriteTest::WriteThreadHangOnWriteStall:3");
  for (auto& t : threads) {
    t.join();
  }
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST_P(DBWriteTest, WriteThreadWaitNanosCounter) {
  Options options = GetOptions();
  std::vector<port::Thread> threads;

  Reopen(options);

  std::function<void()> write_func = [&]() {
    PerfContext* perf_ctx = get_perf_context();
    SetPerfLevel(PerfLevel::kEnableWait);
    perf_ctx->Reset();
    TEST_SYNC_POINT("DBWriteTest::WriteThreadWaitNanosCounter:WriteFunc");
    ASSERT_OK(dbfull()->Put(WriteOptions(), "bar", "val2"));
    ASSERT_GT(perf_ctx->write_thread_wait_nanos, 2000000U);
  };

  std::function<void()> sleep_func = [&]() {
    TEST_SYNC_POINT("DBWriteTest::WriteThreadWaitNanosCounter:SleepFunc:1");
    SystemClock::Default()->SleepForMicroseconds(2000);
    TEST_SYNC_POINT("DBWriteTest::WriteThreadWaitNanosCounter:SleepFunc:2");
  };

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"WriteThread::EnterAsBatchGroupLeader:End",
        "DBWriteTest::WriteThreadWaitNanosCounter:WriteFunc"},
       {"WriteThread::AwaitState:BlockingWaiting",
        "DBWriteTest::WriteThreadWaitNanosCounter:SleepFunc:1"},
       {"DBWriteTest::WriteThreadWaitNanosCounter:SleepFunc:2",
        "WriteThread::ExitAsBatchGroupLeader:Start"}});
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  threads.emplace_back(sleep_func);
  threads.emplace_back(write_func);

  ASSERT_OK(dbfull()->Put(WriteOptions(), "foo", "val1"));

  for (auto& t : threads) {
    t.join();
  }
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

TEST_P(DBWriteTest, IOErrorOnWALWritePropagateToWriteThreadFollower) {
  constexpr int kNumThreads = 5;
  std::unique_ptr<FaultInjectionTestEnv> mock_env(
      new FaultInjectionTestEnv(env_));
  Options options = GetOptions();
  options.env = mock_env.get();
  Reopen(options);
  std::atomic<int> ready_count{0};
  std::atomic<int> leader_count{0};
  std::vector<port::Thread> threads;
  mock_env->SetFilesystemActive(false);

  // Wait until all threads linked to write threads, to make sure
  // all threads join the same batch group.
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        ready_count++;
        auto* w = static_cast<WriteThread::Writer*>(arg);
        if (w->state == WriteThread::STATE_GROUP_LEADER) {
          leader_count++;
          while (ready_count < kNumThreads) {
            // busy waiting
          }
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back(
        [&](int index) {
          // All threads should fail.
          auto res = Put("key" + std::to_string(index), "value");
          if (options.manual_wal_flush) {
            ASSERT_TRUE(res.ok());
            // we should see fs error when we do the flush

            // TSAN reports a false alarm for lock-order-inversion but Open and
            // FlushWAL are not run concurrently. Disabling this until TSAN is
            // fixed.
            // res = dbfull()->FlushWAL(false);
            // ASSERT_FALSE(res.ok());
          } else {
            ASSERT_FALSE(res.ok());
          }
        },
        i);
  }
  for (int i = 0; i < kNumThreads; i++) {
    threads[i].join();
  }
  ASSERT_EQ(1, leader_count);

  // The Failed PUT operations can cause a BG error to be set.
  // Mark it as Checked for the ASSERT_STATUS_CHECKED
  dbfull()->Resume().PermitUncheckedError();

  // Close before mock_env destruct.
  Close();
}

TEST_F(DBWriteTestUnparameterized, PartitionMergeBufferLifecycle) {
  for (const bool two_write_queues : {false, true}) {
    SCOPED_TRACE("two_write_queues=" + std::to_string(two_write_queues));
    Options options = CurrentOptions();
    options.create_if_missing = true;
    options.two_write_queues = two_write_queues;
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);

    const WriteBatch* const partition_merge_buffer =
        dbfull()->TEST_GetWalMergeBuffer(0);
    std::atomic<bool> expect_merged_batch{false};
    std::atomic<bool> single_writer_used_merge_buffer{false};
    std::atomic<bool> group_borrowed_merge_buffer{false};
    std::atomic<int> group_wal_records{0};
    std::string group_wal_payload;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
          const auto* log_entry = static_cast<Slice*>(arg);
          const Slice merge_buffer_contents =
              WriteBatchInternal::Contents(partition_merge_buffer);
          const bool borrowed_merge_buffer =
              log_entry->data() == merge_buffer_contents.data();
          if (expect_merged_batch.load()) {
            group_borrowed_merge_buffer = borrowed_merge_buffer;
            group_wal_payload.assign(log_entry->data(), log_entry->size());
            ++group_wal_records;
          } else {
            single_writer_used_merge_buffer = borrowed_merge_buffer;
          }
        });
    SyncPoint::GetInstance()->EnableProcessing();

    ASSERT_OK(Put("single", "value"));
    ASSERT_FALSE(single_writer_used_merge_buffer.load());

    std::atomic<bool> follower_started{false};
    std::atomic<bool> follower_joined{false};
    Status follower_status;
    port::Thread follower_thread;
    SyncPoint::GetInstance()->SetCallBack(
        "WriteThread::JoinBatchGroup:Wait2", [&](void* arg) {
          auto* writer = static_cast<WriteThread::Writer*>(arg);
          if (writer->state != WriteThread::STATE_GROUP_LEADER ||
              follower_started.exchange(true)) {
            return;
          }
          SyncPoint::GetInstance()->SetCallBack(
              "WriteThread::JoinBatchGroup:Wait", [&](void* follower_arg) {
                auto* follower =
                    static_cast<WriteThread::Writer*>(follower_arg);
                if (follower->state != WriteThread::STATE_GROUP_LEADER) {
                  follower_joined = true;
                }
              });
          follower_thread = port::Thread(
              [&]() { follower_status = Put("follower", "value"); });
          while (!follower_joined.load()) {
          }
        });

    expect_merged_batch = true;
    const Status leader_status = Put("leader", "value");
    follower_thread.join();
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();

    ASSERT_OK(leader_status);
    ASSERT_OK(follower_status);
    ASSERT_EQ(1, group_wal_records.load());
    // ConcurrentWriteGroupToWAL merges into invocation-local scratch and must
    // never reach the partition buffer.
    ASSERT_EQ(!two_write_queues, group_borrowed_merge_buffer.load());
    ASSERT_EQ(0u, WriteBatchInternal::Count(partition_merge_buffer));

    WriteBatch expected_group;
    ASSERT_OK(expected_group.Put("leader", "value"));
    ASSERT_OK(expected_group.Put("follower", "value"));
    WriteBatchInternal::SetSequence(&expected_group, 2);
    ASSERT_EQ(WriteBatchInternal::Contents(&expected_group), group_wal_payload);
    ASSERT_EQ("value", Get("leader"));
    ASSERT_EQ("value", Get("follower"));

    Reopen(options);
    ASSERT_EQ("value", Get("single"));
    ASSERT_EQ("value", Get("leader"));
    ASSERT_EQ("value", Get("follower"));
  }
}

TEST_F(DBWriteTestUnparameterized, WriteGroupHistograms) {
  // One leader plus two followers gated so all three land in a single write
  // group, across the three write paths that form WAL-side groups. Each path
  // must record one writers-per-group sample, one cap-exclusion sample, and
  // one join-wait sample per write.
  const std::vector<std::string> modes = {"default", "pipelined_write",
                                          "unordered_write"};
  for (const auto& mode : modes) {
    SCOPED_TRACE("mode=" + mode);
    Options options = CurrentOptions();
    options.create_if_missing = true;
    options.statistics = ROCKSDB_NAMESPACE::CreateDBStatistics();
    // GC28 Q6: the join timer records only above kExceptDetailedTimers.
    options.statistics->set_stats_level(StatsLevel::kExceptTimeForMutex);
    options.enable_pipelined_write = (mode == "pipelined_write");
    options.unordered_write = (mode == "unordered_write");
    DestroyAndReopen(options);

    constexpr int kFollowers = 2;
    std::atomic<bool> group_started{false};
    std::atomic<int> followers_joined{0};
    std::vector<port::Thread> follower_threads;
    std::vector<Status> follower_statuses(kFollowers);
    SyncPoint::GetInstance()->SetCallBack(
        "WriteThread::JoinBatchGroup:Wait2", [&](void* arg) {
          auto* writer = static_cast<WriteThread::Writer*>(arg);
          if (writer->state != WriteThread::STATE_GROUP_LEADER ||
              group_started.exchange(true)) {
            return;
          }
          SyncPoint::GetInstance()->SetCallBack(
              "WriteThread::JoinBatchGroup:Wait", [&](void* follower_arg) {
                auto* follower =
                    static_cast<WriteThread::Writer*>(follower_arg);
                if (follower->state != WriteThread::STATE_GROUP_LEADER) {
                  ++followers_joined;
                }
              });
          for (int i = 0; i < kFollowers; ++i) {
            follower_threads.emplace_back(
                [&](int index) {
                  follower_statuses[index] =
                      Put("follower" + std::to_string(index), "value");
                },
                i);
          }
          while (followers_joined.load() < kFollowers) {
          }
        });
    SyncPoint::GetInstance()->EnableProcessing();

    const Status leader_status = Put("leader", "value");
    for (auto& t : follower_threads) {
      t.join();
    }
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();

    ASSERT_OK(leader_status);
    for (const auto& s : follower_statuses) {
      ASSERT_OK(s);
    }
    ASSERT_EQ("value", Get("leader"));
    ASSERT_EQ("value", Get("follower0"));
    ASSERT_EQ("value", Get("follower1"));

    HistogramData writers_per_group;
    options.statistics->histogramData(WRITERS_PER_WRITE_GROUP,
                                      &writers_per_group);
    ASSERT_EQ(1u, writers_per_group.count);
    ASSERT_EQ(3u, writers_per_group.sum);
    ASSERT_EQ(3, writers_per_group.max);

    HistogramData cap_excluded;
    options.statistics->histogramData(WRITE_GROUP_CAP_EXCLUDED, &cap_excluded);
    ASSERT_EQ(1u, cap_excluded.count);
    ASSERT_EQ(0u, cap_excluded.sum);

    HistogramData join_wait;
    options.statistics->histogramData(JOIN_BATCH_GROUP_WAIT_MICROS, &join_wait);
    ASSERT_EQ(3u, join_wait.count);
    ASSERT_GT(join_wait.sum, 0u);
  }
}

TEST_F(DBWriteTestUnparameterized, WriteGroupCapExcludedHistogram) {
  // A group-size cap smaller than two batches: the follower cannot join the
  // leader's group and forms a second group alone. The cap-exclusion
  // histogram must show the one turned-away writer.
  Options options = CurrentOptions();
  options.create_if_missing = true;
  options.statistics = ROCKSDB_NAMESPACE::CreateDBStatistics();
  // GC28 Q6: the join timer records only above kExceptDetailedTimers.
  options.statistics->set_stats_level(StatsLevel::kExceptTimeForMutex);
  options.max_write_batch_group_size_bytes = 1024;
  DestroyAndReopen(options);

  const std::string big_value(1500, 'v');
  std::atomic<bool> group_started{false};
  std::atomic<bool> follower_joined{false};
  Status follower_status;
  port::Thread follower_thread;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait2", [&](void* arg) {
        auto* writer = static_cast<WriteThread::Writer*>(arg);
        if (writer->state != WriteThread::STATE_GROUP_LEADER ||
            group_started.exchange(true)) {
          return;
        }
        SyncPoint::GetInstance()->SetCallBack(
            "WriteThread::JoinBatchGroup:Wait", [&](void* follower_arg) {
              auto* follower = static_cast<WriteThread::Writer*>(follower_arg);
              if (follower->state != WriteThread::STATE_GROUP_LEADER) {
                follower_joined = true;
              }
            });
        follower_thread = port::Thread(
            [&]() { follower_status = Put("follower", big_value); });
        while (!follower_joined.load()) {
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();

  const Status leader_status = Put("leader", big_value);
  follower_thread.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();

  ASSERT_OK(leader_status);
  ASSERT_OK(follower_status);
  ASSERT_EQ(big_value, Get("leader"));
  ASSERT_EQ(big_value, Get("follower"));

  HistogramData writers_per_group;
  options.statistics->histogramData(WRITERS_PER_WRITE_GROUP,
                                    &writers_per_group);
  ASSERT_EQ(2u, writers_per_group.count);
  ASSERT_EQ(2u, writers_per_group.sum);
  ASSERT_EQ(1, writers_per_group.max);

  HistogramData cap_excluded;
  options.statistics->histogramData(WRITE_GROUP_CAP_EXCLUDED, &cap_excluded);
  ASSERT_EQ(2u, cap_excluded.count);
  ASSERT_EQ(1u, cap_excluded.sum);
  ASSERT_EQ(1, cap_excluded.max);

  HistogramData join_wait;
  options.statistics->histogramData(JOIN_BATCH_GROUP_WAIT_MICROS, &join_wait);
  ASSERT_EQ(2u, join_wait.count);
}

TEST_F(DBWriteTestUnparameterized, WriteGroupCapExcludedAttribution) {
  // One cap exclusion and one compatibility exclusion in the same group
  // formation: only the cap exclusion must be counted. The leader is a
  // non-sync write whose batch alone exceeds the cap; behind it queue a sync
  // write (incompatible with a non-sync leader) and a compatible non-sync
  // write that cannot fit under the cap. Sync mismatch is the non-cap
  // exclusion because it is the simplest to force -- one WriteOptions flag,
  // no helper class -- and it exercises the first incompatible clause.
  //
  // Followers spawn sequentially: the Wait sync point fires after LinkOne, so
  // a counted follower is already queued, and waiting for each before starting
  // the next fixes the queue order deterministically. Group 1 is the leader
  // alone (cap sample 1); group 2 is the sync follower leading with the small
  // compatible follower joining (cap sample 0), since both small batches fit
  // under the small-leader effective cap.
  Options options = CurrentOptions();
  options.create_if_missing = true;
  options.statistics = ROCKSDB_NAMESPACE::CreateDBStatistics();
  // GC28 Q6: the join timer records only above kExceptDetailedTimers.
  options.statistics->set_stats_level(StatsLevel::kExceptTimeForMutex);
  options.max_write_batch_group_size_bytes = 1024;
  DestroyAndReopen(options);

  const std::string big_value(1500, 'v');
  std::atomic<bool> group_started{false};
  std::atomic<int> followers_joined{0};
  Status sync_status;
  Status compat_status;
  port::Thread sync_thread;
  port::Thread compat_thread;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait2", [&](void* arg) {
        auto* writer = static_cast<WriteThread::Writer*>(arg);
        if (writer->state != WriteThread::STATE_GROUP_LEADER ||
            group_started.exchange(true)) {
          return;
        }
        SyncPoint::GetInstance()->SetCallBack(
            "WriteThread::JoinBatchGroup:Wait", [&](void* follower_arg) {
              auto* follower = static_cast<WriteThread::Writer*>(follower_arg);
              if (follower->state != WriteThread::STATE_GROUP_LEADER) {
                ++followers_joined;
              }
            });
        sync_thread = port::Thread([&]() {
          WriteOptions sync_options;
          sync_options.sync = true;
          sync_status = Put("sync_follower", "v", sync_options);
        });
        while (followers_joined.load() < 1) {
        }
        compat_thread = port::Thread(
            [&]() { compat_status = Put("compat_follower", "v"); });
        while (followers_joined.load() < 2) {
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();

  const Status leader_status = Put("leader", big_value);
  sync_thread.join();
  compat_thread.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();

  ASSERT_OK(leader_status);
  ASSERT_OK(sync_status);
  ASSERT_OK(compat_status);
  ASSERT_EQ(big_value, Get("leader"));
  ASSERT_EQ("v", Get("sync_follower"));
  ASSERT_EQ("v", Get("compat_follower"));

  HistogramData writers_per_group;
  options.statistics->histogramData(WRITERS_PER_WRITE_GROUP,
                                    &writers_per_group);
  ASSERT_EQ(2u, writers_per_group.count);
  ASSERT_EQ(3u, writers_per_group.sum);
  ASSERT_EQ(2, writers_per_group.max);

  // Exactly one cap exclusion across both groups. Incrementing
  // num_cap_excluded on every exclusion would record 2 in group 1 and fail
  // the sum and max assertions below.
  HistogramData cap_excluded;
  options.statistics->histogramData(WRITE_GROUP_CAP_EXCLUDED, &cap_excluded);
  ASSERT_EQ(2u, cap_excluded.count);
  ASSERT_EQ(1u, cap_excluded.sum);
  ASSERT_EQ(1, cap_excluded.max);

  HistogramData join_wait;
  options.statistics->histogramData(JOIN_BATCH_GROUP_WAIT_MICROS, &join_wait);
  ASSERT_EQ(3u, join_wait.count);
  ASSERT_GT(join_wait.sum, 0u);
}

TEST_F(DBWriteTestUnparameterized, PipelinedWriteRace) {
  // This test was written to trigger a race in ExitAsBatchGroupLeader in case
  // enable_pipelined_write_ was true.
  // Writers for which ShouldWriteToMemtable() evaluates to false are removed
  // from the write_group via CompleteFollower/ CompleteLeader. Writers in the
  // middle of the group are fully unlinked, but if that writers is the
  // last_writer, then we did not update the predecessor's link_older, i.e.,
  // this writer was still reachable via newest_writer_.
  //
  // But the problem was, that CompleteFollower already wakes up the thread
  // owning that writer before the writer has been removed. This resulted in a
  // race - if the leader thread was fast enough, then everything was fine.
  // However, if the woken up thread finished the current write operation and
  // then performed yet another write, then a new writer instance was added
  // to newest_writer_. It is possible that the new writer is located on the
  // same address on stack, and if this happened, then we had a problem,
  // because the old code tried to find the last_writer in the list to unlink
  // it, which in this case produced a cycle in the list.
  // Whether two invocations of PipelinedWriteImpl() by the same thread actually
  // allocate the writer on the same address depends on the OS and/or compiler,
  // so it is rather hard to create a deterministic test for this.

  Options options = GetDefaultOptions();
  options.create_if_missing = true;
  options.enable_pipelined_write = true;
  std::vector<port::Thread> threads;

  std::atomic<int> write_counter{0};
  std::atomic<int> active_writers{0};
  std::atomic<bool> second_write_starting{false};
  std::atomic<bool> second_write_in_progress{false};
  std::atomic<WriteThread::Writer*> leader{nullptr};
  std::atomic<bool> finished_WAL_write{false};

  DestroyAndReopen(options);

  auto write_one_doc = [&]() {
    int a = write_counter.fetch_add(1);
    std::string key = "foo" + std::to_string(a);
    WriteOptions wo;
    ASSERT_OK(dbfull()->Put(wo, key, "bar"));
    --active_writers;
  };

  auto write_two_docs = [&]() {
    write_one_doc();
    second_write_starting = true;
    write_one_doc();
  };

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        if (second_write_starting.load()) {
          second_write_in_progress = true;
          return;
        }
        auto* w = static_cast<WriteThread::Writer*>(arg);
        if (w->state == WriteThread::STATE_GROUP_LEADER) {
          active_writers++;
          if (leader.load() == nullptr) {
            leader.store(w);
            while (active_writers.load() < 2) {
              // wait for another thread to join the write_group
            }
          }
        } else {
          // we disable the memtable for all followers so that they they are
          // removed from the write_group before enqueuing it for the memtable
          // write
          w->disable_memtable = true;
          active_writers++;
        }
      });

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::ExitAsBatchGroupLeader:Start", [&](void* arg) {
        auto* wg = static_cast<WriteThread::WriteGroup*>(arg);
        if (wg->leader == leader && !finished_WAL_write) {
          finished_WAL_write = true;
          while (active_writers.load() < 3) {
            // wait for the new writer to be enqueued
          }
        }
      });

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::ExitAsBatchGroupLeader:AfterCompleteWriters",
      [&](void* arg) {
        auto* wg = static_cast<WriteThread::WriteGroup*>(arg);
        if (wg->leader == leader) {
          while (!second_write_in_progress.load()) {
            // wait for the old follower thread to start the next write
          }
        }
      });

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  // start leader + one follower
  threads.emplace_back(write_one_doc);
  while (leader.load() == nullptr) {
    // wait for leader
  }

  // we perform two writes in the follower, so that for the second write
  // the thread reinserts a Writer with the same address
  threads.emplace_back(write_two_docs);

  // wait for the leader to enter ExitAsBatchGroupLeader
  while (!finished_WAL_write.load()) {
    // wait for write_group to have finished the WAL writes
  }

  // start another writer thread to be enqueued before the leader can
  // complete the writers from its write_group
  threads.emplace_back(write_one_doc);

  for (auto& t : threads) {
    t.join();
  }
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST_P(DBWriteTest, ManualWalFlushInEffect) {
  Options options = GetOptions();
  Reopen(options);
  // try the 1st WAL created during open
  ASSERT_TRUE(Put("key" + std::to_string(0), "value").ok());
  ASSERT_TRUE(options.manual_wal_flush != dbfull()->WALBufferIsEmpty());
  ASSERT_TRUE(dbfull()->FlushWAL(false).ok());
  ASSERT_TRUE(dbfull()->WALBufferIsEmpty());
  // try the 2nd wal created during SwitchWAL
  ASSERT_OK(dbfull()->TEST_SwitchWAL());
  ASSERT_TRUE(Put("key" + std::to_string(0), "value").ok());
  ASSERT_TRUE(options.manual_wal_flush != dbfull()->WALBufferIsEmpty());
  ASSERT_TRUE(dbfull()->FlushWAL(false).ok());
  ASSERT_TRUE(dbfull()->WALBufferIsEmpty());
}

TEST_P(DBWriteTest, UnflushedPutRaceWithTrackedWalSync) {
  // Repro race condition bug where unflushed WAL data extended the synced size
  // recorded to MANIFEST despite being unrecoverable.
  Options options = GetOptions();
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  options.env = fault_env.get();
  options.manual_wal_flush = true;
  options.track_and_verify_wals_in_manifest = true;
  Reopen(options);

  ASSERT_OK(Put("key1", "val1"));

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWAL:Begin",
      [this](void* /* arg */) { ASSERT_OK(Put("key2", "val2")); });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  ASSERT_OK(db_->FlushWAL(true /* sync */));

  // Ensure callback ran.
  ASSERT_EQ("val2", Get("key2"));

  Close();

  // Simulate full loss of unsynced data. This drops "key2" -> "val2" from the
  // DB WAL.
  ASSERT_OK(fault_env->DropUnsyncedFileData());

  Reopen(options);

  // Need to close before `fault_env` goes out of scope.
  Close();
}

TEST_P(DBWriteTest, InactiveWalFullySyncedBeforeUntracked) {
  // Repro bug where a WAL is appended and switched after
  // `FlushWAL(true /* sync */)`'s sync finishes and before it untracks fully
  // synced inactive logs. Previously such a WAL would be wrongly untracked
  // so the final append would never be synced.
  Options options = GetOptions();
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  options.env = fault_env.get();
  Reopen(options);

  ASSERT_OK(Put("key1", "val1"));

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWAL:BeforeMarkLogsSynced:1", [this](void* /* arg */) {
        ASSERT_OK(Put("key2", "val2"));
        ASSERT_OK(dbfull()->TEST_SwitchMemtable());
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  ASSERT_OK(db_->FlushWAL(true /* sync */));

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();

  ASSERT_OK(Put("key3", "val3"));

  ASSERT_OK(db_->FlushWAL(true /* sync */));

  Close();

  // Simulate full loss of unsynced data. This should drop nothing since we did
  // `FlushWAL(true /* sync */)` before `Close()`.
  ASSERT_OK(fault_env->DropUnsyncedFileData());

  Reopen(options);

  ASSERT_EQ("val1", Get("key1"));
  ASSERT_EQ("val2", Get("key2"));
  ASSERT_EQ("val3", Get("key3"));

  // Need to close before `fault_env` goes out of scope.
  Close();
}

TEST_P(DBWriteTest, IOErrorOnWALWriteTriggersReadOnlyMode) {
  std::unique_ptr<FaultInjectionTestEnv> mock_env(
      new FaultInjectionTestEnv(env_));
  Options options = GetOptions();
  options.env = mock_env.get();
  Reopen(options);
  for (int i = 0; i < 2; i++) {
    // Forcibly fail WAL write for the first Put only. Subsequent Puts should
    // fail due to read-only mode
    mock_env->SetFilesystemActive(i != 0);
    auto res = Put("key" + std::to_string(i), "value");
    // TSAN reports a false alarm for lock-order-inversion but Open and
    // FlushWAL are not run concurrently. Disabling this until TSAN is
    // fixed.
    /*
    if (options.manual_wal_flush && i == 0) {
      // even with manual_wal_flush the 2nd Put should return error because of
      // the read-only mode
      ASSERT_TRUE(res.ok());
      // we should see fs error when we do the flush
      res = dbfull()->FlushWAL(false);
    }
    */
    if (!options.manual_wal_flush) {
      ASSERT_NOK(res);
    } else {
      ASSERT_OK(res);
    }
  }
  // Close before mock_env destruct.
  Close();
}

TEST_P(DBWriteTest, IOErrorOnSwitchMemtable) {
  Random rnd(301);
  std::unique_ptr<FaultInjectionTestEnv> mock_env(
      new FaultInjectionTestEnv(env_));
  Options options = GetOptions();
  options.env = mock_env.get();
  options.writable_file_max_buffer_size = 4 * 1024 * 1024;
  options.write_buffer_size = 3 * 512 * 1024;
  options.wal_bytes_per_sync = 256 * 1024;
  options.manual_wal_flush = true;
  Reopen(options);
  mock_env->SetFilesystemActive(false, Status::IOError("Not active"));
  Status s;
  for (int i = 0; i < 4 * 512; ++i) {
    s = Put(Key(i), rnd.RandomString(1024));
    if (!s.ok()) {
      break;
    }
  }
  ASSERT_EQ(s.severity(), Status::Severity::kFatalError);

  mock_env->SetFilesystemActive(true);
  // Close before mock_env destruct.
  Close();
}

// Test that db->LockWAL() flushes the WAL after locking, which can fail
TEST_P(DBWriteTest, LockWALInEffect) {
  if (mem_env_ || encrypted_env_) {
    ROCKSDB_GTEST_SKIP("Test requires non-mem or non-encrypted environment");
    return;
  }
  Options options = GetOptions();
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  options.env = fault_fs_env.get();
  options.disable_auto_compactions = true;
  options.paranoid_checks = false;
  options.max_bgerror_resume_count = 0;  // manual Resume()
  Reopen(options);
  // try the 1st WAL created during open
  ASSERT_OK(Put("key0", "value"));
  ASSERT_NE(options.manual_wal_flush, dbfull()->WALBufferIsEmpty());

  ASSERT_OK(db_->LockWAL());

  ASSERT_TRUE(dbfull()->WALBufferIsEmpty());
  uint64_t wal_num = dbfull()->TEST_GetCurrentLogNumber();
  // Manual flush with wait=false should abruptly fail with TryAgain
  FlushOptions flush_opts;
  flush_opts.wait = false;
  for (bool allow_write_stall : {true, false}) {
    flush_opts.allow_write_stall = allow_write_stall;
    ASSERT_TRUE(db_->Flush(flush_opts).IsTryAgain());
  }
  ASSERT_EQ(wal_num, dbfull()->TEST_GetCurrentLogNumber());

  ASSERT_OK(db_->UnlockWAL());

  // try the 2nd wal created during SwitchWAL (not locked this time)
  ASSERT_OK(dbfull()->TEST_SwitchWAL());
  ASSERT_NE(wal_num, dbfull()->TEST_GetCurrentLogNumber());
  ASSERT_OK(Put("key1", "value"));
  ASSERT_NE(options.manual_wal_flush, dbfull()->WALBufferIsEmpty());
  ASSERT_OK(db_->LockWAL());
  ASSERT_TRUE(dbfull()->WALBufferIsEmpty());
  ASSERT_OK(db_->UnlockWAL());

  // The above `TEST_SwitchWAL()` triggered a flush. That flush needs to finish
  // before we make the filesystem inactive, otherwise the flush might hit an
  // unrecoverable error (e.g., failed MANIFEST update).
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(nullptr));

  // Fail the WAL flush if applicable
  fault_fs->SetFilesystemActive(false);
  Status s = Put("key2", "value");
  if (options.manual_wal_flush) {
    ASSERT_OK(s);
    // I/O failure
    ASSERT_NOK(db_->LockWAL());
    // Should not need UnlockWAL after LockWAL fails
  } else {
    ASSERT_NOK(s);
    ASSERT_OK(db_->LockWAL());
    ASSERT_OK(db_->UnlockWAL());
  }
  fault_fs->SetFilesystemActive(true);
  ASSERT_OK(db_->Resume());
  // Writes should work again
  ASSERT_OK(Put("key3", "value"));
  ASSERT_EQ(Get("key3"), "value");

  // Should be extraneous, but allowed
  ASSERT_NOK(db_->UnlockWAL());

  // Close before mock_env destruct.
  Close();
}

TEST_P(DBWriteTest, LockWALConcurrentRecursive) {
  // This is a micro-stress test of LockWAL and concurrency handling.
  // It is considered the most convenient way to balance functional
  // coverage and reproducibility (vs. the two extremes of (a) unit tests
  // tailored to specific interleavings and (b) db_stress)
  Options options = GetOptions();
  Reopen(options);
  ASSERT_OK(Put("k1", "k1_orig"));
  ASSERT_OK(db_->LockWAL());  // 0 -> 1
  auto frozen_seqno = db_->GetLatestSequenceNumber();

  std::string ingest_file = dbname_ + "/external.sst";
  {
    SstFileWriter sst_file_writer(EnvOptions(), options);
    ASSERT_OK(sst_file_writer.Open(ingest_file));
    ASSERT_OK(sst_file_writer.Put("k2", "k2_val"));
    ExternalSstFileInfo external_info;
    ASSERT_OK(sst_file_writer.Finish(&external_info));
  }
  Atomic<bool> parallel_ingest_completed{false};
  port::Thread parallel_ingest{[&]() {
    IngestExternalFileOptions ingest_opts;
    ingest_opts.move_files = true;  // faster than copy
    // Shouldn't finish until WAL unlocked
    ASSERT_OK(db_->IngestExternalFile({ingest_file}, ingest_opts));
    parallel_ingest_completed.Store(true);
  }};

  Atomic<bool> flush_completed{false};
  port::Thread parallel_flush{[&]() {
    FlushOptions flush_opts;
    // NB: Flush with wait=false case is tested above in LockWALInEffect
    flush_opts.wait = true;
    // allow_write_stall = true blocks in fewer cases
    flush_opts.allow_write_stall = true;
    // Shouldn't finish until WAL unlocked
    ASSERT_OK(db_->Flush(flush_opts));
    flush_completed.Store(true);
  }};

  Atomic<bool> parallel_put_completed{false};
  port::Thread parallel_put{[&]() {
    // This can make certain failure scenarios more likely:
    //   sleep(1);
    // Shouldn't finish until WAL unlocked
    ASSERT_OK(Put("k1", "k1_mod"));
    parallel_put_completed.Store(true);
  }};

  ASSERT_OK(db_->LockWAL());  // 1 -> 2
  // Read-only ops are OK
  ASSERT_EQ(Get("k1"), "k1_orig");
  {
    std::vector<LiveFileStorageInfo> files;
    LiveFilesStorageInfoOptions lf_opts;
    // A DB flush could deadlock
    lf_opts.wal_size_for_flush = UINT64_MAX;
    ASSERT_OK(db_->GetLiveFilesStorageInfo({lf_opts}, &files));
  }

  SyncPoint::GetInstance()->LoadDependency(
      {{"DBWriteTest::LockWALConcurrentRecursive:ParallelLockAcquired",
        "DBWriteTest::LockWALConcurrentRecursive:MainUnlocks"},
       {"DBWriteTest::LockWALConcurrentRecursive:MainAllowsParallelUnlock",
        "DBWriteTest::LockWALConcurrentRecursive:ParallelUnlocks"}});
  SyncPoint::GetInstance()->EnableProcessing();

  port::Thread parallel_lock_wal{[&]() {
    ASSERT_OK(db_->LockWAL());  // 2 -> 3
    TEST_SYNC_POINT(
        "DBWriteTest::LockWALConcurrentRecursive:ParallelLockAcquired");
    TEST_SYNC_POINT("DBWriteTest::LockWALConcurrentRecursive:ParallelUnlocks");
    ASSERT_OK(db_->UnlockWAL());
  }};
  TEST_SYNC_POINT("DBWriteTest::LockWALConcurrentRecursive:MainUnlocks");

  ASSERT_OK(db_->UnlockWAL());  // 3 -> 2
  // Give parallel_put an extra chance to jump in case of bug
  std::this_thread::yield();
  ASSERT_FALSE(parallel_put_completed.Load());
  ASSERT_FALSE(parallel_ingest_completed.Load());
  ASSERT_FALSE(flush_completed.Load());

  // Should still have 2 outstanding LockWAL().
  ASSERT_EQ(Get("k1"), "k1_orig");

  ASSERT_OK(db_->UnlockWAL());  // 2 -> 1

  ASSERT_FALSE(parallel_put_completed.Load());
  ASSERT_FALSE(parallel_ingest_completed.Load());
  ASSERT_FALSE(flush_completed.Load());

  ASSERT_EQ(Get("k1"), "k1_orig");
  ASSERT_EQ(Get("k2"), "NOT_FOUND");
  ASSERT_EQ(frozen_seqno, db_->GetLatestSequenceNumber());

  TEST_SYNC_POINT(
      "DBWriteTest::LockWALConcurrentRecursive:MainAllowsParallelUnlock");
  parallel_lock_wal.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->LoadDependency({});

  // Write can proceed
  parallel_put.join();
  ASSERT_TRUE(parallel_put_completed.Load());
  ASSERT_EQ(Get("k1"), "k1_mod");
  parallel_ingest.join();
  ASSERT_TRUE(parallel_ingest_completed.Load());
  ASSERT_EQ(Get("k2"), "k2_val");
  parallel_flush.join();
  ASSERT_TRUE(flush_completed.Load());
  // And new writes
  ASSERT_OK(Put("k3", "val"));
  ASSERT_EQ(Get("k3"), "val");
}

TEST_F(DBWriteTestUnparameterized, WalWriteMutexConfigurationMatrix) {
  for (const bool two_write_queues : {false, true}) {
    for (const bool manual_wal_flush : {false, true}) {
      Options options = CurrentOptions();
      options.two_write_queues = two_write_queues;
      options.manual_wal_flush = manual_wal_flush;
      Reopen(options);

      ASSERT_OK(Put("before_lock", "value"));
      ASSERT_EQ(!manual_wal_flush, dbfull()->WALBufferIsEmpty());
      ASSERT_OK(db_->LockWAL());
      ASSERT_TRUE(dbfull()->WALBufferIsEmpty());

      port::Thread concurrent_writer([&]() {
        WriteOptions write_options;
        write_options.no_slowdown = true;
        ASSERT_TRUE(
            db_->Put(write_options, "while_locked", "value").IsIncomplete());
      });
      concurrent_writer.join();

      ASSERT_OK(db_->UnlockWAL());
      ASSERT_OK(Put("after_unlock", "value"));
      ASSERT_EQ("value", Get("after_unlock"));
    }
  }
}

TEST_P(DBWriteTest, ConcurrentlyDisabledWAL) {
  Options options = GetOptions();
  options.statistics = ROCKSDB_NAMESPACE::CreateDBStatistics();
  options.statistics->set_stats_level(StatsLevel::kAll);
  Reopen(options);
  std::string wal_key_prefix = "WAL_KEY_";
  std::string no_wal_key_prefix = "K_";
  // 100 KB value each for NO-WAL operation
  std::string no_wal_value(1024 * 100, 'X');
  // 1B value each for WAL operation
  std::string wal_value = "0";
  std::thread threads[10];
  for (int t = 0; t < 10; t++) {
    threads[t] = std::thread([t, wal_key_prefix, wal_value, no_wal_key_prefix,
                              no_wal_value, &options, this] {
      for (int i = 0; i < 10; i++) {
        ROCKSDB_NAMESPACE::WriteOptions write_option_disable;
        write_option_disable.disableWAL = true;
        ROCKSDB_NAMESPACE::WriteOptions write_option_default;
        std::string no_wal_key =
            no_wal_key_prefix + std::to_string(t) + "_" + std::to_string(i);
        ASSERT_OK(this->Put(no_wal_key, no_wal_value, write_option_disable));
        std::string wal_key =
            wal_key_prefix + std::to_string(i) + "_" + std::to_string(i);
        ASSERT_OK(this->Put(wal_key, wal_value, write_option_default));
        ASSERT_OK(dbfull()->SyncWAL())
            << "options.env: " << options.env << ", env_: " << env_
            << ", env_->is_wal_sync_thread_safe_: "
            << env_->is_wal_sync_thread_safe_.load();
      }
      return;
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  uint64_t bytes_num = options.statistics->getTickerCount(
      ROCKSDB_NAMESPACE::Tickers::WAL_FILE_BYTES);
  // written WAL size should less than 100KB (even included HEADER & FOOTER
  // overhead)
  ASSERT_LE(bytes_num, 1024 * 100);
}

void CorruptLogFile(Env* env, Options& options, std::string log_path,
                    uint64_t log_num, int record_num) {
  std::shared_ptr<FileSystem> fs = env->GetFileSystem();
  std::unique_ptr<SequentialFileReader> file_reader;
  Status status;
  {
    std::unique_ptr<FSSequentialFile> file;
    status = fs->NewSequentialFile(log_path, FileOptions(), &file, nullptr);
    ASSERT_EQ(status, IOStatus::OK());
    file_reader.reset(new SequentialFileReader(std::move(file), log_path));
  }
  std::unique_ptr<log::Reader> reader(new log::Reader(
      nullptr, std::move(file_reader), nullptr, false, log_num));
  std::string scratch;
  Slice record;
  uint64_t record_checksum;
  for (int i = 0; i < record_num; ++i) {
    ASSERT_TRUE(reader->ReadRecord(&record, &scratch, options.wal_recovery_mode,
                                   &record_checksum));
  }
  uint64_t rec_start = reader->LastRecordOffset();
  reader.reset();
  {
    std::unique_ptr<FSRandomRWFile> file;
    status = fs->NewRandomRWFile(log_path, FileOptions(), &file, nullptr);
    ASSERT_EQ(status, IOStatus::OK());
    uint32_t bad_lognum = 0xff;
    ASSERT_EQ(file->Write(
                  rec_start + 7,
                  Slice(reinterpret_cast<char*>(&bad_lognum), sizeof(uint32_t)),
                  IOOptions(), nullptr),
              IOStatus::OK());
    ASSERT_OK(file->Close(IOOptions(), nullptr));
    file.reset();
  }
}

TEST_P(DBWriteTest, RecycleLogTest) {
  Options options = GetOptions();
  options.recycle_log_file_num = 0;
  options.avoid_flush_during_recovery = true;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;

  Reopen(options);
  ASSERT_OK(Put(Key(1), "val1"));
  ASSERT_OK(Put(Key(2), "val1"));

  uint64_t latest_log_num = 0;
  std::unique_ptr<LogFile> log_file;
  ASSERT_OK(dbfull()->GetCurrentWalFile(&log_file));
  latest_log_num = log_file->LogNumber();
  Reopen(options);
  ASSERT_OK(Put(Key(3), "val3"));

  // Corrupt second entry of first log
  std::string log_path = LogFileName(dbname_, latest_log_num);
  CorruptLogFile(env_, options, log_path, latest_log_num, 2);

  Reopen(options);
  ASSERT_EQ(Get(Key(1)), "val1");
  ASSERT_EQ(Get(Key(2)), "NOT_FOUND");
  ASSERT_EQ(Get(Key(3)), "NOT_FOUND");
}

TEST_P(DBWriteTest, RecycleLogTestCFAheadOfWAL) {
  Options options = GetOptions();
  options.recycle_log_file_num = 0;
  options.avoid_flush_during_recovery = true;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;

  CreateAndReopenWithCF({"pikachu"}, options);
  ASSERT_OK(Put(1, Key(1), "val1"));
  ASSERT_OK(Put(0, Key(2), "val2"));

  uint64_t latest_log_num = 0;
  std::unique_ptr<LogFile> log_file;
  ASSERT_OK(dbfull()->GetCurrentWalFile(&log_file));
  latest_log_num = log_file->LogNumber();
  ASSERT_OK(Flush(1));
  ASSERT_OK(Put(1, Key(3), "val3"));

  // Corrupt second entry of first log
  std::string log_path = LogFileName(dbname_, latest_log_num);
  CorruptLogFile(env_, options, log_path, latest_log_num, 2);

  ASSERT_EQ(TryReopenWithColumnFamilies({"default", "pikachu"}, options),
            Status::Corruption());
}

TEST_P(DBWriteTest, RecycleLogToggleTest) {
  Options options = GetOptions();
  options.recycle_log_file_num = 0;
  options.avoid_flush_during_recovery = true;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;

  Destroy(options);
  Reopen(options);
  // After opening, a new log gets created, say 1.log
  ASSERT_OK(Put(Key(1), "val1"));

  options.recycle_log_file_num = 1;
  Reopen(options);
  // 1.log is added to alive_wal_files_
  ASSERT_OK(Put(Key(2), "val1"));
  ASSERT_OK(Flush());
  // 1.log should be deleted and not recycled, since it
  // was created by the previous Reopen
  ASSERT_OK(Put(Key(1), "val2"));
  ASSERT_OK(Flush());

  options.recycle_log_file_num = 1;
  Reopen(options);
  ASSERT_EQ(Get(Key(1)), "val2");
}

TEST_P(DBWriteTest, IngestWriteBatchWithIndex) {
  if (GetParam() == kPipelinedWrite) {
    return;
  }

  Options options = GetOptions();
  options.disable_auto_compactions = true;
  Reopen(options);
  Options cf_options = GetOptions();
  cf_options.merge_operator = MergeOperators::CreateStringAppendOperator();
  CreateColumnFamilies({"cf1", "cf2"}, cf_options);
  ReopenWithColumnFamilies({"default", "cf1", "cf2"},
                           {options, cf_options, cf_options});

  // default cf
  auto wbwi1 = std::make_shared<WriteBatchWithIndex>(options.comparator, 0,
                                                     /*overwrite_key=*/true);
  ASSERT_OK(wbwi1->Put("key1", "value1"));
  ASSERT_OK(wbwi1->Put("key2", "value2"));
  if (GetParam() == kPipelinedWrite) {
    ASSERT_TRUE(db_->IngestWriteBatchWithIndex({}, wbwi1).IsNotSupported());
    return;
  }
  // Test disableWAL=false
  ASSERT_TRUE(db_->IngestWriteBatchWithIndex({}, wbwi1).IsNotSupported());

  WriteOptions wo;
  wo.disableWAL = true;
  ASSERT_OK(db_->IngestWriteBatchWithIndex(wo, wbwi1));
  ASSERT_EQ("value1", Get("key1"));
  ASSERT_EQ("value2", Get("key2"));

  // Test with overwrites
  auto wbwi = std::make_shared<WriteBatchWithIndex>(options.comparator, 0,
                                                    /*overwrite_key=*/true);
  ASSERT_OK(wbwi->Put("key2", "value3"));
  ASSERT_OK(wbwi->Delete("key1"));  // Delete an existing key
  ASSERT_OK(db_->IngestWriteBatchWithIndex(wo, wbwi));
  ASSERT_EQ("NOT_FOUND", Get("key1"));
  ASSERT_EQ("value3", Get("key2"));

  auto wbwi2 = std::make_shared<WriteBatchWithIndex>(options.comparator, 0,
                                                     /*overwrite_key=*/true);
  ASSERT_OK(wbwi2->Put(handles_[1], "cf1_key1", "cf1_value1"));
  ASSERT_OK(wbwi2->Delete(handles_[1], "cf1_key2"));
  // Test ingestion with column family
  ASSERT_OK(db_->IngestWriteBatchWithIndex(wo, wbwi2));
  ASSERT_EQ("cf1_value1", Get(1, "cf1_key1"));
  ASSERT_EQ("NOT_FOUND", Get(1, "cf1_key2"));

  auto wbwi3 = std::make_shared<WriteBatchWithIndex>(options.comparator, 0,
                                                     /*overwrite_key=*/true);
  ASSERT_OK(wbwi3->Merge(handles_[2], "cf2_key1", "cf2_value1"));
  ASSERT_OK(wbwi3->Merge(handles_[2], "cf2_key1", "cf2_value2"));
  // Test ingestion with merge operations
  ASSERT_OK(db_->IngestWriteBatchWithIndex(wo, wbwi3));
  ASSERT_EQ("cf2_value1,cf2_value2", Get(2, "cf2_key1"));

  // Test with overwrite_key = false
  auto wbwi_no_overwrite = std::make_shared<WriteBatchWithIndex>(
      options.comparator, 0, /*overwrite_key=*/false);
  ASSERT_OK(wbwi_no_overwrite->Put("key1", "value1"));
  Status s = db_->IngestWriteBatchWithIndex(wo, wbwi_no_overwrite);
  ASSERT_TRUE(s.IsNotSupported());

  auto empty_wbwi = std::make_shared<WriteBatchWithIndex>(
      options.comparator, 0, /*overwrite_key=*/true);
  ASSERT_OK(db_->IngestWriteBatchWithIndex(wo, empty_wbwi));

  DestroyAndReopen(options);
  // Should fail when trying to ingest to non-existent column family
  ASSERT_NOK(db_->IngestWriteBatchWithIndex(wo, wbwi2));
}

TEST_P(DBWriteTest, CompletionGateNeverInstallsAtOnePartition) {
  // R6: the single-WAL path never installs the gate -- no registration, no
  // frontier wait -- across sync, async, and disableWAL writes.
  Open();
  ASSERT_EQ(1U, dbfull()->TEST_NumWalPartitions());
  std::atomic<int> installs{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::InstallCompletionGate",
                                        [&](void*) { ++installs; });
  SyncPoint::GetInstance()->EnableProcessing();

  ASSERT_OK(Put("k1", "v1"));
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("k2", "v2", sync_wo));
  WriteOptions no_wal;
  no_wal.disableWAL = true;
  ASSERT_OK(Put("k3", "v3", no_wal));

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  EXPECT_EQ(0, installs.load());
}

TEST_F(DBWriteTestUnparameterized, WalPartitionHistogramsSilentAtN1) {
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.statistics->set_stats_level(StatsLevel::kAll);
  Reopen(options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("k1", "v1"));
  ASSERT_OK(Put("k2", "v2", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Flush());
  for (Histograms h :
       {WAL_DURABILITY_WAIT_MICROS, WAL_PARTITION_COMPLETION_MICROS,
        WAL_SYNC_FSYNCED_PARTITIONS, WAL_SYNC_UNSYNCED_PARTITIONS,
        WAL_SYNC_BARRIER_MICROS, WAL_PARTITION_INGESTION_STALL_MICROS,
        WAL_WRITE_MUTEX_WAIT_MICROS, WAL_WRITE_MUTEX_ALL_WAIT_MICROS}) {
    HistogramData data;
    options.statistics->histogramData(h, &data);
    EXPECT_EQ(0U, data.count) << h;
  }
}

TEST_F(DBWriteTestUnparameterized, IndexedWalPartitionsOpenFromOptionsFile) {
  // Uses no sync point: indexed wal_partitions > 1 opens on its own, and so
  // does a reopen from the OPTIONS file it writes, which is how tools such as
  // ldb open a DB.
  constexpr uint32_t kPartitions = 4;
  Options options = CurrentOptions();
  options.wal_partitions = kPartitions;
  options.track_and_verify_wals_in_manifest = true;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_shutdown = true;
  options.avoid_flush_during_recovery = true;
  DestroyDB(dbname_, options).PermitUncheckedError();
  ASSERT_OK(TryReopen(options));
  std::vector<std::string> children;
  ASSERT_OK(env_->GetChildren(dbname_, &children));
  size_t num_wals = 0;
  for (const std::string& child : children) {
    uint64_t number = 0;
    FileType type;
    if (ParseFileName(child, &number, &type) && type == kWalFile) {
      ++num_wals;
    }
  }
  ASSERT_EQ(kPartitions, num_wals);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("k1", "v1", sync_wo));
  ASSERT_OK(Put("k2", "v2"));
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ("v1", Get("k1"));
  ASSERT_EQ("v2", Get("k2"));

  DBOptions loaded_db_options;
  std::vector<ColumnFamilyDescriptor> loaded_cfs;
  ConfigOptions config_options;
  config_options.env = env_;
  ASSERT_OK(LoadLatestOptions(config_options, dbname_, &loaded_db_options,
                              &loaded_cfs));
  ASSERT_EQ(kPartitions, loaded_db_options.wal_partitions);
  ASSERT_EQ(PartitionWALUsage::kPartitionByColumnFamily,
            loaded_db_options.partition_wal_usage);
  Close();
  ASSERT_OK(TryReopen(Options(loaded_db_options, loaded_cfs[0].options)));
  ASSERT_EQ(kPartitions, dbfull()->TEST_NumWalPartitions());
  ASSERT_EQ("v1", Get("k1"));
  ASSERT_EQ("v2", Get("k2"));
}

// Unindexed WALs cannot be merged at recovery, so kNone with N > 1 is rejected,
// naming both options, while an indexing usage opens.
TEST_F(DBWriteTestUnparameterized, UnindexedWalPartitionsAlwaysRejected) {
  for (const uint32_t partitions : {2U, 4U}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.wal_partitions = partitions;
    options.partition_wal_usage = PartitionWALUsage::kNone;
    options.track_and_verify_wals_in_manifest = true;
    Destroy(options);
    const Status s = TryReopen(options);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
    EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
    EXPECT_NE(nullptr, std::strstr(s.getState(), "partition_wal_usage"));
  }
  Options options = CurrentOptions();
  options.wal_partitions = 4;
  options.track_and_verify_wals_in_manifest = true;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_shutdown = true;
  Destroy(options);
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ(4U, dbfull()->TEST_NumWalPartitions());
  ASSERT_OK(Put("key", "value"));
  ASSERT_EQ("value", Get("key"));
}

// wal_partitions above kMaxWalPartitions is rejected; kMaxWalPartitions itself
// opens.
TEST_F(DBWriteTestUnparameterized, WalPartitionsUpperBound) {
  Options options = CurrentOptions();
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_shutdown = true;
  options.wal_partitions = DBImpl::kMaxWalPartitions + 1;
  options.track_and_verify_wals_in_manifest = true;
  Destroy(options);
  const Status s = TryReopen(options);
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
  EXPECT_NE(nullptr,
            std::strstr(s.getState(),
                        std::to_string(DBImpl::kMaxWalPartitions).c_str()));
  options.wal_partitions = DBImpl::kMaxWalPartitions;
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ(DBImpl::kMaxWalPartitions, dbfull()->TEST_NumWalPartitions());
  ASSERT_OK(Put("key", "value"));
  ASSERT_EQ("value", Get("key"));
}

// Every rejection of wal_partitions > 1: each names
// wal_partitions and its conflicting option, and the two M1.5 cells (kNone,
// and two_write_queues, which rejects indexing) also name partition_wal_usage.
// The runtime rejections follow on a DB that opens.
TEST_F(DBWriteTestUnparameterized, WalPartitionsRejectionsNameTheirOptions) {
  struct Case {
    const char* name;
    std::function<void(Options*)> tweak;
    Status::Code code;
    std::vector<const char*> names;
  };
  const std::vector<Case> cases = {
      {"kNone",
       [](Options* o) { o->partition_wal_usage = PartitionWALUsage::kNone; },
       Status::kNotSupported,
       {"partition_wal_usage"}},
      {"two_write_queues",
       [](Options* o) { o->two_write_queues = true; },
       Status::kNotSupported,
       {"two_write_queues", "partition_wal_usage"}},
      {"allow_concurrent_memtable_write",
       [](Options* o) { o->allow_concurrent_memtable_write = false; },
       Status::kNotSupported,
       {"allow_concurrent_memtable_write"}},
      {"enable_pipelined_write",
       [](Options* o) { o->enable_pipelined_write = true; },
       Status::kNotSupported,
       {"enable_pipelined_write"}},
      {"track_and_verify_wals",
       [](Options* o) { o->track_and_verify_wals = true; },
       Status::kNotSupported,
       {"track_and_verify_wals"}},
      {"unordered_write",
       [](Options* o) { o->unordered_write = true; },
       Status::kNotSupported,
       {"unordered_write"}},
      {"recycle_log_file_num",
       [](Options* o) { o->recycle_log_file_num = 1; },
       Status::kNotSupported,
       {"recycle_log_file_num"}},
      // SanitizeOptions would zero recycle_log_file_num in these two.
      {"recycle_log_file_num with WAL_ttl_seconds",
       [](Options* o) {
         o->recycle_log_file_num = 1;
         o->WAL_ttl_seconds = 60;
       },
       Status::kNotSupported,
       {"recycle_log_file_num"}},
      {"recycle_log_file_num with kAbsoluteConsistency",
       [](Options* o) {
         o->recycle_log_file_num = 1;
         o->wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
       },
       Status::kNotSupported,
       {"recycle_log_file_num"}},
      {"max_successive_merges",
       [](Options* o) { o->max_successive_merges = 10; },
       Status::kInvalidArgument,
       {"max_successive_merges"}},
      {"memtable_factory",
       [](Options* o) { o->memtable_factory.reset(new VectorRepFactory()); },
       Status::kNotSupported,
       {"SkipList"}},
      {"inplace_update_support",
       [](Options* o) { o->inplace_update_support = true; },
       Status::kNotSupported,
       {"inplace_update_support"}},
      {"enable_blob_direct_write",
       [](Options* o) {
         o->enable_blob_files = true;
         o->enable_blob_direct_write = true;
       },
       Status::kNotSupported,
       {"enable_blob_direct_write"}},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Options options = CurrentOptions();
    options.wal_partitions = 2;
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    c.tweak(&options);
    Destroy(options);
    const Status s = TryReopen(options);
    EXPECT_EQ(c.code, s.code()) << s.ToString();
    if (s.ok()) {
      continue;
    }
    EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"))
        << s.ToString();
    for (const char* name : c.names) {
      EXPECT_NE(nullptr, std::strstr(s.getState(), name)) << s.ToString();
    }
  }

  Options options = CurrentOptions();
  options.wal_partitions = 2;
  options.track_and_verify_wals_in_manifest = true;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_shutdown = true;
  Destroy(options);
  ASSERT_OK(TryReopen(options));
  class AcceptingWriteCallback : public WriteCallback {
   public:
    Status Callback(DB* /*db*/) override { return Status::OK(); }
    bool AllowWriteBatching() override { return true; }
  };
  WriteBatch batch;
  ASSERT_OK(batch.Put("callback_key", "value"));
  AcceptingWriteCallback callback;
  Status s = dbfull()->WriteWithCallback(WriteOptions(), &batch, &callback);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
  EXPECT_EQ("NOT_FOUND", Get("callback_key"));
  // A UserWriteCallback validates nothing and is not rejected.
  class CountingUserWriteCallback : public UserWriteCallback {
   public:
    void OnWriteEnqueued() override { ++enqueued; }
    void OnWalWriteFinish() override {}
    int enqueued = 0;
  };
  CountingUserWriteCallback user_write_cb;
  ASSERT_OK(
      dbfull()->WriteWithCallback(WriteOptions(), &batch, &user_write_cb));
  EXPECT_EQ(1, user_write_cb.enqueued);
  EXPECT_EQ("value", Get("callback_key"));

  auto wbwi = std::make_shared<WriteBatchWithIndex>(options.comparator, 0,
                                                    /*overwrite_key=*/true);
  ASSERT_OK(wbwi->Put("ingest_key", "value"));
  WriteOptions disable_wal;
  disable_wal.disableWAL = true;
  s = db_->IngestWriteBatchWithIndex(disable_wal, wbwi);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
}

// Opens with more than one WAL partition, which Open() rejects outside tests,
// and with a column family routed to every partition. The parameter is
// two_write_queues, which rejects the WAL indexing N > 1 needs, so tests that
// open N > 1 bypass the true instance.
class DBWritePartitionTest : public DBTestBase,
                             public testing::WithParamInterface<bool> {
 public:
  DBWritePartitionTest()
      : DBTestBase("db_write_partition_test", /*env_do_fsync=*/false) {}

  void OpenWithPartitions(size_t num_partitions, Options options) {
    SyncPoint::GetInstance()->EnableProcessing();

    // Column family ids are assigned in creation order, so cf_id is also the
    // handle index once the families are created.
    std::vector<std::string> cf_names;
    cf_for_partition_.assign(num_partitions, 0);
    std::vector<bool> covered(num_partitions, false);
    size_t num_covered = 0;
    for (uint32_t cf_id = 1; num_covered < num_partitions; ++cf_id) {
      WriteBatch batch;
      ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
      size_t partition = num_partitions;
      ASSERT_OK(
          DBImpl::TEST_WalPartitionForWrite(batch, num_partitions, &partition));
      cf_names.push_back("cf" + std::to_string(cf_id));
      if (!covered[partition]) {
        covered[partition] = true;
        cf_for_partition_[partition] = static_cast<int>(cf_id);
        ++num_covered;
      }
    }
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    if (num_partitions > 1) {
      options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
      options.track_and_verify_wals_in_manifest = true;
    }
    options.two_write_queues = GetParam();
    CreateAndReopenWithCF(cf_names, options);
    ASSERT_EQ(num_partitions, dbfull()->TEST_NumWalPartitions());
  }

  // A write with no WAL record. It still inserts the memtable (so it can
  // fill, schedule, and switch) but never waits on the durability frontier.
  Status DisableWalPut(int cf, const std::string& key,
                       const std::string& value) {
    WriteBatch batch;
    Status s = batch.Put(handles_[cf], key, value);
    if (!s.ok()) {
      return s;
    }
    WriteOptions write_options;
    write_options.disableWAL = true;
    return db_->Write(write_options, &batch);
  }

  // Fills `cf`'s memtable with disableWAL puts, and keeps putting until
  // `switching()` reports a memtable switch under way. Each put runs on its
  // own thread: above a held lower write a put parks in its completion gate
  // until that write is published, but only after its memtable insert, which
  // is what fills the memtable and schedules the flush; the put that finds
  // the flush scheduled switches before it allocates. The caller joins the
  // returned threads once the held write is released.
  std::vector<port::Thread> FillAndSwitchMemtable(
      int cf, const std::function<bool()>& switching) {
    std::vector<port::Thread> puts;
    const size_t parked_before = dbfull()->TEST_WalDurabilityWaiters();
    puts.emplace_back([this, cf] {
      ASSERT_OK(DisableWalPut(cf, "fill", std::string(128 << 10, 'x')));
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (int i = 0; i < 3; ++i) {
      while (!switching() &&
             dbfull()->TEST_WalDurabilityWaiters() <
                 parked_before + puts.size() &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (switching()) {
        break;
      }
      puts.emplace_back(
          [this, cf] { ASSERT_OK(DisableWalPut(cf, "switch", "value")); });
    }
    return puts;
  }

  // A write with no memtable insert, which goes through the nonmem queue
  // when two_write_queues is set.
  Status WalOnlyPut(int cf, const std::string& key, const std::string& value) {
    WriteBatch batch;
    Status s = batch.Put(handles_[cf], key, value);
    if (!s.ok()) {
      return s;
    }
    return dbfull()->WriteImpl(WriteOptions(), &batch, /*callback=*/nullptr,
                               /*user_write_cb=*/nullptr, /*wal_used=*/nullptr,
                               /*log_ref=*/0, /*disable_memtable=*/true);
  }

  // WriteWithCallback rejects a callback at N > 1; internal writers pass one
  // to WriteImpl directly.
  Status WriteImplWithCallback(WriteBatch* batch, WriteCallback* callback) {
    return dbfull()->WriteImpl(WriteOptions(), batch, callback,
                               /*user_write_cb=*/nullptr);
  }

  // Runs one write per partition on its own thread, each holding its WAL
  // append open until every partition is inside one, and returns the most
  // appends seen in flight at once. Appends that serialize never get past
  // one, and each waiter gives up at a shared deadline.
  size_t MaxConcurrentAppends(
      const std::function<Status(size_t partition)>& write) {
    const size_t num_partitions = cf_for_partition_.size();
    std::mutex mu;
    std::condition_variable cv;
    size_t in_append = 0;
    size_t max_in_append = 0;
    bool all_in = false;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          max_in_append = std::max(max_in_append, ++in_append);
          if (in_append == num_partitions) {
            all_in = true;
            cv.notify_all();
          }
          cv.wait_until(lock, deadline, [&] { return all_in; });
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          --in_append;
        });

    std::vector<port::Thread> writers;
    for (size_t p = 0; p < num_partitions; ++p) {
      writers.emplace_back([&write, p] { ASSERT_OK(write(p)); });
    }
    for (auto& writer : writers) {
      writer.join();
    }
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord");
    return max_in_append;
  }

  // Waits until exactly `expected` groups park in their completion
  // gates. Returns false on timeout instead of failing, so callers can
  // ASSERT_TRUE it next to the injection the parked state gates.
  bool WaitForDurabilityWaiters(size_t expected) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (dbfull()->TEST_WalDurabilityWaiters() != expected) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }

  // Options for tests that reopen with N > 1: only indexed WALs recover as
  // more than one file, and the WALs stay behind for recovery instead of
  // being flushed away at shutdown or open.
  static Options IndexedWalOptions(Options options) {
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    options.avoid_flush_during_shutdown = true;
    options.avoid_flush_during_recovery = true;
    options.track_and_verify_wals_in_manifest = true;
    // Room for the immutable memtables tests leave behind by rotating with
    // TEST_SwitchMemtable, which schedules no flush.
    options.max_write_buffer_number = 8;
    return options;
  }

  // Reopens the column families OpenWithPartitions created, with
  // `num_partitions` WAL partitions, and recomputes cf_for_partition_ for it.
  void ReopenWithPartitions(size_t num_partitions, Options options) {
    SyncPoint::GetInstance()->EnableProcessing();
    // Listed in id order, which is also handle order.
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    if (num_partitions > 1) {
      options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
      options.track_and_verify_wals_in_manifest = true;
    }
    options.two_write_queues = GetParam();
    ReopenWithColumnFamilies(cf_names, options);
    ASSERT_EQ(num_partitions, dbfull()->TEST_NumWalPartitions());
    cf_for_partition_.assign(num_partitions, 0);
    for (uint32_t cf_id = static_cast<uint32_t>(cf_names.size()) - 1;
         cf_id >= 1; --cf_id) {
      WriteBatch batch;
      ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
      size_t partition = num_partitions;
      ASSERT_OK(
          DBImpl::TEST_WalPartitionForWrite(batch, num_partitions, &partition));
      cf_for_partition_[partition] = static_cast<int>(cf_id);
    }
  }

  // Every partition's alive WAL numbers, in deque order.
  std::vector<std::vector<uint64_t>> AliveWalNumbers() {
    std::vector<std::vector<uint64_t>> numbers(
        dbfull()->TEST_NumWalPartitions());
    for (size_t p = 0; p < numbers.size(); ++p) {
      for (const auto& file : dbfull()->TEST_AliveWalFiles(p)) {
        numbers[p].push_back(file.first);
      }
    }
    return numbers;
  }

  // Invariant 3 and 4: each partition's deque is ascending and ends with the
  // WAL its writer appends to.
  void AssertPartitionDequesConsistent() {
    const std::vector<std::vector<uint64_t>> alive = AliveWalNumbers();
    for (size_t p = 0; p < alive.size(); ++p) {
      ASSERT_FALSE(alive[p].empty());
      ASSERT_TRUE(std::is_sorted(alive[p].begin(), alive[p].end()));
      ASSERT_EQ(dbfull()->TEST_CurrentWalNumber(p), alive[p].back());
    }
  }

  std::vector<uint64_t> WalNumbersOnDisk() {
    std::vector<std::string> children;
    EXPECT_OK(env_->GetChildren(dbname_, &children));
    std::vector<uint64_t> numbers;
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (ParseFileName(child, &number, &type) && type == kWalFile) {
        numbers.push_back(number);
      }
    }
    std::sort(numbers.begin(), numbers.end());
    return numbers;
  }

  // Opens the DB copy at `dir` (a checkpoint or a restored backup) with this
  // DB's column families and partition count, and returns the value of each
  // {partition, key} read from that partition's column family, or
  // "NOT_FOUND".
  std::vector<std::string> ReadFromCopy(
      const std::string& dir, Options options,
      const std::vector<std::pair<size_t, std::string>>& keys) {
    std::vector<std::string> cf_names;
    EXPECT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.create_if_missing = false;
    options.wal_partitions = static_cast<uint32_t>(cf_for_partition_.size());
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    const Status s =
        DB::Open(DBOptions(options), dir, column_families, &handles, &db);
    EXPECT_OK(s);
    std::vector<std::string> values;
    if (!s.ok()) {
      return values;
    }
    for (const auto& [partition, key] : keys) {
      std::string value;
      const Status get = db->Get(
          ReadOptions(), handles[cf_for_partition_[partition]], key, &value);
      EXPECT_TRUE(get.ok() || get.IsNotFound()) << get.ToString();
      values.push_back(get.ok() ? value : "NOT_FOUND");
    }
    for (ColumnFamilyHandle* handle : handles) {
      EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    return values;
  }

  // Opens the DB at `dir`, holding the column families OpenWithPartitions
  // created, as a WritePrepared TransactionDB with one write queue and
  // `num_partitions` WAL partitions.
  std::unique_ptr<TransactionDB> OpenWritePreparedTxnDB(
      const std::string& dir, size_t num_partitions, Options options,
      std::vector<ColumnFamilyHandle*>* handles) {
    std::vector<std::string> cf_names;
    EXPECT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.two_write_queues = false;
    TransactionDBOptions txn_db_options;
    txn_db_options.write_policy = TxnDBWritePolicy::WRITE_PREPARED;
    TransactionDB* txn_db = nullptr;
    EXPECT_OK(TransactionDB::Open(options, txn_db_options, dir, column_families,
                                  handles, &txn_db));
    return std::unique_ptr<TransactionDB>(txn_db);
  }

  // Copies every file of the open DB but its LOCK to `dir`: with every write
  // synced, what a crash at this point leaves behind.
  void CopyDBAsCrashImage(const std::string& dir) {
    ASSERT_OK(DestroyDir(env_, dir));
    ASSERT_OK(env_->CreateDirIfMissing(dir));
    std::vector<std::string> children;
    ASSERT_OK(env_->GetChildren(dbname_, &children));
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (!ParseFileName(child, &number, &type) || type == kDBLockFile) {
        continue;
      }
      CopyFile(dbname_ + "/" + child, dir + "/" + child);
    }
  }

  // GC26 Q4 (D-10): opens a fresh copy of the closed DB at `from` with
  // kSkipAnyCorruptedRecords, at options.wal_partitions, and checks that each
  // (handle index, key) of `present` reads back and each of `absent` does
  // not. `from` itself is left as it was, for the other modes.
  void AssertSkipOpensCopy(
      const std::string& from, Options options,
      const std::vector<std::pair<int, std::string>>& present,
      const std::vector<std::pair<int, std::string>>& absent) {
    const std::string dir = from + "_skip";
    ASSERT_OK(DestroyDir(env_, dir));
    ASSERT_OK(env_->CreateDirIfMissing(dir));
    std::vector<std::string> children;
    ASSERT_OK(env_->GetChildren(from, &children));
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (ParseFileName(child, &number, &type) && type != kDBLockFile) {
        CopyFile(from + "/" + child, dir + "/" + child);
      }
    }
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.create_if_missing = false;
    options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    ASSERT_OK(
        DB::Open(DBOptions(options), dir, column_families, &handles, &db));
    for (const auto& [cf, key] : present) {
      std::string value;
      EXPECT_OK(db->Get(ReadOptions(), handles[cf], key, &value)) << key;
    }
    for (const auto& [cf, key] : absent) {
      std::string value;
      const Status s = db->Get(ReadOptions(), handles[cf], key, &value);
      EXPECT_TRUE(s.IsNotFound()) << key << ": " << s.ToString();
    }
    for (ColumnFamilyHandle* handle : handles) {
      EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
    }
  }

  // Creates column families, named as OpenWithPartitions names them, until
  // one routes to `partition`, and returns its handle index.
  int CreateColumnFamilyOnPartition(size_t partition, const Options& options) {
    const size_t num_partitions = cf_for_partition_.size();
    while (true) {
      const uint32_t cf_id = static_cast<uint32_t>(handles_.size());
      CreateColumnFamilies({"cf" + std::to_string(cf_id)}, options);
      WriteBatch batch;
      EXPECT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
      size_t routed = num_partitions;
      EXPECT_OK(
          DBImpl::TEST_WalPartitionForWrite(batch, num_partitions, &routed));
      if (routed == partition) {
        return static_cast<int>(cf_id);
      }
    }
  }

  void FlushAllColumnFamilies() {
    for (size_t cf = 0; cf < handles_.size(); ++cf) {
      ASSERT_OK(Flush(static_cast<int>(cf)));
    }
  }

  // Puts key0 .. key<num_keys - 1> alternating between partitions 0 and 1 of
  // a two-partition DB, so wal_index i + 1 holds key i, and returns the size
  // of partition 1's WAL after each of its records.
  std::vector<uint64_t> PutAlternatingKeys(int num_keys,
                                           const std::string& value) {
    std::vector<uint64_t> partition1_record_ends;
    const std::string wal =
        LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(1));
    for (int i = 0; i < num_keys; ++i) {
      EXPECT_OK(
          Put(cf_for_partition_[i % 2], "key" + std::to_string(i), value));
      if (i % 2 == 1) {
        uint64_t size = 0;
        EXPECT_OK(env_->GetFileSize(wal, &size));
        partition1_record_ends.push_back(size);
      }
    }
    return partition1_record_ends;
  }

  // The first wal_index `declaring_wal` declares superseded in `cut_wal`.
  uint64_t FirstSupersededWALIndex(uint64_t declaring_wal, uint64_t cut_wal) {
    std::unique_ptr<SequentialFileReader> file_reader;
    EXPECT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), LogFileName(dbname_, declaring_wal),
        FileOptions(), &file_reader, /*dbg=*/nullptr,
        /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, declaring_wal);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
    }
    return reader.GetFirstSupersededWALIndex(cut_wal);
  }

  // Writes "r<round>_p<partition>" to every partition per round, partitions
  // in ascending order (descending when `descending`), and rotates every
  // partition between rounds without flushing, so every round's WALs stay
  // live. Returns each partition's alive WAL numbers, oldest first.
  std::vector<std::vector<uint64_t>> WriteRounds(int rounds, bool descending) {
    const size_t num_partitions = cf_for_partition_.size();
    ColumnFamilyData* rotating_cfd =
        static_cast_with_check<ColumnFamilyHandleImpl>(
            handles_[cf_for_partition_[0]])
            ->cfd();
    for (int round = 0; round < rounds; ++round) {
      if (round > 0) {
        EXPECT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
      }
      for (size_t i = 0; i < num_partitions; ++i) {
        const size_t p = descending ? num_partitions - 1 - i : i;
        EXPECT_OK(Put(cf_for_partition_[p], RoundKey(round, p), "value"));
      }
    }
    return AliveWalNumbers();
  }
  static std::string RoundKey(int round, size_t partition) {
    return "r" + std::to_string(round) + "_p" + std::to_string(partition);
  }

  // `wal`, read to its end, for its WAL index metadata.
  std::unique_ptr<log::Reader> ReadWALThrough(uint64_t wal) {
    std::unique_ptr<SequentialFileReader> file_reader;
    EXPECT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), LogFileName(dbname_, wal), FileOptions(),
        &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    std::unique_ptr<log::Reader> reader(
        new log::Reader(/*info_log=*/nullptr, std::move(file_reader),
                        /*reporter=*/nullptr, /*checksum=*/true, wal));
    std::string scratch;
    Slice record;
    while (reader->ReadRecord(&record, &scratch)) {
    }
    return reader;
  }

  // Handle index of a column family routed to each partition.
  std::vector<int> cf_for_partition_;
};

// Ignores the records found at the given positions, counting from zero.
class IgnoreFoundRecords : public WalFilter {
 public:
  explicit IgnoreFoundRecords(std::vector<int> ignored)
      : ignored_(std::move(ignored)) {}
  const char* Name() const override { return "IgnoreFoundRecords"; }
  WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                     const std::string& /*log_file_name*/,
                                     const WriteBatch& /*batch*/,
                                     WriteBatch* /*new_batch*/,
                                     bool* /*batch_changed*/) override {
    const int found = records_found_++;
    return std::find(ignored_.begin(), ignored_.end(), found) != ignored_.end()
               ? WalProcessingOption::kIgnoreCurrentRecord
               : WalProcessingOption::kContinueProcessing;
  }

 private:
  const std::vector<int> ignored_;
  int records_found_ = 0;
};

TEST_P(DBWritePartitionTest, MemtableDecisionEnforcedAtOpen) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  Options options = CurrentOptions();
  options.wal_partitions = 4;
  options.track_and_verify_wals_in_manifest = true;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.allow_concurrent_memtable_write = false;
  Status s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported());
  ASSERT_NE(nullptr,
            std::strstr(s.getState(), "allow_concurrent_memtable_write"));

  options.allow_concurrent_memtable_write = true;
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ(4U, dbfull()->TEST_NumWalPartitions());
}

TEST_P(DBWritePartitionTest, PartitionLeadersAppendConcurrently) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 16;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  ASSERT_EQ(kPartitions, MaxConcurrentAppends([&](size_t p) {
              return Put(cf_for_partition_[p], "main" + std::to_string(p),
                         "value");
            }));
  if (GetParam()) {
    ASSERT_EQ(kPartitions, MaxConcurrentAppends([&](size_t p) {
                return WalOnlyPut(cf_for_partition_[p],
                                  "wal_only" + std::to_string(p), "value");
              }));
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_EQ("value", Get(cf_for_partition_[p], "main" + std::to_string(p)));
  }

  // Each partition appended to a WAL of its own.
  VectorWalPtr wal_files;
  ASSERT_OK(db_->GetSortedWalFiles(wal_files));
  ASSERT_GE(wal_files.size(), kPartitions);
}

TEST_P(DBWritePartitionTest, SwitchWaitsForOtherPartitionsLeaders) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int blocked_cf = cf_for_partition_[0];
  const int switching_cf = cf_for_partition_[1];

  // The blocked leader holds its append open until a switch is waiting on
  // it. The switch must not start before that append completes.
  std::mutex mu;
  std::condition_variable cv;
  bool switch_waiting = false;
  bool blocked_append_open = false;
  std::atomic<bool> blocked_append_done{false};
  std::atomic<bool> switched_after_append{false};
  std::atomic<bool> switched{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        if (!static_cast<Slice*>(arg)->ends_with("blocked")) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        blocked_append_open = true;
        cv.notify_all();
        cv.wait_for(lock, std::chrono::seconds(10),
                    [&] { return switch_waiting; });
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (switch_waiting) {
                                            blocked_append_done = true;
                                          }
                                        });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CloseMemTableSwitchGate:WaitForLeaders", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        switch_waiting = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SwitchMemtable:AfterCreateWAL", [&](void*) {
        switched_after_append = blocked_append_done.load();
        switched = true;
      });

  // No put issued after the blocked append can return until the blocked
  // record is published, but the blocked append is held open pre-record
  // until the switch waits on it, so the fill and the switch-triggering puts
  // each run on their own thread (FillAndSwitchMemtable). The main thread
  // waits for the blocked append to open (hence past the switch gate) before
  // issuing them, so the switch deterministically observes the in-flight
  // leader it must wait for. A concurrent insert schedules the flush on the
  // memtable's next insert, and the write after that switches.
  port::Thread blocked_writer(
      [&] { ASSERT_OK(Put(blocked_cf, "key", "blocked")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(10),
                            [&] { return blocked_append_open; }));
  }
  std::vector<port::Thread> puts = FillAndSwitchMemtable(switching_cf, [&] {
    std::lock_guard<std::mutex> lock(mu);
    return switch_waiting;
  });
  blocked_writer.join();
  for (auto& put : puts) {
    put.join();
  }

  ASSERT_TRUE(switched.load());
  ASSERT_TRUE(switched_after_append.load());
  ASSERT_EQ("blocked", Get(blocked_cf, "key"));
  ASSERT_EQ("value", Get(switching_cf, "switch"));
}

TEST_P(DBWritePartitionTest, SequenceOrderHoldsUnderConcurrentSwitches) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 16;
  constexpr int kThreads = 16;
  constexpr int kWritesPerThread = 200;
  Options options = CurrentOptions();
  // Small enough that leaders keep switching memtables and rotating every
  // partition's WAL while other partitions append.
  options.db_write_buffer_size = 256 << 10;
  options.max_total_wal_size = 512 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int num_cfs = static_cast<int>(handles_.size());

  std::atomic<int> rotations{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::SwitchMemtable:AfterCreateWAL",
                                        [&](void*) { rotations.fetch_add(1); });

  // Every append asserts in WriteToWAL that it starts above the sequences
  // its file already holds.
  const std::string value(1 << 10, 'v');
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      Random rnd(301 + t);
      for (int i = 0; i < kWritesPerThread; ++i) {
        const std::string key =
            "t" + std::to_string(t) + "_" + std::to_string(i);
        const int cf = static_cast<int>(rnd.Uniform(num_cfs));
        if (GetParam() && i % 4 == 0) {
          ASSERT_OK(WalOnlyPut(cf, key, value));
        } else {
          ASSERT_OK(Put(cf, key, value));
        }
      }
    });
  }
  // Switches from outside the write path enter every partition's queue.
  threads.emplace_back([&] {
    for (int i = 0; i < 5; ++i) {
      ASSERT_OK(Flush(i % num_cfs));
    }
  });
  for (auto& thread : threads) {
    thread.join();
  }
  ASSERT_GT(rotations.load(), 0);

  for (int t = 0; t < kThreads; ++t) {
    Random rnd(301 + t);
    for (int i = 0; i < kWritesPerThread; ++i) {
      const std::string key = "t" + std::to_string(t) + "_" + std::to_string(i);
      const int cf = static_cast<int>(rnd.Uniform(num_cfs));
      if (GetParam() && i % 4 == 0) {
        ASSERT_EQ("NOT_FOUND", Get(cf, key));
      } else {
        ASSERT_EQ(value, Get(cf, key));
      }
    }
  }
}

TEST_P(DBWritePartitionTest, MergeBatchesInsertConcurrentlyAcrossPartitions) {
  // Q5: at N>1 every memtable insert must be concurrent-safe, because other
  // partitions' leaders may be inserting into the same memtable at the same
  // time (a batch routes by its minimum CF id but inserts into every CF it
  // touches). A batch containing a Merge therefore stays on the parallel
  // path instead of leaving it for a non-concurrent insert. Merges from both
  // partitions land and merge correctly. (Without the fix, any Merge batch
  // trips the debug assert that the non-parallel path is unreachable at N>1;
  // under TSan the unsynchronized same-memtable inserts race.)
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.merge_operator = MergeOperators::CreateStringAppendOperator();
  OpenWithPartitions(kPartitions, options);

  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  const int shared_cf = std::max(cf_a, cf_b);
  // Each thread merges into its own partition's CF plus the shared CF, so
  // the two partitions insert into the shared CF's memtable concurrently.
  constexpr int kThreads = 4;
  constexpr int kMergesPerThread = 25;
  std::atomic<int> failures{0};
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const int own_cf = (t % 2 == 0) ? cf_a : cf_b;
      for (int i = 0; i < kMergesPerThread; ++i) {
        WriteBatch batch;
        if (!batch
                 .Merge(handles_[own_cf], "shared_merge_key",
                        "t" + std::to_string(t))
                 .ok()) {
          ++failures;
          continue;
        }
        std::string key =
            "merge_t" + std::to_string(t) + "_" + std::to_string(i);
        if (!batch.Merge(handles_[shared_cf], key, "solo").ok()) {
          ++failures;
          continue;
        }
        if (!batch
                 .Merge(handles_[shared_cf], "cross_key",
                        "t" + std::to_string(t))
                 .ok()) {
          ++failures;
          continue;
        }
        if (!db_->Write(WriteOptions(), &batch).ok()) {
          ++failures;
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  ASSERT_EQ(0, failures.load());

  for (int t = 0; t < kThreads; ++t) {
    for (int i = 0; i < kMergesPerThread; ++i) {
      std::string key = "merge_t" + std::to_string(t) + "_" + std::to_string(i);
      ASSERT_EQ("solo", Get(shared_cf, key));
    }
  }
  // Every thread's marker merged into the cross-partition key.
  std::string cross = Get(shared_cf, "cross_key");
  for (int t = 0; t < kThreads; ++t) {
    EXPECT_NE(std::string::npos, cross.find("t" + std::to_string(t)));
  }
  // Each partition's own shared key merged its two writers' markers.
  for (int t = 0; t < kThreads; ++t) {
    const int own_cf = (t % 2 == 0) ? cf_a : cf_b;
    EXPECT_NE(std::string::npos,
              Get(own_cf, "shared_merge_key").find("t" + std::to_string(t)));
  }
}

TEST_P(DBWritePartitionTest, IngestWriteBatchWithIndexRejectedAtNPartitions) {
  // Q5: ingesting a WriteBatchWithIndex switches memtables inside the
  // leader, which is not part of the concurrent contract, so N>1 rejects it
  // at runtime in WriteImpl's wbwi block (the direct call and commit-time
  // ingestion both funnel through WriteImpl). Reverting the check trips the
  // debug assert in IngestWBWIAsMemtable instead, and fails this test
  // either way.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  auto wbwi = std::make_shared<WriteBatchWithIndex>(options.comparator, 0,
                                                    /*overwrite_key=*/true);
  ASSERT_OK(wbwi->Put("ingest_key", "ingest_value"));
  WriteOptions wo;
  wo.disableWAL = true;
  Status s = db_->IngestWriteBatchWithIndex(wo, wbwi);
  ASSERT_TRUE(s.IsNotSupported());
  ASSERT_NE(nullptr, s.getState());
  EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
}

TEST_P(DBWritePartitionTest, ConcurrentUnsafeConfigsRejectedAtOpen) {
  // Q5: configurations whose memtable work cannot be concurrent-safe are
  // rejected at Open when N>1. The blob and inplace-update rejections name
  // wal_partitions ahead of the concurrent-writes checks (N>1 forces
  // allow_concurrent_memtable_write on); the merge-counting and non-SkipList
  // rejections are new. Reverting either new guard opens successfully and
  // fails its case below; the last case is the allowed-config control.

  auto try_open =
      [](const std::string& name,
         std::function<void(Options*, ColumnFamilyOptions*)> tweak) {
        Options options;
        options.create_if_missing = true;
        options.wal_partitions = 4;
        options.track_and_verify_wals_in_manifest = true;
        options.partition_wal_usage =
            PartitionWALUsage::kPartitionByColumnFamily;
        options.two_write_queues = false;
        options.allow_concurrent_memtable_write = true;
        options.avoid_flush_during_shutdown = true;
        ColumnFamilyOptions cf_options(options);
        tweak(&options, &cf_options);
        const std::string dbname = test::PerThreadDBPath(name);
        std::vector<ColumnFamilyDescriptor> cfs;
        cfs.emplace_back(kDefaultColumnFamilyName, cf_options);
        std::vector<ColumnFamilyHandle*> handles;
        std::unique_ptr<DB> db;
        Status s = DBImpl::Open(options, dbname, cfs, &handles, &db,
                                /*seq_per_batch=*/false, /*batch_per_txn=*/true,
                                /*is_retry=*/false, /*can_retry=*/nullptr);
        if (s.ok()) {
          for (auto* h : handles) {
            delete h;
          }
          db.reset();
          EXPECT_OK(DestroyDB(dbname, options));
        }
        return s;
      };

  auto expect_reject =
      [&](const std::string& name,
          std::function<void(Options*, ColumnFamilyOptions*)> tweak,
          const char* fragment) {
        Status s = try_open(name, tweak);
        EXPECT_TRUE(s.IsNotSupported() || s.IsInvalidArgument())
            << name << ": " << s.ToString();
        ASSERT_NE(nullptr, s.getState()) << name;
        EXPECT_NE(nullptr, std::strstr(s.getState(), fragment)) << name;
      };

  expect_reject(
      "wal_q5_blob",
      [](Options* /*db_options*/, ColumnFamilyOptions* cf_options) {
        cf_options->enable_blob_files = true;
        cf_options->enable_blob_direct_write = true;
      },
      "enable_blob_direct_write");
  // VectorRep passes the generic concurrent-inserts check, so only the new
  // SkipList-only guard rejects it.
  expect_reject(
      "wal_q5_vector_rep",
      [](Options* /*db_options*/, ColumnFamilyOptions* cf_options) {
        cf_options->memtable_factory.reset(new VectorRepFactory());
      },
      "SkipList");
  expect_reject(
      "wal_q5_inplace",
      [](Options* /*db_options*/, ColumnFamilyOptions* cf_options) {
        cf_options->inplace_update_support = true;
      },
      "inplace_update_support");
  expect_reject(
      "wal_q5_merges",
      [](Options* /*db_options*/, ColumnFamilyOptions* cf_options) {
        cf_options->max_successive_merges = 10;
      },
      "max_successive_merges");
  EXPECT_OK(try_open(
      "wal_q5_allowed",
      [](Options* /*db_options*/, ColumnFamilyOptions* /*cf_options*/) {}));
}

TEST_P(DBWriteTest, N1ConcurrentUnsafeOptionsStillAccepted) {
  // Q5: the N>1 Open rejections are scoped to N>1: the same options open
  // fine at N=1 and Merge writes work there. (The existing
  // IngestWriteBatchWithIndex test covers WBWI ingestion at N=1.)
  Options options = GetOptions();
  options.create_if_missing = true;
  options.max_successive_merges = 10;
  options.memtable_factory.reset(new VectorRepFactory());
  options.merge_operator = MergeOperators::CreateStringAppendOperator();
  Reopen(options);
  ASSERT_OK(Merge("merge_key", "a"));
  ASSERT_OK(Merge("merge_key", "b"));
  ASSERT_EQ("a,b", Get("merge_key"));
}

TEST_P(DBWritePartitionTest, WatermarkAdvancesOnWriteCompletion) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  EXPECT_EQ(0, dbfull()->WalDurabilitySafePoint());
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "key" + std::to_string(p), "value"));
    EXPECT_EQ(dbfull()->GetLatestSequenceNumber(),
              dbfull()->WalDurabilitySafePoint());
  }
}

TEST_F(DBWriteTestUnparameterized, RejectsSeqPerBatchTwoQueues) {
  // R8: N > 1 + seq_per_batch + two_write_queues breaks the contiguous
  // published prefix the commit callbacks assume. WritePrepared without two
  // queues stays openable: WritePrepared in general is not rejected.
  // (WriteUnprepared shares seq_per_batch=true, so the same gate covers it.)
  // Two queues are rejected at N > 1 with or without seq_per_batch, ahead of
  // the seq_per_batch check, since they reject the WAL indexing N > 1 needs.

  auto try_open = [](const std::string& name, bool seq_per_batch,
                     bool two_write_queues) {
    Options options;
    options.create_if_missing = true;
    options.wal_partitions = 4;
    options.track_and_verify_wals_in_manifest = true;
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    options.two_write_queues = two_write_queues;
    options.allow_concurrent_memtable_write = true;
    options.avoid_flush_during_shutdown = true;
    const std::string dbname = test::PerThreadDBPath(name);
    std::vector<ColumnFamilyDescriptor> cfs;
    cfs.emplace_back(kDefaultColumnFamilyName, ColumnFamilyOptions(options));
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    Status s = DBImpl::Open(options, dbname, cfs, &handles, &db, seq_per_batch,
                            /*batch_per_txn=*/true,
                            /*is_retry=*/false, /*can_retry=*/nullptr);
    if (s.ok()) {
      for (auto* h : handles) {
        delete h;
      }
      db.reset();
      EXPECT_OK(DestroyDB(dbname, options));
    }
    return s;
  };

  Status s = try_open("wal_durability_r8_reject", /*seq_per_batch=*/true,
                      /*two_write_queues=*/true);
  EXPECT_TRUE(s.IsNotSupported());
  ASSERT_NE(nullptr, s.getState());
  EXPECT_NE(nullptr, std::strstr(s.getState(), "two_write_queues"));

  EXPECT_OK(try_open("wal_durability_r8_wp_no_2q", /*seq_per_batch=*/true,
                     /*two_write_queues=*/false));
  s = try_open("wal_durability_r8_2q_no_spb", /*seq_per_batch=*/false,
               /*two_write_queues=*/true);
  EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
}

TEST_P(DBWritePartitionTest, N1IndexedSkipsDurabilityTracker) {
  // Q4: at N==1 nothing consumes the durability tracker (the gate and
  // its snapshots are all N>1-gated), so even with WAL indexing on,
  // writes never enter it. The safe point stays at its initial 0 while
  // sequences advance; WatermarkAdvancesOnWriteCompletion is the N>1
  // positive control. Reverting the N>1 guard advances the safe point
  // here and fails this test.
  Options options = IndexedWalOptions(CurrentOptions());
  options.two_write_queues = false;
  Reopen(options);
  EXPECT_EQ(0, dbfull()->WalDurabilitySafePoint());
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(Put("key" + std::to_string(i), "value"));
  }
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("sync_key", "value", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  EXPECT_GT(dbfull()->GetLatestSequenceNumber(), 0U);
  EXPECT_EQ(0, dbfull()->WalDurabilitySafePoint());
}

TEST_P(DBWritePartitionTest, N1PipelinedAndTwoQueueWritesSkipTracker) {
  // As N1IndexedSkipsDurabilityTracker, for the other N==1 write paths:
  // the false run writes pipelined with WAL indexing on, the true run
  // writes through two queues (which Open allows only unindexed). Each
  // append runs after its registration point and before its completion
  // point, so a registration shows there as a pending range whether or not
  // it is later completed; none may appear. Afterwards nothing is pending
  // or was ever completed, and the safe point stays at 0 while sequences
  // advance. Reverting only a registration guard leaves a range pending;
  // reverting both guards advances the safe point. Reverting only a
  // completion guard trips the tracker's unpaired-completion assert.
  Options options = CurrentOptions();
  if (GetParam()) {
    options.two_write_queues = true;
  } else {
    options = IndexedWalOptions(options);
    options.enable_pipelined_write = true;
  }
  Reopen(options);

  std::atomic<int> appends{0};
  std::atomic<int> appends_with_pending_range{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        ++appends;
        if (dbfull()->TEST_WalDurabilityPendingLow(0) !=
            WalDurabilityTracker::kIdle) {
          ++appends_with_pending_range;
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  constexpr int kAsyncWrites = 10;
  for (int i = 0; i < kAsyncWrites; ++i) {
    ASSERT_OK(Put("key" + std::to_string(i), "value"));
  }
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("sync_key", "value", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");

  EXPECT_EQ(kAsyncWrites + 1, appends.load());
  EXPECT_EQ(0, appends_with_pending_range.load());
  EXPECT_EQ(WalDurabilityTracker::kIdle,
            dbfull()->TEST_WalDurabilityPendingLow(0));
  EXPECT_EQ(0, dbfull()->TEST_WalDurabilityDurableUpto(0));
  EXPECT_GT(dbfull()->GetLatestSequenceNumber(), 0U);
  EXPECT_EQ(0, dbfull()->WalDurabilitySafePoint());
}

TEST_P(DBWritePartitionTest, CompletionWaitsForPriorPartitionAppend) {
  // M2.5: a write completes once its own record is written and every previous
  // WAL record has completed. A's append on partition 0 is held open while B
  // appends on partition 1: B's append must proceed (the gate never touches
  // the append path) but B must not return until A lands.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool b_appended = false;
  std::atomic<bool> b_returned{false};
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            // A is still held before its
                                            // AddRecord, so this is B's.
                                            b_appended = true;
                                            cv.notify_all();
                                          }
                                        });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_b", "value_b");
    b_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  // B's append landed while A was held. The poll below gives B every chance
  // to (incorrectly) return before A is released; with the gate B cannot.
  const auto return_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  while (!b_returned.load() &&
         std::chrono::steady_clock::now() < return_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const bool b_returned_before_release = b_returned.load();
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");

  ASSERT_OK(b_status);
  EXPECT_TRUE(b_appended);
  EXPECT_FALSE(b_returned_before_release);
  ASSERT_EQ("value_a", Get(cf_a, "key_a"));
  ASSERT_EQ("value_b", Get(cf_b, "key_b"));
}

// A writer carrying this can never be folded into an earlier group as a
// follower: the group-formation check tests the joining writer, so its write
// always forms a group of its own.
class NoBatchWriteCallback : public WriteCallback {
 public:
  Status Callback(DB* /*db*/) override { return Status::OK(); }
  bool AllowWriteBatching() override { return false; }
};

TEST_P(DBWritePartitionTest, NextAppendProceedsWhileGroupWaits) {
  // R2: the thread performing exit duties promotes the next leader first,
  // then waits, then completes its followers -- so a slow partition delays
  // another partition's *completion*, never its *next append*. A is held
  // open on partition 0; B appends on partition 1 and parks in its
  // completion gate; C follows on partition 1 with batching disallowed so
  // it always forms its own group. C's append must land while B is still
  // waiting, and neither B nor C may return until A lands.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  int appends_while_held = 0;
  std::atomic<bool> b_returned{false};
  std::atomic<bool> c_returned{false};
  Status b_status;
  Status c_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            // A is still held before its
                                            // AddRecord, so these are B's and
                                            // C's.
                                            ++appends_while_held;
                                            cv.notify_all();
                                          }
                                        });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_b", "value_b");
    b_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(
        cv.wait_until(lock, deadline, [&] { return appends_while_held >= 1; }));
  }
  port::Thread c_thread([&] {
    WriteBatch batch;
    c_status = batch.Put(handles_[cf_b], "key_c", "value_c");
    if (c_status.ok()) {
      NoBatchWriteCallback no_batch;
      c_status = WriteImplWithCallback(&batch, &no_batch);
    }
    c_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(
        cv.wait_until(lock, deadline, [&] { return appends_while_held >= 2; }));
  }
  // Both appends landed while A was held. The poll below gives B and C every
  // chance to (incorrectly) return before A is released; with the gate
  // neither can.
  const auto return_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  while ((!b_returned.load() || !c_returned.load()) &&
         std::chrono::steady_clock::now() < return_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const bool returned_before_release = b_returned.load() || c_returned.load();
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  c_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");

  ASSERT_OK(b_status);
  ASSERT_OK(c_status);
  EXPECT_GE(appends_while_held, 2);
  EXPECT_FALSE(returned_before_release);
  ASSERT_EQ("value_a", Get(cf_a, "key_a"));
  ASSERT_EQ("value_b", Get(cf_b, "key_b"));
  ASSERT_EQ("value_c", Get(cf_b, "key_c"));
}

TEST_P(DBWritePartitionTest, MixedSyncWritesCompleteUnderConcurrency) {
  // M2.6 under contention: sync and async writes interleave across
  // partitions while memtables switch underneath, with an external flush
  // thread compounding the interleavings. Every write completes with OK
  // and every key is visible afterwards. This shakes the post-wait SyncWAL
  // path -- all-N locks taken from inside group exit -- against concurrent
  // appends, syncs, and switches, and would ABBA-deadlock in sync-marking
  // without the wal_sync_mark_mutex_ serialization (leadership marks run
  // others-then-own while SyncWalImpl claims ascending).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr int kThreads = 8;
  constexpr int kWritesPerThread = 50;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int num_cfs = static_cast<int>(handles_.size());
  WriteOptions sync_wo;
  sync_wo.sync = true;
  const std::string value(1 << 10, 'v');
  const std::string sync_value(1 << 10, 's');

  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      Random rnd(417 + t);
      for (int i = 0; i < kWritesPerThread; ++i) {
        const std::string key =
            "t" + std::to_string(t) + "_" + std::to_string(i);
        const int cf = static_cast<int>(rnd.Uniform(num_cfs));
        if (i % 3 == 0) {
          ASSERT_OK(Put(cf, key, sync_value, sync_wo));
        } else {
          ASSERT_OK(Put(cf, key, value));
        }
      }
    });
  }
  threads.emplace_back([&] {
    for (int i = 0; i < 5; ++i) {
      ASSERT_OK(Flush(i % num_cfs));
    }
  });
  for (auto& thread : threads) {
    thread.join();
  }

  for (int t = 0; t < kThreads; ++t) {
    Random rnd(417 + t);
    for (int i = 0; i < kWritesPerThread; ++i) {
      const std::string key = "t" + std::to_string(t) + "_" + std::to_string(i);
      const int cf = static_cast<int>(rnd.Uniform(num_cfs));
      ASSERT_EQ(i % 3 == 0 ? sync_value : value, Get(cf, key));
    }
  }
}

TEST_P(DBWritePartitionTest, LeadershipSyncFinishesMarksPastHorizon) {
  // A sync=true write's leadership sync marks every partition's WALs, but
  // its completion's horizon is cur_wal_number_, partition 0's current WAL,
  // and every other partition's current WAL is numbered above it. The
  // completion must still finish those marks, or the next sync waits on
  // WalSyncCV forever.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  OpenWithPartitions(kPartitions, CurrentOptions());
  const uint64_t horizon = dbfull()->TEST_WalSyncMarks(0).back().first;
  for (size_t p = 1; p < kPartitions; ++p) {
    ASSERT_GT(dbfull()->TEST_WalSyncMarks(p).back().first, horizon);
  }
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "k0", "v0", sync_wo));
  for (size_t p = 0; p < kPartitions; ++p) {
    for (const auto& mark : dbfull()->TEST_WalSyncMarks(p)) {
      ASSERT_FALSE(mark.second) << "partition " << p << " WAL " << mark.first;
    }
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "k" + std::to_string(p), "v", sync_wo));
  }
  ASSERT_OK(db_->SyncWAL());
}

TEST_P(DBWritePartitionTest, LeadershipSyncRecordsWalsBelowClaimHorizon) {
  // A sync write's completion judges its horizon by the claim-time
  // cur_wal_number_. Every inactive WAL it claimed is numbered below that,
  // so it is still recorded in the MANIFEST, at one partition and above.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("two_write_queues syncs through SyncWAL instead");
    return;
  }
  for (size_t num_partitions : {size_t{1}, size_t{2}}) {
    SCOPED_TRACE("num_partitions=" + std::to_string(num_partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.track_and_verify_wals_in_manifest = true;
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    std::vector<uint64_t> old_wals;
    for (size_t p = 0; p < num_partitions; ++p) {
      old_wals.push_back(dbfull()->TEST_CurrentWalNumber(p));
      ASSERT_OK(Put(cf_for_partition_[p], "k", "v"));
    }
    // Rotates every partition; the unflushed column families keep the old
    // WALs live.
    ASSERT_OK(dbfull()->TEST_SwitchMemtable());
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf_for_partition_[0], "k2", "v", sync_wo));
    const auto& recorded = dbfull()->GetVersionSet()->GetWalSet().GetWals();
    for (uint64_t wal : old_wals) {
      EXPECT_NE(recorded.end(), recorded.find(wal)) << "WAL " << wal;
    }
  }
}

TEST_P(DBWritePartitionTest, GateSyncToleratesCoupledRotation) {
  // The sync write's claim and fsync run in its completion gate, after its
  // group left the write queue. The sync write is held in its gate fsync by
  // slow WAL syncs, and a manual flush (a coupled rotation of every
  // partition) is started inside that window. D-9: the write is neither
  // inserted nor published until the fsync, so the rotation's memtable
  // switch waits for it (WaitForUnpublishedSyncGroups) and its WAL creation
  // falls after the sync's end. The sync completes correctly: the write is
  // acknowledged, every partition's deque stays consistent, and no mark is
  // left behind. Ported from GC17's Muse candidate.
  // Neuter: no WaitForUnpublishedSyncGroups in WaitForPendingWrites; the
  // rotation lands inside the sync window.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  auto latency_fs = std::make_shared<LatencyInjectionFileSystem>(
      FileSystem::Default(), LatencyInjectionFileSystem::Latency(),
      /*seed=*/0);
  std::unique_ptr<Env> latency_env(NewCompositeEnv(latency_fs));
  Defer close_before_env([this] { Close(); });
  Options options = CurrentOptions();
  options.env = latency_env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[1], "pending", "v"));
  LatencyInjectionFileSystem::Latency slow_sync;
  slow_sync.sync_micros = 50000;
  latency_fs->SetDefaultLatency(slow_sync);

  std::mutex mu;
  std::condition_variable cv;
  bool in_gate_sync = false;
  std::atomic<int> events{0};
  std::atomic<int> sync_started_at{0};
  std::atomic<int> rotated_at{0};
  std::atomic<int> sync_finished_at{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        in_gate_sync = true;
        cv.notify_all();
        int expected = 0;
        sync_started_at.compare_exchange_strong(expected, ++events);
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SwitchMemtable:AfterCreateWAL", [&](void*) {
        int expected = 0;
        rotated_at.compare_exchange_strong(expected, ++events);
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void*) {
        int expected = 0;
        sync_finished_at.compare_exchange_strong(expected, ++events);
      });
  SyncPoint::GetInstance()->EnableProcessing();

  WriteOptions sync_wo;
  sync_wo.sync = true;
  port::Thread writer(
      [&] { ASSERT_OK(Put(cf_for_partition_[0], "synced", "v", sync_wo)); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return in_gate_sync; }));
  }
  ASSERT_OK(Flush(cf_for_partition_[1]));
  writer.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  latency_fs->SetDefaultLatency(LatencyInjectionFileSystem::Latency());

  // The rotation waited for the gate sync to finish.
  ASSERT_NE(0, sync_started_at.load());
  ASSERT_NE(0, rotated_at.load());
  ASSERT_NE(0, sync_finished_at.load());
  EXPECT_LT(sync_started_at.load(), sync_finished_at.load());
  EXPECT_LT(sync_finished_at.load(), rotated_at.load());
  AssertPartitionDequesConsistent();
  // The sync completed and left no mark behind: a second sync write, which
  // would hang on a stuck getting_synced mark, succeeds.
  ASSERT_OK(Put(cf_for_partition_[0], "synced2", "v", sync_wo));
  AssertPartitionDequesConsistent();
}

// The partitioned-WAL histograms (G-50) instrument the single-queue path;
// with two_write_queues, Open() rejects WAL indexing and so N > 1 outside
// tests.
TEST_P(DBWritePartitionTest, DurabilityWaitAndCompletionHistograms) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr uint64_t kHeldMicros = 20000;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(options.statistics->Reset());

  // A's append is held; B appends behind it on the other partition and
  // parks in its gate until A is released at least kHeldMicros later.
  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  port::Thread a_thread(
      [&] { ASSERT_OK(Put(cf_for_partition_[0], "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return a_in_append; });
  }
  port::Thread b_thread(
      [&] { ASSERT_OK(Put(cf_for_partition_[1], "key_b", "value_b")); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  env_->SleepForMicroseconds(static_cast<int>(kHeldMicros));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");

  HistogramData wait;
  options.statistics->histogramData(WAL_DURABILITY_WAIT_MICROS, &wait);
  EXPECT_GE(wait.count, 1U);
  EXPECT_GE(wait.max, static_cast<double>(kHeldMicros));
  HistogramData completion;
  options.statistics->histogramData(WAL_PARTITION_COMPLETION_MICROS,
                                    &completion);
  EXPECT_EQ(2U, completion.count);
  EXPECT_GE(completion.max, static_cast<double>(kHeldMicros));
}

TEST_P(DBWritePartitionTest, SyncFanoutAndMutexHistograms) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  // Leaves every partition synced.
  ASSERT_OK(Put(cf_for_partition_[0], "k0", "v", sync_wo));
  ASSERT_OK(options.statistics->Reset());
  auto count = [&](Histograms h) {
    HistogramData data;
    options.statistics->histogramData(h, &data);
    return data.count;
  };
  auto sum = [&](Histograms h) {
    HistogramData data;
    options.statistics->histogramData(h, &data);
    return data.sum;
  };

  // A non-sync write takes only its own partition's WAL mutex.
  ASSERT_OK(Put(cf_for_partition_[1], "k1", "v"));
  EXPECT_EQ(1U, count(WAL_WRITE_MUTEX_WAIT_MICROS));
  EXPECT_EQ(0U, count(WAL_WRITE_MUTEX_ALL_WAIT_MICROS));
  EXPECT_EQ(0U, count(WAL_SYNC_FSYNCED_PARTITIONS));

  // Partitions 0 and 1 hold unsynced bytes, 2 and 3 are idle; the sync
  // write fsyncs only the two holding unsynced records (GC28 Q1), and takes
  // all N WAL mutexes for its marks.
  ASSERT_OK(Put(cf_for_partition_[0], "k2", "v"));
  ASSERT_OK(Put(cf_for_partition_[0], "k3", "v", sync_wo));
  EXPECT_EQ(1U, count(WAL_SYNC_FSYNCED_PARTITIONS));
  EXPECT_EQ(2U, sum(WAL_SYNC_FSYNCED_PARTITIONS));
  EXPECT_EQ(1U, count(WAL_SYNC_UNSYNCED_PARTITIONS));
  EXPECT_EQ(2U, sum(WAL_SYNC_UNSYNCED_PARTITIONS));
  EXPECT_EQ(1U, count(WAL_SYNC_BARRIER_MICROS));
  EXPECT_GE(count(WAL_WRITE_MUTEX_ALL_WAIT_MICROS), 1U);
}

TEST_P(DBWritePartitionTest, IngestionStallHistogram) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  OpenWithPartitions(kPartitions, options);
  const std::string sst = dbname_ + "/ingest.sst";
  SstFileWriter writer(EnvOptions(), options);
  ASSERT_OK(writer.Open(sst));
  ASSERT_OK(writer.Put("key", "value"));
  ASSERT_OK(writer.Finish());
  ASSERT_OK(options.statistics->Reset());
  ASSERT_OK(db_->IngestExternalFile(handles_[cf_for_partition_[1]], {sst},
                                    IngestExternalFileOptions()));
  HistogramData stall;
  options.statistics->histogramData(WAL_PARTITION_INGESTION_STALL_MICROS,
                                    &stall);
  EXPECT_EQ(1U, stall.count);
  ASSERT_EQ("value", Get(cf_for_partition_[1], "key"));
}

TEST_P(DBWritePartitionTest, NonLeadershipSyncsAdvanceSyncedSize) {
  // A successful SyncWAL or FlushWAL(true) (SyncWalImpl, as is the
  // completion gate's second sync) advances synced_size for the WALs it
  // fsynced, so a later sync write counts as unsynced only the partitions
  // written since: the other one, and its own (its claim follows its own
  // append, D-G11 (c)).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  auto unsynced_partitions_after = [&](const std::function<Status()>& sync) {
    EXPECT_OK(Put(cf_for_partition_[0], "a", "v"));
    EXPECT_OK(Put(cf_for_partition_[1], "b", "v"));
    EXPECT_OK(sync());
    EXPECT_OK(Put(cf_for_partition_[1], "c", "v"));
    EXPECT_OK(options.statistics->Reset());
    EXPECT_OK(Put(cf_for_partition_[0], "d", "v", sync_wo));
    HistogramData data;
    options.statistics->histogramData(WAL_SYNC_UNSYNCED_PARTITIONS, &data);
    EXPECT_EQ(1U, data.count);
    return data.sum;
  };
  EXPECT_EQ(2U, unsynced_partitions_after([&] { return db_->SyncWAL(); }));
  EXPECT_EQ(2U, unsynced_partitions_after(
                    [&] { return db_->FlushWAL(/*sync=*/true); }));
}

TEST_P(DBWritePartitionTest, SyncWalAdvancesOtherPartitionsSyncedSize) {
  // The multi-partition branch of FinishWalPartitionSync (G-50): a SyncWAL
  // advances synced_size for the other partitions' WALs too, so with both
  // partitions synced and nothing written after, a sync write records only
  // its own partition as unsynced. The count is taken at the claim in its
  // completion gate, after its own append (D-G11 (c)), so its own record is
  // the only unsynced byte range. Test idea ported from D126193958.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[0], "k0", "v"));
  ASSERT_OK(Put(cf_for_partition_[1], "k1", "v"));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(options.statistics->Reset());

  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "k2", "v", sync_wo));
  HistogramData unsynced;
  options.statistics->histogramData(WAL_SYNC_UNSYNCED_PARTITIONS, &unsynced);
  EXPECT_EQ(1U, unsynced.count);
  EXPECT_EQ(1U, unsynced.sum);
}

TEST_P(DBWritePartitionTest, SyncWalClaimTimeHorizonExcludesCoupledRotation) {
  // GC03's claim-time SyncWAL horizon, regressed: a coupled rotation landing
  // between a SyncWAL's claim and its completion must not mark the
  // rotated-in WALs synced. SyncWAL holds no WAL mutexes across its fsync
  // loop, so holding the loop open with slow WAL syncs and firing a coupled
  // rotation inside (observed via the latency counters) lands the rotation
  // mid-sync deterministically; sync points prove the order. Ported from
  // D126193958's CoupledRotationDuringSyncWal, renamed for the GC03 horizon
  // it pins rather than the leadership fix.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  auto latency_fs = std::make_shared<LatencyInjectionFileSystem>(
      FileSystem::Default(), LatencyInjectionFileSystem::Latency(),
      /*seed=*/0);
  std::unique_ptr<Env> latency_env(NewCompositeEnv(latency_fs));
  Defer close_before_env([this] { Close(); });
  Options options = CurrentOptions();
  options.env = latency_env.get();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  // Non-empty WALs everywhere, so the rotation installs a new generation.
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "init", "v"));
  }
  const uint64_t wal_before = dbfull()->TEST_CurrentWalNumber(0);
  LatencyInjectionFileSystem::Latency slow_sync;
  slow_sync.sync_micros = 100000;
  latency_fs->SetDefaultLatency(slow_sync);

  auto now_ns = [] {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  };
  std::atomic<uint64_t> sync_begin_ns{0};
  std::atomic<uint64_t> before_mark_ns{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWAL:Begin", [&](void*) { sync_begin_ns.store(now_ns()); });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWAL:BeforeMarkLogsSynced:1",
      [&](void*) { before_mark_ns.store(now_ns()); });
  SyncPoint::GetInstance()->EnableProcessing();

  Status sync_status;
  port::Thread syncer([&] { sync_status = db_->SyncWAL(); });
  // Wait until the SyncWAL is inside its slow fsync loop: the first fsync's
  // counter lands after its 100ms sleep with three still queued, so no
  // timing.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(60);
  for (;;) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "SyncWAL never reached its fsync loop";
    if (latency_fs->GetAggregateCounters().syncs >= 1) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  const uint64_t switch_ns = now_ns();
  // Bytes in the rotated-in WALs, before the SyncWAL completes: its
  // claim-time horizon leaves them untouched however this interleaves.
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "after-rotation", "v"));
  }
  syncer.join();
  ASSERT_OK(sync_status);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  latency_fs->SetDefaultLatency(LatencyInjectionFileSystem::Latency());

  ASSERT_GT(sync_begin_ns.load(), 0u);
  ASSERT_GT(before_mark_ns.load(), 0u);
  EXPECT_LT(sync_begin_ns.load(), switch_ns);
  EXPECT_LT(switch_ns, before_mark_ns.load());
  EXPECT_NE(wal_before, dbfull()->TEST_CurrentWalNumber(0));
  for (size_t p = 0; p < kPartitions; ++p) {
    for (const auto& mark : dbfull()->TEST_WalSyncMarks(p)) {
      ASSERT_FALSE(mark.second) << "partition " << p << " WAL " << mark.first;
    }
  }
  EXPECT_EQ("v", Get(cf_for_partition_[0], "init"));
  // The rotated-in WALs were not marked synced by that call: every
  // partition holds bytes past synced_size, so a sync write counts all
  // four (its own bytes land post-claim and are the only ones excluded).
  ASSERT_OK(options.statistics->Reset());
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "measured", "v", sync_wo));
  HistogramData unsynced;
  options.statistics->histogramData(WAL_SYNC_UNSYNCED_PARTITIONS, &unsynced);
  EXPECT_EQ(1U, unsynced.count);
  EXPECT_EQ(kPartitions, unsynced.sum);
  AssertPartitionDequesConsistent();
}

TEST_P(DBWritePartitionTest, ImportStallHistogram) {
  // CreateColumnFamilyWithImport holds every partition's writers like
  // ingestion does, and records one stall sample.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[1];
  ASSERT_OK(Put(cf, "key", "value"));
  ASSERT_OK(Flush(cf));
  Checkpoint* checkpoint = nullptr;
  ASSERT_OK(Checkpoint::Create(db_.get(), &checkpoint));
  std::unique_ptr<Checkpoint> checkpoint_guard(checkpoint);
  const std::string export_dir = dbname_ + "/export";
  ASSERT_OK(DestroyDir(env_, export_dir));
  ExportImportFilesMetaData* metadata = nullptr;
  ASSERT_OK(
      checkpoint->ExportColumnFamily(handles_[cf], export_dir, &metadata));
  std::unique_ptr<ExportImportFilesMetaData> metadata_guard(metadata);
  ASSERT_OK(options.statistics->Reset());
  ColumnFamilyHandle* imported = nullptr;
  ASSERT_OK(db_->CreateColumnFamilyWithImport(
      options, "imported", ImportColumnFamilyOptions(), *metadata, &imported));
  std::unique_ptr<ColumnFamilyHandle> imported_guard(imported);
  HistogramData stall;
  options.statistics->histogramData(WAL_PARTITION_INGESTION_STALL_MICROS,
                                    &stall);
  EXPECT_EQ(1U, stall.count);
  std::string value;
  ASSERT_OK(db_->Get(ReadOptions(), imported, "key", &value));
  ASSERT_EQ("value", value);
}

TEST_P(DBWritePartitionTest, DisableWalWritesWaitOnlyForLowerRanges) {
  // Sequences that never produce a WAL record never wait for durability, but
  // are published only above an applied prefix. With nothing in flight below
  // it, a disableWAL write is visible as soon as it returns; above a held WAL
  // write it parks until that write is released.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions no_wal;
  no_wal.disableWAL = true;

  // Idle: a naive min with the durability safe point (the highest durable
  // sequence when every partition is idle) would hide this write.
  ASSERT_OK(Put(cf_a, "key_wal", "value_wal"));
  const SequenceNumber before_idle = db_->GetLatestSequenceNumber();
  ASSERT_OK(Put(cf_b, "key_idle", "value_idle", no_wal));
  EXPECT_EQ(before_idle + 1, db_->GetLatestSequenceNumber());
  ASSERT_EQ("value_idle", Get(cf_b, "key_idle"));

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  std::atomic<bool> b_returned{false};
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  const SequenceNumber before_held = db_->GetLatestSequenceNumber();
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_nowal", "value_nowal", no_wal);
    b_returned.store(true);
  });
  // Inserted but parked behind A, and not visible while A is held.
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_FALSE(b_returned.load());
  EXPECT_EQ(before_held, db_->GetLatestSequenceNumber());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_nowal"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");

  ASSERT_OK(b_status);
  ASSERT_EQ("value_nowal", Get(cf_b, "key_nowal"));
  ASSERT_EQ("value_a", Get(cf_a, "key_a"));
}

TEST_P(DBWritePartitionTest, BackgroundErrorFailsWaitingWrite) {
  // R4: a background error arriving during the wait fails the group instead
  // of hanging it or reporting OK. A's append fails (pinned frontier plus a
  // background error under paranoid checks); B, waiting on A, aborts with
  // the error. A itself returns promptly: a failed append is never waited on.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool fail_next_append = false;
  bool b_appended = false;
  std::atomic<bool> b_returned{false};
  Status a_status;
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (!release_a) {
          b_appended = true;
          cv.notify_all();
        }
        if (fail_next_append) {
          fail_next_append = false;
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        }
      });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_b", "value_b");
    b_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  {
    // B's append already passed this point unmutated, so arming now fails
    // only A's.
    std::lock_guard<std::mutex> lock(mu);
    fail_next_append = true;
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");

  EXPECT_TRUE(b_returned.load());
  ASSERT_NOK(a_status);
  ASSERT_NOK(b_status);
  EXPECT_TRUE(b_status.IsIOError());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, PostWaitSyncErrorFailsSyncWrite) {
  // R4 (second half): for sync writes the gate's outcome includes the gate
  // sync. B waits on held A, then its gate sync fails; B reports the sync
  // error. GC28 Q3: B waits first and runs one gate sync; the post-wait
  // SyncWAL this test failed is gone on the standard path, so the failure
  // is aimed at B's gate sync instead. Standard path only: here no other
  // gate sync exists, so arming the failure from the start can only hit B's.
  // (The two-queues leadership sync is covered by the counts in
  // SyncWriteSyncsAgainOnlyAfterWait; the gate itself is shared code.)
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool b_appended = false;
  Status a_status;
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            b_appended = true;
                                            cv.notify_all();
                                          }
                                        });
  // B parks in its pre-sync wait while A is held, then runs one gate sync
  // once A lands; arming the failure from the start can only hit that sync.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        *static_cast<IOStatus*>(arg) = IOStatus::IOError("injected sync error");
      });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  WriteOptions sync_wo;
  sync_wo.sync = true;
  port::Thread b_thread(
      [&] { b_status = Put(cf_b, "key_b", "value_b", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");

  ASSERT_OK(a_status);
  ASSERT_NOK(b_status);
  EXPECT_TRUE(b_status.IsIOError());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, RejectsPipelinedWalPartitions) {
  // The gate runs between promotion and follower completion in the
  // single-queue exit; the pipelined exit has nowhere to run it, so
  // pipelined+N>1 is rejected even for tests (bypass enabled).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  Options options = CurrentOptions();
  options.wal_partitions = 4;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.allow_concurrent_memtable_write = true;
  options.enable_pipelined_write = true;
  Status s = TryReopen(options);
  EXPECT_TRUE(s.IsNotSupported());
  ASSERT_NE(nullptr, s.getState());
  EXPECT_NE(nullptr, std::strstr(s.getState(), "pipelined"));
}

TEST_P(DBWritePartitionTest, N1DefaultWritePathTakesPre08Locks) {
  // Q3: a default N=1 configuration pays no lock on the write path that
  // it did not pay before diff 08. Each take site 08 added or widened
  // (WriteToWAL's unconditional take, both sync-mark takes, the
  // leadership fsync's per-partition takes) fires a SyncPoint when
  // taken; at N=1 none fires for plain writes, sync writes, or SyncWAL.
  // The pre-08 take set is read off 60bb245c3869: WriteToWAL took only
  // under manual flush without two write queues, there was no mark
  // mutex, and the leadership fsync took nothing without manual flush.
  // Reverting any guard turns a zero below into a nonzero.
  std::atomic<int> wal_take{0};
  std::atomic<int> pre_mark{0};
  std::atomic<int> sync_loop_take{0};
  std::atomic<int> syncwal_mark{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:TakesWalMutex",
                                        [&](void*) { ++wal_take; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex",
      [&](void*) { ++pre_mark; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex",
      [&](void*) { ++sync_loop_take; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalImpl:TakesSyncMarkMutex", [&](void*) { ++syncwal_mark; });

  SyncPoint::GetInstance()->EnableProcessing();
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.two_write_queues = GetParam();
  Reopen(options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("async" + std::to_string(i), "value"));
  }
  EXPECT_EQ(0, wal_take.load());
  EXPECT_EQ(0, pre_mark.load());
  EXPECT_EQ(0, sync_loop_take.load());
  wal_take.store(0);
  pre_mark.store(0);
  sync_loop_take.store(0);
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("sync" + std::to_string(i), "value", sync_wo));
  }
  EXPECT_EQ(0, wal_take.load());
  EXPECT_EQ(0, pre_mark.load());
  EXPECT_EQ(0, sync_loop_take.load());
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(db_->SyncWAL());
  }
  EXPECT_EQ(0, syncwal_mark.load());

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:TakesWalMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalImpl:TakesSyncMarkMutex");
}

TEST_P(DBWritePartitionTest, N1ManualFlushWritePathTakesPre08Locks) {
  // Q3, the other half of the pre-08 set: at N=1 with manual flush the
  // pre-existing WriteToWAL take still fires (single queue only, as
  // before), the leadership sync takes all N (the pre-08 take, never
  // the per-partition takes the point observes), and no mark mutex
  // fires. Pins the fix against over-narrowing.
  std::atomic<int> wal_take{0};
  std::atomic<int> pre_mark{0};
  std::atomic<int> sync_loop_take{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:TakesWalMutex",
                                        [&](void*) { ++wal_take; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex",
      [&](void*) { ++pre_mark; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex",
      [&](void*) { ++sync_loop_take; });

  SyncPoint::GetInstance()->EnableProcessing();
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.manual_wal_flush = true;
  options.two_write_queues = GetParam();
  Reopen(options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  const int expect_take = GetParam() ? 0 : 3;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("async" + std::to_string(i), "value"));
  }
  EXPECT_EQ(expect_take, wal_take.load());
  EXPECT_EQ(0, pre_mark.load());
  EXPECT_EQ(0, sync_loop_take.load());
  wal_take.store(0);
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("sync" + std::to_string(i), "value", sync_wo));
  }
  EXPECT_EQ(expect_take, wal_take.load());
  EXPECT_EQ(0, pre_mark.load());
  EXPECT_EQ(0, sync_loop_take.load());

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:TakesWalMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
}

TEST_P(DBWritePartitionTest, N2WritePathTakesPartitionLocks) {
  // Q3, positive controls: every guard fires where the partitioned path
  // needs it. (With two write queues the append holds its mutex across
  // WriteToWAL and the leadership sync runs through SyncWAL, so the take
  // points observe the other sites instead.)
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  std::atomic<int> wal_take{0};
  std::atomic<int> pre_mark{0};
  std::atomic<int> sync_loop_take{0};
  std::atomic<int> syncwal_mark{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:TakesWalMutex",
                                        [&](void*) { ++wal_take; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex",
      [&](void*) { ++pre_mark; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex",
      [&](void*) { ++sync_loop_take; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalImpl:TakesSyncMarkMutex", [&](void*) { ++syncwal_mark; });

  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(2, options);
  const int cf = cf_for_partition_[0];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(cf, "async" + std::to_string(i), "value"));
  }
  EXPECT_EQ(GetParam() ? 0 : 3, wal_take.load());
  wal_take.store(0);
  pre_mark.store(0);
  sync_loop_take.store(0);
  syncwal_mark.store(0);
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(cf, "sync" + std::to_string(i), "value", sync_wo));
  }
  EXPECT_EQ(3, pre_mark.load());
  EXPECT_EQ(GetParam() ? 0 : 3, sync_loop_take.load());
  EXPECT_EQ(GetParam() ? 3 : 0, syncwal_mark.load());
  syncwal_mark.store(0);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(db_->SyncWAL());
  }
  EXPECT_EQ(2, syncwal_mark.load());

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:TakesWalMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalImpl:TakesSyncMarkMutex");
}

TEST_P(DBWritePartitionTest, SlowPartitionParksInGateWithoutError) {
  // R3 neuter: a slow partition parks the waiter in its completion gate
  // without any error. B appends on partition 1 while A is held open on
  // partition 0: B must reach the parked state (one waiter), stay blocked
  // over a bounded observation interval, and complete OK once A lands.
  // (CompletionWaitsForPriorPartitionAppend proves the same wait without
  // observing the parked state; this one pins the mechanism.)
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool b_appended = false;
  std::atomic<bool> b_returned{false};
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            b_appended = true;
                                            cv.notify_all();
                                          }
                                        });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_b", "value_b");
    b_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  // B's append landed while A was held: B must now be parked in its gate,
  // with no background error anywhere.
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_TRUE(dbfull()->TEST_GetBGError().ok());
  EXPECT_FALSE(dbfull()->TEST_WalDurabilityBGError());
  // Bounded observation only: B must still be blocked, by policy, not by
  // error. (Correctness never depends on this interval.)
  const auto return_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
  while (!b_returned.load() &&
         std::chrono::steady_clock::now() < return_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_FALSE(b_returned.load());
  EXPECT_EQ(1U, dbfull()->TEST_WalDurabilityWaiters());
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");

  ASSERT_OK(b_status);
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  ASSERT_EQ("value_a", Get(cf_a, "key_a"));
  ASSERT_EQ("value_b", Get(cf_b, "key_b"));
}

TEST_P(DBWritePartitionTest, BackgroundErrorWakesParkedGateWithFollowers) {
  // R1/R2/R6: a real WAL I/O failure on partition 0 fails every group
  // parked behind it, leader and followers alike. A is held open; B1 and
  // B2 are forced into one group on partition 1 and park in its gate (one
  // waiter); A's append then fails. A returns promptly without ever
  // waiting; B1 and B2 wake promptly with the background error. The DB
  // enters the same read-only state as a single-WAL failure, the failed
  // append ran exactly once on its own partition, and no acknowledged
  // sequence lies above the failed hole.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool fail_a_append = false;
  bool b1_in_wait = false;
  bool b2_linked = false;
  std::atomic<bool> b1_returned{false};
  std::atomic<bool> b2_returned{false};
  Status a_status;
  Status b1_status;
  Status b2_status;
  std::atomic<int> gate_installs{0};
  std::atomic<int> bg_error_wakes{0};
  std::atomic<int> resume_syncs{0};
  std::atomic<int> resume_fences{0};
  std::vector<DBImpl::WalAppendOutcome> append_outcomes;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_a_append) {
          fail_a_append = false;
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AppendOutcome", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        append_outcomes.push_back(*static_cast<DBImpl::WalAppendOutcome*>(arg));
      });
  // Force B1 and B2 into one group: B1 (partition 1's leader) waits here
  // until B2 has linked behind it, so group formation observes both.
  const size_t b_partition = 1;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        auto* w = static_cast<WriteThread::Writer*>(arg);
        if (w->wal_partition != b_partition) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!b1_in_wait) {
          b1_in_wait = true;
          cv.notify_all();
          cv.wait(lock, [&] { return b2_linked; });
        } else {
          b2_linked = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::InstallCompletionGate",
                                        [&](void*) { ++gate_installs; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ResumeImpl:SyncLastSequenceWithAllocated",
      [&](void*) { ++resume_syncs; });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::ResumeImpl:AfterSyncSeq",
                                        [&](void*) { ++resume_fences; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b1_thread([&] {
    b1_status = Put(cf_b, "key_b1", "value_b1");
    b1_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b1_in_wait; }));
  }
  port::Thread b2_thread([&] {
    b2_status = Put(cf_b, "key_b2", "value_b2");
    b2_returned.store(true);
  });
  // One group, one waiter: both B writes park behind the same gate.
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_a_append = true;
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b1_thread.join();
  b2_thread.join();

  EXPECT_TRUE(b1_returned.load());
  EXPECT_TRUE(b2_returned.load());
  ASSERT_NOK(a_status);
  EXPECT_TRUE(a_status.IsIOError());
  ASSERT_NOK(b1_status);
  ASSERT_NOK(b2_status);
  EXPECT_TRUE(b1_status.IsIOError());
  EXPECT_TRUE(b2_status.IsIOError());
  // Three writes, one gate install: B1 and B2 shared a group, so one of
  // the two failing statuses belongs to a follower.
  EXPECT_EQ(1, gate_installs.load());
  // Direct notification fired; nothing was left to the poll.
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  // Same background error and read-only state as a single-WAL failure: a
  // later write fails fast with the published error.
  Status bg_error = dbfull()->TEST_GetBGError();
  EXPECT_FALSE(bg_error.ok());
  EXPECT_TRUE(bg_error.IsIOError());
  Status read_only_status = Put(cf_b, "key_after", "value_after");
  EXPECT_FALSE(read_only_status.ok());
  EXPECT_TRUE(read_only_status.IsIOError());
  {
    std::lock_guard<std::mutex> lock(mu);
    // Each write attempted its append exactly once, on its own partition:
    // A on 0 (failed), the B group on 1 (landed but never acknowledged).
    ASSERT_EQ(2U, append_outcomes.size());
    const DBImpl::WalAppendOutcome* a_outcome = nullptr;
    const DBImpl::WalAppendOutcome* b_outcome = nullptr;
    const size_t a_partition = 0;
    for (const auto& outcome : append_outcomes) {
      if (outcome.wal_partition == a_partition) {
        a_outcome = &outcome;
      } else {
        b_outcome = &outcome;
      }
    }
    ASSERT_NE(nullptr, a_outcome);
    ASSERT_NE(nullptr, b_outcome);
    EXPECT_FALSE(a_outcome->ok);
    EXPECT_EQ(1U, a_outcome->sequence_count);
    EXPECT_TRUE(b_outcome->ok);
    EXPECT_EQ(2U, b_outcome->sequence_count);
    EXPECT_LT(a_outcome->first_sequence, b_outcome->first_sequence);
  }

  // Reconciliation still uses the existing path exactly once per Resume:
  // the two-queues Resume fence runs it, the standard path never needs it.
  dbfull()->Resume().PermitUncheckedError();
  EXPECT_EQ(1, resume_fences.load());
  EXPECT_EQ(GetParam() ? 1 : 0, resume_syncs.load());

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AppendOutcome");
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ResumeImpl:SyncLastSequenceWithAllocated");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::ResumeImpl:AfterSyncSeq");
}

TEST_P(DBWritePartitionTest, GatePostWaitSyncFailureWakesParkedGate) {
  // M2.7 (second half of R2): the completion gate's own sync establishes
  // the same background error as a leadership sync failure and wakes every
  // other parked gate. A (async, partition 0) is held open; B (sync,
  // partition 1) appends and parks; M (async, partition 2) allocates then
  // holds its append open; W (async, partition 0) appends after A is
  // released and parks behind M's still-open range. Releasing A lets B reach
  // its gate sync, which fails: W wakes and, being above B's failed range,
  // fails with the background error while M is still held (D-9; at GC24,
  // before D-9, it parked again as D-N71 (b) had it). M's append then fails
  // too.
  // GC28 Q3: B runs one gate sync after its wait; the post-wait SyncWAL this
  // test failed is gone on the standard path, so the failure is aimed at
  // B's gate sync instead. All assertions below are unchanged.
  // Neuter: no fence for a failed sync (FailUnpublishedSyncRange always
  // publishing the range as a hole): W is not failed while M is held.
  // Standard path only (as in PostWaitSyncErrorFailsSyncWrite): with two
  // write queues the leadership sync itself goes through SyncWAL, so the
  // failure cannot be aimed at the gate sync alone.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  const int cf_m = cf_for_partition_[2];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool m_in_append = false;
  bool release_m = false;
  bool fail_m_append = false;
  bool b_appended = false;
  std::atomic<bool> w_returned{false};
  std::atomic<bool> m_returned{false};
  Status a_status;
  Status b_status;
  Status w_status;
  Status m_status;
  std::atomic<int> gate_installs{0};
  std::atomic<int> bg_error_wakes{0};
  std::atomic<int> resume_syncs{0};
  std::atomic<int> resume_fences{0};
  std::atomic<int> gate_sync_begins{0};
  std::vector<DBImpl::WalAppendOutcome> append_outcomes;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        } else if (static_cast<Slice*>(arg)->ends_with("value_m")) {
          m_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_m; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (!release_a && !m_in_append) {
          b_appended = true;
          cv.notify_all();
        }
        if (fail_m_append) {
          fail_m_append = false;
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AppendOutcome", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        append_outcomes.push_back(*static_cast<DBImpl::WalAppendOutcome*>(arg));
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::InstallCompletionGate",
                                        [&](void*) {
                                          ++gate_installs;
                                          std::lock_guard<std::mutex> lock(mu);
                                          cv.notify_all();
                                        });
  // Only B runs a gate sync on this path (A, M and W are async). Hold it
  // until W installed its gate and parked, so W is provably parked when B's
  // sync fails.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
        ++gate_sync_begins;
        {
          std::unique_lock<std::mutex> lock(mu);
          cv.wait_until(lock, deadline,
                        [&] { return gate_installs.load() >= 3; });
        }
        const auto park_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (dbfull()->TEST_WalDurabilityWaiters() != 1 &&
               std::chrono::steady_clock::now() < park_deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        *static_cast<IOStatus*>(arg) = IOStatus::IOError("injected sync error");
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ResumeImpl:SyncLastSequenceWithAllocated",
      [&](void*) { ++resume_syncs; });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::ResumeImpl:AfterSyncSeq",
                                        [&](void*) { ++resume_fences; });
  std::atomic<std::thread::id> w_thread_id{std::thread::id()};
  std::atomic<int> w_parks{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForWalDurability:BeforePark", [&](void*) {
        if (std::this_thread::get_id() == w_thread_id.load()) {
          ++w_parks;
        }
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread(
      [&] { b_status = Put(cf_b, "key_b", "value_b", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  port::Thread m_thread([&] {
    m_status = Put(cf_m, "key_m", "value_m");
    m_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return m_in_append; }));
  }
  // W shares partition 0 with A, so it queues behind A and appends only
  // after A is released, allocating after M.
  port::Thread w_thread([&] {
    w_thread_id.store(std::this_thread::get_id());
    w_status = Put(cf_a, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  // B's sync failure wakes W, which fails without parking again while M is
  // held.
  const auto w_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while ((bg_error_wakes.load() < 1 || !w_returned.load()) &&
         std::chrono::steady_clock::now() < w_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  b_thread.join();
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_GE(w_parks.load(), 1);
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  EXPECT_TRUE(w_returned.load());
  EXPECT_FALSE(m_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_m_append = true;
    release_m = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  m_thread.join();

  ASSERT_OK(a_status);
  ASSERT_NOK(b_status);
  EXPECT_TRUE(b_status.IsIOError());
  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
  ASSERT_NOK(m_status);
  EXPECT_TRUE(m_status.IsIOError());
  // A, B, and W installed gates; M's append failed, so it installed none.
  EXPECT_EQ(3, gate_installs.load());
  // Exactly one gate sync ran: B's. One notification per failure: B's gate
  // sync, then M's append.
  EXPECT_EQ(1, gate_sync_begins.load());
  EXPECT_EQ(2, bg_error_wakes.load());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  Status bg_error = dbfull()->TEST_GetBGError();
  EXPECT_FALSE(bg_error.ok());
  EXPECT_TRUE(bg_error.IsIOError());
  Status read_only_status = Put(cf_b, "key_after", "value_after");
  EXPECT_FALSE(read_only_status.ok());
  {
    std::lock_guard<std::mutex> lock(mu);
    ASSERT_EQ(4U, append_outcomes.size());
    const DBImpl::WalAppendOutcome* a_outcome = nullptr;
    const DBImpl::WalAppendOutcome* b_outcome = nullptr;
    const DBImpl::WalAppendOutcome* m_outcome = nullptr;
    const DBImpl::WalAppendOutcome* w_outcome = nullptr;
    const size_t p1 = 1;
    const size_t p2 = 2;
    for (const auto& outcome : append_outcomes) {
      if (outcome.wal_partition == p1) {
        b_outcome = &outcome;
      } else if (outcome.wal_partition == p2) {
        m_outcome = &outcome;
      } else if (a_outcome == nullptr ||
                 outcome.first_sequence < a_outcome->first_sequence) {
        // Partition 0 holds A and W; A allocated first.
        w_outcome = a_outcome;
        a_outcome = &outcome;
      } else {
        w_outcome = &outcome;
      }
    }
    ASSERT_NE(nullptr, a_outcome);
    ASSERT_NE(nullptr, b_outcome);
    ASSERT_NE(nullptr, m_outcome);
    ASSERT_NE(nullptr, w_outcome);
    EXPECT_TRUE(a_outcome->ok);
    EXPECT_TRUE(b_outcome->ok);
    EXPECT_TRUE(w_outcome->ok);
    EXPECT_FALSE(m_outcome->ok);
    EXPECT_EQ(1U, a_outcome->sequence_count);
    EXPECT_EQ(1U, b_outcome->sequence_count);
    EXPECT_EQ(1U, m_outcome->sequence_count);
    EXPECT_EQ(1U, w_outcome->sequence_count);
    EXPECT_LT(a_outcome->first_sequence, b_outcome->first_sequence);
    EXPECT_LT(b_outcome->first_sequence, m_outcome->first_sequence);
    EXPECT_LT(m_outcome->first_sequence, w_outcome->first_sequence);
  }

  dbfull()->Resume().PermitUncheckedError();
  EXPECT_EQ(1, resume_fences.load());
  EXPECT_EQ(0, resume_syncs.load());

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AppendOutcome");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::CompletionGate:BeforeCoalescedSync");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ResumeImpl:SyncLastSequenceWithAllocated");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::ResumeImpl:AfterSyncSeq");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WaitForWalDurability:BeforePark");
}

TEST_P(DBWritePartitionTest, DurabilityWaitErrorBeforeAdvanceFailsWaiter) {
  // Invariant 4, error-first order: W parks behind held A and held F; F's
  // append, below W, fails while A is still held. W must wake promptly with
  // the error rather than waiting for A, with no lost wake and no leaked
  // waiter-count. A, released afterwards, still completes OK: its own
  // range is below F's, and coverage never depended on the error.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];
  const int cf_f = cf_for_partition_[2];

  std::mutex mu;
  std::condition_variable cv;
  int appends = 0;
  bool a_in_append = false;
  bool f_in_append = false;
  bool release_a = false;
  bool release_f = false;
  bool fail_next_append = false;
  bool w_appended = false;
  std::atomic<bool> w_returned{false};
  Status a_status;
  Status f_status;
  Status w_status;
  std::atomic<int> bg_error_wakes{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        const int append = appends++;
        if (append == 0) {
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        } else if (append == 1) {
          f_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_f; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_next_append) {
          fail_next_append = false;
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        } else if (!release_f) {
          w_appended = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread f_thread([&] { f_status = Put(cf_f, "key_f", "value_f"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return f_in_append; }));
  }
  port::Thread w_thread([&] {
    w_status = Put(cf_w, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_next_append = true;
    release_f = true;
    cv.notify_all();
  }
  f_thread.join();
  const auto w_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!w_returned.load() && std::chrono::steady_clock::now() < w_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(w_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(f_status);
  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
  EXPECT_EQ(dbfull()->TEST_GetBGError().ToString(), w_status.ToString());
  ASSERT_OK(a_status);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, DurabilityWaitAdvanceBeforeErrorCompletesWaiter) {
  // Invariant 4, advance-first order: W parks behind held A; A lands and
  // both complete OK. Publishing a background error afterwards, with no
  // waiters, must be safe and set the predicate flag for future waiters.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool w_appended = false;
  std::atomic<bool> w_returned{false};
  Status a_status;
  Status w_status;
  std::atomic<int> bg_error_wakes{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            w_appended = true;
                                            cv.notify_all();
                                          }
                                        });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread w_thread([&] {
    w_status = Put(cf_w, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();

  ASSERT_OK(a_status);
  ASSERT_OK(w_status);
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  EXPECT_EQ(0, bg_error_wakes.load());

  dbfull()->TEST_SetBGError(IOStatus::IOError("injected race error"),
                            BackgroundErrorReason::kWriteCallback);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_FALSE(dbfull()->TEST_GetBGError().ok());
  Status read_only_status = Put(cf_w, "key_after", "value_after");
  EXPECT_FALSE(read_only_status.ok());

  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, DurabilityWaitAdvanceRacesError) {
  // Invariant 4, true race: W parks behind held A; releasing A (the final
  // frontier advance covering W) races with background-error publication
  // from a barrier start. Either order must lose no wake, invent no stale
  // OK, deadlock nothing, and leak no waiter-count: an OK is valid only
  // below a covering frontier, and any failure carries the error.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kIterations = 10;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;

  for (int iter = 0; iter < kIterations; ++iter) {
    if (iter > 0) {
      // Fresh database per iteration: the previous one holds a background
      // error, and its WALs must not reopen.
      Destroy(options);
      Reopen(options);
    }
    OpenWithPartitions(kPartitions, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_w = cf_for_partition_[1];

    std::mutex mu;
    std::condition_variable cv;
    bool first_append = true;
    bool a_in_append = false;
    bool release_a = false;
    bool w_appended = false;
    Status a_status;
    Status w_status;
    SequenceNumber w_target = 0;
    std::atomic<int> bg_error_wakes{0};

    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (first_append) {
            first_append = false;
            a_in_append = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_a; });
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          if (!release_a) {
            w_appended = true;
            cv.notify_all();
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::InstallCompletionGate", [&](void* arg) {
          std::lock_guard<std::mutex> lock(mu);
          // A installs below W; the maximum install is W's target.
          w_target = std::max(w_target, *static_cast<SequenceNumber*>(arg));
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SignalWalDurabilityWaitersOnBGError",
        [&](void*) { ++bg_error_wakes; });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
    }
    port::Thread w_thread([&] { w_status = Put(cf_w, "key_w", "value_w"); });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
    }
    EXPECT_TRUE(WaitForDurabilityWaiters(1));

    std::mutex latch_mu;
    std::condition_variable latch_cv;
    bool start_race = false;
    port::Thread advance_thread([&] {
      {
        std::unique_lock<std::mutex> lock(latch_mu);
        latch_cv.wait(lock, [&] { return start_race; });
      }
      std::lock_guard<std::mutex> lock(mu);
      release_a = true;
      cv.notify_all();
    });
    port::Thread error_thread([&] {
      {
        std::unique_lock<std::mutex> lock(latch_mu);
        latch_cv.wait(lock, [&] { return start_race; });
      }
      dbfull()->TEST_SetBGError(IOStatus::IOError("injected race error"),
                                BackgroundErrorReason::kWriteCallback);
    });
    {
      std::lock_guard<std::mutex> lock(latch_mu);
      start_race = true;
      latch_cv.notify_all();
    }
    advance_thread.join();
    error_thread.join();
    a_thread.join();
    w_thread.join();

    ASSERT_OK(a_status);
    if (w_status.ok()) {
      // No stale OK: an acknowledged write sits below a covering frontier.
      std::lock_guard<std::mutex> lock(mu);
      EXPECT_NE(0U, w_target);
      EXPECT_LE(w_target, dbfull()->WalDurabilitySafePoint());
    } else {
      EXPECT_TRUE(w_status.IsIOError());
    }
    EXPECT_EQ(1, bg_error_wakes.load());
    EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());

    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord");
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SignalWalDurabilityWaitersOnBGError");
    dbfull()->Resume().PermitUncheckedError();
  }
}

TEST_P(DBWritePartitionTest, WalFailureMatchesSingleWALBehavior) {
  // R1: a WAL I/O failure on one partition produces the same DB-level
  // behaviour as a single-WAL failure. Run the identical injection at N=1
  // and N=2 and compare every observable: the failing write's status, the
  // published background error, and the read-only behaviour after it.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  options.two_write_queues = GetParam();

  auto run_single_write_failure = [&](size_t num_partitions,
                                      Status* append_status, Status* bg_error,
                                      Status* followup_status) {
    if (num_partitions == 1) {
      options.wal_partitions = 1;
      DestroyAndReopen(options);
      ASSERT_EQ(1U, dbfull()->TEST_NumWalPartitions());
    } else {
      OpenWithPartitions(num_partitions, options);
    }
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        });
    SyncPoint::GetInstance()->EnableProcessing();
    if (num_partitions == 1) {
      // Plain reopen leaves no column family handles behind.
      *append_status = Put("key_fail", "value_fail");
      *bg_error = dbfull()->TEST_GetBGError();
      *followup_status = Put("key_after", "value_after");
    } else {
      const int cf = cf_for_partition_[0];
      *append_status = Put(cf, "key_fail", "value_fail");
      *bg_error = dbfull()->TEST_GetBGError();
      *followup_status = Put(cf, "key_after", "value_after");
    }
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord");
    dbfull()->Resume().PermitUncheckedError();
  };

  Status append_1, bg_1, followup_1;
  run_single_write_failure(1, &append_1, &bg_1, &followup_1);
  Status append_n, bg_n, followup_n;
  run_single_write_failure(2, &append_n, &bg_n, &followup_n);

  ASSERT_NOK(append_1);
  EXPECT_TRUE(append_1.IsIOError());
  ASSERT_NOK(bg_1);
  ASSERT_NOK(followup_1);
  for (const auto& pair :
       {std::make_pair(&append_1, &append_n), std::make_pair(&bg_1, &bg_n),
        std::make_pair(&followup_1, &followup_n)}) {
    EXPECT_EQ(pair.first->code(), pair.second->code());
    EXPECT_EQ(pair.first->subcode(), pair.second->subcode());
    EXPECT_EQ(pair.first->severity(), pair.second->severity());
    EXPECT_EQ(pair.first->ToString(), pair.second->ToString());
  }
}

TEST_P(DBWritePartitionTest, DurabilityBGErrorFlagClearsWithBackgroundError) {
  // The park predicate's error leg mirrors the background error: set on
  // publication (with no waiters, safely lock-free) and cleared only when
  // the background error itself clears. Uses a soft error recovered
  // through TEST_ResumeImpl, which runs the clearing path directly:
  // Resume() itself reports a soft error as nothing to do.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(2, options);

  std::atomic<int> bg_error_wakes{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  EXPECT_FALSE(dbfull()->TEST_WalDurabilityBGError());
  dbfull()->TEST_SetBGError(IOStatus::IOError("injected soft error"),
                            BackgroundErrorReason::kAsyncFileOpen);
  Status bg = dbfull()->TEST_GetBGError();
  EXPECT_FALSE(bg.ok());
  EXPECT_TRUE(bg.IsIOError());
  EXPECT_EQ(Status::Severity::kSoftError, bg.severity());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(1, bg_error_wakes.load());

  ASSERT_OK(dbfull()->TEST_ResumeImpl(DBRecoverContext()));
  EXPECT_TRUE(dbfull()->TEST_GetBGError().ok());
  EXPECT_FALSE(dbfull()->TEST_WalDurabilityBGError());

  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");
}

// A listener that clears the next background error it sees, modelling a
// suppression: the error reaches no stored state and must wake nothing.
class SuppressNextBgErrorListener : public EventListener {
 public:
  void OnBackgroundError(BackgroundErrorReason /*reason*/,
                         Status* bg_error) override {
    if (suppress_next_) {
      suppress_next_ = false;
      *bg_error = Status::OK();
    }
  }

  void SuppressNext() { suppress_next_ = true; }

 private:
  bool suppress_next_ = false;
};

TEST_P(DBWritePartitionTest, CompactionErrorWakesParkedGate) {
  // Q1, non-WAL source: a real compaction failure wakes a parked gate
  // through the same transition hook as a WAL error. A is held open on
  // partition 0; W appends on partition 1 and parks in its gate; a file
  // compaction then fails at output verification. W wakes on the
  // publication while A is still held, but the error fences no range, so W
  // parks again, once, instead of failing or spinning (D-N71 (b)): once A
  // is released W is acknowledged, as one WAL acknowledges a write whose
  // append landed before an unrelated background error, while a new write
  // fails with the error. CompactFiles is used because CompactRange and Flush
  // seal the memtable first, and the switch waits for the held append.
  // Standard path only: even CompactFiles runs obsolete-file cleanup
  // (all N WAL mutexes) around its error publication, so with two write
  // queues it cannot publish while A's append holds its partition mutex.
  // The compaction-to-SetBGError wiring is queue-independent code, and
  // the two-queue wake mechanics are covered by the WAL run below.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(Put(cf_a, "pre_key" + std::to_string(i), "value"));
  }
  ASSERT_OK(Flush(cf_a));
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(Put(cf_a, "pre_key2_" + std::to_string(i), "value"));
  }
  ASSERT_OK(Flush(cf_a));
  ColumnFamilyMetaData meta;
  db_->GetColumnFamilyMetaData(handles_[cf_a], &meta);
  std::vector<std::string> l0_files;
  for (const auto& f : meta.levels[0].files) {
    l0_files.push_back(f.name);
  }
  ASSERT_GE(l0_files.size(), 2U);

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool w_appended = false;
  bool fail_compaction = false;
  int compaction_verifies = 0;
  std::atomic<bool> w_returned{false};
  Status compact_status;
  Status a_status;
  Status w_status;
  std::atomic<int> bg_error_wakes{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            w_appended = true;
                                            cv.notify_all();
                                          }
                                        });
  SyncPoint::GetInstance()->SetCallBack(
      "CompactionJob::Run():AfterVerifyOutputFiles", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_compaction) {
          fail_compaction = false;
          ++compaction_verifies;
          *static_cast<Status*>(arg) =
              IOStatus::IOError("injected compaction error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });
  std::atomic<std::thread::id> w_thread_id{std::thread::id()};
  std::atomic<int> w_parks{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForWalDurability:BeforePark", [&](void*) {
        if (std::this_thread::get_id() == w_thread_id.load()) {
          ++w_parks;
        }
      });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread w_thread([&] {
    w_thread_id.store(std::this_thread::get_id());
    w_status = Put(cf_w, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  // Armed before the compaction starts, so it cannot finish uninjected.
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_compaction = true;
  }
  port::Thread compact_thread([&] {
    compact_status = db_->CompactFiles(CompactionOptions(), handles_[cf_a],
                                       l0_files, /*output_level=*/1);
  });
  // The compaction error wakes W while A is still held, and W parks again.
  compact_thread.join();
  const auto w_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (w_parks.load() < 2 && std::chrono::steady_clock::now() < w_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_EQ(2, w_parks.load());
  EXPECT_EQ(1U, dbfull()->TEST_WalDurabilityWaiters());
  EXPECT_FALSE(w_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    ASSERT_FALSE(release_a);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "CompactionJob::Run():AfterVerifyOutputFiles");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WaitForWalDurability:BeforePark");

  ASSERT_NOK(compact_status);
  ASSERT_OK(w_status);
  ASSERT_OK(a_status);
  EXPECT_EQ(1, compaction_verifies);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  const Status bg_error = dbfull()->TEST_GetBGError();
  ASSERT_NOK(bg_error);
  EXPECT_EQ(bg_error.ToString(), Put(cf_w, "key_new", "value_new").ToString());
  EXPECT_EQ("value_w", Get(cf_w, "key_w"));
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, GateSyncFailureWakesParkedGate) {
  // Q1, gate sync failure, adapted to GC28 Q3's wait-then-sync order (the
  // base script pins sync-before-wait: C can no longer fail its gate fsync
  // while A is held open, since C parks behind A before syncing anything).
  // A is held open on partition 0; W appends on partition 1 and parks in
  // its gate; C (sync) appends and parks behind A with no gate sync
  // running. Once A is released W is acknowledged, as one WAL acknowledges
  // an async write whose append landed before a later sync failed, and C
  // runs exactly its gate fsync and fails it. The failure fences C's range,
  // which lies above W's, so W (already acknowledged below the fence) is
  // unaffected. Standard path only: with two write queues a sync write
  // takes all N WAL mutexes in PreprocessWrite, so C would block on A's
  // held append before ever reaching its sync. The two-queue write-path
  // sync still funnels through the same SetBGError transition the two-queue
  // append-failure run already covers; only the funnel (WriteStatusCheck)
  // differs, and it is pre-existing code.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool w_appended = false;
  bool fail_gate_sync = false;
  int gate_syncs = 0;
  std::atomic<bool> w_returned{false};
  Status c_status;
  Status a_status;
  Status w_status;
  std::atomic<int> bg_error_wakes{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a && !w_appended) {
                                            w_appended = true;
                                            cv.notify_all();
                                          }
                                        });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_gate_sync) {
          fail_gate_sync = false;
          ++gate_syncs;
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected gate sync error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread w_thread([&] {
    w_status = Put(cf_w, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  WriteOptions sync_options;
  sync_options.sync = true;
  std::atomic<bool> c_returned{false};
  port::Thread c_thread([&] {
    c_status = db_->Put(sync_options, handles_[cf_w], "key_c", "value_c");
    c_returned.store(true);
  });
  // C parks behind held A before syncing anything (GC28 Q3): no gate sync
  // runs while it parks, and it stays invisible.
  EXPECT_TRUE(WaitForDurabilityWaiters(2));
  EXPECT_EQ(0, gate_syncs);
  EXPECT_EQ("NOT_FOUND", Get(cf_w, "key_c"));
  {
    std::lock_guard<std::mutex> lock(mu);
    ASSERT_FALSE(release_a);
    fail_gate_sync = true;
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  c_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(c_status);
  EXPECT_TRUE(c_status.IsIOError());
  ASSERT_OK(w_status);
  ASSERT_OK(a_status);
  EXPECT_EQ(1, gate_syncs);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(c_status.ToString(), dbfull()->TEST_GetBGError().ToString());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, SyncWriteVisibleOnlyOnceDurable) {
  // GC25 Q1 (D-9): a sync write becomes visible only once it and every
  // record below it are fsynced, and a write above it neither returns nor
  // becomes visible before that. S (sync, partition 0) is held at the start
  // of its gate fsync; C (async, partition 1, above S) parks behind it. While
  // S is held, neither key reads through Get, MultiGet, a plain or a tailing
  // iterator, and the published sequence has not moved. Released, both
  // return OK and read back. Failed instead, S returns an IOError and C the
  // background error, and both are absent before and after Resume(), after
  // a flush, and after a reopen in every recovery mode.
  // Neuter: insert and publish the sync range before its fsync (RunSyncGate's
  // callbacks, insert and PublishLastSequence moved before
  // RunCoalescedGateSync): S reads back
  // while held, and C returns.
  if (GetParam()) {
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    for (const bool fail : {false, true}) {
      SCOPED_TRACE("partitions=" + std::to_string(partitions) +
                   " fail=" + std::to_string(fail));
      Options options = IndexedWalOptions(CurrentOptions());
      options.paranoid_checks = true;
      options.max_bgerror_resume_count = 0;
      DestroyAndReopen(options);
      OpenWithPartitions(partitions, options);
      const int cf_s = cf_for_partition_[0];
      const int cf_c = cf_for_partition_[1];
      WriteOptions sync_wo;
      sync_wo.sync = true;
      ASSERT_OK(Put(cf_s, "below", "value_below", sync_wo));
      const SequenceNumber before = db_->GetLatestSequenceNumber();

      std::mutex mu;
      std::condition_variable cv;
      bool s_in_sync = false;
      bool release_s = false;
      bool fail_sync = false;
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
            std::unique_lock<std::mutex> lock(mu);
            if (!s_in_sync) {
              s_in_sync = true;
              cv.notify_all();
              cv.wait(lock, [&] { return release_s; });
            }
          });
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
            std::lock_guard<std::mutex> lock(mu);
            if (fail_sync) {
              fail_sync = false;
              IOStatus error = IOStatus::IOError("injected gate sync error");
              error.SetRetryable(true);
              *static_cast<IOStatus*>(arg) = error;
            }
          });
      Status s_status;
      Status c_status;
      std::atomic<bool> c_returned{false};
      port::Thread s_thread(
          [&] { s_status = Put(cf_s, "key_s", "value_s", sync_wo); });
      {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                                [&] { return s_in_sync; }));
      }
      port::Thread c_thread([&] {
        c_status = Put(cf_c, "key_c", "value_c");
        c_returned = true;
      });
      ASSERT_TRUE(WaitForDurabilityWaiters(1));

      auto expect_reads = [&](const std::string& phase, bool visible) {
        SCOPED_TRACE(phase);
        const std::string s_value = visible ? "value_s" : "NOT_FOUND";
        const std::string c_value = visible ? "value_c" : "NOT_FOUND";
        EXPECT_EQ("value_below", Get(cf_s, "below"));
        EXPECT_EQ(s_value, Get(cf_s, "key_s"));
        EXPECT_EQ(c_value, Get(cf_c, "key_c"));
        std::vector<ColumnFamilyHandle*> cfhs = {handles_[cf_s],
                                                 handles_[cf_c]};
        std::vector<Slice> keys = {"key_s", "key_c"};
        std::vector<std::string> values;
        std::vector<Status> statuses =
            db_->MultiGet(ReadOptions(), cfhs, keys, &values);
        ASSERT_EQ(2U, statuses.size());
        for (const Status& st : statuses) {
          EXPECT_EQ(visible, st.ok()) << st.ToString();
          EXPECT_EQ(!visible, st.IsNotFound()) << st.ToString();
        }
        for (const bool tailing : {false, true}) {
          ReadOptions ro;
          ro.tailing = tailing;
          std::unique_ptr<Iterator> it(db_->NewIterator(ro, handles_[cf_s]));
          it->Seek("key_s");
          EXPECT_EQ(visible, it->Valid() && it->key() == "key_s")
              << (tailing ? "tailing" : "plain");
          EXPECT_OK(it->status());
        }
      };
      // Held: neither is visible, and C has not returned.
      EXPECT_FALSE(c_returned.load());
      EXPECT_EQ(before, db_->GetLatestSequenceNumber());
      expect_reads("held", /*visible=*/false);
      {
        std::lock_guard<std::mutex> lock(mu);
        fail_sync = fail;
        release_s = true;
        cv.notify_all();
      }
      s_thread.join();
      c_thread.join();
      SyncPoint::GetInstance()->ClearCallBack(
          "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
      SyncPoint::GetInstance()->ClearCallBack(
          "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
      if (!fail) {
        ASSERT_OK(s_status);
        ASSERT_OK(c_status);
        expect_reads("released", /*visible=*/true);
        Close();
        continue;
      }
      ASSERT_TRUE(s_status.IsIOError()) << s_status.ToString();
      ASSERT_NOK(c_status);
      EXPECT_EQ(dbfull()->TEST_GetBGError().ToString(), c_status.ToString());
      expect_reads("failed", /*visible=*/false);
      ASSERT_OK(dbfull()->Resume());
      expect_reads("resumed", /*visible=*/false);
      ASSERT_OK(Flush(cf_s));
      ASSERT_OK(Flush(cf_c));
      expect_reads("flushed", /*visible=*/false);
      for (const WALRecoveryMode mode :
           {WALRecoveryMode::kAbsoluteConsistency,
            WALRecoveryMode::kTolerateCorruptedTailRecords,
            WALRecoveryMode::kPointInTimeRecovery,
            WALRecoveryMode::kSkipAnyCorruptedRecords}) {
        options.wal_recovery_mode = mode;
        ReopenWithPartitions(partitions, options);
        expect_reads(
            "reopened in mode " + std::to_string(static_cast<int>(mode)),
            /*visible=*/false);
      }
      Close();
    }
  }
}

TEST_P(DBWritePartitionTest, SyncLedGroupAsyncWritersReturnBeforeFsync) {
  // GC25 Q2 (D-9): in a group led by a sync write, the async writers take
  // the group's lower sequences and return, visible, without waiting for
  // the fsync; the sync leader stays parked and invisible until it. W0
  // (async, partition 1) is held in its append as leader, while L (sync)
  // and then A1 and A2 (async) queue behind it on partition 1; released,
  // W0 hands L the group [L, A1, A2]. L's gate fsync is held: A1 and A2
  // return OK and read back, the published sequence stops just below L's,
  // and L has not returned and does not read. Released, L returns and reads
  // back.
  // Neuter: the async writers wait for an fsync before they complete
  // (SyncWalsFromCompletionGate at the start of async_writers_gate): A1 and
  // A2 do not return while L is held.
  if (GetParam()) {
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf, "below", "v", sync_wo));
    const SequenceNumber before = db_->GetLatestSequenceNumber();

    std::mutex mu;
    std::condition_variable cv;
    bool hold_append = true;
    bool w0_in_append = false;
    bool release_w0 = false;
    size_t waiting = 0;
    bool l_in_sync = false;
    bool release_l = false;
    size_t async_returned = 0;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (hold_append) {
            hold_append = false;
            w0_in_append = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_w0; });
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "WriteThread::JoinBatchGroup:BeganWaiting", [&](void* arg) {
          if (static_cast<WriteThread::Writer*>(arg)->wal_partition != 1) {
            return;
          }
          std::lock_guard<std::mutex> lock(mu);
          ++waiting;
          cv.notify_all();
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (!l_in_sync) {
            l_in_sync = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_l; });
          }
        });
    auto wait_for = [&](const std::function<bool()>& pred) {
      std::unique_lock<std::mutex> lock(mu);
      return cv.wait_for(lock, std::chrono::seconds(30), pred);
    };
    Status w0_status;
    Status l_status;
    Status a1_status;
    Status a2_status;
    port::Thread w0([&] { w0_status = Put(cf, "w0", "v"); });
    ASSERT_TRUE(wait_for([&] { return w0_in_append; }));
    port::Thread l([&] { l_status = Put(cf, "l", "v", sync_wo); });
    ASSERT_TRUE(wait_for([&] { return waiting == 1; }));
    auto async_writer = [&](const std::string& key, Status* status) {
      *status = Put(cf, key, "v");
      std::lock_guard<std::mutex> lock(mu);
      ++async_returned;
      cv.notify_all();
    };
    port::Thread a1([&] { async_writer("a1", &a1_status); });
    ASSERT_TRUE(wait_for([&] { return waiting == 2; }));
    port::Thread a2([&] { async_writer("a2", &a2_status); });
    ASSERT_TRUE(wait_for([&] { return waiting == 3; }));
    {
      std::lock_guard<std::mutex> lock(mu);
      release_w0 = true;
      cv.notify_all();
    }
    w0.join();
    ASSERT_OK(w0_status);
    ASSERT_TRUE(wait_for([&] { return l_in_sync; }));
    // The async writers return while L's fsync is held.
    EXPECT_TRUE(wait_for([&] { return async_returned == 2; }));
    EXPECT_EQ(before + 3, db_->GetLatestSequenceNumber());
    EXPECT_EQ("v", Get(cf, "w0"));
    EXPECT_EQ("v", Get(cf, "a1"));
    EXPECT_EQ("v", Get(cf, "a2"));
    EXPECT_EQ("NOT_FOUND", Get(cf, "l"));
    {
      std::lock_guard<std::mutex> lock(mu);
      release_l = true;
      cv.notify_all();
    }
    l.join();
    a1.join();
    a2.join();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    ASSERT_OK(l_status);
    ASSERT_OK(a1_status);
    ASSERT_OK(a2_status);
    EXPECT_EQ(before + 4, db_->GetLatestSequenceNumber());
    EXPECT_EQ("v", Get(cf, "l"));
    Close();
  }
}

TEST_P(DBWritePartitionTest, FailedSyncReopenWithoutResumeMayRecoverIt) {
  // GC25 Q4 (G3-A-3): the documented reopen rule. A sync write whose record
  // reached the WAL but whose gate sync failed returns the error and is not
  // visible; a reopen without Resume(), with the error still set at Close,
  // replays the record, as one WAL does, in every recovery mode. (After
  // Resume() it never reads back: SyncWriteVisibleOnlyOnceDurable.)
  if (GetParam()) {
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    for (const WALRecoveryMode mode :
         {WALRecoveryMode::kAbsoluteConsistency,
          WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kPointInTimeRecovery,
          WALRecoveryMode::kSkipAnyCorruptedRecords}) {
      SCOPED_TRACE("partitions=" + std::to_string(partitions) +
                   " mode=" + std::to_string(static_cast<int>(mode)));
      Options options = IndexedWalOptions(CurrentOptions());
      options.paranoid_checks = true;
      options.max_bgerror_resume_count = 0;
      DestroyAndReopen(options);
      OpenWithPartitions(partitions, options);
      const int cf = cf_for_partition_[0];
      WriteOptions sync_wo;
      sync_wo.sync = true;
      ASSERT_OK(Put(cf, "below", "v", sync_wo));
      std::atomic<bool> fail_sync{true};
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
            if (fail_sync.exchange(false)) {
              IOStatus error = IOStatus::IOError("injected gate sync error");
              error.SetRetryable(true);
              *static_cast<IOStatus*>(arg) = error;
            }
          });
      const Status s = Put(cf, "failed", "v", sync_wo);
      SyncPoint::GetInstance()->ClearCallBack(
          "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
      ASSERT_TRUE(s.IsIOError()) << s.ToString();
      ASSERT_NOK(dbfull()->TEST_GetBGError());
      EXPECT_EQ("NOT_FOUND", Get(cf, "failed"));
      Close();
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(partitions, options);
      EXPECT_EQ("v", Get(cf, "below"));
      EXPECT_EQ("v", Get(cf, "failed"));
      Close();
    }
  }
}

TEST_P(DBWritePartitionTest, FailedGateSyncWriteStaysInvisible) {
  // A sync write whose gate fsync fails stays invisible, as at N=1, where a
  // failed sync never inserted the write. Async A on partition 0 is held in
  // its append, so B (sync, partition 1, above A) cannot publish; B's gate
  // sync then fails. Once A is released A is acknowledged and
  // published, while B stays absent: before Resume(), after Resume() and a
  // flush, and after a reopen in every recovery mode. Then, above A2 held the
  // same way, C (sync) is held inside its gate sync while D (sync, above C)
  // queues behind it in the coalescer; C's sync fails, and both get the error
  // and stay absent. Every write acknowledged below them stays present.
  // GC28 Q3: the sync write waits for every record at or below its target
  // before its gate sync runs, so B (then C) parks while A (then A2) is
  // held; the held append is released once it parks. At the base the sync
  // ran first and its failure short-circuited the wait, so B returned while
  // A was still held.
  // Neuter: insert and publish the sync range before its fsync (RunSyncGate's
  // callbacks, insert and PublishLastSequence moved before
  // RunCoalescedGateSync); A's publication
  // exposes B.
  if (GetParam()) {
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf_a, "below", "value_below", sync_wo));

    std::mutex mu;
    std::condition_variable cv;
    bool hold_append = true;
    bool a_in_append = false;
    bool release_a = false;
    int fail_syncs = 0;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (hold_append) {
            hold_append = false;
            a_in_append = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_a; });
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
          std::lock_guard<std::mutex> lock(mu);
          if (fail_syncs > 0) {
            --fail_syncs;
            // Retryable, so Resume() can recover from it.
            IOStatus error = IOStatus::IOError("injected gate sync error");
            error.SetRetryable(true);
            *static_cast<IOStatus*>(arg) = error;
          }
        });
    // GC28 Q3: B parks in its pre-sync wait while A is held (C while A2 is
    // held, in the second half); release it once it parks. The held write
    // never reaches its own gate while held, so the first park after it is
    // held is the sync write's. All assertions below are unchanged.
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::RunCompletionGate:BeforeWait", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          if (a_in_append && !release_a) {
            release_a = true;
            cv.notify_all();
          }
        });

    Status a_status;
    port::Thread a_thread(
        [&] { a_status = Put(cf_a, "key_a", "value_a", WriteOptions()); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return a_in_append; }));
      fail_syncs = 1;
    }
    const Status b_status = Put(cf_b, "key_b", "value_b", sync_wo);
    ASSERT_TRUE(b_status.IsIOError()) << b_status.ToString();
    ASSERT_NOK(dbfull()->TEST_GetBGError());
    {
      std::lock_guard<std::mutex> lock(mu);
      release_a = true;
      cv.notify_all();
    }
    a_thread.join();
    // A lies below B's range, so its gate passes once it is published.
    ASSERT_OK(a_status);
    ASSERT_EQ("value_a", Get(cf_a, "key_a"));
    ASSERT_EQ("NOT_FOUND", Get(cf_b, "key_b"));

    ASSERT_OK(dbfull()->Resume());
    ASSERT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
    ASSERT_OK(Flush(cf_b));
    ASSERT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
    ASSERT_EQ("value_a", Get(cf_a, "key_a"));
    ASSERT_OK(Put(cf_b, "after", "value_after", sync_wo));
    ASSERT_EQ("value_after", Get(cf_b, "after"));

    // Coalesced: with A2 held below them, D queues behind C's running sync
    // and shares its failure.
    const int cf_d = cf_for_partition_[partitions - 1];
    {
      std::lock_guard<std::mutex> lock(mu);
      hold_append = true;
      a_in_append = false;
      release_a = false;
    }
    Status a2_status;
    port::Thread a2_thread(
        [&] { a2_status = Put(cf_a, "key_a2", "value_a2", WriteOptions()); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return a_in_append; }));
    }
    bool c_in_sync = false;
    bool release_c = false;
    bool d_waiting = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (!c_in_sync) {
            c_in_sync = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_c; });
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::RunCoalescedGateSync:BeforeWait", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          d_waiting = true;
          cv.notify_all();
        });
    Status c_status;
    Status d_status;
    port::Thread c_thread(
        [&] { c_status = Put(cf_b, "key_c", "value_c", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return c_in_sync; }));
    }
    port::Thread d_thread(
        [&] { d_status = Put(cf_d, "key_d", "value_d", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return d_waiting; }));
      fail_syncs = 1;
      release_c = true;
      cv.notify_all();
    }
    c_thread.join();
    d_thread.join();
    ASSERT_TRUE(c_status.IsIOError()) << c_status.ToString();
    ASSERT_TRUE(d_status.IsIOError()) << d_status.ToString();
    {
      std::lock_guard<std::mutex> lock(mu);
      release_a = true;
      cv.notify_all();
    }
    a2_thread.join();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    ASSERT_OK(a2_status);
    ASSERT_EQ("value_a2", Get(cf_a, "key_a2"));
    ASSERT_EQ("NOT_FOUND", Get(cf_b, "key_c"));
    ASSERT_EQ("NOT_FOUND", Get(cf_d, "key_d"));
    ASSERT_OK(dbfull()->Resume());
    ASSERT_OK(Flush(cf_b));
    ASSERT_OK(Flush(cf_d));
    ASSERT_EQ("NOT_FOUND", Get(cf_b, "key_c"));
    ASSERT_EQ("NOT_FOUND", Get(cf_d, "key_d"));

    for (const WALRecoveryMode mode :
         {WALRecoveryMode::kAbsoluteConsistency,
          WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kPointInTimeRecovery,
          WALRecoveryMode::kSkipAnyCorruptedRecords}) {
      SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(partitions, options);
      EXPECT_EQ("value_below", Get(cf_a, "below"));
      EXPECT_EQ("value_a", Get(cf_a, "key_a"));
      EXPECT_EQ("value_after", Get(cf_b, "after"));
      EXPECT_EQ("value_a2", Get(cf_a, "key_a2"));
      EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
      EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_c"));
      EXPECT_EQ("NOT_FOUND", Get(cf_d, "key_d"));
    }
    Close();
  }
}

TEST_P(DBWritePartitionTest, FailedGateSyncFencesCoalescedWaiterAtomically) {
  // Ported from GC21's losing candidate (Muse), for GC24 Q3 and Q9; under
  // D-9, two sync groups coalesce onto one failed gate sync, and with
  // auto-resume on, recovery's flush waits until neither group's entries
  // can still be published (WaitForUnpublishedSyncGroups). The runner (C)
  // is held between its failed sync and its fence while D waits on it,
  // until recovery has reached that wait; both writes are absent after
  // recovery and after a reopen.
  // Neuter: no WaitForUnpublishedSyncGroups in WaitForPendingWrites;
  // recovery never waits (recovery_waited stays false).
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 5;
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_a, "below", "value_below", sync_wo));

  std::mutex mu;
  std::condition_variable cv;
  int fail_syncs = 0;
  bool c_in_sync = false;
  bool release_c = false;
  bool d_waiting = false;
  bool recovery_waiting = false;
  std::atomic<bool> recovery_waited{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_syncs > 0) {
          --fail_syncs;
          // Retryable, so auto-resume can recover from it.
          IOStatus error = IOStatus::IOError("injected gate sync error");
          error.SetRetryable(true);
          *static_cast<IOStatus*>(arg) = error;
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!c_in_sync) {
          c_in_sync = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_c; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:BeforeWait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        d_waiting = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForUnpublishedSyncGroups:Wait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        recovery_waiting = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:AfterFailedSync", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        recovery_waited = cv.wait_for(lock, std::chrono::seconds(30),
                                      [&] { return recovery_waiting; });
      });

  Status c_status;
  Status d_status;
  port::Thread c_thread(
      [&] { c_status = Put(cf_b, "key_c", "value_c", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return c_in_sync; }));
  }
  port::Thread d_thread(
      [&] { d_status = Put(cf_b, "key_d", "value_d", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return d_waiting; }));
    fail_syncs = 1;
    release_c = true;
    cv.notify_all();
  }
  c_thread.join();
  d_thread.join();
  ASSERT_TRUE(recovery_waited.load());
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!dbfull()->TEST_GetBGError().ok()) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "auto-resume recovery never completed";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_TRUE(c_status.IsIOError()) << c_status.ToString();
  ASSERT_TRUE(d_status.IsIOError()) << d_status.ToString();
  // D shared C's failed sync (it did not run its own).
  EXPECT_NE(std::string::npos,
            d_status.ToString().find("injected gate sync error"))
      << d_status.ToString();
  EXPECT_EQ("value_below", Get(cf_a, "below"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_c"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_d"));

  ASSERT_OK(dbfull()->Resume());
  ASSERT_OK(Flush(cf_b));
  for (const WALRecoveryMode mode :
       {WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kPointInTimeRecovery,
        WALRecoveryMode::kSkipAnyCorruptedRecords}) {
    SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
    options.wal_recovery_mode = mode;
    ReopenWithPartitions(kPartitions, options);
    EXPECT_EQ("value_below", Get(cf_a, "below"));
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_c"));
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_d"));
  }
  Close();
}

TEST_P(DBWritePartitionTest, FailedSyncRangeDeletionHiddenUntilResumeDropsIt) {
  // Ported from GC21's losing candidate (Muse), for GC24 Q2; under D-9
  // (GC21 Q4): a failed sync write's point keys and range deletions are
  // invisible through Get, MultiGet, plain iterators and tailing iterators,
  // before and after Resume() drops them. The write below deletes
  // ["a", "m"), covering the acked key "f", and puts "point" outside the
  // range; it is held inside its gate sync while an async write above it,
  // "zebra", parks behind it. Its gate sync then fails, and so does
  // "zebra". "f" still reads and neither "point" nor "zebra" does, through
  // all four paths, before Resume, after Resume plus a flush, and after a
  // reopen in every recovery mode. The one exception: before Resume a
  // tailing iterator, which reads the memtable at kMaxSequenceNumber, may
  // show "zebra", which inserted before it failed, as it may any write
  // above a failed append (D-1).
  // Neuters: insert and publish the sync range before its fsync
  // (RunSyncGate's callbacks, insert and PublishLastSequence moved before
  // RunCoalescedGateSync): "zebra" no longer parks behind F. No drop at the
  // fence (UnreleasedWalGateFenceLow() returning kMaxSequenceNumber):
  // Resume()'s flush keeps "zebra", which then reads back.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 0;
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[0];
  ASSERT_OK(Put(cf, "f", "value_f"));

  // The failing write runs first and is held inside its gate sync; an async
  // write above it then parks behind it.
  std::mutex mu;
  std::condition_variable cv;
  int fail_syncs = 0;
  bool f_in_sync = false;
  bool release_f = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_syncs > 0) {
          --fail_syncs;
          IOStatus error = IOStatus::IOError("injected gate sync error");
          error.SetRetryable(true);
          *static_cast<IOStatus*>(arg) = error;
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!f_in_sync) {
          f_in_sync = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_f; });
        }
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status w_status;
  port::Thread f_thread([&] {
    WriteBatch batch;
    EXPECT_OK(batch.DeleteRange(handles_[cf], "a", "m"));
    EXPECT_OK(batch.Put(handles_[cf], "point", "value_point"));
    w_status = db_->Write(sync_wo, &batch);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return f_in_sync; }));
  }
  // Outside ["a", "m"). Same partition is fine: F already left its write
  // thread, so this forms its own group above F's range.
  Status zebra_status;
  port::Thread zebra_thread(
      [&] { zebra_status = Put(cf, "zebra", "value_zebra"); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_syncs = 1;
    release_f = true;
    cv.notify_all();
  }
  f_thread.join();
  zebra_thread.join();
  ASSERT_TRUE(zebra_status.IsIOError()) << zebra_status.ToString();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  ASSERT_TRUE(w_status.IsIOError()) << w_status.ToString();

  auto collect = [&](bool tailing) {
    ReadOptions ro;
    ro.tailing = tailing;
    std::map<std::string, std::string> kv;
    std::unique_ptr<Iterator> it(db_->NewIterator(ro, handles_[cf]));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      kv[it->key().ToString()] = it->value().ToString();
    }
    EXPECT_OK(it->status());
    return kv;
  };
  const std::map<std::string, std::string> kExpected = {{"f", "value_f"}};
  bool resumed = false;
  auto check_reads = [&](const std::string& phase) {
    SCOPED_TRACE(phase);
    EXPECT_EQ("value_f", Get(cf, "f"));
    EXPECT_EQ("NOT_FOUND", Get(cf, "point"));
    EXPECT_EQ("NOT_FOUND", Get(cf, "zebra"));
    {
      std::vector<ColumnFamilyHandle*> cfhs = {handles_[cf], handles_[cf],
                                               handles_[cf]};
      std::vector<Slice> keys = {"f", "point", "zebra"};
      std::vector<std::string> values;
      std::vector<Status> statuses =
          db_->MultiGet(ReadOptions(), cfhs, keys, &values);
      ASSERT_EQ(3U, statuses.size());
      EXPECT_OK(statuses[0]) << statuses[0].ToString();
      EXPECT_TRUE(statuses[1].IsNotFound()) << statuses[1].ToString();
      EXPECT_TRUE(statuses[2].IsNotFound()) << statuses[2].ToString();
      ASSERT_EQ(3U, values.size());
      EXPECT_EQ("value_f", values[0]);
    }
    EXPECT_EQ(kExpected, collect(false));
    std::map<std::string, std::string> tailing = collect(true);
    if (!resumed) {
      tailing.erase("zebra");
    }
    EXPECT_EQ(kExpected, tailing);
  };
  check_reads("before resume");

  ASSERT_OK(dbfull()->Resume());
  ASSERT_OK(Flush(cf));
  resumed = true;
  check_reads("after resume");

  for (const WALRecoveryMode mode :
       {WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kPointInTimeRecovery,
        WALRecoveryMode::kSkipAnyCorruptedRecords}) {
    SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
    options.wal_recovery_mode = mode;
    ReopenWithPartitions(kPartitions, options);
    check_reads("reopen");
  }
  Close();
}

TEST_P(DBWritePartitionTest, FailedZeroCountSyncLeavesNoIndeterminateRecord) {
  // Ported from GC21's losing candidate (Muse), for GC24 Q9, with its gap
  // count mapped to TEST_HasIndeterminateWriteSequence.
  // GC21 Q5 (Q8), under D-9: a zero-count sync write (LogData only) whose
  // gate sync fails holds no sequence, so once nothing was allocated above
  // it, its fence closes empty at Resume and no indeterminate record names
  // it. After Resume the WAL manager's indeterminate set is empty, and a
  // write after it succeeds.
  // Neuter: close the fence at max(low, last allocated), as before; the
  // indeterminate assertion below fails.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 0;
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[0];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf, "key", "value", sync_wo));

  int fail_syncs = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        if (fail_syncs > 0) {
          --fail_syncs;
          IOStatus error = IOStatus::IOError("injected gate sync error");
          error.SetRetryable(true);
          *static_cast<IOStatus*>(arg) = error;
        }
      });
  WriteBatch log_data_only;
  ASSERT_OK(log_data_only.PutLogData(Slice("zero-count-failed-marker")));
  fail_syncs = 1;
  const Status w_status = db_->Write(sync_wo, &log_data_only);
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  ASSERT_TRUE(w_status.IsIOError()) << w_status.ToString();

  ASSERT_OK(dbfull()->Resume());
  EXPECT_FALSE(dbfull()->TEST_HasIndeterminateWriteSequence());

  ASSERT_OK(Put(cf, "after", "value", sync_wo));
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("value", Get(cf, "key"));
  EXPECT_EQ("value", Get(cf, "after"));
  Close();
}

TEST_P(DBWritePartitionTest, SuppressedErrorLeavesGateUndisturbed) {
  // Q1: errors that never become the background error wake nothing, stick
  // nothing, and leave a parked gate undisturbed. W parks behind held A;
  // a compaction retryable (soft, never stored) and a listener-suppressed
  // error are published; neither fires the wake nor sets the flag, and W
  // stays parked. Repeating the suppressed error without the listener
  // then lands it, proving the listener did the suppressing; it fences no
  // range, so W still waits for A and is acknowledged (D-N71 (b)).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  auto listener = std::make_shared<SuppressNextBgErrorListener>();
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  options.listeners.push_back(listener);
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool w_appended = false;
  std::atomic<bool> w_returned{false};
  Status a_status;
  Status w_status;
  std::atomic<int> bg_error_wakes{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            w_appended = true;
                                            cv.notify_all();
                                          }
                                        });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread w_thread([&] {
    w_status = Put(cf_w, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
  }
  EXPECT_TRUE(WaitForDurabilityWaiters(1));

  // A compaction retryable maps to soft and is never stored.
  IOStatus retryable = IOStatus::IOError("injected compaction retryable");
  retryable.SetRetryable(true);
  dbfull()->TEST_SetBGError(retryable, BackgroundErrorReason::kCompaction);
  EXPECT_EQ(0, bg_error_wakes.load());
  EXPECT_FALSE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_TRUE(dbfull()->TEST_GetBGError().ok());
  EXPECT_EQ(1U, dbfull()->TEST_WalDurabilityWaiters());

  // A listener suppression clears the error before it is stored.
  listener->SuppressNext();
  dbfull()->TEST_SetBGError(IOStatus::IOError("suppressed error"),
                            BackgroundErrorReason::kWriteCallback);
  EXPECT_EQ(0, bg_error_wakes.load());
  EXPECT_FALSE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_TRUE(dbfull()->TEST_GetBGError().ok());
  EXPECT_EQ(1U, dbfull()->TEST_WalDurabilityWaiters());
  EXPECT_FALSE(w_returned.load());

  // The same error without suppression lands and wakes the parked gate:
  // the listener, not the classification, suppressed it above.
  dbfull()->TEST_SetBGError(IOStatus::IOError("suppressed error"),
                            BackgroundErrorReason::kWriteCallback);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_FALSE(dbfull()->TEST_GetBGError().ok());
  EXPECT_FALSE(w_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    ASSERT_FALSE(release_a);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_OK(a_status);
  ASSERT_OK(w_status);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, CheckThenParkRaceWakesWaiter) {
  // Q1, the check-then-park window: a failure published between the
  // waiter's predicate check and its park is never missed. W parks behind
  // held A and held F; the BeforePark hook holds W after its check with the
  // park mutex held; F's append, below W, then fails while W is held there
  // (its wake and fence block on the mutex W holds); W is then released and
  // must fail with the error instead of sleeping past it. There is no timer
  // to save a missed wake, so a lost signal would hang this test.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];
  const int cf_f = cf_for_partition_[2];

  std::mutex mu;
  std::condition_variable cv;
  int appends = 0;
  bool a_in_append = false;
  bool f_in_append = false;
  bool release_a = false;
  bool release_f = false;
  bool fail_next_append = false;
  bool w_appended = false;
  bool w_at_park = false;
  bool release_w = false;
  std::atomic<bool> w_returned{false};
  Status a_status;
  Status f_status;
  Status w_status;
  std::atomic<int> bg_error_wakes{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        const int append = appends++;
        if (append == 0) {
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        } else if (append == 1) {
          f_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_f; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_next_append) {
          fail_next_append = false;
          *static_cast<IOStatus*>(arg) = IOStatus::IOError("park-window error");
        } else if (!release_f) {
          w_appended = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForWalDurability:BeforePark", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!w_at_park) {
          w_at_park = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_w; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread f_thread([&] { f_status = Put(cf_f, "key_f", "value_f"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return f_in_append; }));
  }
  port::Thread w_thread([&] {
    w_status = Put(cf_w, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_appended; }));
  }
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return w_at_park; }));
  }
  // W has checked (no fence then) and is held before its park with the park
  // mutex held. Fail F: its wake and its fence block on that mutex until W
  // is released.
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_next_append = true;
    release_f = true;
    cv.notify_all();
  }
  // Release W only once the error is published: the blocked wake and fence
  // are then all that is left to wake W, and they must still reach W. The
  // bound only turns a flag that never appears into a reported failure; W
  // is released either way so the test can finish.
  const auto flag_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!dbfull()->TEST_WalDurabilityBGError() &&
         std::chrono::steady_clock::now() < flag_deadline) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  {
    std::lock_guard<std::mutex> lock(mu);
    release_w = true;
    cv.notify_all();
  }
  f_thread.join();
  const auto w_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!w_returned.load() && std::chrono::steady_clock::now() < w_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(w_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    ASSERT_FALSE(release_a);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WaitForWalDurability:BeforePark");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(f_status);
  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
  ASSERT_OK(a_status);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(w_status.ToString(), dbfull()->TEST_GetBGError().ToString());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, WalErrorFailsGatedGroupWithFollowers) {
  // Q1 (WAL source) and Q10 (follower receipt): a real WAL I/O failure
  // fails a parked multi-writer group, and every member receives the
  // identical published error. A is held open on partition 0; B1 and B2
  // are forced into one group on partition 1 and park in its single
  // gate; A's append then fails. Both members must return the same
  // status string as the stored background error: the follower's error
  // is the gate's outcome, not a generic failure.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  constexpr size_t b_partition = 1;
  const int cf_b = cf_for_partition_[b_partition];

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool fail_a_append = false;
  bool b1_in_wait = false;
  bool b2_linked = false;
  std::atomic<bool> b1_returned{false};
  std::atomic<bool> b2_returned{false};
  Status a_status;
  Status b1_status;
  Status b2_status;
  std::atomic<int> bg_error_wakes{0};
  std::atomic<int> gate_installs{0};

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_a_append) {
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        }
      });
  // Force B1 and B2 into one group: B1 (partition 1's first writer) waits
  // here until B2 has linked behind it, so group formation observes both.
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        auto* w = static_cast<WriteThread::Writer*>(arg);
        if (w->wal_partition != b_partition) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!b1_in_wait) {
          b1_in_wait = true;
          cv.notify_all();
          cv.wait(lock, [&] { return b2_linked; });
        } else {
          b2_linked = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::InstallCompletionGate",
                                        [&](void*) { ++gate_installs; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b1_thread([&] {
    b1_status = Put(cf_b, "key_b1", "value_b1");
    b1_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b1_in_wait; }));
  }
  port::Thread b2_thread([&] {
    b2_status = Put(cf_b, "key_b2", "value_b2");
    b2_returned.store(true);
  });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_a_append = true;
    release_a = true;
    cv.notify_all();
  }
  const auto b_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while ((!b1_returned.load() || !b2_returned.load()) &&
         std::chrono::steady_clock::now() < b_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(b1_returned.load());
  EXPECT_TRUE(b2_returned.load());
  a_thread.join();
  b1_thread.join();
  b2_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(a_status);
  ASSERT_NOK(b1_status);
  ASSERT_NOK(b2_status);
  EXPECT_TRUE(b1_status.IsIOError());
  EXPECT_TRUE(b2_status.IsIOError());
  // One group parked one gate; both members received its outcome, which
  // is the published background error verbatim.
  EXPECT_EQ(1, gate_installs.load());
  EXPECT_EQ(b1_status.ToString(), b2_status.ToString());
  EXPECT_EQ(b1_status.ToString(), dbfull()->TEST_GetBGError().ToString());
  EXPECT_EQ(b2_status.ToString(), dbfull()->TEST_GetBGError().ToString());
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, SkewedWritesRotateEveryPartitionAndReclaim) {
  // R1/R3 under skew: partition 1 takes almost every write, partition 2 a
  // trickle, partitions 0 and 3 none. Coupled rotation gives every partition
  // a new WAL at every memtable switch, so the quiet ones accumulate
  // near-empty files, and nothing is deleted while memtables are unflushed.
  // Once every column family flushes, all partitions retire in lockstep down
  // to their current WAL: no partition that stops writing keeps a file.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr size_t kHot = 1;
  constexpr size_t kCold = 2;
  constexpr int kHotWrites = 2000;
  constexpr int kColdEvery = 100;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 10;
  options.max_write_buffer_number = 64;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  // Hold every flush so each WAL is still alive, with its size, when counted.
  std::mutex mu;
  std::condition_variable cv;
  bool release_flush = false;
  std::atomic<int> rotations{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::BackgroundCallFlush:start", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return release_flush; });
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::SwitchMemtable:AfterCreateWAL",
                                        [&](void*) { rotations.fetch_add(1); });

  const std::string value(1 << 10, 'v');
  int cold_writes = 0;
  for (int i = 0; i < kHotWrites; ++i) {
    ASSERT_OK(Put(cf_for_partition_[kHot], "hot" + std::to_string(i), value));
    if (i % kColdEvery == 0) {
      ASSERT_OK(Put(cf_for_partition_[kCold], "cold" + std::to_string(i), "v"));
      ++cold_writes;
    }
  }
  const size_t num_rotations = static_cast<size_t>(rotations.load());
  ASSERT_GE(num_rotations, 10U);

  std::vector<uint64_t> all_wals;
  for (size_t p = 0; p < kPartitions; ++p) {
    const std::vector<std::pair<uint64_t, uint64_t>> files =
        dbfull()->TEST_AliveWalFiles(p);
    // The initial WAL plus one per rotation, in every partition.
    ASSERT_EQ(num_rotations + 1, files.size());
    size_t files_with_data = 0;
    for (const auto& file : files) {
      all_wals.push_back(file.first);
      files_with_data += file.second > 0 ? 1 : 0;
    }
    if (p == kHot) {
      EXPECT_GE(files_with_data, num_rotations);
    } else if (p == kCold) {
      EXPECT_LE(files_with_data, static_cast<size_t>(cold_writes));
    } else {
      EXPECT_EQ(0U, files_with_data);
    }
  }
  AssertPartitionDequesConsistent();
  // Invariant 1: every WAL is still on disk while its memtables are unflushed.
  std::sort(all_wals.begin(), all_wals.end());
  std::vector<uint64_t> on_disk = WalNumbersOnDisk();
  ASSERT_TRUE(std::includes(on_disk.begin(), on_disk.end(), all_wals.begin(),
                            all_wals.end()));

  {
    std::lock_guard<std::mutex> lock(mu);
    release_flush = true;
    cv.notify_all();
  }
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForPurge());
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::BackgroundCallFlush:start");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SwitchMemtable:AfterCreateWAL");

  // Invariant 2: every partition, quiet or not, is down to its current WAL,
  // and the reclaimed files are gone from disk.
  std::vector<uint64_t> current_wals;
  for (size_t p = 0; p < kPartitions; ++p) {
    current_wals.push_back(dbfull()->TEST_CurrentWalNumber(p));
    ASSERT_EQ(std::vector<uint64_t>{current_wals.back()}, AliveWalNumbers()[p]);
  }
  ASSERT_EQ(current_wals, WalNumbersOnDisk());
  ASSERT_EQ(value, Get(cf_for_partition_[kHot], "hot0"));
  ASSERT_EQ("v", Get(cf_for_partition_[kCold], "cold0"));
}

TEST_P(DBWritePartitionTest, RotationWaitsForHeldAppendAndKeepsItsWal) {
  // R6: write A on partition 0 is held while another write's memtable switch
  // waits. Held inside its append or between its append and its memtable
  // insert (B, on partition 1, parks behind it either way), the switch neither
  // deadlocks nor raises an error, and does not rotate until A is released.
  // Then every write completes, A's record is in partition 0's pre-switch
  // WAL, and that WAL is still alive. A's column family holds nothing but
  // A's record, so a switch landing before A's memtable insert would treat
  // it as empty and advance its log number past the record; reopening
  // recovers it.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  if (!GetParam()) {
    options = IndexedWalOptions(options);
  }
  options.write_buffer_size = 64 << 10;
  options.avoid_flush_during_shutdown = true;
  for (const bool hold_in_append : {true, false}) {
    SCOPED_TRACE(hold_in_append ? "held in append" : "held before insert");
    if (!hold_in_append) {
      Destroy(options);
      Reopen(options);
    }
    OpenWithPartitions(kPartitions, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    // The switch runs on partition 1, whose queue B has left, in a column
    // family that is neither A's nor B's.
    const int cf_switch = CreateColumnFamilyOnPartition(1, options);
    const uint64_t wal_a_before = dbfull()->TEST_CurrentWalNumber(0);
    const uint64_t wal_b_before = dbfull()->TEST_CurrentWalNumber(1);

    std::mutex mu;
    std::condition_variable cv;
    bool a_held = false;
    bool release_a = false;
    bool b_appended = false;
    bool switch_waiting = false;
    std::atomic<int> rotations{0};
    std::atomic<bool> b_returned{false};
    Status a_status;
    Status b_status;
    std::vector<DBImpl::WalAppendOutcome> outcomes;
    auto hold_a = [&]() {
      std::unique_lock<std::mutex> lock(mu);
      if (a_held) {
        return;
      }
      a_held = true;
      cv.notify_all();
      cv.wait(lock, [&] { return release_a; });
    };
    // A is the first WAL write, so it is also the first gate install.
    SyncPoint::GetInstance()->SetCallBack(hold_in_append
                                              ? "DBImpl::WriteToWAL:log_entry"
                                              : "DBImpl::InstallCompletionGate",
                                          [&](void*) { hold_a(); });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AppendOutcome", [&](void* arg) {
          std::lock_guard<std::mutex> lock(mu);
          outcomes.push_back(*static_cast<DBImpl::WalAppendOutcome*>(arg));
          if (a_held && !release_a) {
            b_appended = true;
            cv.notify_all();
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::CloseMemTableSwitchGate:WaitForLeaders", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          switch_waiting = true;
          cv.notify_all();
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SwitchMemtable:AfterCreateWAL",
        [&](void*) { rotations.fetch_add(1); });

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_until(lock, deadline, [&] { return a_held; }));
    }
    port::Thread b_thread([&] {
      b_status = Put(cf_b, "key_b", "value_b");
      b_returned.store(true);
    });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
    }
    // A is unpublished either way (its record has not landed, or it is not
    // inserted yet), so B parks behind it.
    ASSERT_TRUE(WaitForDurabilityWaiters(1));
    std::vector<port::Thread> switch_puts =
        FillAndSwitchMemtable(cf_switch, [&] {
          std::lock_guard<std::mutex> lock(mu);
          return switch_waiting;
        });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(
          cv.wait_until(lock, deadline, [&] { return switch_waiting; }));
    }
    // Held: a wait, not an error. Bounded observation only: the switch must
    // still be waiting on A. (Correctness never depends on this interval.)
    const auto observe_deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    while (rotations.load() == 0 &&
           std::chrono::steady_clock::now() < observe_deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(0, rotations.load());
    EXPECT_OK(dbfull()->TEST_GetBGError());
    EXPECT_FALSE(b_returned.load());
    {
      std::lock_guard<std::mutex> lock(mu);
      release_a = true;
      cv.notify_all();
    }
    a_thread.join();
    b_thread.join();
    for (auto& put : switch_puts) {
      put.join();
    }
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AppendOutcome");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::CloseMemTableSwitchGate:WaitForLeaders");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SwitchMemtable:AfterCreateWAL");

    ASSERT_OK(a_status);
    ASSERT_OK(b_status);
    ASSERT_GE(rotations.load(), 1);
    ASSERT_EQ(2U, outcomes.size());
    for (const DBImpl::WalAppendOutcome& outcome : outcomes) {
      EXPECT_TRUE(outcome.ok);
      EXPECT_EQ(outcome.wal_partition == 0 ? wal_a_before : wal_b_before,
                outcome.wal_number);
    }
    ASSERT_NE(wal_a_before, dbfull()->TEST_CurrentWalNumber(0));
    ASSERT_EQ(wal_a_before, AliveWalNumbers()[0].front());

    if (GetParam()) {
      // Reopening at N > 1 needs indexed WALs, which two_write_queues
      // rejects.
      continue;
    }
    ReopenWithPartitions(kPartitions, options);
    ASSERT_EQ("value_a", Get(cf_a, "key_a"));
    ASSERT_EQ("value_b", Get(cf_b, "key_b"));
  }
}

TEST_P(DBWritePartitionTest, RotationWalCreationFailureWakesGates) {
  // R6: a CreateWAL failure for partition 1 during rotation publishes the
  // background error the same way an append failure does: the predicate
  // flag is set and the wake fires once, so a parked gate wakes and fails on
  // it. No gate can be parked at that point (the rotation runs only
  // after every in-flight leader has exited), which the test records. The
  // failed rotation publishes nothing: every partition keeps its WALs.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "key", "value"));
  }
  const std::vector<std::vector<uint64_t>> alive_before = AliveWalNumbers();

  int wal_starts = 0;
  size_t waiters_at_failure = std::numeric_limits<size_t>::max();
  std::atomic<int> bg_error_wakes{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::StartWALFile:AfterCompressionTypeRecord", [&](void* arg) {
        // Partition 0's replacement starts first; fail partition 1's.
        if (++wal_starts == 2) {
          waiters_at_failure = dbfull()->TEST_WalDurabilityWaiters();
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL creation error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });

  Status flush_status = Flush(cf_for_partition_[0]);
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::StartWALFile:AfterCompressionTypeRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(flush_status);
  EXPECT_TRUE(flush_status.IsIOError());
  EXPECT_EQ(2, wal_starts);
  EXPECT_EQ(0U, waiters_at_failure);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_NOK(dbfull()->TEST_GetBGError());
  EXPECT_EQ(alive_before, AliveWalNumbers());
  AssertPartitionDequesConsistent();
  EXPECT_NOK(Put(cf_for_partition_[1], "key_after", "value_after"));
  // No gate can be parked at the failure (waiters_at_failure == 0:
  // rotation runs only after every in-flight leader has exited). What is
  // observable after it: a new write fails fast on the published error,
  // rejected at the stopped check before it could reach a gate, so the
  // waiter count never moves.
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  Status after_fail = Put(cf_for_partition_[0], "key_after2", "value_after2");
  EXPECT_NOK(after_fail);
  EXPECT_TRUE(after_fail.IsIOError());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, DataLessTornPartitionWalIsABoundedTail) {
  // G-21 at N > 1: a partition's current WAL torn mid-first-record, or
  // mid-marker, while the other partition's holds data below it. The
  // reopen's spans anchor the lower bound (the torn file's own, or the lower
  // partition's when the marker tore), so what the torn file lost would
  // surface as a hole: it is a lost tail, and every mode that tolerates a
  // tail opens with the other partition's data and nothing durable lost.
  // Neuter: stop writing spans; every mode fails with "head damaged".
  if (GetParam()) {
    // Reopening at N > 1 needs indexed WALs, which two_write_queues rejects.
    return;
  }
  constexpr size_t kPartitions = 2;
  for (const bool mid_marker : {false, true}) {
    SCOPED_TRACE(mid_marker ? "mid-marker" : "mid-first-record");
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    ASSERT_OK(Put(cf_for_partition_[0], "key0", "value"));
    ASSERT_OK(Put(cf_for_partition_[1], "key1", "value"));
    const std::string torn_wal =
        LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(1));
    Close();
    uint64_t size = 0;
    ASSERT_OK(env_->GetFileSize(torn_wal, &size));
    ASSERT_OK(test::TruncateFile(env_, torn_wal,
                                 mid_marker ? log::kHeaderSize - 2 : size - 1));

    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_partitions = kPartitions;
    for (const WALRecoveryMode recovery_mode :
         {WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kPointInTimeRecovery,
          WALRecoveryMode::kSkipAnyCorruptedRecords}) {
      SCOPED_TRACE(static_cast<int>(recovery_mode));
      options.wal_recovery_mode = recovery_mode;
      ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
      EXPECT_EQ("value", Get(cf_for_partition_[0], "key0"));
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "key1"));
      Close();
    }
  }
}

TEST_P(DBWritePartitionTest, ReopenRestoresWalsToTheirPartitions) {
  // R5: closed WALs return to the partition that wrote them, read from the
  // marker. Two rotations leave three WALs in every partition. After reopen
  // each partition holds exactly its own, ascending, then the new WAL its
  // writer appends to; new appends land in that file; and flushing retires
  // the recovered WALs from every partition.
  if (GetParam()) {
    // Reopening at N > 1 needs indexed WALs, which two_write_queues rejects.
    return;
  }
  constexpr size_t kPartitions = 4;
  const Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(kPartitions, options);
  ColumnFamilyData* rotating_cfd =
      static_cast_with_check<ColumnFamilyHandleImpl>(
          handles_[cf_for_partition_[0]])
          ->cfd();
  for (int round = 0; round < 3; ++round) {
    if (round > 0) {
      ASSERT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
    }
    for (size_t p = 0; p < kPartitions; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p],
                    "key" + std::to_string(round) + "_" + std::to_string(p),
                    "value"));
    }
  }
  const std::vector<std::vector<uint64_t>> before = AliveWalNumbers();

  ReopenWithPartitions(kPartitions, options);

  AssertPartitionDequesConsistent();
  const std::vector<std::vector<uint64_t>> after = AliveWalNumbers();
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_EQ(3U, before[p].size());
    std::vector<uint64_t> expected = before[p];
    expected.push_back(dbfull()->TEST_CurrentWalNumber(p));
    ASSERT_EQ(expected, after[p]);
    // SwitchWAL takes partition 0's front as the oldest alive WAL.
    ASSERT_LE(after[0].front(), after[p].front());
  }

  std::vector<DBImpl::WalAppendOutcome> outcomes;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AppendOutcome", [&](void* arg) {
        outcomes.push_back(*static_cast<DBImpl::WalAppendOutcome*>(arg));
      });
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "after" + std::to_string(p), "value"));
  }
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AppendOutcome");
  ASSERT_EQ(kPartitions, outcomes.size());
  for (size_t p = 0; p < kPartitions; ++p) {
    EXPECT_EQ(p, outcomes[p].wal_partition);
    EXPECT_EQ(dbfull()->TEST_CurrentWalNumber(p), outcomes[p].wal_number);
  }
  for (int round = 0; round < 3; ++round) {
    for (size_t p = 0; p < kPartitions; ++p) {
      ASSERT_EQ("value",
                Get(cf_for_partition_[p],
                    "key" + std::to_string(round) + "_" + std::to_string(p)));
    }
  }

  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForPurge());
  std::vector<uint64_t> current_wals;
  for (size_t p = 0; p < kPartitions; ++p) {
    current_wals.push_back(dbfull()->TEST_CurrentWalNumber(p));
    ASSERT_EQ(std::vector<uint64_t>{current_wals.back()}, AliveWalNumbers()[p]);
  }
  ASSERT_EQ(current_wals, WalNumbersOnDisk());
}

TEST_P(DBWritePartitionTest, ReopenPlacesUnpartitionedWalsInPartitionZero) {
  // R5: WALs written before partitioning was enabled carry no partition
  // (an unindexed file has no marker; an indexed one at one partition has an
  // empty one), so they return to partition 0. Partitions above 0 start with
  // only their new WAL, and retirement still empties partition 0.
  if (GetParam()) {
    return;
  }
  for (PartitionWALUsage usage :
       {PartitionWALUsage::kNone,
        PartitionWALUsage::kPartitionByColumnFamily}) {
    Options options = IndexedWalOptions(CurrentOptions());
    options.partition_wal_usage = usage;
    options.wal_partitions = 1;
    DestroyAndReopen(options);
    CreateAndReopenWithCF({"cf1", "cf2", "cf3", "cf4"}, options);
    for (int round = 0; round < 2; ++round) {
      if (round > 0) {
        ASSERT_OK(dbfull()->TEST_SwitchMemtable(
            static_cast_with_check<ColumnFamilyHandleImpl>(handles_[1])
                ->cfd()));
      }
      for (int cf = 1; cf < static_cast<int>(handles_.size()); ++cf) {
        ASSERT_OK(Put(cf, "key" + std::to_string(round), "value"));
      }
    }
    const std::vector<uint64_t> unpartitioned = AliveWalNumbers()[0];
    ASSERT_EQ(2U, unpartitioned.size());

    ReopenWithPartitions(2, IndexedWalOptions(CurrentOptions()));

    AssertPartitionDequesConsistent();
    std::vector<uint64_t> expected = unpartitioned;
    expected.push_back(dbfull()->TEST_CurrentWalNumber(0));
    ASSERT_EQ(expected, AliveWalNumbers()[0]);
    ASSERT_EQ(std::vector<uint64_t>{dbfull()->TEST_CurrentWalNumber(1)},
              AliveWalNumbers()[1]);
    for (int cf = 1; cf < static_cast<int>(handles_.size()); ++cf) {
      ASSERT_EQ("value", Get(cf, "key0"));
      ASSERT_EQ("value", Get(cf, "key1"));
    }

    FlushAllColumnFamilies();
    ASSERT_EQ(std::vector<uint64_t>{dbfull()->TEST_CurrentWalNumber(0)},
              AliveWalNumbers()[0]);
  }
}

TEST_P(DBWritePartitionTest, ReopenWithFewerPartitionsPlacesOrphansInZero) {
  // R5: a marker naming a partition this open does not have (written at
  // four partitions, reopened at two) sends that WAL to partition 0. The
  // deques stay ascending, so FIFO retirement still holds.
  if (GetParam()) {
    return;
  }
  const Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(4, options);
  ColumnFamilyData* rotating_cfd =
      static_cast_with_check<ColumnFamilyHandleImpl>(
          handles_[cf_for_partition_[0]])
          ->cfd();
  for (int round = 0; round < 2; ++round) {
    if (round > 0) {
      ASSERT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
    }
    for (size_t p = 0; p < 4; ++p) {
      ASSERT_OK(
          Put(cf_for_partition_[p], "key" + std::to_string(round), "value"));
    }
  }
  const std::vector<std::vector<uint64_t>> before = AliveWalNumbers();

  ReopenWithPartitions(2, options);

  AssertPartitionDequesConsistent();
  std::vector<uint64_t> expected_0 = before[0];
  expected_0.insert(expected_0.end(), before[2].begin(), before[2].end());
  expected_0.insert(expected_0.end(), before[3].begin(), before[3].end());
  std::sort(expected_0.begin(), expected_0.end());
  expected_0.push_back(dbfull()->TEST_CurrentWalNumber(0));
  std::vector<uint64_t> expected_1 = before[1];
  expected_1.push_back(dbfull()->TEST_CurrentWalNumber(1));
  const std::vector<std::vector<uint64_t>> after = AliveWalNumbers();
  ASSERT_EQ(expected_0, after[0]);
  ASSERT_EQ(expected_1, after[1]);

  FlushAllColumnFamilies();
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_EQ(std::vector<uint64_t>{dbfull()->TEST_CurrentWalNumber(p)},
              AliveWalNumbers()[p]);
  }
}

TEST_P(DBWritePartitionTest, CrashMidRotationUnderSkewRecoversPrefix) {
  // Crash recovery under skew with partitions at different rotation points:
  // two complete rotations of a skewed history (partition 1 hot, partition 2
  // cold, partition 0 idle), a synced write partway, then a rotation that
  // dies after partition 0's new WAL starts and before partition 1's does.
  // Unsynced-data loss is on, so the drop is real (N-7-4): the reopened
  // state must be a prefix of the write order that includes everything up
  // to the last synced write, and at least one acknowledged write past it
  // must be gone.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 3;
  constexpr int kWrites = 150;
  constexpr int kLastSynced = 119;
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  // Before the open: files capture this flag when created, and without it
  // DropUnsyncedFileData is a silent no-op and the test checks a clean
  // restart (found by D126193958).
  fault_fs->SetInjectUnsyncedDataLoss(true);
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_fs_env.get();
  OpenWithPartitions(kPartitions, options);
  const int hot = cf_for_partition_[1];
  const int cold = cf_for_partition_[2];
  ColumnFamilyData* hot_cfd =
      static_cast_with_check<ColumnFamilyHandleImpl>(handles_[hot])->cfd();

  auto cf_of = [&](int i) { return i % 10 == 9 ? cold : hot; };
  auto key_of = [](int i) { return "key" + std::to_string(1000 + i); };
  for (int i = 0; i < kWrites; ++i) {
    if (i == 50 || i == 100) {
      ASSERT_OK(dbfull()->TEST_SwitchMemtable(hot_cfd));
    }
    WriteOptions write_options;
    write_options.sync = i == kLastSynced;
    ASSERT_OK(Put(cf_of(i), key_of(i), "value", write_options));
  }

  int wal_starts = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::StartWALFile:AfterCompressionTypeRecord", [&](void*) {
        if (++wal_starts == 2) {
          fault_fs->SetFilesystemActive(false);
        }
      });
  ASSERT_NOK(dbfull()->TEST_SwitchMemtable(hot_cfd));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::StartWALFile:AfterCompressionTypeRecord");
  Close();
  ASSERT_OK(fault_fs->DropUnsyncedFileData());
  fault_fs->SetFilesystemActive(true);

  ReopenWithPartitions(kPartitions, options);

  AssertPartitionDequesConsistent();
  bool missing_seen = false;
  for (int i = 0; i < kWrites; ++i) {
    const bool present = Get(cf_of(i), key_of(i)) == "value";
    if (i <= kLastSynced) {
      ASSERT_TRUE(present) << i;
    }
    if (present) {
      ASSERT_FALSE(missing_seen) << i;
    } else {
      missing_seen = true;
    }
  }
  // Every write past the last sync was acknowledged and async, so the drop
  // must have taken at least one: without unsynced-data loss this is false
  // and the test checks a clean restart.
  EXPECT_TRUE(missing_seen) << "the crash dropped no acknowledged write";
  Close();
}

TEST_P(DBWritePartitionTest, SyncWriteMakesPriorPartitionAppendSurviveCrash) {
  // Q10 ([08-4]): a sync=true write's predecessors on other partitions are
  // durable when it returns: shown on the filesystem, not inferred from
  // SyncWAL call counts. Same orchestration as SyncWriteSyncsAgainOnlyAfter-
  // Wait phase 2 (held async A on partition 0, sync B on partition 1, so
  // only the gate's post-wait sync can make A durable), then all unsynced
  // file data is dropped (simulated OS crash) and the DB recovers with WAL
  // indexing on. Both keys come back. A later unsynced control write must
  // NOT come back, proving the crash actually dropped unsynced data.
  // Single-queue only: under two queues B-sync would block on A's held
  // partition mutex before ever appending (see the SyncWriteSyncsAgainOnly-
  // AfterWait comment), so the contended interleaving cannot be scripted
  // there; the gate and post-wait sync are shared code. Disabling the
  // post-wait sync loses key_a and fails this test.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  // Close before `fault_env` goes out of scope on every exit path, including
  // a fatal assertion's early return.
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;

  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool b_appended = false;
  std::atomic<bool> b_returned{false};
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            // A is still held before its
                                            // AddRecord, so this is B's.
                                            b_appended = true;
                                            cv.notify_all();
                                          }
                                        });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_b", "value_b", sync_wo);
    b_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  // B parks only after its leadership fsync and pre-sync snapshot, so A's
  // append below is covered by neither.
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  ASSERT_OK(b_status);
  ASSERT_TRUE(b_returned.load());
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");

  // Control: written after B returned and never synced; the crash drops it.
  ASSERT_OK(Put(cf_a, "key_control", "value_control"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);

  ASSERT_EQ("value_a", Get(cf_for_partition_[0], "key_a"));
  ASSERT_EQ("value_b", Get(cf_for_partition_[1], "key_b"));
  ASSERT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key_control"));
}

TEST_P(DBWritePartitionTest, MalformedBatchFailsAsAtN1) {
  // G-19b: routing never fails a write; a batch whose records stop parsing
  // fails on the normal write path with the same status, and leaves the
  // same background error, at N=4 as at N=1. Restoring the router's error
  // return fails it before the append, with no background error.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  auto write_damaged = [this](size_t partitions) {
    Options options = CurrentOptions();
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    WriteBatch valid;
    EXPECT_OK(WriteBatchInternal::Put(&valid, /*cf=*/1, "k", "v"));
    std::string rep = valid.Data();
    rep.push_back('\x7f');
    WriteBatch damaged;
    EXPECT_OK(WriteBatchInternal::SetContents(&damaged, rep));
    const Status s = db_->Write(WriteOptions(), &damaged);
    const Status bg_error = dbfull()->TEST_GetBGError();
    Close();
    return std::make_pair(s.ToString(), bg_error.ToString());
  };
  const auto at_one = write_damaged(1);
  const auto at_four = write_damaged(4);
  EXPECT_NE("OK", at_one.first);
  EXPECT_EQ(at_one.first, at_four.first);
  EXPECT_EQ(at_one.second, at_four.second);
}

TEST_P(DBWritePartitionTest, SyncWriteAboveAsyncWriteSurvivesCrash) {
  // Async A on partition 0, then sync B on partition 1. B fsyncs every
  // partition's WAL before its group publishes, so after a crash that
  // drops every unsynced byte both come back; a later async write on
  // partition 0 does not, proving the crash dropped unsynced data. Without
  // the gate's fsync both A and B are lost.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "key_a", "value_a"));
  ASSERT_OK(Put(cf_for_partition_[1], "key_b", "value_b", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[0], "key_control", "value_control"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("value_a", Get(cf_for_partition_[0], "key_a"));
  EXPECT_EQ("value_b", Get(cf_for_partition_[1], "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key_control"));
}

TEST_P(DBWritePartitionTest, PartialGateSyncCutLosesHigherAsyncWrite) {
  // D-G11 (c) accepted risk: records in one partition's WAL that follow a
  // lower record never fsynced in another partition's WAL exist on disk
  // but cannot be used; recovery cuts at the hole. D (async) on partition
  // 2, then R (async) on partition 0 above D; a sync write's gate sync
  // then fails after fsyncing partitions 0 and 1, so R's bytes are on
  // disk while D's are not. After the crash the recovered state is a
  // prefix: A (below the hole, fsynced) survives, while D and the
  // acknowledged R are both gone. Ported from GC17's Muse candidate.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  // Close before `fault_env` goes out of scope on every exit path, including
  // a fatal assertion's early return.
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_d = cf_for_partition_[2];
  const int cf_s = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  // Baseline sync: every partition's WAL file exists on disk synced, so
  // the crash below truncates partition 2's WAL at a record boundary.
  ASSERT_OK(Put(cf_a, "key_s0", "value_s0", sync_wo));
  ASSERT_OK(Put(cf_a, "key_a", "value_a"));
  ASSERT_OK(Put(cf_d, "key_d", "value_d"));
  ASSERT_OK(Put(cf_a, "key_r", "value_r"));

  // Fail the next sync write's gate sync inside partition 2's slice, after
  // partitions 0 and 1 (holding A and R) really fsynced. The filesystem
  // goes inactive only for that slice; nothing else runs concurrently.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) == 2) {
          fault_env->SetFilesystemActive(false);
        }
      });
  Status s_status = Put(cf_s, "key_s", "value_s", sync_wo);
  fault_env->SetFilesystemActive(true);
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  ASSERT_NOK(s_status);
  EXPECT_TRUE(s_status.IsIOError());
  EXPECT_FALSE(dbfull()->TEST_GetBGError().ok());
  dbfull()->Resume().PermitUncheckedError();
  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());

  // Default recovery fails closed on the unexplained hole (pinned
  // behavior); point-in-time recovery cuts at it and opens the prefix.
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = static_cast<uint32_t>(kPartitions);
  const Status default_reopen_status =
      TryReopenWithColumnFamilies(cf_names, options);
  ASSERT_TRUE(default_reopen_status.IsCorruption())
      << default_reopen_status.ToString();
  EXPECT_NE(std::string::npos,
            default_reopen_status.ToString().find("unexplained hole"));
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ReopenWithPartitions(kPartitions, options);

  ASSERT_EQ("value_s0", Get(cf_a, "key_s0"));
  ASSERT_EQ("value_a", Get(cf_a, "key_a"));
  ASSERT_EQ("NOT_FOUND", Get(cf_d, "key_d"));
  ASSERT_EQ("NOT_FOUND", Get(cf_a, "key_r"));
  ASSERT_EQ("NOT_FOUND", Get(cf_s, "key_s"));
}

TEST_P(DBWritePartitionTest, CoalescedGateSyncsFsyncEachPartitionOnce) {
  // Sync writes on several partitions share gate syncs. S1's gate sync is
  // held before it fsyncs anything; sync writes on every partition, all above
  // S1, queue behind it in the coalescer. Once S1 is released exactly two
  // rounds run, S1's and one that covers every queued write. GC28 Q1: S1's
  // claim ran before the queued writes appended, when only partition 0 held
  // an unsynced record, so S1's round fsyncs partition 0 only; the covering
  // round fsyncs every partition once. Every write is acknowledged, and all
  // of them survive a crash that drops every unsynced byte. Neuter: a round
  // that skips a partition loses the queued write on it in the crash.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  OpenWithPartitions(kPartitions, options);

  std::mutex mu;
  std::condition_variable cv;
  bool s1_in_sync = false;
  bool release_s1 = false;
  size_t queued = 0;
  std::vector<int> fsyncs(kPartitions, 0);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!s1_in_sync) {
          s1_in_sync = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_s1; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:BeforeWait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        ++queued;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        ++fsyncs[*static_cast<size_t*>(arg)];
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s1_status;
  port::Thread s1(
      [&] { s1_status = Put(cf_for_partition_[0], "s1", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return s1_in_sync; }));
  }
  std::vector<Status> statuses(kPartitions);
  std::vector<port::Thread> writers;
  for (size_t p = 0; p < kPartitions; ++p) {
    writers.emplace_back([&, p] {
      statuses[p] =
          Put(cf_for_partition_[p], "q" + std::to_string(p), "v", sync_wo);
    });
  }
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return queued == kPartitions; }));
    release_s1 = true;
    cv.notify_all();
  }
  s1.join();
  for (port::Thread& writer : writers) {
    writer.join();
  }
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_OK(s1_status);
  for (const Status& s : statuses) {
    ASSERT_OK(s);
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    // S1's round fsyncs partition 0 only (GC28 Q1 targeting); the covering
    // round fsyncs every partition once.
    EXPECT_EQ(p == 0 ? 2 : 1, fsyncs[p]) << "partition " << p;
  }

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s1"));
  for (size_t p = 0; p < kPartitions; ++p) {
    EXPECT_EQ("v", Get(cf_for_partition_[p], "q" + std::to_string(p)));
  }
}

// GC28 Q1-Q4, Q6: the gate sync fsyncs only the WAL files a sync write
// depends on, once, without holding partition locks across I/O.
TEST_P(DBWritePartitionTest, TargetedGateSyncSkipsNewerOnlyPartitions) {
  // GC28 Q1 (F2): partition 1 holds an unsynced record below the sync
  // write's target T, partition 2 only unsynced records above T (allocated
  // after the sync write parks in its gate, appended before its claim), and
  // partition 3 is clean. The sync write on partition 0 fsyncs exactly
  // partitions 0 and 1. Both histograms agree.
  // Neuter: claim every partition; partitions 2 and 3 are fsynced too.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  ASSERT_OK(Put(cf_for_partition_[1], "a1", "v"));
  ASSERT_OK(options.statistics->Reset());

  std::mutex mu;
  std::condition_variable cv;
  bool s_parked = false;
  bool release_s = false;
  bool a2_appended = false;
  std::vector<int> fsyncs(kPartitions, 0);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!s_parked) {
          s_parked = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_s; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        ++fsyncs[*static_cast<size_t*>(arg)];
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s_status;
  port::Thread s_thread(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return s_parked; }));
  }
  // S allocated its target before parking; A2 allocates above it, and its
  // append lands before S's claim. A2 parks in its own gate (above an
  // unpublished sync) and returns once S publishes.
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          a2_appended = true;
                                          cv.notify_all();
                                        });
  Status a2_status;
  port::Thread a2_thread(
      [&] { a2_status = Put(cf_for_partition_[2], "a2", "v"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return a2_appended; }));
    release_s = true;
    cv.notify_all();
  }
  s_thread.join();
  a2_thread.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::CompletionGate:BeforeCoalescedSync");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  ASSERT_OK(s_status);
  ASSERT_OK(a2_status);
  EXPECT_EQ(1, fsyncs[0]) << "own partition";
  EXPECT_EQ(1, fsyncs[1]) << "below-T record";
  EXPECT_EQ(0, fsyncs[2]) << "only above-T records";
  EXPECT_EQ(0, fsyncs[3]) << "clean";
  HistogramData fsynced;
  options.statistics->histogramData(WAL_SYNC_FSYNCED_PARTITIONS, &fsynced);
  EXPECT_EQ(1U, fsynced.count);
  EXPECT_EQ(2U, fsynced.sum);
  HistogramData unsynced;
  options.statistics->histogramData(WAL_SYNC_UNSYNCED_PARTITIONS, &unsynced);
  EXPECT_EQ(1U, unsynced.count);
  EXPECT_EQ(2U, unsynced.sum);
}

TEST_P(DBWritePartitionTest, TargetedGateSyncCoversRotatedUnsyncedWal) {
  // GC28 Q1 (F2): an old, rotated WAL holds an unsynced record below the
  // sync write's target while the current WAL is clean. The targeted claim
  // still fsyncs that partition, and a power loss keeps the old record.
  // Neuter: claim only the current WAL; the old record is lost.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "a" + std::to_string(p), "v"));
  }
  std::vector<uint64_t> before(kPartitions);
  for (size_t p = 0; p < kPartitions; ++p) {
    before[p] = dbfull()->TEST_CurrentWalNumber(p);
  }
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  size_t rotated = 0;
  for (size_t p = 0; p < kPartitions; ++p) {
    rotated += dbfull()->TEST_CurrentWalNumber(p) != before[p] ? 1 : 0;
  }
  ASSERT_GT(rotated, 0U) << "rotation created no new WAL";

  std::mutex mu;
  std::vector<int> fsyncs(kPartitions, 0);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        ++fsyncs[*static_cast<size_t*>(arg)];
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s", "v", sync_wo));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  for (size_t p = 0; p < kPartitions; ++p) {
    if (dbfull()->TEST_CurrentWalNumber(p) != before[p]) {
      EXPECT_GE(fsyncs[p], 1) << "rotated partition " << p;
    }
  }
  // Never synced: the power loss below must drop it.
  ASSERT_OK(Put(cf_for_partition_[1], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    EXPECT_EQ("v", Get(cf_for_partition_[p], "a" + std::to_string(p)))
        << "partition " << p;
  }
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "control"));
}

TEST_P(DBWritePartitionTest, TargetedGateSyncLeavesLaterAppendUnsynced) {
  // GC28 Q1 (F2): a record appended to a claimed partition after its size
  // was captured is never reported as synced: a power loss drops it, and a
  // later sync write fsyncs the file again.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  std::mutex mu;
  std::condition_variable cv;
  bool s_in_fsync = false;
  bool release_s = false;
  bool e_appended = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) != 0) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!s_in_fsync) {
          s_in_fsync = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_s; });
        }
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s_status;
  port::Thread s_thread(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return s_in_fsync; }));
  }
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          e_appended = true;
                                          cv.notify_all();
                                        });
  Status e_status;
  port::Thread e_thread(
      [&] { e_status = Put(cf_for_partition_[0], "e", "v"); });
  {
    // At the base the append waits for the fsync (partition mutex held
    // across it), so the deadline passes and the record lands after the
    // release; either way it is never reported as synced.
    std::unique_lock<std::mutex> lock(mu);
    cv.wait_for(lock, std::chrono::seconds(2), [&] { return e_appended; });
    release_s = true;
    cv.notify_all();
  }
  s_thread.join();
  e_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  ASSERT_OK(s_status);
  ASSERT_OK(e_status);

  // A later sync write syncs the file again.
  std::atomic<int> p0_fsyncs{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) == 0) {
          ++p0_fsyncs;
        }
      });
  ASSERT_OK(Put(cf_for_partition_[1], "s2", "v", sync_wo));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  EXPECT_GE(p0_fsyncs.load(), 1);

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "e"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "s2"));
}

TEST_P(DBWritePartitionTest, TargetedCoalescingNeedsCoveringClaim) {
  // GC28 Q1 (F2): both sync writes append before either claims, with
  // T1 < T2. S1's sync skips S2's partition (its only unsynced record is
  // above T1), so S2 must not return on S1: it runs its own sync, and S2
  // survives a power loss. When S2's files were all claimed by S1 instead
  // (a lower async record shares S2's partition), no new fsync runs.
  // Neuter: epoch-only coverage; S2 returns covered by S1 and loses its
  // record.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  for (int covered_case = 0; covered_case < 2; ++covered_case) {
    std::unique_ptr<FaultInjectionTestEnv> fault_env(
        new FaultInjectionTestEnv(env_));
    Defer close_before_env([this] { Close(); });
    Options options = IndexedWalOptions(CurrentOptions());
    options.env = fault_env.get();
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    if (covered_case) {
      ASSERT_OK(Put(cf_for_partition_[2], "a", "v"));
    }

    std::mutex mu;
    std::condition_variable cv;
    int installs = 0;
    bool release_s1 = false;
    bool release_s2 = false;
    bool s1_in_fsync = false;
    bool release_fsync = false;
    bool s2_waiting = false;
    std::atomic<int> gate_syncs{0};
    std::vector<int> fsyncs(kPartitions, 0);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::InstallCompletionGate", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (++installs == 1) {
            cv.notify_all();
            cv.wait(lock, [&] { return release_s1; });
          } else if (installs == 2) {
            cv.notify_all();
            cv.wait(lock, [&] { return release_s2; });
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
          std::unique_lock<std::mutex> lock(mu);
          if (*static_cast<size_t*>(arg) == 0 && !s1_in_fsync) {
            s1_in_fsync = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_fsync; });
          } else {
            ++fsyncs[*static_cast<size_t*>(arg)];
          }
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::RunCoalescedGateSync:BeforeWait", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          s2_waiting = true;
          cv.notify_all();
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalsFromCompletionGate:AfterGateSync",
        [&](void*) { ++gate_syncs; });

    WriteOptions sync_wo;
    sync_wo.sync = true;
    Status s1_status;
    port::Thread s1_thread(
        [&] { s1_status = Put(cf_for_partition_[0], "s1", "v", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return installs >= 1; }));
    }
    Status s2_status;
    port::Thread s2_thread(
        [&] { s2_status = Put(cf_for_partition_[2], "s2", "v", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return installs >= 2; }));
      release_s1 = true;
      cv.notify_all();
    }
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return s1_in_fsync; }));
      release_s2 = true;
      cv.notify_all();
    }
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return s2_waiting; }));
      release_fsync = true;
      cv.notify_all();
    }
    s1_thread.join();
    s2_thread.join();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    ASSERT_OK(s1_status);
    ASSERT_OK(s2_status);
    if (!covered_case) {
      EXPECT_EQ(2, gate_syncs.load())
          << "S2 runs its own sync; S1 skipped its partition";
    } else {
      EXPECT_EQ(1, gate_syncs.load()) << "S1 covered S2's files";
    }

    Close();
    ASSERT_OK(fault_env->DropUnsyncedFileData());
    ReopenWithPartitions(kPartitions, options);
    EXPECT_EQ("v", Get(cf_for_partition_[0], "s1"));
    EXPECT_EQ("v", Get(cf_for_partition_[2], "s2"));
    if (covered_case) {
      EXPECT_EQ("v", Get(cf_for_partition_[2], "a"));
    }
    Close();
  }
}

TEST_P(DBWritePartitionTest, TargetedZeroCountSyncSyncsItsOwnFile) {
  // GC28 Q1 (F2, G3-D-1): a sync Put on partition 1 (target T), then a
  // LogData-only sync write (keyless batches route to partition 0) whose
  // file holds only that record, stamped T+1. Partition 0 is fsynced, the
  // record survives a power loss, and an unsynced control record written
  // after it is lost. (Partition roles are swapped from the issue text
  // because LogData-only batches always route to partition 0.)
  // Neuter: claim by "first unsynced sequence <= T" alone; partition 0 is
  // skipped and the record is lost.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[1], "key", "value", sync_wo));

  std::atomic<int> p0_fsyncs{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) == 0) {
          ++p0_fsyncs;
        }
      });
  WriteBatch log_data_only;
  ASSERT_OK(log_data_only.PutLogData("blob"));
  ASSERT_OK(db_->Write(sync_wo, &log_data_only));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  EXPECT_GE(p0_fsyncs.load(), 1) << "zero-count group syncs its own file";

  // Never synced: the power loss below must drop it.
  WriteBatch control;
  ASSERT_OK(control.PutLogData("control"));
  ASSERT_OK(db_->Write(WriteOptions(), &control));

  std::vector<uint64_t> wals;
  for (size_t p = 0; p < kPartitions; ++p) {
    wals.push_back(dbfull()->TEST_CurrentWalNumber(p));
  }
  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  bool found_record = false;
  bool found_control = false;
  for (const uint64_t wal : wals) {
    std::unique_ptr<SequentialFileReader> file_reader;
    ASSERT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), LogFileName(dbname_, wal), FileOptions(),
        &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, wal);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
      found_record |= record.ToString().find("blob") != std::string::npos;
      found_control |= record.ToString().find("control") != std::string::npos;
    }
  }
  EXPECT_TRUE(found_record);
  EXPECT_FALSE(found_control);
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("value", Get(cf_for_partition_[1], "key"));
}

TEST_P(DBWritePartitionTest, TargetedSyncInvisibleUntilNeededFsyncDone) {
  // GC28 Q1 (F2, D-9): partition 1 holds a lower async record the sync
  // write needs. While that partition's fsync is held, the sync write stays
  // invisible (Get misses, published sequence unchanged); once released it
  // returns and becomes visible.
  // Neuter: claim only the writer's own partition; the write completes
  // while the needed fsync is still held.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  ASSERT_OK(Put(cf_for_partition_[1], "a", "v"));
  const SequenceNumber published_before = db_->GetLatestSequenceNumber();

  std::mutex mu;
  std::condition_variable cv;
  bool q_in_fsync = false;
  bool release_q = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) != 1) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!q_in_fsync) {
          q_in_fsync = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_q; });
        }
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s_status = Status::Incomplete("not started");
  std::atomic<bool> s_returned{false};
  port::Thread s_thread([&] {
    s_status = Put(cf_for_partition_[0], "s", "v", sync_wo);
    s_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return q_in_fsync; }));
  }
  // Room for the write to (wrongly) publish while the needed fsync is held.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(s_returned.load());
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ(published_before, db_->GetLatestSequenceNumber());
  {
    std::lock_guard<std::mutex> lock(mu);
    release_q = true;
    cv.notify_all();
  }
  s_thread.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  ASSERT_OK(s_status);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
}

TEST_P(DBWritePartitionTest,
       TargetedManualFlushSyncLeavesNewerBuffersBuffered) {
  // GC28 Q1 (F2, D-F2d): under manual_wal_flush, a sync write flushes only
  // the claimed partitions' buffers. A1 (below T) is flushed and survives a
  // power loss; A2's buffer (above T, appended before the claim) stays
  // buffered.
  // Neuter: flush every partition; partition 2's buffer comes back empty.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.manual_wal_flush = true;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  ASSERT_OK(Put(cf_for_partition_[1], "a1", "v"));

  std::mutex mu;
  std::condition_variable cv;
  bool s_parked = false;
  bool release_s = false;
  bool a2_appended = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!s_parked) {
          s_parked = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_s; });
        }
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s_status;
  port::Thread s_thread(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return s_parked; }));
  }
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          a2_appended = true;
                                          cv.notify_all();
                                        });
  Status a2_status;
  port::Thread a2_thread(
      [&] { a2_status = Put(cf_for_partition_[2], "a2", "v"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return a2_appended; }));
    release_s = true;
    cv.notify_all();
  }
  s_thread.join();
  a2_thread.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::CompletionGate:BeforeCoalescedSync");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  ASSERT_OK(s_status);
  ASSERT_OK(a2_status);
  EXPECT_TRUE(dbfull()->TEST_WalBufferEmpty(0));
  EXPECT_TRUE(dbfull()->TEST_WalBufferEmpty(1));
  EXPECT_FALSE(dbfull()->TEST_WalBufferEmpty(2))
      << "newer buffer stays buffered";
  EXPECT_TRUE(dbfull()->TEST_WalBufferEmpty(3));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "a1"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[2], "a2"));
}

TEST_P(DBWritePartitionTest, FlushWalTakesOnePartitionMutexAtATime) {
  // GC28 Q1 (F2, D-F2d): while FlushWAL() is blocked inside partition 3's
  // buffer flush (an append pins that partition's WAL mutex), an async
  // write on partition 1 appends and returns.
  // Neuter: the all-N lock; the write never completes (RunWithin aborts).
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.manual_wal_flush = true;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  std::mutex mu;
  std::condition_variable cv;
  bool e_parked = false;
  bool release_e = false;
  bool e_appended = false;
  bool a_pinned = false;
  bool release_a = false;
  bool first_log_entry = true;
  bool first_append = true;
  // E (async, partition 1) parks before its append, pinning no mutex; A
  // (async, partition 3) pins its partition's mutex inside its append.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_log_entry) {
          first_log_entry = false;
          e_parked = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_e; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_pinned = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        } else {
          e_appended = true;
          cv.notify_all();
        }
      });

  Status e_status = Status::Incomplete("not started");
  std::atomic<bool> e_returned{false};
  port::Thread e_thread([&] {
    e_status = Put(cf_for_partition_[1], "e", "v");
    e_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return e_parked; }));
  }
  Status a_status;
  port::Thread a_thread(
      [&] { a_status = Put(cf_for_partition_[3], "a", "v"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return a_pinned; }));
  }
  // FlushWAL takes partitions in order and blocks on partition 3's pinned
  // mutex; at the base it holds partitions 0-2 meanwhile.
  Status f_status;
  port::Thread f_thread([&] { f_status = db_->FlushWAL(false); });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_e = true;
    cv.notify_all();
  }
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    e_thread.join();
    return e_status;
  }));
  EXPECT_TRUE(e_appended);
  EXPECT_TRUE(e_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  f_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  ASSERT_OK(e_status);
  ASSERT_OK(a_status);
  ASSERT_OK(f_status);
  EXPECT_EQ("v", Get(cf_for_partition_[1], "e"));
}

TEST_P(DBWritePartitionTest, TargetedSyncRetiresSkippedPartitionWals) {
  // GC28 Q1 (F2): partitions the targeted gate keeps skipping still retire
  // their closed WALs once synced. Each round writes async on partition 3,
  // syncs from partition 0 (covering partition 3), then rotates; extra
  // syncs on partition 0 skip the clean partition 3. Its deque stays
  // bounded either way.
  // Neuter: never retire an unclaimed WAL; logs_[3] grows every round.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  // Gate syncs retire fully-synced inactive WALs only with this option; with
  // it, the base retires every round's leftover, so the bound below holds at
  // the base and fails if targeting never retires an unclaimed WAL.
  options.background_close_inactive_wals = true;
  OpenWithPartitions(kPartitions, options);

  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int round = 0; round < 3; ++round) {
    ASSERT_OK(Put(cf_for_partition_[3], "a" + std::to_string(round), "v"));
    ASSERT_OK(
        Put(cf_for_partition_[0], "s" + std::to_string(round), "v", sync_wo));
    const uint64_t before = dbfull()->TEST_CurrentWalNumber(3);
    ASSERT_OK(dbfull()->TEST_SwitchMemtable());
    ASSERT_NE(before, dbfull()->TEST_CurrentWalNumber(3))
        << "round " << round << ": rotation created no new WAL on partition 3";
  }
  ASSERT_OK(Put(cf_for_partition_[0], "s3", "v", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[0], "s4", "v", sync_wo));
  EXPECT_LE(dbfull()->TEST_NumOpenWals(3), 2U);
  for (int round = 0; round < 3; ++round) {
    EXPECT_EQ("v", Get(cf_for_partition_[3], "a" + std::to_string(round)));
  }
}

TEST_P(DBWritePartitionTest,
       ManualFlushAppendProceedsWhileOtherPartitionFsyncHeld) {
  // GC28 Q2 (FU3, D-6): under manual_wal_flush, with the sync write's fsync
  // of partition q held, an async write on r != q completes its append. It
  // stays invisible and unreturned until the sync publishes; then both
  // return OK.
  // Neuter: today's LockWalWriteMutexes() branch; the append never lands
  // (RunWithin aborts).
  if (GetParam()) {
    return;
  }
  for (size_t kPartitions : {2, 8}) {
    Options options = IndexedWalOptions(CurrentOptions());
    options.manual_wal_flush = true;
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    const size_t q = 0;
    const size_t r = kPartitions - 1;

    std::mutex mu;
    std::condition_variable cv;
    bool s_in_fsync = false;
    bool release_s = false;
    bool e_appended = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
          if (*static_cast<size_t*>(arg) != q) {
            return;
          }
          std::unique_lock<std::mutex> lock(mu);
          if (!s_in_fsync) {
            s_in_fsync = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_s; });
          }
        });

    WriteOptions sync_wo;
    sync_wo.sync = true;
    Status s_status;
    port::Thread s_thread(
        [&] { s_status = Put(cf_for_partition_[q], "s", "v", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return s_in_fsync; }))
          << "N=" << kPartitions;
    }
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          e_appended = true;
          cv.notify_all();
        });
    Status e_status = Status::Incomplete("not started");
    std::atomic<bool> e_returned{false};
    port::Thread e_thread([&] {
      e_status = Put(cf_for_partition_[r], "e", "v");
      e_returned.store(true);
    });
    ASSERT_OK(RunWithin(std::chrono::seconds(30),
                        [&] {
                          std::unique_lock<std::mutex> lock(mu);
                          cv.wait(lock, [&] { return e_appended; });
                          return Status::OK();
                        }))
        << "N=" << kPartitions;
    EXPECT_FALSE(e_returned.load()) << "N=" << kPartitions;
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[r], "e"))
        << "N=" << kPartitions;
    {
      std::lock_guard<std::mutex> lock(mu);
      release_s = true;
      cv.notify_all();
    }
    s_thread.join();
    e_thread.join();
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord");
    ASSERT_OK(s_status) << "N=" << kPartitions;
    ASSERT_OK(e_status) << "N=" << kPartitions;
    EXPECT_EQ("v", Get(cf_for_partition_[r], "e")) << "N=" << kPartitions;
    Close();
  }
}

TEST_P(DBWritePartitionTest,
       AsyncAppendOnSyncingPartitionProceedsWhileItsFsyncHeld) {
  // GC28 Q2 (FU3, D-6): with the sync write's fsync of partition q held,
  // an async write on q itself completes its append (the partition mutex is
  // no longer held across the fsync). It stays invisible and unreturned
  // until the sync publishes; then both return OK. Replaces
  // AsyncOnSyncingPartitionWaitsForItsFsync: the append no longer waits,
  // the return still waits for the sync's publication (D-9).
  // Neuter: hold p's mutex across p's fsync; the append never lands
  // (RunWithin aborts).
  if (GetParam()) {
    return;
  }
  for (bool manual_flush : {false, true}) {
    Options options = IndexedWalOptions(CurrentOptions());
    options.manual_wal_flush = manual_flush;
    options.avoid_flush_during_shutdown = true;
    constexpr size_t kPartitions = 4;
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    const size_t q = 1;

    std::mutex mu;
    std::condition_variable cv;
    bool s_in_fsync = false;
    bool release_s = false;
    bool e_appended = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
          if (*static_cast<size_t*>(arg) != q) {
            return;
          }
          std::unique_lock<std::mutex> lock(mu);
          if (!s_in_fsync) {
            s_in_fsync = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_s; });
          }
        });

    WriteOptions sync_wo;
    sync_wo.sync = true;
    Status s_status;
    port::Thread s_thread(
        [&] { s_status = Put(cf_for_partition_[q], "s", "v", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                              [&] { return s_in_fsync; }))
          << "manual=" << manual_flush;
    }
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          e_appended = true;
          cv.notify_all();
        });
    Status e_status = Status::Incomplete("not started");
    std::atomic<bool> e_returned{false};
    port::Thread e_thread([&] {
      e_status = Put(cf_for_partition_[q], "e", "v");
      e_returned.store(true);
    });
    ASSERT_OK(RunWithin(std::chrono::seconds(30),
                        [&] {
                          std::unique_lock<std::mutex> lock(mu);
                          cv.wait(lock, [&] { return e_appended; });
                          return Status::OK();
                        }))
        << "manual=" << manual_flush;
    EXPECT_FALSE(e_returned.load()) << "manual=" << manual_flush;
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[q], "e"))
        << "manual=" << manual_flush;
    {
      std::lock_guard<std::mutex> lock(mu);
      release_s = true;
      cv.notify_all();
    }
    s_thread.join();
    e_thread.join();
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord");
    ASSERT_OK(s_status) << "manual=" << manual_flush;
    ASSERT_OK(e_status) << "manual=" << manual_flush;
    EXPECT_EQ("v", Get(cf_for_partition_[q], "e")) << "manual=" << manual_flush;
    Close();
  }
}

TEST_P(DBWritePartitionTest,
       NonSyncThreadSafeWalHoldsOnePartitionMutexAcrossFsync) {
  // GC28 Q2 (FU3, D-6 fallback): with WAL files that are not
  // IsSyncThreadSafe(), the gate sync holds only the syncing partition's
  // mutex across its fsync. An append on the syncing partition waits; an
  // append on another partition proceeds. Never all N.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  auto non_safe_fs =
      std::make_shared<NonSyncThreadSafeFS>(env_->GetFileSystem());
  std::unique_ptr<Env> non_safe_env(NewCompositeEnv(non_safe_fs));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = non_safe_env.get();
  options.manual_wal_flush = true;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const size_t q = 0;
  const size_t r = 1;

  std::mutex mu;
  std::condition_variable cv;
  bool s_in_fsync = false;
  bool release_s = false;
  bool q_appended = false;
  bool r_appended = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) != q) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!s_in_fsync) {
          s_in_fsync = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_s; });
        }
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s_status;
  port::Thread s_thread(
      [&] { s_status = Put(cf_for_partition_[q], "s", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return s_in_fsync; }));
  }
  // Set after S is held (S already appended). ER (other partition) appends
  // first; EQ must not append while the fallback holds q's mutex.
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!r_appended) {
                                            r_appended = true;
                                          } else {
                                            q_appended = true;
                                          }
                                          cv.notify_all();
                                        });
  // ER starts first so the append callback attributes the first append to
  // it; EQ's append must not land while the fallback holds q's mutex.
  Status er_status;
  port::Thread er_thread(
      [&] { er_status = Put(cf_for_partition_[r], "er", "v"); });
  Status eq_status;
  port::Thread eq_thread(
      [&] { eq_status = Put(cf_for_partition_[q], "eq", "v"); });
  {
    // EXPECT, not ASSERT: at the base ER never appends (all-N hold), and
    // the test must still release S and join every thread below.
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30), [&] {
      return r_appended;
    })) << "append on another partition proceeds";
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(q_appended) << "append on syncing partition waits";
  {
    std::lock_guard<std::mutex> lock(mu);
    release_s = true;
    cv.notify_all();
  }
  s_thread.join();
  er_thread.join();
  eq_thread.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  ASSERT_OK(s_status);
  ASSERT_OK(er_status);
  ASSERT_OK(eq_status);
}

TEST_P(DBWritePartitionTest,
       GateSyncFlushesBufferedLowerRecordsBeforeUnlockedFsync) {
  // GC28 Q2 (FU3, D-6 step 1): under manual_wal_flush, buffered async
  // records below the sync write, on two partitions, are flushed before
  // the unlocked fsync and survive a power loss.
  // Neuter: skip step 1's buffer flush; the records are lost.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.manual_wal_flush = true;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  ASSERT_OK(Put(cf_for_partition_[1], "a1", "v"));
  ASSERT_OK(Put(cf_for_partition_[2], "a2", "v"));
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s", "v", sync_wo));
  // Never synced: the power loss below must drop it.
  ASSERT_OK(Put(cf_for_partition_[1], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "a1"));
  EXPECT_EQ("v", Get(cf_for_partition_[2], "a2"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "control"));
}

TEST_P(DBWritePartitionTest, UnlockedFsyncRacesWithSyncCallersAndRotation) {
  // GC28 Q2 (FU3, D-6): FlushWAL(), FlushWAL(true), SyncWAL(),
  // LockWAL()/UnlockWAL() and a forced rotation, each released between
  // steps 1-2 and between steps 2-3 of a held gate sync, never corrupt a
  // buffer or file and retire nothing mid-fsync. A reopen in
  // kAbsoluteConsistency reads every acknowledged write, and a power loss
  // drops the unflushed control, proving nothing was marked synced beyond
  // what was flushed.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  const std::vector<std::string> racers = {"flush", "flush_sync", "sync_wal",
                                           "lock_wal", "rotation"};
  const std::vector<std::string> gaps = {"between_claim_and_fsync",
                                         "between_fsync_and_marks"};
  int scenario = 0;
  for (const std::string& racer : racers) {
    for (const std::string& gap : gaps) {
      const std::string tag =
          racer + "/" + gap + "/" + std::to_string(scenario);
      std::unique_ptr<FaultInjectionTestEnv> fault_env(
          new FaultInjectionTestEnv(env_));
      Defer close_before_env([this] { Close(); });
      Options options = IndexedWalOptions(CurrentOptions());
      options.env = fault_env.get();
      options.manual_wal_flush = true;
      options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
      options.avoid_flush_during_shutdown = true;
      DestroyAndReopen(options);
      OpenWithPartitions(kPartitions, options);

      ASSERT_OK(Put(cf_for_partition_[1], "a" + tag, "v"));

      std::mutex mu;
      std::condition_variable cv;
      bool s_held = false;
      bool release_s = false;
      const char* hold_point =
          gap == "between_claim_and_fsync"
              ? "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition"
              : "DBImpl::SyncWalsFromCompletionGate:AfterGateSync";
      SyncPoint::GetInstance()->SetCallBack(hold_point, [&](void* arg) {
        if (gap == "between_claim_and_fsync" &&
            *static_cast<size_t*>(arg) != 0) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!s_held) {
          s_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_s; });
        }
      });

      WriteOptions sync_wo;
      sync_wo.sync = true;
      Status s_status;
      port::Thread s_thread([&] {
        s_status = Put(cf_for_partition_[0], "s" + tag, "v", sync_wo);
      });
      {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30), [&] {
          return s_held;
        })) << tag;
      }
      Status racer_status = Status::Incomplete("not started");
      port::Thread racer_thread([&] {
        if (racer == "flush") {
          racer_status = db_->FlushWAL(false);
        } else if (racer == "flush_sync") {
          racer_status = db_->FlushWAL(true);
        } else if (racer == "sync_wal") {
          racer_status = db_->SyncWAL();
        } else if (racer == "lock_wal") {
          racer_status = db_->LockWAL();
          if (racer_status.ok()) {
            racer_status = db_->UnlockWAL();
          }
        } else {
          racer_status = dbfull()->TEST_SwitchMemtable();
        }
      });
      // Room for the racer to reach its block point (or finish) mid-sync.
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      {
        std::lock_guard<std::mutex> lock(mu);
        release_s = true;
        cv.notify_all();
      }
      s_thread.join();
      racer_thread.join();
      SyncPoint::GetInstance()->ClearCallBack(hold_point);
      ASSERT_OK(s_status) << tag;
      ASSERT_OK(racer_status) << tag;

      ASSERT_OK(Put(cf_for_partition_[1], "control" + tag, "v"));
      Close();
      ASSERT_OK(fault_env->DropUnsyncedFileData());
      ReopenWithPartitions(kPartitions, options);
      EXPECT_EQ("v", Get(cf_for_partition_[1], "a" + tag)) << tag;
      EXPECT_EQ("v", Get(cf_for_partition_[0], "s" + tag)) << tag;
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "control" + tag)) << tag;
      Close();
      ++scenario;
    }
  }
}

TEST_P(DBWritePartitionTest, SyncWriteSyncsOnceAfterItsWait) {
  // GC28 Q3 (AMB-M4-21; the idea of Muse's GC25 SyncWriteSyncsOnceAfterWait):
  // a sync write waits until every record at or below its target is written,
  // then runs one coalesced gate sync. (a) Uncontended sync writes fsync
  // once each. (b) A sync write parked behind a held lower async append runs
  // no gate sync while parked and stays invisible; after the release it runs
  // exactly one gate sync and returns, with no SyncWAL second round.
  // Neuter: sync before the wait (today's order); the parked-phase count is 1.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];

  std::atomic<int> gate_sync_count{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync",
      [&](void*) { ++gate_sync_count; });
  std::atomic<int> sync_wal_count{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::SyncWAL:Begin",
                                        [&](void*) { ++sync_wal_count; });

  // Phase 1: sequential uncontended sync writes each fsync exactly once.
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "uncontended" + std::to_string(i), "value", sync_wo));
  }
  EXPECT_EQ(3, gate_sync_count.load());

  // Phase 2: B (sync) parks on held A (async) before any fsync, invisible;
  // no gate sync runs while it parks, and exactly one runs after the
  // release.
  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool b_appended = false;
  std::atomic<bool> b_returned{false};
  Status b_status;

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (first_append) {
          first_append = false;
          a_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::WriteToWAL:AfterAddRecord",
                                        [&](void*) {
                                          std::lock_guard<std::mutex> lock(mu);
                                          if (!release_a) {
                                            b_appended = true;
                                            cv.notify_all();
                                          }
                                        });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "key_a", "value_a")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_in_append; }));
  }
  port::Thread b_thread([&] {
    b_status = Put(cf_b, "key_b", "value_b", sync_wo);
    b_returned.store(true);
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
  }
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_EQ(3, gate_sync_count.load()) << "no gate sync while parked";
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWAL:Begin");

  ASSERT_OK(b_status);
  EXPECT_TRUE(b_returned.load());
  EXPECT_EQ(4, gate_sync_count.load()) << "exactly one gate sync after release";
  EXPECT_EQ(0, sync_wal_count.load()) << "no SyncWAL second round";
  ASSERT_EQ("value_b", Get(cf_b, "key_b"));
}

TEST_P(DBWritePartitionTest, GateSyncFsyncsClaimedPartitionsConcurrently) {
  // GC28 Q4 (b) (D-GC28c): a gate sync's claimed partitions are fsynced
  // concurrently by threads already parked on the coalescer. With partition
  // 1's fsync held, partition 2's fsync still starts.
  // Neuter: the serial loop; partition 2's fsync never starts (RunWithin
  // aborts).
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  ASSERT_OK(Put(cf_for_partition_[1], "a1", "v"));
  ASSERT_OK(Put(cf_for_partition_[2], "a2", "v"));

  std::mutex mu;
  std::condition_variable cv;
  bool p1_held = false;
  bool release_p1 = false;
  bool p2_started = false;
  bool s2_waiting = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        const size_t p = *static_cast<size_t*>(arg);
        std::unique_lock<std::mutex> lock(mu);
        if (p == 1 && !p1_held) {
          p1_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_p1; });
        } else if (p == 2) {
          p2_started = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:BeforeWait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        s2_waiting = true;
        cv.notify_all();
      });

  WriteOptions sync_wo;
  sync_wo.sync = true;
  Status s1_status;
  // Spawn S2 only after S1 is stuck in its fsync: the two must form
  // separate write groups, or no waiter ever queues on the coalescer.
  port::Thread s1_thread(
      [&] { s1_status = Put(cf_for_partition_[0], "s1", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return p1_held; }));
  }
  Status s2_status;
  port::Thread s2_thread(
      [&] { s2_status = Put(cf_for_partition_[3], "s2", "v", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return s2_waiting; }));
  }
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return p2_started; });
    return Status::OK();
  }));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_p1 = true;
    cv.notify_all();
  }
  s1_thread.join();
  s2_thread.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::RunCoalescedGateSync:BeforeWait");
  ASSERT_OK(s1_status);
  ASSERT_OK(s2_status);
}

TEST_P(DBWritePartitionTest, ConcurrentGateSyncFailureFailsTheGroup) {
  // GC28 Q4 (b): partition 2's fsync fails while partition 1's fsync runs.
  // The write returns the IOError and is invisible before and after Resume();
  // the DB stops; a crash image reopens with every acknowledged sync write.
  // (a1/a2 are async below the failure, so the crash may or may not keep
  // them; b1/b2 are acknowledged sync writes and must survive.)
  // Neuter: ignore the helper's status; the write returns OK.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[1], "b1", "v", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[2], "b2", "v", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[1], "a1", "v"));
  ASSERT_OK(Put(cf_for_partition_[2], "a2", "v"));

  std::atomic<int> p1_fsyncs{0};
  std::atomic<int> p2_fsyncs{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) == 1) {
          ++p1_fsyncs;
        } else if (*static_cast<size_t*>(arg) == 2) {
          ++p2_fsyncs;
          fault_env->SetFilesystemActive(false);
        }
      });

  Status s_status = Put(cf_for_partition_[0], "s", "v", sync_wo);
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  fault_env->SetFilesystemActive(true);
  EXPECT_TRUE(s_status.IsIOError()) << s_status.ToString();
  EXPECT_GE(p1_fsyncs.load(), 1);
  EXPECT_GE(p2_fsyncs.load(), 1);
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "s"));
  ASSERT_NOK(dbfull()->TEST_GetBGError());

  // The DB stopped; Resume keeps the failed write invisible.
  db_->Resume().PermitUncheckedError();
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "s"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());

  // Default recovery fails closed on the unexplained hole left by the
  // failed fsync; point-in-time recovery cuts at it and opens the prefix.
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = static_cast<uint32_t>(kPartitions);
  const Status default_reopen_status =
      TryReopenWithColumnFamilies(cf_names, options);
  ASSERT_TRUE(default_reopen_status.IsCorruption())
      << default_reopen_status.ToString();
  EXPECT_NE(std::string::npos,
            default_reopen_status.ToString().find("unexplained hole"));
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[1], "b1"));
  EXPECT_EQ("v", Get(cf_for_partition_[2], "b2"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "s"));
}

TEST_P(DBWritePartitionTest, JoinWaitTimerOffAtDefaultStatsLevel) {
  // GC28 Q6 (F006, D-GC28f): at stats level kExceptDetailedTimers or below
  // the join timer reads no clock and records nothing; above it, one sample
  // per write, as today.
  // Neuter: the ungated StopWatch; the count is 3 at the default level.
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(1, options);
  ASSERT_EQ(StatsLevel::kExceptDetailedTimers,
            options.statistics->get_stats_level());
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("k" + std::to_string(i), "v"));
  }
  HistogramData at_default;
  options.statistics->histogramData(JOIN_BATCH_GROUP_WAIT_MICROS, &at_default);
  EXPECT_EQ(0U, at_default.count);
  options.statistics->set_stats_level(StatsLevel::kExceptTimeForMutex);
  for (int i = 3; i < 6; ++i) {
    ASSERT_OK(Put("k" + std::to_string(i), "v"));
  }
  HistogramData above;
  options.statistics->histogramData(JOIN_BATCH_GROUP_WAIT_MICROS, &above);
  EXPECT_EQ(3U, above.count);
}

TEST_P(DBWritePartitionTest, FrontierLockTakenOnlyForAllocation) {
  // GC28 Q8 (R6): sequence_wal_index_mutex_ is taken exactly once per write
  // group, at allocation; completion, publication and waiting take the
  // per-partition tracker mutex and the wait mutex instead. 16 threads of
  // non-sync writes at N=8: every allocation (the Take marker) pairs with
  // exactly one group append (AfterAddRecord).
  // Neuter: take the mutex again in PublishAppliedPrefix; takes rise to
  // twice the appends.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kThreads = 16;
  constexpr int kPutsPerThread = 25;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);

  std::atomic<uint64_t> takes{0};
  std::atomic<uint64_t> appends{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SequenceWalIndex:Take",
      [&](void*) { takes.fetch_add(1, std::memory_order_relaxed); });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord",
      [&](void*) { appends.fetch_add(1, std::memory_order_relaxed); });
  SyncPoint::GetInstance()->EnableProcessing();
  std::vector<port::Thread> threads;
  std::atomic<int> failures{0};
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kPutsPerThread; ++i) {
        const int cf = cf_for_partition_[(t + i) % kPartitions];
        Status s =
            Put(cf, "k" + std::to_string(t) + "_" + std::to_string(i), "v");
        if (!s.ok()) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_EQ(0, failures.load());
  EXPECT_GT(takes.load(), 0U);
  EXPECT_EQ(appends.load(), takes.load());
}

TEST_P(DBWritePartitionTest, FrontierRegistrationHoldsFrontierBelowHeldRange) {
  // GC28 Q8 ordering: a range registered under the allocation lock is
  // visible to every later scan, so the WAL and applied frontiers never pass
  // a registered-but-incomplete range. Partition 1's append is held between
  // its registration and its completion while partition 0 completes higher
  // ranges; neither frontier passes the held range.
  // Neuter: register after releasing the allocation lock, holding the window
  // open at AllocateSequenceAndWALIndex:NeuterWindow; the held range sits
  // unregistered while partition 0 completes, and both frontiers pass it.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kP0Puts = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_p0 = cf_for_partition_[0];
  const int cf_p1 = cf_for_partition_[1];
  ASSERT_OK(Put(cf_p0, "base", "v"));
  const SequenceNumber base = dbfull()->GetLatestSequenceNumber();
  const SequenceNumber held_low = base + 1;

  std::mutex mu;
  std::condition_variable cv;
  bool hold_append = true;
  bool p1_in_append = false;
  bool release_append = false;
  bool neuter_hold = true;
  bool neuter_held = false;
  bool release_neuter = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (hold_append) {
          hold_append = false;
          p1_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_append; });
        }
      });
  // Fires only under the neuter, which parks the held range between its
  // counter advance and its registration.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::AllocateSequenceAndWALIndex:NeuterWindow", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (neuter_hold) {
          neuter_hold = false;
          neuter_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_neuter; });
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();

  Status p1_status;
  port::Thread p1_thread(
      [&] { p1_status = Put(cf_p1, "held", "v", WriteOptions()); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return p1_in_append || neuter_held; }));
    if (neuter_held) {
      // Neuter only: the held range sits unregistered at the window; disarm
      // the append hold so partition 0's appends run free below.
      hold_append = false;
    }
  }
  // Partition 0 completes higher ranges on threads; their gates park behind
  // the held range, so they cannot join yet.
  std::vector<Status> p0_statuses(kP0Puts);
  std::vector<port::Thread> p0_threads;
  for (int i = 0; i < kP0Puts; ++i) {
    p0_threads.emplace_back([&, i] {
      p0_statuses[i] = Put(cf_p0, "p0_" + std::to_string(i), "v");
    });
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (dbfull()->TEST_WalDurabilityDurableUpto(0) < held_low + kP0Puts - 1 ||
         dbfull()->TEST_WalDurabilityPendingLow(0) !=
             WalDurabilityTracker::kIdle) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "partition 0 appends never completed";
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  // Higher ranges completed on partition 0, but neither frontier passes the
  // held range on partition 1.
  EXPECT_LT(dbfull()->WalDurabilitySafePoint(), held_low);
  EXPECT_LT(dbfull()->GetLatestSequenceNumber(), held_low);
  {
    std::lock_guard<std::mutex> lock(mu);
    release_neuter = true;
    release_append = true;
    cv.notify_all();
  }
  p1_thread.join();
  for (auto& th : p0_threads) {
    th.join();
  }
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_OK(p1_status);
  for (const Status& s : p0_statuses) {
    ASSERT_OK(s);
  }
  EXPECT_EQ("v", Get(cf_p1, "held"));
  for (int i = 0; i < kP0Puts; ++i) {
    EXPECT_EQ("v", Get(cf_p0, "p0_" + std::to_string(i)));
  }
}

TEST_P(DBWritePartitionTest, TargetedWakeOnWalAdvance) {
  // GC28 Q8 targeted wake, WAL leg: S1 (sync, target h+1) and S2 (sync,
  // target h+3) park in their pre-sync waits above held appends H1 [h,h] and
  // H2 [h+2,h+2]; releasing H1 advances the WAL frontier to h+1, so S1 runs
  // its sync and returns while S2 stays parked; releasing H2 lets S2 return.
  // H2 rides its own partition: a second held append on H1's partition would
  // queue behind H1's group and never reach the hold.
  // Neuter: drop the wake at the WAL Complete site; S1 never wakes.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_h = cf_for_partition_[0];
  const int cf_s = cf_for_partition_[1];
  const int cf_h2 = cf_for_partition_[2];
  WriteOptions sync_wo;
  sync_wo.sync = true;

  std::mutex mu;
  std::condition_variable cv;
  bool hold_append = false;
  bool h_in_append = false;
  int hold_gen = 0;
  int release_gen = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (hold_append) {
          hold_append = false;
          h_in_append = true;
          const int my_gen = hold_gen;
          cv.notify_all();
          cv.wait(lock, [&] { return release_gen >= my_gen; });
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  auto hold_next_append = [&] {
    std::unique_lock<std::mutex> lock(mu);
    hold_append = true;
    h_in_append = false;
    ++hold_gen;
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return h_in_append; }));
  };
  auto wait_for_waiters = [&](size_t n) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (dbfull()->TEST_WalDurabilityWaiters() != n) {
      ASSERT_LT(std::chrono::steady_clock::now(), deadline)
          << "parked-waiter count never reached " << n;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  };

  Status h1_status;
  port::Thread h1_thread(
      [&] { h1_status = Put(cf_h, "h1", "v", WriteOptions()); });
  hold_next_append();
  Status s1_status;
  std::atomic<bool> s1_done{false};
  port::Thread s1_thread([&] {
    s1_status = Put(cf_s, "s1", "v", sync_wo);
    s1_done.store(true);
  });
  wait_for_waiters(1);
  Status h2_status;
  port::Thread h2_thread(
      [&] { h2_status = Put(cf_h2, "h2", "v", WriteOptions()); });
  hold_next_append();
  Status s2_status;
  std::atomic<bool> s2_done{false};
  port::Thread s2_thread([&] {
    s2_status = Put(cf_s, "s2", "v", sync_wo);
    s2_done.store(true);
  });
  wait_for_waiters(2);

  // Releasing H1 covers S1 (target h+1) but not S2 (target h+3).
  {
    std::lock_guard<std::mutex> lock(mu);
    ++release_gen;
    cv.notify_all();
  }
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    s1_thread.join();
    return s1_status;
  }));
  ASSERT_OK(s1_status);
  // S2 cannot have proceeded; give a wrongful wake time to manifest.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(s2_done.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    ++release_gen;
    cv.notify_all();
  }
  h1_thread.join();
  h2_thread.join();
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    s2_thread.join();
    return s2_status;
  }));
  ASSERT_OK(s2_status);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_OK(h1_status);
  ASSERT_OK(h2_status);
  EXPECT_EQ("v", Get(cf_h, "h1"));
  EXPECT_EQ("v", Get(cf_h2, "h2"));
  EXPECT_EQ("v", Get(cf_s, "s1"));
  EXPECT_EQ("v", Get(cf_s, "s2"));
}

TEST_P(DBWritePartitionTest, TargetedWakeOnPublish) {
  // GC28 Q8 targeted wake, published leg: W1 (async, target h+1) and W2
  // (async, target h+3) park in their completion gates above held appends H1
  // [h,h] and H2 [h+2,h+2]; releasing H1 publishes through h+1, so W1
  // returns while W2 stays parked; releasing H2 lets W2 return.
  // H2 rides its own partition: a second held append on H1's partition would
  // queue behind H1's group and never reach the hold.
  // Neuter: drop the wake in PublishAppliedPrefix; W1 never wakes.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = IndexedWalOptions(CurrentOptions());
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_h = cf_for_partition_[0];
  const int cf_w = cf_for_partition_[1];
  const int cf_h2 = cf_for_partition_[2];

  std::mutex mu;
  std::condition_variable cv;
  bool hold_append = false;
  bool h_in_append = false;
  int hold_gen = 0;
  int release_gen = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (hold_append) {
          hold_append = false;
          h_in_append = true;
          const int my_gen = hold_gen;
          cv.notify_all();
          cv.wait(lock, [&] { return release_gen >= my_gen; });
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  auto hold_next_append = [&] {
    std::unique_lock<std::mutex> lock(mu);
    hold_append = true;
    h_in_append = false;
    ++hold_gen;
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return h_in_append; }));
  };
  auto wait_for_waiters = [&](size_t n) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (dbfull()->TEST_WalDurabilityWaiters() != n) {
      ASSERT_LT(std::chrono::steady_clock::now(), deadline)
          << "parked-waiter count never reached " << n;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  };

  Status h1_status;
  port::Thread h1_thread(
      [&] { h1_status = Put(cf_h, "h1", "v", WriteOptions()); });
  hold_next_append();
  Status w1_status;
  std::atomic<bool> w1_done{false};
  port::Thread w1_thread([&] {
    w1_status = Put(cf_w, "w1", "v", WriteOptions());
    w1_done.store(true);
  });
  wait_for_waiters(1);
  Status h2_status;
  port::Thread h2_thread(
      [&] { h2_status = Put(cf_h2, "h2", "v", WriteOptions()); });
  hold_next_append();
  Status w2_status;
  std::atomic<bool> w2_done{false};
  port::Thread w2_thread([&] {
    w2_status = Put(cf_w, "w2", "v", WriteOptions());
    w2_done.store(true);
  });
  wait_for_waiters(2);

  // Releasing H1 covers W1 (target h+1) but not W2 (target h+3).
  {
    std::lock_guard<std::mutex> lock(mu);
    ++release_gen;
    cv.notify_all();
  }
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    w1_thread.join();
    return w1_status;
  }));
  ASSERT_OK(w1_status);
  // W2 cannot have proceeded; give a wrongful wake time to manifest.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_FALSE(w2_done.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    ++release_gen;
    cv.notify_all();
  }
  h1_thread.join();
  h2_thread.join();
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    w2_thread.join();
    return w2_status;
  }));
  ASSERT_OK(w2_status);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_OK(h1_status);
  ASSERT_OK(h2_status);
  EXPECT_EQ("v", Get(cf_h, "h1"));
  EXPECT_EQ("v", Get(cf_h2, "h2"));
  EXPECT_EQ("v", Get(cf_w, "w1"));
  EXPECT_EQ("v", Get(cf_w, "w2"));
}

TEST_P(DBWritePartitionTest, ManualWalFlushSyncWriteSurvivesCrash) {
  // manual_wal_flush with sync writes at N > 1: async A on partition 0 and B
  // on partition 2 sit in their WAL buffers until sync S on partition 1 runs
  // its gate sync, which flushes and fsyncs every partition. A crash that
  // drops every unsynced byte keeps A, B and S; a later async write, never
  // flushed, is lost, proving the crash dropped unsynced data.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.manual_wal_flush = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "key_a", "value_a"));
  ASSERT_OK(Put(cf_for_partition_[2], "key_b", "value_b"));
  ASSERT_OK(Put(cf_for_partition_[1], "key_s", "value_s", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[3], "key_control", "value_control"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("value_a", Get(cf_for_partition_[0], "key_a"));
  EXPECT_EQ("value_b", Get(cf_for_partition_[2], "key_b"));
  EXPECT_EQ("value_s", Get(cf_for_partition_[1], "key_s"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[3], "key_control"));
}

TEST_P(DBWritePartitionTest, SyncZeroCountWriteAfterSyncWriteSyncs) {
  // GC20 Q1 (FU14 H1) without transactions: a sync=true batch holding only
  // LogData consumes no sequence, so its gate targets the last sequence,
  // which the sync write before it already covered. It must still fsync
  // after its own append: WAL_FILE_SYNCED advances, and its record reads back
  // after a power loss that drops every unsynced byte, while an unsynced
  // control record written after it is lost (GC21 Q5, the read-back design
  // of GC20's SyncLogDataOnlyBatchSyncsAndSurvivesCrash).
  // Neuter: RunCoalescedGateSync returns on `wal_gate_synced_upto_ >=
  // target` alone; no sync runs and the crash loses the record.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.statistics = CreateDBStatistics();
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "key", "value", sync_wo));
  const SequenceNumber last = db_->GetLatestSequenceNumber();

  WriteBatch log_data_only;
  ASSERT_OK(log_data_only.PutLogData("blob"));
  const uint64_t synced_before =
      options.statistics->getTickerCount(WAL_FILE_SYNCED);
  ASSERT_OK(db_->Write(sync_wo, &log_data_only));
  EXPECT_GT(options.statistics->getTickerCount(WAL_FILE_SYNCED), synced_before);
  EXPECT_EQ(last, db_->GetLatestSequenceNumber());
  // Never synced: the power loss below must drop it.
  WriteBatch control;
  ASSERT_OK(control.PutLogData("control"));
  ASSERT_OK(db_->Write(WriteOptions(), &control));

  std::vector<uint64_t> wals;
  for (size_t p = 0; p < kPartitions; ++p) {
    wals.push_back(dbfull()->TEST_CurrentWalNumber(p));
  }
  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  bool found_record = false;
  bool found_control = false;
  for (const uint64_t wal : wals) {
    std::unique_ptr<SequentialFileReader> file_reader;
    ASSERT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), LogFileName(dbname_, wal), FileOptions(),
        &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, wal);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
      found_record |= record.ToString().find("blob") != std::string::npos;
      found_control |= record.ToString().find("control") != std::string::npos;
    }
  }
  EXPECT_TRUE(found_record);
  EXPECT_FALSE(found_control);
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("value", Get(cf_for_partition_[0], "key"));
}

TEST_P(DBWritePartitionTest, ZeroCountMarkerNeverPublishesInFlightSequence) {
  // GC20 Q3 (K1): a WAL record that consumes no sequence (here LogData
  // only) completes while every other partition is idle. It must not move
  // the applied safe point past the last allocated sequence L: X then
  // allocates L + 1 on partition 1 and is held before its append, and Y
  // allocates L + 2 on partition 2, completes, publishes and parks in its
  // gate. LastSequence() stays at L until X completes.
  // Neuter: the tracker treats the marker's empty range [L + 1, L] as
  // [L + 1, L + 1]; Y's completion publishes L + 1, X's in-flight sequence
  // (in debug, PublishAppliedFrontierLocked's pending-low assertion fires).
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[0], "base", "value"));
  WriteBatch marker;
  ASSERT_OK(marker.PutLogData("marker"));
  ASSERT_OK(db_->Write(WriteOptions(), &marker));
  const SequenceNumber last = db_->GetLatestSequenceNumber();

  std::mutex mu;
  std::condition_variable cv;
  bool x_in_append = false;
  bool release_x = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        if (!static_cast<Slice*>(arg)->ends_with("value_x")) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        x_in_append = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_x; });
      });
  Status x_status;
  port::Thread x_thread(
      [&] { x_status = Put(cf_for_partition_[1], "key_x", "value_x"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return x_in_append; }));
  }
  Status y_status;
  port::Thread y_thread(
      [&] { y_status = Put(cf_for_partition_[2], "key_y", "value_y"); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_EQ(last, db_->GetLatestSequenceNumber());
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[2], "key_y"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_x = true;
    cv.notify_all();
  }
  x_thread.join();
  y_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_OK(x_status);
  ASSERT_OK(y_status);
  EXPECT_EQ(last + 2, db_->GetLatestSequenceNumber());
  EXPECT_EQ("value_x", Get(cf_for_partition_[1], "key_x"));
  EXPECT_EQ("value_y", Get(cf_for_partition_[2], "key_y"));
}

TEST_P(DBWritePartitionTest, CoverAppendSerializedWithFlushWAL) {
  // GC20 Q4 (K3): with manual_wal_flush at N=2, a protected batch whose log
  // entry is corrupted fails WriteBatch::VerifyChecksum inside WriteToWAL,
  // before any append, so its burned WAL index gets a void cover on a
  // healthy writer, while another thread flushes every WAL buffer in a loop.
  // The cover appends to the same buffer the flush drains; TSan reports the
  // race unless the cover holds the partition's WAL mutex.
  // Neuter: CoverBurnedWALIndex takes no lock; TSan reports a data race on
  // the writer's buffer.
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  options.manual_wal_flush = true;
  options.paranoid_checks = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[0], "before", "value"));

  std::atomic<bool> corrupt{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        if (!corrupt.exchange(false)) {
          return;
        }
        Slice* entry = static_cast<Slice*>(arg);
        char* buf = const_cast<char*>(entry->data());
        buf[entry->size() - 1] ^= 0x1;
      });
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> flushes{0};
  port::Thread flusher([&] {
    while (!stop.load()) {
      db_->FlushWAL(/*sync=*/false).PermitUncheckedError();
      ++flushes;
    }
  });
  while (flushes.load() < 10) {
    std::this_thread::yield();
  }
  WriteBatch batch(/*reserved_bytes=*/0, /*max_bytes=*/0,
                   /*protection_bytes_per_key=*/8, /*default_cf_ts_sz=*/0);
  ASSERT_OK(batch.Put(handles_[cf_for_partition_[0]], "key", "value"));
  corrupt = true;
  const Status s = db_->Write(WriteOptions(), &batch);
  const uint64_t flushes_after_write = flushes.load();
  while (flushes.load() < flushes_after_write + 10) {
    std::this_thread::yield();
  }
  stop = true;
  flusher.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
  EXPECT_FALSE(corrupt.load());
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key"));
  EXPECT_EQ("value", Get(cf_for_partition_[0], "before"));
}

TEST_P(DBWritePartitionTest, AckedAsyncWriteAboveUnsyncedHoleIsCut) {
  // The risk D-G11 (c) accepts: an acknowledged sync=false write can sit in
  // one partition's WAL above a lower record that was written but never
  // fsynced in another's. A crash that keeps partition 1's WAL (as if the OS
  // wrote it back) and drops partition 0's unsynced bytes leaves C's record
  // on disk above the hole A left; recovery cuts at the hole, so the
  // recovered state is still a prefix and omits C. (Point-in-time recovery
  // cuts there; the other modes refuse to open past an unexplained hole.)
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  // Partition 0's WAL holds a synced record below A, which bounds the hole.
  ASSERT_OK(Put(cf_for_partition_[0], "key_synced", "value", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[0], "key_a", "value_a"));
  ASSERT_OK(Put(cf_for_partition_[1], "key_c", "value_c"));
  const std::string c_wal =
      LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(1));
  uint64_t c_wal_size = 0;
  ASSERT_OK(env_->GetFileSize(c_wal, &c_wal_size));
  fault_env->UntrackFile(c_wal);

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  uint64_t c_wal_size_after = 0;
  ASSERT_OK(env_->GetFileSize(c_wal, &c_wal_size_after));
  // C's record is still on disk.
  EXPECT_EQ(c_wal_size, c_wal_size_after);
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("value", Get(cf_for_partition_[0], "key_synced"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key_a"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "key_c"));
}

TEST_P(DBWritePartitionTest, SupersessionsAreDurableWhenOpenReturns) {
  // Recovery writes its supersessions to partition 0's new WAL, while the
  // first write after a cut may reissue an obsoleted index into partition 1.
  // A crash that keeps partition 1's WAL and loses partition 0's unsynced
  // bytes must still leave the supersessions, for every cut source. Only the
  // filter stop obsoletes a readable record, so only there would losing them
  // fail the next open; the torn cuts obsolete an unreadable tail.
  if (GetParam()) {
    // Indexing is rejected with two write queues, so no recovery there writes
    // supersessions. Pin the rejection: lifting it must bring this test along.
    Options options = IndexedWalOptions(CurrentOptions());
    options.two_write_queues = true;
    Destroy(options);
    const Status s = TryReopen(options);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
    return;
  }
  class StopAfterRecords : public WalFilter {
   public:
    explicit StopAfterRecords(int stop_after) : stop_after_(stop_after) {}
    const char* Name() const override { return "StopAfterRecords"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      return records_found_++ < stop_after_
                 ? WalProcessingOption::kContinueProcessing
                 : WalProcessingOption::kStopReplay;
    }

   private:
    const int stop_after_;
    int records_found_ = 0;
  };
  enum class Cut { kWalFilterStop, kSkipTornTail, kPointInTimeTornTail };
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 6;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  for (const Cut cut :
       {Cut::kWalFilterStop, Cut::kSkipTornTail, Cut::kPointInTimeTornTail}) {
    SCOPED_TRACE(static_cast<int>(cut));
    Options options = IndexedWalOptions(CurrentOptions());
    options.env = fault_env.get();
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    // Alternating partitions: index i + 1 holds key i, and partition 1's
    // WAL holds the odd keys.
    for (int i = 0; i < kKeys; ++i) {
      ASSERT_OK(
          Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
    }
    const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(1);
    ASSERT_OK(db_->SyncWAL());
    Close();

    // The filter stops at key 3, so the next write reissues its index; a torn
    // tail loses key 5, the last record.
    StopAfterRecords filter(/*stop_after=*/3);
    int surviving = 3;
    if (cut == Cut::kWalFilterStop) {
      options.wal_filter = &filter;
    } else {
      uint64_t size = 0;
      const std::string torn = LogFileName(dbname_, cut_wal);
      ASSERT_OK(env_->GetFileSize(torn, &size));
      ASSERT_OK(test::TruncateFile(env_, torn, size - 1));
      options.wal_recovery_mode =
          cut == Cut::kSkipTornTail ? WALRecoveryMode::kSkipAnyCorruptedRecords
                                    : WALRecoveryMode::kPointInTimeRecovery;
      surviving = kKeys - 1;
    }
    ReopenWithPartitions(kPartitions, options);
    const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
    const std::string reissuing_wal =
        LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(1));
    ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
    // Partition 1's new WAL survives the crash as if written back.
    fault_env->UntrackFile(reissuing_wal);
    Close();
    ASSERT_OK(fault_env->DropUnsyncedFileData());

    std::unique_ptr<SequentialFileReader> file_reader;
    ASSERT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), LogFileName(dbname_, declaring_wal),
        FileOptions(), &file_reader, /*dbg=*/nullptr,
        /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, declaring_wal);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
    }
    EXPECT_TRUE(reader.IsWALIndexSuperseded(
        cut_wal, std::numeric_limits<uint64_t>::max()));

    options.wal_filter = nullptr;
    ReopenWithPartitions(kPartitions, options);
    for (int i = 0; i < surviving; ++i) {
      EXPECT_EQ("value",
                Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
          << i;
    }
    EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
    Close();
  }
}

TEST_P(DBWritePartitionTest, CleanWALCutStaysAtOrAboveNextIndex) {
  // A filter stop at N=2 where partition 1's WAL finishes clean holding
  // records the filter ignored: index i + 1 holds key i, partition 1 holds
  // indices 2, 4 and 6, and the filter ignores 4 and 6 and stops at 7. The
  // next index is 6, so partition 1 is cut from 6. A cut from 3, where its
  // applied records end, would leave index 4 a hole only supersession
  // explains; at 6, index 4 stays live, as one WAL at a time keeps an ignored
  // record below the cut.
  class IgnoreThenStop : public WalFilter {
   public:
    const char* Name() const override { return "IgnoreThenStop"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      const int found = records_found_++;
      if (found == 3 || found == 5) {
        return WalProcessingOption::kIgnoreCurrentRecord;
      }
      return found < 6 ? WalProcessingOption::kContinueProcessing
                       : WalProcessingOption::kStopReplay;
    }

   private:
    int records_found_ = 0;
  };
  Options options = IndexedWalOptions(CurrentOptions());
  if (GetParam()) {
    // Reopening at N > 1 needs indexed WALs, which two_write_queues rejects.
    options.two_write_queues = true;
    Destroy(options);
    const Status s = TryReopen(options);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 7;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  const uint64_t clean_wal = dbfull()->TEST_CurrentWalNumber(1);
  Close();

  IgnoreThenStop filter;
  options.wal_filter = &filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  // Takes index 6, which the cut obsoletes in partition 1.
  ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
  Close();

  std::unique_ptr<SequentialFileReader> file_reader;
  ASSERT_OK(SequentialFileReader::Create(
      env_->GetFileSystem(), LogFileName(dbname_, declaring_wal), FileOptions(),
      &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
  log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                     /*reporter=*/nullptr, /*checksum=*/true, declaring_wal);
  std::string scratch;
  Slice record;
  while (reader.ReadRecord(&record, &scratch)) {
  }
  EXPECT_EQ(6U, reader.GetFirstSupersededWALIndex(clean_wal));

  options.wal_filter = nullptr;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys; ++i) {
    EXPECT_EQ(i < 5 ? "value" : "NOT_FOUND",
              Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
  Close();
}

// Ported from D126193586.
TEST_P(DBWritePartitionTest, CutFirstObsoleteAtLeastNextWALIndex) {
  // Uneven partition tails at a point-in-time stop: partition 1's WAL loses
  // its last record to a torn tail while partition 0 applied further. The
  // cut on partition 1's WAL must name first_obsolete at or above the seed,
  // not the member's own applied high plus one, leaving no superseded-only
  // hole the next open would have to explain only by supersession.
  if (GetParam()) {
    // Indexed WALs are rejected with two write queues; pin the rejection
    // instead of running the single-queue scenario.
    Options options = IndexedWalOptions(CurrentOptions());
    options.two_write_queues = true;
    const Status s = TryReopen(options);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find("two_write_queues"));
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 6;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  // Alternating partitions: index i + 1 holds key i.
  for (int i = 0; i < kKeys; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(db_->SyncWAL());
  Close();

  uint64_t size = 0;
  const std::string torn = LogFileName(dbname_, cut_wal);
  ASSERT_OK(env_->GetFileSize(torn, &size));
  ASSERT_OK(test::TruncateFile(env_, torn, size - 1));
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
  Close();

  std::unique_ptr<SequentialFileReader> file_reader;
  ASSERT_OK(SequentialFileReader::Create(
      env_->GetFileSystem(), LogFileName(dbname_, declaring_wal), FileOptions(),
      &file_reader, /*dbg=*/nullptr,
      /*rate_limiter=*/nullptr));
  log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                     /*reporter=*/nullptr, /*checksum=*/true, declaring_wal);
  std::string scratch;
  Slice record;
  while (reader.ReadRecord(&record, &scratch)) {
  }
  // Applied indices run 1-5, so the seed is 6; the unclamped cut would name
  // 5 (partition 1's applied high plus one).
  EXPECT_EQ(6U, reader.GetFirstSupersededWALIndex(cut_wal));

  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys - 1; ++i) {
    EXPECT_EQ("value", Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
  Close();
}

TEST_P(DBWritePartitionTest, StoppedWALCutStaysAtOrAboveNextIndex) {
  // CleanWALCutStaysAtOrAboveNextIndex with partition 1 stopped mid-file:
  // it also holds index 8, after the filter stop at 7, so the stop leaves it
  // unfinished. The filter ignores indices 4 and 6, both in partition 1, and
  // the next index is 6. A cut from 3, where partition 1's applied records
  // end, would drop the ignored key at index 4 at the next open; the cut from
  // 6 keeps it live, while the key at 8 after the stop stays dropped.
  class IgnoreThenStop : public WalFilter {
   public:
    const char* Name() const override { return "IgnoreThenStop"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      const int found = records_found_++;
      if (found == 3 || found == 5) {
        return WalProcessingOption::kIgnoreCurrentRecord;
      }
      return found < 6 ? WalProcessingOption::kContinueProcessing
                       : WalProcessingOption::kStopReplay;
    }

   private:
    int records_found_ = 0;
  };
  Options options = IndexedWalOptions(CurrentOptions());
  if (GetParam()) {
    // Reopening at N > 1 needs indexed WALs, which two_write_queues rejects.
    options.two_write_queues = true;
    Destroy(options);
    const Status s = TryReopen(options);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 8;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  const uint64_t stopped_wal = dbfull()->TEST_CurrentWalNumber(1);
  Close();

  IgnoreThenStop filter;
  options.wal_filter = &filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
  Close();

  std::unique_ptr<SequentialFileReader> file_reader;
  ASSERT_OK(SequentialFileReader::Create(
      env_->GetFileSystem(), LogFileName(dbname_, declaring_wal), FileOptions(),
      &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
  log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                     /*reporter=*/nullptr, /*checksum=*/true, declaring_wal);
  std::string scratch;
  Slice record;
  while (reader.ReadRecord(&record, &scratch)) {
  }
  EXPECT_EQ(6U, reader.GetFirstSupersededWALIndex(stopped_wal));

  options.wal_filter = nullptr;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys; ++i) {
    EXPECT_EQ(i < 5 ? "value" : "NOT_FOUND",
              Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
  Close();
}

// The torn-tail cut kind, ported from D126218376: partition 1's WAL loses its
// last record (index 6) to a torn tail, and the filter ignores index 4, also
// partition 1's, while index 5 on partition 0 applies. The next index is 6;
// partition 1's applied records end at 2, so an unclamped cut from 3 would
// supersede the ignored index 4, which one WAL at a time keeps live below the
// cut. Neuter: drop the clamp; the cut names 3 and key3 reads NOT_FOUND.
TEST_P(DBWritePartitionTest, TornWALCutKeepsIgnoredRecordBelowSeedLive) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 6;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  PutAlternatingKeys(kKeys, "value");
  const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(db_->SyncWAL());
  Close();

  uint64_t size = 0;
  const std::string torn = LogFileName(dbname_, cut_wal);
  ASSERT_OK(env_->GetFileSize(torn, &size));
  ASSERT_OK(test::TruncateFile(env_, torn, size - 1));
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  IgnoreFoundRecords filter({3});
  options.wal_filter = &filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
  Close();
  EXPECT_EQ(6U, FirstSupersededWALIndex(declaring_wal, cut_wal));

  options.wal_filter = nullptr;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys; ++i) {
    EXPECT_EQ(i < 5 ? "value" : "NOT_FOUND",
              Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
  Close();
}

// The tail-gap cut kind: partition 1's record at index 6 fails its checksum
// with index 7 on partition 0 above it, so point-in-time recovery surfaces a
// tail gap at 6 and cuts partition 1 there. As in the torn case, the filter
// ignores index 4 on partition 1 and index 5 applies, so the next index is 6
// and an unclamped cut from 3 would supersede the ignored index 4. Neuter:
// drop the clamp; the cut names 3 and key3 reads NOT_FOUND.
TEST_P(DBWritePartitionTest, GapWALCutKeepsIgnoredRecordBelowSeedLive) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 8;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  const std::vector<uint64_t> ends = PutAlternatingKeys(kKeys, "value");
  const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(db_->SyncWAL());
  Close();

  // ends[2] is where index 6 ends; its last byte is payload.
  ASSERT_OK(test::CorruptFile(env_, LogFileName(dbname_, cut_wal),
                              static_cast<int>(ends[2]) - 1,
                              /*bytes_to_corrupt=*/1,
                              /*verify_checksum=*/false));
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  IgnoreFoundRecords filter({3});
  options.wal_filter = &filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
  Close();
  EXPECT_EQ(6U, FirstSupersededWALIndex(declaring_wal, cut_wal));

  options.wal_filter = nullptr;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < kKeys; ++i) {
    EXPECT_EQ(i < 5 ? "value" : "NOT_FOUND",
              Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
  Close();
}

// The frontier cut kind: partition 1's WAL ends clean after index 4, losing 6
// and 8, while partition 0 holds 7, so point-in-time recovery stops at the
// frontier 6 and cuts the unfinished partition 0. The filter ignores indices
// 3 and 5, both partition 0's, and index 4 applies. Recovery seeds to the
// frontier before it cuts, so the cut is clamped to 6 and keeps the ignored
// index 5 live, as one WAL at a time keeps it below the cut. Partition 0's
// applied records end at 1, so an unclamped cut from 2 would supersede the
// ignored index 3. Neuters: drop the clamp, and the cut names 2 and key2
// reads NOT_FOUND; cut before seeding, and the cut names 5 and key4 reads
// NOT_FOUND.
TEST_P(DBWritePartitionTest, FrontierWALCutKeepsIgnoredRecordBelowSeedLive) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 8;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  const std::vector<uint64_t> ends = PutAlternatingKeys(kKeys, "value");
  const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(0);
  const uint64_t clean_wal = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(db_->SyncWAL());
  Close();

  // ends[1] is where index 4 ends.
  ASSERT_OK(test::TruncateFile(env_, LogFileName(dbname_, clean_wal), ends[1]));
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  IgnoreFoundRecords filter({2, 4});
  options.wal_filter = &filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  Close();
  EXPECT_EQ(6U, FirstSupersededWALIndex(declaring_wal, cut_wal));

  // key4 stays live and reads back (ported from D126834369). The final reopen
  // ignores the recovery dummy (the sixth record found), which reuses key4's
  // sequence, so replaying it would go backwards.
  IgnoreFoundRecords ignore_dummy({5});
  options.wal_filter = &ignore_dummy;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ("value", Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  for (int i = 5; i < kKeys; ++i) {
    EXPECT_EQ("NOT_FOUND",
              Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  ASSERT_OK(Put(cf_for_partition_[1], "after", "value"));
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after"));
  Close();
}

TEST_P(DBWritePartitionTest, FrontierCutKeepsIgnoredRecordBelowSeedLive) {
  // The frontier cut kind: the first recovery stops at index 7 via the
  // filter, superseding the old copy of 7, and the old WAL's tail is torn
  // over 7. The second recovery (point-in-time) finds 7's only readable
  // copy superseded and no live copy, so the merge ends at a frontier at
  // 7. The clamp cuts the unfinished member at the seed 7, keeping the
  // ignored index 4 live; without it key3 is lost to a superseded-only
  // hole. Index 6 is ignored in both recoveries: it keeps partition 1's
  // applied high below the seed, and reapplying it after "after2" reused
  // its sequence would go backwards, as one WAL at a time would too.
  // Neuter: remove the clamp; key3 reads NOT_FOUND below. Ported from
  // D126568250 as a second shape of the seed-before-cut order.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  class StopAtSeventh : public WalFilter {
   public:
    const char* Name() const override { return "StopAtSeventh"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      return records_found_++ < 6 ? WalProcessingOption::kContinueProcessing
                                  : WalProcessingOption::kStopReplay;
    }

   private:
    int records_found_ = 0;
  };
  class IgnoreFourthAndSixth : public WalFilter {
   public:
    const char* Name() const override { return "IgnoreFourthAndSixth"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      const int found = records_found_++;
      return (found == 3 || found == 5)
                 ? WalProcessingOption::kIgnoreCurrentRecord
                 : WalProcessingOption::kContinueProcessing;
    }

   private:
    int records_found_ = 0;
  };
  class IgnoreSixth : public WalFilter {
   public:
    const char* Name() const override { return "IgnoreSixth"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      return records_found_++ == 5 ? WalProcessingOption::kIgnoreCurrentRecord
                                   : WalProcessingOption::kContinueProcessing;
    }

   private:
    int records_found_ = 0;
  };
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 8;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  // Alternating partitions: index i + 1 holds key i.
  for (int i = 0; i < kKeys; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  const uint64_t old_wal_p1 = dbfull()->TEST_CurrentWalNumber(1);
  Close();

  // Recovery 1: stop at 7, declaring cuts that supersede the old copy.
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  StopAtSeventh stop_filter;
  options.wal_filter = &stop_filter;
  ReopenWithPartitions(kPartitions, options);
  options.wal_filter = nullptr;
  Close();

  // Tear the old WAL's tail over 7 (losing index 8).
  uint64_t size = 0;
  ASSERT_OK(env_->GetFileSize(LogFileName(dbname_, old_wal_p1), &size));
  ASSERT_OK(
      test::TruncateFile(env_, LogFileName(dbname_, old_wal_p1), size - 1));

  // Recovery 2: frontier at 7; the clamp cuts at the seed 7.
  IgnoreFourthAndSixth ignore_filter;
  options.wal_filter = &ignore_filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  options.wal_filter = nullptr;
  ASSERT_OK(Put(cf_for_partition_[1], "after2", "value"));
  Close();

  std::unique_ptr<SequentialFileReader> file_reader;
  ASSERT_OK(SequentialFileReader::Create(
      env_->GetFileSystem(), LogFileName(dbname_, declaring_wal), FileOptions(),
      &file_reader, /*dbg=*/nullptr,
      /*rate_limiter=*/nullptr));
  log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                     /*reporter=*/nullptr, /*checksum=*/true, declaring_wal);
  std::string scratch;
  Slice record;
  while (reader.ReadRecord(&record, &scratch)) {
  }
  // Partition 1 applied only index 2, so the unclamped cut would name 3,
  // killing ignored index 4.
  EXPECT_EQ(7U, reader.GetFirstSupersededWALIndex(old_wal_p1));

  // Index 6 stays ignored: reapplying it after "after2" reused its sequence
  // would go backwards.
  IgnoreSixth ignore_sixth;
  options.wal_filter = &ignore_sixth;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ("value", Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
        << i;
  }
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key6"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "key7"));
  EXPECT_EQ("value", Get(cf_for_partition_[1], "after2"));
  Close();
}

// Under skip, members read past corruption, so an unreadable record can sit
// below the next index: partition 1's index 4 fails its checksum between the
// records at 2 and 6, and its torn tail at 8 cuts it. Values span blocks, so
// the reader resumes at index 6 in the next block. The cut stays at or above
// the next index, above the unreadable record, which therefore stays in place
// uncut, as the sequential path leaves it: the next skip open skips it again,
// and the records around it stay live.
TEST_P(DBWritePartitionTest, SkipModeCutLeavesUnreadableRecordBelowSeedUncut) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kKeys = 8;
  const std::string value(log::kBlockSize + log::kBlockSize / 4, 'v');
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  const std::vector<uint64_t> ends = PutAlternatingKeys(kKeys, value);
  const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(db_->SyncWAL());
  Close();

  // Index 4 starts at ends[0]; corrupt its first fragment's payload.
  const std::string wal = LogFileName(dbname_, cut_wal);
  ASSERT_OK(test::CorruptFile(env_, wal, static_cast<int>(ends[0]) + 64,
                              /*bytes_to_corrupt=*/1,
                              /*verify_checksum=*/false));
  ASSERT_OK(test::TruncateFile(env_, wal, ends[3] - 1));
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  Close();
  EXPECT_EQ(8U, FirstSupersededWALIndex(declaring_wal, cut_wal));

  for (int reopen = 0; reopen < 2; ++reopen) {
    SCOPED_TRACE(reopen);
    ReopenWithPartitions(kPartitions, options);
    for (int i = 0; i < kKeys; ++i) {
      EXPECT_EQ(i == 3 || i == 7 ? "NOT_FOUND" : value,
                Get(cf_for_partition_[i % 2], "key" + std::to_string(i)))
          << i;
    }
    Close();
  }
}

TEST_P(DBWritePartitionTest, LiveWalReadsRaceFreeUnderConcurrentWrites) {
  // Q7: a live-WAL entry's size accumulates under the WAL mutex, and the
  // mutex_-only oldest-entry copy skips the size field, so no reader
  // observes a field a writer can modify without a common lock. Writers run
  // on both partitions (single-queue writes take the WAL mutex inside
  // WriteToWAL) while reader threads hammer the WAL-mutex snapshot and the
  // mutex_-only oldest-entry read; small max_total_wal_size adds real
  // SwitchWALs to the mix. Clean under TSan. Unlocking before the
  // bookkeeping reports a race on the single-queue param; restoring the
  // full size copy reports one on both params while the deques still hold
  // a single entry.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.max_total_wal_size = 1 << 20;
  OpenWithPartitions(kPartitions, options);

  constexpr int kPutsPerWriter = 2000;
  const std::string big_value(1024, 'x');
  std::atomic<bool> stop{false};
  std::atomic<int> failures{0};
  std::vector<port::Thread> threads;
  for (size_t p = 0; p < kPartitions; ++p) {
    threads.emplace_back([&, p] {
      for (int i = 0; i < kPutsPerWriter; ++i) {
        if (!Put(cf_for_partition_[p],
                 "q7_p" + std::to_string(p) + "_" + std::to_string(i),
                 big_value)
                 .ok()) {
          ++failures;
        }
      }
    });
  }
  // The two live-WAL read contracts: entry sizes under the WAL mutex, and
  // the oldest entry's stable fields under mutex_ only.
  for (int r = 0; r < 2; ++r) {
    threads.emplace_back([&, r] {
      while (!stop.load()) {
        for (size_t p = 0; p < kPartitions; ++p) {
          if (r == 0) {
            (void)dbfull()->TEST_AliveWalFiles(p);
          } else {
            (void)dbfull()->TEST_OldestAliveWalStableFields(p);
          }
        }
      }
    });
  }
  for (size_t i = 0; i < kPartitions; ++i) {
    threads[i].join();
  }
  stop.store(true);
  for (size_t i = kPartitions; i < threads.size(); ++i) {
    threads[i].join();
  }
  ASSERT_EQ(0, failures.load());
  ASSERT_EQ(big_value, Get(cf_for_partition_[0], "q7_p0_0"));
}

// Set by a test's writer thread so that a SyncPoint callback, which runs on
// every writer's thread, acts on that writer only.
thread_local bool hold_this_writer = false;

// G-19: a stop, a timed delay or a WriteBufferManager stall begun
// through one partition's leader holds on every partition: a no_slowdown
// write on any partition, including one queued behind another partition's
// in-flight group when the stall begins, returns Incomplete and lands
// nothing, while a normal write waits; when the stall ends, every partition
// resumes.
TEST_P(DBWritePartitionTest, StallBegunOnOnePartitionHoldsOnAll) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr size_t kBufferSize = 1 << 20;
  enum class Stall { kStop, kDelay, kWriteBufferManager };
  for (Stall stall :
       {Stall::kStop, Stall::kDelay, Stall::kWriteBufferManager}) {
    SCOPED_TRACE("stall=" + std::to_string(static_cast<int>(stall)));
    Options options = IndexedWalOptions(CurrentOptions());
    std::shared_ptr<WriteBufferManager> wbm;
    if (stall == Stall::kWriteBufferManager) {
      wbm = std::make_shared<WriteBufferManager>(kBufferSize, nullptr,
                                                 /*allow_stall=*/true);
      options.write_buffer_manager = wbm;
    }
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    // Every partition has a nonzero last group for the delay to charge.
    for (size_t p = 0; p < kPartitions; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p], "warm", "v"));
    }

    std::mutex mu;
    std::condition_variable cv;
    bool held_in_append = false;
    bool release_held = false;
    bool queued_linked = false;
    bool stall_begun = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [&](void*) {
          if (!hold_this_writer) {
            return;
          }
          std::unique_lock<std::mutex> lock(mu);
          held_in_append = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_held; });
        });
    SyncPoint::GetInstance()->SetCallBack(
        "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
          if (static_cast<WriteThread::Writer*>(arg)->no_slowdown) {
            std::lock_guard<std::mutex> lock(mu);
            queued_linked = true;
            cv.notify_all();
          }
        });
    auto mark_stall_begun = [&](void*) {
      std::lock_guard<std::mutex> lock(mu);
      stall_begun = true;
      cv.notify_all();
    };
    SyncPoint::GetInstance()->SetCallBack("DBImpl::DelayWrite:Wait",
                                          mark_stall_begun);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::DelayWrite:BeginWriteStallDone", mark_stall_begun);
    SyncPoint::GetInstance()->SetCallBack("WBMStallInterface::BlockDB",
                                          mark_stall_begun);
    SyncPoint::GetInstance()->EnableProcessing();
    auto wait_for = [&](const bool& flag) {
      std::unique_lock<std::mutex> lock(mu);
      return cv.wait_for(lock, std::chrono::seconds(30), [&] { return flag; });
    };

    WriteOptions no_slowdown;
    no_slowdown.no_slowdown = true;
    // Partition 1's group is in flight, with a no_slowdown write queued
    // behind it, when the stall begins.
    port::Thread held([&] {
      hold_this_writer = true;
      ASSERT_OK(Put(cf_for_partition_[1], "held", "v"));
      hold_this_writer = false;
    });
    ASSERT_TRUE(wait_for(held_in_append));
    port::Thread queued([&] {
      ASSERT_TRUE(
          Put(cf_for_partition_[1], "queued", "v", no_slowdown).IsIncomplete());
    });
    ASSERT_TRUE(wait_for(queued_linked));

    // The stall begins through partition 0's leader.
    std::unique_ptr<WriteControllerToken> token;
    dbfull()->TEST_LockMutex();
    if (stall == Stall::kStop) {
      token = dbfull()->TEST_write_controler().GetStopToken();
    } else if (stall == Stall::kDelay) {
      token = dbfull()->TEST_write_controler().GetDelayToken(1);
    }
    dbfull()->TEST_UnlockMutex();
    if (stall == Stall::kWriteBufferManager) {
      // Stalls without making a flush due, so no leader closes the switch
      // gate that the held group is past.
      wbm->ReserveMem(kBufferSize);
      wbm->ScheduleFreeMem(kBufferSize);
      ASSERT_TRUE(wbm->ShouldStall());
      ASSERT_FALSE(wbm->ShouldFlush());
    }
    port::Thread stalled(
        [&] { ASSERT_OK(Put(cf_for_partition_[0], "stalled", "v")); });
    ASSERT_TRUE(wait_for(stall_begun));

    for (size_t p : {size_t{0}, size_t{2}, size_t{3}}) {
      ASSERT_TRUE(
          Put(cf_for_partition_[p], "ns" + std::to_string(p), "v", no_slowdown)
              .IsIncomplete());
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      release_held = true;
      cv.notify_all();
    }
    held.join();
    queued.join();
    ASSERT_TRUE(
        Put(cf_for_partition_[1], "ns1", "v", no_slowdown).IsIncomplete());
    ASSERT_EQ("v", Get(cf_for_partition_[1], "held"));
    ASSERT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "queued"));
    for (size_t p = 0; p < kPartitions; ++p) {
      ASSERT_EQ("NOT_FOUND",
                Get(cf_for_partition_[p], "ns" + std::to_string(p)));
    }

    // End the stall; the stalled write and every partition proceed.
    if (stall == Stall::kWriteBufferManager) {
      wbm->FreeMem(kBufferSize);
    } else {
      dbfull()->TEST_LockMutex();
      token.reset();
      dbfull()->TEST_SignalAllBgCv();
      dbfull()->TEST_UnlockMutex();
    }
    stalled.join();
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    for (size_t p = 0; p < kPartitions; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p], "after", "v"));
      ASSERT_EQ("v", Get(cf_for_partition_[p], "after"));
    }
    ASSERT_EQ("v", Get(cf_for_partition_[0], "stalled"));
    Close();
  }
}

// G-19: under a delay, each partition's leader charges the write
// controller with its own partition's last group, as the only leader does
// at N=1, not with whichever partition's group was last.
TEST_P(DBWritePartitionTest, DelayChargesOwnPartitionsGroup) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  OpenWithPartitions(kPartitions, IndexedWalOptions(CurrentOptions()));
  dbfull()->TEST_LockMutex();
  std::unique_ptr<WriteControllerToken> token =
      dbfull()->TEST_write_controler().GetDelayToken(1 << 30);
  dbfull()->TEST_UnlockMutex();
  std::vector<uint64_t> charged;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::DelayWrite:Start",
      [&](void* arg) { charged.push_back(*static_cast<uint64_t*>(arg)); });
  SyncPoint::GetInstance()->EnableProcessing();

  const std::string big(4000, 'b');
  const std::string small(10, 's');
  auto group_size = [&](int cf, const std::string& value) {
    WriteBatch batch;
    EXPECT_OK(batch.Put(handles_[cf], "k", value));
    return static_cast<uint64_t>(WriteBatchInternal::ByteSize(&batch));
  };
  const int big_cf = cf_for_partition_[0];
  const int small_cf = cf_for_partition_[1];
  for (int round = 0; round < 3; ++round) {
    ASSERT_OK(Put(big_cf, "k", big));
    ASSERT_OK(Put(small_cf, "k", small));
  }
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  dbfull()->TEST_LockMutex();
  token.reset();
  dbfull()->TEST_UnlockMutex();

  // Each partition's first group charges nothing (no earlier group), then
  // each charges that partition's previous group.
  ASSERT_EQ(6U, charged.size());
  EXPECT_EQ(0U, charged[0]);
  EXPECT_EQ(0U, charged[1]);
  for (size_t i = 2; i < charged.size(); i += 2) {
    EXPECT_EQ(group_size(big_cf, big), charged[i]) << i;
    EXPECT_EQ(group_size(small_cf, small), charged[i + 1]) << i;
  }
}

// N-3-2: every partition's leader applies timed delays, in parallel,
// against the one write controller, and together they admit about
// delayed_write_rate, as the single leader does at N=1.
// A clock that moves only when a thread sleeps, and whose sleeps return at
// once. A sleep starts at the time its thread last read, so threads sleeping
// in parallel from the same reading move it once, not once per sleeper.
class ParallelSleepClock : public SystemClockWrapper {
 public:
  ParallelSleepClock()
      : SystemClockWrapper(SystemClock::Default()),
        now_(target_->NowMicros()) {}
  static const char* kClassName() { return "ParallelSleepClock"; }
  const char* Name() const override { return kClassName(); }
  uint64_t NowMicros() override { return Read(); }
  uint64_t NowNanos() override { return Read() * 1000; }
  void SleepForMicroseconds(int micros) override {
    const uint64_t wake = LastRead() + static_cast<uint64_t>(micros);
    uint64_t cur = now_.load();
    while (cur < wake && !now_.compare_exchange_weak(cur, wake)) {
    }
  }

 private:
  static uint64_t& LastRead() {
    static thread_local uint64_t last_read = 0;
    return last_read;
  }
  uint64_t Read() {
    LastRead() = now_.load();
    return LastRead();
  }
  std::atomic<uint64_t> now_;
};

TEST_P(DBWritePartitionTest, DelayedWriteRateHoldsAcrossPartitions) {
  // N-3-2: under a delay token, the bytes admitted across all partitions stay
  // within delayed_write_rate times the elapsed time on ParallelSleepClock,
  // whose parallel sleeps do not add up, at N=1 and N=4 alike.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr uint64_t kRate = 1 << 20;
  constexpr int kWriterThreadsPerPartition = 2;
  constexpr int kBatchesPerWriter = 16;
  const std::string value(8 << 10, 'v');
  auto clock = std::make_shared<ParallelSleepClock>();
  std::unique_ptr<Env> env(new CompositeEnvWrapper(env_, clock));
  Defer close_before_env([this] { Close(); });
  for (size_t num_partitions : {size_t{1}, size_t{4}}) {
    SCOPED_TRACE("num_partitions=" + std::to_string(num_partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.env = env.get();
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    dbfull()->TEST_LockMutex();
    std::unique_ptr<WriteControllerToken> token =
        dbfull()->TEST_write_controler().GetDelayToken(kRate);
    dbfull()->TEST_UnlockMutex();

    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> batch_bytes{0};
    std::vector<port::Thread> writers;
    const uint64_t start = clock->NowMicros();
    for (size_t p = 0; p < num_partitions; ++p) {
      for (int t = 0; t < kWriterThreadsPerPartition; ++t) {
        writers.emplace_back([&, p, t] {
          WriteBatch batch;
          for (int i = 0; i < kBatchesPerWriter; ++i) {
            batch.Clear();
            ASSERT_OK(batch.Put(handles_[cf_for_partition_[p]],
                                std::to_string(t) + "_" + std::to_string(i),
                                value));
            ASSERT_OK(db_->Write(WriteOptions(), &batch));
            bytes += WriteBatchInternal::ByteSize(&batch);
            batch_bytes.store(WriteBatchInternal::ByteSize(&batch));
          }
        });
      }
    }
    for (auto& writer : writers) {
      writer.join();
    }
    const double seconds =
        static_cast<double>(clock->NowMicros() - start) / 1000000;
    dbfull()->TEST_LockMutex();
    token.reset();
    dbfull()->TEST_UnlockMutex();
    const double admitted = static_cast<double>(bytes.load());
    fprintf(
        stderr,
        "partitions=%zu admitted %.0f B in %.4f s of clock time (rate %" PRIu64
        ")\n",
        num_partitions, admitted, seconds, kRate);
    // Every group but the first on each partition is charged before it is
    // admitted, a group holding at most one batch per writer, and the
    // controller hands out one 1 ms interval of credit up front: that is
    // the only allowance. One budget per partition would admit about N
    // times the rate.
    const double allowance = static_cast<double>(
        num_partitions * kWriterThreadsPerPartition * batch_bytes.load() +
        2 * kRate / 1000);
    EXPECT_LE(admitted, kRate * seconds + allowance);
    // Not over-throttled either: a delay ends within one 1 ms sleep step of
    // its deadline, and the controller refills credit for any overshoot.
    EXPECT_GE(admitted, 0.8 * kRate * seconds);
    Close();
  }
}

// N-3-1 (F008): leader L0 is woken by the end of a stall but, before
// it rechecks, leader L1 enters a new stall. Both must still wake when that
// stall ends: with one stall object shared by all leaders, L1's BLOCKED
// overwrote L0's RUNNING and the single wakeup for L1 left one of them
// blocked forever.
TEST_P(DBWritePartitionTest, WriteBufferManagerStallWakesEveryLeader) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kBufferSize = 1 << 20;
  auto wbm = std::make_shared<WriteBufferManager>(kBufferSize, nullptr,
                                                  /*allow_stall=*/true);
  Options options = IndexedWalOptions(CurrentOptions());
  options.write_buffer_manager = wbm;
  OpenWithPartitions(2, options);

  std::mutex mu;
  std::condition_variable cv;
  int before_block = 0;
  bool release_l0 = false;
  bool l0_blocked_again = false;
  std::atomic<int> done{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteBufferManagerStallWrites:BeforeBlock", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        ++before_block;
        cv.notify_all();
        if (hold_this_writer) {
          cv.wait(lock, [&] { return release_l0; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "WBMStallInterface::BlockDB", [&](void*) {
        if (hold_this_writer) {
          std::lock_guard<std::mutex> lock(mu);
          l0_blocked_again = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  auto wait_until = [&](const std::function<bool()>& pred) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(30), pred);
  };

  wbm->ReserveMem(kBufferSize);
  wbm->ScheduleFreeMem(kBufferSize);
  port::Thread l0([&] {
    hold_this_writer = true;
    ASSERT_OK(Put(cf_for_partition_[0], "l0", "v"));
    hold_this_writer = false;
    ++done;
    cv.notify_all();
  });
  ASSERT_TRUE(wait_until([&] { return before_block == 1; }));
  // The stall ends (L0 is signaled), then a new one begins before L0
  // rechecks.
  wbm->FreeMem(kBufferSize);
  wbm->ReserveMem(kBufferSize);
  wbm->ScheduleFreeMem(kBufferSize);
  port::Thread l1([&] {
    ASSERT_OK(Put(cf_for_partition_[1], "l1", "v"));
    ++done;
    cv.notify_all();
  });
  ASSERT_TRUE(wait_until([&] { return before_block == 2; }));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_l0 = true;
    cv.notify_all();
  }
  // L0 either passed on its own wakeup or is waiting again; only then end
  // the second stall, so a lost wakeup cannot be masked by timing.
  ASSERT_TRUE(wait_until([&] { return done.load() >= 1 || l0_blocked_again; }));
  wbm->FreeMem(kBufferSize);
  ASSERT_TRUE(wait_until([&] { return done.load() == 2; }))
      << "a stalled leader was never woken";
  l0.join();
  l1.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_EQ("v", Get(cf_for_partition_[0], "l0"));
  ASSERT_EQ("v", Get(cf_for_partition_[1], "l1"));
}

// N-3-1 (F008): with every partition's leader stalled at once, under
// both flush policies, all wake when the manager ends the stall, none is
// released while it still reports a stall (other than to retry a local
// flush), and the DB is never marked flushable while any leader that hid it
// is still stalled.
TEST_P(DBWritePartitionTest, WriteBufferManagerStallsEveryLeaderTogether) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr size_t kBufferSize = 1 << 20;
  constexpr int kRounds = 5;
  for (auto policy : {WriteBufferFlushPolicy::kFlushOldest,
                      WriteBufferFlushPolicy::kFlushLargestAcrossDBs}) {
    SCOPED_TRACE("policy=" + std::to_string(static_cast<int>(policy)));
    auto wbm = std::make_shared<WriteBufferManager>(
        kBufferSize, nullptr, /*allow_stall=*/true, policy);
    Options options = IndexedWalOptions(CurrentOptions());
    options.write_buffer_manager = wbm;
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    const bool tracks_initiator = wbm->ShouldTrackFlushInitiator();

    std::mutex mu;
    std::condition_variable cv;
    int stalled = 0;
    int early_passes = 0;
    int flushable_while_hidden = 0;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteBufferManagerStallWrites:BeforeBlock", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          ++stalled;
          cv.notify_all();
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteBufferManagerStallWrites:Unblocked", [&](void* arg) {
          std::lock_guard<std::mutex> lock(mu);
          --stalled;
          if (!*static_cast<bool*>(arg) && wbm->ShouldStall()) {
            ++early_passes;
          }
        });
    SyncPoint::GetInstance()->EnableProcessing();

    for (int round = 0; round < kRounds; ++round) {
      wbm->ReserveMem(kBufferSize);
      wbm->ScheduleFreeMem(kBufferSize);
      std::atomic<size_t> done{0};
      std::vector<port::Thread> writers;
      for (size_t p = 0; p < kPartitions; ++p) {
        writers.emplace_back([&, p] {
          ASSERT_OK(Put(cf_for_partition_[p],
                        "r" + std::to_string(round) + "_" + std::to_string(p),
                        "v"));
          ++done;
        });
      }
      {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30), [&] {
          return stalled == static_cast<int>(kPartitions);
        }));
      }
      // Sample the flushable flag while leaders are stalled. Holding mu
      // keeps a counted leader from leaving its stall (and restoring the
      // flag) until the sample is taken.
      const auto sample_end =
          std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
      while (tracks_initiator &&
             std::chrono::steady_clock::now() < sample_end) {
        {
          std::lock_guard<std::mutex> lock(mu);
          if (stalled > 0 && dbfull()->TEST_IsWBMFlushInitiatorFlushable()) {
            ++flushable_while_hidden;
          }
        }
        std::this_thread::yield();
      }
      ASSERT_EQ(0U, done.load());
      wbm->FreeMem(kBufferSize);
      for (auto& writer : writers) {
        writer.join();
      }
      ASSERT_EQ(kPartitions, done.load());
    }
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    EXPECT_EQ(0, early_passes);
    EXPECT_EQ(0, flushable_while_hidden);
    if (tracks_initiator) {
      EXPECT_TRUE(dbfull()->TEST_IsWBMFlushInitiatorFlushable());
    }
    for (int round = 0; round < kRounds; ++round) {
      for (size_t p = 0; p < kPartitions; ++p) {
        ASSERT_EQ("v", Get(cf_for_partition_[p], "r" + std::to_string(round) +
                                                     "_" + std::to_string(p)));
      }
    }
    Close();
  }
}

// N-3-3: the close-time WAL marker covers the highest current WAL
// across partitions, not partition 0's (the lowest of its generation), so
// the next open replays no WAL whose data was flushed.
TEST_P(DBWritePartitionTest, CloseMarkerCoversEveryPartitionsWal) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.optimize_manifest_for_recovery = true;
  OpenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "k" + std::to_string(p), "v"));
  }
  FlushAllColumnFamilies();
  uint64_t max_current_wal = 0;
  for (size_t p = 0; p < kPartitions; ++p) {
    max_current_wal =
        std::max(max_current_wal, dbfull()->TEST_CurrentWalNumber(p));
  }
  ASSERT_LT(dbfull()->TEST_CurrentWalNumber(0), max_current_wal);
  uint64_t captured = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CloseHelper:CapturedMaxWal",
      [&](void* arg) { captured = *static_cast<uint64_t*>(arg); });
  SyncPoint::GetInstance()->EnableProcessing();
  Close();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::CloseHelper:CapturedMaxWal");
  ASSERT_EQ(max_current_wal, captured);

  ReopenWithPartitions(kPartitions, options);
  // Every column family's recorded log number is past every WAL that was
  // current at close, so recovery replays none of them.
  for (ColumnFamilyHandle* handle : handles_) {
    ColumnFamilyData* cfd =
        static_cast_with_check<ColumnFamilyHandleImpl>(handle)->cfd();
    EXPECT_GT(cfd->GetLogNumber(), max_current_wal) << cfd->GetName();
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_EQ("v", Get(cf_for_partition_[p], "k" + std::to_string(p)));
  }
}

// N-3-3: with manual_wal_flush, Open flushes every partition's
// buffered current WAL and must sync each of them, not only partition 0's;
// a power loss right after Open then loses nothing.
TEST_P(DBWritePartitionTest, PostOpenManualFlushSyncsEveryPartition) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  fault_fs->SetInjectUnsyncedDataLoss(true);
  auto latency_fs = std::make_shared<LatencyInjectionFileSystem>(
      fault_fs, LatencyInjectionFileSystem::Latency(), /*seed=*/0);
  std::unique_ptr<Env> env(NewCompositeEnv(latency_fs));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = env.get();
  options.manual_wal_flush = true;
  OpenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "k" + std::to_string(p), "v"));
  }
  ASSERT_OK(db_->FlushWAL(/*sync=*/true));
  Close();

  latency_fs->ResetCounters();
  ReopenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    const uint64_t wal = dbfull()->TEST_CurrentWalNumber(p);
    EXPECT_GE(latency_fs->GetWalCounters(wal).syncs, 1U)
        << "partition " << p << " WAL " << wal;
  }

  // Power loss right after Open.
  fault_fs->SetFilesystemActive(false);
  Close();
  ASSERT_OK(fault_fs->DropUnsyncedFileData());
  fault_fs->SetFilesystemActive(true);
  ReopenWithPartitions(kPartitions, options);
  AssertPartitionDequesConsistent();
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_EQ("v", Get(cf_for_partition_[p], "k" + std::to_string(p)));
  }
}

// G-20 / MRR R3: each partition's WALs attest their predecessor, so deleting
// the oldest live WAL of a partition whose successor is present is detected
// instead of raising the merge's lower bound. The lost file holds the lowest
// live index (partition 1 writes first, above a flushed generation), so only
// the spans show it. Point-in-time keeps the prefix below the lost file's
// first index; tolerate and absolute fail naming the missing WAL. Neuter:
// stop writing spans; every mode opens with every key but r0_p1, silently.
TEST_P(DBWritePartitionTest, DeletedOldestPartitionWalIsDetected) {
  if (GetParam()) {
    // Reopening at N > 1 needs indexed WALs, which two_write_queues rejects.
    return;
  }
  constexpr int kRounds = 3;
  for (const size_t num_partitions : {size_t{2}, size_t{4}}) {
    SCOPED_TRACE(num_partitions);
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    for (size_t p = 0; p < num_partitions; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p], "flushed", "value"));
    }
    FlushAllColumnFamilies();
    const std::vector<std::vector<uint64_t>> alive =
        WriteRounds(kRounds, /*descending=*/true);
    Close();
    ASSERT_OK(env_->DeleteFile(LogFileName(dbname_, alive[1].front())));

    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    for (const WALRecoveryMode recovery_mode :
         {WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kAbsoluteConsistency}) {
      SCOPED_TRACE(static_cast<int>(recovery_mode));
      options.wal_recovery_mode = recovery_mode;
      const Status s = TryReopenWithColumnFamilies(cf_names, options);
      ASSERT_TRUE(s.IsCorruption()) << s.ToString();
      EXPECT_NE(std::string::npos,
                s.ToString().find("WAL " + std::to_string(alive[1].front()) +
                                  " is attested by its successor but "
                                  "missing"))
          << s.ToString();
    }
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
    // Round 0 wrote partitions 3 and 2 below the lost r0_p1.
    for (size_t p = 0; p < num_partitions; ++p) {
      EXPECT_EQ("value", Get(cf_for_partition_[p], "flushed"));
      for (int round = 0; round < kRounds; ++round) {
        EXPECT_EQ(round == 0 && p > 1 ? "value" : "NOT_FOUND",
                  Get(cf_for_partition_[p], RoundKey(round, p)))
            << RoundKey(round, p);
      }
    }
    // The loss is flagged durably (GC12 Q8): the reopen's first WAL cuts the
    // lost file's unfinished successor, so no later open replays past it.
    const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
    Close();
    EXPECT_NE(0U, FirstSupersededWALIndex(declaring_wal, alive[1][1]));
  }
}

// G-20 / G-21: a damaged head on a member holding indexed data, either a
// non-newest WAL whose successor attests it or the newest WAL of a partition
// whose lower-numbered gen-mate anchors the bound, is bounded: its lost index
// surfaces as a hole with records above it. Point-in-time keeps the prefix
// below the hole; tolerate and absolute fail at the hole, not with "head
// damaged and unattested", and skip skips it (GC26 Q4, D-10). Neuter: stop
// writing spans; every mode fails with "head damaged".
TEST_P(DBWritePartitionTest, DamagedPartitionWalHeadIsBoundedPerMode) {
  if (GetParam()) {
    // Reopening at N > 1 needs indexed WALs, which two_write_queues rejects.
    return;
  }
  constexpr size_t kPartitions = 2;
  constexpr int kRounds = 2;
  for (const int damaged_round : {0, 1}) {
    SCOPED_TRACE(damaged_round);
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    // Partition 1 first in each round, so its record sits below partition
    // 0's: round r holds r<r>_p1 at 2r + 1 and r<r>_p0 at 2r + 2.
    const std::vector<std::vector<uint64_t>> alive =
        WriteRounds(kRounds, /*descending=*/true);
    Close();
    // Corrupt partition 1's round's only record, the file's first data
    // record, right after the marker and the span.
    const std::string damaged =
        LogFileName(dbname_, alive[1][static_cast<size_t>(damaged_round)]);
    constexpr int kFirstDataPayload =
        3 * log::kHeaderSize +
        static_cast<int>(log::kWALIndexMarkerPartitionSize) +
        static_cast<int>(log::kWALIndexPredecessorSpanPayloadSize);
    ASSERT_OK(test::CorruptFile(env_, damaged, kFirstDataPayload + 2, 1,
                                /*verify_checksum=*/false));

    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_partitions = kPartitions;
    for (const WALRecoveryMode recovery_mode :
         {WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kAbsoluteConsistency}) {
      SCOPED_TRACE(static_cast<int>(recovery_mode));
      options.wal_recovery_mode = recovery_mode;
      const Status s = TryReopenWithColumnFamilies(cf_names, options);
      ASSERT_TRUE(s.IsCorruption()) << s.ToString();
      EXPECT_EQ(std::string::npos, s.ToString().find("head damaged"))
          << s.ToString();
    }
    // GC26 Q4 (D-10): skip skips the hole and applies every record above it.
    {
      std::vector<std::pair<int, std::string>> present;
      for (int round = 0; round < kRounds; ++round) {
        for (size_t p = 0; p < kPartitions; ++p) {
          if (round != damaged_round || p != 1) {
            present.emplace_back(cf_for_partition_[p], RoundKey(round, p));
          }
        }
      }
      AssertSkipOpensCopy(dbname_, options, present,
                          {{cf_for_partition_[1], RoundKey(damaged_round, 1)}});
    }
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
    for (int round = 0; round < kRounds; ++round) {
      for (size_t p = 0; p < kPartitions; ++p) {
        EXPECT_EQ(round < damaged_round ? "value" : "NOT_FOUND",
                  Get(cf_for_partition_[p], RoundKey(round, p)))
            << RoundKey(round, p);
      }
    }
    Close();
  }
}

// G-20 completeness (GC12 Q6): a brand-new DB's first WALs attest no
// predecessor. With partition 0's lowest record lost and partition 1's higher
// one kept, the narrowed history floor (ported from D126834369) makes the
// merge expect index 1, so the loss is a hole: point-in-time stops there,
// skip skips it (GC26 Q4, D-10), and tolerate and absolute fail. Neuter:
// drop the floor; every mode opens and replays key1 above the lost key0.
TEST_P(DBWritePartitionTest, FirstSessionBottomLossIsDetected) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  options.wal_partitions = kPartitions;
  options.create_missing_column_families = true;
  // Created with the DB by its first open, so no reopen precedes the WALs
  // under test.
  std::vector<std::string> cf_names = {kDefaultColumnFamilyName};
  cf_for_partition_.assign(kPartitions, -1);
  for (uint32_t cf_id = 1;
       std::count(cf_for_partition_.begin(), cf_for_partition_.end(), -1) > 0;
       ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = kPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kPartitions, &partition));
    cf_names.push_back("cf" + std::to_string(cf_id));
    if (cf_for_partition_[partition] == -1) {
      cf_for_partition_[partition] = static_cast<int>(cf_id);
    }
  }
  Destroy(options);
  ReopenWithColumnFamilies(cf_names, options);
  ASSERT_OK(Put(cf_for_partition_[0], "key0", "value"));
  ASSERT_OK(Put(cf_for_partition_[1], "key1", "value"));
  const std::string lost_wal =
      LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(0));
  Close();
  // Keep partition 0's marker, drop its record.
  uint64_t first_record = 0;
  {
    std::unique_ptr<SequentialFileReader> file_reader;
    ASSERT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), lost_wal, FileOptions(), &file_reader,
        /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, 0);
    std::string scratch;
    Slice record;
    ASSERT_TRUE(reader.ReadRecord(&record, &scratch));
    first_record = reader.LastRecordOffset();
  }
  ASSERT_GT(first_record, 0U);
  ASSERT_OK(test::TruncateFile(env_, lost_wal, first_record));

  options.create_if_missing = false;
  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kAbsoluteConsistency}) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    options.wal_recovery_mode = recovery_mode;
    const Status s = TryReopenWithColumnFamilies(cf_names, options);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos,
              s.ToString().find("unexplained hole at wal_index 1 "))
        << s.ToString();
  }
  // GC26 Q4 (D-10): skip skips the hole and replays key1 above it.
  AssertSkipOpensCopy(dbname_, options, {{cf_for_partition_[1], "key1"}},
                      {{cf_for_partition_[0], "key0"}});
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key0"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "key1"));
  Close();
}

// GC12 Q6: the floor never applies once a flush has persisted a sequence:
// the WALs below the live ones were retired with their indices, so the live
// set starts above index 1 by design. Spans are suppressed so no attestation
// keeps the merge from applying a floor; only the open's conditions do.
// Neuter: set the floor whatever the conditions; the first reopen fails with
// a hole at index 1.
TEST_P(DBWritePartitionTest, HistoryFloorOffAfterFlush) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::StartWALFile:WriteSpan",
      [](void* arg) { *static_cast<bool*>(arg) = false; });
  SyncPoint::GetInstance()->EnableProcessing();
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "flushed", "value"));
  }
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "live", "value"));
  }
  Close();
  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kPointInTimeRecovery}) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    options.wal_recovery_mode = recovery_mode;
    ReopenWithPartitions(kPartitions, options);
    for (size_t p = 0; p < kPartitions; ++p) {
      EXPECT_EQ("value", Get(cf_for_partition_[p], "flushed"));
      EXPECT_EQ("value", Get(cf_for_partition_[p], "live"));
    }
    Close();
  }
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::StartWALFile:WriteSpan");
}

// Q1: the first WALs of an open whose N differs from the previous session's
// attest the newest recovered WAL of every recorded partition placed in
// their partition (a recorded partition the open lacks goes to partition 0,
// which attests each such chain, GC26 Q2), and the open after that recovers
// every key, in both directions.
TEST_P(DBWritePartitionTest, ReopenWithChangedPartitionsWritesSpans) {
  if (GetParam()) {
    return;
  }
  for (const auto& [from, to] :
       std::vector<std::pair<size_t, size_t>>{{4, 2}, {2, 4}}) {
    SCOPED_TRACE(std::to_string(from) + " -> " + std::to_string(to));
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(std::max(from, to), options);
    ReopenWithPartitions(from, options);
    const std::vector<std::vector<uint64_t>> before =
        WriteRounds(2, /*descending=*/false);
    std::vector<std::pair<int, std::string>> written;
    for (int round = 0; round < 2; ++round) {
      for (size_t p = 0; p < from; ++p) {
        written.emplace_back(cf_for_partition_[p], RoundKey(round, p));
      }
    }
    ReopenWithPartitions(to, options);
    for (size_t p = 0; p < to; ++p) {
      // Partition p's newest WAL of the previous session and, at partition
      // 0, the newest of each partition placed there, in ascending order; a
      // partition the previous session lacked has no previous WAL and
      // attests none.
      std::vector<uint64_t> expected_preds;
      for (size_t q = 0; q < from; ++q) {
        if (q == p || (p == 0 && q >= to)) {
          expected_preds.push_back(before[q].back());
        }
      }
      std::sort(expected_preds.begin(), expected_preds.end());
      uint64_t pred = 0;
      uint64_t first = 0;
      uint64_t last = 0;
      const auto reader = ReadWALThrough(dbfull()->TEST_CurrentWalNumber(p));
      EXPECT_EQ(!expected_preds.empty(),
                reader->GetWALIndexPredecessorSpan(&pred, &first, &last))
          << p;
      std::vector<uint64_t> preds;
      if (pred != 0) {
        preds.push_back(pred);
      }
      for (const auto& span : reader->GetFurtherWALIndexPredecessorSpans()) {
        preds.push_back(span[0]);
      }
      EXPECT_EQ(expected_preds, preds) << p;
    }
    for (size_t p = 0; p < to; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p], "after" + std::to_string(p), "v"));
    }
    ReopenWithPartitions(to, options);
    for (const auto& [cf, key] : written) {
      EXPECT_EQ("value", Get(cf, key)) << key;
    }
    for (size_t p = 0; p < to; ++p) {
      EXPECT_EQ("v", Get(cf_for_partition_[p], "after" + std::to_string(p)));
    }
    Close();
  }
}

// MRR R2 policy: at every rotation each retiring WAL, idle or not, asserts
// progress up to the new generation's first index, and the new WAL of an idle
// partition still attests its empty predecessor with the high mark.
TEST_P(DBWritePartitionTest, RotationSealsEveryRetiringPartitionWal) {
  if (GetParam()) {
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(2, options);
  ASSERT_OK(Put(cf_for_partition_[0], "key", "value"));
  const uint64_t old0 = dbfull()->TEST_CurrentWalNumber(0);
  const uint64_t old1 = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(dbfull()->TEST_SwitchMemtable(
      static_cast_with_check<ColumnFamilyHandleImpl>(
          handles_[cf_for_partition_[0]])
          ->cfd()));
  const uint64_t new1 = dbfull()->TEST_CurrentWalNumber(1);
  Close();
  // The only allocated index is 1, so the new generation starts at 2.
  EXPECT_EQ(2U, ReadWALThrough(old1)->GetWALIndexProgressBound());
  EXPECT_EQ(2U, ReadWALThrough(old0)->GetWALIndexProgressBound());
  uint64_t pred = 0;
  uint64_t first = 0;
  uint64_t last = 0;
  ASSERT_TRUE(
      ReadWALThrough(new1)->GetWALIndexPredecessorSpan(&pred, &first, &last));
  EXPECT_EQ(old1, pred);
  EXPECT_EQ(1U, first);
  EXPECT_EQ(1U, last);
}

// G-24 under D-10 (GC26 Q4, which replaces D-G24 (a)): skip mode skips an
// unexplained pristine hole at N > 1, the routine shape after a crash
// (partition 1 lost its only record, index 2, partition 0 holds 3), and
// applies the record above it; point-in-time keeps the prefix below the hole.
TEST_P(DBWritePartitionTest, SkipModeSkipsPristineHole) {
  if (GetParam()) {
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(2, options);
  ASSERT_OK(Put(cf_for_partition_[0], "below", "value"));
  ASSERT_OK(Put(cf_for_partition_[1], "hole", "value"));
  ASSERT_OK(Put(cf_for_partition_[0], "above", "value"));
  const std::string torn =
      LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(1));
  Close();
  uint64_t size = 0;
  ASSERT_OK(env_->GetFileSize(torn, &size));
  ASSERT_OK(test::TruncateFile(env_, torn, size - 1));

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = 2;
  // GC26 Q4 (D-10): skip skips the hole and applies the record above it.
  AssertSkipOpensCopy(
      dbname_, options,
      {{cf_for_partition_[0], "below"}, {cf_for_partition_[0], "above"}},
      {{cf_for_partition_[1], "hole"}});
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  EXPECT_EQ("value", Get(cf_for_partition_[0], "below"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "hole"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "above"));
  Close();
}

// G-25 under D-G25 (a): with MANIFEST WAL tracking, a synced WAL the MANIFEST
// names that is missing fails the open naming it, even under point-in-time,
// at one partition and at two.
TEST_P(DBWritePartitionTest, MissingTrackedWalFailsTheOpen) {
  if (GetParam()) {
    return;
  }
  for (const size_t num_partitions : {size_t{1}, size_t{2}}) {
    SCOPED_TRACE(num_partitions);
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    ColumnFamilyData* rotating_cfd =
        static_cast_with_check<ColumnFamilyHandleImpl>(
            handles_[cf_for_partition_[0]])
            ->cfd();
    for (int round = 0; round < 3; ++round) {
      if (round > 0) {
        ASSERT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
      }
      for (size_t p = 0; p < num_partitions; ++p) {
        ASSERT_OK(Put(cf_for_partition_[p], RoundKey(round, p), "value"));
      }
      ASSERT_OK(db_->SyncWAL());
    }
    const uint64_t middle = AliveWalNumbers()[0][1];
    Close();
    ASSERT_OK(env_->DeleteFile(LogFileName(dbname_, middle)));

    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    const Status s = TryReopenWithColumnFamilies(cf_names, options);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos,
              s.ToString().find("Missing WAL with log number: " +
                                std::to_string(middle)))
        << s.ToString();
  }
}

// G-25 under D-G25 (a), every recovery mode (ported from D126834369): a
// synced, MANIFEST-tracked WAL in the middle of partition 0's chain is
// missing; every mode fails the open naming it, at one partition and at two.
// WalSet::CheckWals refuses first; the merge's expected set and the
// successor's span back it. Neuters: skip the missing case in CheckWals, and
// the status is the merge's instead; skip the expected-set check too, and
// point-in-time opens.
TEST_P(DBWritePartitionTest, ManifestTrackedMissingWALFailsEveryMode) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  for (const size_t num_partitions : {size_t{1}, size_t{2}}) {
    for (const WALRecoveryMode recovery_mode :
         {WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kAbsoluteConsistency,
          WALRecoveryMode::kPointInTimeRecovery,
          WALRecoveryMode::kSkipAnyCorruptedRecords}) {
      SCOPED_TRACE(std::to_string(num_partitions) + " " +
                   std::to_string(static_cast<int>(recovery_mode)));
      Options options = IndexedWalOptions(CurrentOptions());
      DestroyAndReopen(options);
      OpenWithPartitions(num_partitions, options);
      ColumnFamilyData* rotating_cfd =
          static_cast_with_check<ColumnFamilyHandleImpl>(
              handles_[cf_for_partition_[0]])
              ->cfd();
      for (int i = 0; i < 2; ++i) {
        ASSERT_OK(
            Put(cf_for_partition_[0], "key" + std::to_string(i), "value"));
      }
      ASSERT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
      const uint64_t mid = dbfull()->TEST_CurrentWalNumber(0);
      for (int i = 2; i < 4; ++i) {
        ASSERT_OK(
            Put(cf_for_partition_[0], "key" + std::to_string(i), "value"));
      }
      ASSERT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
      for (int i = 4; i < 6; ++i) {
        ASSERT_OK(
            Put(cf_for_partition_[0], "key" + std::to_string(i), "value"));
      }
      // Sync every WAL so the middle one is MANIFEST-tracked, then delete it.
      ASSERT_OK(db_->SyncWAL());
      Close();
      ASSERT_OK(env_->DeleteFile(LogFileName(dbname_, mid)));

      std::vector<std::string> cf_names;
      ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
      options.wal_partitions = static_cast<uint32_t>(num_partitions);
      options.wal_recovery_mode = recovery_mode;
      const Status s = TryReopenWithColumnFamilies(cf_names, options);
      EXPECT_TRUE(s.IsCorruption()) << s.ToString();
      EXPECT_NE(std::string::npos,
                s.ToString().find("Missing WAL with log number: " +
                                  std::to_string(mid)))
          << s.ToString();
      Close();
    }
  }
}

// G-25 (ported from D126834369): a point-in-time reopen stopped by a filter
// declares cuts for the unfinished members, so a supersession names each of
// them. Deleting a WAL that a supersession names and the MANIFEST tracks
// fails the next open naming it, rather than being read as cut.
// WalSet::CheckWals refuses first and the merge's expected set backs it.
// Neuters: skip the missing case in CheckWals, and the status is the merge's
// instead; skip the expected-set check too, and the open succeeds.
TEST_P(DBWritePartitionTest, SupersessionNamingTrackedMissingWALFailsOpen) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  ColumnFamilyData* rotating_cfd =
      static_cast_with_check<ColumnFamilyHandleImpl>(
          handles_[cf_for_partition_[0]])
          ->cfd();
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  // Synced and closed by the switch, so the MANIFEST tracks it.
  const uint64_t gen1_p1 = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(dbfull()->TEST_SwitchMemtable(rotating_cfd));
  for (int i = 2; i < 4; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  ASSERT_OK(db_->SyncWAL());
  Close();

  // Apply one record, then stop at partition 1's first: the memtables stay
  // non-empty so the old WALs are retained, and partition 1's first WAL is
  // unfinished, so it gets a cut.
  class StopAtSecond : public WalFilter {
   public:
    const char* Name() const override { return "StopAtSecond"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      return records_found_++ < 1 ? WalProcessingOption::kContinueProcessing
                                  : WalProcessingOption::kStopReplay;
    }

   private:
    int records_found_ = 0;
  } stop_filter;
  options.wal_filter = &stop_filter;
  ReopenWithPartitions(kPartitions, options);
  const uint64_t declaring_wal = dbfull()->TEST_CurrentWalNumber(0);
  options.wal_filter = nullptr;
  Close();
  ASSERT_NE(0U, FirstSupersededWALIndex(declaring_wal, gen1_p1));

  ASSERT_OK(env_->DeleteFile(LogFileName(dbname_, gen1_p1)));
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = kPartitions;
  const Status s = TryReopenWithColumnFamilies(cf_names, options);
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
  EXPECT_NE(std::string::npos,
            s.ToString().find("Missing WAL with log number: " +
                              std::to_string(gen1_p1)))
      << s.ToString();
  Close();
}

// G-26 under D-G26 (a) and CDR-F017: the files decide whether recovery
// merges. An indexed N = 2 set opens under a kNone N = 1 opener, read-only
// and read-write, and under an indexed N = 1 opener, with every key. A set
// whose unindexed WALs (written before indexing) precede partitioned ones
// replays the prefix sequentially and merges the rest.
TEST_P(DBWritePartitionTest, OpenerUsageDoesNotDecideTheMerge) {
  if (GetParam()) {
    return;
  }
  constexpr int kRounds = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  for (const bool indexed_opener : {false, true}) {
    SCOPED_TRACE(indexed_opener ? "indexed N=1" : "kNone N=1");
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    // Partition 1 first, so file-number order is not sequence order.
    WriteRounds(kRounds, /*descending=*/true);
    Close();
    Options opener = options;
    opener.wal_partitions = 1;
    if (!indexed_opener) {
      opener.partition_wal_usage = PartitionWALUsage::kNone;
      std::vector<std::string> cf_names;
      ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
      std::vector<ColumnFamilyDescriptor> cfs;
      for (const std::string& name : cf_names) {
        cfs.emplace_back(name, ColumnFamilyOptions(opener));
      }
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> read_only;
      ASSERT_OK(
          DB::OpenForReadOnly(opener, dbname_, cfs, &handles, &read_only));
      for (int round = 0; round < kRounds; ++round) {
        for (size_t p = 0; p < 2; ++p) {
          std::string value;
          EXPECT_OK(read_only->Get(ReadOptions(), handles[cf_for_partition_[p]],
                                   RoundKey(round, p), &value));
        }
      }
      for (ColumnFamilyHandle* handle : handles) {
        ASSERT_OK(read_only->DestroyColumnFamilyHandle(handle));
      }
    }
    // cf_for_partition_ keeps the N = 2 routing the keys were written with.
    const std::vector<int> written_cfs = cf_for_partition_;
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    ReopenWithColumnFamilies(cf_names, opener);
    for (int round = 0; round < kRounds; ++round) {
      for (size_t p = 0; p < 2; ++p) {
        EXPECT_EQ("value", Get(written_cfs[p], RoundKey(round, p)));
      }
    }
    FlushAllColumnFamilies();
    ReopenWithColumnFamilies(cf_names, opener);
    EXPECT_EQ("value", Get(written_cfs[1], RoundKey(1, 1)));
    Close();
  }

  // CDR-F017: kNone N = 1 WALs stay live across the upgrade to indexed N = 2.
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  FlushAllColumnFamilies();
  Options none = options;
  none.partition_wal_usage = PartitionWALUsage::kNone;
  ReopenWithPartitions(1, none);
  const std::vector<int> cfs = {static_cast<int>(handles_.size()) - 1,
                                static_cast<int>(handles_.size()) - 2};
  ASSERT_OK(Put(cfs[0], "unindexed0", "value"));
  ASSERT_OK(Put(cfs[1], "unindexed1", "value"));
  ReopenWithPartitions(2, options);
  for (const size_t p : {size_t{1}, size_t{0}}) {
    ASSERT_OK(Put(cf_for_partition_[p], "indexed" + std::to_string(p), "v"));
  }
  ReopenWithPartitions(2, options);
  EXPECT_EQ("value", Get(cfs[0], "unindexed0"));
  EXPECT_EQ("value", Get(cfs[1], "unindexed1"));
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_EQ("v", Get(cf_for_partition_[p], "indexed" + std::to_string(p)));
  }
  Close();
}

// Records the WAL files recovery reopens for writing, which it does only to
// truncate preallocated space.
class ReopenRecordingFileSystem : public FileSystemWrapper {
 public:
  explicit ReopenRecordingFileSystem(const std::shared_ptr<FileSystem>& base)
      : FileSystemWrapper(base) {}
  static const char* kClassName() { return "ReopenRecordingFileSystem"; }
  const char* Name() const override { return kClassName(); }
  IOStatus ReopenWritableFile(const std::string& fname,
                              const FileOptions& file_opts,
                              std::unique_ptr<FSWritableFile>* result,
                              IODebugContext* dbg) override {
    uint64_t number = 0;
    FileType type;
    const size_t slash = fname.find_last_of('/');
    if (ParseFileName(fname.substr(slash + 1), &number, &type) &&
        type == kWalFile) {
      std::lock_guard<std::mutex> lock(mu_);
      reopened_.insert(number);
    }
    return target()->ReopenWritableFile(fname, file_opts, result, dbg);
  }
  std::set<uint64_t> TakeReopened() {
    std::lock_guard<std::mutex> lock(mu_);
    std::set<uint64_t> reopened;
    reopened.swap(reopened_);
    return reopened;
  }

 private:
  std::mutex mu_;
  std::set<uint64_t> reopened_;
};

// G-28 (preallocation half): recovery truncates preallocated space on the
// newest WAL of each recorded partition and on no other, whether it restores
// the WALs alive or flushes them; at one partition that is still exactly the
// newest WAL. Neuter: truncate only the globally newest WAL; at N = 4 three
// partitions' newest WALs go untruncated.
TEST_P(DBWritePartitionTest, RecoveryTrimsEachPartitionsNewestWal) {
  if (GetParam()) {
    return;
  }
  auto recording_fs =
      std::make_shared<ReopenRecordingFileSystem>(env_->GetFileSystem());
  std::unique_ptr<Env> recording_env(NewCompositeEnv(recording_fs));
  for (const size_t num_partitions : {size_t{1}, size_t{4}}) {
    for (const bool restore : {true, false}) {
      SCOPED_TRACE(std::to_string(num_partitions) +
                   (restore ? " restore" : " flush"));
      Options options = IndexedWalOptions(CurrentOptions());
      options.env = recording_env.get();
      DestroyAndReopen(options);
      OpenWithPartitions(num_partitions, options);
      const std::vector<std::vector<uint64_t>> alive =
          WriteRounds(2, /*descending=*/false);
      Close();
      recording_fs->TakeReopened();
      options.avoid_flush_during_recovery = restore;
      ReopenWithPartitions(num_partitions, options);
      std::set<uint64_t> expected;
      for (const std::vector<uint64_t>& partition_wals : alive) {
        expected.insert(partition_wals.back());
      }
      EXPECT_EQ(expected, recording_fs->TakeReopened());
      Close();
    }
  }
}

// G-07 opener gap: read-only and secondary opens apply the same N > 1 option
// checks as a read-write open, with the same status and message. Every
// DBOptions check runs for them, including the write-only ones. Neuter: drop
// the check from the read-only open; its rows open.
TEST_P(DBWritePartitionTest, ReadOnlyAndSecondaryApplyPartitionChecks) {
  if (GetParam()) {
    return;
  }
  Options base = CurrentOptions();
  DestroyAndReopen(base);
  Close();
  struct Row {
    const char* name;
    std::function<void(Options*)> tweak;
    const char* message;
  };
  const std::vector<Row> rows = {
      {"kNone",
       [](Options* o) { o->partition_wal_usage = PartitionWALUsage::kNone; },
       "requires partition_wal_usage other than kNone"},
      {"track_and_verify_wals",
       [](Options* o) { o->track_and_verify_wals = true; },
       "not supported with track_and_verify_wals"},
      {"recycle_log_file_num", [](Options* o) { o->recycle_log_file_num = 1; },
       "recycle_log_file_num"},
      {"bound",
       [](Options* o) { o->wal_partitions = DBImpl::kMaxWalPartitions + 1; },
       "wal_partitions must be at most"},
      {"tracking",
       [](Options* o) { o->track_and_verify_wals_in_manifest = false; },
       "wal_partitions > 1 requires track_and_verify_wals_in_manifest"},
  };
  for (const Row& row : rows) {
    SCOPED_TRACE(row.name);
    Options options = base;
    options.create_if_missing = false;
    options.wal_partitions = 2;
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    options.track_and_verify_wals_in_manifest = true;
    row.tweak(&options);
    const Status read_write = TryReopen(options);
    std::unique_ptr<DB> db;
    const Status read_only = DB::OpenForReadOnly(options, dbname_, &db);
    const Status secondary =
        DB::OpenAsSecondary(options, dbname_, dbname_ + "_secondary", &db);
    for (const Status& s : {read_write, read_only, secondary}) {
      EXPECT_EQ(read_write.code(), s.code()) << s.ToString();
      EXPECT_NE(std::string::npos, s.ToString().find(row.message))
          << s.ToString();
      EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"))
          << s.ToString();
    }
  }
  // A configuration a read-write open accepts opens read-only too.
  Options options = base;
  options.wal_partitions = 2;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  std::unique_ptr<DB> db;
  ASSERT_OK(DB::OpenForReadOnly(options, dbname_, &db));
}

// G-07 (GC12 Q8): a read-only open of a fully compacted DB takes the
// CompactedDBImpl path first; it applies the same wal_partitions checks as
// the other read-only opens. Neuter: run the checks after the compacted
// attempt again; the kNone row opens.
TEST_P(DBWritePartitionTest, ReadOnlyCompactedOpenAppliesPartitionChecks) {
  if (GetParam()) {
    return;
  }
  Options options = CurrentOptions();
  options.max_open_files = -1;
  DestroyAndReopen(options);
  ASSERT_OK(Put("key", "value"));
  ASSERT_OK(Flush());
  Close();
  // A configuration the checks accept opens through the compacted path.
  options.wal_partitions = 2;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  std::unique_ptr<DB> db;
  ASSERT_OK(DB::OpenForReadOnly(options, dbname_, &db));
  EXPECT_NE(nullptr, dynamic_cast<CompactedDBImpl*>(db.get()));
  std::string value;
  ASSERT_OK(db->Get(ReadOptions(), "key", &value));
  db.reset();

  options.partition_wal_usage = PartitionWALUsage::kNone;
  const Status s = DB::OpenForReadOnly(options, dbname_, &db);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos,
            s.ToString().find("requires partition_wal_usage other than kNone"))
      << s.ToString();
}

// G-30 / G-31 / F014: a checkpoint or backup at N > 1 copies every
// partition's current WAL, not only partition 0's (the lowest numbered), so a
// copy taken without a flush, or with writes after its flush's memtable
// switch, holds every write acknowledged before the capture. Neuter: bound the
// copied WALs by cur_wal_number_ again; the copies lose partitions 1..N-1.
TEST_P(DBWritePartitionTest, CheckpointAndBackupCopyEveryPartitionsWal) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE(num_partitions);
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    std::vector<std::pair<size_t, std::string>> keys;
    for (size_t p = 0; p < num_partitions; ++p) {
      keys.emplace_back(p, "before" + std::to_string(p));
      ASSERT_OK(Put(cf_for_partition_[p], keys.back().second, "v"));
    }

    LiveFilesStorageInfoOptions live_options;
    live_options.wal_size_for_flush = std::numeric_limits<uint64_t>::max();
    std::vector<LiveFileStorageInfo> live_files;
    ASSERT_OK(db_->DisableFileDeletions());
    ASSERT_OK(db_->GetLiveFilesStorageInfo(live_options, &live_files));
    ASSERT_OK(db_->EnableFileDeletions());
    std::set<uint64_t> listed_wals;
    for (const LiveFileStorageInfo& info : live_files) {
      if (info.file_type == kWalFile) {
        listed_wals.insert(info.file_number);
      }
    }
    for (size_t p = 0; p < num_partitions; ++p) {
      EXPECT_EQ(1U, listed_wals.count(dbfull()->TEST_CurrentWalNumber(p)))
          << "partition " << p;
    }

    for (const bool flush : {false, true}) {
      SCOPED_TRACE(flush);
      const std::string checkpoint_dir = dbname_ + "_checkpoint";
      const std::string backup_dir = dbname_ + "_backup";
      const std::string restore_dir = dbname_ + "_restore";
      ASSERT_OK(DestroyDB(checkpoint_dir, options));
      ASSERT_OK(DestroyDB(restore_dir, options));
      ASSERT_OK(DestroyDir(env_, backup_dir));
      int capture = 0;
      if (flush) {
        // Lands in the WALs the flush's memtable switch created, after the
        // switch and before the WALs are copied.
        SyncPoint::GetInstance()->SetCallBack(
            "CheckpointImpl::CreateCheckpoint:FlushDone", [&](void*) {
              ++capture;
              for (size_t p = 0; p < num_partitions; ++p) {
                const std::string key =
                    "after" + std::to_string(capture) + "_" + std::to_string(p);
                ASSERT_OK(Put(cf_for_partition_[p], key, "v"));
                keys.emplace_back(p, key);
              }
            });
      }
      Checkpoint* checkpoint = nullptr;
      ASSERT_OK(Checkpoint::Create(db_.get(), &checkpoint));
      ASSERT_OK(checkpoint->CreateCheckpoint(
          checkpoint_dir, flush ? 0 : std::numeric_limits<uint64_t>::max()));
      delete checkpoint;
      const std::vector<std::pair<size_t, std::string>> checkpoint_keys = keys;

      BackupEngine* backup_engine = nullptr;
      ASSERT_OK(BackupEngine::Open(env_, BackupEngineOptions(backup_dir),
                                   &backup_engine));
      CreateBackupOptions backup_options;
      backup_options.flush_before_backup = flush;
      ASSERT_OK(backup_engine->CreateNewBackup(backup_options, db_.get()));
      ASSERT_OK(
          backup_engine->RestoreDBFromLatestBackup(restore_dir, restore_dir));
      delete backup_engine;
      SyncPoint::GetInstance()->ClearCallBack(
          "CheckpointImpl::CreateCheckpoint:FlushDone");
      if (flush) {
        ASSERT_EQ(2, capture);
      }

      EXPECT_EQ(std::vector<std::string>(checkpoint_keys.size(), "v"),
                ReadFromCopy(checkpoint_dir, options, checkpoint_keys));
      EXPECT_EQ(std::vector<std::string>(keys.size(), "v"),
                ReadFromCopy(restore_dir, options, keys));
    }
    Close();
  }
}

// G-30 / G-31 under concurrency (TSan): writers on all eight partitions keep
// writing while a checkpoint is taken without a flush. The reopened copy holds
// every write acknowledged before the capture began, and what it holds is a
// prefix of the source's sequence order. Neuter: as above; the copy loses the
// acknowledged writes of partitions 1..7.
TEST_P(DBWritePartitionTest, ConcurrentCheckpointIsAPrefixOfAckedWrites) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kMaxWritesPerPartition = 3000;
  Options options = IndexedWalOptions(CurrentOptions());
  options.write_buffer_size = 64 << 20;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  auto key_of = [](size_t p, int i) {
    return "w" + std::to_string(p) + "_" + std::to_string(i);
  };
  std::vector<std::atomic<int>> acked(kPartitions);
  std::atomic<bool> stop{false};
  std::vector<port::Thread> writers;
  for (size_t p = 0; p < kPartitions; ++p) {
    acked[p].store(0);
    writers.emplace_back([&, p] {
      for (int i = 0; i < kMaxWritesPerPartition && !stop.load(); ++i) {
        ASSERT_OK(Put(cf_for_partition_[p], key_of(p, i), "v"));
        acked[p].store(i + 1);
      }
    });
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    while (acked[p].load() < 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  std::vector<int> acked_before(kPartitions);
  for (size_t p = 0; p < kPartitions; ++p) {
    acked_before[p] = acked[p].load();
  }
  const std::string checkpoint_dir = dbname_ + "_checkpoint";
  ASSERT_OK(DestroyDB(checkpoint_dir, options));
  Checkpoint* checkpoint = nullptr;
  ASSERT_OK(Checkpoint::Create(db_.get(), &checkpoint));
  ASSERT_OK(checkpoint->CreateCheckpoint(checkpoint_dir,
                                         std::numeric_limits<uint64_t>::max()));
  delete checkpoint;
  stop.store(true);
  for (auto& writer : writers) {
    writer.join();
  }

  // Every write's sequence in the source, in sequence order.
  std::map<SequenceNumber, std::pair<size_t, std::string>> by_sequence;
  for (size_t p = 0; p < kPartitions; ++p) {
    std::vector<KeyVersion> versions;
    ASSERT_OK(GetAllKeyVersions(db_.get(), handles_[cf_for_partition_[p]], {},
                                {}, std::numeric_limits<size_t>::max(),
                                &versions));
    for (const KeyVersion& version : versions) {
      by_sequence.emplace(version.sequence,
                          std::make_pair(p, version.user_key));
    }
  }
  std::vector<std::pair<size_t, std::string>> keys;
  for (const auto& [sequence, key] : by_sequence) {
    keys.push_back(key);
  }
  const std::vector<std::string> values =
      ReadFromCopy(checkpoint_dir, options, keys);
  ASSERT_EQ(keys.size(), values.size());
  std::set<std::string> present;
  size_t prefix_end = 0;
  while (prefix_end < values.size() && values[prefix_end] == "v") {
    present.insert(keys[prefix_end].second);
    ++prefix_end;
  }
  for (size_t i = prefix_end; i < values.size(); ++i) {
    EXPECT_EQ("NOT_FOUND", values[i]) << keys[i].second << " past the prefix";
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    for (int i = 0; i < acked_before[p]; ++i) {
      EXPECT_EQ(1U, present.count(key_of(p, i))) << key_of(p, i);
    }
  }
  Close();
}

// G-30 (F014): GetCurrentWalFile cannot name one current WAL at N > 1, so it
// refuses rather than returning partition 0's. N=1 is unchanged.
TEST_P(DBWritePartitionTest, GetCurrentWalFileRefusesAtNAboveOne) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(1, options);
  ASSERT_OK(Put(cf_for_partition_[0], "key", "v"));
  std::unique_ptr<WalFile> wal;
  ASSERT_OK(db_->GetCurrentWalFile(&wal));
  EXPECT_EQ(dbfull()->TEST_CurrentWalNumber(0), wal->LogNumber());
  EXPECT_EQ(kAliveLogFile, wal->Type());

  ReopenWithPartitions(2, options);
  wal.reset();
  const Status s = db_->GetCurrentWalFile(&wal);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"));
  EXPECT_EQ(nullptr, wal);
}

// G-10 and N-2-1 (F011): A on partition 0 allocates the lower sequence and
// is held either inside its WAL append or between its append and its memtable
// insert; B on partition 1 (the higher sequence) finishes its insert. Nothing
// at or above A's sequence is visible until A is released: not B through
// GetLatestSequenceNumber(), a new snapshot, a plain Get or an iterator, and
// B's Put does not return. A snapshot taken while A is held reads the same
// before and after A lands.
TEST_P(DBWritePartitionTest, PublicationStopsBelowHeldLowerWrite) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  for (const bool hold_in_append : {true, false}) {
    SCOPED_TRACE(hold_in_append ? "held in append" : "held before insert");
    if (!hold_in_append) {
      Destroy(options);
      Reopen(options);
    }
    OpenWithPartitions(kPartitions, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    ASSERT_OK(Put(cf_a, "seed", "v"));
    const SequenceNumber base = db_->GetLatestSequenceNumber();
    const SequenceNumber seq_a = base + 1;
    const SequenceNumber seq_b = base + 2;

    std::mutex mu;
    std::condition_variable cv;
    bool a_held = false;
    bool release_a = false;
    std::atomic<bool> b_returned{false};
    Status a_status;
    Status b_status;
    // A is the next WAL write, so it is the first to reach either point.
    SyncPoint::GetInstance()->SetCallBack(
        hold_in_append ? "DBImpl::WriteToWAL:log_entry"
                       : "DBImpl::InstallCompletionGate",
        [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (a_held) {
            return;
          }
          a_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    port::Thread a_thread([&] { a_status = Put(cf_a, "key_a", "value_a"); });
    {
      std::unique_lock<std::mutex> lock(mu);
      ASSERT_TRUE(cv.wait_until(lock, deadline, [&] { return a_held; }));
    }
    port::Thread b_thread([&] {
      b_status = Put(cf_b, "key_b", "value_b");
      b_returned.store(true);
    });
    // B parks in its completion gate only after its memtable insert.
    ASSERT_TRUE(WaitForDurabilityWaiters(1));
    if (!hold_in_append) {
      // A's record landed, so the durability safe point alone would admit
      // B; only the applied bound keeps it hidden.
      EXPECT_GE(dbfull()->WalDurabilitySafePoint(), seq_b);
    }

    EXPECT_LT(db_->GetLatestSequenceNumber(), seq_a);
    const Snapshot* snapshot = db_->GetSnapshot();
    EXPECT_LT(snapshot->GetSequenceNumber(), seq_a);
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b", snapshot));
    EXPECT_EQ("NOT_FOUND", Get(cf_a, "key_a", snapshot));
    {
      std::unique_ptr<Iterator> it(
          db_->NewIterator(ReadOptions(), handles_[cf_b]));
      it->SeekToFirst();
      EXPECT_FALSE(it->Valid());
      ASSERT_OK(it->status());
    }
    EXPECT_FALSE(b_returned.load());

    {
      std::lock_guard<std::mutex> lock(mu);
      release_a = true;
      cv.notify_all();
    }
    a_thread.join();
    b_thread.join();
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
    ASSERT_OK(a_status);
    ASSERT_OK(b_status);

    // The snapshot reads exactly what it read while A was held.
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b", snapshot));
    EXPECT_EQ("NOT_FOUND", Get(cf_a, "key_a", snapshot));
    db_->ReleaseSnapshot(snapshot);
    EXPECT_EQ(seq_b, db_->GetLatestSequenceNumber());
    EXPECT_EQ("value_a", Get(cf_a, "key_a"));
    EXPECT_EQ("value_b", Get(cf_b, "key_b"));
  }
}

// N-2-2: with more than one WAL partition, publishing above the allocated
// sequence fails the debug assertion even without two_write_queues.
TEST_P(DBWritePartitionTest, PublishAboveAllocatedAsserts) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
#ifdef NDEBUG
  ROCKSDB_GTEST_BYPASS("the assertion exists in debug builds only");
#else
#ifdef GTEST_FLAG_SET
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#else
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(2, options);
  ASSERT_OK(Put(cf_for_partition_[1], "key", "value"));
  VersionSet* versions = dbfull()->GetVersionSet();
  const SequenceNumber above = versions->LastAllocatedSequence() + 1;
  EXPECT_DEATH(versions->SetLastSequence(above), "");
  EXPECT_DEATH(versions->SetLastSequenceAtLeast(above), "");
#endif
}

// A sync write's record lands, then recording the synced WAL in the MANIFEST
// fails, so the write returns an error without inserting anything. Its range
// must not stay pending in the applied prefix: a later write on the other
// partition returns and is visible, instead of parking for good in its
// completion gate (the M2 prefix-oracle hang, seed 20261023).
TEST_P(DBWritePartitionTest, FailedSyncTrackingReleasesItsRange) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  Defer close_before_env([this] { Close(); });
  Options options = CurrentOptions();
  options.env = fault_fs_env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  // Only a retired WAL is recorded in the MANIFEST when synced, so the sync
  // below must cover one.
  ASSERT_OK(Put(cf_a, "unsynced", "value"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable(
      static_cast_with_check<ColumnFamilyHandleImpl>(handles_[cf_a])->cfd()));

  // An inactive fault FS fails the write with Corruption, not an IOError, so
  // no background error is recorded and nothing else wakes a parked gate.
  std::atomic<bool> fail_manifest{true};
  SyncPoint::GetInstance()->SetCallBack(
      "VersionSet::LogAndApply:WriteManifestStart", [&](void*) {
        if (fail_manifest.exchange(false)) {
          fault_fs->SetFilesystemActive(false);
        }
      });
  WriteOptions sync_write;
  sync_write.sync = true;
  ASSERT_NOK(Put(cf_a, "a", "value", sync_write));
  ASSERT_FALSE(fail_manifest.load());
  fault_fs->SetFilesystemActive(true);
  SyncPoint::GetInstance()->ClearCallBack(
      "VersionSet::LogAndApply:WriteManifestStart");

  std::mutex mu;
  std::condition_variable cv;
  bool b_returned = false;
  Status b_status;
  port::Thread b_writer([&] {
    const Status s = Put(cf_b, "b", "value");
    std::lock_guard<std::mutex> lock(mu);
    b_status = s;
    b_returned = true;
    cv.notify_all();
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, std::chrono::seconds(30),
                     [&] { return b_returned; })) {
      // Nothing can release a parked gate now; fail instead of hanging.
      fprintf(stderr, "write on partition 1 never returned\n");
      std::abort();
    }
  }
  b_writer.join();
  ASSERT_OK(b_status);
  EXPECT_EQ("value", Get(cf_b, "b"));
  EXPECT_EQ(dbfull()->GetVersionSet()->LastAllocatedSequence(),
            db_->GetLatestSequenceNumber());
}

// G-10: an ingestion jumps LastSequence() forward outside the write path;
// partition leaders publishing concurrently afterwards never move it back,
// and every acknowledged write is published.
TEST_P(DBWritePartitionTest, IngestionThenWritesKeepLastSequenceMonotone) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr int kWritesPerPartition = 200;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[1];
  ASSERT_OK(Put(cf, "key", "old"));
  const SequenceNumber before_ingest = db_->GetLatestSequenceNumber();
  const std::string sst = dbname_ + "/ingest.sst";
  SstFileWriter writer(EnvOptions(), options);
  ASSERT_OK(writer.Open(sst));
  ASSERT_OK(writer.Put("key", "ingested"));
  ASSERT_OK(writer.Finish());
  // Overlaps the memtable, so the file takes a global sequence number.
  ASSERT_OK(db_->IngestExternalFile(handles_[cf], {sst},
                                    IngestExternalFileOptions()));
  const SequenceNumber after_ingest = db_->GetLatestSequenceNumber();
  ASSERT_EQ(before_ingest + 1, after_ingest);
  ASSERT_EQ("ingested", Get(cf, "key"));

  std::atomic<bool> writers_done{false};
  std::atomic<bool> went_backwards{false};
  port::Thread sampler([&] {
    SequenceNumber last = db_->GetLatestSequenceNumber();
    while (!writers_done.load()) {
      const SequenceNumber now = db_->GetLatestSequenceNumber();
      if (now < last) {
        went_backwards.store(true);
      }
      last = now;
    }
  });
  std::vector<port::Thread> writers;
  for (size_t p = 0; p < kPartitions; ++p) {
    writers.emplace_back([&, p] {
      for (int i = 0; i < kWritesPerPartition; ++i) {
        const std::string key =
            "p" + std::to_string(p) + "_" + std::to_string(i);
        ASSERT_OK(Put(cf_for_partition_[p], key, "v"));
        // Read-your-writes: an acknowledged write is published.
        ASSERT_EQ("v", Get(cf_for_partition_[p], key));
      }
    });
  }
  for (auto& w : writers) {
    w.join();
  }
  writers_done.store(true);
  sampler.join();
  EXPECT_FALSE(went_backwards.load());
  EXPECT_EQ(after_ingest + kPartitions * kWritesPerPartition,
            db_->GetLatestSequenceNumber());
}

// G-35: ingestion quiesces every partition's queue, so no durability range
// is pending when it jumps the published sequence. Writers on all eight
// partitions run through 100 ingestions alternating between files that overlap
// a writer's memtable range (flushed first, global sequence number) and files
// past it. Under TSan this is the race check; every acknowledged write and
// every ingested key reads back before and after a reopen.
TEST_P(DBWritePartitionTest, ConcurrentIngestionKeepsWritesAndIngestedKeys) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kIngestions = 100;
  constexpr int kMaxWritesPerPartition = 2000;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  auto key_of = [](size_t p, int i) {
    return "w" + std::to_string(p) + "_" + std::to_string(i);
  };
  std::vector<std::atomic<int>> acked(kPartitions);
  std::atomic<bool> stop{false};
  std::vector<port::Thread> writers;
  for (size_t p = 0; p < kPartitions; ++p) {
    acked[p].store(0);
    writers.emplace_back([&, p] {
      for (int i = 0; i < kMaxWritesPerPartition && !stop.load(); ++i) {
        ASSERT_OK(Put(cf_for_partition_[p], key_of(p, i), "v"));
        acked[p].store(i + 1);
      }
    });
  }
  std::vector<std::pair<size_t, std::string>> ingested;
  for (int n = 0; n < kIngestions; ++n) {
    const size_t p = static_cast<size_t>(n) % kPartitions;
    // "w<p>_ingest<n>" sorts inside the writer's keys; "x..." sorts past them.
    const std::string key =
        n % 2 == 0 ? "w" + std::to_string(p) + "_ingest" + std::to_string(n)
                   : "x_ingest" + std::to_string(n);
    const std::string sst = dbname_ + "/ingest" + std::to_string(n) + ".sst";
    SstFileWriter writer(EnvOptions(), options);
    ASSERT_OK(writer.Open(sst));
    ASSERT_OK(writer.Put(key, "ingested"));
    ASSERT_OK(writer.Finish());
    ASSERT_OK(db_->IngestExternalFile(handles_[cf_for_partition_[p]], {sst},
                                      IngestExternalFileOptions()));
    ingested.emplace_back(p, key);
  }
  stop.store(true);
  for (auto& writer : writers) {
    writer.join();
  }
  auto check = [&] {
    for (size_t p = 0; p < kPartitions; ++p) {
      for (int i = 0; i < acked[p].load(); ++i) {
        ASSERT_EQ("v", Get(cf_for_partition_[p], key_of(p, i)))
            << p << " " << i;
      }
    }
    for (const auto& [p, key] : ingested) {
      ASSERT_EQ("ingested", Get(cf_for_partition_[p], key)) << key;
    }
  };
  check();
  ReopenWithPartitions(kPartitions, options);
  check();
}

// G-35: the invariant ingestion asserts is that no partition's range is
// pending, not that the safe point equals the allocated sequence: after a
// disableWAL write the safe point stays below allocated, yet nothing is
// pending and ingestion proceeds. Neuter: assert safe point == allocated
// instead; this ingestion fires it (debug builds).
TEST_P(DBWritePartitionTest, IngestionAfterDisableWalWriteFindsRangesIdle) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  Options options = CurrentOptions();
  OpenWithPartitions(2, options);
  const int cf = cf_for_partition_[1];
  ASSERT_OK(Put(cf_for_partition_[0], "logged", "v"));
  ASSERT_OK(DisableWalPut(cf, "key", "unlogged"));
  ASSERT_LT(dbfull()->WalDurabilitySafePoint(),
            dbfull()->GetVersionSet()->LastAllocatedSequence());
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_EQ(WalDurabilityTracker::kIdle,
              dbfull()->TEST_WalDurabilityPendingLow(p));
  }
  const std::string sst = dbname_ + "/ingest.sst";
  SstFileWriter writer(EnvOptions(), options);
  ASSERT_OK(writer.Open(sst));
  ASSERT_OK(writer.Put("key", "ingested"));
  ASSERT_OK(writer.Finish());
  // Overlaps the memtable, so the file takes a global sequence number and
  // ingestion jumps the published sequence.
  const SequenceNumber before = db_->GetLatestSequenceNumber();
  ASSERT_OK(db_->IngestExternalFile(handles_[cf], {sst},
                                    IngestExternalFileOptions()));
  EXPECT_EQ(before + 1, db_->GetLatestSequenceNumber());
  EXPECT_EQ("ingested", Get(cf, "key"));
}

// G-35: a failed WAL append leaves its range pending for good (the burned
// index rule). Ingestion finds it after quiescing and fails instead of
// publishing sequences above that hole: with the background error when
// paranoid_checks set one, else Aborted naming the partition. Neuter: drop
// the check; ingestion proceeds and, in debug builds, the idle assertion
// fires.
TEST_P(DBWritePartitionTest, IngestionFailsCleanlyOverPinnedFailedAppend) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  // paranoid_checks=false is rejected at Open above one partition (GC08,
  // D-1), so only the paranoid leg remains.
  {
    Options options = CurrentOptions();
    options.paranoid_checks = true;
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    const int cf = cf_for_partition_[1];
    ASSERT_OK(Put(cf, "key", "v"));
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected WAL append error");
        });
    ASSERT_NOK(Put(cf, "failed", "v"));
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord");
    ASSERT_NE(WalDurabilityTracker::kIdle,
              dbfull()->TEST_WalDurabilityPendingLow(1));

    const std::string sst = dbname_ + "/ingest.sst";
    SstFileWriter writer(EnvOptions(), options);
    ASSERT_OK(writer.Open(sst));
    ASSERT_OK(writer.Put("key", "ingested"));
    ASSERT_OK(writer.Finish());
    const SequenceNumber before = db_->GetLatestSequenceNumber();
    const Status s = db_->IngestExternalFile(handles_[cf], {sst},
                                             IngestExternalFileOptions());
    ASSERT_NOK(s);
    EXPECT_EQ(dbfull()->TEST_GetBGError().ToString(), s.ToString());
    EXPECT_EQ(before, db_->GetLatestSequenceNumber());
    EXPECT_EQ("v", Get(cf, "key"));
    Close();
  }
}

// G-34: the change stream stops at the same published prefix. With A held in
// its append and B (higher, other partition) inserted, the published sequence
// stays below A, and GetUpdatesSince, which refuses every call at N > 1
// (G-32, D-G32 (a)), returns nothing. After a crash that drops unsynced data
// the writes acknowledged before A are recovered.
TEST_P(DBWritePartitionTest, ChangeStreamStopsBelowHeldLowerWrite) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  fault_fs->SetInjectUnsyncedDataLoss(true);
  std::unique_ptr<Env> env(NewCompositeEnv(fault_fs));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = env.get();
  options.WAL_ttl_seconds = 1000;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_a, "k0", "v", sync_wo));
  const SequenceNumber first = db_->GetLatestSequenceNumber();
  ASSERT_OK(Put(cf_b, "k1", "v", sync_wo));
  const SequenceNumber seq_a = db_->GetLatestSequenceNumber() + 1;

  std::mutex mu;
  std::condition_variable cv;
  bool a_held = false;
  bool release_a = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (a_held) {
          return;
        }
        a_held = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_a; });
      });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  port::Thread a_thread(
      [&] { Put(cf_a, "key_a", "v").PermitUncheckedError(); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_until(lock, deadline, [&] { return a_held; }));
  }
  port::Thread b_thread(
      [&] { Put(cf_b, "key_b", "v").PermitUncheckedError(); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_LT(db_->GetLatestSequenceNumber(), seq_a);

  // D-G32 (a): at N > 1 GetUpdatesSince refuses before reading any file, so
  // no change stream exists to run past A.
  {
    std::unique_ptr<TransactionLogIterator> iter;
    const Status s = db_->GetUpdatesSince(first, &iter);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  }

  // Crash before A lands: A's append fails and B is never acknowledged.
  fault_fs->SetFilesystemActive(false);
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  Close();
  ASSERT_OK(fault_fs->DropUnsyncedFileData());
  fault_fs->SetFilesystemActive(true);
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "k0"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "k1"));
}

// G-36: a secondary instance replays WALs one file at a time, which is not
// the write order of a set written with wal_partitions > 1. It refuses such a
// set at open and at catch-up, before applying any of it, rather than
// exposing partition 0's records without the ones interleaved with them. N=1
// catches up as before. Neuter: drop the head check when a WAL's reader is
// created; the catch-up applies partition 0's WAL and exposes "n8_p0" before
// the in-loop check refuses partition 1's.
TEST_P(DBWritePartitionTest, SecondaryRejectsPartitionedWalsWithoutApplying) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 8;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  // Created at N=1, enough column families to route to every partition at
  // N=8.
  std::vector<std::string> new_cfs;
  std::vector<bool> covered(kPartitions, false);
  size_t num_covered = 0;
  for (uint32_t cf_id = 1; num_covered < kPartitions; ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = kPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kPartitions, &partition));
    new_cfs.push_back("cf" + std::to_string(cf_id));
    if (!covered[partition]) {
      covered[partition] = true;
      ++num_covered;
    }
  }
  CreateAndReopenWithCF(new_cfs, options);
  ASSERT_OK(Put(1, "n1", "v"));
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  std::vector<ColumnFamilyDescriptor> column_families;
  for (const std::string& name : cf_names) {
    column_families.emplace_back(name, options);
  }
  Options secondary_options = options;
  secondary_options.max_open_files = -1;
  const std::string secondary_path = dbname_ + "_secondary";
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> secondary;
  ASSERT_OK(DB::OpenAsSecondary(secondary_options, dbname_, secondary_path,
                                column_families, &handles, &secondary));
  std::string value;
  ASSERT_OK(secondary->Get(ReadOptions(), handles[1], "n1", &value));

  ReopenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "n8_p" + std::to_string(p), "v"));
  }
  Status s = secondary->TryCatchUpWithPrimary();
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"));
  for (size_t p = 0; p < kPartitions; ++p) {
    EXPECT_TRUE(secondary
                    ->Get(ReadOptions(), handles[cf_for_partition_[p]],
                          "n8_p" + std::to_string(p), &value)
                    .IsNotFound())
        << p;
  }
  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(secondary->DestroyColumnFamilyHandle(handle));
  }
  handles.clear();
  secondary.reset();

  s = DB::OpenAsSecondary(secondary_options, dbname_, secondary_path,
                          column_families, &handles, &secondary);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"));
}

// G-36: a read-only opener writing no partitions but indexing its WALs opens
// a crash image written at N=8 through the merge, and sees a prefix holding
// every synced write. Each write is a single put from one thread, so sequence
// order is write order.
TEST_P(DBWritePartitionTest, ReadOnlyOpenerRecoversPartitionedCrashPrefix) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kWrites = 64;
  constexpr int kSynced = 40;
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  fault_fs->SetInjectUnsyncedDataLoss(true);
  std::unique_ptr<Env> env(NewCompositeEnv(fault_fs));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = env.get();
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  std::vector<std::pair<size_t, std::string>> keys;
  for (int i = 0; i < kWrites; ++i) {
    const size_t p = static_cast<size_t>(i * 3) % kPartitions;
    keys.emplace_back(p, "k" + std::to_string(i));
    WriteOptions write_options;
    write_options.sync = i == kSynced - 1;
    ASSERT_OK(
        Put(cf_for_partition_[p], keys.back().second, "v", write_options));
  }
  fault_fs->SetFilesystemActive(false);
  Close();
  ASSERT_OK(fault_fs->DropUnsyncedFileData());
  fault_fs->SetFilesystemActive(true);

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  std::vector<ColumnFamilyDescriptor> column_families;
  for (const std::string& name : cf_names) {
    column_families.emplace_back(name, options);
  }
  options.wal_partitions = 1;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> read_only;
  ASSERT_OK(DB::OpenForReadOnly(options, dbname_, column_families, &handles,
                                &read_only));
  int prefix_end = 0;
  bool past_prefix = false;
  for (int i = 0; i < kWrites; ++i) {
    std::string value;
    const Status get =
        read_only->Get(ReadOptions(), handles[cf_for_partition_[keys[i].first]],
                       keys[i].second, &value);
    ASSERT_TRUE(get.ok() || get.IsNotFound()) << get.ToString();
    if (get.ok()) {
      EXPECT_FALSE(past_prefix) << keys[i].second << " past the prefix";
      prefix_end = i + 1;
    } else {
      past_prefix = true;
    }
  }
  EXPECT_GE(prefix_end, kSynced);
  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(read_only->DestroyColumnFamilyHandle(handle));
  }
}

// G-36: RepairDB converts each WAL to a table on its own, so on a set written
// with wal_partitions > 1 it could keep records above a hole. It refuses the
// set before changing anything; at N=1 it repairs as before. Neuter: drop
// the check; RepairDB returns OK and rewrites the MANIFEST.
TEST_P(DBWritePartitionTest, RepairDBRejectsPartitionedWalsUnchanged) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  // WAL, table, MANIFEST, CURRENT and OPTIONS files with their sizes: the DB,
  // as opposed to its LOCK and info logs.
  auto db_files = [this] {
    std::vector<std::string> children;
    EXPECT_OK(env_->GetChildren(dbname_, &children));
    std::map<std::string, uint64_t> files;
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (ParseFileName(child, &number, &type) &&
          (type == kWalFile || type == kTableFile || type == kDescriptorFile ||
           type == kCurrentFile || type == kOptionsFile)) {
        uint64_t size = 0;
        EXPECT_OK(env_->GetFileSize(dbname_ + "/" + child, &size));
        files[child] = size;
      }
    }
    return files;
  };
  for (const size_t num_partitions : {size_t{1}, size_t{8}}) {
    SCOPED_TRACE(num_partitions);
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    for (size_t p = 0; p < num_partitions; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p], "key" + std::to_string(p), "v"));
    }
    // RepairDB salvages WAL records only for column families it found tables
    // for, which the default column family always is.
    ASSERT_OK(Put(0, "default_key", "v"));
    Close();
    const std::map<std::string, uint64_t> before = db_files();
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    const Status s = RepairDB(dbname_, options, column_families);
    if (num_partitions == 1) {
      ASSERT_OK(s);
      ReopenWithPartitions(1, options);
      EXPECT_EQ("v", Get(0, "default_key"));
      Close();
    } else {
      ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
      EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"));
      EXPECT_EQ(before, db_files());
    }
  }
}

// G-36: in indexed recovery a WalFilter sees records in the merged order,
// interleaved across files, and log_number names each record's own file.
TEST_P(DBWritePartitionTest, WalFilterSeesMergedOrderWithEachRecordsFile) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kRounds = 3;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  WriteRounds(kRounds, /*descending=*/true);
  Close();

  // The file each sequence was written to.
  std::map<SequenceNumber, uint64_t> file_of;
  for (const uint64_t wal : WalNumbersOnDisk()) {
    std::unique_ptr<SequentialFileReader> file_reader;
    ASSERT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), LogFileName(dbname_, wal), FileOptions(),
        &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, wal);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
      WriteBatch batch;
      ASSERT_OK(WriteBatchInternal::SetContents(&batch, record));
      if (WriteBatchInternal::Count(&batch) > 0) {
        file_of[WriteBatchInternal::Sequence(&batch)] = wal;
      }
    }
  }
  ASSERT_EQ(kPartitions * kRounds, file_of.size());

  class RecordingFilter : public WalFilter {
   public:
    WalProcessingOption LogRecordFound(unsigned long long log_number,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& batch,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      if (WriteBatchInternal::Count(&batch) > 0) {
        calls.emplace_back(WriteBatchInternal::Sequence(&batch), log_number);
      }
      return WalProcessingOption::kContinueProcessing;
    }
    const char* Name() const override { return "RecordingFilter"; }
    std::vector<std::pair<SequenceNumber, uint64_t>> calls;
  } filter;
  options.wal_filter = &filter;
  ReopenWithPartitions(kPartitions, options);
  ASSERT_EQ(file_of.size(), filter.calls.size());
  std::set<uint64_t> files_seen;
  for (size_t i = 0; i < filter.calls.size(); ++i) {
    const auto& [sequence, log_number] = filter.calls[i];
    if (i > 0) {
      EXPECT_LT(filter.calls[i - 1].first, sequence);
    }
    EXPECT_EQ(file_of[sequence], log_number) << sequence;
    files_seen.insert(log_number);
  }
  EXPECT_GT(files_seen.size(), kPartitions);
}

// G-32 / D-G32 (a): GetUpdatesSince refuses before reading any file at N > 1,
// and an N=1 opener whose retained or archived WALs were written at N=4 gets
// the same explicit status instead of "Gap in sequence numbers" at the first
// partition boundary. Neuters: drop the check in DBImpl::GetUpdatesSince, and
// the N=2 status comes from the WAL scan instead; drop the head check in
// WalManager::GetUpdatesSince, and the N=1 reopen's iterator fails with "Gap
// in sequence numbers".
TEST_P(DBWritePartitionTest, GetUpdatesSinceRefusesPartitionedWals) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  options.WAL_ttl_seconds = 1000;
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  ASSERT_OK(Put(cf_for_partition_[0], "key", "v"));
  std::unique_ptr<TransactionLogIterator> iter;
  Status s = db_->GetUpdatesSince(1, &iter);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  // The open DB's own N refuses before any file is read.
  EXPECT_NE(std::string::npos,
            s.ToString().find("not supported with wal_partitions > 1"))
      << s.ToString();
  EXPECT_EQ(nullptr, iter);

  ReopenWithPartitions(4, options);
  WriteRounds(2, /*descending=*/true);
  // Retires the first round's WALs into the archive.
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  WriteRounds(2, /*descending=*/true);
  ReopenWithPartitions(1, options);
  std::vector<std::string> archived;
  ASSERT_OK(env_->GetChildren(ArchivalDirectory(dbname_), &archived));
  EXPECT_FALSE(archived.empty());
  s = db_->GetUpdatesSince(1, &iter);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"));
  EXPECT_EQ(std::string::npos, s.ToString().find("Gap"));
}

// N-5-4: WritePrepared without two write queues at N=4. Each transaction's
// prepare routes by its column family and its commit marker by its XID, here
// always to another partition. From a crash image, committed and rolled-back
// transactions recover to their outcome, and the ones prepared before the
// crash recover as prepared and can then be committed or rolled back. Merged
// recovery replays in sequence order, so each prepare is replayed before its
// commit whichever partition holds them.
TEST_P(DBWritePartitionTest, WritePreparedRecoversAcrossPartitions) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  Close();

  int next_xid = 0;
  // An XID whose commit marker routes away from `prepare_partition`.
  auto xid_committing_elsewhere = [&](size_t prepare_partition) {
    while (true) {
      const std::string xid = "xid" + std::to_string(next_xid++);
      WriteBatch commit;
      EXPECT_OK(WriteBatchInternal::MarkCommit(&commit, xid));
      size_t partition = kPartitions;
      EXPECT_OK(
          DBImpl::TEST_WalPartitionForWrite(commit, kPartitions, &partition));
      if (partition != prepare_partition) {
        return xid;
      }
    }
  };
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<TransactionDB> db =
      OpenWritePreparedTxnDB(dbname_, kPartitions, options, &handles);
  ASSERT_NE(nullptr, db);
  WriteOptions sync;
  sync.sync = true;
  std::map<std::string, size_t> partition_of;
  auto prepare = [&](size_t p, const std::string& key, const std::string& xid) {
    partition_of[key] = p;
    Transaction* txn = db->BeginTransaction(sync);
    EXPECT_OK(txn->SetName(xid));
    EXPECT_OK(txn->Put(handles[cf_for_partition_[p]], key, "v"));
    EXPECT_OK(txn->Prepare());
    return txn;
  };
  auto close = [&] {
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    handles.clear();
    db.reset();
  };
  Transaction* committed = prepare(0, "committed", xid_committing_elsewhere(0));
  ASSERT_OK(committed->Commit());
  delete committed;
  const std::string prepared_xid = xid_committing_elsewhere(1);
  Transaction* prepared = prepare(1, "prepared", prepared_xid);
  // Interleaved: a, c and b prepare, then b commits and a rolls back, and c
  // stays prepared.
  Transaction* a = prepare(2, "a", xid_committing_elsewhere(2));
  const std::string c_xid = xid_committing_elsewhere(3);
  Transaction* c = prepare(3, "c", c_xid);
  Transaction* b = prepare(3, "b", xid_committing_elsewhere(3));
  ASSERT_OK(b->Commit());
  ASSERT_OK(a->Rollback());
  delete a;
  delete b;
  const std::string crash_dir = dbname_ + "_crash";
  CopyDBAsCrashImage(crash_dir);
  ASSERT_OK(prepared->Rollback());
  ASSERT_OK(c->Rollback());
  delete prepared;
  delete c;
  close();

  auto value_of = [&](const std::string& key) {
    std::string value;
    const Status s =
        db->Get(ReadOptions(), handles[cf_for_partition_[partition_of[key]]],
                key, &value);
    EXPECT_TRUE(s.ok() || s.IsNotFound()) << s.ToString();
    return s.ok() ? value : "NOT_FOUND";
  };
  db = OpenWritePreparedTxnDB(crash_dir, kPartitions, options, &handles);
  ASSERT_NE(nullptr, db);
  EXPECT_EQ("v", value_of("committed"));
  EXPECT_EQ("v", value_of("b"));
  EXPECT_EQ("NOT_FOUND", value_of("a"));
  EXPECT_EQ("NOT_FOUND", value_of("prepared"));
  EXPECT_EQ("NOT_FOUND", value_of("c"));
  std::vector<Transaction*> recovered;
  db->GetAllPreparedTransactions(&recovered);
  std::set<std::string> recovered_xids;
  for (Transaction* txn : recovered) {
    recovered_xids.insert(txn->GetName());
  }
  EXPECT_EQ((std::set<std::string>{prepared_xid, c_xid}), recovered_xids);
  Transaction* recovered_prepared = db->GetTransactionByName(prepared_xid);
  ASSERT_NE(nullptr, recovered_prepared);
  ASSERT_OK(recovered_prepared->Commit());
  delete recovered_prepared;
  Transaction* recovered_c = db->GetTransactionByName(c_xid);
  ASSERT_NE(nullptr, recovered_c);
  ASSERT_OK(recovered_c->Rollback());
  delete recovered_c;
  EXPECT_EQ("v", value_of("prepared"));
  EXPECT_EQ("NOT_FOUND", value_of("c"));
  close();

  db = OpenWritePreparedTxnDB(crash_dir, kPartitions, options, &handles);
  ASSERT_NE(nullptr, db);
  recovered.clear();
  db->GetAllPreparedTransactions(&recovered);
  EXPECT_TRUE(recovered.empty());
  EXPECT_EQ("v", value_of("committed"));
  EXPECT_EQ("v", value_of("prepared"));
  EXPECT_EQ("v", value_of("b"));
  EXPECT_EQ("NOT_FOUND", value_of("a"));
  EXPECT_EQ("NOT_FOUND", value_of("c"));
  close();
}

// N-5-4 (audit finding): WritePrepared's prepare callbacks run after each
// partition's WAL append, so at N > 1 prepares reach the prepared heap out of
// sequence order. The heap keeps itself sorted; every prepared entry leaves
// it at commit and every committed key reads back. Neuter: append in push
// again; the heap keeps entries past their commit and its destructor asserts
// it is empty.
TEST_P(DBWritePartitionTest, WritePreparedConcurrentPreparesKeepHeapSorted) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr int kTxnsPerPartition = 500;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  Close();
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<TransactionDB> db =
      OpenWritePreparedTxnDB(dbname_, kPartitions, options, &handles);
  ASSERT_NE(nullptr, db);
  std::vector<port::Thread> threads;
  for (size_t p = 0; p < kPartitions; ++p) {
    threads.emplace_back([&, p] {
      for (int i = 0; i < kTxnsPerPartition; ++i) {
        Transaction* txn = db->BeginTransaction(WriteOptions());
        ASSERT_OK(
            txn->SetName("xid" + std::to_string(p) + "_" + std::to_string(i)));
        ASSERT_OK(txn->Put(handles[cf_for_partition_[p]],
                           "key" + std::to_string(i), "v"));
        ASSERT_OK(txn->Prepare());
        ASSERT_OK(txn->Commit());
        delete txn;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    for (int i = 0; i < kTxnsPerPartition; ++i) {
      std::string value;
      ASSERT_OK(db->Get(ReadOptions(), handles[cf_for_partition_[p]],
                        "key" + std::to_string(i), &value));
    }
  }
  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
  }
  db.reset();
}

// GC12 Q10: under 2PC a prepare on partition 1 pins that partition's WAL
// after rotations and flushes retire partition 0's WAL of the same
// generation, whose indices interleave with the pinned file's. Recovering a
// crash image of that set opens in every mode and recovers the prepared
// transaction. Neuter: stop treating a retired predecessor's span as
// explaining the indices nobody holds; every mode but point-in-time fails
// with an unexplained hole, and point-in-time cuts below the prepare's
// generation-mates.
TEST_P(DBWritePartitionTest, PinnedPrepareWalRecoversAfterGenerationRetired) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  // No compaction rewrites the files while they are copied.
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  Close();
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<TransactionDB> db =
      OpenWritePreparedTxnDB(dbname_, kPartitions, options, &handles);
  ASSERT_NE(nullptr, db);
  DBImpl* db_impl = static_cast_with_check<DBImpl>(db->GetRootDB());
  auto put = [&](size_t p, const std::string& key) {
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_for_partition_[p]], key, "v"));
  };
  // The prepare's generation holds records of both partitions around it.
  put(0, "before0");
  put(1, "before1");
  Transaction* txn = db->BeginTransaction(WriteOptions());
  ASSERT_OK(txn->SetName("xid"));
  ASSERT_OK(txn->Put(handles[cf_for_partition_[1]], "prepared", "v"));
  ASSERT_OK(txn->Prepare());
  const uint64_t prep_log = txn->GetLogNumber();
  ASSERT_EQ(db_impl->TEST_CurrentWalNumber(1), prep_log);
  const uint64_t retiring_wal = db_impl->TEST_CurrentWalNumber(0);
  ASSERT_LT(retiring_wal, prep_log);
  put(0, "after0");
  put(1, "after1");
  for (int round = 0; round < 2; ++round) {
    for (size_t p = 0; p < kPartitions; ++p) {
      put(p, "round" + std::to_string(round));
    }
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->Flush(FlushOptions(), handle));
    }
  }
  ASSERT_EQ(prep_log, db_impl->TEST_AliveWalFiles(1).front().first);
  ASSERT_GT(db_impl->TEST_AliveWalFiles(0).front().first, prep_log);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (env_->FileExists(LogFileName(dbname_, retiring_wal)).ok() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(
      env_->FileExists(LogFileName(dbname_, retiring_wal)).IsNotFound());
  ASSERT_OK(db->DisableFileDeletions());
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords};
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    CopyDBAsCrashImage(dbname_ + "_crash" +
                       std::to_string(static_cast<int>(recovery_mode)));
  }
  ASSERT_OK(db->EnableFileDeletions());
  ASSERT_OK(txn->Rollback());
  delete txn;
  auto close = [&] {
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    handles.clear();
    db.reset();
  };
  close();

  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    options.wal_recovery_mode = recovery_mode;
    const std::string crash_dir =
        dbname_ + "_crash" + std::to_string(static_cast<int>(recovery_mode));
    db = OpenWritePreparedTxnDB(crash_dir, kPartitions, options, &handles);
    ASSERT_NE(nullptr, db);
    Transaction* recovered = db->GetTransactionByName("xid");
    ASSERT_NE(nullptr, recovered);
    ASSERT_OK(recovered->Commit());
    delete recovered;
    for (const auto& [p, key] :
         std::vector<std::pair<size_t, std::string>>{{0, "before0"},
                                                     {1, "before1"},
                                                     {1, "prepared"},
                                                     {0, "after0"},
                                                     {1, "after1"},
                                                     {0, "round1"},
                                                     {1, "round1"}}) {
      std::string value;
      EXPECT_OK(
          db->Get(ReadOptions(), handles[cf_for_partition_[p]], key, &value))
          << key;
    }
    close();
  }
}

// GC08 (D-1): a failed WAL append stops the whole DB at wal_partitions > 1,
// and writes resume after recovery.

// Fails Append, Sync and Fsync on one WAL file while armed: a device fault
// confined to one partition's file.
class FailingWalFS : public FileSystemWrapper {
 public:
  explicit FailingWalFS(const std::shared_ptr<FileSystem>& base)
      : FileSystemWrapper(base) {}
  static const char* kClassName() { return "FailingWalFS"; }
  const char* Name() const override { return kClassName(); }

  // How much of a failing append reaches the file before the error returns.
  enum class Landing { kNothing, kWhole, kTorn };

  void Arm(uint64_t wal_number, const IOStatus& error,
           Landing landing = Landing::kNothing) {
    std::lock_guard<std::mutex> lock(mu_);
    wal_number_ = wal_number;
    error_ = error;
    error_.PermitUncheckedError();
    landing_ = landing;
  }
  void Disarm() { Arm(0, IOStatus::OK()); }

  IOStatus NewWritableFile(const std::string& fname, const FileOptions& opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, opts, result, dbg);
    uint64_t number = 0;
    FileType type;
    if (s.ok() &&
        ParseFileName(fname.substr(fname.find_last_of('/') + 1), &number,
                      &type) &&
        type == kWalFile) {
      result->reset(new WalFile(std::move(*result), this, number));
    }
    return s;
  }

 private:
  class WalFile : public FSWritableFileOwnerWrapper {
   public:
    WalFile(std::unique_ptr<FSWritableFile>&& file, FailingWalFS* fs,
            uint64_t number)
        : FSWritableFileOwnerWrapper(std::move(file)),
          fs_(fs),
          number_(number) {}
    IOStatus Append(const Slice& data, const IOOptions& options,
                    IODebugContext* dbg) override {
      Landing landing = Landing::kNothing;
      IOStatus s = fs_->Check(number_, &landing);
      if (s.ok()) {
        return target()->Append(data, options, dbg);
      }
      Land(data, landing, options, dbg);
      return s;
    }
    IOStatus Append(const Slice& data, const IOOptions& options,
                    const DataVerificationInfo& info,
                    IODebugContext* dbg) override {
      Landing landing = Landing::kNothing;
      IOStatus s = fs_->Check(number_, &landing);
      if (s.ok()) {
        return target()->Append(data, options, info, dbg);
      }
      Land(data, landing, options, dbg);
      return s;
    }
    IOStatus Sync(const IOOptions& options, IODebugContext* dbg) override {
      IOStatus s = fs_->Check(number_);
      return s.ok() ? target()->Sync(options, dbg) : s;
    }
    IOStatus Fsync(const IOOptions& options, IODebugContext* dbg) override {
      IOStatus s = fs_->Check(number_);
      return s.ok() ? target()->Fsync(options, dbg) : s;
    }

   private:
    // Writes what an armed failure lets through, ignoring the outcome.
    void Land(const Slice& data, Landing landing, const IOOptions& options,
              IODebugContext* dbg) {
      if (landing == Landing::kNothing) {
        return;
      }
      const size_t size =
          landing == Landing::kWhole ? data.size() : data.size() / 2;
      target()
          ->Append(Slice(data.data(), size), options, dbg)
          .PermitUncheckedError();
    }

    FailingWalFS* const fs_;
    const uint64_t number_;
  };

  IOStatus Check(uint64_t number, Landing* landing = nullptr) {
    std::lock_guard<std::mutex> lock(mu_);
    if (number != wal_number_) {
      return IOStatus::OK();
    }
    if (landing != nullptr) {
      *landing = landing_;
    }
    return error_;
  }

  std::mutex mu_;
  uint64_t wal_number_ = 0;
  IOStatus error_;
  Landing landing_ = Landing::kNothing;
};

class DBWalAppendFailureTest : public DBWritePartitionTest {
 public:
  DBWalAppendFailureTest()
      : fail_fs_(std::make_shared<FailingWalFS>(FileSystem::Default())),
        fail_env_(NewCompositeEnv(fail_fs_)) {}
  // The DB must close before the env it runs on goes away.
  ~DBWalAppendFailureTest() override { Close(); }

  // Paranoid, with the recorded background error left for inspection unless
  // a test turns auto-resume back on.
  Options FailureOptions() {
    Options options = IndexedWalOptions(CurrentOptions());
    options.env = fail_env_.get();
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    return options;
  }

  static IOStatus Retryable(const std::string& msg) {
    IOStatus s = IOStatus::IOError(msg);
    s.SetRetryable(true);
    return s;
  }
  static IOStatus FileScoped(const std::string& msg) {
    IOStatus s = IOStatus::IOError(msg);
    s.SetScope(IOStatus::IOErrorScope::kIOErrorScopeFile);
    return s;
  }

  // A write on its own thread that the test joins with Join(), which aborts
  // instead of hanging: nothing could release a write parked for good.
  class Writer {
   public:
    explicit Writer(std::function<Status()> write)
        : thread_([this, write] {
            Status s = write();
            std::lock_guard<std::mutex> lock(mu_);
            status_ = s;
            done_ = true;
            cv_.notify_all();
          }) {}
    Status Join() {
      {
        std::unique_lock<std::mutex> lock(mu_);
        if (!cv_.wait_for(lock, std::chrono::seconds(30),
                          [&] { return done_; })) {
          fprintf(stderr, "a write never returned\n");
          std::abort();
        }
      }
      thread_.join();
      return status_;
    }

   private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool done_ = false;
    Status status_;
    port::Thread thread_;
  };

  Status PutWithin(int cf, const std::string& key) {
    Writer writer([this, cf, key] { return Put(cf, key, "value"); });
    return writer.Join();
  }

  // Holds the next WAL append at its start until ReleaseHeldAppend().
  void HoldNextAppend() {
    {
      std::lock_guard<std::mutex> lock(hold_mu_);
      hold_armed_ = true;
      held_ = false;
      release_ = false;
    }
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [this](void*) {
          std::unique_lock<std::mutex> lock(hold_mu_);
          if (!hold_armed_) {
            return;
          }
          hold_armed_ = false;
          held_ = true;
          hold_cv_.notify_all();
          hold_cv_.wait(lock, [this] { return release_; });
        });
  }
  bool WaitUntilHeld() {
    std::unique_lock<std::mutex> lock(hold_mu_);
    return hold_cv_.wait_for(lock, std::chrono::seconds(30),
                             [this] { return held_; });
  }
  void ReleaseHeldAppend() {
    std::lock_guard<std::mutex> lock(hold_mu_);
    release_ = true;
    hold_cv_.notify_all();
  }

  // Starts `held` (the write whose append fails) and holds it at its append;
  // starts `above` on another partition, which appends above it and parks in
  // its completion gate; arms `error` on `failing_partition`'s WAL and
  // releases. Returns both writes' statuses.
  std::pair<Status, Status> FailHeldWriteUnderParkedWrite(
      size_t failing_partition, const IOStatus& error,
      const std::function<Status()>& held,
      const std::function<Status()>& above) {
    HoldNextAppend();
    Writer held_writer(held);
    EXPECT_TRUE(WaitUntilHeld());
    const size_t parked_before = dbfull()->TEST_WalDurabilityWaiters();
    Writer above_writer(above);
    EXPECT_TRUE(WaitForDurabilityWaiters(parked_before + 1));
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(failing_partition), error);
    ReleaseHeldAppend();
    std::pair<Status, Status> result{held_writer.Join(), above_writer.Join()};
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    return result;
  }

  std::shared_ptr<FailingWalFS> fail_fs_;
  std::unique_ptr<Env> fail_env_;

 private:
  std::mutex hold_mu_;
  std::condition_variable hold_cv_;
  bool hold_armed_ = false;
  bool held_ = false;
  bool release_ = false;
};

TEST_F(DBWriteTestUnparameterized, WalPartitionsRequireParanoidChecks) {
  // Q1 (G-12, G-22 N>1 half): a failed append must stop an N>1 DB, so Open
  // refuses paranoid_checks=false there and names both options; N=1 is
  // unaffected.
  Options options = CurrentOptions();
  options.wal_partitions = 2;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  options.paranoid_checks = false;
  Destroy(options);
  Status s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"))
      << s.ToString();
  EXPECT_NE(nullptr, std::strstr(s.getState(), "paranoid_checks"))
      << s.ToString();

  options.paranoid_checks = true;
  ASSERT_OK(TryReopen(options));
  options.wal_partitions = 1;
  options.paranoid_checks = false;
  ASSERT_OK(TryReopen(options));
}

TEST_P(DBWalAppendFailureTest, EveryAppendFailureStopsTheDb) {
  // Q1 (G-12, D-1): an append failure on partition 1 stops the whole DB for
  // every status class, Busy and Incomplete included. An IOError gets the
  // severity one WAL gives it; Busy and Incomplete, which one WAL shrugs
  // off, get the error handler's severity for a write error. A write parked
  // on partition 0 above the failed one, and a new write, fail promptly.
  // Q5: none reaches WALIOStatusCheck's reset-and-continue branch, whose
  // assertion would fire.
  struct Case {
    const char* name;
    IOStatus error;
  };
  const std::vector<Case> cases = {
      {"IOError", IOStatus::IOError("injected")},
      {"retryable IOError", Retryable("injected")},
      {"file-scoped IOError", FileScoped("injected")},
      {"Busy", IOStatus::Busy("injected")},
      {"Incomplete", status_to_io_status(Status::Incomplete("injected"))},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Options options = FailureOptions();
    options.wal_partitions = 1;
    DestroyAndReopen(options);
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), c.error);
    const Status write_1 = Put("key", "value");
    const Status bg_1 = dbfull()->TEST_GetBGError();
    fail_fs_->Disarm();

    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    const auto statuses = FailHeldWriteUnderParkedWrite(
        /*failing_partition=*/1, c.error,
        [&] { return Put(cf_b, "failed", "value"); },
        [&] { return Put(cf_a, "above", "value"); });
    const Status bg_n = dbfull()->TEST_GetBGError();
    const Status later_a = PutWithin(cf_a, "later");
    const Status later_b = PutWithin(cf_b, "later");
    fail_fs_->Disarm();

    ASSERT_EQ(c.error.code(), write_1.code()) << write_1.ToString();
    ASSERT_EQ(c.error.code(), statuses.first.code())
        << statuses.first.ToString();
    ASSERT_NOK(bg_n);
    if (c.error.IsIOError()) {
      ASSERT_NOK(bg_1);
      EXPECT_EQ(bg_1.code(), bg_n.code());
      EXPECT_EQ(bg_1.severity(), bg_n.severity()) << bg_n.ToString();
    } else {
      // One WAL resets the writer and continues.
      ASSERT_OK(bg_1);
      EXPECT_EQ(Status::Severity::kFatalError, bg_n.severity())
          << bg_n.ToString();
    }
    EXPECT_EQ(bg_n.ToString(), statuses.second.ToString());
    EXPECT_NOK(later_a);
    EXPECT_NOK(later_b);
    EXPECT_EQ("NOT_FOUND", Get(cf_a, "above"));
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "failed"));
  }
}

TEST_P(DBWalAppendFailureTest, OnePartitionContinueBehaviourUnchanged) {
  // Q5 (G-16): at N=1 a Busy or Incomplete append, or any failure with
  // paranoid_checks=false, still resets the writer and continues: the write
  // fails, nothing is recorded, and the next write succeeds.
  struct Case {
    const char* name;
    bool paranoid_checks;
    IOStatus error;
  };
  const std::vector<Case> cases = {
      {"Busy", true, IOStatus::Busy("injected")},
      {"Incomplete", true, status_to_io_status(Status::Incomplete("injected"))},
      {"IOError without paranoid_checks", false, IOStatus::IOError("injected")},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Options options = CurrentOptions();
    options.env = fail_env_.get();
    options.paranoid_checks = c.paranoid_checks;
    DestroyAndReopen(options);
    ASSERT_EQ(1U, dbfull()->TEST_NumWalPartitions());
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), c.error);
    const Status failed = Put("failed", "value");
    fail_fs_->Disarm();
    ASSERT_EQ(c.error.code(), failed.code()) << failed.ToString();
    ASSERT_OK(dbfull()->TEST_GetBGError());
    ASSERT_OK(Put("after", "value"));
    EXPECT_EQ("value", Get("after"));
    EXPECT_EQ("NOT_FOUND", Get("failed"));
  }
}

TEST_P(DBWalAppendFailureTest, PartitionFailureFailsEveryLaterWrite) {
  // Q2: partition 1's group (a leader and its follower) fails its append
  // while a write on partition 0 has appended above it and parked. Every
  // receipt: the group's members get the append error; the parked write and
  // new arrivals on both partitions get the background error; the write
  // acknowledged below the failure is untouched; nothing above it is visible.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  ASSERT_OK(Put(cf_a, "below", "value"));

  // The leader waits at group formation until its follower has queued.
  std::mutex mu;
  std::condition_variable cv;
  bool leader_waiting = false;
  bool follower_linked = false;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        if (static_cast<WriteThread::Writer*>(arg)->wal_partition != 1) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!leader_waiting) {
          leader_waiting = true;
          cv.notify_all();
          cv.wait(lock, [&] { return follower_linked; });
        } else {
          follower_linked = true;
          cv.notify_all();
        }
      });
  std::atomic<int> gate_installs{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::InstallCompletionGate",
                                        [&](void*) { ++gate_installs; });
  HoldNextAppend();
  Writer leader([&] { return Put(cf_b, "leader", "value"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return leader_waiting; }));
  }
  Writer follower([&] { return Put(cf_b, "follower", "value"); });
  ASSERT_TRUE(WaitUntilHeld());
  Writer parked([&] { return Put(cf_a, "parked", "value"); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  ReleaseHeldAppend();
  const Status leader_status = leader.Join();
  const Status follower_status = follower.Join();
  const Status parked_status = parked.Join();
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  const Status bg = dbfull()->TEST_GetBGError();
  const Status new_a = PutWithin(cf_a, "new");
  const Status new_b = PutWithin(cf_b, "new");
  fail_fs_->Disarm();

  ASSERT_NOK(bg);
  EXPECT_EQ(Status::Severity::kHardError, bg.severity());
  // Only the parked write installed a gate: the failed group never waits.
  EXPECT_EQ(1, gate_installs.load());
  ASSERT_TRUE(leader_status.IsIOError()) << leader_status.ToString();
  EXPECT_EQ(leader_status.ToString(), follower_status.ToString());
  EXPECT_EQ(bg.ToString(), parked_status.ToString());
  EXPECT_EQ(bg.ToString(), new_a.ToString());
  EXPECT_EQ(bg.ToString(), new_b.ToString());
  EXPECT_EQ("value", Get(cf_a, "below"));
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "parked"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "leader"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "follower"));
}

TEST_P(DBWalAppendFailureTest, CrashAtPartitionFailureRecoversAcknowledgedSet) {
  // Q3 (N-2-3): partition 0's append fails while partition 1 has appended a
  // write above it (physically present in its WAL, not acknowledged). A crash
  // right there, then a reopen in WAL-indexed mode, recovers exactly the
  // acknowledged writes: the merge stops at the failed write's missing index.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  const std::vector<std::pair<int, std::string>> acknowledged = {
      {cf_a, "a0"}, {cf_b, "b0"}, {cf_a, "a1"}, {cf_b, "b1"}};
  for (const auto& write : acknowledged) {
    ASSERT_OK(Put(write.first, write.second, "value"));
  }
  const auto statuses = FailHeldWriteUnderParkedWrite(
      /*failing_partition=*/0, IOStatus::IOError("injected"),
      [&] { return Put(cf_a, "failed", "value"); },
      [&] { return Put(cf_b, "above", "value"); });
  ASSERT_NOK(statuses.first);
  ASSERT_NOK(statuses.second);

  // Crash: the failed WAL stays failing until the process is gone.
  Close();
  fail_fs_->Disarm();
  // A hole with records above it is not a torn tail: modes that do not
  // truncate at the first hole refuse to open rather than skip it.
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = 2;
  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  const Status refused = TryReopenWithColumnFamilies(cf_names, options);
  ASSERT_TRUE(refused.IsCorruption()) << refused.ToString();
  EXPECT_NE(std::string::npos, refused.ToString().find("unexplained hole"))
      << refused.ToString();
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ReopenWithPartitions(2, options);
  for (const auto& write : acknowledged) {
    EXPECT_EQ("value", Get(write.first, write.second)) << write.second;
  }
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "failed"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "above"));
}

TEST_P(DBWalAppendFailureTest,
       DISABLED_CrashAfterLandedFailedAppendRecoversNoFailedWrite) {
  // N-2-3 residual (GC08 Not done): the failed append's record lands whole
  // and only then reports an error. Nothing on disk marks it failed, so a
  // crash-reopen replays it, and with it the write above it on partition 1:
  // both reported failure. One WAL replays such a record too; hiding both
  // needs a durable marker written to the failing device.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  ASSERT_OK(Put(cf_a, "a0", "value"));
  std::atomic<bool> fail_after_record{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        if (fail_after_record.exchange(false)) {
          *static_cast<IOStatus*>(arg) = IOStatus::IOError("injected");
        }
      });
  HoldNextAppend();
  Writer failed([&] { return Put(cf_a, "failed", "value"); });
  ASSERT_TRUE(WaitUntilHeld());
  Writer above([&] { return Put(cf_b, "above", "value"); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  fail_after_record.store(true);
  ReleaseHeldAppend();
  ASSERT_NOK(failed.Join());
  ASSERT_NOK(above.Join());
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  Close();
  ReopenWithPartitions(2, options);
  EXPECT_EQ("value", Get(cf_for_partition_[0], "a0"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "failed"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "above"));
}

// Q4 (G-15, F012): a file-scoped failure on partition 1 quarantines
// partition 1's WAL, not partition 0's lower-numbered one, and no sync
// touches it again. Every WAL holding acknowledged data stays live, and a
// reopen recovers all of it. `fail` arms the fault on partition 1's WAL and
// runs the failing operation.
void ExpectPartitionOneQuarantined(DBWalAppendFailureTest* test,
                                   const Options& options,
                                   const std::function<Status()>& fail) {
  test->OpenWithPartitions(2, options);
  DBImpl* db = test->dbfull();
  const int cf_a = test->cf_for_partition_[0];
  const int cf_b = test->cf_for_partition_[1];
  ASSERT_OK(test->Put(cf_a, "a", "value"));
  ASSERT_OK(test->Put(cf_b, "b", "value"));
  ASSERT_OK(db->FlushWAL(/*sync=*/true));
  const std::vector<std::vector<uint64_t>> alive = test->AliveWalNumbers();
  const uint64_t failed_wal = db->TEST_CurrentWalNumber(1);
  ASSERT_GT(failed_wal, db->TEST_CurrentWalNumber(0));

  test->fail_fs_->Arm(failed_wal,
                      DBWalAppendFailureTest::FileScoped("injected"));
  ASSERT_NOK(fail());
  ASSERT_NOK(db->TEST_GetBGError());
  EXPECT_EQ(failed_wal, db->TEST_WalRecoveryCutoff());

  std::vector<uint64_t> synced;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalImpl:BeforeSyncWAL", [&](void* arg) {
        synced.push_back(static_cast<log::Writer*>(arg)->get_log_number());
      });
  ASSERT_NOK(db->SyncWAL());
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWalImpl:BeforeSyncWAL");
  EXPECT_EQ(synced.end(), std::find(synced.begin(), synced.end(), failed_wal));
  EXPECT_EQ(alive, test->AliveWalNumbers());

  test->Close();
  test->fail_fs_->Disarm();
  test->ReopenWithPartitions(2, options);
  EXPECT_EQ("value", test->Get(test->cf_for_partition_[0], "a"));
  EXPECT_EQ("value", test->Get(test->cf_for_partition_[1], "b"));
}

TEST_P(DBWalAppendFailureTest, RotationFailureQuarantinesFailedPartitionsWal) {
  ExpectPartitionOneQuarantined(this, FailureOptions(), [this] {
    return dbfull()->TEST_SwitchMemtable(
        static_cast_with_check<ColumnFamilyHandleImpl>(
            handles_[cf_for_partition_[0]])
            ->cfd());
  });
}

TEST_P(DBWalAppendFailureTest,
       ManualFlushFailureQuarantinesFailedPartitionsWal) {
  Options options = FailureOptions();
  options.manual_wal_flush = true;
  ExpectPartitionOneQuarantined(this, options, [this] {
    EXPECT_OK(Put(cf_for_partition_[0], "buffered_a", "value"));
    EXPECT_OK(Put(cf_for_partition_[1], "buffered_b", "value"));
    return dbfull()->FlushWAL(/*sync=*/false);
  });
}

TEST_P(DBWalAppendFailureTest,
       LeadershipSyncFailureQuarantinesFailedPartitionsWal) {
  // The sync write appends to partition 0; its gate sync claims partition
  // 1's file for the uncommitted record below it and fails on it. (GC28 Q1:
  // a targeted sync skips partitions holding nothing it needs, so the fault
  // must sit on a file the sync claims; at the base the all-partition fsync
  // failed on partition 1's clean file.)
  ExpectPartitionOneQuarantined(this, FailureOptions(), [this] {
    // The uncommitted record must land before the fault arms, or the append
    // itself trips it: disarm, write, re-arm on the same WAL.
    const uint64_t p1_wal = dbfull()->TEST_CurrentWalNumber(1);
    fail_fs_->Disarm();
    EXPECT_OK(Put(cf_for_partition_[1], "uncommitted", "value"));
    fail_fs_->Arm(p1_wal, FileScoped("injected"));
    WriteOptions sync_write;
    sync_write.sync = true;
    return Put(cf_for_partition_[0], "synced", "value", sync_write);
  });
}

TEST_P(DBWalAppendFailureTest, FenceHidesFailedWriteFromWalIterator) {
  // Q4: a file-scoped failure after the record landed whole fences its
  // sequence range, so TransactionLogIterator stops before it.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_b = cf_for_partition_[1];
  const SequenceNumber first = db_->GetLatestSequenceNumber() + 1;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(cf_b, "k" + std::to_string(i), "value"));
  }
  std::atomic<bool> fail_after_record{true};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
        if (fail_after_record.exchange(false)) {
          *static_cast<IOStatus*>(arg) = FileScoped("injected");
        }
      });
  ASSERT_NOK(Put(cf_b, "failed", "value"));
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");

  // GetUpdatesSince refuses every call above one partition (GC12, D-G32 (a)),
  // so the failed record cannot surface through the change stream at all.
  std::unique_ptr<TransactionLogIterator> iter;
  const Status s = db_->GetUpdatesSince(first, &iter);
  EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
}

// Q6 (G-14, F003): after live recovery succeeds, writes on every partition
// complete instead of parking behind the failed range; a reopen sees every
// acknowledged write and not the failed one.
void ExpectWritesResumeOnEveryPartition(DBWalAppendFailureTest* test,
                                        const Options& options) {
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(test->PutWithin(test->cf_for_partition_[p],
                              "after" + std::to_string(p)));
  }
  test->ReopenWithPartitions(2, options);
  for (size_t p = 0; p < 2; ++p) {
    const int cf = test->cf_for_partition_[p];
    EXPECT_EQ("value", test->Get(cf, "before" + std::to_string(p)));
    EXPECT_EQ("value", test->Get(cf, "after" + std::to_string(p)));
    EXPECT_EQ("NOT_FOUND", test->Get(cf, "failed"));
  }
}

void PutBeforeFailure(DBWalAppendFailureTest* test) {
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(test->Put(test->cf_for_partition_[p],
                        "before" + std::to_string(p), "value"));
  }
}

TEST_P(DBWalAppendFailureTest, AutoResumeRestoresWritesOnEveryPartition) {
  Options options = FailureOptions();
  options.max_bgerror_resume_count = 2;
  options.bgerror_resume_retry_interval = 10000;
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  std::mutex mu;
  std::condition_variable cv;
  bool recovered = false;
  SyncPoint::GetInstance()->LoadDependency(
      {{"DBWalAppendFailureTest::AutoResume:Disarmed",
        "RecoverFromRetryableBGIOError:BeforeStart"}});
  SyncPoint::GetInstance()->SetCallBack(
      "RecoverFromRetryableBGIOError:RecoverSuccess", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        recovered = true;
        cv.notify_all();
      });
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), Retryable("injected"));
  ASSERT_NOK(Put(cf_for_partition_[0], "failed", "value"));
  fail_fs_->Disarm();
  TEST_SYNC_POINT("DBWalAppendFailureTest::AutoResume:Disarmed");
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return recovered; }));
  }
  SyncPoint::GetInstance()->ClearCallBack(
      "RecoverFromRetryableBGIOError:RecoverSuccess");
  ASSERT_OK(dbfull()->TEST_GetBGError());
  ExpectWritesResumeOnEveryPartition(this, options);
}

TEST_P(DBWalAppendFailureTest, ResumeRestoresWritesOnEveryPartition) {
  // Auto-resume off: the retryable error is a hard error until Resume().
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  ASSERT_NOK(Put(cf_for_partition_[1], "failed", "value"));
  fail_fs_->Disarm();
  const Status bg = dbfull()->TEST_GetBGError();
  ASSERT_EQ(Status::Severity::kHardError, bg.severity()) << bg.ToString();
  ASSERT_OK(dbfull()->Resume());
  ExpectWritesResumeOnEveryPartition(this, options);
}

TEST_P(DBWalAppendFailureTest, FailedResumeKeepsFailingLaterWrites) {
  // While recovery has not succeeded the failed range stays: a resume whose
  // flush fails leaves writes failing with the error, without hanging. The
  // next resume that succeeds releases it.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), Retryable("injected"));
  ASSERT_NOK(Put(cf_for_partition_[0], "failed", "value"));
  fail_fs_->Disarm();
  std::atomic<bool> fail_flush{true};
  SyncPoint::GetInstance()->SetCallBack(
      "FlushJob::WriteLevel0Table:s", [&](void* arg) {
        if (fail_flush.exchange(false)) {
          *static_cast<Status*>(arg) = Retryable("injected flush error");
        }
      });
  ASSERT_NOK(dbfull()->Resume());
  ASSERT_FALSE(fail_flush.load());
  SyncPoint::GetInstance()->ClearCallBack("FlushJob::WriteLevel0Table:s");
  const Status bg = dbfull()->TEST_GetBGError();
  ASSERT_NOK(bg);
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_EQ(bg.ToString(),
              PutWithin(cf_for_partition_[p], "stopped").ToString());
  }
  ASSERT_OK(dbfull()->Resume());
  ExpectWritesResumeOnEveryPartition(this, options);
}

TEST_P(DBWalAppendFailureTest, FileScopedFailureStaysStoppedAsAtOnePartition) {
  // A file-scoped failure abandons its WAL on resume, which upstream refuses
  // with WAL tracking in the MANIFEST; N > 1 requires that tracking, so the
  // DB stays stopped until reopened, exactly as N=1 with tracking does. The
  // failed range is never released and writes fail instead of hanging.
  Options options = FailureOptions();
  options.wal_partitions = 1;
  DestroyAndReopen(options);
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), FileScoped("injected"));
  ASSERT_NOK(Put("failed", "value"));
  fail_fs_->Disarm();
  const Status resume_1 = dbfull()->Resume();

  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), FileScoped("injected"));
  ASSERT_NOK(Put(cf_for_partition_[1], "failed", "value"));
  fail_fs_->Disarm();
  const Status resume_n = dbfull()->Resume();
  ASSERT_TRUE(resume_1.IsNotSupported()) << resume_1.ToString();
  EXPECT_EQ(resume_1.ToString(), resume_n.ToString());
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_NOK(PutWithin(cf_for_partition_[p], "stopped"));
  }
  ReopenWithPartitions(2, options);
  ExpectWritesResumeOnEveryPartition(this, options);
}

TEST_P(DBWalAppendFailureTest,
       IngestionBehindParkedWriteAssignsUnheldSequence) {
  // Q9: a higher write parks in its completion gate behind a lower write held
  // in its append, and an ingestion waits for every write queue. Every group
  // publishes before it leaves its queue, so once the ingestion holds every
  // queue LastSequence() covers every allocated sequence (asserted in
  // IngestExternalFiles) and the ingested key takes a sequence no write holds.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  const std::string sst = dbname_ + "/ingest.sst";
  {
    SstFileWriter writer(EnvOptions(), options);
    ASSERT_OK(writer.Open(sst));
    ASSERT_OK(writer.Put("parked", "ingested"));
    ASSERT_OK(writer.Finish());
  }
  std::atomic<bool> ingestion_waiting{false};
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::EnterUnbatched:Wait",
      [&](void*) { ingestion_waiting.store(true); });
  HoldNextAppend();
  Writer held([&] { return Put(cf_a, "held", "value"); });
  ASSERT_TRUE(WaitUntilHeld());
  Writer parked([&] { return Put(cf_b, "parked", "value"); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  const SequenceNumber parked_seq =
      dbfull()->GetVersionSet()->LastAllocatedSequence();
  EXPECT_LT(db_->GetLatestSequenceNumber(), parked_seq);
  Writer ingestion([&] {
    return db_->IngestExternalFile(handles_[cf_b], {sst},
                                   IngestExternalFileOptions());
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!ingestion_waiting.load() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(ingestion_waiting.load());
  ReleaseHeldAppend();
  ASSERT_OK(held.Join());
  ASSERT_OK(parked.Join());
  ASSERT_OK(ingestion.Join());
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::EnterUnbatched:Wait");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");

  EXPECT_GT(db_->GetLatestSequenceNumber(), parked_seq);
  EXPECT_EQ(dbfull()->GetVersionSet()->LastAllocatedSequence(),
            db_->GetLatestSequenceNumber());
  EXPECT_EQ("value", Get(cf_a, "held"));
  EXPECT_EQ("ingested", Get(cf_b, "parked"));
}

TEST_P(DBWalAppendFailureTest, MemtableInsertFailureFailsParkedWrite) {
  // Q10: a group whose memtable insert fails after its append never marks
  // its range applied; it records a background error instead (D-1's rule for
  // a WAL failure), so a write parked above it on the other partition, and
  // every later write, fails within bounded time instead of waiting forever.
  // The insert fails on a batch naming a column family that does not exist.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_b = cf_for_partition_[1];
  uint32_t missing_cf = static_cast<uint32_t>(handles_.size()) + 100;
  while (true) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, missing_cf, "key", "value"));
    size_t partition = 2;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(batch, 2, &partition));
    if (partition == 0) {
      break;
    }
    ++missing_cf;
  }
  HoldNextAppend();
  Writer failed([&] {
    WriteBatch batch;
    EXPECT_OK(WriteBatchInternal::Put(&batch, missing_cf, "key", "value"));
    return db_->Write(WriteOptions(), &batch);
  });
  ASSERT_TRUE(WaitUntilHeld());
  Writer parked([&] { return Put(cf_b, "parked", "value"); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  ReleaseHeldAppend();
  const Status failed_status = failed.Join();
  const Status parked_status = parked.Join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_TRUE(failed_status.IsInvalidArgument()) << failed_status.ToString();
  const Status bg = dbfull()->TEST_GetBGError();
  ASSERT_NOK(bg);
  EXPECT_EQ(bg.ToString(), parked_status.ToString());
  EXPECT_EQ(bg.ToString(), PutWithin(cf_for_partition_[0], "later").ToString());
}

// GC09 Q1 (G-13): a listener that clears every background error it sees.
class SuppressAllBgErrorsListener : public EventListener {
 public:
  void OnBackgroundError(BackgroundErrorReason /*reason*/,
                         Status* bg_error) override {
    ++calls;
    *bg_error = Status::OK();
  }

  std::atomic<int> calls{0};
};

TEST_P(DBWalAppendFailureTest, ListenerCannotSuppressPartitionedAppendFailure) {
  // Q1 (G-13, D-1): at N=2 a listener that suppresses every error is still
  // notified of a failed append, but the error is recorded anyway, with the
  // severity one WAL gives it: the failed range stays pinned until recovery
  // from that error, so a write parked above it fails promptly and new
  // writes fail. At N=1 the same listener still suppresses it. The plain
  // IOError goes through the suppressible classification; the retryable one,
  // with auto-resume off, is recorded before listeners run at either N.
  struct Case {
    const char* name;
    IOStatus error;
    bool suppressible_at_one;
  };
  const std::vector<Case> cases = {
      {"IOError", IOStatus::IOError("injected"), true},
      {"retryable IOError", Retryable("injected"), false},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Options options = FailureOptions();
    options.wal_partitions = 1;
    DestroyAndReopen(options);
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), c.error);
    ASSERT_NOK(Put("key", "value"));
    const Status bg_1 = dbfull()->TEST_GetBGError();
    fail_fs_->Disarm();
    ASSERT_NOK(bg_1);

    auto listener = std::make_shared<SuppressAllBgErrorsListener>();
    options.listeners.push_back(listener);
    DestroyAndReopen(options);
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), c.error);
    ASSERT_NOK(Put("key", "value"));
    const Status suppressed_1 = dbfull()->TEST_GetBGError();
    fail_fs_->Disarm();
    EXPECT_EQ(c.suppressible_at_one, suppressed_1.ok())
        << suppressed_1.ToString();

    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    listener->calls.store(0);
    const auto statuses = FailHeldWriteUnderParkedWrite(
        /*failing_partition=*/0, c.error,
        [&] { return Put(cf_a, "failed", "value"); },
        [&] { return Put(cf_b, "above", "value"); });
    const Status bg_n = dbfull()->TEST_GetBGError();
    const Status later = PutWithin(cf_b, "later");
    fail_fs_->Disarm();

    EXPECT_LE(1, listener->calls.load());
    ASSERT_NOK(statuses.first);
    ASSERT_NOK(bg_n);
    EXPECT_EQ(bg_1.code(), bg_n.code());
    EXPECT_EQ(bg_1.severity(), bg_n.severity()) << bg_n.ToString();
    EXPECT_EQ(bg_n.ToString(), statuses.second.ToString());
    EXPECT_EQ(bg_n.ToString(), later.ToString());
    EXPECT_EQ("NOT_FOUND", Get(cf_b, "above"));
  }
}

// Q2 (N-2-4): A is held in its append on partition 0 and W parks above it on
// partition 1; with `background_error`, C on partition 1 then fails its
// append above W, which leaves W parked (D-N71 (b)). Close() runs on another
// thread: W returns ShutdownInProgress while A is still held, and Close()
// waits for every parked write to leave the gate before it frees it. Close()
// is held right after that wait until A, released, has finished: a write
// still running when teardown starts is a write concurrent with Close().
void ExpectCloseWakesParkedWrite(DBWalAppendFailureTest* test,
                                 bool background_error) {
  Options options = test->FailureOptions();
  test->DestroyAndReopen(options);
  test->OpenWithPartitions(2, options);
  const int cf_a = test->cf_for_partition_[0];
  const int cf_b = test->cf_for_partition_[1];
  std::mutex mu;
  std::condition_variable cv;
  bool parked_once = false;
  bool close_drained = false;
  bool resume_close = false;
  // Reported under `mu`, so everything the parked write did before parking
  // (reading the handle Close() destroys) happens before Close() starts.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForWalDurability:BeforePark", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        parked_once = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CloseHelper:WalDurabilityWaitersLeft", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        close_drained = true;
        cv.notify_all();
        cv.wait(lock, [&] { return resume_close; });
      });
  test->HoldNextAppend();
  DBWalAppendFailureTest::Writer held(
      [&] { return test->Put(cf_a, "held", "value"); });
  ASSERT_TRUE(test->WaitUntilHeld());
  DBWalAppendFailureTest::Writer parked(
      [&] { return test->Put(cf_b, "parked", "value"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return parked_once; }));
  }
  ASSERT_TRUE(test->WaitForDurabilityWaiters(1));
  if (background_error) {
    test->fail_fs_->Arm(test->dbfull()->TEST_CurrentWalNumber(1),
                        DBWalAppendFailureTest::Retryable("injected"));
    ASSERT_NOK(test->Put(cf_b, "failed", "value"));
    test->fail_fs_->Disarm();
    ASSERT_NOK(test->dbfull()->TEST_GetBGError());
    ASSERT_TRUE(test->WaitForDurabilityWaiters(1));
  }
  DBWalAppendFailureTest::Writer closer([&] {
    test->Close();
    return Status::OK();
  });
  const Status parked_status = parked.Join();
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return close_drained; }));
  }
  test->ReleaseHeldAppend();
  const Status held_status = held.Join();
  {
    std::lock_guard<std::mutex> lock(mu);
    resume_close = true;
    cv.notify_all();
  }
  ASSERT_OK(closer.Join());
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::CloseHelper:WalDurabilityWaitersLeft");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WaitForWalDurability:BeforePark");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");

  EXPECT_TRUE(parked_status.IsShutdownInProgress()) << parked_status.ToString();
  ASSERT_OK(held_status);
  test->ReopenWithPartitions(2, options);
  EXPECT_EQ("value", test->Get(test->cf_for_partition_[0], "held"));
  EXPECT_EQ("NOT_FOUND", test->Get(test->cf_for_partition_[1], "failed"));
}

TEST_P(DBWalAppendFailureTest, CloseWakesParkedWrite) {
  ExpectCloseWakesParkedWrite(this, /*background_error=*/false);
}

TEST_P(DBWalAppendFailureTest, CloseWakesParkedWriteUnderBackgroundError) {
  ExpectCloseWakesParkedWrite(this, /*background_error=*/true);
}

// True when a live SST file of `cf` holds a version of `key`.
bool LiveSstHasKey(DBWalAppendFailureTest* test, const Options& options, int cf,
                   const std::string& key) {
  std::vector<LiveFileMetaData> files;
  test->db_->GetLiveFilesMetaData(&files);
  const std::string cf_name = test->handles_[cf]->GetName();
  bool found = false;
  for (const LiveFileMetaData& file : files) {
    if (file.column_family_name != cf_name) {
      continue;
    }
    SstFileReader reader(options);
    EXPECT_OK(reader.Open(file.directory + "/" + file.relative_filename));
    std::unique_ptr<Iterator> iter(reader.NewIterator(ReadOptions()));
    iter->Seek(key);
    found |= iter->Valid() && iter->key() == key;
    EXPECT_OK(iter->status());
  }
  return found;
}

TEST_P(DBWalAppendFailureTest, GateFailedWriteNeverSurvivesRecovery) {
  // Q3 (N-2-5, D-1): A's append on partition 0 fails (retryable) after B on
  // partition 1 has appended and inserted above it; B fails at its gate.
  // Whether recovery is Resume(), auto-resume or a reopen with the error
  // still set, B never becomes visible and no SST holds it: recovery's flush
  // drops everything at or above A's range. Writes resume, and a crash-like
  // reopen afterwards (no flush at close) recovers every acknowledged write
  // and still not B.
  enum class Recovery { kResume, kAutoResume, kReopen };
  for (Recovery recovery :
       {Recovery::kResume, Recovery::kAutoResume, Recovery::kReopen}) {
    SCOPED_TRACE(static_cast<int>(recovery));
    Options options = FailureOptions();
    if (recovery == Recovery::kAutoResume) {
      options.max_bgerror_resume_count = 2;
      options.bgerror_resume_retry_interval = 10000;
    }
    // A reopen with the error still set finds the failed range as a hole
    // with records above it, which only point-in-time recovery truncates.
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    PutBeforeFailure(this);
    std::mutex mu;
    std::condition_variable cv;
    bool recovered = false;
    if (recovery == Recovery::kAutoResume) {
      SyncPoint::GetInstance()->LoadDependency(
          {{"DBWalAppendFailureTest::GateFailed:Disarmed",
            "RecoverFromRetryableBGIOError:BeforeStart"}});
      SyncPoint::GetInstance()->SetCallBack(
          "RecoverFromRetryableBGIOError:RecoverSuccess", [&](void*) {
            std::lock_guard<std::mutex> lock(mu);
            recovered = true;
            cv.notify_all();
          });
    }
    const auto statuses = FailHeldWriteUnderParkedWrite(
        /*failing_partition=*/0, Retryable("injected"),
        [&] { return Put(cf_a, "failed", "value"); },
        [&] { return Put(cf_b, "above", "value"); });
    fail_fs_->Disarm();
    ASSERT_NOK(statuses.first);
    ASSERT_NOK(statuses.second);
    switch (recovery) {
      case Recovery::kResume:
        ASSERT_OK(dbfull()->Resume());
        break;
      case Recovery::kAutoResume: {
        TEST_SYNC_POINT("DBWalAppendFailureTest::GateFailed:Disarmed");
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                                [&] { return recovered; }));
        break;
      }
      case Recovery::kReopen:
        ReopenWithPartitions(2, options);
        break;
    }
    SyncPoint::GetInstance()->ClearCallBack(
        "RecoverFromRetryableBGIOError:RecoverSuccess");
    ASSERT_OK(dbfull()->TEST_GetBGError());
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "above"));
    if (recovery != Recovery::kReopen) {
      EXPECT_FALSE(LiveSstHasKey(this, options, cf_b, "above"));
    }
    ExpectWritesResumeOnEveryPartition(this, options);
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "above"));
    EXPECT_FALSE(LiveSstHasKey(this, options, cf_for_partition_[1], "above"));
  }
}

TEST_P(DBWalAppendFailureTest, UnrelatedErrorLeavesParkedWriteToComplete) {
  // Q4 (N-7-1, F013, D-N71 (b)): a background error that fences no range at
  // or below a parked write leaves it waiting, and it is acknowledged once
  // covered, as one WAL acknowledges a write whose append succeeded. A soft
  // error (compaction NoSpace) at N=1 and N=2, and a fatal one (a flush
  // IOError) at N=2, after which a new write fails with it. The wake makes
  // W recheck and park again exactly once; it does not spin.
  struct Case {
    const char* name;
    IOStatus error;
    BackgroundErrorReason reason;
    Status::Severity severity;
  };
  const std::vector<Case> cases = {
      {"soft", IOStatus::NoSpace("injected"),
       BackgroundErrorReason::kCompaction, Status::Severity::kSoftError},
      {"fatal", IOStatus::IOError("injected"), BackgroundErrorReason::kFlush,
       Status::Severity::kFatalError},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    Options options = FailureOptions();
    if (c.severity == Status::Severity::kSoftError) {
      options.wal_partitions = 1;
      DestroyAndReopen(options);
      SyncPoint::GetInstance()->EnableProcessing();
      HoldNextAppend();
      Writer held([&] { return Put("held", "value"); });
      ASSERT_TRUE(WaitUntilHeld());
      dbfull()->TEST_SetBGError(c.error, c.reason);
      EXPECT_EQ(c.severity, dbfull()->TEST_GetBGError().severity());
      ReleaseHeldAppend();
      ASSERT_OK(held.Join());
      SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    }

    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    std::atomic<std::thread::id> parked_thread{std::thread::id()};
    std::atomic<int> parks{0};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WaitForWalDurability:BeforePark", [&](void*) {
          if (std::this_thread::get_id() == parked_thread.load()) {
            ++parks;
          }
        });
    HoldNextAppend();
    Writer held([&] { return Put(cf_a, "held", "value"); });
    ASSERT_TRUE(WaitUntilHeld());
    Writer parked([&] {
      parked_thread.store(std::this_thread::get_id());
      return Put(cf_b, "parked", "value");
    });
    ASSERT_TRUE(WaitForDurabilityWaiters(1));
    dbfull()->TEST_SetBGError(c.error, c.reason);
    const Status bg = dbfull()->TEST_GetBGError();
    EXPECT_EQ(c.severity, bg.severity()) << bg.ToString();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (parks.load() < 2 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Room for a spinning waiter to show itself.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(2, parks.load());
    EXPECT_EQ(1U, dbfull()->TEST_WalDurabilityWaiters());
    ReleaseHeldAppend();
    ASSERT_OK(held.Join());
    ASSERT_OK(parked.Join());
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WaitForWalDurability:BeforePark");
    EXPECT_EQ("value", Get(cf_b, "parked"));
    if (c.severity == Status::Severity::kFatalError) {
      EXPECT_EQ(bg.ToString(), PutWithin(cf_a, "new").ToString());
    }
  }
}

TEST_P(DBWalAppendFailureTest, WriteHeldPastRecoveryStillFails) {
  // Q5 (N-2-5, D-1): B appends and inserts on partition 1 above A, then is
  // held between leaving its write queue and entering its gate. A's append
  // fails (retryable) and auto-resume completes, releasing A's range. B,
  // released only then, finds its target covered but inside the fence the
  // recovery closed: it fails, its write is gone (recovery's flush dropped
  // it), and a reopen does not bring it back. Later writes succeed.
  Options options = FailureOptions();
  options.max_bgerror_resume_count = 2;
  options.bgerror_resume_retry_interval = 10000;
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  PutBeforeFailure(this);
  std::mutex mu;
  std::condition_variable cv;
  bool b_at_gate = false;
  bool release_b = false;
  bool recovered = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCompletionGate:BeforeWait", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!b_at_gate) {
          b_at_gate = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_b; });
        }
      });
  SyncPoint::GetInstance()->LoadDependency(
      {{"DBWalAppendFailureTest::HeldPastRecovery:Disarmed",
        "RecoverFromRetryableBGIOError:BeforeStart"}});
  SyncPoint::GetInstance()->SetCallBack(
      "RecoverFromRetryableBGIOError:RecoverSuccess", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        recovered = true;
        cv.notify_all();
      });
  HoldNextAppend();
  Writer failed([&] { return Put(cf_a, "failed", "value"); });
  ASSERT_TRUE(WaitUntilHeld());
  Writer above([&] { return Put(cf_b, "above", "value"); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return b_at_gate; }));
  }
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), Retryable("injected"));
  ReleaseHeldAppend();
  ASSERT_NOK(failed.Join());
  fail_fs_->Disarm();
  TEST_SYNC_POINT("DBWalAppendFailureTest::HeldPastRecovery:Disarmed");
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return recovered; }));
  }
  ASSERT_OK(dbfull()->TEST_GetBGError());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "above"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_b = true;
    cv.notify_all();
  }
  const Status above_status = above.Join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::RunCompletionGate:BeforeWait");
  SyncPoint::GetInstance()->ClearCallBack(
      "RecoverFromRetryableBGIOError:RecoverSuccess");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  EXPECT_TRUE(above_status.IsIOError()) << above_status.ToString();
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "above"));
  ExpectWritesResumeOnEveryPartition(this, options);
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "above"));
}

TEST_P(DBWalAppendFailureTest, WriteAdmittedByRecoveryAllocatesAboveFences) {
  // Q7 (N-2-5): ClearBGError closes every open gate fence at the last
  // allocated sequence before it clears the stopped flag, so a write the
  // cleared flag admits allocates above every closed fence. Recovery is held
  // between the two steps, where the fences must exist and be closed, while
  // W retries until the cleared flag admits it; W returns OK above every
  // fence and survives a reopen. Swapping the two steps leaves the fences
  // open at the hold. Fence assertions and retry ported from GC17's Muse
  // candidate (ClearBGErrorAdmitsWritesAboveClosedFences).
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_w = cf_for_partition_[1];
  PutBeforeFailure(this);
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), Retryable("injected"));
  ASSERT_NOK(Put(cf_for_partition_[0], "failed", "value"));
  fail_fs_->Disarm();
  ASSERT_NOK(dbfull()->TEST_GetBGError());

  std::mutex mu;
  std::condition_variable cv;
  bool reached = false;
  bool release = false;
  std::vector<std::pair<SequenceNumber, SequenceNumber>> fences_at_hold;
  SyncPoint::GetInstance()->SetCallBack(
      "ErrorHandler::ClearBGError:BeforeClearStopped", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        fences_at_hold = dbfull()->TEST_WalGateFences();
        reached = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release; });
      });

  std::atomic<int> attempts{0};
  std::string admitted_key;
  SequenceNumber admitted_seq = 0;
  Writer w([&] {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(60);
    Status s = Status::TimedOut("writer never admitted");
    for (int i = 0; std::chrono::steady_clock::now() < deadline; ++i) {
      attempts.fetch_add(1);
      WriteBatch batch;
      const std::string key = "admitted" + std::to_string(i);
      s = batch.Put(handles_[cf_w], key, "value");
      if (s.ok()) {
        s = db_->Write(WriteOptions(), &batch);
      }
      if (s.ok()) {
        admitted_key = key;
        admitted_seq = WriteBatchInternal::Sequence(&batch);
        break;
      }
      std::this_thread::yield();
    }
    return s;
  });
  Writer resume([&] { return dbfull()->Resume(); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return reached; }));
    release = true;
    cv.notify_all();
  }
  ASSERT_OK(resume.Join());
  const Status w_status = w.Join();
  SyncPoint::GetInstance()->ClearCallBack(
      "ErrorHandler::ClearBGError:BeforeClearStopped");
  ASSERT_OK(w_status) << "after " << attempts.load() << " attempts";
  // At the hold, recovery had closed every fence and admitted nothing.
  ASSERT_FALSE(fences_at_hold.empty());
  for (const auto& fence : fences_at_hold) {
    EXPECT_NE(kMaxSequenceNumber, fence.second);
  }
  for (const auto& fence : dbfull()->TEST_WalGateFences()) {
    EXPECT_GT(admitted_seq, fence.second);
  }
  EXPECT_EQ("value", Get(cf_w, admitted_key));
  ReopenWithPartitions(2, options);
  EXPECT_EQ("value", Get(cf_for_partition_[1], admitted_key));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "failed"));
}

// FU14 H7: at N>1 with 2PC, a retryable append failure whose record never
// reaches the file burns its WAL index. The void cover is attempted on the
// writer that just failed (seen_error), so it fails too; a live Resume() then
// rotates every partition, writing only progress records, so the burned
// index is never covered on disk. An outstanding prepare keeps every WAL from
// its own onward live for recovery (MinLogNumberToKeep under allow_2pc),
// including the WAL with the hole. Correct behaviour: a crash image taken
// after acknowledged post-resume writes (some sync=true) opens in every
// recovery mode, recovers every acknowledged write and the prepared
// transaction.
TEST_P(DBWalAppendFailureTest, ResumedTwoPCHoleKeepsAckedWritesAfterCrash) {
  constexpr size_t kPartitions = 2;
  Options options = FailureOptions();
  options.disable_auto_compactions = true;
  OpenWithPartitions(kPartitions, options);
  Close();
  auto open_txn_db = [&](const std::string& dir,
                         std::vector<ColumnFamilyHandle*>* handles,
                         Status* open_status) {
    std::vector<std::string> cf_names;
    EXPECT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    Options open_options = options;
    open_options.wal_partitions = static_cast<uint32_t>(kPartitions);
    open_options.two_write_queues = false;
    TransactionDBOptions txn_db_options;
    txn_db_options.write_policy = TxnDBWritePolicy::WRITE_COMMITTED;
    TransactionDB* txn_db = nullptr;
    *open_status = TransactionDB::Open(open_options, txn_db_options, dir,
                                       column_families, handles, &txn_db);
    return std::unique_ptr<TransactionDB>(txn_db);
  };
  std::vector<ColumnFamilyHandle*> handles;
  Status open_status;
  std::unique_ptr<TransactionDB> db =
      open_txn_db(dbname_, &handles, &open_status);
  ASSERT_OK(open_status);
  ASSERT_NE(nullptr, db);
  DBImpl* db_impl = static_cast_with_check<DBImpl>(db->GetRootDB());
  ASSERT_EQ(kPartitions, db_impl->TEST_NumWalPartitions());
  auto close = [&] {
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    handles.clear();
    db.reset();
  };

  std::vector<std::pair<size_t, std::string>> acknowledged;
  for (size_t p = 0; p < kPartitions; ++p) {
    const std::string key = "before" + std::to_string(p);
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_for_partition_[p]], key, "v"));
    acknowledged.emplace_back(p, key);
  }
  // The prepare goes to the partition with the older WAL, so the failing
  // partition's WAL is numbered above the prepare's and stays live for
  // recovery.
  const size_t prep_partition =
      db_impl->TEST_CurrentWalNumber(0) < db_impl->TEST_CurrentWalNumber(1) ? 0
                                                                            : 1;
  const size_t fail_partition = 1 - prep_partition;
  Transaction* txn = db->BeginTransaction(WriteOptions());
  ASSERT_OK(txn->SetName("xid"));
  ASSERT_OK(
      txn->Put(handles[cf_for_partition_[prep_partition]], "prepared", "v"));
  ASSERT_OK(txn->Prepare());
  const uint64_t prep_log = txn->GetLogNumber();
  const uint64_t failed_wal = db_impl->TEST_CurrentWalNumber(fail_partition);
  ASSERT_LT(prep_log, failed_wal);

  // The append fails in the FS: WritableFileWriter drops its buffer, so the
  // record never lands, and the writer is poisoned, so the cover fails too.
  fail_fs_->Arm(failed_wal, Retryable("injected"));
  ASSERT_NOK(db->Put(WriteOptions(), handles[cf_for_partition_[fail_partition]],
                     "failed", "v"));
  fail_fs_->Disarm();
  const Status bg = db_impl->TEST_GetBGError();
  ASSERT_EQ(Status::Severity::kHardError, bg.severity()) << bg.ToString();
  ASSERT_OK(db->Resume());
  ASSERT_OK(db_impl->TEST_GetBGError());

  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int round = 0; round < 3; ++round) {
    for (size_t p = 0; p < kPartitions; ++p) {
      const std::string key =
          "after" + std::to_string(round) + "_p" + std::to_string(p);
      ASSERT_OK(db->Put(round == 2 ? sync_wo : WriteOptions(),
                        handles[cf_for_partition_[p]], key, "v"));
      acknowledged.emplace_back(p, key);
    }
  }
  // Precondition evidence: the cover of the burned index failed.
  std::string info_log;
  ASSERT_OK(ReadFileToString(env_, dbname_ + "/LOG", &info_log));
  EXPECT_NE(std::string::npos,
            info_log.find("Failed to cover burned WAL index"));

  ASSERT_OK(db->DisableFileDeletions());
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kSkipAnyCorruptedRecords};
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    CopyDBAsCrashImage(dbname_ + "_crash" +
                       std::to_string(static_cast<int>(recovery_mode)));
  }
  ASSERT_OK(db->EnableFileDeletions());
  ASSERT_OK(txn->Rollback());
  delete txn;
  close();

  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE("wal_recovery_mode=" +
                 std::to_string(static_cast<int>(recovery_mode)));
    options.wal_recovery_mode = recovery_mode;
    const std::string crash_dir =
        dbname_ + "_crash" + std::to_string(static_cast<int>(recovery_mode));
    db = open_txn_db(crash_dir, &handles, &open_status);
    EXPECT_OK(open_status);
    if (!open_status.ok() || db == nullptr) {
      handles.clear();
      db.reset();
      continue;
    }
    Transaction* recovered = db->GetTransactionByName("xid");
    EXPECT_NE(nullptr, recovered);
    if (recovered != nullptr) {
      ASSERT_OK(recovered->Rollback());
      delete recovered;
    }
    for (const auto& [p, key] : acknowledged) {
      std::string value;
      EXPECT_OK(
          db->Get(ReadOptions(), handles[cf_for_partition_[p]], key, &value))
          << key;
    }
    close();
  }
}

// FU14 M-a: partition 1's append fails with a retryable IOError (a hard,
// resumable error) and poisons its writer. A sync=true write on partition 0,
// allocated below the failed range and held between its append and its
// completion gate, then reaches its gate sync, which fsyncs every partition
// including partition 1's poisoned writer. That Sync reports "Writer has
// previous error." (non-retryable) and the WAL error path escalates the
// background error. Correct behaviour: the error stays hard and Resume()
// succeeds.
TEST_P(DBWalAppendFailureTest, GateSyncKeepsOtherPartitionsErrorResumable) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;

  std::mutex mu;
  std::condition_variable cv;
  bool hold_armed = true;
  bool a_held = false;
  bool release_a = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::InstallCompletionGate", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!hold_armed) {
          return;
        }
        hold_armed = false;
        a_held = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_a; });
      });
  Writer a([&] { return Put(cf_a, "a", "value", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return a_held; }));
  }
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  const Status b_status = PutWithin(cf_b, "failed");
  fail_fs_->Disarm();
  ASSERT_NOK(b_status);
  const Status bg_before = dbfull()->TEST_GetBGError();
  ASSERT_EQ(Status::Severity::kHardError, bg_before.severity())
      << bg_before.ToString();
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  const Status a_status = a.Join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  fprintf(stderr, "M-a: held sync write on partition 0 returned: %s\n",
          a_status.ToString().c_str());
  const Status bg_after = dbfull()->TEST_GetBGError();
  EXPECT_EQ(Status::Severity::kHardError, bg_after.severity())
      << bg_after.ToString();
  EXPECT_OK(dbfull()->Resume());
}

// D-9 (superseding GC20 Q5's D-7 for sync failures): a write ordered above
// a pending sync write neither returns nor becomes visible before that
// sync's fsync, and fails with the background error when it fails, as at
// N=1. B (sync, on partition 0) is held inside its fsync of partition 0; C
// (async, on partition 1, above B) parks behind it, unreturned and absent.
// B's fsync then fails. Both return the error, and both are absent before
// and after Resume(), after its flush, and after a reopen.
// Neuter: B publishes before its fsync (PublishOnGroupExit ignoring
// sync_first_sequence, RunSyncGate not publishing): C returns OK while B is
// held, and key_b reads back.
TEST_P(DBWalAppendFailureTest, FailedGateSyncFailsWritesAbove) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_b = cf_for_partition_[0];
  const int cf_c = cf_for_partition_[1];
  // Read before B holds partition 0's WAL mutex inside its fsync.
  const uint64_t wal0 = dbfull()->TEST_CurrentWalNumber(0);
  std::mutex mu;
  std::condition_variable cv;
  bool b_in_fsync = false;
  bool release_b = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) != 0) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (b_in_fsync) {
          return;
        }
        b_in_fsync = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_b; });
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  Writer b([&] { return Put(cf_b, "key_b", "value_b", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return b_in_fsync; }));
  }
  Writer c([&] { return Put(cf_c, "key_c", "value_c"); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_c, "key_c"));
  fail_fs_->Arm(wal0, Retryable("injected sync error"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_b = true;
    cv.notify_all();
  }
  const Status b_status = b.Join();
  const Status c_status = c.Join();
  fail_fs_->Disarm();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  ASSERT_NOK(b_status);
  ASSERT_NOK(c_status);
  EXPECT_EQ(dbfull()->TEST_GetBGError().ToString(), c_status.ToString());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_c, "key_c"));

  ASSERT_OK(dbfull()->Resume());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_c, "key_c"));
  ASSERT_OK(Flush(cf_b));
  ASSERT_OK(Flush(cf_c));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_c, "key_c"));

  ReopenWithPartitions(2, options);
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_c, "key_c"));
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_EQ("value", Get(cf_for_partition_[p], "before" + std::to_string(p)));
  }
}

// GC20 Q6 (D-8), ported from GC18's Muse candidate (pwal/ab/gc18-muse-PR9,
// a304becec854): a sync write whose gate fsync fails is hidden once it
// returns the error, and never visible again. The write routes to partition
// 0; its append lands and it inserts and publishes, but partition 0's WAL
// fails Sync with a retryable error once the gate sync starts. The key is
// absent before Resume, after Resume plus a flush (writes above the closed
// fence are still admitted), and after a crash and a reopen in every
// recovery mode, each of which must open; every acknowledged write below it
// stays present, and an unsynced control key written last is lost, so the
// crash really dropped data.
// Neuter (D-9): insert and publish the sync range before its fsync
// (RunSyncGate's callbacks, insert and PublishLastSequence moved before
// RunCoalescedGateSync); the key is
// readable right after the failure.
TEST_P(DBWalAppendFailureTest, FailedGateSyncStaysInvisibleAtNPartitions) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("two_write_queues rejects N>1");
    return;
  }
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  // FailingWalFS outside, FaultInjectionTestFS inside: the armed Sync
  // short-circuits before the inner layer sees it, so the failed write's
  // bytes stay unsynced there for the crash below. Function scope: the
  // fixture's last_options_ keeps pointing at it across loop iterations.
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(FileSystem::Default());
  // Tracks unsynced bytes, so DropUnsyncedFileData() below really drops them
  // (the control key proves it).
  fault_fs->SetInjectUnsyncedDataLoss(true);
  auto fail_fs = std::make_shared<FailingWalFS>(fault_fs);
  std::unique_ptr<Env> fail_env(NewCompositeEnv(fail_fs));
  Defer close_before_env([this] { Close(); });
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(num_partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.env = fail_env.get();
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);

    // Below: an async key on every partition, then a sync barrier so every
    // byte below the failing write is fsynced.
    for (size_t p = 0; p < num_partitions; ++p) {
      ASSERT_OK(Put(cf_for_partition_[p], "below", "v"));
    }
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf_for_partition_[0], "barrier", "v", sync_wo));
    const SequenceNumber before_w = db_->GetLatestSequenceNumber();

    // The failing write routes to partition 0. Its append lands; arming
    // partition 0's WAL once its gate sync starts fails the fsync with a
    // retryable error.
    const int cf_w = cf_for_partition_[0];
    const uint64_t wal0 = dbfull()->TEST_CurrentWalNumber(0);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex",
        [&](void*) { fail_fs->Arm(wal0, Retryable("q6")); });
    const Status ws = Put(cf_w, "w", "v", sync_wo);
    fail_fs->Disarm();
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
    ASSERT_NOK(ws);
    EXPECT_TRUE(ws.IsIOError()) << ws.ToString();
    const Status bg = dbfull()->TEST_GetBGError();
    EXPECT_FALSE(bg.ok());
    EXPECT_EQ(Status::Severity::kHardError, bg.severity()) << bg.ToString();

    // Before Resume: the failed key is absent, everything below is present.
    EXPECT_EQ("NOT_FOUND", Get(cf_w, "w"));
    for (size_t p = 0; p < num_partitions; ++p) {
      EXPECT_EQ("v", Get(cf_for_partition_[p], "below")) << p;
    }
    EXPECT_EQ("v", Get(cf_for_partition_[0], "barrier"));

    // The failed write was never inserted or published. Resume clears the
    // error; the key stays absent through a later flush, and a write above
    // it is admitted.
    ASSERT_OK(dbfull()->Resume());
    EXPECT_GT(db_->GetLatestSequenceNumber(), before_w);
    EXPECT_EQ("NOT_FOUND", Get(cf_w, "w"));
    ASSERT_OK(Put(cf_for_partition_[0], "above", "v"));
    EXPECT_EQ("v", Get(cf_for_partition_[0], "above"));
    for (size_t p = 0; p < num_partitions; ++p) {
      ASSERT_OK(Flush(cf_for_partition_[p]));
    }
    EXPECT_EQ("NOT_FOUND", Get(cf_w, "w"));
    for (size_t p = 0; p < num_partitions; ++p) {
      EXPECT_EQ("v", Get(cf_for_partition_[p], "below")) << p;
    }
    // Written last and never synced: the crash must drop it.
    ASSERT_OK(Put(cf_for_partition_[0], "control", "v"));

    // The crash drops the never-synced bytes; every recovery mode reopens
    // without the failed key or the control key, and with the prefix.
    Close();
    ASSERT_OK(fault_fs->DropUnsyncedFileData());
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    for (const WALRecoveryMode mode : recovery_modes) {
      SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
      options.wal_recovery_mode = mode;
      ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
      EXPECT_EQ("NOT_FOUND", Get(cf_w, "w"));
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "control"));
      for (size_t p = 0; p < num_partitions; ++p) {
        EXPECT_EQ("v", Get(cf_for_partition_[p], "below")) << p;
      }
      EXPECT_EQ("v", Get(cf_for_partition_[0], "barrier"));
      EXPECT_EQ("v", Get(cf_for_partition_[0], "above"));
      Close();
    }
  }
}

// GC20 Q7, under D-9: recovery from a failed gate sync never flushes the
// failed write. B's entries are unpublished, so recovery's flush waits until
// B's range is fenced (WaitForUnpublishedSyncGroups). Auto-resume is on, and
// B's gate is held right after its sync fails until recovery has reached
// that wait. B is a two-column-family batch, and recovery flushes each
// column family on its own. B is absent from both after recovery and after
// a reopen.
// Neuter: no WaitForUnpublishedSyncGroups in WaitForPendingWrites; recovery
// never waits for B (recovery_waited stays false).
TEST_P(DBWalAppendFailureTest, FailedGateSyncFenceGoesInWithItsError) {
  Options options = FailureOptions();
  options.max_bgerror_resume_count = 2;
  options.bgerror_resume_retry_interval = 10000;
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteBatch batch;
  ASSERT_OK(batch.Put(handles_[cf_a], "key_b", "value_b"));
  ASSERT_OK(batch.Put(handles_[cf_b], "key_b", "value_b"));

  std::atomic<bool> fail_sync{true};
  std::mutex mu;
  std::condition_variable cv;
  bool recovery_waiting = false;
  std::atomic<bool> recovery_waited{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        if (fail_sync.exchange(false)) {
          *static_cast<IOStatus*>(arg) = Retryable("injected gate sync error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForUnpublishedSyncGroups:Wait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        recovery_waiting = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:AfterFailedSync", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        recovery_waited = cv.wait_for(lock, std::chrono::seconds(30),
                                      [&] { return recovery_waiting; });
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  const Status b_status = db_->Write(sync_wo, &batch);
  ASSERT_NOK(b_status);
  ASSERT_TRUE(recovery_waited.load());
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!dbfull()->TEST_GetBGError().ok()) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "auto-resume recovery never completed";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  SyncPoint::GetInstance()->ClearAllCallBacks();
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));

  ReopenWithPartitions(2, options);
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "key_b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_EQ("value", Get(cf_for_partition_[p], "before" + std::to_string(p)));
  }
}

// GC21 Q1: when the void that explains a burned index cannot be written and
// synced, Resume() fails and the DB stays stopped: no write above the burned
// index is acknowledged. A later Resume() writes it and admits writes again.
// Neuter: ignore ExplainBurnedWALIndices' status in ResumeImpl; the first
// Resume() succeeds and the write above is acknowledged.
TEST_P(DBWalAppendFailureTest, ResumeFailsWhenBurnedIndexCannotBeExplained) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_fail = cf_for_partition_[1];
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  ASSERT_NOK(Put(cf_fail, "failed", "value"));
  fail_fs_->Disarm();
  ASSERT_EQ(Status::Severity::kHardError,
            dbfull()->TEST_GetBGError().severity());

  std::atomic<int> explain_attempts{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ExplainBurnedWALIndices:io_s", [&](void* arg) {
        if (explain_attempts.fetch_add(1) == 0) {
          *static_cast<IOStatus*>(arg) = Retryable("injected explain error");
        }
      });
  const Status first = dbfull()->Resume();
  EXPECT_NOK(first);
  EXPECT_EQ(1, explain_attempts.load());
  EXPECT_FALSE(dbfull()->TEST_GetBGError().ok());
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_NOK(Put(cf_for_partition_[p], "above", "value")) << p;
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[p], "above")) << p;
  }

  ASSERT_OK(dbfull()->Resume());
  EXPECT_EQ(2, explain_attempts.load());
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ExplainBurnedWALIndices:io_s");
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "above", "value"));
  }
  ReopenWithPartitions(2, options);
  for (size_t p = 0; p < 2; ++p) {
    const int cf = cf_for_partition_[p];
    EXPECT_EQ("value", Get(cf, "before" + std::to_string(p)));
    EXPECT_EQ("value", Get(cf, "above"));
  }
  EXPECT_EQ("NOT_FOUND", Get(cf_fail, "failed"));
}

// GC21 Q1, repeated failures: as ResumedTwoPCHoleKeepsAckedWritesAfterCrash,
// but the failing partition fails twice, with a Resume() and acknowledged
// writes after each, so the recovery set holds two burned indices in two
// WALs, each explained by a different Resume(). Every recovery mode opens a
// crash image with every acknowledged write and the prepared transaction.
// Neuter: skip ExplainBurnedWALIndices; the strict modes fail to open with
// an unexplained hole and point-in-time recovery loses the later writes.
TEST_P(DBWalAppendFailureTest, RepeatedTwoPCHolesStayExplainedAfterCrash) {
  constexpr size_t kPartitions = 2;
  Options options = FailureOptions();
  options.disable_auto_compactions = true;
  OpenWithPartitions(kPartitions, options);
  Close();
  auto open_txn_db = [&](const std::string& dir,
                         std::vector<ColumnFamilyHandle*>* handles,
                         Status* open_status) {
    std::vector<std::string> cf_names;
    EXPECT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    Options open_options = options;
    open_options.wal_partitions = static_cast<uint32_t>(kPartitions);
    open_options.two_write_queues = false;
    TransactionDBOptions txn_db_options;
    txn_db_options.write_policy = TxnDBWritePolicy::WRITE_COMMITTED;
    TransactionDB* txn_db = nullptr;
    *open_status = TransactionDB::Open(open_options, txn_db_options, dir,
                                       column_families, handles, &txn_db);
    return std::unique_ptr<TransactionDB>(txn_db);
  };
  std::vector<ColumnFamilyHandle*> handles;
  Status open_status;
  std::unique_ptr<TransactionDB> db =
      open_txn_db(dbname_, &handles, &open_status);
  ASSERT_OK(open_status);
  ASSERT_NE(nullptr, db);
  DBImpl* db_impl = static_cast_with_check<DBImpl>(db->GetRootDB());
  auto close = [&] {
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    handles.clear();
    db.reset();
  };

  std::vector<std::pair<size_t, std::string>> acknowledged;
  for (size_t p = 0; p < kPartitions; ++p) {
    const std::string key = "before" + std::to_string(p);
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_for_partition_[p]], key, "v"));
    acknowledged.emplace_back(p, key);
  }
  const size_t prep_partition =
      db_impl->TEST_CurrentWalNumber(0) < db_impl->TEST_CurrentWalNumber(1) ? 0
                                                                            : 1;
  const size_t fail_partition = 1 - prep_partition;
  Transaction* txn = db->BeginTransaction(WriteOptions());
  ASSERT_OK(txn->SetName("xid"));
  ASSERT_OK(
      txn->Put(handles[cf_for_partition_[prep_partition]], "prepared", "v"));
  ASSERT_OK(txn->Prepare());
  const uint64_t prep_log = txn->GetLogNumber();

  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int failure = 0; failure < 2; ++failure) {
    const uint64_t failed_wal = db_impl->TEST_CurrentWalNumber(fail_partition);
    ASSERT_LT(prep_log, failed_wal);
    fail_fs_->Arm(failed_wal, Retryable("injected"));
    ASSERT_NOK(db->Put(WriteOptions(),
                       handles[cf_for_partition_[fail_partition]],
                       "failed" + std::to_string(failure), "v"));
    fail_fs_->Disarm();
    ASSERT_OK(db->Resume());
    for (size_t p = 0; p < kPartitions; ++p) {
      const std::string key =
          "after" + std::to_string(failure) + "_p" + std::to_string(p);
      ASSERT_OK(db->Put(p == fail_partition ? sync_wo : WriteOptions(),
                        handles[cf_for_partition_[p]], key, "v"));
      acknowledged.emplace_back(p, key);
    }
  }

  ASSERT_OK(db->DisableFileDeletions());
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kSkipAnyCorruptedRecords};
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    CopyDBAsCrashImage(dbname_ + "_crash" +
                       std::to_string(static_cast<int>(recovery_mode)));
  }
  ASSERT_OK(db->EnableFileDeletions());
  ASSERT_OK(txn->Rollback());
  delete txn;
  close();

  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE("wal_recovery_mode=" +
                 std::to_string(static_cast<int>(recovery_mode)));
    options.wal_recovery_mode = recovery_mode;
    db = open_txn_db(
        dbname_ + "_crash" + std::to_string(static_cast<int>(recovery_mode)),
        &handles, &open_status);
    EXPECT_OK(open_status);
    if (!open_status.ok() || db == nullptr) {
      handles.clear();
      db.reset();
      continue;
    }
    Transaction* recovered = db->GetTransactionByName("xid");
    EXPECT_NE(nullptr, recovered);
    if (recovered != nullptr) {
      ASSERT_OK(recovered->Rollback());
      delete recovered;
    }
    for (const auto& [p, key] : acknowledged) {
      std::string value;
      EXPECT_OK(
          db->Get(ReadOptions(), handles[cf_for_partition_[p]], key, &value))
          << key;
    }
    for (int failure = 0; failure < 2; ++failure) {
      std::string value;
      EXPECT_TRUE(db->Get(ReadOptions(),
                          handles[cf_for_partition_[fail_partition]],
                          "failed" + std::to_string(failure), &value)
                      .IsNotFound());
    }
    close();
  }
}

// GC21 Q3, under D-9: a group queued on a failed gate sync shares its
// failure, and recovery's flush waits until its range too is fenced. A
// (sync, partition 0) runs the gate sync and is held inside it; B (sync,
// partition 1) queues on it. A's sync fails. Auto-resume is on, and B is
// held where it learns of the shared failure until recovery has reached
// that wait (WaitForUnpublishedSyncGroups). Both are absent after recovery
// and after a reopen.
// Neuter: no WaitForUnpublishedSyncGroups in WaitForPendingWrites; recovery
// never waits for B (recovery_waited stays false).
TEST_P(DBWalAppendFailureTest, CoalescedFailedGateSyncFencedWithItsError) {
  Options options = FailureOptions();
  options.max_bgerror_resume_count = 2;
  options.bgerror_resume_retry_interval = 10000;
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  std::mutex mu;
  std::condition_variable cv;
  bool a_in_fsync = false;
  bool release_a = false;
  bool b_queued = false;
  bool recovery_waiting = false;
  std::atomic<bool> fail_sync{true};
  std::atomic<bool> recovery_waited{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) != 0) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (a_in_fsync) {
          return;
        }
        a_in_fsync = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_a; });
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:BeforeWait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        b_queued = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        if (fail_sync.exchange(false)) {
          *static_cast<IOStatus*>(arg) = Retryable("injected gate sync error");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForUnpublishedSyncGroups:Wait", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        recovery_waiting = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RunCoalescedGateSync:SharedFailure", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        recovery_waited = cv.wait_for(lock, std::chrono::seconds(30),
                                      [&] { return recovery_waiting; });
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  Writer a([&] { return Put(cf_a, "key_a", "value_a", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return a_in_fsync; }));
  }
  Writer b([&] { return Put(cf_b, "key_b", "value_b", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return b_queued; }));
    release_a = true;
    cv.notify_all();
  }
  const Status a_status = a.Join();
  const Status b_status = b.Join();
  ASSERT_NOK(a_status);
  ASSERT_NOK(b_status);
  ASSERT_TRUE(recovery_waited.load());
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!dbfull()->TEST_GetBGError().ok()) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "auto-resume recovery never completed";
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  SyncPoint::GetInstance()->ClearAllCallBacks();
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "key_a"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));

  ReopenWithPartitions(2, options);
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "key_a"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_EQ("value", Get(cf_for_partition_[p], "before" + std::to_string(p)));
  }
}

// GC21 Q4, under D-9: a failed sync write's range deletion and point key
// are never visible on any read path: Get, MultiGet, a plain iterator, a
// tailing iterator created after the failure and one created before it,
// before Resume(), after it, after its flush, and after a reopen.
// Neuter: insert and publish the sync range before its fsync (RunSyncGate's
// callbacks, insert and PublishLastSequence moved before
// RunCoalescedGateSync); Get misses k1 and
// k2 and "w" reads back.
TEST_P(DBWalAppendFailureTest, FailedGateSyncRangeDeletionStaysHidden) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf = cf_for_partition_[0];
  for (const char* key : {"k1", "k2", "k3"}) {
    ASSERT_OK(Put(cf, key, "v"));
  }
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf, "k0", "v", sync_wo));
  ReadOptions tailing_ro;
  tailing_ro.tailing = true;
  std::unique_ptr<Iterator> early_tailing(
      db_->NewIterator(tailing_ro, handles_[cf]));
  early_tailing->SeekToFirst();
  ASSERT_OK(early_tailing->status());

  WriteBatch batch;
  ASSERT_OK(batch.DeleteRange(handles_[cf], "k1", "k3"));
  ASSERT_OK(batch.Put(handles_[cf], "w", "v"));
  std::atomic<bool> fail_sync{true};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        if (fail_sync.exchange(false)) {
          *static_cast<IOStatus*>(arg) = Retryable("injected gate sync error");
        }
      });
  ASSERT_NOK(db_->Write(sync_wo, &batch));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  ASSERT_FALSE(dbfull()->TEST_GetBGError().ok());

  const std::vector<std::string> expected = {"k0", "k1", "k2", "k3"};
  auto keys_of = [](Iterator* it) {
    std::vector<std::string> keys;
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      keys.push_back(it->key().ToString());
    }
    EXPECT_OK(it->status());
    return keys;
  };
  auto expect_hidden = [&]() {
    EXPECT_EQ("v", Get(cf, "k1"));
    EXPECT_EQ("v", Get(cf, "k2"));
    EXPECT_EQ("NOT_FOUND", Get(cf, "w"));
    std::vector<ColumnFamilyHandle*> cfs(3, handles_[cf]);
    std::vector<Slice> keys = {"k1", "k2", "w"};
    std::vector<std::string> values;
    std::vector<Status> statuses =
        db_->MultiGet(ReadOptions(), cfs, keys, &values);
    EXPECT_OK(statuses[0]);
    EXPECT_OK(statuses[1]);
    EXPECT_TRUE(statuses[2].IsNotFound()) << statuses[2].ToString();
    std::unique_ptr<Iterator> plain(
        db_->NewIterator(ReadOptions(), handles_[cf]));
    EXPECT_EQ(expected, keys_of(plain.get()));
    std::unique_ptr<Iterator> tailing(
        db_->NewIterator(tailing_ro, handles_[cf]));
    EXPECT_EQ(expected, keys_of(tailing.get()));
  };
  expect_hidden();
  EXPECT_EQ(expected, keys_of(early_tailing.get()));
  early_tailing.reset();

  ASSERT_OK(dbfull()->Resume());
  expect_hidden();
  ASSERT_OK(Flush(cf));
  expect_hidden();

  ReopenWithPartitions(2, options);
  expect_hidden();
}

// GC24 Q2, under D-9: under TransactionDB's defaults (memtable history
// kept), a sync write whose gate sync fails stays absent from Get, MultiGet,
// a plain iterator and a tailing iterator, before Resume(), after Resume()
// and Flush(), and after a reopen.
// Neuter: insert and publish the sync range before its fsync (RunSyncGate's
// callbacks, insert and PublishLastSequence moved before
// RunCoalescedGateSync): the failed write reads back.
TEST_P(DBWalAppendFailureTest, FailedGateSyncAbsentUnderTransactionDB) {
  constexpr size_t kPartitions = 2;
  Options options = FailureOptions();
  options.disable_auto_compactions = true;
  OpenWithPartitions(kPartitions, options);
  Close();
  auto open_txn_db = [&](std::vector<ColumnFamilyHandle*>* handles) {
    std::vector<std::string> cf_names;
    EXPECT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    Options open_options = options;
    open_options.wal_partitions = static_cast<uint32_t>(kPartitions);
    open_options.two_write_queues = false;
    TransactionDB* txn_db = nullptr;
    EXPECT_OK(TransactionDB::Open(open_options, TransactionDBOptions(), dbname_,
                                  column_families, handles, &txn_db));
    return std::unique_ptr<TransactionDB>(txn_db);
  };
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<TransactionDB> db = open_txn_db(&handles);
  ASSERT_NE(nullptr, db);
  const int cf = cf_for_partition_[0];
  // The configuration GC21 never retired under: flushed memtables are kept.
  ASSERT_NE(0, db->GetOptions(handles[cf]).max_write_buffer_size_to_maintain);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(db->Put(sync_wo, handles[cf], "k0", "v"));

  std::atomic<bool> fail_sync{true};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        if (fail_sync.exchange(false)) {
          *static_cast<IOStatus*>(arg) = Retryable("injected gate sync error");
        }
      });
  ASSERT_NOK(db->Put(sync_wo, handles[cf], "failed", "v"));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");

  auto expect_absent = [&](const std::string& phase) {
    SCOPED_TRACE(phase);
    std::string value;
    EXPECT_OK(db->Get(ReadOptions(), handles[cf], "k0", &value));
    EXPECT_TRUE(
        db->Get(ReadOptions(), handles[cf], "failed", &value).IsNotFound());
    std::vector<ColumnFamilyHandle*> cfs(2, handles[cf]);
    std::vector<Slice> keys = {"k0", "failed"};
    std::vector<std::string> values;
    std::vector<Status> statuses =
        db->MultiGet(ReadOptions(), cfs, keys, &values);
    EXPECT_OK(statuses[0]);
    EXPECT_TRUE(statuses[1].IsNotFound()) << statuses[1].ToString();
    for (const bool tailing : {false, true}) {
      ReadOptions ro;
      ro.tailing = tailing;
      std::unique_ptr<Iterator> it(db->NewIterator(ro, handles[cf]));
      std::vector<std::string> seen;
      for (it->SeekToFirst(); it->Valid(); it->Next()) {
        seen.push_back(it->key().ToString());
      }
      EXPECT_OK(it->status());
      EXPECT_EQ(std::vector<std::string>{"k0"}, seen)
          << (tailing ? "tailing" : "plain");
    }
  };
  expect_absent("failed");
  ASSERT_OK(db->Resume());
  ASSERT_OK(db->Flush(FlushOptions(), handles[cf]));
  expect_absent("resumed");

  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
  }
  handles.clear();
  db.reset();
  db = open_txn_db(&handles);
  ASSERT_NE(nullptr, db);
  expect_absent("reopened");
  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
  }
  db.reset();
}

// GC24 Q5: a real I/O fault on the WAL that explains a burned index. The
// file system fails the explanation's void append on partition 0's new WAL.
// Resume() fails with that error and the DB stays stopped: the background
// error stays, the failed range's fence is still open-ended, and a write is
// refused. A second Resume() rotates past the poisoned WAL, explains the
// index there and admits writes. Ported from GC21's losing candidate (Muse),
// FailedHoleExplanationFailsResume: the open-ended fence check, a second
// fail-and-resume cycle, and the reopen in all four modes.
// Neuter: drop ExplainBurnedWALIndices' `WalEmpty(0) = false`; the second
// Resume() reuses the poisoned WAL and fails again.
TEST_P(DBWalAppendFailureTest, FailedHoleExplanationFailsResume) {
  constexpr size_t kPartitions = 2;
  Options options = FailureOptions();
  options.disable_auto_compactions = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_ok = cf_for_partition_[0];
  const int cf_fail = cf_for_partition_[1];
  ASSERT_OK(Put(cf_ok, "before", "value"));

  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  ASSERT_NOK(Put(cf_fail, "failed", "value"));
  fail_fs_->Disarm();
  ASSERT_EQ(Status::Severity::kHardError,
            dbfull()->TEST_GetBGError().severity());

  uint64_t poisoned = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ExplainBurnedWALIndices:Target", [&](void* arg) {
        poisoned = *static_cast<uint64_t*>(arg);
        fail_fs_->Arm(poisoned, Retryable("injected explanation error"));
      });
  const Status first = dbfull()->Resume();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ExplainBurnedWALIndices:Target");
  fail_fs_->Disarm();
  ASSERT_NOK(first);
  EXPECT_NE(std::string::npos,
            first.ToString().find("injected explanation error"))
      << first.ToString();
  ASSERT_NE(0U, poisoned);
  ASSERT_NOK(dbfull()->TEST_GetBGError());
  // The release never ran: the failed range's fence is still open-ended.
  ASSERT_FALSE(dbfull()->TEST_WalGateFences().empty());
  for (const auto& fence : dbfull()->TEST_WalGateFences()) {
    EXPECT_EQ(kMaxSequenceNumber, fence.second);
  }
  // Nothing above the failed range is acknowledged while stopped.
  ASSERT_NOK(Put(cf_ok, "above", "value"));

  ASSERT_OK(dbfull()->Resume());
  ASSERT_OK(dbfull()->TEST_GetBGError());
  EXPECT_GT(dbfull()->TEST_CurrentWalNumber(0), poisoned);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_ok, "above", "value", sync_wo));

  // A second fail-and-resume cycle works the same way.
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  ASSERT_NOK(Put(cf_fail, "failed2", "value"));
  fail_fs_->Disarm();
  ASSERT_OK(dbfull()->Resume());
  ASSERT_OK(Put(cf_ok, "above2", "value", sync_wo));

  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kPointInTimeRecovery,
        WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kSkipAnyCorruptedRecords}) {
    SCOPED_TRACE("wal_recovery_mode=" +
                 std::to_string(static_cast<int>(recovery_mode)));
    options.wal_recovery_mode = recovery_mode;
    ReopenWithPartitions(kPartitions, options);
    EXPECT_EQ("value", Get(cf_ok, "before"));
    EXPECT_EQ("value", Get(cf_ok, "above"));
    EXPECT_EQ("value", Get(cf_ok, "above2"));
    EXPECT_EQ("NOT_FOUND", Get(cf_fail, "failed"));
    EXPECT_EQ("NOT_FOUND", Get(cf_fail, "failed2"));
  }
}

// GC24 Q5: Resume() on the retry-flush path with a burned index. A flush
// fails with a retryable error, a soft error whose automatic recovery
// retries the flush (kErrorRecoveryRetryFlush) while writes continue. While
// that retry flush runs, a write on partition 1 fails its WAL append and
// burns an index. The retry flush switches no memtable, so no WAL above the
// burning one exists yet. Recovery must still make progress: automatic
// recovery explains the index and clears the error, writes are admitted, and
// every recovery mode reopens with them.
TEST_P(DBWalAppendFailureTest, RetryFlushResumeExplainsBurnedIndex) {
  constexpr size_t kPartitions = 2;
  Options options = FailureOptions();
  options.max_bgerror_resume_count = 4;
  options.bgerror_resume_retry_interval = 10000;
  options.disable_auto_compactions = true;
  OpenWithPartitions(kPartitions, options);
  PutBeforeFailure(this);
  const int cf_ok = cf_for_partition_[0];
  const int cf_fail = cf_for_partition_[1];

  std::mutex mu;
  std::condition_variable cv;
  int flushes = 0;
  bool retry_flush_held = false;
  bool release_retry_flush = false;
  std::atomic<int> explanation_failures{0};
  SyncPoint::GetInstance()->SetCallBack(
      "FlushJob::WriteLevel0Table:s", [&](void* arg) {
        std::unique_lock<std::mutex> lock(mu);
        const int flush = flushes++;
        if (flush == 0) {
          *static_cast<Status*>(arg) = Retryable("injected flush error");
        } else if (flush == 1) {
          retry_flush_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_retry_flush; });
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ExplainBurnedWALIndices:io_s", [&](void* arg) {
        if (!static_cast<IOStatus*>(arg)->ok()) {
          explanation_failures++;
        }
      });
  ASSERT_OK(Put(cf_ok, "flushed", "value"));
  const Status flush_status = Flush(cf_ok);
  ASSERT_NOK(flush_status);
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return retry_flush_held; }));
  }
  // The first flush rotated every WAL: fail partition 1's current one.
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  const Status burned = PutWithin(cf_fail, "failed");
  fail_fs_->Disarm();
  {
    std::lock_guard<std::mutex> lock(mu);
    release_retry_flush = true;
    cv.notify_all();
  }
  ASSERT_NOK(burned);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!dbfull()->TEST_GetBGError().ok() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  SyncPoint::GetInstance()->ClearCallBack("FlushJob::WriteLevel0Table:s");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ExplainBurnedWALIndices:io_s");
  ASSERT_OK(dbfull()->TEST_GetBGError()) << "automatic recovery stalled";
  EXPECT_EQ(0, explanation_failures.load());
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "above" + std::to_string(p), "value",
                  sync_wo));
  }
  EXPECT_EQ("NOT_FOUND", Get(cf_fail, "failed"));

  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kPointInTimeRecovery,
        WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kSkipAnyCorruptedRecords}) {
    SCOPED_TRACE("wal_recovery_mode=" +
                 std::to_string(static_cast<int>(recovery_mode)));
    options.wal_recovery_mode = recovery_mode;
    ReopenWithPartitions(kPartitions, options);
    EXPECT_EQ("value", Get(cf_ok, "flushed"));
    for (size_t p = 0; p < kPartitions; ++p) {
      const int cf = cf_for_partition_[p];
      EXPECT_EQ("value", Get(cf, "before" + std::to_string(p)));
      EXPECT_EQ("value", Get(cf, "above" + std::to_string(p)));
    }
    EXPECT_EQ("NOT_FOUND", Get(cf_fail, "failed"));
  }
}

// GC24 Q4: the timestamp-stripping flush path never meets a read mask. Masks
// exist only above one partition, which requires concurrent memtable writes;
// not persisting user-defined timestamps, the only setting that strips them
// in a flush, rejects those. This pins the refusal, so the path needs no
// masking of its own while it holds.
TEST_P(DBWalAppendFailureTest, StrippingFlushNeverMeetsAMask) {
  Options options = FailureOptions();
  options.comparator = test::BytewiseComparatorWithU64TsWrapper();
  options.persist_user_defined_timestamps = false;
  Close();
  for (const bool concurrent : {false, true}) {
    options.allow_concurrent_memtable_write = concurrent;
    options.wal_partitions = 2;
    options.create_if_missing = true;
    const Status s = DB::Open(options, dbname_ + "_strip", &db_);
    EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
    EXPECT_NE(std::string::npos,
              s.ToString().find(concurrent ? "user-defined timestamps"
                                           : "allow_concurrent_memtable_write"))
        << s.ToString();
    if (s.ok()) {
      Close();
    }
  }
}

// GC21 Q5 (GC20 Q8), under D-9: a sync write that consumes no sequence
// (LogData only) and whose gate fsync fails holds no sequence: once nothing
// was allocated above it, its fence closes empty at Resume(), holding no
// sequence, and the WAL manager holds no indeterminate write.
// Neuter: close the fence at max(low, last allocated), as before; an
// indeterminate write is recorded and the fence holds a sequence.
TEST_P(DBWalAppendFailureTest, FailedZeroCountGateSyncRecordsNoIndeterminate) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  ASSERT_FALSE(dbfull()->TEST_HasIndeterminateWriteSequence());
  std::atomic<bool> fail_sync{true};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        if (fail_sync.exchange(false)) {
          *static_cast<IOStatus*>(arg) = Retryable("injected gate sync error");
        }
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  WriteBatch log_data_only;
  ASSERT_OK(log_data_only.PutLogData("marker"));
  ASSERT_NOK(db_->Write(sync_wo, &log_data_only));
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  ASSERT_FALSE(fail_sync.load());
  ASSERT_FALSE(dbfull()->TEST_GetBGError().ok());
  ASSERT_OK(dbfull()->Resume());
  EXPECT_FALSE(dbfull()->TEST_HasIndeterminateWriteSequence());
  for (const auto& [low, high] : dbfull()->TEST_WalGateFences()) {
    EXPECT_GT(low, high);
  }
}

TEST_P(DBWritePartitionTest, PublicationBoundedForParallelGroup) {
  // G-10 on the parallel-memtable path (ported from D126813582): B1 and B2
  // are forced into one group on partition 1, above A held in its append on
  // partition 0. B2, the follower, is made the last writer out, so the
  // group publishes from the follower's exit; it publishes nothing above A's
  // sequence and parks until A lands.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  ASSERT_OK(Put(cf_a, "warmup", "value"));
  const SequenceNumber seq_a = db_->GetLatestSequenceNumber() + 1;

  std::mutex mu;
  std::condition_variable cv;
  bool a_held = false;
  bool release_a = false;
  bool b1_waiting = false;
  bool b2_linked = false;
  WriteThread::Writer* b1 = nullptr;
  bool b1_blocked = false;
  int follower_exits = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        if (!a_held) {
          a_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_a; });
        }
      });
  port::Thread a_thread([&] { ASSERT_OK(Put(cf_a, "a", "value")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return a_held; }));
  }
  // B1 waits at group formation until B2 has queued behind it.
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        if (static_cast<WriteThread::Writer*>(arg)->wal_partition != 1) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (!b1_waiting) {
          b1 = static_cast<WriteThread::Writer*>(arg);
          b1_waiting = true;
          cv.notify_all();
          cv.wait(lock, [&] { return b2_linked; });
        } else {
          b2_linked = true;
          cv.notify_all();
        }
      });
  // B1 inserts and blocks waiting for B2 before B2 inserts, so B2 exits the
  // group.
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::AwaitState:BlockingWaiting", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (arg == b1) {
          b1_blocked = true;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:DoneWaiting", [&](void* arg) {
        auto* w = static_cast<WriteThread::Writer*>(arg);
        if (w->wal_partition != 1 ||
            w->state != WriteThread::STATE_PARALLEL_MEMTABLE_WRITER) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return b1_blocked; });
        ++follower_exits;
      });
  port::Thread b1_thread([&] { ASSERT_OK(Put(cf_b, "b1", "value")); });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return b1_waiting; }));
  }
  port::Thread b2_thread([&] { ASSERT_OK(Put(cf_b, "b2", "value")); });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));

  EXPECT_LT(db_->GetLatestSequenceNumber(), seq_a);
  const Snapshot* snapshot = db_->GetSnapshot();
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "b1"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "b2"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b1_thread.join();
  b2_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
  SyncPoint::GetInstance()->ClearCallBack(
      "WriteThread::AwaitState:BlockingWaiting");
  SyncPoint::GetInstance()->ClearCallBack(
      "WriteThread::JoinBatchGroup:DoneWaiting");

  EXPECT_EQ(1, follower_exits);
  EXPECT_EQ(seq_a + 2, db_->GetLatestSequenceNumber());
  EXPECT_EQ("value", Get(cf_b, "b1"));
  EXPECT_EQ("value", Get(cf_b, "b2"));
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "b1", snapshot));
  db_->ReleaseSnapshot(snapshot);
}

// GC14a: N > 1 coverage of every recovery mode (Q1), a change of the
// partition count (Q2), MANIFEST WAL tracking (Q3), and the merge buffer,
// FlushWAL and LockWAL (Q4). Instantiated without two_write_queues only: N > 1
// needs indexed WALs, which two_write_queues rejects.
class DBWalPartitionCoverageTest : public DBWritePartitionTest {
 public:
  // A write, in sequence order: its column family's handle index, its key,
  // and the byte range its record occupies in its WAL.
  struct LoggedWrite {
    int cf;
    std::string key;
    uint64_t wal;
    uint64_t begin;
    uint64_t end;
  };

  // What opening a damaged crash image does in one recovery mode: fail with
  // `code` for `reason`, or, for Status::kOk, open holding exactly the first
  // `recovered` logged writes and none after them.
  struct Cell {
    WALRecoveryMode mode;
    Status::Code code;
    size_t recovered;
    const char* reason;
  };

  static constexpr WALRecoveryMode kTolerate =
      WALRecoveryMode::kTolerateCorruptedTailRecords;
  static constexpr WALRecoveryMode kAbsolute =
      WALRecoveryMode::kAbsoluteConsistency;
  static constexpr WALRecoveryMode kPointInTime =
      WALRecoveryMode::kPointInTimeRecovery;
  static constexpr WALRecoveryMode kSkip =
      WALRecoveryMode::kSkipAnyCorruptedRecords;

  static std::vector<Cell> EveryModeOpens(size_t recovered,
                                          const char* reason) {
    std::vector<Cell> cells;
    for (const WALRecoveryMode mode :
         {kTolerate, kAbsolute, kPointInTime, kSkip}) {
      cells.push_back({mode, Status::kOk, recovered, reason});
    }
    return cells;
  }

  static std::string ValueOf(const std::string& key) { return "v_" + key; }

  // Puts `key` to column family `cf` (a handle index, which is also its id)
  // and logs where its record went.
  void LoggedPut(int cf, const std::string& key) {
    WriteBatch routing;
    ASSERT_OK(WriteBatchInternal::Put(&routing, static_cast<uint32_t>(cf), key,
                                      ValueOf(key)));
    size_t partition = 0;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(
        routing, dbfull()->TEST_NumWalPartitions(), &partition));
    const uint64_t wal = dbfull()->TEST_CurrentWalNumber(partition);
    const std::string fname = LogFileName(dbname_, wal);
    uint64_t begin = 0;
    ASSERT_OK(env_->GetFileSize(fname, &begin));
    ASSERT_OK(Put(cf, key, ValueOf(key)));
    uint64_t end = 0;
    ASSERT_OK(env_->GetFileSize(fname, &end));
    ASSERT_GT(end, begin);
    writes_.push_back({cf, key, wal, begin, end});
  }

  // Rotates every partition's WAL without flushing.
  void Rotate() {
    ASSERT_OK(dbfull()->TEST_SwitchMemtable(
        static_cast_with_check<ColumnFamilyHandleImpl>(
            handles_[cf_for_partition_[0]])
            ->cfd()));
  }

  // `rounds` rounds of one logged write to each of `cfs`, in order, rotating
  // every partition's WAL between rounds, so every round's WALs stay live.
  void LoggedRounds(const std::vector<int>& cfs, int rounds,
                    const std::string& tag) {
    for (int round = 0; round < rounds; ++round) {
      if (round > 0) {
        Rotate();
      }
      for (size_t i = 0; i < cfs.size(); ++i) {
        LoggedPut(cfs[i],
                  tag + std::to_string(round) + "_" + std::to_string(i));
      }
    }
  }

  void AssertEveryLoggedWrite() {
    for (const LoggedWrite& write : writes_) {
      EXPECT_EQ(ValueOf(write.key), Get(write.cf, write.key)) << write.key;
    }
  }

  size_t IndexOf(const std::string& key) const {
    for (size_t i = 0; i < writes_.size(); ++i) {
      if (writes_[i].key == key) {
        return i;
      }
    }
    ADD_FAILURE() << key << " was not logged";
    return writes_.size();
  }

  // Truncates `wal` of the crash image `image` to `size` bytes.
  void TruncateImageWal(const std::string& image, uint64_t wal, uint64_t size) {
    ASSERT_OK(test::TruncateFile(env_, LogFileName(image, wal), size));
  }

  // Copies the DB files of directory `from` to a fresh directory `to`.
  void CopyDir(const std::string& from, const std::string& to) {
    ASSERT_OK(DestroyDir(env_, to));
    ASSERT_OK(env_->CreateDirIfMissing(to));
    std::vector<std::string> children;
    ASSERT_OK(env_->GetChildren(from, &children));
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (ParseFileName(child, &number, &type) && type != kDBLockFile) {
        CopyFile(from + "/" + child, to + "/" + child);
      }
    }
  }

  // Opens the DB at `dir`, with every column family it has, at
  // `num_partitions` partitions in `mode`.
  Status OpenCopy(const std::string& dir, size_t num_partitions,
                  Options options, WALRecoveryMode mode,
                  std::vector<ColumnFamilyHandle*>* handles,
                  std::unique_ptr<DB>* db) {
    std::vector<std::string> cf_names;
    Status s = DB::ListColumnFamilies(options, dir, &cf_names);
    if (!s.ok()) {
      return s;
    }
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.create_if_missing = false;
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.wal_recovery_mode = mode;
    return DB::Open(DBOptions(options), dir, column_families, handles, db);
  }

  // Like OpenCopy, as a pessimistic WriteCommitted TransactionDB.
  Status OpenTxnCopy(const std::string& dir, size_t num_partitions,
                     Options options, WALRecoveryMode mode,
                     std::vector<ColumnFamilyHandle*>* handles,
                     std::unique_ptr<TransactionDB>* txn_db) {
    std::vector<std::string> cf_names;
    Status s = DB::ListColumnFamilies(options, dir, &cf_names);
    if (!s.ok()) {
      return s;
    }
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.create_if_missing = false;
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.wal_recovery_mode = mode;
    TransactionDBOptions txn_db_options;
    txn_db_options.write_policy = TxnDBWritePolicy::WRITE_COMMITTED;
    TransactionDB* opened = nullptr;
    s = TransactionDB::Open(options, txn_db_options, dir, column_families,
                            handles, &opened);
    txn_db->reset(opened);
    return s;
  }

  // Opens a fresh copy of the crash image `image` at `num_partitions`
  // partitions once per cell and checks the cell key by key against writes_.
  // Every open that succeeds also holds ValueOf(key) for each of `durable`:
  // writes outside writes_ that the image holds in SST files.
  void AssertCells(const std::string& image, size_t num_partitions,
                   const Options& options, const std::vector<Cell>& cells,
                   const std::vector<std::pair<int, std::string>>& durable) {
    const std::string copy = image + "_open";
    for (const Cell& cell : cells) {
      SCOPED_TRACE(
          "wal_recovery_mode=" + std::to_string(static_cast<int>(cell.mode)) +
          " (" + cell.reason + ")");
      CopyDir(image, copy);
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> db;
      const Status s =
          OpenCopy(copy, num_partitions, options, cell.mode, &handles, &db);
      if (cell.code != Status::kOk) {
        EXPECT_EQ(cell.code, s.code()) << s.ToString();
        for (ColumnFamilyHandle* handle : handles) {
          EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
        }
        continue;
      }
      ASSERT_OK(s);
      for (size_t i = 0; i < writes_.size(); ++i) {
        std::string value;
        const Status get = db->Get(ReadOptions(), handles[writes_[i].cf],
                                   writes_[i].key, &value);
        if (i < cell.recovered) {
          EXPECT_OK(get) << writes_[i].key << " lost below the prefix end";
          EXPECT_EQ(ValueOf(writes_[i].key), value);
        } else {
          EXPECT_TRUE(get.IsNotFound())
              << writes_[i].key << " recovered above the prefix end "
              << get.ToString();
        }
      }
      for (const auto& [cf, key] : durable) {
        std::string value;
        EXPECT_OK(db->Get(ReadOptions(), handles[cf], key, &value)) << key;
        EXPECT_EQ(ValueOf(key), value);
      }
      for (ColumnFamilyHandle* handle : handles) {
        EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
      }
    }
  }

  // Creates a brand-new DB directly at `num_partitions` partitions, with
  // column families covering every partition (cf_for_partition_), so its
  // history floor applies: no sequence persisted and no WAL retired.
  void CreateBrandNewPartitionedDb(size_t num_partitions, Options options) {
    Close();
    ASSERT_OK(DestroyDB(dbname_, options));
    std::vector<std::string> cf_names = {kDefaultColumnFamilyName};
    cf_for_partition_.assign(num_partitions, -1);
    size_t covered = 0;
    for (uint32_t cf_id = 1; covered < num_partitions; ++cf_id) {
      WriteBatch batch;
      ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
      size_t partition = num_partitions;
      ASSERT_OK(
          DBImpl::TEST_WalPartitionForWrite(batch, num_partitions, &partition));
      cf_names.push_back("cf" + std::to_string(cf_id));
      if (cf_for_partition_[partition] < 0) {
        cf_for_partition_[partition] = static_cast<int>(cf_id);
        ++covered;
      }
    }
    options.create_if_missing = true;
    options.create_missing_column_families = true;
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.two_write_queues = GetParam();
    ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
    ASSERT_EQ(num_partitions, dbfull()->TEST_NumWalPartitions());
    ASSERT_EQ(0U, dbfull()->GetVersionSet()->LastSequence());
    writes_.clear();
  }

  // Flips one byte in the middle of `write`'s record in the crash image.
  void CorruptImageRecord(const std::string& image, const LoggedWrite& write) {
    const std::string fname = LogFileName(image, write.wal);
    std::string contents;
    ASSERT_OK(ReadFileToString(env_, fname, &contents));
    ASSERT_LT(write.end, contents.size() + 1);
    contents[(write.begin + write.end) / 2] ^= 0x5a;
    ASSERT_OK(WriteStringToFile(env_, contents, fname, /*should_sync=*/true));
  }

  std::vector<LoggedWrite> writes_;
};

// Q1 (G-55), torn tails in several partitions: three rotated generations with
// a record in every partition, then a fourth whose WALs hold only their
// marker and span until the tail writes. The image tears partition 1's
// current WAL mid-first-record (it holds only t0), and partition 0's either
// mid-marker (t1 also lost: the two newest writes are a lost tail) or
// mid-t2 (t1 kept above the lost t0: a hole). At N = 8 partition 2's
// data-less current WAL also tears mid-marker. Every mode opens with exactly
// the writes below t0 or fails as listed, except skip at the hole, which
// keeps t1 (GC26 Q4, D-10). Neuter: let the merge step past
// the torn index (WALIndexMerger::Next's lost-tail branch emits nothing and
// moves on as if a void covered it); the point-in-time and tolerate cells
// recover t1 above the lost t0.
TEST_P(DBWalPartitionCoverageTest, TornTailsInSeveralPartitionsPerMode) {
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    for (const bool hole : {false, true}) {
      SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions) +
                   (hole ? " hole" : " lost tail"));
      const Options options = IndexedWalOptions(CurrentOptions());
      DestroyAndReopen(options);
      writes_.clear();
      OpenWithPartitions(num_partitions, options);
      LoggedRounds(cf_for_partition_, 3, "r");
      Rotate();
      LoggedPut(cf_for_partition_[1], "t0");
      LoggedPut(cf_for_partition_[0], "t1");
      if (hole) {
        LoggedPut(cf_for_partition_[0], "t2");
      }
      const uint64_t idle_wal =
          num_partitions > 2 ? dbfull()->TEST_CurrentWalNumber(2) : 0;
      const std::string image = dbname_ + "_image";
      CopyDBAsCrashImage(image);
      Close();

      const LoggedWrite& t0 = writes_[IndexOf("t0")];
      TruncateImageWal(image, t0.wal, (t0.begin + t0.end) / 2);
      if (hole) {
        const LoggedWrite& t2 = writes_[IndexOf("t2")];
        TruncateImageWal(image, t2.wal, (t2.begin + t2.end) / 2);
      } else {
        TruncateImageWal(image, writes_[IndexOf("t1")].wal,
                         log::kHeaderSize - 2);
      }
      if (num_partitions > 2) {
        TruncateImageWal(image, idle_wal, log::kHeaderSize - 2);
      }

      const size_t below_t0 = IndexOf("t0");
      std::vector<Cell> cells;
      if (!hole) {
        cells = {
            {kTolerate, Status::kOk, below_t0, "the losses are a tail"},
            {kAbsolute, Status::kCorruption, 0,
             "a torn record is corruption to kAbsoluteConsistency"},
            {kPointInTime, Status::kOk, below_t0, "the losses are a tail"},
            {kSkip, Status::kOk, below_t0, "the losses are a tail"},
        };
      } else {
        cells = {
            {kTolerate, Status::kCorruption, 0,
             "a tail gap with records above it is not a tail"},
            {kAbsolute, Status::kCorruption, 0,
             "a torn record is corruption to kAbsoluteConsistency"},
            {kPointInTime, Status::kOk, below_t0, "stops at the hole"},
        };
      }
      AssertCells(image, num_partitions, options, cells, {});
      if (hole) {
        // GC26 Q4 (D-10): skip skips the hole at t0 and keeps t1 above it.
        std::vector<std::pair<int, std::string>> present;
        for (const LoggedWrite& write : writes_) {
          if (write.key != "t0" && write.key != "t2") {
            present.emplace_back(write.cf, write.key);
          }
        }
        Options skip_options = options;
        skip_options.wal_partitions = static_cast<uint32_t>(num_partitions);
        AssertSkipOpensCopy(image, skip_options, present,
                            {{writes_[IndexOf("t0")].cf, "t0"},
                             {writes_[IndexOf("t2")].cf, "t2"}});
      }
    }
  }
}

// Q1 (G-55), a missing file: partition 1's middle WAL of three is deleted.
// Unsynced, the MANIFEST does not name it, and its successor's span attests
// it: tolerate and absolute fail, point-in-time opens with the writes below
// its record, and skip skips its record (GC26 Q4, D-10). Synced, the
// MANIFEST names it, and every mode fails (D-G25 (a)).
TEST_P(DBWalPartitionCoverageTest, MissingMidChainWalPerMode) {
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    for (const bool synced : {false, true}) {
      SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions) +
                   (synced ? " synced" : " unsynced"));
      const Options options = IndexedWalOptions(CurrentOptions());
      DestroyAndReopen(options);
      writes_.clear();
      OpenWithPartitions(num_partitions, options);
      for (int round = 0; round < 3; ++round) {
        if (round > 0) {
          Rotate();
        }
        for (size_t p = 0; p < num_partitions; ++p) {
          LoggedPut(cf_for_partition_[p], RoundKey(round, p));
        }
        if (synced) {
          ASSERT_OK(db_->SyncWAL());
        }
      }
      const uint64_t missing = AliveWalNumbers()[1][1];
      ASSERT_EQ(missing, writes_[IndexOf(RoundKey(1, 1))].wal);
      const std::string image = dbname_ + "_image";
      CopyDBAsCrashImage(image);
      Close();
      ASSERT_OK(env_->DeleteFile(LogFileName(image, missing)));

      std::vector<Cell> cells;
      if (synced) {
        for (const WALRecoveryMode mode :
             {kTolerate, kAbsolute, kPointInTime, kSkip}) {
          cells.push_back({mode, Status::kCorruption, 0,
                           "the MANIFEST names the missing WAL"});
        }
      } else {
        const size_t below_missing = IndexOf(RoundKey(1, 1));
        cells = {
            {kTolerate, Status::kCorruption, 0,
             "its successor attests the missing WAL"},
            {kAbsolute, Status::kCorruption, 0,
             "its successor attests the missing WAL"},
            {kPointInTime, Status::kOk, below_missing,
             "stops at the missing WAL's first record"},
        };
      }
      AssertCells(image, num_partitions, options, cells, {});
      if (!synced) {
        // GC26 Q4 (D-10): skip skips the missing WAL's record and keeps
        // every other.
        std::vector<std::pair<int, std::string>> present;
        for (const LoggedWrite& write : writes_) {
          if (write.key != RoundKey(1, 1)) {
            present.emplace_back(write.cf, write.key);
          }
        }
        Options skip_options = options;
        skip_options.wal_partitions = static_cast<uint32_t>(num_partitions);
        AssertSkipOpensCopy(image, skip_options, present,
                            {{cf_for_partition_[1], RoundKey(1, 1)}});
      }
    }
  }
}

// Q1 (G-55), 2PC spanning partitions: a pessimistic WriteCommitted
// transaction prepares on one partition and commits on another. Its commit
// record carries the prepared data past the WAL termination point, so it
// routes by the lowest column family id like the prepare does; a commit-time
// write to a lower-id column family on another partition sends it there
// (checked through the append outcomes). In every mode, an image taken
// before the commit restores the prepared transaction, one taken after it
// holds the committed data, and one whose commit record is torn restores the
// prepared transaction, or fails under kAbsoluteConsistency: never half.
TEST_P(DBWalPartitionCoverageTest,
       TwoPhaseCommitAcrossPartitionsRecoversWhole) {
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.allow_2pc = true;
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    const std::vector<int> cfs = cf_for_partition_;
    const size_t commit_partition = static_cast<size_t>(
        std::min_element(cfs.begin(), cfs.end()) - cfs.begin());
    const size_t prepare_partition = static_cast<size_t>(
        std::max_element(cfs.begin(), cfs.end()) - cfs.begin());
    Close();

    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<TransactionDB> txn_db;
    ASSERT_OK(OpenTxnCopy(dbname_, num_partitions, options, kPointInTime,
                          &handles, &txn_db));
    DBImpl* db_impl = static_cast_with_check<DBImpl>(txn_db->GetRootDB());
    std::vector<size_t> appended_partitions;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AppendOutcome", [&](void* arg) {
          appended_partitions.push_back(
              static_cast<DBImpl::WalAppendOutcome*>(arg)->wal_partition);
        });
    for (size_t p = 0; p < num_partitions; ++p) {
      ASSERT_OK(txn_db->Put(WriteOptions(), handles[cfs[p]], "before",
                            ValueOf("before")));
    }
    Transaction* txn = txn_db->BeginTransaction(WriteOptions());
    ASSERT_OK(txn->SetName("xid"));
    ASSERT_OK(txn->Put(handles[cfs[prepare_partition]], "txn_key",
                       ValueOf("txn_key")));
    ASSERT_OK(txn->GetCommitTimeWriteBatch()->Put(
        handles[cfs[commit_partition]], "commit_time_key",
        ValueOf("commit_time_key")));
    appended_partitions.clear();
    ASSERT_OK(txn->Prepare());
    ASSERT_EQ(std::vector<size_t>{prepare_partition}, appended_partitions);
    const std::string prepared_image = dbname_ + "_prepared";
    CopyDBAsCrashImage(prepared_image);

    const uint64_t commit_wal_number =
        db_impl->TEST_CurrentWalNumber(commit_partition);
    const std::string commit_wal = LogFileName(dbname_, commit_wal_number);
    uint64_t commit_begin = 0;
    ASSERT_OK(env_->GetFileSize(commit_wal, &commit_begin));
    appended_partitions.clear();
    ASSERT_OK(txn->Commit());
    ASSERT_EQ(std::vector<size_t>{commit_partition}, appended_partitions);
    uint64_t commit_end = 0;
    ASSERT_OK(env_->GetFileSize(commit_wal, &commit_end));
    const std::string committed_image = dbname_ + "_committed";
    CopyDBAsCrashImage(committed_image);
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AppendOutcome");
    delete txn;
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(txn_db->DestroyColumnFamilyHandle(handle));
    }
    txn_db.reset();
    const std::string torn_image = dbname_ + "_torn_commit";
    CopyDir(committed_image, torn_image);
    TruncateImageWal(torn_image, commit_wal_number,
                     (commit_begin + commit_end) / 2);

    auto check = [&](const std::string& image, WALRecoveryMode mode,
                     bool committed) {
      SCOPED_TRACE(image + " wal_recovery_mode=" +
                   std::to_string(static_cast<int>(mode)));
      const std::string copy = image + "_open";
      CopyDir(image, copy);
      std::vector<ColumnFamilyHandle*> copy_handles;
      std::unique_ptr<TransactionDB> copy_db;
      ASSERT_OK(OpenTxnCopy(copy, num_partitions, options, mode, &copy_handles,
                            &copy_db));
      for (size_t p = 0; p < num_partitions; ++p) {
        std::string value;
        EXPECT_OK(copy_db->Get(ReadOptions(), copy_handles[cfs[p]], "before",
                               &value));
      }
      Transaction* recovered = copy_db->GetTransactionByName("xid");
      for (const auto& [partition, key] :
           std::vector<std::pair<size_t, std::string>>{
               {prepare_partition, "txn_key"},
               {commit_partition, "commit_time_key"}}) {
        std::string value;
        const Status s = copy_db->Get(
            ReadOptions(), copy_handles[cfs[partition]], key, &value);
        if (committed) {
          EXPECT_OK(s) << key;
          EXPECT_EQ(ValueOf(key), value);
        } else {
          EXPECT_TRUE(s.IsNotFound()) << key << " " << s.ToString();
        }
      }
      if (committed) {
        EXPECT_EQ(nullptr, recovered);
      } else {
        ASSERT_NE(nullptr, recovered);
        EXPECT_OK(recovered->Rollback());
        delete recovered;
      }
      for (ColumnFamilyHandle* handle : copy_handles) {
        EXPECT_OK(copy_db->DestroyColumnFamilyHandle(handle));
      }
    };
    for (const WALRecoveryMode mode :
         {kTolerate, kAbsolute, kPointInTime, kSkip}) {
      check(prepared_image, mode, /*committed=*/false);
      check(committed_image, mode, /*committed=*/true);
      if (mode == kAbsolute) {
        std::vector<ColumnFamilyHandle*> copy_handles;
        std::unique_ptr<TransactionDB> copy_db;
        CopyDir(torn_image, torn_image + "_open");
        EXPECT_TRUE(OpenTxnCopy(torn_image + "_open", num_partitions, options,
                                mode, &copy_handles, &copy_db)
                        .IsCorruption());
        for (ColumnFamilyHandle* handle : copy_handles) {
          EXPECT_OK(copy_db->DestroyColumnFamilyHandle(handle));
        }
      } else {
        check(torn_image, mode, /*committed=*/false);
      }
    }
  }
}

// Q1 (G-55), gaps: a disableWAL write (flushed, so its data outlives the
// crash) and an ingested file take sequence numbers but no WAL index. An
// image whose WALs hold writes on both sides of the gaps opens in every mode
// with every write: the gaps truncate nothing (Tasks:680). One partition
// (indexed) runs too, so the disableWAL and ingestion WAL-index cases are
// also reopened at N = 1.
TEST_P(DBWalPartitionCoverageTest, DisableWalAndIngestionGapsDoNotTruncate) {
  for (const size_t num_partitions : {size_t{1}, size_t{2}, size_t{8}}) {
    SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions));
    const Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    writes_.clear();
    OpenWithPartitions(num_partitions, options);
    for (size_t p = 0; p < num_partitions; ++p) {
      LoggedPut(cf_for_partition_[p], "before" + std::to_string(p));
    }
    WriteOptions unlogged;
    unlogged.disableWAL = true;
    ASSERT_OK(
        Put(cf_for_partition_[0], "unlogged", ValueOf("unlogged"), unlogged));
    ASSERT_OK(Flush(cf_for_partition_[0]));
    const std::string sst = dbname_ + "/ingest.sst";
    SstFileWriter writer(EnvOptions(), options);
    ASSERT_OK(writer.Open(sst));
    ASSERT_OK(writer.Put("ingested", ValueOf("ingested")));
    ASSERT_OK(writer.Finish());
    const int ingest_cf = cf_for_partition_[num_partitions - 1];
    ASSERT_OK(db_->IngestExternalFile(handles_[ingest_cf], {sst},
                                      IngestExternalFileOptions()));
    for (size_t p = 0; p < num_partitions; ++p) {
      LoggedPut(cf_for_partition_[p], "after" + std::to_string(p));
    }
    const std::string image = dbname_ + "_image";
    CopyDBAsCrashImage(image);
    Close();
    AssertCells(image, num_partitions, options,
                EveryModeOpens(writes_.size(), "the gaps are no damage"),
                {{cf_for_partition_[0], "unlogged"}, {ingest_cf, "ingested"}});
  }
}

// Q2 (N-3-5, G-55): written at `from` partitions with every WAL retained
// (rotations without flushes) and reopened at `to`. A crash image opens at
// `to` with every write in every mode; the clean reopen places each WAL with
// the partition that wrote it, or in partition 0 when this open lacks that
// partition, and reads every write; flushing every column family then
// reclaims every old WAL. Neuter: place orphans in the last partition
// instead of partition 0 (RestoreRecoveredWals); the 4 -> 2 placement assert
// fails.
TEST_P(DBWalPartitionCoverageTest, ReopenAtAnotherPartitionCountKeepsEveryKey) {
  for (const auto& [from, to] :
       std::vector<std::pair<size_t, size_t>>{{4, 2}, {2, 4}}) {
    SCOPED_TRACE(std::to_string(from) + " -> " + std::to_string(to));
    const Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    writes_.clear();
    OpenWithPartitions(std::max(from, to), options);
    const std::vector<int> cfs = cf_for_partition_;
    ReopenWithPartitions(from, options);
    LoggedRounds(cfs, 3, "r");
    const std::vector<std::vector<uint64_t>> before = AliveWalNumbers();
    const std::string image = dbname_ + "_image";
    CopyDBAsCrashImage(image);
    AssertCells(image, to, options,
                EveryModeOpens(writes_.size(), "a change of N is no damage"),
                {});

    ReopenWithPartitions(to, options);
    AssertPartitionDequesConsistent();
    std::vector<std::vector<uint64_t>> expected(to);
    for (size_t q = 0; q < from; ++q) {
      std::vector<uint64_t>& into = expected[q < to ? q : 0];
      into.insert(into.end(), before[q].begin(), before[q].end());
    }
    for (size_t p = 0; p < to; ++p) {
      std::sort(expected[p].begin(), expected[p].end());
      expected[p].push_back(dbfull()->TEST_CurrentWalNumber(p));
    }
    ASSERT_EQ(expected, AliveWalNumbers());
    AssertEveryLoggedWrite();

    FlushAllColumnFamilies();
    ASSERT_OK(dbfull()->TEST_WaitForPurge());
    std::vector<uint64_t> current_wals;
    for (size_t p = 0; p < to; ++p) {
      current_wals.push_back(dbfull()->TEST_CurrentWalNumber(p));
      ASSERT_EQ(std::vector<uint64_t>{current_wals.back()},
                AliveWalNumbers()[p]);
    }
    ASSERT_EQ(current_wals, WalNumbersOnDisk());
    ReopenWithPartitions(to, options);
    AssertEveryLoggedWrite();
    Close();
  }
}

// Q2 (N-3-5): one indexed partition -> four -> one, with WALs retained across
// both changes. Each crash image opens at the next count with every write in
// every mode; each reopen keeps every write and puts every WAL the new count
// cannot place in partition 0; a final flush reclaims every old WAL.
TEST_P(DBWalPartitionCoverageTest, ReopenOneFourOneKeepsEveryKey) {
  const Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(4, options);
  const std::vector<int> cfs = cf_for_partition_;
  ReopenWithPartitions(1, options);
  LoggedRounds(cfs, 2, "n1_");
  const std::vector<uint64_t> n1_wals = AliveWalNumbers()[0];
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  AssertCells(image, 4, options,
              EveryModeOpens(writes_.size(), "a change of N is no damage"), {});

  ReopenWithPartitions(4, options);
  AssertPartitionDequesConsistent();
  std::vector<uint64_t> expected_0 = n1_wals;
  expected_0.push_back(dbfull()->TEST_CurrentWalNumber(0));
  ASSERT_EQ(expected_0, AliveWalNumbers()[0]);
  for (size_t p = 1; p < 4; ++p) {
    ASSERT_EQ(std::vector<uint64_t>{dbfull()->TEST_CurrentWalNumber(p)},
              AliveWalNumbers()[p]);
  }
  AssertEveryLoggedWrite();
  LoggedRounds(cfs, 2, "n4_");
  std::vector<uint64_t> all_wals;
  for (const std::vector<uint64_t>& wals : AliveWalNumbers()) {
    all_wals.insert(all_wals.end(), wals.begin(), wals.end());
  }
  std::sort(all_wals.begin(), all_wals.end());
  CopyDBAsCrashImage(image);
  AssertCells(image, 1, options,
              EveryModeOpens(writes_.size(), "a change of N is no damage"), {});

  ReopenWithPartitions(1, options);
  AssertPartitionDequesConsistent();
  all_wals.push_back(dbfull()->TEST_CurrentWalNumber(0));
  ASSERT_EQ(all_wals, AliveWalNumbers()[0]);
  AssertEveryLoggedWrite();
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForPurge());
  ASSERT_EQ(std::vector<uint64_t>{dbfull()->TEST_CurrentWalNumber(0)},
            AliveWalNumbers()[0]);
  ASSERT_EQ(std::vector<uint64_t>{dbfull()->TEST_CurrentWalNumber(0)},
            WalNumbersOnDisk());
  ReopenWithPartitions(1, options);
  AssertEveryLoggedWrite();
  Close();
}

// Q3 (muse #19, test half): MANIFEST WAL tracking at N = 4 under skew
// (partition 1 hot, partition 3 idle after its first write). After every
// SyncWAL the MANIFEST's WalSet holds every inactive live WAL of every
// partition and nothing that is not live, through rotations, a flush of the
// hot column family, a flush of every column family, and rotations with
// empty column families after it; a reopen then verifies the set and reads
// every write. Neuter: take the SwitchMemtable deletion horizon (and the
// log number of empty column families) from the highest partition's new WAL
// instead of partition 0's; the reopen loses the writes partitions 0 and 2
// made after the rotation.
TEST_P(DBWalPartitionCoverageTest, ManifestTracksEveryPartitionsWalsUnderSkew) {
  constexpr size_t kPartitions = 4;
  const Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(kPartitions, options);
  ASSERT_TRUE(options.track_and_verify_wals_in_manifest);
  const int hot = cf_for_partition_[1];
  int n = 0;
  auto write_skewed = [&](int hot_writes) {
    for (int i = 0; i < hot_writes; ++i) {
      LoggedPut(hot, "hot" + std::to_string(n++));
    }
    LoggedPut(cf_for_partition_[0], "p0_" + std::to_string(n++));
    LoggedPut(cf_for_partition_[2], "p2_" + std::to_string(n++));
  };
  // WALs a reopen recovered have no writer, so SyncWAL does not track them,
  // as at one partition: after the reopen only "nothing untracked is live"
  // is checked.
  auto assert_tracked = [&](const std::string& step, bool recovered_wals) {
    SCOPED_TRACE(step);
    ASSERT_OK(db_->SyncWAL());
    std::set<uint64_t> tracked;
    for (const auto& wal : dbfull()->GetVersionSet()->GetWalSet().GetWals()) {
      tracked.insert(wal.first);
    }
    std::set<uint64_t> live;
    std::set<uint64_t> inactive;
    for (const std::vector<uint64_t>& wals : AliveWalNumbers()) {
      live.insert(wals.begin(), wals.end());
      inactive.insert(wals.begin(), wals.end() - 1);
    }
    for (const uint64_t wal : tracked) {
      EXPECT_EQ(1U, live.count(wal)) << "tracked WAL " << wal << " not live";
    }
    for (const uint64_t wal : inactive) {
      EXPECT_TRUE(recovered_wals || tracked.count(wal) == 1)
          << "live WAL " << wal << " untracked";
    }
  };

  LoggedPut(cf_for_partition_[3], "idle");
  write_skewed(20);
  assert_tracked("first generation", false);
  for (int round = 0; round < 2; ++round) {
    Rotate();
    write_skewed(20);
    assert_tracked("rotation " + std::to_string(round), false);
  }
  ASSERT_OK(Flush(hot));
  assert_tracked("hot column family flushed", false);
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForPurge());
  assert_tracked("every column family flushed", false);
  // Every column family but the hot one is empty when this rotates.
  LoggedPut(hot, "hot" + std::to_string(n++));
  Rotate();
  write_skewed(10);
  assert_tracked("rotation with empty column families", false);
  // Partitions 0 and 2 now keep their WALs for column families that were
  // empty at the rotation alone.
  ASSERT_OK(Flush(hot));
  assert_tracked("hot column family flushed after the rotation", false);
  Rotate();
  write_skewed(10);
  assert_tracked("second rotation after the flushes", false);

  ReopenWithPartitions(kPartitions, options);
  AssertEveryLoggedWrite();
  assert_tracked("reopened", true);
  Close();
}

// Q4 (G-57, Design :2113): multi-writer groups forced on all eight
// partitions at once: each partition's first leader waits until every other
// writer of the round has queued behind some leader. Each partition's WAL
// decodes record by record into whole groups: every acknowledged batch
// appears exactly once, in its own partition's WAL, at its own sequence,
// with its bytes intact, and every partition wrote a multi-writer group.
TEST_P(DBWalPartitionCoverageTest, ForcedGroupsOnEveryPartitionAreByteExact) {
  constexpr size_t kPartitions = 8;
  constexpr int kWritersPerPartition = 4;
  constexpr int kRounds = 3;
  const Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(kPartitions, options);

  // port::Mutex, not std::mutex: its constructor registers it with TSan,
  // which otherwise can mistake it for a destroyed mutex at the same address.
  port::Mutex mu;
  port::CondVar cv(&mu);
  // Guarded by mu
  int followers = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
        if (static_cast<WriteThread::Writer*>(arg)->state !=
            WriteThread::STATE_GROUP_LEADER) {
          MutexLock lock(&mu);
          ++followers;
          cv.SignalAll();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:Wait2", [&](void* arg) {
        if (static_cast<WriteThread::Writer*>(arg)->state !=
            WriteThread::STATE_GROUP_LEADER) {
          return;
        }
        MutexLock lock(&mu);
        const uint64_t deadline = env_->NowMicros() + 10 * 1000 * 1000;
        while (followers <
                   static_cast<int>(kPartitions) * (kWritersPerPartition - 1) &&
               !cv.TimedWait(deadline)) {
        }
      });

  std::vector<WriteBatch> batches(kRounds * kPartitions * kWritersPerPartition);
  std::vector<size_t> partition_of_batch(batches.size());
  for (int round = 0; round < kRounds; ++round) {
    {
      MutexLock lock(&mu);
      followers = 0;
    }
    std::vector<port::Thread> threads;
    for (size_t p = 0; p < kPartitions; ++p) {
      for (int w = 0; w < kWritersPerPartition; ++w) {
        const size_t i = (round * kPartitions + p) * kWritersPerPartition + w;
        partition_of_batch[i] = p;
        const std::string key = "r" + std::to_string(round) + "_p" +
                                std::to_string(p) + "_w" + std::to_string(w);
        ASSERT_OK(batches[i].Put(
            handles_[cf_for_partition_[p]], key + "a",
            std::string(1 + 7 * w + round, static_cast<char>('a' + w))));
        ASSERT_OK(
            batches[i].Put(handles_[cf_for_partition_[p]], key + "b", key));
        threads.emplace_back(
            [&, i] { ASSERT_OK(db_->Write(WriteOptions(), &batches[i])); });
      }
    }
    for (port::Thread& thread : threads) {
      thread.join();
    }
  }
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
  SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait2");

  std::map<SequenceNumber, size_t> batch_at;
  for (size_t i = 0; i < batches.size(); ++i) {
    const SequenceNumber sequence = WriteBatchInternal::Sequence(&batches[i]);
    ASSERT_NE(0U, sequence);
    ASSERT_TRUE(batch_at.emplace(sequence, i).second);
  }
  std::vector<int> seen(batches.size(), 0);
  std::vector<int> groups(kPartitions, 0);
  const std::vector<std::vector<uint64_t>> alive = AliveWalNumbers();
  for (size_t p = 0; p < kPartitions; ++p) {
    for (const uint64_t wal : alive[p]) {
      SCOPED_TRACE("WAL " + std::to_string(wal));
      std::unique_ptr<SequentialFileReader> file_reader;
      ASSERT_OK(SequentialFileReader::Create(
          env_->GetFileSystem(), LogFileName(dbname_, wal), FileOptions(),
          &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
      log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                         /*reporter=*/nullptr, /*checksum=*/true, wal);
      std::string scratch;
      Slice record;
      while (reader.ReadRecord(&record, &scratch)) {
        ASSERT_GE(record.size(), WriteBatchInternal::kHeader);
        const SequenceNumber first = DecodeFixed64(record.data());
        const uint32_t count = DecodeFixed32(record.data() + 8);
        if (count == 0) {
          continue;
        }
        size_t offset = WriteBatchInternal::kHeader;
        SequenceNumber next = first;
        int members = 0;
        while (offset < record.size()) {
          const auto at = batch_at.find(next);
          ASSERT_NE(batch_at.end(), at) << "no batch at sequence " << next;
          const WriteBatch& batch = batches[at->second];
          const Slice body(batch.Data().data() + WriteBatchInternal::kHeader,
                           batch.Data().size() - WriteBatchInternal::kHeader);
          ASSERT_LE(offset + body.size(), record.size());
          ASSERT_EQ(body.ToString(/*hex=*/true),
                    Slice(record.data() + offset, body.size())
                        .ToString(/*hex=*/true));
          ASSERT_EQ(p, partition_of_batch[at->second]);
          ++seen[at->second];
          offset += body.size();
          next += WriteBatchInternal::Count(&batch);
          ++members;
        }
        ASSERT_EQ(first + count, next);
        if (members > 1) {
          ++groups[p];
        }
      }
    }
  }
  for (size_t i = 0; i < batches.size(); ++i) {
    EXPECT_EQ(1, seen[i]) << "batch " << i;
  }
  for (size_t p = 0; p < kPartitions; ++p) {
    EXPECT_GE(groups[p], 1) << "partition " << p << " wrote no group";
  }
}

// Q4 (G-57): FlushWAL(true) at N = 4 with manual_wal_flush and a writer on
// every partition: a crash that drops unsynced data afterwards keeps every
// write acknowledged before the call started, and each writer's recovered
// writes are a prefix of its own. Neuter: FlushWAL flushes only partition
// 0's buffer; writes acknowledged on partitions 1..3 are lost.
TEST_P(DBWalPartitionCoverageTest, FlushWalSyncsEveryPartitionsAckedWrites) {
  constexpr size_t kPartitions = 4;
  constexpr uint64_t kMaxKeys = 1000000;
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  fault_fs->SetInjectUnsyncedDataLoss(true);
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_fs_env.get();
  options.manual_wal_flush = true;
  OpenWithPartitions(kPartitions, options);
  const std::vector<uint32_t> cf_ids(cf_for_partition_.begin(),
                                     cf_for_partition_.end());
  std::vector<uint64_t> acked_before;
  {
    PartitionWriters writers(db_.get(), handles_, cf_ids, kMaxKeys);
    writers.WaitForAcked(100);
    acked_before = writers.Acked();
    ASSERT_OK(db_->FlushWAL(/*sync=*/true));
    writers.Stop();
  }
  fault_fs->SetFilesystemActive(false);
  Close();
  ASSERT_OK(fault_fs->DropUnsyncedFileData());
  fault_fs->SetFilesystemActive(true);

  // Writes after the call may have reached some partitions' files and not
  // others; point-in-time recovery opens at the prefix below such a hole.
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ReopenWithPartitions(kPartitions, options);
  for (size_t p = 0; p < kPartitions; ++p) {
    SCOPED_TRACE("partition " + std::to_string(p));
    bool missing = false;
    for (uint64_t i = 0;; ++i) {
      const std::string value =
          Get(cf_for_partition_[p], "key" + std::to_string(i));
      if (value == "NOT_FOUND") {
        EXPECT_GE(i, acked_before[p]) << "acknowledged key" << i << " lost";
        missing = true;
        // Past the drop no later write of this writer may survive.
        for (uint64_t j = i + 1; j < i + 1000; ++j) {
          EXPECT_EQ("NOT_FOUND",
                    Get(cf_for_partition_[p], "key" + std::to_string(j)));
        }
        break;
      }
      EXPECT_EQ("value" + std::to_string(i), value);
    }
    EXPECT_TRUE(missing);
  }
  Close();
}

// Q4 (G-57): LockWAL at N = 4 under writers on every partition. A write held
// inside partition 1's WAL append when LockWAL starts must finish before
// LockWAL returns; while locked, no partition's WAL grows and no writer is
// acknowledged; after UnlockWAL every writer completes again. Neuter: lock
// only partition 0's write queue; LockWAL returns while the held append is
// pending, and partition 1's WAL grows under the lock.
TEST_P(DBWalPartitionCoverageTest, LockWalHoldsEveryPartitionUnderWrites) {
  constexpr size_t kPartitions = 4;
  constexpr uint64_t kMaxKeys = 1000000;
  const Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(kPartitions, options);
  auto wal_sizes = [&] {
    std::vector<uint64_t> sizes;
    for (size_t p = 0; p < kPartitions; ++p) {
      uint64_t size = 0;
      EXPECT_OK(env_->GetFileSize(
          LogFileName(dbname_, dbfull()->TEST_CurrentWalNumber(p)), &size));
      sizes.push_back(size);
    }
    return sizes;
  };

  port::Mutex mu;
  port::CondVar cv(&mu);
  bool held = false;
  bool release = false;
  std::thread::id held_thread;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void*) {
        MutexLock lock(&mu);
        if (held || std::this_thread::get_id() != held_thread) {
          return;
        }
        held = true;
        cv.SignalAll();
        while (!release) {
          cv.Wait();
        }
      });
  port::Thread held_writer([&] {
    {
      MutexLock lock(&mu);
      held_thread = std::this_thread::get_id();
    }
    ASSERT_OK(Put(cf_for_partition_[1], "held", "value"));
  });
  {
    MutexLock lock(&mu);
    while (!held) {
      cv.Wait();
    }
  }
  const std::vector<uint32_t> cf_ids(cf_for_partition_.begin(),
                                     cf_for_partition_.end());
  PartitionWriters writers(db_.get(), handles_, cf_ids, kMaxKeys);
  std::atomic<bool> locked{false};
  std::atomic<bool> unlock{false};
  // UnlockWAL must come from the thread that called LockWAL.
  port::Thread locker([&] {
    ASSERT_OK(db_->LockWAL());
    locked.store(true);
    while (!unlock.load()) {
      std::this_thread::yield();
    }
    ASSERT_OK(db_->UnlockWAL());
  });
  // LockWAL has to wait for the held append; give it a second to return
  // without it, which only a LockWAL missing partition 1 does.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!locked.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::vector<uint64_t> sizes_locked;
  if (locked.load()) {
    sizes_locked = wal_sizes();
  }
  {
    MutexLock lock(&mu);
    release = true;
    cv.SignalAll();
  }
  held_writer.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    while (!locked.load()) {
      std::this_thread::yield();
    }
    return Status::OK();
  }));
  if (sizes_locked.empty()) {
    sizes_locked = wal_sizes();
  }
  // Let acknowledgements of appends finished before the lock land.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const std::vector<uint64_t> acked_locked = writers.Acked();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(sizes_locked, wal_sizes());
  EXPECT_EQ(acked_locked, writers.Acked());

  unlock.store(true);
  locker.join();
  const uint64_t resume_to =
      *std::max_element(acked_locked.begin(), acked_locked.end()) + 10;
  ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
    writers.WaitForAcked(resume_to);
    return Status::OK();
  }));
  writers.Stop();
  ASSERT_EQ("value", Get(cf_for_partition_[1], "held"));
}

INSTANTIATE_TEST_CASE_P(DBWalPartitionCoverageTest, DBWalPartitionCoverageTest,
                        testing::Values(false));

INSTANTIATE_TEST_CASE_P(DBWalAppendFailureTest, DBWalAppendFailureTest,
                        testing::Values(false));

INSTANTIATE_TEST_CASE_P(DBWritePartitionTest, DBWritePartitionTest,
                        testing::Bool());

INSTANTIATE_TEST_CASE_P(DBWriteTestInstance, DBWriteTest,
                        testing::Values(DBTestBase::kDefault,
                                        DBTestBase::kConcurrentWALWrites,
                                        DBTestBase::kPipelinedWrite));

// A non-blocking unbatched join must refuse an active write stall.
TEST(WriteThreadTest, EnterUnbatchedNonBlockingRefusesDuringStall) {
  Options options;
  ImmutableDBOptions db_options(options);
  WriteThread write_thread(db_options);
  InstrumentedMutex mu;

  // No stall: joining succeeds and must be paired with ExitUnbatched().
  {
    WriteThread::Writer w;
    InstrumentedMutexLock l(&mu);
    ASSERT_TRUE(write_thread.EnterUnbatchedNonBlocking(&w, &mu));
    ASSERT_OK(w.status);
    EXPECT_FALSE(w.no_slowdown);
    EXPECT_FALSE(w.non_blocking_join);
    write_thread.ExitUnbatched(&w);
  }

  // Stalled: refuses rather than parking on stall_cv_. Reaching the assertion
  // at all is the point -- EnterUnbatched() would never have returned.
  {
    InstrumentedMutexLock l(&mu);
    write_thread.BeginWriteStall();
    ASSERT_NE(0, write_thread.GetBegunCountOfOutstandingStall());

    WriteThread::Writer w;
    ASSERT_FALSE(write_thread.EnterUnbatchedNonBlocking(&w, &mu));
    ASSERT_TRUE(w.status.IsIncomplete());
    EXPECT_FALSE(w.no_slowdown);
    EXPECT_FALSE(w.non_blocking_join);

    // Not linked, so ExitUnbatched() must not be called; ending the stall is
    // enough to leave the queue clean.
    write_thread.EndWriteStall();
    ASSERT_EQ(0, write_thread.GetBegunCountOfOutstandingStall());
  }

  {
    WriteThread::Writer w;
    InstrumentedMutexLock l(&mu);
    ASSERT_TRUE(write_thread.EnterUnbatchedNonBlocking(&w, &mu));
    write_thread.ExitUnbatched(&w);
  }
}

TEST(WriteThreadTest, EnterUnbatchedNonBlockingFinishesAfterLeadershipStall) {
  Options options;
  ImmutableDBOptions db_options(options);
  WriteThread write_thread(db_options);
  InstrumentedMutex mu;
  InstrumentedMutex signal_mu;
  InstrumentedCondVar signal_cv(&signal_mu);
  bool before_lock = false;
  bool release_joiner = false;
  bool entered = true;

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::EnterUnbatchedNonBlocking:BeforeLock", [&](void*) {
        InstrumentedMutexLock l(&signal_mu);
        before_lock = true;
        signal_cv.SignalAll();
        while (!release_joiner) {
          signal_cv.Wait();
        }
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_points([] {
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  });

  port::Thread joiner([&] {
    WriteThread::Writer w;
    InstrumentedMutexLock l(&mu);
    entered = write_thread.EnterUnbatchedNonBlocking(&w, &mu);
    if (entered) {
      write_thread.ExitUnbatched(&w);
    }
  });

  {
    InstrumentedMutexLock l(&signal_mu);
    while (!before_lock) {
      signal_cv.Wait();
    }
  }
  {
    InstrumentedMutexLock l(&mu);
    write_thread.BeginWriteStall();
  }
  {
    InstrumentedMutexLock l(&signal_mu);
    release_joiner = true;
    signal_cv.SignalAll();
  }

  joiner.join();
  EXPECT_TRUE(entered);
  {
    InstrumentedMutexLock l(&mu);
    write_thread.EndWriteStall();
  }
}

// A WBM join queues behind an ordinary leader so sustained write traffic
// cannot starve the memtable switch.
TEST(WriteThreadTest, EnterUnbatchedNonBlockingQueuesBehindOrdinaryLeader) {
  Options options;
  ImmutableDBOptions db_options(options);
  WriteThread write_thread(db_options);
  InstrumentedMutex mu;
  InstrumentedMutex signal_mu;
  InstrumentedCondVar signal_cv(&signal_mu);
  bool queued = false;

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::EnterUnbatched:Wait", [&](void*) {
        InstrumentedMutexLock l(&signal_mu);
        queued = true;
        signal_cv.SignalAll();
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_points([] {
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  });

  WriteThread::Writer leader;
  mu.Lock();
  write_thread.EnterUnbatched(&leader, &mu);
  mu.Unlock();

  bool entered = false;
  port::Thread joiner([&] {
    WriteThread::Writer follower;
    InstrumentedMutexLock l(&mu);
    entered = write_thread.EnterUnbatchedNonBlocking(&follower, &mu);
    if (entered) {
      write_thread.ExitUnbatched(&follower);
    }
  });

  {
    InstrumentedMutexLock l(&signal_mu);
    while (!queued) {
      signal_cv.Wait();
    }
  }
  mu.Lock();
  write_thread.ExitUnbatched(&leader);
  mu.Unlock();
  joiner.join();
  EXPECT_TRUE(entered);
}

// FU14 repro H3: a point-in-time open must stop at a reported tail gap even
// after an earlier pass reset the shared recovery status. Partition `torn`'s
// WAL (numbered below the other's) holds idx 1 and a torn idx 3; the other
// holds idx 2 and idx 4. Idx 2 passes the torn WAL's loss (the stop arms and
// resumes, since idx 2 continues the sequence) and resets status, so the
// later tail gap at idx 3 maps through the no-report branch, which cuts
// without stopping. Correct: keys of idx 1 and 2 present, idx 3 and 4 absent.
TEST_P(DBWalPartitionCoverageTest, FU14H3ReportedTailGapAfterPassStopsPIT) {
  Options options = IndexedWalOptions(CurrentOptions());
  options.paranoid_checks = true;
  DestroyAndReopen(options);
  writes_.clear();
  OpenWithPartitions(2, options);
  const size_t torn =
      dbfull()->TEST_CurrentWalNumber(0) < dbfull()->TEST_CurrentWalNumber(1)
          ? 0
          : 1;
  const size_t other = 1 - torn;
  LoggedPut(cf_for_partition_[torn], "i1");
  LoggedPut(cf_for_partition_[other], "i2");
  LoggedPut(cf_for_partition_[torn], "i3");
  LoggedPut(cf_for_partition_[other], "i4");
  ASSERT_EQ(writes_[0].wal, writes_[2].wal);
  ASSERT_EQ(writes_[1].wal, writes_[3].wal);
  ASSERT_LT(writes_[0].wal, writes_[1].wal);
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  Close();

  const LoggedWrite& i3 = writes_[IndexOf("i3")];
  TruncateImageWal(image, i3.wal, (i3.begin + i3.end) / 2);
  AssertCells(image, 2, options,
              {{kPointInTime, Status::kOk, IndexOf("i3"),
                "point-in-time stops at the reported tail gap at idx 3"}},
              {});
}

// FU14 repro H4: a superseded-only hole with a live record above it. Crash 1
// loses idx 5 cleanly at a record boundary of partition a's WAL, while
// partition b's WAL holds idx 2, 4 and 6..9. Open 1 stops at 5 and cuts
// partition b's WAL at 5 (the supersession goes into a new WAL); the old
// WALs stay (avoid_flush_during_recovery). The reopened DB writes na (a), nb
// (b), na2 (a), which reissue indices partition b's old WAL still holds.
// Power loss 2 truncates partition b's new WAL back to its head, losing nb
// cleanly. On open 2 the only holder of nb's index is the superseded old copy
// and no member is tail-impure, so the merge skips the index and applies na2
// above it. Correct: point-in-time stops at nb's index, so na2 is absent.
// (Open 1 also writes an empty record at idx 5, so na lands at 6, nb at 7
// and na2 at 8; the test reads the indices back rather than assuming them.)
static std::vector<uint64_t> WalNumbersIn(const std::string& dir) {
  std::vector<std::string> children;
  EXPECT_OK(Env::Default()->GetChildren(dir, &children));
  std::vector<uint64_t> numbers;
  for (const std::string& child : children) {
    uint64_t number = 0;
    FileType type;
    if (ParseFileName(child, &number, &type) && type == kWalFile) {
      numbers.push_back(number);
    }
  }
  std::sort(numbers.begin(), numbers.end());
  return numbers;
}

// Every indexed record of WAL `wal` in `dir`: {wal_index, payload}.
static std::vector<std::pair<uint64_t, std::string>> IndexedRecordsIn(
    Env* env, const std::string& dir, uint64_t wal) {
  std::vector<std::pair<uint64_t, std::string>> records;
  std::unique_ptr<SequentialFileReader> file_reader;
  EXPECT_OK(SequentialFileReader::Create(
      env->GetFileSystem(), LogFileName(dir, wal), FileOptions(), &file_reader,
      /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
  log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                     /*reporter=*/nullptr, /*checksum=*/true, wal);
  std::string scratch;
  Slice record;
  while (reader.ReadRecord(&record, &scratch)) {
    if (reader.GetLastReadWALIndex() != 0) {
      records.emplace_back(reader.GetLastReadWALIndex(), record.ToString());
    }
  }
  return records;
}

// wal_index of the record in `dir` whose payload holds `key`, or zero.
static uint64_t WalIndexOfKey(Env* env, const std::string& dir,
                              const std::string& key) {
  for (uint64_t wal : WalNumbersIn(dir)) {
    for (const auto& [index, payload] : IndexedRecordsIn(env, dir, wal)) {
      if (payload.find(key) != std::string::npos) {
        return index;
      }
    }
  }
  return 0;
}

TEST_P(DBWalPartitionCoverageTest, FU14H4SupersededOnlyHoleWithLiveAboveStops) {
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  writes_.clear();
  OpenWithPartitions(2, options);
  const size_t a =
      dbfull()->TEST_CurrentWalNumber(0) < dbfull()->TEST_CurrentWalNumber(1)
          ? 0
          : 1;
  const size_t b = 1 - a;
  const int cf_a = cf_for_partition_[a];
  const int cf_b = cf_for_partition_[b];
  LoggedPut(cf_a, "o1");
  LoggedPut(cf_b, "o2");
  LoggedPut(cf_a, "o3");
  LoggedPut(cf_b, "o4");
  LoggedPut(cf_a, "o5");
  for (const char* key : {"o6", "o7", "o8", "o9"}) {
    LoggedPut(cf_b, key);
  }
  const uint64_t old_wal_a = writes_[0].wal;
  const uint64_t old_wal_b = writes_[1].wal;
  ASSERT_EQ(old_wal_a, writes_[IndexOf("o5")].wal);
  ASSERT_EQ(old_wal_b, writes_[IndexOf("o9")].wal);
  const std::string image1 = dbname_ + "_crash1";
  CopyDBAsCrashImage(image1);
  Close();
  // Crash 1: idx 5 lost cleanly, at its record's start.
  const LoggedWrite& o5 = writes_[IndexOf("o5")];
  TruncateImageWal(image1, o5.wal, o5.begin);

  // Open 1, point-in-time: stops at the hole at idx 5.
  const std::string image2 = dbname_ + "_crash2";
  uint64_t wal_b = 0;
  uint64_t wal_b_head = 0;
  {
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    ASSERT_OK(OpenCopy(image1, 2, options, kPointInTime, &handles, &db));
    for (const char* key : {"o1", "o2", "o3", "o4"}) {
      std::string value;
      ASSERT_OK(db->Get(ReadOptions(), handles[writes_[IndexOf(key)].cf], key,
                        &value))
          << key;
    }
    for (const char* key : {"o5", "o6", "o7", "o8", "o9"}) {
      std::string value;
      ASSERT_TRUE(
          db->Get(ReadOptions(), handles[writes_[IndexOf(key)].cf], key, &value)
              .IsNotFound())
          << key;
    }
    DBImpl* impl = static_cast_with_check<DBImpl>(db.get());
    ASSERT_EQ(2U, impl->TEST_NumWalPartitions());
    // The old WALs are retained (avoid_flush_during_recovery).
    ASSERT_OK(env_->FileExists(LogFileName(image1, old_wal_a)));
    ASSERT_OK(env_->FileExists(LogFileName(image1, old_wal_b)));
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_a], "na", ValueOf("na")));
    wal_b = impl->TEST_CurrentWalNumber(b);
    ASSERT_GT(wal_b, old_wal_b);
    ASSERT_OK(env_->GetFileSize(LogFileName(image1, wal_b), &wal_b_head));
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_b], "nb", ValueOf("nb")));
    uint64_t wal_b_after = 0;
    ASSERT_OK(env_->GetFileSize(LogFileName(image1, wal_b), &wal_b_after));
    ASSERT_GT(wal_b_after, wal_b_head);
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_a], "na2", ValueOf("na2")));
    // Power loss 2: the files as they stand.
    CopyDir(image1, image2);
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
  }
  // The construction: nb reissued an index partition b's old WAL still holds
  // (superseded), and na2 sits above it.
  const uint64_t nb_index = WalIndexOfKey(env_, image2, "nb");
  const uint64_t na2_index = WalIndexOfKey(env_, image2, "na2");
  ASSERT_NE(0U, nb_index);
  ASSERT_GT(na2_index, nb_index);
  bool old_b_holds_nb_index = false;
  for (const auto& record : IndexedRecordsIn(env_, image2, old_wal_b)) {
    old_b_holds_nb_index = old_b_holds_nb_index || record.first == nb_index;
  }
  ASSERT_TRUE(old_b_holds_nb_index) << "nb at wal_index " << nb_index;
  // New nb lost cleanly: partition b's new WAL back to its head.
  TruncateImageWal(image2, wal_b, wal_b_head);
  ASSERT_EQ(0U, WalIndexOfKey(env_, image2, "nb"));

  // Open 2, point-in-time.
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> db;
  const Status s = OpenCopy(image2, 2, options, kPointInTime, &handles, &db);
  ASSERT_OK(s);
  auto get = [&](int cf, const std::string& key) {
    std::string value;
    const Status g = db->Get(ReadOptions(), handles[cf], key, &value);
    EXPECT_TRUE(g.ok() || g.IsNotFound()) << g.ToString();
    return g.ok() ? value : std::string("NOT_FOUND");
  };
  EXPECT_EQ(ValueOf("o1"), get(cf_a, "o1"));
  EXPECT_EQ(ValueOf("o2"), get(cf_b, "o2"));
  EXPECT_EQ(ValueOf("o3"), get(cf_a, "o3"));
  EXPECT_EQ(ValueOf("o4"), get(cf_b, "o4"));
  EXPECT_EQ("NOT_FOUND", get(cf_a, "o5"));
  for (const char* key : {"o6", "o7", "o8", "o9"}) {
    EXPECT_EQ("NOT_FOUND", get(cf_b, key)) << key << " is superseded";
  }
  EXPECT_EQ(ValueOf("na"), get(cf_a, "na"));
  EXPECT_EQ("NOT_FOUND", get(cf_b, "nb")) << "nb was lost";
  EXPECT_EQ("NOT_FOUND", get(cf_a, "na2"))
      << "na2 (wal_index " << na2_index << ") applied above the lost nb "
      << "(wal_index " << nb_index << ")";
  for (ColumnFamilyHandle* handle : handles) {
    EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
  }
}

// FU14 repro H5: a kNone N = 1 WAL whose synced size the MANIFEST tracks,
// kept alive by an unflushed column family across the upgrade to indexed
// N = 2. The second N = 2 open replays it as the unindexed prefix ahead of
// the merge, but the merge's expected membership still names it. Correct:
// the open succeeds and every key is present.
TEST_P(DBWalPartitionCoverageTest, FU14H5SyncedUnindexedPrefixWalReopens) {
  if (GetParam()) {
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  FlushAllColumnFamilies();
  const int cf_flushed = cf_for_partition_[0];
  const int cf_kept = cf_for_partition_[1];
  Options none = options;
  none.partition_wal_usage = PartitionWALUsage::kNone;
  ReopenWithPartitions(1, none);
  ASSERT_OK(Put(cf_flushed, "unindexed_flushed", "value"));
  ASSERT_OK(Put(cf_kept, "unindexed_kept", "value"));
  const uint64_t unindexed_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(Flush(cf_flushed));
  ASSERT_NE(unindexed_wal, dbfull()->TEST_CurrentWalNumber(0));
  {
    const auto& wals = dbfull()->GetVersionSet()->GetWalSet().GetWals();
    const auto it = wals.find(unindexed_wal);
    ASSERT_TRUE(it != wals.end()) << "WAL " << unindexed_wal;
    ASSERT_TRUE(it->second.HasSyncedSize()) << "WAL " << unindexed_wal;
  }
  SCOPED_TRACE("unindexed kNone WAL " + std::to_string(unindexed_wal));
  ReopenWithPartitions(2, options);
  ASSERT_OK(env_->FileExists(LogFileName(dbname_, unindexed_wal)));
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "indexed" + std::to_string(p), "v"));
  }
  const std::vector<int> indexed_cfs = cf_for_partition_;
  Close();

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  Options reopen = options;
  reopen.wal_partitions = 2;
  reopen.two_write_queues = GetParam();
  const Status s = TryReopenWithColumnFamilies(cf_names, reopen);
  ASSERT_OK(s);
  EXPECT_EQ("value", Get(cf_flushed, "unindexed_flushed"));
  EXPECT_EQ("value", Get(cf_kept, "unindexed_kept"));
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_EQ("v", Get(indexed_cfs[p], "indexed" + std::to_string(p)));
  }
}

// FU14 repro H6: a DB created directly at N = 2, so its first WALs attest no
// predecessor span. Idx 1 lands in partition 0's WAL; idx 2 in partition 1's,
// torn mid-record. Correct: point-in-time (and tolerate) open with idx 1's
// key present and idx 2's absent.
TEST_P(DBWalPartitionCoverageTest, FU14H6TornFirstRecordInBrandNewNTwoDb) {
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  Close();
  ASSERT_OK(DestroyDB(dbname_, options));
  SyncPoint::GetInstance()->EnableProcessing();
  std::vector<std::string> cf_names = {kDefaultColumnFamilyName};
  cf_for_partition_.assign(kPartitions, -1);
  size_t covered = 0;
  for (uint32_t cf_id = 1; covered < kPartitions; ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = kPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kPartitions, &partition));
    cf_names.push_back("cf" + std::to_string(cf_id));
    if (cf_for_partition_[partition] < 0) {
      cf_for_partition_[partition] = static_cast<int>(cf_id);
      ++covered;
    }
  }
  options.create_if_missing = true;
  options.create_missing_column_families = true;
  options.wal_partitions = kPartitions;
  options.two_write_queues = GetParam();
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  ASSERT_EQ(kPartitions, dbfull()->TEST_NumWalPartitions());
  ASSERT_EQ(0U, dbfull()->GetVersionSet()->LastSequence());
  writes_.clear();
  LoggedPut(cf_for_partition_[0], "k1");
  LoggedPut(cf_for_partition_[1], "k2");
  ASSERT_NE(writes_[0].wal, writes_[1].wal);
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  Close();
  const LoggedWrite& k2 = writes_[IndexOf("k2")];
  SCOPED_TRACE("torn partition-1 WAL " + std::to_string(k2.wal));
  TruncateImageWal(image, k2.wal, (k2.begin + k2.end) / 2);
  options.create_if_missing = false;
  // One call per mode: AssertCells stops at its first failed open.
  for (const WALRecoveryMode mode : {kPointInTime, kTolerate}) {
    AssertCells(image, kPartitions, options,
                {{mode, Status::kOk, 1, "idx 2 is a torn tail"}}, {});
  }
}

// GC19 Q1: a reported tail gap stops point-in-time replay whichever
// partition's WAL tore. With the lower-numbered WAL torn this is H3's shape
// (an earlier pass consumed the report); with the higher-numbered one the
// report is still pending when the gap surfaces. Tolerate and absolute fail
// at the gap as before. Neuter: ignore TailGap::reported in map_tail_gap; the
// lower-numbered case recovers i4 above the gap. The higher-numbered case is
// caught by map_tail_gap's pending-report branch instead: neuter that
// branch's point-in-time stop (consume the report without arming the stop)
// and it recovers i4 above the gap.
TEST_P(DBWalPartitionCoverageTest, GC19ReportedTailGapStopsPITEitherPartition) {
  for (const bool torn_lower : {true, false}) {
    SCOPED_TRACE(torn_lower ? "torn WAL numbered lower"
                            : "torn WAL numbered higher");
    Options options = IndexedWalOptions(CurrentOptions());
    options.paranoid_checks = true;
    DestroyAndReopen(options);
    writes_.clear();
    OpenWithPartitions(2, options);
    const bool zero_lower =
        dbfull()->TEST_CurrentWalNumber(0) < dbfull()->TEST_CurrentWalNumber(1);
    const size_t torn = zero_lower == torn_lower ? 0 : 1;
    const size_t other = 1 - torn;
    LoggedPut(cf_for_partition_[torn], "i1");
    LoggedPut(cf_for_partition_[other], "i2");
    LoggedPut(cf_for_partition_[torn], "i3");
    LoggedPut(cf_for_partition_[other], "i4");
    ASSERT_EQ(writes_[0].wal, writes_[2].wal);
    ASSERT_EQ(writes_[1].wal, writes_[3].wal);
    ASSERT_EQ(torn_lower, writes_[0].wal < writes_[1].wal);
    const std::string image = dbname_ + "_image";
    CopyDBAsCrashImage(image);
    Close();

    const LoggedWrite& i3 = writes_[IndexOf("i3")];
    TruncateImageWal(image, i3.wal, (i3.begin + i3.end) / 2);
    AssertCells(image, 2, options,
                {{kPointInTime, Status::kOk, IndexOf("i3"),
                  "point-in-time stops at the reported tail gap"},
                 {kTolerate, Status::kCorruption, 0, "records above the gap"},
                 {kAbsolute, Status::kCorruption, 0, "records above the gap"}},
                {});
  }
}

// GC19 Q3: an unindexed prefix replayed ahead of the merge leaves indexed
// WALs demanded. A MANIFEST-tracked mid-chain indexed WAL deleted after the
// upgrade still fails every mode, naming it: WalSet::CheckWals refuses first,
// and the merge's expected set backs it ("named but not held").
TEST_P(DBWalPartitionCoverageTest, GC19MissingIndexedWalAfterPrefixFails) {
  if (GetParam()) {
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  FlushAllColumnFamilies();
  const int cf_flushed = cf_for_partition_[0];
  const int cf_kept = cf_for_partition_[1];
  Options none = options;
  none.partition_wal_usage = PartitionWALUsage::kNone;
  ReopenWithPartitions(1, none);
  ASSERT_OK(Put(cf_flushed, "unindexed_flushed", "value"));
  ASSERT_OK(Put(cf_kept, "unindexed_kept", "value"));
  const uint64_t unindexed_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(Flush(cf_flushed));
  ReopenWithPartitions(2, options);
  ASSERT_OK(env_->FileExists(LogFileName(dbname_, unindexed_wal)));
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "indexed" + std::to_string(p), "v"));
  }
  const uint64_t missing = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_GT(missing, unindexed_wal);
  Rotate();
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "rotated" + std::to_string(p), "v"));
  }
  ASSERT_NE(missing, dbfull()->TEST_CurrentWalNumber(1));
  ASSERT_OK(db_->SyncWAL());
  Close();
  ASSERT_OK(env_->DeleteFile(LogFileName(dbname_, missing)));

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  for (const WALRecoveryMode mode :
       {kTolerate, kAbsolute, kPointInTime, kSkip}) {
    SCOPED_TRACE(static_cast<int>(mode));
    Options reopen = options;
    reopen.wal_partitions = 2;
    reopen.wal_recovery_mode = mode;
    const Status s = TryReopenWithColumnFamilies(cf_names, reopen);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find(std::to_string(missing)))
        << s.ToString();
  }
}

// GC23 Q2: as GC19MissingIndexedWalAfterPrefixFails, but the deleted WAL is
// partition 1's current (never rotated) indexed WAL, after SyncWAL. The
// MANIFEST records a WAL's synced size only once the WAL is inactive
// (FinishWalPartitionSync, as upstream), so a current WAL stays untracked and
// neither WalSet::CheckWals nor the merge's expected set demands it. Nothing
// attests it either: it has no successor. Its records are therefore judged by
// the surviving files alone. On top, the lost record is a lost tail and every
// mode opens without it. Below a survivor it was a gap at the base: the lost
// index sat under the lower bound the survivors set, so every mode kept the
// later write without the earlier one. GC24 Q1 flips those cells: each WAL
// the upgrade open created carries the index floor in its head, the lost
// index surfaces as a hole, and point-in-time opens the prefix below it while
// tolerate and absolute fail closed. Skip skips the hole (GC26 Q4, D-10).
TEST_P(DBWalPartitionCoverageTest, GC23MissingCurrentIndexedWalAfterPrefix) {
  if (GetParam()) {
    return;
  }
  for (const bool lost_on_top : {false, true}) {
    SCOPED_TRACE(lost_on_top ? "lost record on top" : "lost record below");
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    FlushAllColumnFamilies();
    const int cf_flushed = cf_for_partition_[0];
    const int cf_kept = cf_for_partition_[1];
    Options none = options;
    none.partition_wal_usage = PartitionWALUsage::kNone;
    ReopenWithPartitions(1, none);
    ASSERT_OK(Put(cf_flushed, "unindexed_flushed", "value"));
    ASSERT_OK(Put(cf_kept, "unindexed_kept", "value"));
    const uint64_t unindexed_wal = dbfull()->TEST_CurrentWalNumber(0);
    ASSERT_OK(Flush(cf_flushed));
    ReopenWithPartitions(2, options);
    ASSERT_OK(env_->FileExists(LogFileName(dbname_, unindexed_wal)));
    writes_.clear();
    if (lost_on_top) {
      LoggedPut(cf_for_partition_[0], "kept");
      LoggedPut(cf_for_partition_[1], "lost");
    } else {
      LoggedPut(cf_for_partition_[1], "lost");
      LoggedPut(cf_for_partition_[0], "kept");
    }
    const uint64_t missing = dbfull()->TEST_CurrentWalNumber(1);
    ASSERT_EQ(missing, writes_[IndexOf("lost")].wal);
    ASSERT_GT(missing, unindexed_wal);
    ASSERT_OK(db_->SyncWAL());
    ASSERT_EQ(missing, dbfull()->TEST_CurrentWalNumber(1));
    {
      const auto& wals = dbfull()->GetVersionSet()->GetWalSet().GetWals();
      ASSERT_TRUE(wals.find(missing) == wals.end()) << "WAL " << missing;
      const auto it = wals.find(unindexed_wal);
      ASSERT_TRUE(it != wals.end() && it->second.HasSyncedSize())
          << "WAL " << unindexed_wal;
    }
    const std::string image = dbname_ + "_image";
    CopyDBAsCrashImage(image);
    Close();
    ASSERT_OK(env_->DeleteFile(LogFileName(image, missing)));
    SCOPED_TRACE("deleted current WAL " + std::to_string(missing));
    if (!lost_on_top) {
      // AssertCells checks a prefix of writes_: list the survivor first.
      std::swap(writes_[0], writes_[1]);
    }
    if (lost_on_top) {
      AssertCells(image, 2, options, EveryModeOpens(1, "lost tail"), {});
    } else {
      AssertCells(image, 2, options,
                  {{kTolerate, Status::kCorruption, 0, "hole at the floor"},
                   {kAbsolute, Status::kCorruption, 0, "hole at the floor"},
                   {kPointInTime, Status::kOk, 0, "prefix below the hole"},
                   {kSkip, Status::kOk, 1,
                    "skips the hole at the floor (GC26 Q4, D-10)"}},
                  {});
    }
  }
}

// GC24 Q1: a lost current WAL whose synced record lies below a surviving
// record of another partition, at N in {2, 8}, with and without
// manual_wal_flush. The WALs the upgrade open creates attest no predecessor,
// so only their head floor bounds the loss. Tolerate and absolute fail at
// the hole; point-in-time opens the prefix below it, which holds the
// unindexed history and neither indexed write; skip skips the hole and keeps
// "kept" (GC26 Q4, D-10). Neuter: drop the head floors (writer or merge
// side); every mode opens with "kept" and without "lost".
TEST_P(DBWalPartitionCoverageTest, GC24LostCurrentWalBelowSurvivorIsAHole) {
  if (GetParam()) {
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    for (const bool manual_flush : {false, true}) {
      SCOPED_TRACE("N=" + std::to_string(partitions) +
                   (manual_flush ? " manual_wal_flush" : ""));
      Options options = IndexedWalOptions(CurrentOptions());
      options.manual_wal_flush = manual_flush;
      DestroyAndReopen(options);
      OpenWithPartitions(partitions, options);
      FlushAllColumnFamilies();
      Options none = options;
      none.partition_wal_usage = PartitionWALUsage::kNone;
      ReopenWithPartitions(1, none);
      const int unindexed_cf = cf_for_partition_[0];
      ASSERT_OK(Put(unindexed_cf, "unindexed", "value"));
      ASSERT_OK(db_->FlushWAL(/*sync=*/true));
      ReopenWithPartitions(partitions, options);
      const size_t lost_partition = partitions - 1;
      ASSERT_OK(Put(cf_for_partition_[lost_partition], "lost", "value"));
      for (size_t p = 0; p < lost_partition; ++p) {
        ASSERT_OK(Put(cf_for_partition_[p], "kept" + std::to_string(p), "v"));
      }
      ASSERT_OK(db_->FlushWAL(/*sync=*/true));
      const uint64_t missing = dbfull()->TEST_CurrentWalNumber(lost_partition);
      {
        const auto& wals = dbfull()->GetVersionSet()->GetWalSet().GetWals();
        ASSERT_TRUE(wals.find(missing) == wals.end()) << "WAL " << missing;
      }
      const std::string image = dbname_ + "_image";
      CopyDBAsCrashImage(image);
      const std::vector<int> cfs = cf_for_partition_;
      Close();
      ASSERT_OK(env_->DeleteFile(LogFileName(image, missing)));
      const std::string copy = image + "_open";
      for (const WALRecoveryMode mode :
           {kTolerate, kAbsolute, kPointInTime, kSkip}) {
        SCOPED_TRACE("wal_recovery_mode=" +
                     std::to_string(static_cast<int>(mode)));
        CopyDir(image, copy);
        std::vector<ColumnFamilyHandle*> handles;
        std::unique_ptr<DB> db;
        const Status s =
            OpenCopy(copy, partitions, options, mode, &handles, &db);
        if (mode == kTolerate || mode == kAbsolute) {
          ASSERT_TRUE(s.IsCorruption()) << s.ToString();
          EXPECT_NE(std::string::npos,
                    s.ToString().find("unexplained hole at wal_index"))
              << s.ToString();
          continue;
        }
        ASSERT_OK(s);
        std::string value;
        EXPECT_OK(
            db->Get(ReadOptions(), handles[unindexed_cf], "unindexed", &value));
        EXPECT_TRUE(
            db->Get(ReadOptions(), handles[cfs[lost_partition]], "lost", &value)
                .IsNotFound());
        // GC26 Q4 (D-10): skip skips the hole and keeps the writes above it.
        for (size_t p = 0; p < lost_partition; ++p) {
          const Status kept = db->Get(ReadOptions(), handles[cfs[p]],
                                      "kept" + std::to_string(p), &value);
          if (mode == kSkip) {
            EXPECT_OK(kept) << "kept" << p;
          } else {
            EXPECT_TRUE(kept.IsNotFound())
                << "kept" << p << " recovered above the lost record";
          }
        }
        for (ColumnFamilyHandle* handle : handles) {
          EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
        }
      }
    }
  }
}

// GC19 Q4: a DB created directly at N = 2 with partition 1's first record
// torn. The history floor bounds the head, so absolute fails on the torn
// tail as for any other and skip opens with idx 1 (point-in-time and
// tolerate are H6). Neuter: drop the floor from HeadBounded; every mode
// fails with "head damaged and unattested".
TEST_P(DBWalPartitionCoverageTest, GC19TornFirstRecordInBrandNewNTwoDbModes) {
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  Close();
  ASSERT_OK(DestroyDB(dbname_, options));
  std::vector<std::string> cf_names = {kDefaultColumnFamilyName};
  cf_for_partition_.assign(kPartitions, -1);
  size_t covered = 0;
  for (uint32_t cf_id = 1; covered < kPartitions; ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = kPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kPartitions, &partition));
    cf_names.push_back("cf" + std::to_string(cf_id));
    if (cf_for_partition_[partition] < 0) {
      cf_for_partition_[partition] = static_cast<int>(cf_id);
      ++covered;
    }
  }
  options.create_if_missing = true;
  options.create_missing_column_families = true;
  options.wal_partitions = kPartitions;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  ASSERT_EQ(0U, dbfull()->GetVersionSet()->LastSequence());
  writes_.clear();
  LoggedPut(cf_for_partition_[0], "k1");
  LoggedPut(cf_for_partition_[1], "k2");
  ASSERT_NE(writes_[0].wal, writes_[1].wal);
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  Close();
  const LoggedWrite& k2 = writes_[IndexOf("k2")];
  TruncateImageWal(image, k2.wal, (k2.begin + k2.end) / 2);
  options.create_if_missing = false;
  AssertCells(image, kPartitions, options,
              {{kAbsolute, Status::kCorruption, 0, "torn tail"},
               {kSkip, Status::kOk, 1, "idx 2 is a torn tail"}},
              {});
}

// GC19 Q4 negative: once a flush persisted a sequence the history floor is
// off, and with spans suppressed nothing else bounds a partition WAL's head.
// A torn first record there may hide a loss below it, so every mode still
// fails with "head damaged and unattested". Neuter: let HeadBounded take the
// floor whatever the open's conditions; the opens succeed.
TEST_P(DBWalPartitionCoverageTest, GC19TornFirstRecordAfterFlushStaysDamaged) {
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  OpenWithPartitions(kPartitions, options);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::StartWALFile:WriteSpan",
      [](void* arg) { *static_cast<bool*>(arg) = false; });
  SyncPoint::GetInstance()->EnableProcessing();
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "flushed", "value"));
  }
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  writes_.clear();
  LoggedPut(cf_for_partition_[0], "live0");
  LoggedPut(cf_for_partition_[1], "live1");
  ASSERT_NE(writes_[0].wal, writes_[1].wal);
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  Close();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::StartWALFile:WriteSpan");
  const LoggedWrite& live1 = writes_[IndexOf("live1")];
  TruncateImageWal(image, live1.wal, (live1.begin + live1.end) / 2);
  const std::string copy = image + "_open";
  for (const WALRecoveryMode mode :
       {kTolerate, kAbsolute, kPointInTime, kSkip}) {
    SCOPED_TRACE(static_cast<int>(mode));
    CopyDir(image, copy);
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    const Status s = OpenCopy(copy, kPartitions, options, mode, &handles, &db);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos,
              s.ToString().find("head damaged and unattested"))
        << s.ToString();
  }
}

// GC23 Q3, ported from GC19's Muse candidate: the H6 image's remaining
// modes. Absolute fails as for any torn tail; skip opens with it dropped.
// (Point-in-time and tolerate are H6.) Neuter: drop the floor term from
// HeadBounded; both modes fail with "head damaged and unattested".
TEST_P(DBWalPartitionCoverageTest, GC19H6FloorSetAbsoluteAndSkipOutcomes) {
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  CreateBrandNewPartitionedDb(kPartitions, options);
  LoggedPut(cf_for_partition_[0], "k1");
  LoggedPut(cf_for_partition_[1], "k2");
  ASSERT_NE(writes_[0].wal, writes_[1].wal);
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  Close();
  const LoggedWrite& k2 = writes_[IndexOf("k2")];
  SCOPED_TRACE("torn partition-1 WAL " + std::to_string(k2.wal));
  TruncateImageWal(image, k2.wal, (k2.begin + k2.end) / 2);
  options.create_if_missing = false;
  AssertCells(
      image, kPartitions, options,
      {{kAbsolute, Status::kCorruption, 0, "absolute fails on the torn tail"}},
      {});
  AssertCells(image, kPartitions, options,
              {{kSkip, Status::kOk, 1, "skip drops the torn tail"}}, {});
}

// GC23 Q3, ported from GC19's Muse candidate: the H6 shape (spanless
// first-generation WALs, torn mid-first-record), but a rotation created
// second-generation WALs whose spans attest the first generation. The
// rotation records a minimum WAL to keep, so the history floor is off, and
// the torn partition-1 WAL is unattested (its successor's span tore with its
// marker) and numbered below the anchoring WAL, so nothing bounds it and
// every mode fails closed. Neuter: let HeadBounded hold for every member;
// the opens succeed. (GC23FloorSetLaterGenerationHeads covers the same shape
// with the floor set.)
TEST_P(DBWalPartitionCoverageTest, GC19H6FloorUnsetHeadDamageStillFails) {
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  CreateBrandNewPartitionedDb(kPartitions, options);
  LoggedPut(cf_for_partition_[0], "k1");
  LoggedPut(cf_for_partition_[1], "k2");
  ASSERT_NE(writes_[0].wal, writes_[1].wal);
  const uint64_t torn_wal = writes_[IndexOf("k2")].wal;
  Rotate();
  const uint64_t gen2_p0 = dbfull()->TEST_CurrentWalNumber(0);
  const uint64_t gen2_p1 = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_GT(gen2_p0, torn_wal);
  ASSERT_GT(gen2_p1, torn_wal);
  auto synced_size_recorded = [&](uint64_t wal) {
    const auto& wals = dbfull()->GetVersionSet()->GetWalSet().GetWals();
    const auto it = wals.find(wal);
    return it != wals.end() && it->second.HasSyncedSize();
  };
  // Nothing synced the torn WALs, so the tears below reach the merge instead
  // of tripping MANIFEST size verification.
  ASSERT_OK(db_->FlushWAL(false));
  ASSERT_FALSE(synced_size_recorded(torn_wal));
  ASSERT_FALSE(synced_size_recorded(gen2_p1));
  const std::string image = dbname_ + "_image";
  CopyDBAsCrashImage(image);
  Close();
  const LoggedWrite& k2 = writes_[IndexOf("k2")];
  SCOPED_TRACE("torn partition-1 WAL " + std::to_string(k2.wal));
  TruncateImageWal(image, k2.wal, (k2.begin + k2.end) / 2);
  TruncateImageWal(image, gen2_p1, log::kHeaderSize - 2);
  options.create_if_missing = false;
  const char* reason =
      "the floor is off, so the torn head may hide loss below it";
  AssertCells(image, kPartitions, options,
              {{kTolerate, Status::kCorruption, 0, reason},
               {kAbsolute, Status::kCorruption, 0, reason},
               {kPointInTime, Status::kCorruption, 0, reason},
               {kSkip, Status::kCorruption, 0, reason}},
              {});
}

// GC23 Q3: the history floor never reaches a member that is not the first
// of history. Recovery sets the floor only while no sequence is persisted and
// the MANIFEST records no minimum WAL to keep, and wal_partitions > 1
// requires MANIFEST WAL tracking. Both ways a brand-new N = 2 DB gets a second
// WAL generation record that minimum: a rotation, and a reopen (here one
// whose first-generation WALs were torn mid-first-record, so it recovered no
// sequence). With spans suppressed nothing attests, so the second
// generation's damaged head (b1 corrupted, b3 surviving behind it) is
// examined by the lower-bound gate without the floor, and every mode fails
// with "head damaged and unattested". The brand-new case, where every member
// is first of history and the floor bounds them, is H6 and
// GC19TornFirstRecordInBrandNewNTwoDbModes. Neuters: set the floor without
// the MinWalNumberToKeep condition, or let HeadBounded hold whenever no
// member attests; both open.
TEST_P(DBWalPartitionCoverageTest, GC23FloorOffForLaterGenerationHeads) {
  if (GetParam()) {
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = IndexedWalOptions(CurrentOptions());
  for (const bool by_reopen : {false, true}) {
    SCOPED_TRACE(by_reopen ? "second generation by reopen"
                           : "second generation by rotation");
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::StartWALFile:WriteSpan",
        [](void* arg) { *static_cast<bool*>(arg) = false; });
    SyncPoint::GetInstance()->EnableProcessing();
    CreateBrandNewPartitionedDb(kPartitions, options);
    LoggedPut(cf_for_partition_[0], "a1");
    LoggedPut(cf_for_partition_[1], "a2");
    const uint64_t first_generation_p1 = writes_[IndexOf("a2")].wal;
    if (by_reopen) {
      const std::vector<LoggedWrite> first = writes_;
      std::vector<std::string> cf_names;
      ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
      ASSERT_OK(db_->FlushWAL(false));
      Close();
      for (const LoggedWrite& write : first) {
        ASSERT_OK(test::TruncateFile(env_, LogFileName(dbname_, write.wal),
                                     (write.begin + write.end) / 2));
      }
      Options reopen = options;
      reopen.wal_partitions = kPartitions;
      reopen.two_write_queues = GetParam();
      ASSERT_OK(TryReopenWithColumnFamilies(cf_names, reopen));
      ASSERT_EQ(0U, dbfull()->GetVersionSet()->LastSequence());
      writes_.clear();
    } else {
      Rotate();
    }
    ASSERT_NE(0U,
              dbfull()->GetVersionSet()->GetWalSet().GetMinWalNumberToKeep());
    LoggedPut(cf_for_partition_[1], "b1");
    LoggedPut(cf_for_partition_[0], "b2");
    LoggedPut(cf_for_partition_[1], "b3");
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::StartWALFile:WriteSpan");
    const LoggedWrite b1 = writes_[IndexOf("b1")];
    ASSERT_GT(b1.wal, first_generation_p1);
    ASSERT_EQ(b1.wal, writes_[IndexOf("b3")].wal);
    ASSERT_OK(db_->FlushWAL(false));
    const std::string image = dbname_ + "_image";
    CopyDBAsCrashImage(image);
    Close();
    CorruptImageRecord(image, b1);
    SCOPED_TRACE("damaged second-generation WAL " + std::to_string(b1.wal));
    const std::string copy = image + "_open";
    for (const WALRecoveryMode mode :
         {kTolerate, kAbsolute, kPointInTime, kSkip}) {
      SCOPED_TRACE(static_cast<int>(mode));
      CopyDir(image, copy);
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> db;
      const Status s =
          OpenCopy(copy, kPartitions, options, mode, &handles, &db);
      for (ColumnFamilyHandle* handle : handles) {
        EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
      }
      ASSERT_TRUE(s.IsCorruption()) << s.ToString();
      EXPECT_NE(std::string::npos,
                s.ToString().find("WAL " + std::to_string(b1.wal) +
                                  " head damaged and unattested"))
          << s.ToString();
    }
  }
}

// GC24 Q7: GC23's recovery flush for a skipped superseded-only index, at
// N in {2, 8} (SkippedSupersededHoleSurvivesLaterWrites runs at N = 1). A
// filter stop after two records cuts k2-k4; the write after it, on another
// partition, reissues only the first cut index, leaving the next two
// superseded-only. The next open skips them as trailing and must flush; its
// write, on a third partition where N allows, lands above them. Every mode
// then opens with k0, k1, a and b, and without k2-k4.
// Neuter: drop the flush (`flushed = true` for SkippedUnexplainedDiscard in
// ProcessLogFilesMerged): the last open reads b as a live record above a
// lost reissue (point-in-time loses b; tolerate and absolute fail with an
// unexplained hole).
TEST_P(DBWalPartitionCoverageTest,
       GC24SkippedSupersededHoleFlushesAtNAboveOne) {
  if (GetParam()) {
    return;
  }
  class StopAfterRecords : public WalFilter {
   public:
    const char* Name() const override { return "StopAfterRecords"; }
    WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                       const std::string& /*log_file_name*/,
                                       const WriteBatch& /*batch*/,
                                       WriteBatch* /*new_batch*/,
                                       bool* /*batch_changed*/) override {
      return records_found_++ < 2 ? WalProcessingOption::kContinueProcessing
                                  : WalProcessingOption::kStopReplay;
    }

   private:
    int records_found_ = 0;
  };
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    for (const WALRecoveryMode mode :
         {kPointInTime, kTolerate, kAbsolute, kSkip}) {
      SCOPED_TRACE("N=" + std::to_string(partitions) + " mode " +
                   std::to_string(static_cast<int>(mode)));
      Options options = IndexedWalOptions(CurrentOptions());
      options.wal_recovery_mode = mode;
      DestroyAndReopen(options);
      OpenWithPartitions(partitions, options);
      const int cf_k = cf_for_partition_[0];
      const int cf_a = cf_for_partition_[1];
      const int cf_b = cf_for_partition_[partitions - 1];
      for (int i = 0; i < 5; ++i) {
        const std::string key = "k" + std::to_string(i);
        ASSERT_OK(Put(cf_k, key, ValueOf(key)));
      }
      StopAfterRecords stop;
      options.wal_filter = &stop;
      ReopenWithPartitions(partitions, options);
      options.wal_filter = nullptr;
      ASSERT_EQ("NOT_FOUND", Get(cf_k, "k2"));
      ASSERT_OK(Put(cf_a, "a", ValueOf("a")));
      ReopenWithPartitions(partitions, options);
      EXPECT_EQ(ValueOf("a"), Get(cf_a, "a"));
      ASSERT_OK(Put(cf_b, "b", ValueOf("b")));
      ReopenWithPartitions(partitions, options);
      for (int i = 0; i < 5; ++i) {
        const std::string key = "k" + std::to_string(i);
        EXPECT_EQ(i < 2 ? ValueOf(key) : "NOT_FOUND", Get(cf_k, key)) << key;
      }
      EXPECT_EQ(ValueOf("a"), Get(cf_a, "a"));
      EXPECT_EQ(ValueOf("b"), Get(cf_b, "b"));
    }
  }
}

// Offsets of the physical records in the first block of a non-recyclable
// WAL's contents, with their types.
static std::vector<std::pair<size_t, uint8_t>> FirstBlockRecords(
    const std::string& contents) {
  std::vector<std::pair<size_t, uint8_t>> records;
  size_t offset = 0;
  while (offset + log::kHeaderSize <= contents.size() &&
         offset + log::kHeaderSize <= static_cast<size_t>(log::kBlockSize)) {
    const uint8_t type = static_cast<uint8_t>(contents[offset + 6]);
    if (type == log::kZeroType) {
      break;
    }
    EXPECT_FALSE(log::IsRecyclableRecordType(type));
    const size_t length =
        static_cast<uint8_t>(contents[offset + 4]) |
        (static_cast<size_t>(static_cast<uint8_t>(contents[offset + 5])) << 8);
    records.emplace_back(offset, type);
    offset += log::kHeaderSize + length;
  }
  return records;
}

// GC23 Q4: the void exception to a pending superseded-only index, through DB
// opens at N > 1. Within N > 1 histories every cut sits at or above the seed,
// which is above every surviving void, so the pre-cut case arises from an
// N = 1 cut carried into an upgrade: keys k0-k4, then a burned index whose
// void ends the WAL; a filter stop after k1 cuts k2-k4, the open seeds past
// the void, and its writes n0-n3 take indices above it, leaving k2-k4's
// indices superseded-only. The DB is then reopened at N and written in every
// partition (p*).
// - Pre-cut void (as written): the void sits in a WAL numbered below the
//   declaring WAL, so it clears the pending index and every mode opens with
//   k0, k1, n* and p*.
// - Post-cut void: the same void moved into the declaring WAL, or into the
//   highest partition WAL the upgrade created, ahead of its first record. It
//   explains nothing below it, so the pending index is the frontier: a
//   point-in-time read-only open holds only k0 and k1; read-write opens fail
//   (point-in-time on an index above the cut a write would reissue, since
//   the end-of-recovery check counts only this open's cuts, naming the
//   pending index since GC24 Q6; tolerate and absolute on the unexplained
//   hole); skip re-trusts the declaration and opens.
// No write path burns an index after a cut while leaving the reissue of a
// lower superseded index absent, so the post-cut layout is crafted.
// Neuters: clear the pending index on any void (the post-cut cases open with
// n* and p*); never clear it (the pre-cut case loses n* and p*).
TEST_P(DBWalPartitionCoverageTest, GC23VoidAgainstPendingSupersededIndex) {
  if (GetParam()) {
    return;
  }
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    ReopenWithPartitions(1, options);
    const int cf = cf_for_partition_[0];
    std::vector<std::pair<int, std::string>> kept;
    std::vector<std::pair<int, std::string>> cut;
    std::vector<std::pair<int, std::string>> above;
    for (int i = 0; i < 5; ++i) {
      const std::string key = "k" + std::to_string(i);
      ASSERT_OK(Put(cf, key, ValueOf(key)));
      (i < 2 ? kept : cut).emplace_back(cf, key);
    }
    const uint64_t cut_wal = dbfull()->TEST_CurrentWalNumber(0);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [](void* arg) {
          Slice* const log_entry = static_cast<Slice*>(arg);
          char* const data = const_cast<char*>(log_entry->data());
          data[log_entry->size() - 1] ^= 0xff;
        });
    SyncPoint::GetInstance()->EnableProcessing();
    WriteBatch burned(/*reserved_bytes=*/0, /*max_bytes=*/0,
                      /*protection_bytes_per_key=*/8, /*default_cf_ts_sz=*/0);
    ASSERT_OK(burned.Put(handles_[cf], "burned", "v"));
    ASSERT_NOK(db_->Write(WriteOptions(), &burned));
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    Close();

    class StopAfterRecords : public WalFilter {
     public:
      const char* Name() const override { return "StopAfterRecords"; }
      WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                         const std::string& /*log_file_name*/,
                                         const WriteBatch& /*batch*/,
                                         WriteBatch* /*new_batch*/,
                                         bool* /*batch_changed*/) override {
        return records_found_++ < 2 ? WalProcessingOption::kContinueProcessing
                                    : WalProcessingOption::kStopReplay;
      }

     private:
      int records_found_ = 0;
    } stop;
    options.wal_filter = &stop;
    ReopenWithPartitions(1, options);
    options.wal_filter = nullptr;
    const uint64_t declarer = dbfull()->TEST_CurrentWalNumber(0);
    ASSERT_GT(declarer, cut_wal);
    for (int i = 0; i < 4; ++i) {
      const std::string key = "n" + std::to_string(i);
      ASSERT_OK(Put(cf, key, ValueOf(key)));
      above.emplace_back(cf, key);
    }
    Close();
    ReopenWithPartitions(num_partitions, options);
    for (size_t p = 0; p < num_partitions; ++p) {
      const std::string key = "p" + std::to_string(p);
      ASSERT_OK(Put(cf_for_partition_[p], key, ValueOf(key)));
      above.emplace_back(cf_for_partition_[p], key);
    }
    const uint64_t later_wal =
        dbfull()->TEST_CurrentWalNumber(num_partitions - 1);
    ASSERT_GT(later_wal, declarer);
    const std::string image = dbname_ + "_image";
    CopyDBAsCrashImage(image);
    Close();

    // Opens a copy of `dir` and checks which keys it holds.
    auto assert_open = [&](const std::string& dir, WALRecoveryMode mode,
                           bool read_only, bool expect_above) {
      std::vector<std::string> cf_names;
      ASSERT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
      std::vector<ColumnFamilyDescriptor> descriptors;
      for (const std::string& name : cf_names) {
        descriptors.emplace_back(name, options);
      }
      Options open_options = options;
      open_options.create_if_missing = false;
      open_options.wal_partitions = static_cast<uint32_t>(num_partitions);
      open_options.wal_recovery_mode = mode;
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> db;
      const Status s = read_only
                           ? DB::OpenForReadOnly(DBOptions(open_options), dir,
                                                 descriptors, &handles, &db)
                           : DB::Open(DBOptions(open_options), dir, descriptors,
                                      &handles, &db);
      ASSERT_OK(s);
      auto get = [&](const std::pair<int, std::string>& key) {
        std::string value;
        const Status g =
            db->Get(ReadOptions(), handles[key.first], key.second, &value);
        return g.ok() ? value : g.ToString();
      };
      for (const auto& key : kept) {
        EXPECT_EQ(ValueOf(key.second), get(key)) << key.second;
      }
      for (const auto& key : cut) {
        EXPECT_EQ("NotFound: ", get(key)) << key.second;
      }
      for (const auto& key : above) {
        EXPECT_EQ(expect_above ? ValueOf(key.second) : "NotFound: ", get(key))
            << key.second;
      }
      for (ColumnFamilyHandle* handle : handles) {
        EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
      }
    };
    auto open_status = [&](const std::string& dir, WALRecoveryMode mode) {
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> db;
      const Status s =
          OpenCopy(dir, num_partitions, options, mode, &handles, &db);
      for (ColumnFamilyHandle* handle : handles) {
        EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
      }
      return s;
    };

    const std::string copy = image + "_open";
    {
      SCOPED_TRACE("pre-cut void");
      for (const WALRecoveryMode mode :
           {kTolerate, kAbsolute, kPointInTime, kSkip}) {
        SCOPED_TRACE(static_cast<int>(mode));
        CopyDir(image, copy);
        assert_open(copy, mode, /*read_only=*/false, /*expect_above=*/true);
      }
    }

    // Takes the void out of the cut WAL and puts it ahead of `target`'s first
    // record that carries a wal_index.
    const std::string moved = image + "_moved";
    auto move_void = [&](uint64_t target) {
      CopyDir(image, moved);
      std::string source;
      ASSERT_OK(ReadFileToString(env_, LogFileName(moved, cut_wal), &source));
      const auto source_records = FirstBlockRecords(source);
      ASSERT_FALSE(source_records.empty());
      const size_t void_offset = source_records.back().first;
      ASSERT_EQ(log::kWALIndexVoidType, source_records.back().second);
      const std::string void_record = source.substr(void_offset);
      ASSERT_EQ(log::kHeaderSize + log::kWALIndexVoidPayloadSize,
                void_record.size());
      ASSERT_OK(WriteStringToFile(env_, source.substr(0, void_offset),
                                  LogFileName(moved, cut_wal),
                                  /*should_sync=*/true));
      std::string contents;
      ASSERT_OK(ReadFileToString(env_, LogFileName(moved, target), &contents));
      ASSERT_LT(contents.size() + void_record.size(),
                static_cast<size_t>(log::kBlockSize));
      size_t insert_at = contents.size();
      for (const auto& [offset, type] : FirstBlockRecords(contents)) {
        if (log::IsWALIndexRecordType(type)) {
          insert_at = offset;
          break;
        }
      }
      ASSERT_LT(insert_at, contents.size());
      contents.insert(insert_at, void_record);
      ASSERT_OK(WriteStringToFile(env_, contents, LogFileName(moved, target),
                                  /*should_sync=*/true));
    };
    for (const uint64_t target : {declarer, later_wal}) {
      SCOPED_TRACE("post-cut void in WAL " + std::to_string(target) +
                   (target == declarer ? " (declarer)" : " (later)"));
      ASSERT_NO_FATAL_FAILURE(move_void(target));
      CopyDir(moved, copy);
      assert_open(copy, kPointInTime, /*read_only=*/true,
                  /*expect_above=*/false);
      for (const WALRecoveryMode mode : {kTolerate, kAbsolute, kPointInTime}) {
        SCOPED_TRACE(static_cast<int>(mode));
        CopyDir(moved, copy);
        const Status s = open_status(copy, mode);
        EXPECT_TRUE(s.IsCorruption()) << s.ToString();
        EXPECT_NE(
            std::string::npos,
            s.ToString().find(mode == kPointInTime ? "a write would reissue it"
                                                   : "unexplained hole"))
            << s.ToString();
        if (mode == kPointInTime) {
          // GC24 Q6: the read-write open fails closed naming the pending
          // index, k2's. Neuter: drop the clause (as before).
          EXPECT_NE(std::string::npos,
                    s.ToString().find(
                        "stopped at superseded-only wal_index " +
                        std::to_string(WalIndexOfKey(env_, moved, "k2"))))
              << s.ToString();
        }
      }
      CopyDir(moved, copy);
      assert_open(copy, kSkip, /*read_only=*/false, /*expect_above=*/true);
    }
  }
}

// GC26 Q2 and Q3: a WritePrepared prepare in partition 1 pins every WAL from
// its own onward (MinLogNumberToKeep under allow_2pc) while the DB reopens at
// another wal_partitions, or with partition_wal_usage kNone, and a flush
// there retires the old partition 0 WAL below the prepare's.
class DBWalPrepareAcrossChangeTest : public DBWalPartitionCoverageTest {
 public:
  // Opens the DB at `dir`, with every column family it has, as a
  // WritePrepared TransactionDB at `num_partitions` partitions writing
  // `usage`, in `mode`.
  Status OpenPreparedCopy(const std::string& dir, size_t num_partitions,
                          PartitionWALUsage usage, WALRecoveryMode mode,
                          std::vector<ColumnFamilyHandle*>* handles,
                          std::unique_ptr<TransactionDB>* txn_db) {
    Options options = IndexedWalOptions(CurrentOptions());
    // No compaction deletes files while a crash image is copied.
    options.disable_auto_compactions = true;
    std::vector<std::string> cf_names;
    Status s = DB::ListColumnFamilies(options, dir, &cf_names);
    if (!s.ok()) {
      return s;
    }
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.create_if_missing = false;
    options.wal_partitions = static_cast<uint32_t>(num_partitions);
    options.partition_wal_usage = usage;
    options.two_write_queues = false;
    options.wal_recovery_mode = mode;
    TransactionDBOptions txn_db_options;
    txn_db_options.write_policy = TxnDBWritePolicy::WRITE_PREPARED;
    TransactionDB* opened = nullptr;
    s = TransactionDB::Open(options, txn_db_options, dir, column_families,
                            handles, &opened);
    txn_db->reset(opened);
    return s;
  }

  static void CloseTxnDb(std::vector<ColumnFamilyHandle*>* handles,
                         std::unique_ptr<TransactionDB>* txn_db) {
    for (ColumnFamilyHandle* handle : *handles) {
      ASSERT_OK((*txn_db)->DestroyColumnFamilyHandle(handle));
    }
    handles->clear();
    txn_db->reset();
  }

  // At `from` partitions: synced writes to every column family, then the
  // prepare "xid" in partition 1, then more writes, and a crash image of that.
  // The image reopens at `to` partitions writing `to_usage`, flushes every
  // column family, purges obsolete WALs, which deletes the old partition 0
  // WAL unless the opener writes kNone, and takes synced writes to every
  // column family. `image` is the crash image taken then. Every write is
  // logged in acknowledged_.
  void ChangeWithPrepareOutstanding(size_t from, size_t to,
                                    PartitionWALUsage to_usage,
                                    const std::string& image) {
    Options options = IndexedWalOptions(CurrentOptions());
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(from, options);
    Close();
    acknowledged_.clear();
    WriteOptions sync_wo;
    sync_wo.sync = true;
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<TransactionDB> db;
    auto put = [&](int cf, const std::string& key) {
      ASSERT_OK(db->Put(sync_wo, handles[cf], key, "v"));
      acknowledged_.emplace_back(cf, key);
    };
    const std::string before = image + "_before";
    uint64_t old_p0_wal = 0;
    {
      ASSERT_OK(OpenPreparedCopy(dbname_, from,
                                 PartitionWALUsage::kPartitionByColumnFamily,
                                 options.wal_recovery_mode, &handles, &db));
      DBImpl* impl = static_cast_with_check<DBImpl>(db->GetRootDB());
      ASSERT_EQ(from, impl->TEST_NumWalPartitions());
      // Every column family, so that the flush below moves every log number
      // past the old WALs.
      for (size_t cf = 0; cf < handles.size(); ++cf) {
        put(static_cast<int>(cf), "a" + std::to_string(cf));
      }
      Transaction* txn = db->BeginTransaction(sync_wo);
      ASSERT_OK(txn->SetName("xid"));
      ASSERT_OK(txn->Put(handles[cf_for_partition_[1]], "prepared", "v"));
      ASSERT_OK(txn->Prepare());
      old_p0_wal = impl->TEST_CurrentWalNumber(0);
      ASSERT_LT(old_p0_wal, txn->GetLogNumber());
      for (size_t p = 0; p < from; ++p) {
        put(cf_for_partition_[p], "b" + std::to_string(p));
      }
      CopyDBAsCrashImage(before);
      ASSERT_OK(txn->Rollback());
      delete txn;
      CloseTxnDb(&handles, &db);
    }
    CopyDir(before, dbname_);
    ASSERT_OK(OpenPreparedCopy(dbname_, to, to_usage, options.wal_recovery_mode,
                               &handles, &db));
    DBImpl* impl = static_cast_with_check<DBImpl>(db->GetRootDB());
    ASSERT_EQ(to, impl->TEST_NumWalPartitions());
    Transaction* recovered = db->GetTransactionByName("xid");
    ASSERT_NE(nullptr, recovered);
    // Two rounds: the first round's flushes may each still see another
    // column family's recovered data below the prepare's WAL.
    for (int round = 0; round < 2; ++round) {
      for (size_t cf = 0; cf < handles.size(); ++cf) {
        put(static_cast<int>(cf), "f" + std::to_string(round));
      }
      ASSERT_OK(db->Flush(FlushOptions(), handles));
    }
    ASSERT_EQ(recovered->GetLogNumber(), impl->MinLogNumberToKeep());
    // The WALs recovered from the previous instance leave only by a full
    // scan, which re-enabling deletions forces.
    ASSERT_OK(db->DisableFileDeletions());
    ASSERT_OK(db->EnableFileDeletions());
    if (to_usage == PartitionWALUsage::kNone) {
      // A kNone opener attests nothing, so the prepare pins the lowest
      // recovered WAL instead and the indexed set stays whole.
      ASSERT_LE(impl->MinLogNumberToKeep(), old_p0_wal);
      ASSERT_OK(env_->FileExists(LogFileName(dbname_, old_p0_wal)));
    } else {
      ASSERT_TRUE(
          env_->FileExists(LogFileName(dbname_, old_p0_wal)).IsNotFound())
          << "WAL " << old_p0_wal;
    }
    for (size_t cf = 0; cf < handles.size(); ++cf) {
      put(static_cast<int>(cf), "c" + std::to_string(cf));
    }
    CopyDBAsCrashImage(image);
    ASSERT_OK(recovered->Rollback());
    delete recovered;
    CloseTxnDb(&handles, &db);
  }

  // Opens a fresh copy of `image` at `num_partitions` writing `usage` in
  // every mode: every acknowledged write and the prepare are recovered, the
  // prepare commits (or rolls back, alternately), and the DB reopens with
  // that outcome.
  void AssertPrepareRecoversInEveryMode(const std::string& image,
                                        size_t num_partitions,
                                        PartitionWALUsage usage) {
    const std::string copy = image + "_open";
    bool commit = true;
    for (const WALRecoveryMode mode :
         {kTolerate, kAbsolute, kPointInTime, kSkip}) {
      SCOPED_TRACE("wal_recovery_mode=" +
                   std::to_string(static_cast<int>(mode)));
      CopyDir(image, copy);
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<TransactionDB> db;
      const Status s =
          OpenPreparedCopy(copy, num_partitions, usage, mode, &handles, &db);
      EXPECT_OK(s);
      if (!s.ok()) {
        continue;
      }
      auto get = [&](int cf, const std::string& key) {
        std::string value;
        const Status g = db->Get(ReadOptions(), handles[cf], key, &value);
        EXPECT_TRUE(g.ok() || g.IsNotFound()) << g.ToString();
        return g.ok() ? value : std::string("NOT_FOUND");
      };
      for (const auto& [cf, key] : acknowledged_) {
        EXPECT_EQ("v", get(cf, key)) << key;
      }
      const int prepared_cf = cf_for_partition_[1];
      EXPECT_EQ("NOT_FOUND", get(prepared_cf, "prepared"));
      Transaction* recovered = db->GetTransactionByName("xid");
      EXPECT_NE(nullptr, recovered);
      if (recovered != nullptr) {
        ASSERT_OK(commit ? recovered->Commit() : recovered->Rollback());
        delete recovered;
      }
      const std::string outcome = commit ? "v" : "NOT_FOUND";
      EXPECT_EQ(outcome, get(prepared_cf, "prepared"));
      CloseTxnDb(&handles, &db);
      ASSERT_OK(
          OpenPreparedCopy(copy, num_partitions, usage, mode, &handles, &db));
      EXPECT_EQ(nullptr, db->GetTransactionByName("xid"));
      EXPECT_EQ(outcome, get(prepared_cf, "prepared"));
      for (const auto& [cf, key] : acknowledged_) {
        EXPECT_EQ("v", get(cf, key)) << key;
      }
      CloseTxnDb(&handles, &db);
      commit = !commit;
    }
  }

  std::vector<std::pair<int, std::string>> acknowledged_;
};

INSTANTIATE_TEST_CASE_P(DBWalPrepareAcrossChangeTest,
                        DBWalPrepareAcrossChangeTest, testing::Values(false));

// GC26 Q2 (G3-B-1, D-GC25b): reducing wal_partitions while a prepare pins an
// old WAL. The old partition 0 WAL is deleted at the new N, and its indices
// interleave with the pinned WAL's. Open attests every recovered chain head,
// including those of partitions the new N folds into partition 0, so the
// deleted WAL's span explains those indices. Every mode opens with every
// acknowledged write and the prepare. 4 -> 4 is the control. Neuter:
// RecoveredWALIndexPredecessors attests only the newest WAL placed in each
// partition; the reduced cells fail with "unexplained hole" (point-in-time:
// "Column family inconsistency").
TEST_P(DBWalPrepareAcrossChangeTest, GC26PartitionReductionKeepsPrepare) {
  struct Change {
    size_t from;
    size_t to;
  };
  for (const Change change :
       {Change{4, 2}, Change{4, 1}, Change{8, 2}, Change{4, 4}}) {
    SCOPED_TRACE("wal_partitions " + std::to_string(change.from) + " -> " +
                 std::to_string(change.to));
    const std::string image = dbname_ + "_gc26_q2";
    ASSERT_NO_FATAL_FAILURE(ChangeWithPrepareOutstanding(
        change.from, change.to, PartitionWALUsage::kPartitionByColumnFamily,
        image));
    AssertPrepareRecoversInEveryMode(
        image, change.to, PartitionWALUsage::kPartitionByColumnFamily);
  }
}

// GC26 Q3 (G3-B-2, D-GC25c): rolling back to kNone while a prepare pins an
// old WAL leaves unindexed WALs above the partitioned indexed set. The
// unindexed suffix replays sequentially after the merge, so every mode opens
// with every acknowledged write and the prepare, which then commits or rolls
// back. Neuter: today's gate ("carries no WAL index but follows indexed WAL"
// in every mode).
TEST_P(DBWalPrepareAcrossChangeTest, GC26RollbackToNoneKeepsPrepare) {
  for (const size_t from : {size_t{2}, size_t{4}}) {
    SCOPED_TRACE("wal_partitions " + std::to_string(from) + " -> kNone");
    const std::string image = dbname_ + "_gc26_q3";
    ASSERT_NO_FATAL_FAILURE(
        ChangeWithPrepareOutstanding(from, 1, PartitionWALUsage::kNone, image));
    AssertPrepareRecoversInEveryMode(image, 1, PartitionWALUsage::kNone);
  }
}

// GC26 Q3 (Review-M3 T6): true alternation still fails closed. After the
// kNone rollback above, partitioning is re-enabled while the prepare is
// still outstanding: that open replays the unindexed suffix after the merge
// and writes indexed WALs above it. The next open finds unindexed WALs
// between indexed ones, which neither the merge nor sequential replay can
// order, and every mode fails naming both WALs. Neuter: replay any mixed
// partitioned set as merge plus suffix; the merge then refuses the inner
// unindexed WAL on its own, naming only that WAL, and the check fails.
TEST_P(DBWalPrepareAcrossChangeTest, GC26TrueAlternationFailsClosed) {
  const std::string image = dbname_ + "_gc26_t6";
  ASSERT_NO_FATAL_FAILURE(
      ChangeWithPrepareOutstanding(4, 1, PartitionWALUsage::kNone, image));
  CopyDir(image, dbname_);
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<TransactionDB> db;
  ASSERT_OK(OpenPreparedCopy(dbname_, 2,
                             PartitionWALUsage::kPartitionByColumnFamily,
                             kPointInTime, &handles, &db));
  ASSERT_NE(nullptr, db->GetTransactionByName("xid"));
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (size_t cf = 1; cf < handles.size(); ++cf) {
    ASSERT_OK(db->Put(sync_wo, handles[cf], "d" + std::to_string(cf), "v"));
  }
  const std::string alternating = dbname_ + "_gc26_t6_alternating";
  CopyDBAsCrashImage(alternating);
  CloseTxnDb(&handles, &db);
  const std::string copy = alternating + "_open";
  for (const WALRecoveryMode mode :
       {kTolerate, kAbsolute, kPointInTime, kSkip}) {
    SCOPED_TRACE("wal_recovery_mode=" + std::to_string(static_cast<int>(mode)));
    CopyDir(alternating, copy);
    const Status s =
        OpenPreparedCopy(copy, 2, PartitionWALUsage::kPartitionByColumnFamily,
                         mode, &handles, &db);
    for (ColumnFamilyHandle* handle : handles) {
      EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    handles.clear();
    db.reset();
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos,
              s.ToString().find("carries no WAL index but follows indexed WAL"))
        << s.ToString();
  }
}

// GC26 Q1 (G3-A-2, D-GC25a): an append whose bytes reach the file before
// the file system reports its error. Resume reads the burned index back from
// the WAL that burned it and explains it with a void only when no whole
// record landed.
class DBWalLandedAppendTest : public DBWalAppendFailureTest {
 public:
  // At `num_partitions` partitions, as a WriteCommitted TransactionDB: writes
  // to every partition, then (with `two_pc`) a prepare on the partition with
  // the oldest WAL, which keeps the failing WAL live for recovery. An append
  // on the partition with the newest WAL fails with `landing`; Resume()
  // succeeds, and three rounds of writes (the last synced) follow. A crash
  // image taken then opens in every mode with every acknowledged write and
  // the prepare.
  void ResumeAfterLandingThenCrash(size_t num_partitions, bool two_pc,
                                   FailingWalFS::Landing landing) {
    Options options = FailureOptions();
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(num_partitions, options);
    Close();
    auto open_txn_db = [&](const std::string& dir,
                           std::vector<ColumnFamilyHandle*>* handles,
                           Status* open_status) {
      std::vector<std::string> cf_names;
      EXPECT_OK(DB::ListColumnFamilies(options, dir, &cf_names));
      std::vector<ColumnFamilyDescriptor> column_families;
      for (const std::string& name : cf_names) {
        column_families.emplace_back(name, options);
      }
      Options open_options = options;
      open_options.wal_partitions = static_cast<uint32_t>(num_partitions);
      open_options.two_write_queues = false;
      TransactionDBOptions txn_db_options;
      txn_db_options.write_policy = TxnDBWritePolicy::WRITE_COMMITTED;
      TransactionDB* txn_db = nullptr;
      *open_status = TransactionDB::Open(open_options, txn_db_options, dir,
                                         column_families, handles, &txn_db);
      return std::unique_ptr<TransactionDB>(txn_db);
    };
    std::vector<ColumnFamilyHandle*> handles;
    Status open_status;
    std::unique_ptr<TransactionDB> db =
        open_txn_db(dbname_, &handles, &open_status);
    ASSERT_OK(open_status);
    ASSERT_NE(nullptr, db);
    DBImpl* db_impl = static_cast_with_check<DBImpl>(db->GetRootDB());
    ASSERT_EQ(num_partitions, db_impl->TEST_NumWalPartitions());
    auto close = [&] {
      for (ColumnFamilyHandle* handle : handles) {
        ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
      }
      handles.clear();
      db.reset();
    };

    std::vector<std::pair<size_t, std::string>> acknowledged;
    for (size_t p = 0; p < num_partitions; ++p) {
      const std::string key = "before" + std::to_string(p);
      ASSERT_OK(
          db->Put(WriteOptions(), handles[cf_for_partition_[p]], key, "v"));
      acknowledged.emplace_back(p, key);
    }
    size_t oldest = 0;
    size_t newest = 0;
    for (size_t p = 1; p < num_partitions; ++p) {
      if (db_impl->TEST_CurrentWalNumber(p) <
          db_impl->TEST_CurrentWalNumber(oldest)) {
        oldest = p;
      }
      if (db_impl->TEST_CurrentWalNumber(p) >
          db_impl->TEST_CurrentWalNumber(newest)) {
        newest = p;
      }
    }
    const uint64_t failed_wal = db_impl->TEST_CurrentWalNumber(newest);
    Transaction* txn = nullptr;
    if (two_pc) {
      txn = db->BeginTransaction(WriteOptions());
      ASSERT_OK(txn->SetName("xid"));
      ASSERT_OK(txn->Put(handles[cf_for_partition_[oldest]], "prepared", "v"));
      ASSERT_OK(txn->Prepare());
      ASSERT_LT(txn->GetLogNumber(), failed_wal);
    }

    fail_fs_->Arm(failed_wal, Retryable("injected"), landing);
    ASSERT_NOK(db->Put(WriteOptions(), handles[cf_for_partition_[newest]],
                       "failed", "v"));
    fail_fs_->Disarm();
    const Status bg = db_impl->TEST_GetBGError();
    ASSERT_EQ(Status::Severity::kHardError, bg.severity()) << bg.ToString();
    ASSERT_OK(db->Resume());
    ASSERT_OK(db_impl->TEST_GetBGError());

    WriteOptions sync_wo;
    sync_wo.sync = true;
    for (int round = 0; round < 3; ++round) {
      for (size_t p = 0; p < num_partitions; ++p) {
        const std::string key =
            "after" + std::to_string(round) + "_p" + std::to_string(p);
        ASSERT_OK(db->Put(round == 2 ? sync_wo : WriteOptions(),
                          handles[cf_for_partition_[p]], key, "v"));
        acknowledged.emplace_back(p, key);
      }
    }
    // The cover on the poisoned writer failed, and Resume told a whole record
    // from a torn or absent one. Without a prepare, Resume's flush retired
    // the failed WAL first, and nothing in it can contradict a void.
    std::string info_log;
    ASSERT_OK(ReadFileToString(env_, dbname_ + "/LOG", &info_log));
    EXPECT_NE(std::string::npos,
              info_log.find("Failed to cover burned WAL index"));
    EXPECT_EQ(two_pc, env_->FileExists(LogFileName(dbname_, failed_wal)).ok());
    EXPECT_EQ(two_pc && landing == FailingWalFS::Landing::kWhole,
              info_log.find("landed whole in WAL #" +
                            std::to_string(failed_wal)) != std::string::npos);
    EXPECT_EQ(two_pc && landing == FailingWalFS::Landing::kTorn,
              info_log.find("landed torn in WAL #" +
                            std::to_string(failed_wal)) != std::string::npos);

    ASSERT_OK(db->DisableFileDeletions());
    const std::vector<WALRecoveryMode> recovery_modes = {
        WALRecoveryMode::kPointInTimeRecovery,
        WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kSkipAnyCorruptedRecords};
    for (const WALRecoveryMode recovery_mode : recovery_modes) {
      CopyDBAsCrashImage(dbname_ + "_crash" +
                         std::to_string(static_cast<int>(recovery_mode)));
    }
    ASSERT_OK(db->EnableFileDeletions());
    if (txn != nullptr) {
      ASSERT_OK(txn->Rollback());
      delete txn;
    }
    close();

    for (const WALRecoveryMode recovery_mode : recovery_modes) {
      SCOPED_TRACE("wal_recovery_mode=" +
                   std::to_string(static_cast<int>(recovery_mode)));
      options.wal_recovery_mode = recovery_mode;
      const std::string crash_dir =
          dbname_ + "_crash" + std::to_string(static_cast<int>(recovery_mode));
      db = open_txn_db(crash_dir, &handles, &open_status);
      EXPECT_OK(open_status);
      if (!open_status.ok() || db == nullptr) {
        handles.clear();
        db.reset();
        continue;
      }
      if (two_pc) {
        Transaction* recovered = db->GetTransactionByName("xid");
        EXPECT_NE(nullptr, recovered);
        if (recovered != nullptr) {
          ASSERT_OK(recovered->Rollback());
          delete recovered;
        }
      }
      for (const auto& [p, key] : acknowledged) {
        std::string value;
        EXPECT_OK(
            db->Get(ReadOptions(), handles[cf_for_partition_[p]], key, &value))
            << key;
      }
      // A landed record is replayed, but Resume's flush moved its column
      // family past the failed WAL, so it applies nothing either way.
      std::string value;
      const Status failed = db->Get(
          ReadOptions(), handles[cf_for_partition_[newest]], "failed", &value);
      EXPECT_TRUE(failed.IsNotFound()) << failed.ToString();
      close();
    }
  }
};

INSTANTIATE_TEST_CASE_P(DBWalLandedAppendTest, DBWalLandedAppendTest,
                        testing::Values(false));

// GC26 Q1: the record of the failed append landed whole, landed torn, or did
// not land, at N in {2, 8}, with and without a prepare keeping its WAL live.
// Every crash image after Resume opens in all four modes with every
// acknowledged write. Neuter: ExplainBurnedWALIndices voids every burned
// index (ignore `landed`); the whole-record cells with 2PC fail to open in
// every mode with "index N bears records and a void cover".
TEST_P(DBWalLandedAppendTest, GC26ResumeExplainsOnlyWhatDidNotLand) {
  using Landing = FailingWalFS::Landing;
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    for (const bool two_pc : {true, false}) {
      for (const Landing landing :
           {Landing::kWhole, Landing::kTorn, Landing::kNothing}) {
        SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions) +
                     (two_pc ? " 2PC" : "") +
                     " landing=" + std::to_string(static_cast<int>(landing)));
        ASSERT_NO_FATAL_FAILURE(
            ResumeAfterLandingThenCrash(num_partitions, two_pc, landing));
      }
    }
  }
}

// GC26 Q1: Resume cannot tell whether the burned record landed (reading its
// WAL back fails), so it explains nothing, fails, and leaves the DB stopped.
// A later Resume reads it back and succeeds. Neuter: ignore the read-back
// error in BurnedWALIndexLanded; the first Resume succeeds.
TEST_P(DBWalLandedAppendTest, GC26ResumeFailsWhenBurnedIndexReadBackFails) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_fail = cf_for_partition_[1];
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"),
                FailingWalFS::Landing::kWhole);
  ASSERT_NOK(Put(cf_fail, "failed", "value"));
  fail_fs_->Disarm();
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::BurnedWALIndexLanded:Open", [](void* arg) {
        *static_cast<Status*>(arg) = Status::IOError("injected read-back");
      });
  const Status first = dbfull()->Resume();
  EXPECT_TRUE(first.IsIOError()) << first.ToString();
  EXPECT_NE(std::string::npos, first.ToString().find("injected read-back"));
  EXPECT_FALSE(dbfull()->TEST_GetBGError().ok());
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_NOK(Put(cf_for_partition_[p], "above", "value")) << p;
  }
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::BurnedWALIndexLanded:Open");
  ASSERT_OK(dbfull()->Resume());
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "above", "value"));
  }
  for (const WALRecoveryMode mode :
       {WALRecoveryMode::kAbsoluteConsistency,
        WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kPointInTimeRecovery,
        WALRecoveryMode::kSkipAnyCorruptedRecords}) {
    SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
    options.wal_recovery_mode = mode;
    ReopenWithPartitions(2, options);
    for (size_t p = 0; p < 2; ++p) {
      EXPECT_EQ("value", Get(cf_for_partition_[p], "above"));
    }
  }
}

// GC26 Q4 (D-10): one lost record in an indexed set, with records above it
// in later WALs, at N in {1, 2}. Partition 0's middle WAL holds only t0,
// which the image tears mid-record ("torn") or drops at its record's start
// ("boundary"); it bore no positioned record, so no tail explains the hole.
// Skip skips the hole and applies every record above it, as N = 1 skip
// skips what it cannot read. Tolerate and absolute fail; point-in-time
// opens the prefix below t0. Neuter: drop SetSkipUnexplainedHoles in
// ProcessLogFilesMerged; skip fails with "unexplained hole".
TEST_P(DBWalPartitionCoverageTest, GC26SkipModeSkipsUnexplainedHole) {
  for (const size_t num_partitions : {size_t{1}, size_t{2}}) {
    for (const bool torn : {true, false}) {
      SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions) +
                   (torn ? " torn" : " boundary"));
      const Options options = IndexedWalOptions(CurrentOptions());
      DestroyAndReopen(options);
      writes_.clear();
      OpenWithPartitions(num_partitions, options);
      ASSERT_TRUE(
          log::WALIndexEnabled(dbfull()->GetDBOptions().partition_wal_usage));
      LoggedRounds(cf_for_partition_, 2, "r");
      Rotate();
      LoggedPut(cf_for_partition_[0], "t0");
      Rotate();
      LoggedRounds(cf_for_partition_, 1, "u");
      const std::string image = dbname_ + "_gc26_q4";
      CopyDBAsCrashImage(image);
      Close();
      const LoggedWrite& t0 = writes_[IndexOf("t0")];
      TruncateImageWal(image, t0.wal,
                       torn ? (t0.begin + t0.end) / 2 : t0.begin);

      const size_t below_t0 = IndexOf("t0");
      AssertCells(image, num_partitions, options,
                  {{kTolerate, Status::kCorruption, 0,
                    "a hole with records above it is not a tail"},
                   {kAbsolute, Status::kCorruption, 0,
                    "a hole with records above it is not a tail"},
                   {kPointInTime, Status::kOk, below_t0, "stops at the hole"}},
                  {});
      const std::string copy = image + "_open";
      CopyDir(image, copy);
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> db;
      ASSERT_OK(OpenCopy(copy, num_partitions, options, kSkip, &handles, &db));
      for (const LoggedWrite& write : writes_) {
        std::string value;
        const Status get =
            db->Get(ReadOptions(), handles[write.cf], write.key, &value);
        if (write.key == "t0") {
          EXPECT_TRUE(get.IsNotFound()) << get.ToString();
        } else {
          EXPECT_OK(get) << write.key;
        }
      }
      for (ColumnFamilyHandle* handle : handles) {
        EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
      }
    }
  }
}

// GC26 Q4 (D-10) guard: skip mode never applies a superseded record. A
// point-in-time open stops at a lost t0 and declares the WALs holding
// records above it superseded; its writes n0 (partition 0) and n1
// (partition 1) reissue those indices. n0 is lost at its record's start,
// and n1 sits beside the superseded copy at the same index. The skip open
// applies n1 and none of the superseded records. The supersession explains
// the lost index, so no hole is skipped here; the test passes at the base
// and with SetSkipUnexplainedHoles dropped, and only pins that the new skip
// path keeps superseded copies out.
TEST_P(DBWalPartitionCoverageTest, GC26SkipModeNeverAppliesSuperseded) {
  const Options options = IndexedWalOptions(CurrentOptions());
  DestroyAndReopen(options);
  writes_.clear();
  OpenWithPartitions(2, options);
  LoggedRounds(cf_for_partition_, 2, "r");
  Rotate();
  LoggedPut(cf_for_partition_[0], "t0");
  Rotate();
  LoggedRounds(cf_for_partition_, 1, "u");
  const std::string image1 = dbname_ + "_gc26_q4s1";
  CopyDBAsCrashImage(image1);
  Close();
  const LoggedWrite& t0 = writes_[IndexOf("t0")];
  TruncateImageWal(image1, t0.wal, t0.begin);

  const std::string image2 = dbname_ + "_gc26_q4s2";
  uint64_t n0_wal = 0;
  uint64_t n0_begin = 0;
  {
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    ASSERT_OK(OpenCopy(image1, 2, options, kPointInTime, &handles, &db));
    DBImpl* impl = static_cast_with_check<DBImpl>(db.get());
    n0_wal = impl->TEST_CurrentWalNumber(0);
    ASSERT_OK(env_->GetFileSize(LogFileName(image1, n0_wal), &n0_begin));
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_for_partition_[0]], "n0",
                      ValueOf("n0")));
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_for_partition_[1]], "n1",
                      ValueOf("n1")));
    ASSERT_OK(db->Put(WriteOptions(), handles[cf_for_partition_[0]], "n2",
                      ValueOf("n2")));
    CopyDir(image1, image2);
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
  }
  // n1 shares its index with a superseded u record.
  const uint64_t n1_index = WalIndexOfKey(env_, image2, "n1");
  ASSERT_NE(0U, n1_index);
  bool superseded_copy = false;
  for (const LoggedWrite& write : writes_) {
    for (const auto& record : IndexedRecordsIn(env_, image2, write.wal)) {
      superseded_copy =
          superseded_copy || (record.first == n1_index && write.wal != t0.wal);
    }
  }
  ASSERT_TRUE(superseded_copy) << "n1 at wal_index " << n1_index;
  // n0, and n2 behind it in the same WAL, lost at n0's record's start.
  TruncateImageWal(image2, n0_wal, n0_begin);

  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> db;
  ASSERT_OK(OpenCopy(image2, 2, options, kSkip, &handles, &db));
  auto get = [&](int cf, const std::string& key) {
    std::string value;
    const Status g = db->Get(ReadOptions(), handles[cf], key, &value);
    EXPECT_TRUE(g.ok() || g.IsNotFound()) << g.ToString();
    return g.ok() ? value : std::string("NOT_FOUND");
  };
  for (size_t i = 0; i < IndexOf("t0"); ++i) {
    EXPECT_EQ(ValueOf(writes_[i].key), get(writes_[i].cf, writes_[i].key))
        << writes_[i].key;
  }
  EXPECT_EQ("NOT_FOUND", get(cf_for_partition_[0], "t0"));
  for (size_t i = IndexOf("t0") + 1; i < writes_.size(); ++i) {
    EXPECT_EQ("NOT_FOUND", get(writes_[i].cf, writes_[i].key))
        << writes_[i].key << " is superseded";
  }
  EXPECT_EQ("NOT_FOUND", get(cf_for_partition_[0], "n0"));
  EXPECT_EQ(ValueOf("n1"), get(cf_for_partition_[1], "n1"));
  EXPECT_EQ("NOT_FOUND", get(cf_for_partition_[0], "n2"));
  for (ColumnFamilyHandle* handle : handles) {
    EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
  }
}

// Ported from GC24's closed loser (Muse, PR 21) for GC26 Q5 (b).
TEST_P(DBWalAppendFailureTest,
       GC24RetryFlushRecoveryWithBurnedIndexProgresses) {
  // GC24 Q5: a burned index is pending while recovery is forced onto the
  // retry-flush path (the Context callback overrides the recorded reason;
  // everything else about the Resume is real). The retry path rotates no
  // WALs and switches no memtables, so without the override the explanation
  // finds partition 0's WAL below the burning one and fails closed; with it
  // Resume takes the full path, rotates, explains, and recovers. Forcing is
  // required: via real faults retry-with-burns is unreachable (a burn always
  // rides a hard WAL error, which wins the recorded reason, and flushes are
  // blocked while hard-stopped), so the forced reason simulates the argued
  // overlap.
  // Neuter: drop the BurnedWalIndicesPending override. The forced Resume
  // fails with Corruption ("no healthy WAL") and never recovers.
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_fail = cf_for_partition_[1];
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
  ASSERT_NOK(Put(cf_fail, "failed", "value"));
  fail_fs_->Disarm();
  SyncPoint::GetInstance()->SetCallBack(
      "ErrorHandler::RecoverFromBGError:Context", [&](void* arg) {
        static_cast<DBRecoverContext*>(arg)->flush_reason =
            FlushReason::kErrorRecoveryRetryFlush;
      });
  ASSERT_OK(dbfull()->Resume());
  SyncPoint::GetInstance()->ClearCallBack(
      "ErrorHandler::RecoverFromBGError:Context");
  for (size_t p = 0; p < 2; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "above", "value"));
  }
  ReopenWithPartitions(2, options);
  for (size_t p = 0; p < 2; ++p) {
    const int cf = cf_for_partition_[p];
    EXPECT_EQ("value", Get(cf, "before" + std::to_string(p)));
    EXPECT_EQ("value", Get(cf, "above"));
  }
  EXPECT_EQ("NOT_FOUND", Get(cf_fail, "failed"));
}

// Ported from GC24's closed loser (Muse, PR 21) for GC26 Q5 (a).
// GC24 Q8: HeadBounded's scope on an N=1 to N=2 upgrade. An empty DB opened
// at N=1 (unindexed WAL) is reopened at N=2; a sequence-less marker lands
// the first record in partition 0's WAL, and the image tears it. The
// upgrade always retires the N=1 WAL, and WAL tracking is mandatory at
// N > 1, so MinWalNumberToKeep is nonzero and the history floor is
// definitionally off on upgrade: the torn head is bounded only by
// attestation or anchoring. With spans on, partition 0's WAL attests the
// N=1 WAL and anchors itself, so the DB opens (absolute still refuses the
// torn tail, with its ordinary tail error, not "head damaged"); with spans
// off, nothing bounds the head and every mode fails closed with "head
// damaged and unattested", whether or not a sequence was persisted. Neuter:
// HeadBounded holds whenever no member attests a predecessor; the
// spans-off layouts open.
TEST_P(DBWalPartitionCoverageTest, GC24HistoryFloorScopeOnUpgrade) {
  if (GetParam()) {
    return;
  }
  // Column families covering both N=2 partitions.
  std::vector<std::string> cf_names = {kDefaultColumnFamilyName};
  std::vector<int> cf_for_partition(2, -1);
  size_t covered = 0;
  for (uint32_t cf_id = 1; covered < 2; ++cf_id) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, cf_id, "key", "value"));
    size_t partition = 2;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(batch, 2, &partition));
    cf_names.push_back("cf" + std::to_string(cf_id));
    if (cf_for_partition[partition] < 0) {
      cf_for_partition[partition] = static_cast<int>(cf_id);
      ++covered;
    }
  }
  // An empty DB at N=1 with an unindexed WAL.
  Close();
  {
    Options n1 = CurrentOptions();
    n1.create_if_missing = true;
    n1.create_missing_column_families = true;
    n1.wal_partitions = 1;
    n1.two_write_queues = false;
    ASSERT_OK(DestroyDB(dbname_, n1));
    ASSERT_OK(TryReopenWithColumnFamilies(cf_names, n1));
    Close();
  }
  auto run_layout = [&](bool with_spans, bool with_sequence,
                        const std::string& tag) {
    if (!with_spans) {
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::StartWALFile:WriteSpan",
          [](void* arg) { *static_cast<bool*>(arg) = false; });
    }
    SyncPoint::GetInstance()->EnableProcessing();
    Options n2 = IndexedWalOptions(CurrentOptions());
    n2.create_missing_column_families = true;
    n2.wal_partitions = 2;
    n2.two_write_queues = false;
    ASSERT_OK(TryReopenWithColumnFamilies(cf_names, n2));
    ASSERT_EQ(2U, dbfull()->TEST_NumWalPartitions());
    WriteBatch marker;
    ASSERT_OK(marker.PutLogData(Slice("floor-probe-" + tag)));
    size_t partition = 2;
    ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(marker, 2, &partition));
    ASSERT_EQ(0U, partition);
    const uint64_t wal = dbfull()->TEST_CurrentWalNumber(0);
    uint64_t begin = 0;
    ASSERT_OK(env_->GetFileSize(LogFileName(dbname_, wal), &begin));
    ASSERT_OK(db_->Write(WriteOptions(), &marker));
    uint64_t end = 0;
    ASSERT_OK(env_->GetFileSize(LogFileName(dbname_, wal), &end));
    ASSERT_GT(end, begin);
    if (with_sequence) {
      ASSERT_OK(Put(cf_for_partition[1], "seq-key", ValueOf("seq-key")));
      ASSERT_NE(0U, dbfull()->GetVersionSet()->LastSequence());
    } else {
      ASSERT_EQ(0U, dbfull()->GetVersionSet()->LastSequence());
    }
    // The upgrade retires the N=1 WAL, so the floor is off in every
    // layout; only spans can bound the torn head.
    ASSERT_NE(0U,
              dbfull()->GetVersionSet()->GetWalSet().GetMinWalNumberToKeep());
    ASSERT_OK(db_->FlushWAL(false));
    const std::string image = dbname_ + "_q8_" + tag;
    CopyDBAsCrashImage(image);
    Close();
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::StartWALFile:WriteSpan");
    // Truncate mid-first-record: a torn tail, so a bounded head opens in
    // every mode but absolute (a byte flip instead would leave a hole,
    // which tolerate refuses too).
    TruncateImageWal(image, wal, (begin + end) / 2);
    const std::string copy = image + "_open";
    for (const WALRecoveryMode mode :
         {kTolerate, kAbsolute, kPointInTime, kSkip}) {
      SCOPED_TRACE("layout=" + tag +
                   " mode=" + std::to_string(static_cast<int>(mode)));
      CopyDir(image, copy);
      std::vector<ColumnFamilyHandle*> handles;
      std::unique_ptr<DB> db;
      const Status s = OpenCopy(copy, 2, n2, mode, &handles, &db);
      for (ColumnFamilyHandle* handle : handles) {
        EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
      }
      if (!with_spans) {
        ASSERT_TRUE(s.IsCorruption()) << s.ToString();
        EXPECT_NE(std::string::npos,
                  s.ToString().find("WAL " + std::to_string(wal) +
                                    " head damaged and unattested"))
            << s.ToString();
      } else if (mode == kAbsolute) {
        // The span anchors the torn head, but absolute still refuses a torn
        // tail; its error names the tail, not an unattested head.
        ASSERT_TRUE(s.IsCorruption()) << s.ToString();
        EXPECT_EQ(std::string::npos,
                  s.ToString().find("head damaged and unattested"))
            << s.ToString();
      } else {
        ASSERT_OK(s) << s.ToString();
      }
    }
    // Restore the pre-layout N=1 image for the next layout.
    ASSERT_OK(DestroyDB(dbname_, n2));
    Options n1 = CurrentOptions();
    n1.create_if_missing = true;
    n1.create_missing_column_families = true;
    n1.wal_partitions = 1;
    n1.two_write_queues = false;
    ASSERT_OK(DestroyDB(dbname_, n1));
    ASSERT_OK(TryReopenWithColumnFamilies(cf_names, n1));
    Close();
  };
  run_layout(true, false, "spans");
  run_layout(false, false, "unattested");
  run_layout(false, true, "sequence");
}

// GC26 Q5 (c): the N>1 column family refusals hold for CreateColumnFamily
// and for SetOptions, not only for DB::Open. max_successive_merges is the
// one of them that can change at runtime. Neuter: none ships here; the
// checks in ColumnFamilyData::ValidateOptions already run on both paths, and
// these cases pin that.
TEST_P(DBWritePartitionTest, GC26ColumnFamilyRefusalsHoldAfterOpen) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  OpenWithPartitions(2, IndexedWalOptions(CurrentOptions()));
  struct Case {
    const char* name;
    std::function<void(ColumnFamilyOptions*)> tweak;
    Status::Code code;
    const char* fragment;
  };
  const std::vector<Case> cases = {
      {"inplace_update_support",
       [](ColumnFamilyOptions* o) { o->inplace_update_support = true; },
       Status::kNotSupported,
       "wal_partitions > 1 is not supported with "
       "inplace_update_support"},
      {"enable_blob_direct_write",
       [](ColumnFamilyOptions* o) {
         o->enable_blob_files = true;
         o->enable_blob_direct_write = true;
       },
       Status::kNotSupported,
       "wal_partitions > 1 is not supported with "
       "enable_blob_direct_write"},
      {"memtable_factory",
       [](ColumnFamilyOptions* o) {
         o->memtable_factory.reset(new VectorRepFactory());
       },
       Status::kNotSupported, "wal_partitions > 1 requires the SkipList"},
      {"max_successive_merges",
       [](ColumnFamilyOptions* o) { o->max_successive_merges = 10; },
       Status::kInvalidArgument,
       "max_successive_merges > 0 is incompatible with wal_partitions > 1"},
      {"timestamp stripping",
       [](ColumnFamilyOptions* o) {
         o->comparator = test::BytewiseComparatorWithU64TsWrapper();
         o->persist_user_defined_timestamps = false;
       },
       Status::kNotSupported, "concurrent memtable write"},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ColumnFamilyOptions cf_options(CurrentOptions());
    c.tweak(&cf_options);
    ColumnFamilyHandle* handle = nullptr;
    const Status s = db_->CreateColumnFamily(
        cf_options, std::string("gc26_") + c.name, &handle);
    EXPECT_EQ(c.code, s.code()) << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find(c.fragment)) << s.ToString();
    EXPECT_EQ(nullptr, handle);
    if (handle != nullptr) {
      ASSERT_OK(db_->DestroyColumnFamilyHandle(handle));
    }
  }
  ColumnFamilyHandle* handle = nullptr;
  ASSERT_OK(db_->CreateColumnFamily(ColumnFamilyOptions(CurrentOptions()),
                                    "gc26_allowed", &handle));
  ASSERT_OK(db_->DestroyColumnFamilyHandle(handle));

  const int cf = cf_for_partition_[1];
  const Status s =
      db_->SetOptions(handles_[cf], {{"max_successive_merges", "10"}});
  EXPECT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(std::string::npos,
            s.ToString().find("max_successive_merges > 0 is incompatible "
                              "with wal_partitions > 1"))
      << s.ToString();
  EXPECT_EQ(0U, db_->GetOptions(handles_[cf]).max_successive_merges);
  ASSERT_OK(db_->SetOptions(handles_[cf], {{"max_successive_merges", "0"}}));

  // N = 1 is unaffected.
  ReopenWithPartitions(1, IndexedWalOptions(CurrentOptions()));
  ASSERT_OK(db_->SetOptions(handles_[cf], {{"max_successive_merges", "10"}}));
  EXPECT_EQ(10U, db_->GetOptions(handles_[cf]).max_successive_merges);
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
