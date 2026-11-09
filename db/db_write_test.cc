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
    if (!cv.wait_for(lock, std::chrono::seconds(10),
                     [&] { return waiting == 2; })) {
      // A follower that never queued would lead a group nobody exits, so
      // the threads cannot be joined: abort with a reason instead of hanging.
      fprintf(stderr, "B and C never both queued behind A\n");
      std::abort();
    }
  }

  WriteThread::WriteGroup group;
  write_thread.EnterAsBatchGroupLeader(&wa, &group);
  EXPECT_EQ(2u, group.size);
  EXPECT_EQ(0u, group.num_cap_excluded);
  std::vector<WriteThread::Writer*> members;
  for (auto* w : group) {
    members.push_back(w);
  }
  if (members.size() != 2 || members[0] != &wa || members[1] != &wb) {
    // The steps below cannot drive any other grouping (with B alone, C
    // queues behind B and nothing exits its group; with C in A's group, C
    // is completed, not a leader), so abort with a reason.
    fprintf(stderr, "group has %zu members, expected A and B\n",
            members.size());
    std::abort();
  }

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

void CorruptLogFile(Env* env, Options& options, const std::string& log_path,
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

TEST_F(DBWriteTestUnparameterized, WalPartitionsTestingOptIn) {
  // Uses no sync point, so it also runs in release builds: there the
  // testing-only opt-in is the only way to open wal_partitions > 1, and it
  // admits indexed WALs only.
  constexpr uint32_t kPartitions = 4;
  Options options = CurrentOptions();
  options.wal_partitions = kPartitions;
  options.track_and_verify_wals_in_manifest = true;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_shutdown = true;
  options.avoid_flush_during_recovery = true;
  DestroyDB(dbname_, options).PermitUncheckedError();
  Status s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  ASSERT_NE(nullptr, std::strstr(s.getState(),
                                 "wal_partitions > 1 is not supported yet"));

  options.experimental_allow_wal_partitions_for_testing = true;
  options.partition_wal_usage = PartitionWALUsage::kNone;
  s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  ASSERT_NE(nullptr, std::strstr(s.getState(),
                                 "wal_partitions > 1 is not supported yet"));

  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
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

  // The OPTIONS file keeps wal_partitions but not the opt-in, so reopening
  // from it is rejected.
  DBOptions loaded_db_options;
  std::vector<ColumnFamilyDescriptor> loaded_cfs;
  ConfigOptions config_options;
  config_options.env = env_;
  ASSERT_OK(LoadLatestOptions(config_options, dbname_, &loaded_db_options,
                              &loaded_cfs));
  ASSERT_EQ(kPartitions, loaded_db_options.wal_partitions);
  ASSERT_FALSE(loaded_db_options.experimental_allow_wal_partitions_for_testing);
  Close();
  s = TryReopen(Options(loaded_db_options, loaded_cfs[0].options));
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
}

// Unindexed WALs cannot be merged at recovery, so kNone with N > 1 is rejected
// with and without the test hook, naming both options.
TEST_F(DBWriteTestUnparameterized, UnindexedWalPartitionsAlwaysRejected) {
  for (const bool hook : {false, true}) {
    if (hook) {
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::ValidateOptions:AllowWalPartitions",
          [](void* arg) { *static_cast<bool*>(arg) = true; });
      SyncPoint::GetInstance()->EnableProcessing();
    }
    for (const uint32_t partitions : {2U, 4U}) {
      SCOPED_TRACE("hook=" + std::to_string(hook) +
                   " partitions=" + std::to_string(partitions));
      Options options = CurrentOptions();
      options.wal_partitions = partitions;
      options.partition_wal_usage = PartitionWALUsage::kNone;
      Destroy(options);
      const Status s = TryReopen(options);
      ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
      EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
      EXPECT_NE(nullptr, std::strstr(s.getState(), "partition_wal_usage"));
    }
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

// wal_partitions above kMaxWalPartitions is rejected with and without the
// test hook; kMaxWalPartitions itself opens with the hook.
TEST_F(DBWriteTestUnparameterized, WalPartitionsUpperBound) {
  Options options = CurrentOptions();
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_shutdown = true;
  options.wal_partitions = DBImpl::kMaxWalPartitions + 1;
  options.track_and_verify_wals_in_manifest = true;
  Destroy(options);
  for (const bool hook : {false, true}) {
    SCOPED_TRACE("hook=" + std::to_string(hook));
    if (hook) {
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::ValidateOptions:AllowWalPartitions",
          [](void* arg) { *static_cast<bool*>(arg) = true; });
      SyncPoint::GetInstance()->EnableProcessing();
    }
    const Status s = TryReopen(options);
    ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
    EXPECT_NE(nullptr, std::strstr(s.getState(), "wal_partitions"));
    EXPECT_NE(nullptr,
              std::strstr(s.getState(),
                          std::to_string(DBImpl::kMaxWalPartitions).c_str()));
  }
  options.wal_partitions = DBImpl::kMaxWalPartitions;
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ(DBImpl::kMaxWalPartitions, dbfull()->TEST_NumWalPartitions());
  ASSERT_OK(Put("key", "value"));
  ASSERT_EQ("value", Get("key"));
}

// Every rejection of wal_partitions > 1, with the test hook on: each names
// wal_partitions and its conflicting option, and the two M1.5 cells (kNone,
// and two_write_queues, which rejects indexing) also name partition_wal_usage.
// The runtime rejections follow on a DB that opens.
TEST_F(DBWriteTestUnparameterized, WalPartitionsRejectionsNameTheirOptions) {
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
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
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::ValidateOptions:AllowWalPartitions",
        [](void* arg) { *static_cast<bool*>(arg) = true; });
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

  // Partition p's lowest range not yet marked applied, or
  // WalDurabilityTracker::kIdle. The applied tracker bounds what N > 1
  // publishes.
  uint64_t AppliedPendingLow(size_t partition) {
    return dbfull()->applied_tracker_->PendingLow(partition);
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

  // The same with `write_options`, as a sync 2PC Prepare writes.
  Status WalOnlyWrite(const WriteOptions& write_options, WriteBatch* batch) {
    return dbfull()->WriteImpl(write_options, batch, /*callback=*/nullptr,
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
  // appends seen in flight at once. An append counts from inside its
  // partition's WAL mutex (after the timestamp-size record, before
  // AddRecord) until AddRecord returns, so appends whose mutexes serialize
  // them never get past one, and each waiter gives up at a shared deadline.
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
        "DBImpl::WriteToWAL:AfterMaybeAddUserDefinedTimestampSizeRecord",
        [&](void*) {
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
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterMaybeAddUserDefinedTimestampSizeRecord");
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
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::ValidateOptions:AllowWalPartitions",
        [](void* arg) { *static_cast<bool*>(arg) = true; });
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
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
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
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_EQ("value", Get(cf_for_partition_[p], "main" + std::to_string(p)));
  }

  // Each partition appended to a WAL of its own.
  VectorWalPtr wal_files;
  ASSERT_OK(db_->GetSortedWalFiles(wal_files));
  ASSERT_GE(wal_files.size(), kPartitions);
}

// Above one partition, one partition's leader checks the flush scheduler
// without the DB mutex while another's takes the scheduled column family
// under it. FlushScheduler::Empty() read the head outside the lock that
// guards its debug checking set, so a dequeue landing between the read and
// the check tripped its assertion in debug builds (a whitebox crash test at
// N=8 did). Here B's check reads a non-empty head and A then dequeues before
// B checks; B must not assert. Neuter: read the head outside the lock again;
// B aborts on "(rv == checking_set_.empty()) || rv".
TEST_P(DBWritePartitionTest, FlushSchedulerEmptyRacesAnotherLeadersDequeue) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  // A memtable past its size schedules its flush at the write after the one
  // that filled it, for the write after that to take.
  ASSERT_OK(Put(cf_for_partition_[0], "big", std::string(128 << 10, 'x')));
  ASSERT_OK(Put(cf_for_partition_[0], "small", "v"));

  std::mutex mu;
  std::condition_variable cv;
  bool b_checking = false;
  bool dequeued = false;
  std::thread::id b_id;
  SyncPoint::GetInstance()->SetCallBack(
      "FlushScheduler::Empty:AfterLoad", [&](void* arg) {
        if (*static_cast<bool*>(arg) || std::this_thread::get_id() != b_id) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (b_checking) {
          return;  // only B's first check waits
        }
        b_checking = true;
        cv.notify_all();
        // Waits for A's dequeue. With the head read under the scheduler's
        // lock A cannot dequeue meanwhile, so this times out instead.
        cv.wait_for(lock, std::chrono::seconds(2), [&] { return dequeued; });
      });
  SyncPoint::GetInstance()->SetCallBack(
      "FlushScheduler::TakeNextColumnFamily:Dequeued", [&](void* /*arg*/) {
        std::lock_guard<std::mutex> lock(mu);
        dequeued = true;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->EnableProcessing();

  Status b_status;
  port::Thread b_thread([&] {
    {
      std::lock_guard<std::mutex> lock(mu);
      b_id = std::this_thread::get_id();
    }
    b_status = Put(cf_for_partition_[1], "b", "vb");
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return b_checking; }));
  }
  const Status a_status = Put(cf_for_partition_[0], "a", "va");
  b_thread.join();
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearCallBack("FlushScheduler::Empty:AfterLoad");
  SyncPoint::GetInstance()->ClearCallBack(
      "FlushScheduler::TakeNextColumnFamily:Dequeued");
  SyncPoint::GetInstance()->EnableProcessing();

  ASSERT_OK(a_status);
  ASSERT_OK(b_status);
  EXPECT_EQ("va", Get(cf_for_partition_[0], "a"));
  EXPECT_EQ("vb", Get(cf_for_partition_[1], "b"));
}

// Coupled rotation retires the WALs of partitions that took no write, holding
// only WAL index metadata, and a closed-WAL sync tracks them in the MANIFEST.
// GetSortedWalFiles must list them rather than skip them as empty and then
// report the tracked files missing. Neuter: in WalManager::GetSortedWalsOfType,
// skip a tracked metadata-only WAL as empty; GetSortedWalFiles returns "WAL
// file N required by manifest but not in directory list".
TEST_P(DBWritePartitionTest, SortedWalFilesListsIdlePartitionWals) {
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
  // Partition 1's column family keeps unflushed data in the first generation,
  // so its WALs stay live; partitions 2 and 3 take no write.
  ASSERT_OK(Put(cf_for_partition_[0], "key0", "value"));
  ASSERT_OK(Put(cf_for_partition_[1], "key1", "value"));
  ASSERT_OK(Flush(cf_for_partition_[0]));

  VectorWalPtr wal_files;
  ASSERT_OK(db_->GetSortedWalFiles(wal_files));
  std::set<uint64_t> listed;
  for (const auto& wal : wal_files) {
    listed.insert(wal->LogNumber());
  }
  size_t tracked = 0;
  for (const auto& [number, metadata] :
       dbfull()->GetVersionSet()->GetWalSet().GetWals()) {
    if (metadata.HasSyncedSize()) {
      ++tracked;
      EXPECT_EQ(1U, listed.count(number)) << "tracked WAL " << number;
    }
  }
  ASSERT_GT(tracked, 0U);
}

// The idle partitions' metadata-only WALs belong to the files, not to the
// opener: a read-only open with default options (one partition, WAL tracking
// off) opens and lists them too. Neuters: give ReadFirstLine's WAL index
// marker case a wal_partitions > 1 condition, and GetSortedWalFiles returns
// "WAL file N required by manifest but not in directory list"; drop the
// read-only condition from Recover's clearing of tracked WALs, and the open
// crashes (debug: assertion `recovery_ctx != nullptr').
TEST_P(DBWritePartitionTest, SortedWalFilesListsIdlePartitionWalsReadOnly) {
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
  ASSERT_OK(Put(cf_for_partition_[0], "key0", "value"));
  ASSERT_OK(Put(cf_for_partition_[1], "key1", "value"));
  ASSERT_OK(Flush(cf_for_partition_[0]));
  std::set<uint64_t> tracked;
  for (const auto& [number, metadata] :
       dbfull()->GetVersionSet()->GetWalSet().GetWals()) {
    if (metadata.HasSyncedSize()) {
      tracked.insert(number);
    }
  }
  ASSERT_FALSE(tracked.empty());
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  Close();

  // Default options: one partition and WAL tracking off. Recovery must not try
  // to clear the tracked WALs, which a read-only open cannot write.
  Options read_only = CurrentOptions();
  ASSERT_FALSE(read_only.track_and_verify_wals_in_manifest);
  std::vector<ColumnFamilyDescriptor> descriptors;
  for (const std::string& name : cf_names) {
    descriptors.emplace_back(name, ColumnFamilyOptions(read_only));
  }
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> owned;
  ASSERT_OK(DB::OpenForReadOnly(DBOptions(read_only), dbname_, descriptors,
                                &handles, &owned));
  Defer destroy_handles([&] {
    for (ColumnFamilyHandle* handle : handles) {
      EXPECT_OK(owned->DestroyColumnFamilyHandle(handle));
    }
  });
  VectorWalPtr wal_files;
  ASSERT_OK(owned->GetSortedWalFiles(wal_files));
  std::set<uint64_t> listed;
  for (const auto& wal : wal_files) {
    listed.insert(wal->LogNumber());
  }
  for (uint64_t number : tracked) {
    EXPECT_EQ(1U, listed.count(number)) << "tracked WAL " << number;
  }
}

// GetLiveFiles followed by GetSortedWalFiles, the copy method include/rocksdb/
// db.h documents, must capture the idle partitions' metadata-only WALs the
// MANIFEST tracks, or the copy fails its WAL check on open.
// Neuters: leave a tracked metadata-only WAL out of GetSortedWalFiles and
// accept it in the MANIFEST cross-check, and the copy's open fails with
// "Missing WAL with log number"; leave it out without accepting it, and
// GetSortedWalFiles fails with "required by manifest but not in directory
// list".
TEST_P(DBWritePartitionTest, LiveFilesAndSortedWalFilesCopyOpens) {
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
  ASSERT_OK(Put(cf_for_partition_[0], "key0", "value0"));
  ASSERT_OK(Put(cf_for_partition_[1], "key1", "value1"));
  ASSERT_OK(Flush(cf_for_partition_[0]));

  std::vector<std::string> live_files;
  uint64_t manifest_size = 0;
  ASSERT_OK(db_->GetLiveFiles(live_files, &manifest_size,
                              /*flush_memtable=*/false));
  VectorWalPtr wal_files;
  ASSERT_OK(db_->GetSortedWalFiles(wal_files));
  std::set<uint64_t> listed;
  for (const auto& wal : wal_files) {
    listed.insert(wal->LogNumber());
  }
  for (const auto& [number, metadata] :
       dbfull()->GetVersionSet()->GetWalSet().GetWals()) {
    if (metadata.HasSyncedSize()) {
      EXPECT_EQ(1U, listed.count(number)) << "tracked WAL " << number;
    }
  }

  const std::string copy_dir = dbname_ + "_copy";
  ASSERT_OK(DestroyDB(copy_dir, options));
  ASSERT_OK(env_->CreateDirIfMissing(copy_dir));
  for (const std::string& file : live_files) {
    uint64_t number = 0;
    FileType type;
    ASSERT_TRUE(ParseFileName(file.substr(1), &number, &type)) << file;
    CopyFile(dbname_ + file, copy_dir + file,
             type == kDescriptorFile ? manifest_size : 0);
  }
  for (const auto& wal : wal_files) {
    ASSERT_EQ(kAliveLogFile, wal->Type());
    CopyFile(dbname_ + wal->PathName(),
             copy_dir + "/" + LogFileName("", wal->LogNumber()).substr(1));
  }

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  Options copy_options = options;
  copy_options.wal_partitions = kPartitions;
  copy_options.partition_wal_usage =
      PartitionWALUsage::kPartitionByColumnFamily;
  copy_options.track_and_verify_wals_in_manifest = true;
  copy_options.create_if_missing = false;
  std::vector<ColumnFamilyDescriptor> descriptors;
  for (const std::string& name : cf_names) {
    descriptors.emplace_back(name, ColumnFamilyOptions(copy_options));
  }
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> copy;
  ASSERT_OK(DB::Open(DBOptions(copy_options), copy_dir, descriptors, &handles,
                     &copy));
  Defer destroy_handles([&] {
    for (ColumnFamilyHandle* handle : handles) {
      EXPECT_OK(copy->DestroyColumnFamilyHandle(handle));
    }
  });
  std::string value;
  ASSERT_OK(
      copy->Get(ReadOptions(), handles[cf_for_partition_[1]], "key1", &value));
  ASSERT_EQ("value1", value);
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
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(10),
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
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();

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
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();

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
  // partition) is started inside that window. D-9: the write is inserted
  // but not published until the fsync, so the rotation's memtable
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
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return in_gate_sync; }));
  }
  EXPECT_OK(Flush(cf_for_partition_[1]));
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
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
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
  // write fsyncs only the two it depends on (GC28 Q1), and marks each under
  // its own WAL mutex, never all N (GC31 Q4).
  ASSERT_OK(Put(cf_for_partition_[0], "k2", "v"));
  ASSERT_OK(Put(cf_for_partition_[0], "k3", "v", sync_wo));
  EXPECT_EQ(1U, count(WAL_SYNC_FSYNCED_PARTITIONS));
  EXPECT_EQ(2U, sum(WAL_SYNC_FSYNCED_PARTITIONS));
  EXPECT_EQ(1U, count(WAL_SYNC_UNSYNCED_PARTITIONS));
  EXPECT_EQ(2U, sum(WAL_SYNC_UNSYNCED_PARTITIONS));
  EXPECT_EQ(1U, count(WAL_SYNC_BARRIER_MICROS));
  EXPECT_EQ(0U, count(WAL_WRITE_MUTEX_ALL_WAIT_MICROS));
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
  // Joins the syncer, which finishes on its own, on every exit.
  Defer join_syncer([&] {
    if (syncer.joinable()) {
      syncer.join();
    }
  });
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
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
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
  // R4 (second half): for sync writes the gate's outcome includes the
  // post-wait sync. B waits on held A, then its gate sync, which runs only
  // after the wait (GC28 Q3), fails; B reports the sync error. Standard path
  // only: here no leadership-time sync exists, so arming the failure from
  // the start can only hit the post-wait sync.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  // A is sync=false, so the only gate sync in this test is B's post-wait
  // one.
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
  // B parks in its gate behind A.
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
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
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
  // M2.7 (second half of R2): the completion gate's own post-wait sync
  // establishes the same background error as a leadership sync failure and
  // wakes every other parked gate. A (async, partition 0) is held open; B
  // (sync, partition 1) appends and parks; M (async, partition 2)
  // allocates then holds its append open; W (async, partition 0) appends
  // after A is released and parks behind M's still-open range. Releasing
  // A lets B reach its post-wait sync, which fails: W wakes and, being above
  // B's failed range, fails with the background error while M is still held
  // (D-9; at GC24, before D-9, it parked again as D-N71 (b) had it). M's
  // append then fails too.
  // Neuter: no fence for a failed sync (FailUnpublishedSyncRange always
  // publishing the range as a hole): W is not failed while M is held.
  // Standard path only (as in PostWaitSyncErrorFailsSyncWrite): with two
  // write queues the leadership sync itself goes through SyncWAL, so the
  // failure cannot be aimed at the post-wait sync alone.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  std::atomic<int> sync_wal_begins{0};
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
  // No leadership-time sync exists on this path, so the only gate sync is
  // B's post-wait one (GC28 Q3: it runs only after B's wait). Hold it until
  // W installed its gate and parked, so W is provably parked when B's sync
  // fails.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex", [&](void*) {
        ++sync_wal_begins;
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
  // Exactly one gate sync ran: B's post-wait one. One notification per
  // failure: B's post-wait sync (classified by SyncWalsFromCompletionGate),
  // then M's append.
  EXPECT_EQ(1, sync_wal_begins.load());
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
      "DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex");
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
  // OK, deadlock nothing, and leak no waiter-count. W's append succeeded and
  // a background error that is not a fence only wakes a parked gate to
  // recheck, so W returns OK once A completes, below a covering frontier.
  // The last iteration is deterministic: the error is published, and its
  // wake delivered, before A is released.
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

  for (int iter = 0; iter <= kIterations; ++iter) {
    const bool error_first = iter == kIterations;
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
        "DBImpl::SignalWalDurabilityWaitersOnBGError", [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          ++bg_error_wakes;
          cv.notify_all();
        });

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
      std::unique_lock<std::mutex> lock(mu);
      if (error_first) {
        EXPECT_TRUE(cv.wait_until(lock, deadline,
                                  [&] { return bg_error_wakes.load() == 1; }));
      }
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
    ASSERT_OK(w_status);
    {
      // No stale OK: an acknowledged write sits below a covering frontier.
      std::lock_guard<std::mutex> lock(mu);
      EXPECT_NE(0U, w_target);
      EXPECT_LE(w_target, dbfull()->WalDurabilitySafePoint());
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    EXPECT_FALSE(release_a);
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
  // Q1, gate sync failure: A is held on partition 0 after its append, before
  // it publishes; W appends on partition 1 and parks in its gate; C (sync)
  // then fails exactly its gate fsync, which runs from its completion gate
  // once every record below it is written (GC28 Q3). The
  // failure fences C's range, which lies above W's, so the wake leaves W
  // parked (D-N71 (b)): once A is released W is acknowledged, as one WAL
  // acknowledges an async write whose append landed before a later sync
  // failed. Standard path only: with two write queues a sync write takes all
  // N WAL mutexes in PreprocessWrite, so C would block on A's held append
  // before ever reaching its sync. The two-queue write-path sync still
  // funnels through the same SetBGError transition the two-queue
  // append-failure run already covers; only the funnel (WriteStatusCheck)
  // differs, and it is pre-existing code.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
      "DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:pause",
      [&](void*) {
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
        if (a_in_append && !release_a && !w_appended) {
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
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_gate_sync = true;
  }
  WriteOptions sync_options;
  sync_options.sync = true;
  port::Thread c_thread([&] {
    c_status = db_->Put(sync_options, handles_[cf_w], "key_c", "value_c");
  });
  c_thread.join();
  // C's failure is published, and W is still parked behind held A.
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_FALSE(w_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    EXPECT_FALSE(release_a);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  w_thread.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:pause");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
        EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                                [&] { return s_in_sync; }));
      }
      port::Thread c_thread([&] {
        c_status = Put(cf_c, "key_c", "value_c");
        c_returned = true;
      });
      EXPECT_TRUE(WaitForDurabilityWaiters(1));

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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    if (!wait_for([&] { return w0_in_append; })) {
      // Stop before a later writer can take the hold below W0, where only
      // release_l could free it.
      {
        std::lock_guard<std::mutex> lock(mu);
        hold_append = false;
        release_w0 = true;
        release_l = true;
        cv.notify_all();
      }
      w0.join();
      SyncPoint::GetInstance()->ClearAllCallBacks();
      FAIL() << "W0 never reached its WAL append";
    }
    port::Thread l([&] { l_status = Put(cf, "l", "v", sync_wo); });
    EXPECT_TRUE(wait_for([&] { return waiting == 1; }));
    auto async_writer = [&](const std::string& key, Status* status) {
      *status = Put(cf, key, "v");
      std::lock_guard<std::mutex> lock(mu);
      ++async_returned;
      cv.notify_all();
    };
    port::Thread a1([&] { async_writer("a1", &a1_status); });
    EXPECT_TRUE(wait_for([&] { return waiting == 2; }));
    port::Thread a2([&] { async_writer("a2", &a2_status); });
    EXPECT_TRUE(wait_for([&] { return waiting == 3; }));
    {
      std::lock_guard<std::mutex> lock(mu);
      release_w0 = true;
      cv.notify_all();
    }
    w0.join();
    EXPECT_OK(w0_status);
    EXPECT_TRUE(wait_for([&] { return l_in_sync; }));
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

TEST_P(DBWritePartitionTest, SyncLedGroupAsyncMembersReturnBeforeFsync) {
  // In a group led by a sync write, the async members take the lower
  // sequences and return once written and visible, without waiting for the
  // fsync; the sync members take the higher sequences and stay parked and
  // invisible until it. S (sync leader), A1, A2 (async) and S2 (sync
  // follower) form one group on partition 1; S's gate sync is held. A1 and
  // A2 return OK and read back while S and S2 have not returned and do not
  // read. Releasing the sync completes S and S2. Both async sequences lie
  // below both sync ones, in link order within each class.
  // Ported from GC25's closed Muse candidate.
  // Neuter: run the async writers' gate after the sync writers' completion
  // gate (WriteThread::ExitAsBatchGroupLeader), so the async members wait
  // for the fsync: they never return while it is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const size_t group_partition = 1;
    const int cf = cf_for_partition_[group_partition];

    std::mutex mu;
    std::condition_variable cv;
    bool s_waiting = false;
    int joiners = 0;
    bool release_s = false;
    bool fsync_held = false;
    bool release_fsync = false;
    int async_returned = 0;

    // S waits at group formation until A1, A2 and S2 have all queued
    // behind it, so the four always form one S-led group.
    SyncPoint::GetInstance()->SetCallBack(
        "WriteThread::JoinBatchGroup:Wait", [&](void* arg) {
          if (static_cast<WriteThread::Writer*>(arg)->wal_partition !=
              group_partition) {
            return;
          }
          std::unique_lock<std::mutex> lock(mu);
          if (!s_waiting) {
            s_waiting = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_s; });
          } else if (++joiners == 3) {
            release_s = true;
            cv.notify_all();
          }
        });
    // S's gate sync is held; S, the group's exit, is the only writer that
    // reaches it while held.
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (!fsync_held) {
            fsync_held = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_fsync; });
          }
        });

    WriteOptions sync_wo;
    sync_wo.sync = true;
    Status s_status, a1_status, a2_status, s2_status;
    std::atomic<bool> s_returned{false};
    std::atomic<bool> s2_returned{false};
    SequenceNumber s_seq = 0, a1_seq = 0, a2_seq = 0, s2_seq = 0;
    auto run_writer = [&](const std::string& key, const WriteOptions& wo,
                          Status* status, SequenceNumber* seq) {
      WriteBatch batch;
      *status = batch.Put(handles_[cf], key, "value_" + key);
      if (status->ok()) {
        *status = db_->Write(wo, &batch);
        *seq = WriteBatchInternal::Sequence(&batch);
      }
    };
    auto run_async = [&](const std::string& key, Status* status,
                         SequenceNumber* seq) {
      run_writer(key, WriteOptions(), status, seq);
      std::lock_guard<std::mutex> lock(mu);
      ++async_returned;
      cv.notify_all();
    };
    auto wait_for = [&](const std::function<bool()>& pred) {
      std::unique_lock<std::mutex> lock(mu);
      return cv.wait_for(lock, std::chrono::seconds(30), pred);
    };
    port::Thread s_thread([&] {
      run_writer("s", sync_wo, &s_status, &s_seq);
      s_returned.store(true);
    });
    EXPECT_TRUE(wait_for([&] { return s_waiting; }));
    port::Thread a1_thread([&] { run_async("a1", &a1_status, &a1_seq); });
    port::Thread a2_thread([&] { run_async("a2", &a2_status, &a2_seq); });
    port::Thread s2_thread([&] {
      run_writer("s2", sync_wo, &s2_status, &s2_seq);
      s2_returned.store(true);
    });
    EXPECT_TRUE(wait_for([&] { return fsync_held; }));
    // The async members return OK and are visible while the sync is held;
    // the sync members are parked and invisible.
    EXPECT_TRUE(wait_for([&] { return async_returned == 2; }));
    {
      std::lock_guard<std::mutex> lock(mu);
      EXPECT_OK(a1_status);
      EXPECT_OK(a2_status);
    }
    EXPECT_FALSE(s_returned.load());
    EXPECT_FALSE(s2_returned.load());
    EXPECT_EQ("value_a1", Get(cf, "a1"));
    EXPECT_EQ("value_a2", Get(cf, "a2"));
    EXPECT_EQ("NOT_FOUND", Get(cf, "s"));
    EXPECT_EQ("NOT_FOUND", Get(cf, "s2"));
    {
      std::lock_guard<std::mutex> lock(mu);
      release_s = true;
      release_fsync = true;
      cv.notify_all();
    }
    s_thread.join();
    a1_thread.join();
    a2_thread.join();
    s2_thread.join();
    SyncPoint::GetInstance()->ClearCallBack("WriteThread::JoinBatchGroup:Wait");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::CompletionGate:BeforeCoalescedSync");

    ASSERT_OK(s_status);
    ASSERT_OK(s2_status);
    EXPECT_EQ("value_s", Get(cf, "s"));
    EXPECT_EQ("value_s2", Get(cf, "s2"));
    // Async members take the lower sequences, link order within each
    // class: S leads, so S before S2.
    EXPECT_LT(a1_seq, s_seq);
    EXPECT_LT(a2_seq, s_seq);
    EXPECT_LT(s_seq, s2_seq);
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  // failed sync never inserted the write. Async A on partition 0 is held
  // after its append, before it publishes, so B (sync, partition 1, above A)
  // cannot publish; B's gate sync, which runs once A's record is written
  // (GC28 Q3), then fails. Once A is released A is acknowledged and
  // published, while B stays absent: before Resume(), after Resume() and a
  // flush, and after a reopen in every recovery mode. Then, above A2 held the
  // same way, C (sync) is held inside its gate sync while D (sync, above C)
  // queues behind it in the coalescer; C's sync fails, and both get the error
  // and stay absent. Every write acknowledged below them stays present.
  // Neuter: insert and publish the sync range before its fsync (RunSyncGate's
  // callbacks, insert and PublishLastSequence moved before
  // RunCoalescedGateSync); A's publication
  // exposes B.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
        "DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:pause",
        [&](void*) {
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

    Status a_status;
    port::Thread a_thread(
        [&] { a_status = Put(cf_a, "key_a", "value_a", WriteOptions()); });
    bool a_held = false;
    {
      std::unique_lock<std::mutex> lock(mu);
      a_held = cv.wait_for(lock, std::chrono::seconds(30),
                           [&] { return a_in_append; });
      if (!a_held) {
        // Otherwise B's own exit would take the hold, and only this thread
        // could release it.
        hold_append = false;
        release_a = true;
        cv.notify_all();
      }
      fail_syncs = 1;
    }
    if (!a_held) {
      a_thread.join();
      SyncPoint::GetInstance()->ClearAllCallBacks();
      FAIL() << "A never reached its hold";
    }
    const Status b_status = Put(cf_b, "key_b", "value_b", sync_wo);
    EXPECT_TRUE(b_status.IsIOError()) << b_status.ToString();
    EXPECT_NOK(dbfull()->TEST_GetBGError());
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
    // False once any wait below times out: C may then be held by the hold
    // meant for A2, or parked behind A2, so A2 is released before C's join.
    bool coordinated = true;
    {
      std::unique_lock<std::mutex> lock(mu);
      coordinated &= cv.wait_for(lock, std::chrono::seconds(30),
                                 [&] { return a_in_append; });
      EXPECT_TRUE(coordinated);
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
      const bool c_ok = cv.wait_for(lock, std::chrono::seconds(30),
                                    [&] { return c_in_sync; });
      EXPECT_TRUE(c_ok);
      coordinated &= c_ok;
    }
    port::Thread d_thread(
        [&] { d_status = Put(cf_d, "key_d", "value_d", sync_wo); });
    {
      std::unique_lock<std::mutex> lock(mu);
      const bool d_ok = cv.wait_for(lock, std::chrono::seconds(30),
                                    [&] { return d_waiting; });
      EXPECT_TRUE(d_ok);
      coordinated &= d_ok;
      fail_syncs = 1;
      release_c = true;
      if (!coordinated) {
        hold_append = false;
        release_a = true;
      }
      cv.notify_all();
    }
    c_thread.join();
    d_thread.join();
    EXPECT_TRUE(c_status.IsIOError()) << c_status.ToString();
    EXPECT_TRUE(d_status.IsIOError()) << d_status.ToString();
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    EXPECT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return c_in_sync; }));
  }
  port::Thread d_thread(
      [&] { d_status = Put(cf_b, "key_d", "value_d", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    EXPECT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return f_in_sync; }));
  }
  // Outside ["a", "m"). Same partition is fine: F already left its write
  // thread, so this forms its own group above F's range.
  Status zebra_status;
  port::Thread zebra_thread(
      [&] { zebra_status = Put(cf, "zebra", "value_zebra"); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    EXPECT_FALSE(release_a);
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
    EXPECT_FALSE(release_a);
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
      EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_held; }));
    }
    port::Thread b_thread([&] {
      b_status = Put(cf_b, "key_b", "value_b");
      b_returned.store(true);
    });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return b_appended; }));
    }
    // A is unpublished either way (its record has not landed, or it is not
    // inserted yet), so B parks behind it.
    EXPECT_TRUE(WaitForDurabilityWaiters(1));
    std::vector<port::Thread> switch_puts =
        FillAndSwitchMemtable(cf_switch, [&] {
          std::lock_guard<std::mutex> lock(mu);
          return switch_waiting;
        });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  // rounds run, S1's and one that covers every queued write, each fsyncing
  // every partition once. Every write is acknowledged, and all of them
  // survive a crash that drops every unsynced byte. Neuter: a round that
  // skips a partition loses the queued write on it in the crash.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
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
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
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
    EXPECT_EQ(2, fsyncs[p]) << "partition " << p;
  }

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s1"));
  for (size_t p = 0; p < kPartitions; ++p) {
    EXPECT_EQ("v", Get(cf_for_partition_[p], "q" + std::to_string(p)));
  }
}

// GC28 tests. Helpers shared by this block only.
namespace {
// Set on the test thread when a failed assertion unwinds a still-running
// write (DBWalAppendFailureTest::Writer), so every test hold lets its
// thread go and the write can be joined; cleared when a hold is armed.
std::atomic<bool> test_holds_abandoned{false};

// Waits on `cv` until `pred` holds or the test abandons its holds.
void WaitForReleaseOrAbandon(std::unique_lock<std::mutex>& lock,
                             std::condition_variable& cv,
                             const std::function<bool()>& pred) {
  while (!cv.wait_for(lock, std::chrono::milliseconds(100), pred)) {
    if (test_holds_abandoned.load()) {
      return;
    }
  }
}

// Holds the first thread that reaches `point` (and passes `match`) until
// Release(), and counts every matching hit.
class GC28Hold {
 public:
  GC28Hold() = default;
  GC28Hold(const GC28Hold&) = delete;
  GC28Hold& operator=(const GC28Hold&) = delete;
  ~GC28Hold() {
    Release();
    if (!point_.empty()) {
      SyncPoint::GetInstance()->ClearCallBack(point_);
    }
  }
  void Arm(const std::string& point,
           const std::function<bool(void*)>& match = nullptr,
           bool hold = true) {
    point_ = point;
    test_holds_abandoned = false;
    SyncPoint::GetInstance()->SetCallBack(
        point, [this, match, hold](void* arg) {
          if (match && !match(arg)) {
            return;
          }
          std::unique_lock<std::mutex> lock(mu_);
          ++hits_;
          cv_.notify_all();
          if (!hold || held_) {
            return;
          }
          held_ = true;
          WaitForReleaseOrAbandon(lock, cv_, [this] { return released_; });
        });
  }
  bool WaitHeld(std::chrono::seconds timeout = std::chrono::seconds(30)) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this] { return held_; });
  }
  bool WaitHits(int n,
                std::chrono::seconds timeout = std::chrono::seconds(30)) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this, n] { return hits_ >= n; });
  }
  int Hits() {
    std::lock_guard<std::mutex> lock(mu_);
    return hits_;
  }
  void ResetHits() {
    std::lock_guard<std::mutex> lock(mu_);
    hits_ = 0;
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::string point_;
  std::mutex mu_;
  std::condition_variable cv_;
  int hits_ = 0;
  bool held_ = false;
  bool released_ = false;
};

// Matches a SyncWalPartitionFiles sync point fired for `partition`.
std::function<bool(void*)> GC28Partition(size_t partition) {
  return [partition](void* arg) {
    return *static_cast<size_t*>(arg) == partition;
  };
}

// Matches a sync point reached on the thread whose id `id` holds.
std::function<bool(void*)> GC28OnThread(
    const std::atomic<std::thread::id>* id) {
  return [id](void*) { return std::this_thread::get_id() == id->load(); };
}

// Counts gate syncs per partition (SyncWalPartitionFiles:BeforeSyncPartition)
// until destroyed.
class GC28FsyncCounter {
 public:
  explicit GC28FsyncCounter(size_t partitions) : fsyncs_(partitions, 0) {
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [this](void* arg) {
          std::lock_guard<std::mutex> lock(mu_);
          ++fsyncs_[*static_cast<size_t*>(arg)];
        });
  }
  ~GC28FsyncCounter() {
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  }
  std::vector<int> Get() {
    std::lock_guard<std::mutex> lock(mu_);
    return fsyncs_;
  }
  int Total() {
    int total = 0;
    for (int n : Get()) {
      total += n;
    }
    return total;
  }

 private:
  std::mutex mu_;
  std::vector<int> fsyncs_;
};

// A FileSystem whose writable files report IsSyncThreadSafe() == false.
class GC28NotSyncThreadSafeFS : public FileSystemWrapper {
 public:
  explicit GC28NotSyncThreadSafeFS(const std::shared_ptr<FileSystem>& target)
      : FileSystemWrapper(target) {}
  static const char* kClassName() { return "GC28NotSyncThreadSafeFS"; }
  const char* Name() const override { return kClassName(); }
  IOStatus NewWritableFile(const std::string& fname,
                           const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, file_opts, result, dbg);
    if (s.ok()) {
      result->reset(new File(std::move(*result)));
    }
    return s;
  }

 private:
  class File : public FSWritableFileOwnerWrapper {
   public:
    explicit File(std::unique_ptr<FSWritableFile>&& t)
        : FSWritableFileOwnerWrapper(std::move(t)) {}
    bool IsSyncThreadSafe() const override { return false; }
  };
};
}  // namespace

TEST_P(DBWritePartitionTest, TargetedGateSyncSkipsNewerOnlyPartitions) {
  // GC28 Q1 (F2): a sync write S on partition 0, target T, fsyncs only the
  // partitions holding an unsynced record at or below T: its own and
  // partition 1 (an async record below T). Partition 2 holds only a record
  // above T, appended while S waits; partition 3 is clean. Neither is
  // fsynced, and both histograms count two partitions.
  // Neuter: claim every partition; the counts are 4 fsynced and 3 unsynced.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[1], "below", "v"));

  GC28Hold hold_s;
  hold_s.Arm("DBImpl::CompletionGate:BeforeCoalescedSync");
  Status s_status;
  port::Thread s(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  EXPECT_TRUE(hold_s.WaitHeld());
  GC28Hold above_appended;
  above_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord", nullptr,
                     /*hold=*/false);
  Status above_status;
  port::Thread above(
      [&] { above_status = Put(cf_for_partition_[2], "above", "v"); });
  EXPECT_TRUE(above_appended.WaitHits(1));
  GC28FsyncCounter fsyncs(kPartitions);
  EXPECT_OK(options.statistics->Reset());
  hold_s.Release();
  s.join();
  above.join();
  ASSERT_OK(s_status);
  ASSERT_OK(above_status);
  EXPECT_EQ((std::vector<int>{1, 1, 0, 0}), fsyncs.Get());
  HistogramData fsynced;
  options.statistics->histogramData(WAL_SYNC_FSYNCED_PARTITIONS, &fsynced);
  EXPECT_EQ(1U, fsynced.count);
  EXPECT_EQ(2U, fsynced.sum);
  HistogramData unsynced;
  options.statistics->histogramData(WAL_SYNC_UNSYNCED_PARTITIONS, &unsynced);
  EXPECT_EQ(1U, unsynced.count);
  EXPECT_EQ(2U, unsynced.sum);
  EXPECT_EQ("v", Get(cf_for_partition_[2], "above"));
}

TEST_P(DBWritePartitionTest, TargetedGateSyncCoversRotatedUnsyncedWal) {
  // GC28 Q1: an older, rotated WAL holding an unsynced record below a sync
  // write's target is claimed although its partition's current WAL holds no
  // record. The record survives a power loss that drops every unsynced
  // byte; a later async write does not.
  // Neuter: claim only each partition's current WAL; the old record is lost
  // (recovery then fails on the hole below the sync write, or loses it).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[1], "old", "v"));
  const uint64_t old_wal = dbfull()->TEST_CurrentWalNumber(1);
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_NE(old_wal, dbfull()->TEST_CurrentWalNumber(1));
  ASSERT_OK(Put(cf_for_partition_[0], "s", "v", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[2], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[1], "old"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[2], "control"));
}

TEST_P(DBWritePartitionTest, TargetedGateSyncLeavesLaterAppendUnsynced) {
  // GC28 Q1/Q2: a record appended to a claimed WAL after the claim captured
  // its size is never reported synced. S's fsync of partition 0 is held
  // after its claim; L (async) appends to partition 0 meanwhile. A later
  // sync write on partition 1, above L, fsyncs partition 0 again, and L
  // survives a power loss.
  // Neuter: a finished sync reports every record of a claimed WAL synced;
  // the later sync write skips partition 0 (its count is 0).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 2;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());

  std::atomic<std::thread::id> l_thread_id{std::thread::id()};
  {
    GC28Hold hold_fsync;
    hold_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                   GC28Partition(0));
    Status s_status;
    port::Thread s(
        [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
    EXPECT_TRUE(hold_fsync.WaitHeld());
    GC28Hold l_appended;
    l_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord",
                   GC28OnThread(&l_thread_id), /*hold=*/false);
    Status l_status;
    port::Thread l([&] {
      l_thread_id.store(std::this_thread::get_id());
      l_status = Put(cf_for_partition_[0], "late", "v");
    });
    EXPECT_TRUE(l_appended.WaitHits(1, std::chrono::seconds(10)));
    hold_fsync.Release();
    s.join();
    l.join();
    ASSERT_OK(s_status);
    ASSERT_OK(l_status);
  }
  {
    GC28FsyncCounter fsyncs(kPartitions);
    ASSERT_OK(Put(cf_for_partition_[1], "s2", "v", sync_wo));
    EXPECT_EQ(1, fsyncs.Get()[0]);
  }
  ASSERT_OK(Put(cf_for_partition_[1], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "late"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "s2"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "control"));
}

TEST_P(DBWritePartitionTest, TargetedCoalescingNeedsCoveringClaim) {
  // GC28 Q1 coalescing. (a) S1 (partition 0) has started its gate sync but
  // not claimed yet; S2 (partition 1, above S1) appends and queues behind
  // it. S1's claim skips partition 1 (S2's record is above S1's target), so
  // S2 is not covered by S1: it runs its own sync, which fsyncs partition 1.
  // (b) S4 (partition 0) appends while S3 (partition 0) is in the same
  // window; S3's claim of partition 0 covers S4's record, so S4 issues no
  // fsync. Every write survives a power loss.
  // Neuter: a group is covered by the sync running when it queued; S2
  // returns on S1 without partition 1 fsynced and is lost.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());

  auto run = [&](int first_cf, const std::string& first_key, int second_cf,
                 const std::string& second_key) {
    GC28Hold hold_first;
    hold_first.Arm("DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex");
    Status first_status;
    port::Thread first(
        [&] { first_status = Put(first_cf, first_key, "v", sync_wo); });
    EXPECT_TRUE(hold_first.WaitHeld());
    GC28Hold queued;
    queued.Arm("DBImpl::RunCoalescedGateSync:BeforeWait", nullptr,
               /*hold=*/false);
    Status second_status;
    port::Thread second(
        [&] { second_status = Put(second_cf, second_key, "v", sync_wo); });
    EXPECT_TRUE(queued.WaitHits(1));
    GC28FsyncCounter fsyncs(kPartitions);
    hold_first.Release();
    first.join();
    second.join();
    EXPECT_OK(first_status);
    EXPECT_OK(second_status);
    return fsyncs.Get();
  };
  // (a) S1 fsyncs partition 0, S2 its own partition 1.
  EXPECT_EQ((std::vector<int>{1, 1, 0, 0}),
            run(cf_for_partition_[0], "s1", cf_for_partition_[1], "s2"));
  // (b) S3 fsyncs partition 0 once; S4 adds nothing.
  EXPECT_EQ((std::vector<int>{1, 0, 0, 0}),
            run(cf_for_partition_[0], "s3", cf_for_partition_[0], "s4"));
  ASSERT_OK(Put(cf_for_partition_[2], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s1"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "s2"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s3"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s4"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[2], "control"));
}

TEST_P(DBWritePartitionTest, TargetedZeroCountSyncSyncsItsOwnFile) {
  // GC28 Q1, G3-D-1: a sync Put on partition 1 (target T), then a
  // LogData-only sync write, which routes to partition 0 and consumes no
  // sequence: its record is stamped T + 1 but its target is T. It is the only
  // unsynced record in partition 0's WAL, which must still be fsynced: the
  // record survives a power loss, while an unsynced control record written
  // after it is lost.
  // Neuter: claim a WAL only for an unsynced record at or below T;
  // partition 0 is skipped and the record is lost.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 2;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[1], "key", "value", sync_wo));
  const SequenceNumber last = db_->GetLatestSequenceNumber();

  auto log_data_only = [&](const WriteOptions& wo, const std::string& blob) {
    WriteBatch batch;
    EXPECT_OK(batch.PutLogData(blob));
    return db_->Write(wo, &batch);
  };
  {
    GC28FsyncCounter fsyncs(kPartitions);
    ASSERT_OK(log_data_only(sync_wo, "zero_count_blob"));
    EXPECT_EQ(1, fsyncs.Get()[0]);
  }
  EXPECT_EQ(last, db_->GetLatestSequenceNumber());
  ASSERT_OK(log_data_only(WriteOptions(), "control_blob"));

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
      found_record |=
          record.ToString().find("zero_count_blob") != std::string::npos;
      found_control |=
          record.ToString().find("control_blob") != std::string::npos;
    }
  }
  EXPECT_TRUE(found_record);
  EXPECT_FALSE(found_control);
}

TEST_P(DBWritePartitionTest, TargetedSyncInvisibleUntilNeededFsyncDone) {
  // GC28 Q1, D-9: partition 1 holds an async record below sync write S
  // (partition 0). With partition 1's fsync held, S stays invisible: Get()
  // misses it and GetLatestSequenceNumber() does not move, until the fsync
  // is released.
  // Neuter: claim only the writer's own partition; partition 1's fsync is
  // never reached and S becomes visible without it.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[1], "low", "v"));
  const SequenceNumber before = db_->GetLatestSequenceNumber();

  GC28Hold hold_fsync;
  hold_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                 GC28Partition(1));
  Status s_status;
  port::Thread s(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  EXPECT_TRUE(hold_fsync.WaitHeld(std::chrono::seconds(10)));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ(before, db_->GetLatestSequenceNumber());
  hold_fsync.Release();
  s.join();
  ASSERT_OK(s_status);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
}

TEST_P(DBWritePartitionTest,
       TargetedManualFlushSyncLeavesNewerBuffersBuffered) {
  // GC28 Q1, D-F2d: under manual_wal_flush a sync write flushes only the
  // claimed partitions' buffers. A buffered record below S on partition 1
  // survives a power loss; partition 2's buffered record above S stays in
  // its buffer.
  // Neuter: flush every partition's buffer; partition 2's buffer is empty.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(dbfull()->FlushWAL(/*sync=*/true));
  ASSERT_OK(Put(cf_for_partition_[1], "below", "v"));

  GC28Hold hold_s;
  hold_s.Arm("DBImpl::CompletionGate:BeforeCoalescedSync");
  Status s_status;
  port::Thread s(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  EXPECT_TRUE(hold_s.WaitHeld());
  GC28Hold above_appended;
  above_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord", nullptr,
                     /*hold=*/false);
  Status above_status;
  port::Thread above(
      [&] { above_status = Put(cf_for_partition_[2], "above", "v"); });
  EXPECT_TRUE(above_appended.WaitHits(1));
  hold_s.Release();
  s.join();
  above.join();
  ASSERT_OK(s_status);
  ASSERT_OK(above_status);
  EXPECT_TRUE(dbfull()->TEST_WalBufferIsEmpty(1));
  EXPECT_FALSE(dbfull()->TEST_WalBufferIsEmpty(2));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[1], "below"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
}

TEST_P(DBWritePartitionTest, FlushWalTakesOnePartitionMutexAtATime) {
  // GC28 Q1, D-F2d: FlushWAL() writes each partition's buffer holding only
  // that partition's WAL mutex. With FlushWAL() held inside partition 0's
  // buffer write, an async write on partition 3 appends and returns.
  // Neuter: hold every partition's WAL mutex across the loop; the write
  // never returns and RunWithin aborts.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.manual_wal_flush = true;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[0], "a", "v"));
  ASSERT_OK(Put(cf_for_partition_[1], "b", "v"));

  GC28Hold hold_flush;
  hold_flush.Arm("DBImpl::FlushWAL:BeforeWriteBuffer", GC28Partition(0));
  Status flush_status;
  port::Thread flush(
      [&] { flush_status = dbfull()->FlushWAL(/*sync=*/false); });
  EXPECT_TRUE(hold_flush.WaitHeld());
  EXPECT_OK(RunWithin(std::chrono::seconds(10),
                      [&] { return Put(cf_for_partition_[3], "c", "v"); }));
  hold_flush.Release();
  flush.join();
  ASSERT_OK(flush_status);
  EXPECT_EQ("v", Get(cf_for_partition_[3], "c"));
}

TEST_P(DBWritePartitionTest, TargetedSyncRetiresSkippedPartitionWals) {
  // GC28 Q1: a closed WAL whose records are all synced still leaves
  // logs_[p] although no sync write needs its records. Partition 3 is
  // written once; then each round rotates and runs a sync write on
  // partition 0. Every retiring WAL of partition 3 gets an unsynced progress
  // record and every new one an unsynced header, so the next gate sync
  // claims partition 3 and retires the fully synced closed WAL
  // (background_close_inactive_wals), as the base's all-partition sync did.
  // Neuter: claim a WAL only for an unsynced sequenced record; partition 3
  // is never claimed again and logs_[3] grows by one WAL per round.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = IndexedWalOptions(CurrentOptions());
  options.background_close_inactive_wals = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[3], "a", "v"));
  for (int i = 0; i < 4; ++i) {
    ASSERT_OK(dbfull()->TEST_SwitchMemtable());
    ASSERT_OK(Put(cf_for_partition_[0], "s" + std::to_string(i), "v", sync_wo));
  }
  EXPECT_EQ(1U, dbfull()->TEST_WalSyncMarks(3).size());
  ASSERT_OK(db_->SyncWAL());
  EXPECT_EQ(1U, dbfull()->TEST_WalSyncMarks(3).size());
  EXPECT_EQ("v", Get(cf_for_partition_[3], "a"));
}

TEST_P(DBWritePartitionTest,
       ManualFlushAppendProceedsWhileOtherPartitionFsyncHeld) {
  // GC28 Q2 (FU3): under manual_wal_flush no partition mutex is held across
  // a gate fsync. With partition 0's fsync held, an async write W on another
  // partition completes its append; it is neither visible nor returned until
  // the sync write publishes, then both return OK.
  // Neuter: hold every WAL mutex across the gate's fsync loop (the base's
  // LockWalWriteMutexes() branch); W never appends.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.manual_wal_flush = true;
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
    ASSERT_OK(dbfull()->FlushWAL(/*sync=*/true));
    const int cf_w = cf_for_partition_[partitions - 1];

    GC28Hold hold_fsync;
    hold_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                   GC28Partition(0));
    Status s_status;
    port::Thread s(
        [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
    EXPECT_TRUE(hold_fsync.WaitHeld());
    std::atomic<std::thread::id> w_thread_id{std::thread::id()};
    GC28Hold w_appended;
    w_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord",
                   GC28OnThread(&w_thread_id), /*hold=*/false);
    std::atomic<bool> w_returned{false};
    Status w_status;
    port::Thread w([&] {
      w_thread_id.store(std::this_thread::get_id());
      w_status = Put(cf_w, "w", "v");
      w_returned.store(true);
    });
    EXPECT_TRUE(w_appended.WaitHits(1, std::chrono::seconds(10)));
    EXPECT_FALSE(w_returned.load());
    EXPECT_EQ("NOT_FOUND", Get(cf_w, "w"));
    hold_fsync.Release();
    s.join();
    w.join();
    ASSERT_OK(s_status);
    ASSERT_OK(w_status);
    EXPECT_EQ("v", Get(cf_w, "w"));
  }
}

TEST_P(DBWritePartitionTest,
       AsyncAppendOnSyncingPartitionProceedsWhileItsFsyncHeld) {
  // GC28 Q2 (FU3, D-FU3a (ii)): replaces GC18 Q9's
  // AsyncOnSyncingPartitionWaitsForItsFsync. With sync write B held inside
  // its fsync of partition 1, async E on partition 1 appends (it no longer
  // waits for that fsync), but returns only once B publishes (D-9). Default
  // config and manual_wal_flush.
  // Neuter: hold the partition's WAL mutex across its fsync; E never
  // appends while B is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  for (const bool manual_flush : {false, true}) {
    SCOPED_TRACE("manual_wal_flush=" + std::to_string(manual_flush));
    Options options = CurrentOptions();
    options.manual_wal_flush = manual_flush;
    options.avoid_flush_during_shutdown = true;
    DestroyAndReopen(options);
    OpenWithPartitions(kPartitions, options);
    const int cf_q = cf_for_partition_[1];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf_q, "s0", "v", sync_wo));

    GC28Hold hold_fsync;
    hold_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                   GC28Partition(1));
    Status b_status;
    port::Thread b([&] { b_status = Put(cf_q, "key_b", "value_b", sync_wo); });
    EXPECT_TRUE(hold_fsync.WaitHeld());
    std::atomic<std::thread::id> e_thread_id{std::thread::id()};
    GC28Hold e_appended;
    e_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord",
                   GC28OnThread(&e_thread_id), /*hold=*/false);
    std::atomic<bool> e_returned{false};
    Status e_status;
    port::Thread e([&] {
      e_thread_id.store(std::this_thread::get_id());
      e_status = Put(cf_q, "key_e", "value_e");
      e_returned.store(true);
    });
    EXPECT_TRUE(e_appended.WaitHits(1, std::chrono::seconds(10)));
    EXPECT_FALSE(e_returned.load());
    EXPECT_EQ("NOT_FOUND", Get(cf_q, "key_e"));
    hold_fsync.Release();
    EXPECT_OK(RunWithin(std::chrono::seconds(30), [&] {
      e.join();
      return e_status;
    }));
    b.join();
    ASSERT_OK(b_status);
    EXPECT_EQ("value_b", Get(cf_q, "key_b"));
    EXPECT_EQ("value_e", Get(cf_q, "key_e"));
  }
}

TEST_P(DBWritePartitionTest,
       NonSyncThreadSafeWalHoldsOnePartitionMutexAcrossFsync) {
  // GC28 Q2 fallback: when the WAL file is not IsSyncThreadSafe(), the gate
  // holds a partition's WAL mutex across that partition's fsync, one
  // partition at a time. With partition 0's fsync held, an append on
  // partition 0 waits, while an append on partition 1 proceeds (never all
  // N).
  // Neuter: fsync without the mutex regardless; the append on partition 0
  // lands while its fsync is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 2;
  std::shared_ptr<FileSystem> fs =
      std::make_shared<GC28NotSyncThreadSafeFS>(env_->GetFileSystem());
  std::unique_ptr<Env> env(NewCompositeEnv(fs));
  Defer close_before_env([this] { Close(); });
  Options options = CurrentOptions();
  options.env = env.get();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));

  GC28Hold hold_fsync;
  hold_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                 GC28Partition(0));
  Status s_status;
  port::Thread s(
      [&] { s_status = Put(cf_for_partition_[0], "s", "v", sync_wo); });
  EXPECT_TRUE(hold_fsync.WaitHeld());
  GC28Hold appended;
  appended.Arm("DBImpl::WriteToWAL:AfterAddRecord", nullptr, /*hold=*/false);
  Status q_status;
  port::Thread q([&] { q_status = Put(cf_for_partition_[0], "q", "v"); });
  Status r_status;
  port::Thread r([&] { r_status = Put(cf_for_partition_[1], "r", "v"); });
  // r appends; q, which needs partition 0's WAL mutex, does not.
  EXPECT_TRUE(appended.WaitHits(1, std::chrono::seconds(10)));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(1, appended.Hits());
  hold_fsync.Release();
  s.join();
  q.join();
  r.join();
  ASSERT_OK(s_status);
  ASSERT_OK(q_status);
  ASSERT_OK(r_status);
}

TEST_P(DBWritePartitionTest,
       GateSyncFlushesBufferedLowerRecordsBeforeUnlockedFsync) {
  // GC28 Q2 step 1: under manual_wal_flush the gate writes each claimed
  // partition's buffer to the OS, under its mutex, before the unlocked
  // fsync. Buffered async records on partitions 1 and 2, below sync write S
  // on partition 0, survive a power loss; a later buffered write does not.
  // Neuter: skip the buffer write; the buffered records (S's included) are
  // lost.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(dbfull()->FlushWAL(/*sync=*/true));
  ASSERT_OK(Put(cf_for_partition_[1], "b1", "v"));
  ASSERT_OK(Put(cf_for_partition_[2], "b2", "v"));
  ASSERT_OK(Put(cf_for_partition_[0], "s", "v", sync_wo));
  ASSERT_OK(Put(cf_for_partition_[3], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[1], "b1"));
  EXPECT_EQ("v", Get(cf_for_partition_[2], "b2"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[3], "control"));
}

TEST_P(DBWritePartitionTest, GateSyncRacesWithWalOperations) {
  // GC28 Q2: FlushWAL(), FlushWAL(true), SyncWAL(), LockWAL()/UnlockWAL()
  // and a rotation, each started while a gate sync is held between its
  // claim and its fsync, or between its fsync and its marks. Every write
  // returns OK; after a power loss a reopen in kAbsoluteConsistency reads
  // every acknowledged write, and the MANIFEST's synced sizes
  // (track_and_verify_wals_in_manifest) never exceed what reached the disk.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));

  const std::vector<std::function<Status()>> ops = {
      [&] { return dbfull()->FlushWAL(/*sync=*/false); },
      [&] { return dbfull()->FlushWAL(/*sync=*/true); },
      [&] { return db_->SyncWAL(); },
      [&] {
        Status s = db_->LockWAL();
        if (s.ok()) {
          s = db_->UnlockWAL();
        }
        return s;
      },
      [&] { return dbfull()->TEST_SwitchMemtable(); },
  };
  const std::vector<std::string> points = {
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync",
  };
  int round = 0;
  for (const std::string& point : points) {
    for (const auto& op : ops) {
      SCOPED_TRACE("round=" + std::to_string(round));
      const std::string suffix = std::to_string(round++);
      ASSERT_OK(Put(cf_for_partition_[1], "pre" + suffix, "v"));
      GC28Hold hold;
      hold.Arm(point);
      Status s_status;
      port::Thread s([&] {
        s_status = Put(cf_for_partition_[0], "s" + suffix, "v", sync_wo);
      });
      EXPECT_TRUE(hold.WaitHeld());
      Status op_status;
      port::Thread o([&] { op_status = op(); });
      // Room for the operation to run into the held sync.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      hold.Release();
      s.join();
      o.join();
      ASSERT_OK(s_status);
      ASSERT_OK(op_status);
    }
  }
  ASSERT_OK(Put(cf_for_partition_[2], "control", "v"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
  ReopenWithPartitions(kPartitions, options);
  for (int i = 0; i < round; ++i) {
    EXPECT_EQ("v", Get(cf_for_partition_[1], "pre" + std::to_string(i)));
    EXPECT_EQ("v", Get(cf_for_partition_[0], "s" + std::to_string(i)));
  }
}

// FaultInjectionTestEnv's files report IsSyncThreadSafe() but update their
// state without a lock, so a gate sync's unlocked fsync racing a FlushWAL()
// or an append to the same file would race inside the test file itself.
// This env serializes each file's calls; the DB still fsyncs holding no
// mutex of its own.
class SerializedFileEnv : public EnvWrapper {
 public:
  explicit SerializedFileEnv(Env* target) : EnvWrapper(target) {}
  static const char* kClassName() { return "SerializedFileEnv"; }
  const char* Name() const override { return kClassName(); }
  Status NewWritableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result,
                         const EnvOptions& options) override {
    Status s = target()->NewWritableFile(fname, result, options);
    if (s.ok()) {
      result->reset(new File(std::move(*result)));
    }
    return s;
  }
  Status ReopenWritableFile(const std::string& fname,
                            std::unique_ptr<WritableFile>* result,
                            const EnvOptions& options) override {
    Status s = target()->ReopenWritableFile(fname, result, options);
    if (s.ok()) {
      result->reset(new File(std::move(*result)));
    }
    return s;
  }
  Status ReuseWritableFile(const std::string& fname,
                           const std::string& old_fname,
                           std::unique_ptr<WritableFile>* result,
                           const EnvOptions& options) override {
    Status s = target()->ReuseWritableFile(fname, old_fname, result, options);
    if (s.ok()) {
      result->reset(new File(std::move(*result)));
    }
    return s;
  }

 private:
  class File : public WritableFileWrapper {
   public:
    explicit File(std::unique_ptr<WritableFile>&& file)
        : WritableFileWrapper(file.get()), file_(std::move(file)) {}
    Status Append(const Slice& data) override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Append(data);
    }
    Status Append(const Slice& data,
                  const DataVerificationInfo& verification_info) override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Append(data, verification_info);
    }
    Status PositionedAppend(const Slice& data, uint64_t offset) override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::PositionedAppend(data, offset);
    }
    Status PositionedAppend(
        const Slice& data, uint64_t offset,
        const DataVerificationInfo& verification_info) override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::PositionedAppend(data, offset,
                                                   verification_info);
    }
    Status Truncate(uint64_t size) override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Truncate(size);
    }
    Status Close() override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Close();
    }
    Status Flush() override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Flush();
    }
    Status Sync() override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Sync();
    }
    Status Fsync() override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::Fsync();
    }
    uint64_t GetFileSize() override {
      std::lock_guard<std::mutex> lock(mu_);
      return WritableFileWrapper::GetFileSize();
    }

   private:
    std::mutex mu_;
    std::unique_ptr<WritableFile> file_;
  };
};

TEST_P(DBWritePartitionTest, UnlockedFsyncRacesWithSyncCallersAndRotation) {
  // FlushWAL(), FlushWAL(true), SyncWAL(), LockWAL()/UnlockWAL() and a
  // rotation, each started while a gate sync is held between its claim and
  // its fsync, or between its fsync and its marks, with and without
  // manual_wal_flush. Each scenario runs on a fresh DB and ends in its own
  // power loss: a reopen in kAbsoluteConsistency reads every acknowledged
  // write, and the control record, written unsynced after the race, is
  // gone, so no sync marked or covered more than reached the disk. Ported
  // from Muse's closed GC28 candidate (one power loss per scenario, both
  // flush modes, the control record asserted).
  // Neuter: mark the claimed WALs synced without writing their buffers
  // first (the claim's WriteBuffer skipped under manual_wal_flush); the
  // acknowledged sync write is lost.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
    return;
  }
  constexpr size_t kPartitions = 4;
  const std::vector<std::string> racers = {"flush", "flush_sync", "sync_wal",
                                           "lock_wal", "rotation"};
  const std::vector<std::string> points = {
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync",
  };
  int scenario = 0;
  for (const bool manual_wal_flush : {true, false}) {
    for (const std::string& racer : racers) {
      for (const std::string& point : points) {
        const std::string tag = std::to_string(scenario++);
        SCOPED_TRACE("manual_wal_flush=" + std::to_string(manual_wal_flush) +
                     " racer=" + racer + " point=" + point);
        std::unique_ptr<FaultInjectionTestEnv> fault_env(
            new FaultInjectionTestEnv(env_));
        SerializedFileEnv serialized_env(fault_env.get());
        Defer close_before_env([this] { Close(); });
        Options options = IndexedWalOptions(CurrentOptions());
        options.env = &serialized_env;
        options.manual_wal_flush = manual_wal_flush;
        options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
        DestroyAndReopen(options);
        OpenWithPartitions(kPartitions, options);
        ASSERT_OK(Put(cf_for_partition_[1], "a" + tag, "v"));

        GC28Hold hold;
        // Between claim and fsync: partition 0's, the sync write's own.
        hold.Arm(point, point == points[0] ? GC28Partition(0) : nullptr);
        WriteOptions sync_wo;
        sync_wo.sync = true;
        Status s_status;
        port::Thread s([&] {
          s_status = Put(cf_for_partition_[0], "s" + tag, "v", sync_wo);
        });
        EXPECT_TRUE(hold.WaitHeld());
        Status racer_status;
        port::Thread r([&] {
          if (racer == "flush") {
            racer_status = db_->FlushWAL(/*sync=*/false);
          } else if (racer == "flush_sync") {
            racer_status = db_->FlushWAL(/*sync=*/true);
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
        // Room for the racer to run into the held sync.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        hold.Release();
        s.join();
        r.join();
        ASSERT_OK(s_status);
        ASSERT_OK(racer_status);

        ASSERT_OK(Put(cf_for_partition_[1], "control" + tag, "v"));
        Close();
        ASSERT_OK(fault_env->DropUnsyncedFileData());
        ReopenWithPartitions(kPartitions, options);
        EXPECT_EQ("v", Get(cf_for_partition_[1], "a" + tag));
        EXPECT_EQ("v", Get(cf_for_partition_[0], "s" + tag));
        EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "control" + tag));
        Close();
      }
    }
  }
}

TEST_P(DBWritePartitionTest, SyncWriteSyncsOnceAfterItsWait) {
  // GC28 Q3 (AMB-M4-21, D-GC28b (i); the idea of Muse's GC25
  // SyncWriteSyncsOnceAfterWait): a sync write first waits until every
  // record at or below it is written, then runs one gate sync. (a)
  // Uncontended sync writes run one gate sync each. (b) B (sync) behind A
  // (async, held before its append) runs no gate sync while parked and
  // stays invisible; once A lands it runs exactly one and returns.
  // Replaces SyncWriteSyncsAgainOnlyAfterWait, which pinned the removed
  // fsync-wait-fsync order.
  // Neuter: sync before the wait (the base's order); the parked-phase count
  // is 1.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_a, "warm", "v", sync_wo));

  // A gate sync round, whichever path runs it.
  std::atomic<int> rounds{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex",
      [&](void*) { ++rounds; });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::SyncWAL:Begin",
                                        [&](void*) { ++rounds; });
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "uncontended" + std::to_string(i), "v", sync_wo));
  }
  EXPECT_EQ(3, rounds.load());

  rounds.store(0);
  GC28Hold hold_a;
  hold_a.Arm("DBImpl::WriteToWAL:log_entry", [](void* arg) {
    return static_cast<Slice*>(arg)->ends_with("value_a");
  });
  Status a_status;
  port::Thread a([&] { a_status = Put(cf_a, "key_a", "value_a"); });
  EXPECT_TRUE(hold_a.WaitHeld());
  Status b_status;
  port::Thread b([&] { b_status = Put(cf_b, "key_b", "value_b", sync_wo); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_EQ(0, rounds.load());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  hold_a.Release();
  a.join();
  b.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWAL:Begin");
  ASSERT_OK(a_status);
  ASSERT_OK(b_status);
  EXPECT_EQ(1, rounds.load());
  EXPECT_EQ("value_b", Get(cf_b, "key_b"));
}

TEST_P(DBWritePartitionTest, GateSyncFsyncsClaimedPartitionsConcurrently) {
  // GC28 Q4 (b), D-GC28c (ii): S1 (partition 1) claims partitions 1 and 2
  // (an async record below it); S2 (partition 3) is parked on the
  // coalescer. With the first claimed partition's fsync held, the other
  // one's fsync starts: S2 runs it while it waits. No thread is created.
  // Neuter: the serial loop; the second fsync never starts while the first
  // is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[2], "low", "v"));

  GC28Hold hold_start;
  hold_start.Arm("DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  Status s1_status;
  port::Thread s1(
      [&] { s1_status = Put(cf_for_partition_[1], "s1", "v", sync_wo); });
  EXPECT_TRUE(hold_start.WaitHeld());
  GC28Hold queued;
  queued.Arm("DBImpl::RunCoalescedGateSync:BeforeWait", nullptr,
             /*hold=*/false);
  Status s2_status;
  port::Thread s2(
      [&] { s2_status = Put(cf_for_partition_[3], "s2", "v", sync_wo); });
  EXPECT_TRUE(queued.WaitHits(1));
  GC28Hold hold_first;
  hold_first.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                 [](void* arg) {
                   const size_t p = *static_cast<size_t*>(arg);
                   return p == 1 || p == 2;
                 });
  hold_start.Release();
  EXPECT_TRUE(hold_first.WaitHeld());
  EXPECT_TRUE(hold_first.WaitHits(2, std::chrono::seconds(10)));
  hold_first.Release();
  s1.join();
  s2.join();
  ASSERT_OK(s1_status);
  ASSERT_OK(s2_status);
}

TEST_P(DBWritePartitionTest, ParkedWriterHelpsGateSyncButReturnsAfterIt) {
  // S (sync, partition 1) claims partitions 1 and 2 (an async record below
  // it); W (async, partition 3) is allocated above S and parks for
  // publication. With S's own first fsync held on S's thread, W, parked,
  // fsyncs the other claimed partition on its own thread
  // (HelpRunningGateSync), then parks again: a write ordered after a pending
  // sync write becomes visible, and returns, only after that sync. Neither
  // is visible until S's held fsync is released; then both return.
  // Neuters: (a) no helping from WaitForWalDurability; nothing runs on W's
  // thread. (b) return from WaitForWalDurability after helping; W never
  // parks again and returns before S's sync completed.
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
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[2], "low", "v"));

  GC28Hold hold_start;
  hold_start.Arm("DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  std::atomic<std::thread::id> s_thread_id{std::thread::id()};
  Status s_status;
  port::Thread s([&] {
    s_thread_id.store(std::this_thread::get_id());
    s_status = Put(cf_for_partition_[1], "s", "v", sync_wo);
  });
  EXPECT_TRUE(hold_start.WaitHeld());
  std::atomic<std::thread::id> w_thread_id{std::thread::id()};
  GC28Hold w_parks;
  w_parks.Arm("DBImpl::WaitForWalDurability:BeforePark",
              GC28OnThread(&w_thread_id), /*hold=*/false);
  Status w_status;
  std::atomic<bool> w_done{false};
  port::Thread w([&] {
    w_thread_id.store(std::this_thread::get_id());
    w_status = Put(cf_for_partition_[3], "w", "v");
    w_done.store(true);
  });
  EXPECT_TRUE(w_parks.WaitHits(1));
  // S's thread runs the sync; its first claimed partition's fsync is held.
  // Fsyncs on W's thread are counted.
  std::atomic<int> w_fsyncs{0};
  GC28Hold hold_s_fsync;
  hold_s_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                   [&](void*) {
                     if (std::this_thread::get_id() == w_thread_id.load()) {
                       ++w_fsyncs;
                       return false;
                     }
                     return std::this_thread::get_id() == s_thread_id.load();
                   });
  hold_start.Release();
  EXPECT_TRUE(hold_s_fsync.WaitHeld());
  EXPECT_TRUE(w_parks.WaitHits(2, std::chrono::seconds(10)))
      << "the parked writer did not help and park again";
  EXPECT_EQ(1, w_fsyncs.load());
  EXPECT_FALSE(w_done.load());
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "s"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[3], "w"));
  hold_s_fsync.Release();
  s.join();
  w.join();
  ASSERT_OK(s_status);
  ASSERT_OK(w_status);
  EXPECT_EQ(1, w_fsyncs.load());
  EXPECT_EQ("v", Get(cf_for_partition_[1], "s"));
  EXPECT_EQ("v", Get(cf_for_partition_[3], "w"));
}

TEST_P(DBWritePartitionTest, ConcurrentGateSyncFailureFailsTheGroup) {
  // GC28 Q4 (b): partitions 1 and 2 are fsynced concurrently (the first
  // held until the second starts, on S2, parked on the coalescer); the
  // second one's fsync fails. Sync write S gets the IOError and stays
  // invisible, before and after Resume(); the DB stops; a crash image
  // reopens with every acknowledged write.
  // Neuter: ignore a helper's fsync status; S returns OK.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  Options options = IndexedWalOptions(CurrentOptions());
  options.env = fault_env.get();
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 0;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "s0", "v", sync_wo));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(cf_for_partition_[2], "low", "v"));

  std::mutex mu;
  std::condition_variable cv;
  int arrivals = 0;
  bool concurrent = false;
  size_t fail_partition = kPartitions;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        const size_t p = *static_cast<size_t*>(arg);
        if (p != 1 && p != 2) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (++arrivals == 1) {
          concurrent = cv.wait_for(lock, std::chrono::seconds(10),
                                   [&] { return arrivals >= 2; });
        } else if (arrivals == 2) {
          fail_partition = p;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:AfterSyncPartition", [&](void* arg) {
        auto* partition_and_status =
            static_cast<std::pair<size_t, IOStatus*>*>(arg);
        std::lock_guard<std::mutex> lock(mu);
        if (partition_and_status->first == fail_partition) {
          IOStatus error = IOStatus::IOError("injected partition fsync error");
          error.SetRetryable(true);
          *partition_and_status->second = error;
          fail_partition = kPartitions;
        }
      });
  GC28Hold hold_start;
  hold_start.Arm("DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  Status s_status;
  port::Thread s(
      [&] { s_status = Put(cf_for_partition_[1], "s", "v", sync_wo); });
  EXPECT_TRUE(hold_start.WaitHeld());
  GC28Hold queued;
  queued.Arm("DBImpl::RunCoalescedGateSync:BeforeWait", nullptr,
             /*hold=*/false);
  Status s2_status;
  port::Thread s2(
      [&] { s2_status = Put(cf_for_partition_[3], "s2", "v", sync_wo); });
  EXPECT_TRUE(queued.WaitHits(1));
  hold_start.Release();
  s.join();
  s2.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:AfterSyncPartition");
  EXPECT_EQ(2, arrivals);
  EXPECT_TRUE(concurrent);
  ASSERT_TRUE(s_status.IsIOError()) << s_status.ToString();
  // S2 queued while S's sync ran, so it shares the failure.
  ASSERT_TRUE(s2_status.IsIOError()) << s2_status.ToString();
  ASSERT_NOK(dbfull()->TEST_GetBGError());
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "s"));
  dbfull()->Resume().PermitUncheckedError();
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "s"));

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ReopenWithPartitions(kPartitions, options);
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s0"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "s"));
}

TEST_P(DBWritePartitionTest, JoinWaitTimerOffAtDefaultStatsLevel) {
  // GC28 Q6 (F006, D-GC28f (i)): at the default stats level
  // (kExceptDetailedTimers) the join-wait timer reads no clock and records
  // nothing, N=1 included; above it every write records one sample.
  // Neuter: the ungated StopWatch; the default level records 3.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  ASSERT_EQ(StatsLevel::kExceptDetailedTimers,
            options.statistics->get_stats_level());
  OpenWithPartitions(1, options);
  auto join_wait_count = [&] {
    HistogramData data;
    options.statistics->histogramData(JOIN_BATCH_GROUP_WAIT_MICROS, &data);
    return data.count;
  };
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("k" + std::to_string(i), "v"));
  }
  EXPECT_EQ(0U, join_wait_count());
  options.statistics->set_stats_level(StatsLevel::kExceptTimeForMutex);
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put("k" + std::to_string(i), "v"));
  }
  EXPECT_EQ(3U, join_wait_count());
}

TEST_P(DBWritePartitionTest, FrontierLockTakenOnlyForAllocation) {
  // GC28 Q8 (R6): at N=8, non-sync writes from 16 threads take
  // sequence_wal_index_mutex_ exactly once per write group, to allocate;
  // completion, publication and waiting take it never.
  // Neuter: take it again in PublishAppliedPrefix; the count exceeds the
  // number of groups.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 8;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(options.statistics->Reset());
  const uint64_t takes_before = dbfull()->TEST_SequenceWalIndexMutexTakes();
  std::vector<port::Thread> threads;
  for (int t = 0; t < 16; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 50; ++i) {
        ASSERT_OK(Put(cf_for_partition_[(t + i) % kPartitions],
                      "k" + std::to_string(t) + "_" + std::to_string(i), "v"));
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  HistogramData groups;
  options.statistics->histogramData(WRITERS_PER_WRITE_GROUP, &groups);
  EXPECT_EQ(16U * 50U, groups.sum);
  EXPECT_EQ(groups.count,
            dbfull()->TEST_SequenceWalIndexMutexTakes() - takes_before);
}

TEST_P(DBWritePartitionTest, FrontierNeverPassesAllocatedRange) {
  // GC28 Q8 ordering: X (partition 1) is held right after its allocation
  // section, before its append; Y1..Y3 (partition 2) allocate above it,
  // append, publish their ranges and park. Neither GetLatestSequenceNumber()
  // nor the WAL safe point passes X's range until X lands.
  // Neuter: register X's range after releasing the allocation lock; the Ys'
  // completions move both frontiers past it.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 3;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[0], "base", "v"));
  const SequenceNumber x_low = db_->GetLatestSequenceNumber() + 1;

  std::atomic<std::thread::id> x_thread_id{std::thread::id()};
  GC28Hold hold_x;
  hold_x.Arm("DBImpl::AllocateSequenceAndWALIndex:Allocated",
             GC28OnThread(&x_thread_id));
  Status x_status;
  port::Thread x([&] {
    x_thread_id.store(std::this_thread::get_id());
    x_status = Put(cf_for_partition_[1], "x", "v");
  });
  EXPECT_TRUE(hold_x.WaitHeld());
  std::vector<Status> y_statuses(3);
  std::vector<port::Thread> ys;
  for (int i = 0; i < 3; ++i) {
    ys.emplace_back([&, i] {
      y_statuses[i] = Put(cf_for_partition_[2], "y" + std::to_string(i), "v");
    });
  }
  // Releases X and joins every writer on every exit.
  Defer join_writers([&] {
    hold_x.Release();
    if (x.joinable()) {
      x.join();
    }
    for (auto& y : ys) {
      if (y.joinable()) {
        y.join();
      }
    }
  });
  // A Y group parks only after its append completed and it published.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (dbfull()->TEST_WalDurabilityWaiters() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_GT(dbfull()->TEST_WalDurabilityWaiters(), 0U);
  // Every Y's append completed on partition 2 (they take x_low + 1 through
  // x_low + 3), so nothing but X's range can hold either frontier.
  while (dbfull()->TEST_WalDurabilityDurableUpto(2) < x_low + 3 ||
         dbfull()->TEST_WalDurabilityPendingLow(2) !=
             WalDurabilityTracker::kIdle) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "partition 2 appends never completed";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_LT(db_->GetLatestSequenceNumber(), x_low);
  EXPECT_LT(dbfull()->WalDurabilitySafePoint(), x_low);
  hold_x.Release();
  x.join();
  for (auto& y : ys) {
    y.join();
  }
  ASSERT_OK(x_status);
  for (const Status& s : y_statuses) {
    ASSERT_OK(s);
  }
  EXPECT_EQ("v", Get(cf_for_partition_[1], "x"));
  EXPECT_EQ("v", Get(cf_for_partition_[2], "y2"));
}

TEST_P(DBWritePartitionTest, TargetedWakeLeavesUncoveredWaiterParked) {
  // GC28 Q8 targeted wake: W1 (target a + 1) and W2 (target a + 3) park
  // behind A (a, held before its append) and C (a + 2, held likewise).
  // Releasing A covers W1 only: W1 wakes and returns, while W2 is never
  // woken. Releasing C then returns W2.
  // Neuter: wake every parked waiter on every frontier advance; W2 wakes.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(Put(cf_for_partition_[0], "base", "v"));

  // A and C are each held before their append until released.
  std::mutex mu;
  std::condition_variable cv;
  bool a_held = false;
  bool c_held = false;
  bool release_a = false;
  bool release_c = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        const Slice* entry = static_cast<Slice*>(arg);
        const bool is_a = entry->ends_with("value_a");
        if (!is_a && !entry->ends_with("value_c")) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        (is_a ? a_held : c_held) = true;
        cv.notify_all();
        cv.wait(lock, [&] { return is_a ? release_a : release_c; });
      });
  auto wait_held = [&](const bool* held) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(30), [&] { return *held; });
  };
  auto release = [&](bool* flag) {
    std::lock_guard<std::mutex> lock(mu);
    *flag = true;
    cv.notify_all();
  };
  std::atomic<std::thread::id> w2_thread_id{std::thread::id()};
  GC28Hold w2_woken;
  w2_woken.Arm("DBImpl::WaitForWalDurability:AfterWake",
               GC28OnThread(&w2_thread_id), /*hold=*/false);

  Status a_status;
  port::Thread a(
      [&] { a_status = Put(cf_for_partition_[0], "key_a", "value_a"); });
  EXPECT_TRUE(wait_held(&a_held));
  Status w1_status;
  port::Thread w1([&] { w1_status = Put(cf_for_partition_[1], "w1", "v"); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  Status c_status;
  port::Thread c(
      [&] { c_status = Put(cf_for_partition_[2], "key_c", "value_c"); });
  EXPECT_TRUE(wait_held(&c_held));
  Status w2_status;
  port::Thread w2([&] {
    w2_thread_id.store(std::this_thread::get_id());
    w2_status = Put(cf_for_partition_[3], "w2", "v");
  });
  EXPECT_TRUE(WaitForDurabilityWaiters(2));
  release(&release_a);
  EXPECT_OK(RunWithin(std::chrono::seconds(30), [&] {
    a.join();
    w1.join();
    return w1_status;
  }));
  EXPECT_OK(a_status);
  EXPECT_EQ(0, w2_woken.Hits());
  EXPECT_EQ(1U, dbfull()->TEST_WalDurabilityWaiters());
  release(&release_c);
  c.join();
  w2.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_OK(c_status);
  ASSERT_OK(w2_status);
  EXPECT_EQ("v", Get(cf_for_partition_[3], "w2"));
}

TEST_P(DBWritePartitionTest, TargetedWakeOnPublish) {
  // Targeted wake, published leg: W1 (async, target h+1) and W2 (async,
  // target h+3) park in their completion gates above held appends H1 [h,h]
  // and H2 [h+2,h+2]; releasing H1 publishes through h+1, so W1 returns
  // while W2 stays parked; releasing H2 lets W2 return. H2 rides its own
  // partition: a second held append on H1's partition would queue behind
  // H1's group and never reach the hold. Ported from Muse's closed GC28
  // candidate.
  // Neuter: drop the wake in PublishAppliedFrontier; W1 never wakes.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
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
  // Arms the hold of the next append: call it before starting the writer
  // to hold, then wait_in_append().
  auto hold_next_append = [&] {
    std::lock_guard<std::mutex> lock(mu);
    hold_append = true;
    h_in_append = false;
    ++hold_gen;
  };
  auto wait_in_append = [&] {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(30),
                       [&] { return h_in_append; });
  };
  auto release_append = [&] {
    std::lock_guard<std::mutex> lock(mu);
    ++release_gen;
    cv.notify_all();
  };
  std::atomic<std::thread::id> w2_thread_id{std::thread::id()};
  GC28Hold w2_woken;
  w2_woken.Arm("DBImpl::WaitForWalDurability:AfterWake",
               GC28OnThread(&w2_thread_id), /*hold=*/false);

  Status h1_status;
  hold_next_append();
  port::Thread h1_thread(
      [&] { h1_status = Put(cf_h, "h1", "v", WriteOptions()); });
  EXPECT_TRUE(wait_in_append());
  Status w1_status;
  port::Thread w1_thread(
      [&] { w1_status = Put(cf_w, "w1", "v", WriteOptions()); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  Status h2_status;
  hold_next_append();
  port::Thread h2_thread(
      [&] { h2_status = Put(cf_h2, "h2", "v", WriteOptions()); });
  EXPECT_TRUE(wait_in_append());
  Status w2_status;
  std::atomic<bool> w2_done{false};
  port::Thread w2_thread([&] {
    w2_thread_id.store(std::this_thread::get_id());
    w2_status = Put(cf_w, "w2", "v", WriteOptions());
    w2_done.store(true);
  });
  EXPECT_TRUE(WaitForDurabilityWaiters(2));

  // Releasing H1 covers W1 (target h+1) but not W2 (target h+3). Every held
  // writer is released and every thread joined before the checks.
  release_append();
  const Status w1_joined = RunWithin(std::chrono::seconds(30), [&] {
    w1_thread.join();
    return w1_status;
  });
  const bool w2_done_early = w2_done.load();
  const int w2_wakes_early = w2_woken.Hits();
  const size_t waiters_early = dbfull()->TEST_WalDurabilityWaiters();
  release_append();
  h1_thread.join();
  h2_thread.join();
  const Status w2_joined = RunWithin(std::chrono::seconds(30), [&] {
    w2_thread.join();
    return w2_status;
  });
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_OK(w1_joined);
  EXPECT_FALSE(w2_done_early);
  EXPECT_EQ(0, w2_wakes_early);
  EXPECT_EQ(1U, waiters_early);
  ASSERT_OK(w2_joined);
  ASSERT_OK(h1_status);
  ASSERT_OK(h2_status);
  EXPECT_EQ("v", Get(cf_h, "h1"));
  EXPECT_EQ("v", Get(cf_h2, "h2"));
  EXPECT_EQ("v", Get(cf_w, "w1"));
  EXPECT_EQ("v", Get(cf_w, "w2"));
}

TEST_P(DBWritePartitionTest, TargetedWakeOnWalAdvance) {
  // Targeted wake, WAL leg: S1 (sync, target h+1) and S2 (sync, target h+3)
  // park in their pre-sync waits (unpublished) above held appends H1 [h,h]
  // and H2 [h+2,h+2]. Releasing H1's append advances the WAL frontier to
  // h+1 while H1 itself stays held after its append, before it publishes
  // (DBImpl::InstallCompletionGate), so no publication can wake S1: S1 runs
  // its gate sync and reaches its publication on the WAL-advance wake alone,
  // while S2 does not return. Releasing H1 lets S1 return; releasing H2 lets S2
  // return. Ported from Muse's closed GC28 candidate, with H1 held between
  // its WAL completion and its publication.
  // Neuter: drop the wake after the WAL completion in WriteGroupToWAL; S1
  // never reaches its publication while H1 is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS(
        "N > 1 needs indexed WALs, which two_write_queues "
        "rejects");
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
  // Arms the hold of the next append: call it before starting the writer
  // to hold, then wait_in_append().
  auto hold_next_append = [&] {
    std::lock_guard<std::mutex> lock(mu);
    hold_append = true;
    h_in_append = false;
    ++hold_gen;
  };
  auto wait_in_append = [&] {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(30),
                       [&] { return h_in_append; });
  };
  auto release_append = [&] {
    std::lock_guard<std::mutex> lock(mu);
    ++release_gen;
    cv.notify_all();
  };
  // H1, once its append completed, is held before it publishes.
  std::atomic<std::thread::id> h1_thread_id{std::thread::id()};
  GC28Hold h1_appended;
  h1_appended.Arm("DBImpl::InstallCompletionGate", GC28OnThread(&h1_thread_id));
  std::atomic<std::thread::id> s1_thread_id{std::thread::id()};
  GC28Hold s1_publishing;
  s1_publishing.Arm("DBImpl::RunSyncGate:BeforePublish",
                    GC28OnThread(&s1_thread_id), /*hold=*/false);

  Status h1_status;
  hold_next_append();
  port::Thread h1_thread([&] {
    h1_thread_id.store(std::this_thread::get_id());
    h1_status = Put(cf_h, "h1", "v", WriteOptions());
  });
  EXPECT_TRUE(wait_in_append());
  Status s1_status;
  port::Thread s1_thread([&] {
    s1_thread_id.store(std::this_thread::get_id());
    s1_status = Put(cf_s, "s1", "v", sync_wo);
  });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  Status h2_status;
  hold_next_append();
  port::Thread h2_thread(
      [&] { h2_status = Put(cf_h2, "h2", "v", WriteOptions()); });
  EXPECT_TRUE(wait_in_append());
  Status s2_status;
  std::atomic<bool> s2_done{false};
  port::Thread s2_thread([&] {
    s2_status = Put(cf_s, "s2", "v", sync_wo);
    s2_done.store(true);
  });
  EXPECT_TRUE(WaitForDurabilityWaiters(2));

  // Releasing H1's append covers S1 (target h+1) but not S2 (target h+3).
  // Every held writer is released and every thread joined before the
  // checks, so a failing check leaves no writer blocked in the DB.
  release_append();
  const bool h1_held = h1_appended.WaitHeld();
  const bool s1_woken = s1_publishing.WaitHits(1, std::chrono::seconds(10));
  // S2 may wake to help S1's gate sync, but its own target stays uncovered.
  const bool s2_done_early = s2_done.load();
  const std::string s2_early = Get(cf_s, "s2");
  h1_appended.Release();
  const Status s1_joined = RunWithin(std::chrono::seconds(30), [&] {
    s1_thread.join();
    return s1_status;
  });
  release_append();
  h1_thread.join();
  h2_thread.join();
  const Status s2_joined = RunWithin(std::chrono::seconds(30), [&] {
    s2_thread.join();
    return s2_status;
  });
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_TRUE(h1_held);
  EXPECT_TRUE(s1_woken) << "S1 was not woken by the WAL advance";
  EXPECT_FALSE(s2_done_early);
  EXPECT_EQ("NOT_FOUND", s2_early);
  ASSERT_OK(s1_joined);
  ASSERT_OK(s2_joined);
  ASSERT_OK(h1_status);
  ASSERT_OK(h2_status);
  EXPECT_EQ("v", Get(cf_h, "h1"));
  EXPECT_EQ("v", Get(cf_h2, "h2"));
  EXPECT_EQ("v", Get(cf_s, "s1"));
  EXPECT_EQ("v", Get(cf_s, "s2"));
}

// GC30 tests. Helpers shared by this block only.
namespace {
// Matches MemTable::Add:BeforeReturn:Encoded for an entry whose user key is
// one of `keys`.
std::function<bool(void*)> GC30AddOf(const std::vector<std::string>& keys) {
  return [keys](void* arg) {
    Slice encoded = *static_cast<Slice*>(arg);
    Slice internal_key;
    if (!GetLengthPrefixedSlice(&encoded, &internal_key) ||
        internal_key.size() < 8) {
      return false;
    }
    const std::string user_key = ExtractUserKey(internal_key).ToString();
    return std::find(keys.begin(), keys.end(), user_key) != keys.end();
  };
}

// A write on its own thread. Joined when it goes out of scope, so declare it
// before every hold that may block it: the holds release first.
class GC30Write {
 public:
  GC30Write() = default;
  GC30Write(const GC30Write&) = delete;
  GC30Write& operator=(const GC30Write&) = delete;
  ~GC30Write() { Join().PermitUncheckedError(); }
  void Start(const std::function<Status()>& write) {
    thread_ = port::Thread([this, write] {
      Status s = write();
      std::lock_guard<std::mutex> lock(mu_);
      status_ = s;
      done_ = true;
      cv_.notify_all();
    });
  }
  bool Done() {
    std::lock_guard<std::mutex> lock(mu_);
    return done_;
  }
  bool WaitDone(std::chrono::seconds timeout = std::chrono::seconds(30)) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this] { return done_; });
  }
  Status Join() {
    if (thread_.joinable()) {
      thread_.join();
    }
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  Status status_;
  port::Thread thread_;
};

// Runs writes as one write group: the first is held before it leads
// (DBImpl::WriteImpl:BeforeLeaderEnters) until the others have queued behind
// it on its partition. Declare it before every hold that may block a write.
// (A hold clears its sync point only once no sync point callback runs, so
// declare the holds that hold last, or release them all first.)
class GC30Group {
 public:
  GC30Group() = default;
  GC30Group(const GC30Group&) = delete;
  GC30Group& operator=(const GC30Group&) = delete;
  ~GC30Group() {
    leader_.Release();
    writes_.clear();
  }
  // Returns true once the group formed: the first write is held and the
  // others wait behind it.
  bool Start(const std::vector<std::function<Status()>>& writes) {
    leader_.Arm("DBImpl::WriteImpl:BeforeLeaderEnters");
    queued_.Arm("WriteThread::JoinBatchGroup:BeganWaiting", nullptr,
                /*hold=*/false);
    for (size_t i = 0; i < writes.size(); ++i) {
      writes_.emplace_back(new GC30Write());
      writes_.back()->Start(writes[i]);
      if (i == 0 && !leader_.WaitHeld()) {
        return false;
      }
    }
    return queued_.WaitHits(static_cast<int>(writes.size()) - 1);
  }
  // Lets the first write lead the group.
  void Lead() { leader_.Release(); }
  int Returned() {
    int returned = 0;
    for (auto& write : writes_) {
      returned += write->Done() ? 1 : 0;
    }
    return returned;
  }
  std::vector<Status> Join() {
    Lead();
    std::vector<Status> statuses;
    for (auto& write : writes_) {
      statuses.push_back(write->Join());
    }
    return statuses;
  }

 private:
  GC28Hold leader_;
  GC28Hold queued_;
  std::vector<std::unique_ptr<GC30Write>> writes_;
};

std::vector<std::function<Status()>> GC30Puts(
    DBTestBase* test, int cf, const std::vector<std::string>& keys,
    const WriteOptions& write_options = WriteOptions()) {
  std::vector<std::function<Status()>> writes;
  for (const std::string& key : keys) {
    writes.push_back([test, cf, key, write_options] {
      return test->Put(cf, key, "v", write_options);
    });
  }
  return writes;
}

// Every read path of `cf` misses each of `keys`: Get, MultiGet, an iterator
// and a tailing iterator.
void GC30ExpectInvisible(DB* db, ColumnFamilyHandle* cf,
                         const std::vector<std::string>& keys) {
  for (const std::string& key : keys) {
    std::string value;
    const Status s = db->Get(ReadOptions(), cf, key, &value);
    EXPECT_TRUE(s.IsNotFound()) << key << ": " << s.ToString();
  }
  std::vector<ColumnFamilyHandle*> cfs(keys.size(), cf);
  std::vector<Slice> slices(keys.begin(), keys.end());
  std::vector<std::string> values;
  for (const Status& s : db->MultiGet(ReadOptions(), cfs, slices, &values)) {
    EXPECT_TRUE(s.IsNotFound()) << s.ToString();
  }
  for (const bool tailing : {false, true}) {
    ReadOptions ro;
    ro.tailing = tailing;
    std::unique_ptr<Iterator> it(db->NewIterator(ro, cf));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      EXPECT_EQ(keys.end(),
                std::find(keys.begin(), keys.end(), it->key().ToString()))
          << "tailing=" << tailing << " " << it->key().ToString();
    }
    EXPECT_OK(it->status());
  }
}

// Every read path of `cf` returns "v" for each of `keys`.
void GC30ExpectVisible(DB* db, ColumnFamilyHandle* cf,
                       const std::vector<std::string>& keys) {
  for (const std::string& key : keys) {
    std::string value;
    EXPECT_OK(db->Get(ReadOptions(), cf, key, &value)) << key;
    EXPECT_EQ("v", value) << key;
  }
  for (const bool tailing : {false, true}) {
    ReadOptions ro;
    ro.tailing = tailing;
    std::unique_ptr<Iterator> it(db->NewIterator(ro, cf));
    for (const std::string& key : keys) {
      it->Seek(key);
      ASSERT_TRUE(it->Valid()) << "tailing=" << tailing << " " << key;
      EXPECT_EQ(key, it->key().ToString());
    }
    EXPECT_OK(it->status());
  }
}

// True when a live SST file of `cf` holds a version of `key`.
bool GC30LiveSstHasKey(DB* db, ColumnFamilyHandle* cf, const std::string& key) {
  std::vector<LiveFileMetaData> files;
  db->GetLiveFilesMetaData(&files);
  bool found = false;
  for (const LiveFileMetaData& file : files) {
    if (file.column_family_name != cf->GetName()) {
      continue;
    }
    SstFileReader reader(db->GetOptions(cf));
    EXPECT_OK(reader.Open(file.directory + "/" + file.relative_filename));
    std::unique_ptr<Iterator> iter(reader.NewIterator(ReadOptions()));
    iter->Seek(key);
    found |= iter->Valid() && iter->key() == key;
    EXPECT_OK(iter->status());
  }
  return found;
}

IOStatus GC30RetryableError(const std::string& msg) {
  IOStatus s = IOStatus::IOError(msg);
  s.SetRetryable(true);
  return s;
}

// Fails the next WAL append on this DB before its record is added (so
// nothing of it lands), with a retryable error, once armed.
class GC30AppendFailure {
 public:
  GC30AppendFailure() {
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterMaybeAddUserDefinedTimestampSizeRecord",
        [this](void* arg) {
          if (armed_.exchange(false)) {
            *static_cast<IOStatus*>(arg) =
                GC30RetryableError("injected append error");
          }
        });
  }
  ~GC30AppendFailure() {
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteToWAL:AfterMaybeAddUserDefinedTimestampSizeRecord");
  }
  void Arm() { armed_.store(true); }

 private:
  std::atomic<bool> armed_{false};
};

// Fails the next gate sync with a retryable error, once armed.
class GC30GateSyncFailure {
 public:
  GC30GateSyncFailure() {
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [this](void* arg) {
          if (armed_.exchange(false)) {
            *static_cast<IOStatus*>(arg) =
                GC30RetryableError("injected gate sync error");
          }
        });
  }
  ~GC30GateSyncFailure() {
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  }
  void Arm() { armed_.store(true); }

 private:
  std::atomic<bool> armed_{false};
};

// Counts background errors raised on the DB it listens to.
class GC30BgErrorCounter : public EventListener {
 public:
  void OnBackgroundError(BackgroundErrorReason /*reason*/,
                         Status* /*bg_error*/) override {
    ++count_;
  }
  int Count() const { return count_.load(); }

 private:
  std::atomic<int> count_{0};
};

const std::vector<WALRecoveryMode> kGC30RecoveryModes = {
    WALRecoveryMode::kAbsoluteConsistency,
    WALRecoveryMode::kTolerateCorruptedTailRecords,
    WALRecoveryMode::kPointInTimeRecovery,
    WALRecoveryMode::kSkipAnyCorruptedRecords};
}  // namespace

TEST_P(DBWritePartitionTest, AsyncGroupInsertOverlapsHeldAppend) {
  // GC30 Q1 (ruling 4d, D-GC30b (i)): above one partition a group's memtable
  // inserts do not wait for its WAL append. Four async writes form one group
  // on one partition and the leader's append is held: the three followers'
  // inserts finish meanwhile, yet no key is visible on any read path, the
  // published sequence does not move and no write returns. Released, every
  // write returns OK and reads back.
  // Neuter: launch the parallel memtable writers after the append (the base
  // order); no follower inserts while the append is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    const std::vector<std::string> keys = {"k0", "k1", "k2", "k3"};
    const SequenceNumber published = db_->GetLatestSequenceNumber();
    GC30Group group;
    GC28Hold inserted;
    inserted.Arm("MemTable::Add:BeforeReturn:Encoded",
                 GC30AddOf({"k1", "k2", "k3"}), /*hold=*/false);
    GC28Hold append;
    append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
    ASSERT_TRUE(group.Start(GC30Puts(this, cf, keys)));
    group.Lead();
    ASSERT_TRUE(append.WaitHeld());
    EXPECT_TRUE(inserted.WaitHits(3, std::chrono::seconds(10)))
        << "followers inserted while the append was held: " << inserted.Hits();
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    EXPECT_EQ(published, db_->GetLatestSequenceNumber());
    EXPECT_EQ(0, group.Returned());
    append.Release();
    for (const Status& s : group.Join()) {
      ASSERT_OK(s);
    }
    GC30ExpectVisible(db_.get(), handles_[cf], keys);
  }
}

TEST_P(DBWritePartitionTest, GroupPublishesOnlyAfterAppendAndInserts) {
  // GC30 Q1: a group publishes only once its append and every insert are
  // done. (a) The append is done and one follower's insert is held: nothing
  // is published. (b) Every follower's insert is done and the append is
  // held: nothing is published. Released, all writes return OK and read back.
  // Neuter: let the last inserter exit without waiting for the append (the
  // leader's unit completed before its append); in (b) the last follower
  // publishes, and in debug PublishAppliedFrontier's assertion fires.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[2];
  {
    SCOPED_TRACE("(a) an insert held after the append");
    const std::vector<std::string> keys = {"a0", "a1", "a2"};
    const SequenceNumber published = db_->GetLatestSequenceNumber();
    GC30Group group;
    GC28Hold appended;
    appended.Arm("DBImpl::WriteToWAL:AfterAddRecord", nullptr,
                 /*hold=*/false);
    GC28Hold insert;
    insert.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"a2"}));
    ASSERT_TRUE(group.Start(GC30Puts(this, cf, keys)));
    group.Lead();
    ASSERT_TRUE(insert.WaitHeld());
    ASSERT_TRUE(appended.WaitHits(1));
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    EXPECT_EQ(published, db_->GetLatestSequenceNumber());
    EXPECT_EQ(0, group.Returned());
    insert.Release();
    for (const Status& s : group.Join()) {
      ASSERT_OK(s);
    }
    GC30ExpectVisible(db_.get(), handles_[cf], keys);
  }
  {
    SCOPED_TRACE("(b) the append held after the inserts");
    const std::vector<std::string> keys = {"b0", "b1", "b2"};
    const SequenceNumber published = db_->GetLatestSequenceNumber();
    GC30Group group;
    GC28Hold inserted;
    inserted.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"b1", "b2"}),
                 /*hold=*/false);
    GC28Hold append;
    append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
    ASSERT_TRUE(group.Start(GC30Puts(this, cf, keys)));
    group.Lead();
    ASSERT_TRUE(append.WaitHeld());
    ASSERT_TRUE(inserted.WaitHits(2, std::chrono::seconds(10)))
        << "followers inserted while the append was held: " << inserted.Hits();
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    EXPECT_EQ(published, db_->GetLatestSequenceNumber());
    EXPECT_EQ(0, group.Returned());
    append.Release();
    for (const Status& s : group.Join()) {
      ASSERT_OK(s);
    }
    GC30ExpectVisible(db_.get(), handles_[cf], keys);
  }
}

TEST_P(DBWritePartitionTest,
       FailedAppendAfterInsertsStaysInvisibleAndIsDropped) {
  // GC30 Q1 (D-1): a group's append fails after its followers inserted. Four
  // async writes form a group; with the leader's append held the followers
  // insert, then the append fails (retryable, nothing lands). Every writer
  // returns the IOError and the DB stops. None of the keys is visible on
  // any read path before Resume(), after it, nor after a crash-like reopen
  // in every recovery mode; no live SST holds one; every acknowledged write
  // survives.
  // Neuters: publish at the exit whatever the append status (the keys read
  // back); install no fence on the failed append (Resume()'s flush keeps the
  // followers' keys, which then read back).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    const int other_cf = cf_for_partition_[0];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf, "acked_before", "v", sync_wo));
    ASSERT_OK(Put(other_cf, "acked_other", "v", sync_wo));
    const std::vector<std::string> keys = {"k0", "k1", "k2", "k3"};
    {
      GC30Group group;
      GC28Hold inserted;
      inserted.Arm("MemTable::Add:BeforeReturn:Encoded",
                   GC30AddOf({"k1", "k2", "k3"}), /*hold=*/false);
      GC30AppendFailure failure;
      GC28Hold append;
      append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
      ASSERT_TRUE(group.Start(GC30Puts(this, cf, keys)));
      group.Lead();
      ASSERT_TRUE(append.WaitHeld());
      // Not vacuous: the inserts happened before the append failed.
      ASSERT_TRUE(inserted.WaitHits(3, std::chrono::seconds(10)))
          << "followers inserted while the append was held: "
          << inserted.Hits();
      failure.Arm();
      append.Release();
      for (const Status& s : group.Join()) {
        EXPECT_TRUE(s.IsIOError()) << s.ToString();
      }
    }
    ASSERT_NOK(dbfull()->TEST_GetBGError());
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    EXPECT_NOK(Put(other_cf, "while_stopped", "v"));

    ASSERT_OK(dbfull()->Resume());
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    ASSERT_OK(Flush(cf));
    for (const std::string& key : keys) {
      EXPECT_FALSE(GC30LiveSstHasKey(db_.get(), handles_[cf], key)) << key;
    }
    ASSERT_OK(Put(cf, "acked_after", "v", sync_wo));
    GC30ExpectVisible(db_.get(), handles_[cf], {"acked_before", "acked_after"});

    for (const WALRecoveryMode mode : kGC30RecoveryModes) {
      SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(partitions, options);
      GC30ExpectInvisible(db_.get(), handles_[cf], keys);
      GC30ExpectVisible(db_.get(), handles_[cf],
                        {"acked_before", "acked_after"});
      GC30ExpectVisible(db_.get(), handles_[other_cf], {"acked_other"});
    }
  }
}

TEST_P(DBWritePartitionTest, AppendAndInsertBothFailClassifiedOnce) {
  // GC30 Q1 (D-1): in one group a follower's insert fails (its batch names a
  // column family that does not exist, ignore_missing_column_families is
  // off) while the leader's append is held, and then the append fails. The
  // group is classified once, as the WAL failure: one background error, the
  // WAL's IOError, which every writer returns (the failed follower
  // included); debug builds do not assert (HandleMemTableInsertFailure never
  // runs). No write is visible before or after Resume(), which succeeds.
  // Neuter: classify the insert failure too (HandleMemTableInsertFailure at
  // the exit whatever the append status); debug builds assert, release
  // builds record a second background error.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 2;
  auto counter = std::make_shared<GC30BgErrorCounter>();
  Options options = IndexedWalOptions(CurrentOptions());
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 0;
  options.listeners.push_back(counter);
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[1];
  uint32_t missing_cf = static_cast<uint32_t>(handles_.size()) + 100;
  while (true) {
    WriteBatch batch;
    ASSERT_OK(WriteBatchInternal::Put(&batch, missing_cf, "key", "value"));
    size_t partition = kPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(batch, kPartitions, &partition));
    if (partition == 1) {
      break;
    }
    ++missing_cf;
  }
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf, "acked", "v", sync_wo));
  std::vector<std::function<Status()>> writes =
      GC30Puts(this, cf, {"k0", "k1"});
  writes.insert(writes.begin() + 1, [this, missing_cf] {
    WriteBatch batch;
    EXPECT_OK(WriteBatchInternal::Put(&batch, missing_cf, "missing", "v"));
    return db_->Write(WriteOptions(), &batch);
  });
  std::vector<Status> statuses;
  {
    GC30Group group;
    GC28Hold followers_done;
    followers_done.Arm("DBImpl::WriteImpl:FollowerMemTableInserted", nullptr,
                       /*hold=*/false);
    GC30AppendFailure failure;
    GC28Hold append;
    append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
    ASSERT_TRUE(group.Start(writes));
    group.Lead();
    ASSERT_TRUE(append.WaitHeld());
    ASSERT_TRUE(followers_done.WaitHits(2, std::chrono::seconds(10)))
        << "followers inserted while the append was held: "
        << followers_done.Hits();
    failure.Arm();
    append.Release();
    statuses = group.Join();
  }
  ASSERT_EQ(3U, statuses.size());
  for (const Status& s : statuses) {
    EXPECT_TRUE(s.IsIOError()) << s.ToString();
  }
  const Status bg = dbfull()->TEST_GetBGError();
  EXPECT_TRUE(bg.IsIOError()) << bg.ToString();
  EXPECT_EQ(1, counter->Count());
  GC30ExpectInvisible(db_.get(), handles_[cf], {"k0", "k1"});
  ASSERT_OK(dbfull()->Resume());
  GC30ExpectInvisible(db_.get(), handles_[cf], {"k0", "k1"});
  ASSERT_OK(Put(cf, "after", "v"));
  GC30ExpectVisible(db_.get(), handles_[cf], {"after"});
}

TEST_P(DBWritePartitionTest,
       SyncWriterInsertedBeforeFsyncInvisibleUntilDurable) {
  // GC30 Q2 (ruling 4b, D-GC30e (i)): a sync write's batch is in the
  // memtable while its gate sync is held, yet invisible to Get, MultiGet, an
  // iterator, a tailing iterator created before the write and one created
  // during the hold, a snapshot taken during the hold, and
  // GetLatestSequenceNumber(). Released, the write returns OK and is visible
  // everywhere but to that snapshot.
  // Neuter: insert the sync writers' batches after the fsync (the base
  // order); the key is not in the memtable while the gate sync is held. The
  // tailing legs rest on GC27 Q4's bound (neuter: tailing at
  // kMaxSequenceNumber; the tailing iterators then show the key).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    ASSERT_OK(Put(cf, "before", "v"));
    ReadOptions tailing_ro;
    tailing_ro.tailing = true;
    std::unique_ptr<Iterator> tailing_before(
        db_->NewIterator(tailing_ro, handles_[cf]));
    tailing_before->Seek("s");
    EXPECT_FALSE(tailing_before->Valid());
    const SequenceNumber published = db_->GetLatestSequenceNumber();
    const Snapshot* during = nullptr;
    std::unique_ptr<Iterator> tailing_during;
    WriteOptions sync_wo;
    sync_wo.sync = true;
    {
      GC30Write s;
      GC28Hold inserted;
      inserted.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"s"}),
                   /*hold=*/false);
      GC28Hold gate;
      gate.Arm("DBImpl::CompletionGate:BeforeCoalescedSync");
      s.Start([&] { return Put(cf, "s", "v", sync_wo); });
      ASSERT_TRUE(gate.WaitHeld());
      EXPECT_EQ(1, inserted.Hits()) << "inserted before its fsync";
      GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
      tailing_before->Seek("s");
      EXPECT_FALSE(tailing_before->Valid());
      tailing_during.reset(db_->NewIterator(tailing_ro, handles_[cf]));
      tailing_during->Seek("s");
      EXPECT_FALSE(tailing_during->Valid());
      during = db_->GetSnapshot();
      EXPECT_EQ(published, db_->GetLatestSequenceNumber());
      EXPECT_FALSE(s.Done());
      gate.Release();
      ASSERT_OK(s.Join());
    }
    GC30ExpectVisible(db_.get(), handles_[cf], {"s"});
    for (Iterator* it : {tailing_before.get(), tailing_during.get()}) {
      it->Seek("s");
      ASSERT_TRUE(it->Valid());
      EXPECT_EQ("s", it->key().ToString());
      EXPECT_OK(it->status());
    }
    ReadOptions at_snapshot;
    at_snapshot.snapshot = during;
    std::string value;
    EXPECT_TRUE(db_->Get(at_snapshot, handles_[cf], "s", &value).IsNotFound());
    db_->ReleaseSnapshot(during);
    tailing_before.reset();
    tailing_during.reset();
  }
}

TEST_P(DBWritePartitionTest, FailedFsyncAfterInsertFencedAndDropped) {
  // GC30 Q2 (D-1, D-9): a sync write is in the memtable when its gate sync
  // fails. It returns the IOError and stays invisible on every read path
  // before Resume(), after it and after a crash-like reopen in every
  // recovery mode; no live SST holds it; every acknowledged write survives.
  // Neuter: fence the failed sync range without recording it for the flushes
  // to drop (FailUnpublishedSyncRange); Resume()'s flush keeps the key, which
  // then reads back. (The hole branch never runs here: the failed fsync set
  // the background error first.)
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    const int other_cf = cf_for_partition_[0];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf, "acked_before", "v", sync_wo));
    ASSERT_OK(Put(other_cf, "acked_other", "v", sync_wo));
    {
      GC30Write s;
      GC28Hold inserted;
      inserted.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"s"}),
                   /*hold=*/false);
      GC30GateSyncFailure failure;
      GC28Hold gate;
      gate.Arm("DBImpl::CompletionGate:BeforeCoalescedSync");
      s.Start([&] { return Put(cf, "s", "v", sync_wo); });
      ASSERT_TRUE(gate.WaitHeld());
      ASSERT_EQ(1, inserted.Hits()) << "inserted before its fsync";
      failure.Arm();
      gate.Release();
      const Status s_status = s.Join();
      EXPECT_TRUE(s_status.IsIOError()) << s_status.ToString();
    }
    ASSERT_NOK(dbfull()->TEST_GetBGError());
    GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
    ASSERT_OK(dbfull()->Resume());
    GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
    ASSERT_OK(Flush(cf));
    EXPECT_FALSE(GC30LiveSstHasKey(db_.get(), handles_[cf], "s"));
    ASSERT_OK(Put(cf, "acked_after", "v", sync_wo));
    for (const WALRecoveryMode mode : kGC30RecoveryModes) {
      SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(partitions, options);
      GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
      GC30ExpectVisible(db_.get(), handles_[cf],
                        {"acked_before", "acked_after"});
      GC30ExpectVisible(db_.get(), handles_[other_cf], {"acked_other"});
    }
  }
}

TEST_P(DBWritePartitionTest,
       SyncFailureWithoutBackgroundErrorAfterInsertNeverPublishes) {
  // GC30 Q2 (D-GC30d (ii)): a sync write whose gate sync succeeds but whose
  // MANIFEST record of the synced WALs fails with a non-I/O status sets no
  // background error of its own: FailUnpublishedSyncRange's first branch.
  // At the base that branch publishes the range as a hole, having inserted
  // nothing; here the batch is already in the memtable, so the range is
  // fenced instead, with a background error. The write returns the error,
  // is never visible, and no flush persists it.
  // Cannot fail at the base: there nothing is inserted before the fsync (and
  // the injection point, AfterManifestRecord, is new).
  // Neuter: keep the hole publication once inserted; the key reads back.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 2;
  Options options = CurrentOptions();
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 0;
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf, "acked", "v", sync_wo));
  std::atomic<bool> fail_record{true};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterManifestRecord", [&](void* arg) {
        if (fail_record.exchange(false)) {
          *static_cast<Status*>(arg) =
              Status::Aborted("injected MANIFEST record failure");
        }
      });
  GC28Hold inserted;
  inserted.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"s"}),
               /*hold=*/false);
  const Status s_status = Put(cf, "s", "v", sync_wo);
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterManifestRecord");
  EXPECT_TRUE(s_status.IsAborted()) << s_status.ToString();
  EXPECT_EQ(1, inserted.Hits());
  GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
  EXPECT_NOK(dbfull()->TEST_GetBGError());
  // A flush either refuses (the DB is stopped) or drops the fenced entry.
  Flush(cf).PermitUncheckedError();
  EXPECT_FALSE(GC30LiveSstHasKey(db_.get(), handles_[cf], "s"));
  GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
  GC30ExpectVisible(db_.get(), handles_[cf], {"acked"});
}

TEST_P(DBWritePartitionTest, SyncLedGroupAsyncMembersInsertDuringAppend) {
  // GC30 Q3 (D-GC30e (i)): in a group led by a sync write, the async
  // members' inserts finish while the leader's append is held, and they
  // return, visible, before the group's gate sync starts; the sync writer
  // returns only after it.
  // Neuter: launch the memtable writers after the append (the base order);
  // no member inserts while the append is held.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[3];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  std::vector<std::function<Status()>> writes =
      GC30Puts(this, cf, {"leader"}, sync_wo);
  for (const auto& write : GC30Puts(this, cf, {"m1", "m2"})) {
    writes.push_back(write);
  }
  GC30Group group;
  GC28Hold inserted;
  inserted.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"m1", "m2"}),
               /*hold=*/false);
  GC28Hold gate;
  gate.Arm("DBImpl::CompletionGate:BeforeCoalescedSync");
  GC28Hold append;
  append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
  // A hold clears its sync point only once no callback runs: release both
  // before either clears.
  Defer release_holds([&] {
    gate.Release();
    append.Release();
  });
  ASSERT_TRUE(group.Start(writes));
  group.Lead();
  ASSERT_TRUE(append.WaitHeld());
  EXPECT_TRUE(inserted.WaitHits(2, std::chrono::seconds(10)))
      << "members inserted while the append was held: " << inserted.Hits();
  EXPECT_EQ(0, group.Returned());
  append.Release();
  ASSERT_TRUE(gate.WaitHeld());
  // Both async members return before the gate sync is released.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (group.Returned() < 2 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(2, group.Returned());
  GC30ExpectVisible(db_.get(), handles_[cf], {"m1", "m2"});
  GC30ExpectInvisible(db_.get(), handles_[cf], {"leader"});
  gate.Release();
  for (const Status& s : group.Join()) {
    ASSERT_OK(s);
  }
  GC30ExpectVisible(db_.get(), handles_[cf], {"leader", "m1", "m2"});
}

TEST_P(DBWritePartitionTest, SwitchWaitsForGroupInsertedBeforeAppend) {
  // GC30 Q4 (ruling 4d): no memtable switch while a group that inserted can
  // still fail. A group of three async writes on partition 1 holds its
  // leader's append while the followers insert; the first of those inserts
  // schedules a flush of their column family (filled past its write buffer
  // beforehand). A write on partition 0 then finds the flush scheduled and
  // closes the memtable switch gate: it waits for the held group
  // (DBImpl::CloseMemTableSwitchGate:WaitForLeaders) and does not return.
  // The append then fails (retryable): the group is fenced before it leaves
  // the gate, so the switch's memtable, once flushed (by Resume()), holds
  // none of the followers' keys, which stay invisible.
  // Neuter: leave the switch gate once the inserts are done, before the
  // append returns (ExitMemTableSwitchGate before the leader's append); the
  // switch does not wait, its flush persists the followers' keys before the
  // fence exists, and they read back after Resume().
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    options.write_buffer_size = 64 << 10;
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    const int other_cf = cf_for_partition_[0];
    // Past the write buffer: the next insert into `cf` schedules its flush.
    ASSERT_OK(Put(cf, "fill", std::string(128 << 10, 'x')));
    const std::vector<std::string> keys = {"k0", "k1", "k2"};
    Status switch_status;
    {
      GC30Write switcher;
      GC30Group group;
      GC28Hold inserted;
      inserted.Arm("MemTable::Add:BeforeReturn:Encoded",
                   GC30AddOf({"k1", "k2"}), /*hold=*/false);
      GC28Hold switch_waits;
      switch_waits.Arm("DBImpl::CloseMemTableSwitchGate:WaitForLeaders",
                       nullptr, /*hold=*/false);
      GC30AppendFailure failure;
      GC28Hold append;
      append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
      ASSERT_TRUE(group.Start(GC30Puts(this, cf, keys)));
      group.Lead();
      ASSERT_TRUE(append.WaitHeld());
      ASSERT_TRUE(inserted.WaitHits(2, std::chrono::seconds(10)))
          << "followers inserted while the append was held: "
          << inserted.Hits();
      switcher.Start([&] { return Put(other_cf, "switch", "v"); });
      const bool waited = switch_waits.WaitHits(1, std::chrono::seconds(10));
      EXPECT_TRUE(waited);
      EXPECT_FALSE(switcher.Done());
      if (!waited) {
        // The switch did not wait: let its flush finish before the failure.
        ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[cf]));
      }
      failure.Arm();
      append.Release();
      for (const Status& s : group.Join()) {
        EXPECT_TRUE(s.IsIOError()) << s.ToString();
      }
      // The switching write allocated above the failed group: it fails too.
      switch_status = switcher.Join();
    }
    EXPECT_NOK(switch_status);
    ASSERT_NOK(dbfull()->TEST_GetBGError());
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    ASSERT_OK(dbfull()->Resume());
    ASSERT_OK(Flush(cf));
    for (const std::string& key : keys) {
      EXPECT_FALSE(GC30LiveSstHasKey(db_.get(), handles_[cf], key)) << key;
    }
    GC30ExpectInvisible(db_.get(), handles_[cf], keys);
    ASSERT_OK(Put(cf, "after", "v"));
    GC30ExpectVisible(db_.get(), handles_[cf], {"after"});
  }
}

TEST_P(DBWritePartitionTest, SwitchWaitsForSyncGroupInsertedBeforeFsync) {
  // GC30 Q4 (A10): a sync write is in the memtable while its gate sync is
  // held. Flush() from another thread waits for its group
  // (DBImpl::WaitForUnpublishedSyncGroups:Wait) and does not return. The
  // gate sync then fails (retryable): the flush returns, no live SST holds
  // the key, and it stays invisible after Resume() and after a crash-like
  // reopen in every recovery mode, while every acknowledged write survives.
  // Neuter: drop WaitForUnpublishedSyncGroups from WaitForPendingWrites; the
  // flush persists the key before the fence exists, and it reads back after
  // Resume() (in debug, AssertLastSequenceCoversAllocated fires first).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = IndexedWalOptions(CurrentOptions());
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    const int other_cf = cf_for_partition_[0];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf, "acked_before", "v", sync_wo));
    ASSERT_OK(Put(other_cf, "acked_other", "v", sync_wo));
    Status flush_status;
    {
      GC30Write s;
      GC30Write flush;
      GC28Hold inserted;
      inserted.Arm("MemTable::Add:BeforeReturn:Encoded", GC30AddOf({"s"}),
                   /*hold=*/false);
      GC28Hold flush_waits;
      flush_waits.Arm("DBImpl::WaitForUnpublishedSyncGroups:Wait", nullptr,
                      /*hold=*/false);
      GC30GateSyncFailure failure;
      GC28Hold gate;
      gate.Arm("DBImpl::CompletionGate:BeforeCoalescedSync");
      s.Start([&] { return Put(cf, "s", "v", sync_wo); });
      ASSERT_TRUE(gate.WaitHeld());
      ASSERT_EQ(1, inserted.Hits()) << "inserted before its fsync";
      flush.Start([&] { return Flush(cf); });
      const bool waited = flush_waits.WaitHits(1, std::chrono::seconds(10));
      EXPECT_TRUE(waited);
      EXPECT_FALSE(flush.Done());
      if (!waited) {
        // The flush did not wait: let it finish before the failure.
        flush.WaitDone();
      }
      failure.Arm();
      gate.Release();
      const Status s_status = s.Join();
      EXPECT_TRUE(s_status.IsIOError()) << s_status.ToString();
      flush_status = flush.Join();
    }
    // The flush returns the background error the failed sync recorded.
    EXPECT_NOK(flush_status);
    ASSERT_NOK(dbfull()->TEST_GetBGError());
    GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
    ASSERT_OK(dbfull()->Resume());
    ASSERT_OK(Flush(cf));
    EXPECT_FALSE(GC30LiveSstHasKey(db_.get(), handles_[cf], "s"));
    GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
    for (const WALRecoveryMode mode : kGC30RecoveryModes) {
      SCOPED_TRACE("mode=" + std::to_string(static_cast<int>(mode)));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(partitions, options);
      GC30ExpectInvisible(db_.get(), handles_[cf], {"s"});
      GC30ExpectVisible(db_.get(), handles_[cf], {"acked_before"});
      GC30ExpectVisible(db_.get(), handles_[other_cf], {"acked_other"});
    }
  }
}

TEST_P(DBWritePartitionTest,
       AsyncMemberOfSyncLedGroupNeverWaitsForItsGroupsFsync) {
  // GC30 Q8 (ruling 4d, D-9): the async member of a sync-led group returns
  // once its range is published, never after an fsync it does not need. L
  // (async, partition 0) is held after its insert, before it publishes; the
  // group S (a sync leader and async member M, partition 1) exits, and its
  // leader waits in async_writers_gate for L. G (sync, partition 1, above S)
  // then runs a gate sync claiming partitions 0 and 1: its runner is held in
  // partition 0's fsync, and whoever takes partition 1's is held too. L is
  // released: M returns while both fsyncs are held. Released, every write
  // returns OK.
  // Neuter: let any parked writer help a running gate sync (the base); S's
  // leader takes partition 1's fsync and M returns only after it.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf_l = cf_for_partition_[0];
    const int cf_s = cf_for_partition_[1];
    WriteOptions sync_wo;
    sync_wo.sync = true;
    std::vector<std::function<Status()>> s_writes =
        GC30Puts(this, cf_s, {"s"}, sync_wo);
    s_writes.push_back(GC30Puts(this, cf_s, {"m"})[0]);
    // Holds the first fsync of each of partitions 0 and 1 until released.
    std::mutex mu;
    std::condition_variable cv;
    bool held[2] = {false, false};
    bool release_syncs = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
          const size_t p = *static_cast<size_t*>(arg);
          std::unique_lock<std::mutex> lock(mu);
          if (p > 1 || held[p] || release_syncs) {
            return;
          }
          held[p] = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_syncs; });
        });
    auto release = [&] {
      std::lock_guard<std::mutex> lock(mu);
      release_syncs = true;
      cv.notify_all();
    };
    {
      GC30Write l;
      GC30Group s_group;
      GC30Write g;
      GC28Hold l_held;
      l_held.Arm(
          "DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:pause");
      Defer release_all([&] {
        l_held.Release();
        release();
      });
      l.Start([&] { return Put(cf_l, "l", "v"); });
      ASSERT_TRUE(l_held.WaitHeld());
      ASSERT_TRUE(s_group.Start(s_writes));
      s_group.Lead();
      ASSERT_TRUE(WaitForDurabilityWaiters(1));
      g.Start([&] { return Put(cf_s, "g", "v", sync_wo); });
      {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                                [&] { return held[0]; }));
      }
      l_held.Release();
      ASSERT_OK(l.Join());
      // M, the second write of S, returns while both fsyncs are held.
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (s_group.Returned() < 1 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      EXPECT_EQ(1, s_group.Returned());
      {
        std::lock_guard<std::mutex> lock(mu);
        EXPECT_FALSE(release_syncs);
      }
      release();
      for (const Status& s : s_group.Join()) {
        ASSERT_OK(s);
      }
      ASSERT_OK(g.Join());
    }
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
    GC30ExpectVisible(db_.get(), handles_[cf_s], {"s", "m", "g"});
    GC30ExpectVisible(db_.get(), handles_[cf_l], {"l"});
  }
}

TEST_P(DBWritePartitionTest, HelpingNeverWakesUncoveredWaiters) {
  // GC30 Q8: a gate sync that hands out partitions to helpers wakes only the
  // waiters that may help (sync groups in their written wait), never a write
  // parked for publication. L (async, partition 0) is held after its insert,
  // before it publishes; W (async, partition 2, above L) parks in its
  // completion gate. G (sync, partition 1) runs a gate sync claiming at
  // least two partitions; W is never woken by it. Released, all return OK.
  // Neuter: wake every parked waiter on a multi-partition handout
  // (SignalAll, the base); W wakes.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  std::atomic<std::thread::id> w_thread_id{std::thread::id()};
  GC30Write l;
  GC30Write w;
  GC30Write g;
  GC28Hold w_woken;
  w_woken.Arm("DBImpl::WaitForWalDurability:AfterWake",
              GC28OnThread(&w_thread_id), /*hold=*/false);
  GC28Hold synced;
  synced.Arm("DBImpl::SyncWalsFromCompletionGate:AfterGateSync", nullptr,
             /*hold=*/false);
  GC28Hold l_held;
  l_held.Arm("DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:pause");
  l.Start([&] { return Put(cf_for_partition_[0], "l", "v"); });
  ASSERT_TRUE(l_held.WaitHeld());
  w.Start([&] {
    w_thread_id.store(std::this_thread::get_id());
    return Put(cf_for_partition_[2], "w", "v");
  });
  ASSERT_TRUE(WaitForDurabilityWaiters(1));
  ASSERT_OK(options.statistics->Reset());
  g.Start([&] { return Put(cf_for_partition_[1], "g", "v", sync_wo); });
  ASSERT_TRUE(synced.WaitHits(1));
  // G parks for publication behind L once its gate sync is done.
  ASSERT_TRUE(WaitForDurabilityWaiters(2));
  HistogramData fsynced;
  options.statistics->histogramData(WAL_SYNC_FSYNCED_PARTITIONS, &fsynced);
  EXPECT_GE(fsynced.sum, 2U);
  EXPECT_EQ(0, w_woken.Hits());
  l_held.Release();
  ASSERT_OK(l.Join());
  ASSERT_OK(w.Join());
  ASSERT_OK(g.Join());
  EXPECT_EQ("v", Get(cf_for_partition_[2], "w"));
  EXPECT_EQ("v", Get(cf_for_partition_[1], "g"));
}

TEST_P(DBWritePartitionTest, HelpingPublicationWaiterAboveRunningSyncHelps) {
  // GC30 Q8: a write parked for publication behind a running gate sync,
  // whose acknowledgment waits for that sync anyway (its target is above
  // the sync's claim), fsyncs the sync's partitions. G (sync, partition 1,
  // above an unsynced write on partition 0) claims at least partitions 0 and
  // 1, and its runner is held in its first partition's fsync; W (async,
  // partition 2, above G) parks and fsyncs the others, and returns only
  // after G. Released, every write returns OK.
  // Neuter: help only in a sync group's written wait; W never fsyncs.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{4}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options options = CurrentOptions();
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    WriteOptions sync_wo;
    sync_wo.sync = true;
    ASSERT_OK(Put(cf_for_partition_[0], "x", "v"));
    std::atomic<std::thread::id> g_thread_id{std::thread::id()};
    std::atomic<std::thread::id> w_thread_id{std::thread::id()};
    std::mutex mu;
    std::condition_variable cv;
    bool runner_held = false;
    bool release_runner = false;
    std::atomic<int> w_fsyncs{0};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void*) {
          if (std::this_thread::get_id() == w_thread_id.load()) {
            w_fsyncs.fetch_add(1);
            return;
          }
          if (std::this_thread::get_id() != g_thread_id.load()) {
            return;
          }
          std::unique_lock<std::mutex> lock(mu);
          if (runner_held) {
            return;
          }
          runner_held = true;
          cv.notify_all();
          cv.wait(lock, [&] { return release_runner; });
        });
    auto release = [&] {
      std::lock_guard<std::mutex> lock(mu);
      release_runner = true;
      cv.notify_all();
    };
    {
      GC30Write g;
      GC30Write w;
      Defer release_all([&] { release(); });
      g.Start([&] {
        g_thread_id.store(std::this_thread::get_id());
        return Put(cf_for_partition_[1], "g", "v", sync_wo);
      });
      {
        std::unique_lock<std::mutex> lock(mu);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                                [&] { return runner_held; }));
      }
      w.Start([&] {
        w_thread_id.store(std::this_thread::get_id());
        return Put(cf_for_partition_[2], "w", "v");
      });
      // W fsyncs G's other claimed partitions while G's runner is held.
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (w_fsyncs.load() == 0 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      EXPECT_GE(w_fsyncs.load(), 1);
      EXPECT_FALSE(w.Done());
      release();
      ASSERT_OK(g.Join());
      ASSERT_OK(w.Join());
    }
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
    EXPECT_EQ("v", Get(cf_for_partition_[1], "g"));
    EXPECT_EQ("v", Get(cf_for_partition_[2], "w"));
  }
}

// GC30 Q9: port of Muse's GC27 test (pwal/ab/gc27-muse-PR29 @ f9e5714d1306),
// unchanged. GetUpdatesSince at an N=1 opener of a partitioned set refuses a
// WAL whose head cannot be read with NotSupported, naming it. The damage
// sits mid-history, so the opener runs in skip mode; the MANIFEST WAL-index
// guard marks the DB indexed.
// Neuter: drop the re-list in GetUpdatesSince; the seqno pass fails first
// and it returns Corruption.
TEST_P(DBWritePartitionTest,
       GC27GetUpdatesSinceRefusesDamagedHeadOfPartitionedWal) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  const uint64_t p1_first = dbfull()->TEST_CurrentWalNumber(1);
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  ASSERT_OK(db_->SyncWAL());
  Close();
  ASSERT_OK(test::CorruptFile(env_, LogFileName(dbname_, p1_first), 0,
                              /*bytes_to_corrupt=*/1,
                              /*verify_checksum=*/false));
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = 1;
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  std::unique_ptr<TransactionLogIterator> iter;
  const Status s = db_->GetUpdatesSince(1, &iter);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos,
            s.ToString().find("WAL " + std::to_string(p1_first)))
      << s.ToString();
  EXPECT_EQ(nullptr, iter);
  Close();
}

TEST_P(DBWritePartitionTest, RepairRefusesDamagedHeadByManifestGuard) {
  // GC30 Q9: an indexed N=2 DB whose partition-1 WAL head is damaged, and
  // options that say wal_partitions = 1. The MANIFEST WAL-index guard is
  // set, so RepairDB treats the DB as indexed: it refuses with NotSupported
  // naming the WAL, and changes nothing in the directory.
  // Neuter: decide from the options alone (the base); RepairDB converts the
  // WALs one by one and returns OK.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  const uint64_t p1_wal = dbfull()->TEST_CurrentWalNumber(1);
  for (int i = 0; i < 8; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % 2], "key" + std::to_string(i), "value"));
  }
  ASSERT_OK(db_->SyncWAL());
  Close();
  uint64_t guard = 0;
  ASSERT_OK(VersionSet::ReadWALIndexGuard(dbname_, env_->GetFileSystem().get(),
                                          &guard));
  ASSERT_NE(0U, guard);
  ASSERT_OK(test::CorruptFile(env_, LogFileName(dbname_, p1_wal), 0,
                              /*bytes_to_corrupt=*/1,
                              /*verify_checksum=*/false));
  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = 1;
  std::vector<ColumnFamilyDescriptor> descriptors;
  for (const std::string& name : cf_names) {
    descriptors.emplace_back(name, options);
  }
  auto list_files = [&] {
    std::vector<std::string> files;
    EXPECT_OK(env_->GetChildren(dbname_, &files));
    files.erase(std::remove_if(files.begin(), files.end(),
                               [](const std::string& f) {
                                 return f == "LOG" ||
                                        f.rfind("LOG.old.", 0) == 0;
                               }),
                files.end());
    std::sort(files.begin(), files.end());
    return files;
  };
  const std::vector<std::string> before = list_files();
  const Status s = RepairDB(dbname_, options, descriptors);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(std::string::npos,
            s.ToString().find("WAL " + std::to_string(p1_wal)))
      << s.ToString();
  EXPECT_EQ(before, list_files());
}

TEST_P(DBWritePartitionTest, WalPartitionsOptionRefusalsUnchanged) {
  // GC30 Q7 guard (it passes at the base): at N in {2, 8} every write-path
  // option that N>1 refuses is still refused with the same status, naming
  // wal_partitions and the option: through DB::Open and, for the column
  // family options, CreateColumnFamily and (where mutable) SetOptions. The
  // timestamp-stripping refusal is GC29a's and is not pinned here.
  // Neuter: drop one refusal; its case opens.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  struct Refusal {
    const char* option;
    std::function<void(Options*)> set;
    bool invalid_argument;
    const char* fragment;
  };
  const std::vector<Refusal> db_refusals = {
      {"allow_concurrent_memtable_write",
       [](Options* o) { o->allow_concurrent_memtable_write = false; }, false,
       "allow_concurrent_memtable_write"},
      {"enable_pipelined_write",
       [](Options* o) { o->enable_pipelined_write = true; }, false,
       "enable_pipelined_write"},
      {"unordered_write", [](Options* o) { o->unordered_write = true; }, false,
       "unordered_write"},
      {"two_write_queues", [](Options* o) { o->two_write_queues = true; },
       false, "two_write_queues"},
      {"track_and_verify_wals",
       [](Options* o) { o->track_and_verify_wals = true; }, false,
       "track_and_verify_wals"},
      {"recycle_log_file_num", [](Options* o) { o->recycle_log_file_num = 1; },
       false, "recycle_log_file_num"},
      {"partition_wal_usage",
       [](Options* o) { o->partition_wal_usage = PartitionWALUsage::kNone; },
       false, "partition_wal_usage"},
      {"track_and_verify_wals_in_manifest",
       [](Options* o) { o->track_and_verify_wals_in_manifest = false; }, false,
       "track_and_verify_wals_in_manifest"},
      {"paranoid_checks", [](Options* o) { o->paranoid_checks = false; }, false,
       "paranoid_checks"},
  };
  const std::vector<Refusal> cf_refusals = {
      {"inplace_update_support",
       [](Options* o) { o->inplace_update_support = true; }, false,
       "inplace_update_support"},
      {"enable_blob_direct_write",
       [](Options* o) { o->enable_blob_direct_write = true; }, false,
       "enable_blob_direct_write"},
      {"max_successive_merges",
       [](Options* o) { o->max_successive_merges = 1; }, true,
       "max_successive_merges"},
      {"memtable_factory",
       [](Options* o) {
         o->memtable_factory.reset(test::NewSpecialSkipListFactory(1000));
       },
       false, "SkipList memtable"},
  };
  auto expect_refused = [](const Status& s, const Refusal& refusal,
                           const std::string& via) {
    SCOPED_TRACE(std::string(refusal.option) + " via " + via);
    if (refusal.invalid_argument) {
      EXPECT_TRUE(s.IsInvalidArgument()) << s.ToString();
    } else {
      EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
    }
    EXPECT_NE(std::string::npos, s.ToString().find("wal_partitions"))
        << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find(refusal.fragment))
        << s.ToString();
  };
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Options base = CurrentOptions();
    base.create_if_missing = true;
    base.wal_partitions = static_cast<uint32_t>(partitions);
    base.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    base.track_and_verify_wals_in_manifest = true;
    for (const std::vector<Refusal>* refusals : {&db_refusals, &cf_refusals}) {
      for (const Refusal& refusal : *refusals) {
        Options options = base;
        refusal.set(&options);
        Close();
        ASSERT_OK(DestroyDB(dbname_, base));
        expect_refused(TryReopen(options), refusal, "DB::Open");
      }
    }
    DestroyAndReopen(base);
    for (const Refusal& refusal : cf_refusals) {
      Options options = base;
      refusal.set(&options);
      ColumnFamilyHandle* handle = nullptr;
      expect_refused(db_->CreateColumnFamily(ColumnFamilyOptions(options),
                                             "refused", &handle),
                     refusal, "CreateColumnFamily");
      EXPECT_EQ(nullptr, handle);
    }
    // The only mutable one.
    expect_refused(db_->SetOptions(db_->DefaultColumnFamily(),
                                   {{"max_successive_merges", "1"}}),
                   cf_refusals[2], "SetOptions");
    ASSERT_OK(Put("after", "v"));
  }
}

TEST_P(DBWritePartitionTest, MergeBatchesOverlapAndStayCorrectAtN8) {
  // GC30 Q7: at N>1 Merge batches insert concurrently (max_successive_merges
  // is 0), and the overlap applies to them. Sixteen threads, two per
  // partition, each write batches of uint64 add merges over shared keys; the
  // groups they form insert their followers' merges while the leader
  // appends. Every key's value equals the sum a single-threaded reference
  // computes, before and after a reopen.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kThreadsPerPartition = 2;
  constexpr int kBatches = 300;
  constexpr int kKeysPerPartition = 4;
  Options options = CurrentOptions();
  options.merge_operator = MergeOperators::CreateUInt64AddOperator();
  ASSERT_EQ(0U, options.max_successive_merges);
  OpenWithPartitions(kPartitions, options);
  std::atomic<int> follower_inserts{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteImpl:FollowerMemTableInserted",
      [&](void*) { follower_inserts.fetch_add(1); });
  const uint32_t seed = static_cast<uint32_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  SCOPED_TRACE("seed=" + std::to_string(seed));
  // The plan: thread t adds plan[t][b][k] to key k of its partition in batch
  // b. The reference sums it single-threaded.
  const int num_threads = static_cast<int>(kPartitions) * kThreadsPerPartition;
  std::vector<std::vector<std::vector<uint64_t>>> plan(num_threads);
  std::map<std::pair<size_t, int>, uint64_t> reference;
  Random rnd(seed);
  for (int t = 0; t < num_threads; ++t) {
    const size_t partition = static_cast<size_t>(t) % kPartitions;
    plan[t].resize(kBatches);
    for (int b = 0; b < kBatches; ++b) {
      plan[t][b].resize(kKeysPerPartition);
      for (int k = 0; k < kKeysPerPartition; ++k) {
        plan[t][b][k] = rnd.Uniform(1000);
        reference[{partition, k}] += plan[t][b][k];
      }
    }
  }
  auto encode = [](uint64_t v) {
    std::string encoded;
    PutFixed64(&encoded, v);
    return encoded;
  };
  std::vector<port::Thread> threads;
  std::vector<Status> statuses(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      const size_t partition = static_cast<size_t>(t) % kPartitions;
      ColumnFamilyHandle* cf = handles_[cf_for_partition_[partition]];
      for (int b = 0; b < kBatches && statuses[t].ok(); ++b) {
        WriteBatch batch;
        for (int k = 0; k < kKeysPerPartition; ++k) {
          if (plan[t][b][k] != 0) {
            EXPECT_OK(batch.Merge(cf, "key" + std::to_string(k),
                                  encode(plan[t][b][k])));
          }
        }
        statuses[t] = db_->Write(WriteOptions(), &batch);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WriteImpl:FollowerMemTableInserted");
  for (const Status& s : statuses) {
    ASSERT_OK(s);
  }
  // Groups formed, so followers inserted their merges during an append.
  EXPECT_GT(follower_inserts.load(), 0);
  auto verify = [&] {
    for (const auto& [where, sum] : reference) {
      std::string value;
      ASSERT_OK(db_->Get(ReadOptions(),
                         handles_[cf_for_partition_[where.first]],
                         "key" + std::to_string(where.second), &value));
      ASSERT_EQ(sizeof(uint64_t), value.size());
      EXPECT_EQ(sum, DecodeFixed64(value.data()))
          << "partition " << where.first << " key " << where.second;
    }
  };
  verify();
  ReopenWithPartitions(kPartitions, options);
  verify();
}

TEST_P(DBWritePartitionTest, DroppedColumnFamilyInsertFailsOnceUnderOverlap) {
  // GC30 Q7: with ignore_missing_column_families=false, a follower's batch
  // for a dropped column family fails its insert while the leader's append
  // is held. The group fails once, as upstream: every write of the group
  // returns the insert's InvalidArgument, one background error is set
  // (kMemTable, which fences the group's range), and nothing of the group
  // becomes visible.
  // Neuter: drop the insert failure's classification on the group exit (or
  // run it twice); the background error count is not 1.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  Options options = CurrentOptions();
  auto listener = std::make_shared<GC30BgErrorCounter>();
  options.listeners.push_back(listener);
  constexpr size_t kPartitions = 2;
  OpenWithPartitions(kPartitions, options);
  const int cf = cf_for_partition_[1];
  // A column family routed to partition 1, dropped.
  ColumnFamilyHandle* dropped = nullptr;
  for (int i = 0; dropped == nullptr; ++i) {
    ColumnFamilyHandle* handle = nullptr;
    ASSERT_OK(db_->CreateColumnFamily(ColumnFamilyOptions(options),
                                      "candidate" + std::to_string(i),
                                      &handle));
    WriteBatch probe;
    ASSERT_OK(WriteBatchInternal::Put(&probe, handle->GetID(), "k", "v"));
    size_t partition = kPartitions;
    ASSERT_OK(
        DBImpl::TEST_WalPartitionForWrite(probe, kPartitions, &partition));
    if (partition == 1) {
      dropped = handle;
    } else {
      ASSERT_OK(db_->DestroyColumnFamilyHandle(handle));
    }
  }
  const uint32_t dropped_id = dropped->GetID();
  ASSERT_OK(db_->DropColumnFamily(dropped));
  ASSERT_OK(db_->DestroyColumnFamilyHandle(dropped));
  ASSERT_GT(dropped_id, static_cast<uint32_t>(cf));

  const std::vector<std::string> keys = {"k0", "k1"};
  std::vector<std::function<Status()>> writes = GC30Puts(this, cf, keys);
  writes.push_back([&] {
    WriteBatch batch;
    EXPECT_OK(WriteBatchInternal::Put(&batch, dropped_id, "lost", "v"));
    WriteOptions wo;
    wo.ignore_missing_column_families = false;
    return db_->Write(wo, &batch);
  });
  std::vector<Status> statuses;
  {
    GC30Group group;
    GC28Hold failed_insert_done;
    failed_insert_done.Arm(
        "DBImpl::WriteImpl:FollowerMemTableInserted",
        [](void* arg) {
          return !static_cast<WriteThread::Writer*>(arg)->status.ok();
        },
        /*hold=*/false);
    GC28Hold append;
    append.Arm("DBImpl::WriteToWAL:TakesWalMutex");
    ASSERT_TRUE(group.Start(writes));
    group.Lead();
    ASSERT_TRUE(append.WaitHeld());
    ASSERT_TRUE(failed_insert_done.WaitHits(1, std::chrono::seconds(10)))
        << "the dropped column family's insert failed during the append";
    EXPECT_EQ(0, group.Returned());
    EXPECT_EQ(0, listener->Count());
    append.Release();
    statuses = group.Join();
  }
  ASSERT_EQ(writes.size(), statuses.size());
  for (const Status& s : statuses) {
    EXPECT_TRUE(s.IsInvalidArgument()) << s.ToString();
  }
  EXPECT_EQ(1, listener->Count());
  ASSERT_NOK(dbfull()->TEST_GetBGError());
  GC30ExpectInvisible(db_.get(), handles_[cf], keys);
}

TEST_P(DBWritePartitionTest, LoneWriterGroupStillPublishesInOrder) {
  // GC30 Q7 (D-GC30b (i)): at N=8, one writer thread per partition, so every
  // group is a lone writer, whose leader appends and inserts on its own
  // thread. Every write is acknowledged only once published: when Put
  // returns, LastSequence() covers its sequence and it reads back. And
  // publication stays in sequence order: at every snapshot an observer
  // takes, the visible writes are exactly those at or below its sequence.
  // Neuter: publish a lone group's range at its insert, before its append
  // returns; a snapshot sees a write above one not yet inserted, or a Put
  // returns before LastSequence() covers it.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kWrites = 300;
  Options options = CurrentOptions();
  OpenWithPartitions(kPartitions, options);
  std::atomic<int> waited{0};
  thread_local SequenceNumber gate_last_sequence = 0;
  SyncPoint::GetInstance()->SetCallBack(
      "WriteThread::JoinBatchGroup:BeganWaiting",
      [&](void*) { waited.fetch_add(1); });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::InstallCompletionGate", [&](void* arg) {
        gate_last_sequence = *static_cast<SequenceNumber*>(arg);
      });
  std::mutex mu;
  // key -> (partition, sequence)
  std::map<std::string, std::pair<size_t, SequenceNumber>> written;
  std::atomic<bool> done{false};
  std::vector<std::pair<const Snapshot*, std::vector<std::set<std::string>>>>
      observed;
  port::Thread observer([&] {
    while (!done.load()) {
      const Snapshot* snapshot = db_->GetSnapshot();
      ReadOptions ro;
      ro.snapshot = snapshot;
      std::vector<std::set<std::string>> visible(kPartitions);
      for (size_t p = 0; p < kPartitions; ++p) {
        std::unique_ptr<Iterator> it(
            db_->NewIterator(ro, handles_[cf_for_partition_[p]]));
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
          visible[p].insert(it->key().ToString());
        }
        EXPECT_OK(it->status());
      }
      observed.emplace_back(snapshot, std::move(visible));
    }
  });
  std::vector<port::Thread> writers;
  for (size_t p = 0; p < kPartitions; ++p) {
    writers.emplace_back([&, p] {
      for (int i = 0; i < kWrites; ++i) {
        const std::string key =
            "p" + std::to_string(p) + "_" + std::to_string(1000 + i);
        gate_last_sequence = 0;
        ASSERT_OK(Put(cf_for_partition_[p], key, "v"));
        const SequenceNumber sequence = gate_last_sequence;
        ASSERT_NE(0U, sequence);
        EXPECT_GE(db_->GetLatestSequenceNumber(), sequence) << key;
        EXPECT_EQ("v", Get(cf_for_partition_[p], key)) << key;
        std::lock_guard<std::mutex> lock(mu);
        written[key] = {p, sequence};
      }
    });
  }
  for (auto& writer : writers) {
    writer.join();
  }
  done.store(true);
  observer.join();
  SyncPoint::GetInstance()->ClearCallBack(
      "WriteThread::JoinBatchGroup:BeganWaiting");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::InstallCompletionGate");
  // One writer per partition: no write ever queued behind another.
  EXPECT_EQ(0, waited.load());
  ASSERT_EQ(kPartitions * kWrites, written.size());
  EXPECT_FALSE(observed.empty());
  for (auto& [snapshot, visible] : observed) {
    const SequenceNumber at = snapshot->GetSequenceNumber();
    for (const auto& [key, where] : written) {
      const bool expected = where.second <= at;
      EXPECT_EQ(expected, visible[where.first].count(key) == 1)
          << key << " sequence " << where.second << " snapshot " << at;
    }
    db_->ReleaseSnapshot(snapshot);
  }
}

// GC31 tests: the write-path locks and shared writes at N > 1 outside the
// two allowed cross-partition sync points.
namespace {
// Holds every thread that reaches its sync point while armed, until
// Release(), and counts the arrivals.
class GC31Gate {
 public:
  GC31Gate() = default;
  GC31Gate(const GC31Gate&) = delete;
  GC31Gate& operator=(const GC31Gate&) = delete;
  ~GC31Gate() {
    Release();
    if (!point_.empty()) {
      SyncPoint::GetInstance()->ClearCallBack(point_);
    }
  }
  void Arm(const std::string& point) {
    point_ = point;
    SyncPoint::GetInstance()->SetCallBack(point, [this](void*) {
      std::unique_lock<std::mutex> lock(mu_);
      ++arrivals_;
      cv_.notify_all();
      cv_.wait(lock, [this] { return released_; });
    });
  }
  bool WaitArrivals(int n) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::seconds(30),
                        [this, n] { return arrivals_ >= n; });
  }
  void Release() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::string point_;
  std::mutex mu_;
  std::condition_variable cv_;
  int arrivals_ = 0;
  bool released_ = false;
};

// Blocks the HIGH and LOW background pools until Release() (or
// destruction), so a flush a switch schedules stays queued.
class GC31BlockedPools {
 public:
  explicit GC31BlockedPools(Env* env) {
    env->SetBackgroundThreads(1, Env::Priority::HIGH);
    env->SetBackgroundThreads(1, Env::Priority::LOW);
    env->Schedule(&test::SleepingBackgroundTask::DoSleepTask, &high_,
                  Env::Priority::HIGH);
    env->Schedule(&test::SleepingBackgroundTask::DoSleepTask, &low_,
                  Env::Priority::LOW);
    high_.WaitUntilSleeping();
    low_.WaitUntilSleeping();
  }
  ~GC31BlockedPools() { Release(); }
  void Release() {
    if (released_) {
      return;
    }
    released_ = true;
    high_.WakeUp();
    low_.WakeUp();
    high_.WaitUntilDone();
    low_.WaitUntilDone();
  }

 private:
  test::SleepingBackgroundTask high_;
  test::SleepingBackgroundTask low_;
  bool released_ = false;
};

// A FileSystem whose writable file for one WAL number blocks in Close()
// until Unblock().
class GC31BlockingCloseFS : public FileSystemWrapper {
 public:
  explicit GC31BlockingCloseFS(const std::shared_ptr<FileSystem>& target)
      : FileSystemWrapper(target) {}
  static const char* kClassName() { return "GC31BlockingCloseFS"; }
  const char* Name() const override { return kClassName(); }
  IOStatus NewWritableFile(const std::string& fname,
                           const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, file_opts, result, dbg);
    uint64_t number = 0;
    FileType type;
    if (s.ok() &&
        ParseFileName(fname.substr(fname.rfind('/') + 1), &number, &type) &&
        type == kWalFile) {
      result->reset(new File(std::move(*result), this, number));
    }
    return s;
  }
  void BlockClose(uint64_t number) {
    std::lock_guard<std::mutex> lock(mu_);
    blocked_number_ = number;
  }
  bool WaitBlocked() {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::seconds(30),
                        [this] { return blocked_; });
  }
  void Unblock() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  class File : public FSWritableFileOwnerWrapper {
   public:
    File(std::unique_ptr<FSWritableFile>&& t, GC31BlockingCloseFS* fs,
         uint64_t number)
        : FSWritableFileOwnerWrapper(std::move(t)), fs_(fs), number_(number) {}
    IOStatus Close(const IOOptions& options, IODebugContext* dbg) override {
      fs_->MaybeBlock(number_);
      return FSWritableFileOwnerWrapper::Close(options, dbg);
    }

   private:
    GC31BlockingCloseFS* const fs_;
    const uint64_t number_;
  };
  void MaybeBlock(uint64_t number) {
    std::unique_lock<std::mutex> lock(mu_);
    if (number != blocked_number_) {
      return;
    }
    blocked_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }
  std::mutex mu_;
  std::condition_variable cv_;
  uint64_t blocked_number_ = 0;
  bool blocked_ = false;
  bool released_ = false;
};

// Counts the hits of a sync point until destroyed.
class GC31Counter {
 public:
  explicit GC31Counter(const std::string& point) : point_(point) {
    SyncPoint::GetInstance()->SetCallBack(
        point_, [this](void*) { hits_.fetch_add(1); });
  }
  ~GC31Counter() { SyncPoint::GetInstance()->ClearCallBack(point_); }
  int Get() const { return hits_.load(); }
  void Reset() { hits_.store(0); }

 private:
  const std::string point_;
  std::atomic<int> hits_{0};
};
}  // namespace

TEST_P(DBWritePartitionTest, SwitchWalQuiesceSkippedWhileOldestWalFlushing) {
  // GC31 Q1 (A1): past max_total_wal_size with more than one column family,
  // a write stops every partition only when SwitchWAL will act. The first
  // switch marks the oldest WAL getting_flushed and schedules the flush,
  // which the blocked pools hold. Meanwhile every SwitchWAL would return at
  // once, so 200 writes from 8 threads quiesce no partition. Once the flush
  // runs the WAL retires, and the next write's switch acts, once.
  // Neuter: drop the pre-check; every write group quiesces.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kThreads = 8;
  constexpr int kWritesPerThread = 25;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 20;
  options.max_total_wal_size = 64 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  GC31BlockedPools pools(env_);
  GC31Counter quiesces("DBImpl::WaitForPendingWrites:BeforeBlock");
  GC31Counter switches("DBImpl::SwitchMemtable:AfterCreateWAL");
  const std::string value(1 << 10, 'v');
  for (int i = 0; switches.Get() == 0; ++i) {
    ASSERT_LT(i, 10000);
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "fill" + std::to_string(i), value));
  }
  ASSERT_EQ(1, quiesces.Get());
  quiesces.Reset();
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kWritesPerThread; ++i) {
        ASSERT_OK(Put(cf_for_partition_[t % kPartitions],
                      "t" + std::to_string(t) + "_" + std::to_string(i),
                      value));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(0, quiesces.Get());
  EXPECT_EQ(1, switches.Get());
  const uint64_t held_total = dbfull()->TEST_wals_total_size();
  ASSERT_GT(held_total, options.max_total_wal_size);

  pools.Release();
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  EXPECT_LT(dbfull()->TEST_wals_total_size(), held_total);
  ASSERT_GT(dbfull()->TEST_wals_total_size(), options.max_total_wal_size);
  quiesces.Reset();
  ASSERT_OK(Put(cf_for_partition_[0], "next", value));
  EXPECT_EQ(1, quiesces.Get());
  EXPECT_EQ(2, switches.Get());
}

TEST_P(DBWritePartitionTest, SwitchWalDecidedAgainAfterGateCloses) {
  // GC31 Q1 (A1), the decision after closing: D and A both find that the
  // switch acts. D closes the switch gate first and waits for C, held in
  // its append past the gate; A then waits for the gate. Once C is let go
  // D switches, marking the oldest WAL getting_flushed (the blocked pools
  // keep it so), and opens the gate. A closes it, finds the switch would no
  // longer act, and opens it again without draining any partition: one
  // quiesce in all, D's.
  // Neuter: decide only before closing; A drains too (two quiesces).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 20;
  options.max_total_wal_size = 256 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  GC31BlockedPools pools(env_);
  const std::string value(1 << 10, 'v');
  for (int i = 0; i < 64; ++i) {
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "fill" + std::to_string(i), value));
  }
  ASSERT_LT(dbfull()->TEST_wals_total_size(), options.max_total_wal_size);
  GC31Counter quiesces("DBImpl::WaitForPendingWrites:BeforeBlock");
  GC31Counter switches("DBImpl::SwitchMemtable:AfterCreateWAL");

  GC30Write c;
  GC30Write b;
  GC30Write d;
  GC30Write a;
  GC28Hold c_append;
  c_append.Arm("DBImpl::WriteToWAL:log_entry", [](void* arg) {
    return static_cast<Slice*>(arg)->ends_with("held");
  });
  c.Start([&] { return Put(cf_for_partition_[2], "c", "held"); });
  ASSERT_TRUE(c_append.WaitHeld());
  // B's own append takes the total past the limit; B then waits for C.
  std::atomic<std::thread::id> b_id;
  GC28Hold b_appended;
  b_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord", GC28OnThread(&b_id),
                 /*hold=*/false);
  b.Start([&] {
    b_id.store(std::this_thread::get_id());
    return Put(cf_for_partition_[0], "b", std::string(256 << 10, 'x'));
  });
  ASSERT_TRUE(b_appended.WaitHits(1));
  ASSERT_GT(dbfull()->TEST_wals_total_size(), options.max_total_wal_size);
  GC28Hold d_waits;
  d_waits.Arm("DBImpl::CloseMemTableSwitchGate:WaitForLeaders", nullptr,
              /*hold=*/false);
  d.Start([&] { return Put(cf_for_partition_[1], "d", "v"); });
  ASSERT_TRUE(d_waits.WaitHits(1));
  std::atomic<std::thread::id> a_id;
  GC28Hold a_decided;
  a_decided.Arm("DBImpl::PreprocessWrite:SwitchWalWouldAct",
                GC28OnThread(&a_id), /*hold=*/false);
  a.Start([&] {
    a_id.store(std::this_thread::get_id());
    return Put(cf_for_partition_[3], "a", "v");
  });
  ASSERT_TRUE(a_decided.WaitHits(1));
  // A now waits for the gate D closed.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(a.Done());
  c_append.Release();
  ASSERT_OK(c.Join());
  ASSERT_OK(b.Join());
  ASSERT_OK(d.Join());
  ASSERT_OK(a.Join());
  EXPECT_EQ(1, switches.Get());
  EXPECT_EQ(1, quiesces.Get());
  EXPECT_EQ("v", Get(cf_for_partition_[3], "a"));
}

TEST_P(DBWritePartitionTest, FlushQuiesceSkippedWhenSchedulerDrained) {
  // GC31 Q1 (A1), the flush path: two leaders on different partitions both
  // find a flush scheduled before taking mutex_; the first to take it
  // schedules the flush, and the second finds the scheduler drained and
  // closes no gate. Exactly one quiesce.
  // Neuter: no recheck under mutex_; both close the gate (two quiesces).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 10;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ColumnFamilyData* cfd = static_cast_with_check<ColumnFamilyHandleImpl>(
                              handles_[cf_for_partition_[0]])
                              ->cfd();
  for (int i = 0; !cfd->mem()->HasFlushScheduled(); ++i) {
    ASSERT_LT(i, 1000);
    ASSERT_OK(Put(cf_for_partition_[0], "fill" + std::to_string(i),
                  std::string(4 << 10, 'v')));
  }
  GC31Counter quiesces("DBImpl::WaitForPendingWrites:BeforeBlock");
  GC31Gate scheduled;
  scheduled.Arm("DBImpl::PreprocessWrite:FlushScheduled");
  GC30Write w1;
  GC30Write w2;
  w1.Start([&] { return Put(cf_for_partition_[1], "w1", "v"); });
  w2.Start([&] { return Put(cf_for_partition_[2], "w2", "v"); });
  ASSERT_TRUE(scheduled.WaitArrivals(2));
  scheduled.Release();
  ASSERT_OK(w1.Join());
  ASSERT_OK(w2.Join());
  EXPECT_EQ(1, quiesces.Get());
  EXPECT_EQ("v", Get(cf_for_partition_[1], "w1"));
  EXPECT_EQ("v", Get(cf_for_partition_[2], "w2"));
}

TEST_P(DBWritePartitionTest, DBStatsExactUnderConcurrentLeaders) {
  // GC31 Q2 (A2): leaders of eight partitions add the hot DB stats
  // concurrently, each to its own partition's shard. Every DB stat still
  // reports the exact total: keys, bytes, writes by self and by other, writes
  // with WAL, and the WAL bytes and syncs, which also equal their tickers.
  // Neuter: GetDBStats reads only shard 0; the totals fall short.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kThreads = 16;
  constexpr int kWritesPerThread = 100;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.write_buffer_size = 64 << 20;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  ASSERT_OK(dbfull()->ResetStats());
  ASSERT_OK(options.statistics->Reset());
  std::atomic<uint64_t> expected_bytes{0};
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const int cf = cf_for_partition_[t % kPartitions];
      WriteOptions sync_wo;
      sync_wo.sync = true;
      for (int i = 0; i < kWritesPerThread; ++i) {
        const std::string key =
            "t" + std::to_string(t) + "_" + std::to_string(i);
        const std::string value(1 + (t * 7 + i) % 300, 'v');
        WriteBatch batch;
        ASSERT_OK(batch.Put(handles_[cf], key, value));
        expected_bytes.fetch_add(batch.GetDataSize());
        ASSERT_OK(db_->Write(i % 10 == 0 ? sync_wo : WriteOptions(), &batch));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  constexpr uint64_t kWrites = kThreads * kWritesPerThread;
  std::map<std::string, std::string> stats;
  ASSERT_TRUE(db_->GetMapProperty(DB::Properties::kDBStats, &stats));
  auto stat = [&](const std::string& name) {
    return std::stoull(stats.at(name));
  };
  auto ticker = [&](Tickers t) {
    return options.statistics->getTickerCount(t);
  };
  EXPECT_EQ(kWrites, stat("db.user_keys_written"));
  // A group's merged size counts one batch header (WriteBatchInternal::
  // AppendedByteSize).
  EXPECT_EQ(expected_bytes.load() -
                WriteBatchInternal::kHeader * stat("db.user_writes_by_other"),
            stat("db.user_bytes_written"));
  EXPECT_EQ(kWrites,
            stat("db.user_writes_by_self") + stat("db.user_writes_by_other"));
  EXPECT_EQ(kWrites, stat("db.user_writes_with_wal"));
  EXPECT_EQ(ticker(WRITE_DONE_BY_SELF), stat("db.user_writes_by_self"));
  EXPECT_EQ(ticker(WRITE_DONE_BY_OTHER), stat("db.user_writes_by_other"));
  EXPECT_EQ(ticker(WAL_FILE_BYTES), stat("db.wal_bytes_written"));
  EXPECT_EQ(ticker(WAL_FILE_SYNCED), stat("db.wal_syncs"));
  EXPECT_GT(stat("db.wal_syncs"), 0U);
  uint64_t live_wal_bytes = 0;
  for (size_t p = 0; p < kPartitions; ++p) {
    for (const auto& file : dbfull()->TEST_AliveWalFiles(p)) {
      live_wal_bytes += file.second;
    }
  }
  EXPECT_EQ(live_wal_bytes, dbfull()->TEST_wals_total_size());
  // The text dump sums the same shards.
  std::string text;
  ASSERT_TRUE(db_->GetProperty(DB::Properties::kDBStats, &text));
  EXPECT_NE(std::string::npos, text.find(std::to_string(kWrites) + " writes, " +
                                         std::to_string(kWrites) + " keys"))
      << text;
  // Reset clears every shard.
  ASSERT_OK(dbfull()->ResetStats());
  stats.clear();
  ASSERT_TRUE(db_->GetMapProperty(DB::Properties::kDBStats, &stats));
  EXPECT_EQ(0U, stat("db.user_keys_written"));
  EXPECT_EQ(0U, stat("db.wal_bytes_written"));
  EXPECT_EQ(0U, stat("db.wal_syncs"));
}

TEST_P(DBWritePartitionTest, WalsTotalSizeSumsPartitions) {
  // GC31 Q2 (A8): each partition accounts its appends on its own counter.
  // The total is the sum of the live WALs' sizes on every partition, drops
  // by the retired WALs' sizes once a flush retires them, and the
  // max_total_wal_size trigger fires on the sum, which no single partition
  // reaches.
  // Neuter: read one partition's counter; the totals fall short and the
  // trigger never fires.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 20;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  auto live_wal_bytes = [&] {
    uint64_t total = 0;
    for (size_t p = 0; p < kPartitions; ++p) {
      for (const auto& file : dbfull()->TEST_AliveWalFiles(p)) {
        total += file.second;
      }
    }
    return total;
  };
  const std::string value(4 << 10, 'v');
  for (int i = 0; i < 40; ++i) {
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions], "k" + std::to_string(i),
                  value));
  }
  const uint64_t before = dbfull()->TEST_wals_total_size();
  EXPECT_EQ(live_wal_bytes(), before);
  EXPECT_GT(before, 4 * 10 * value.size());
  // Flush every family: the old WALs retire.
  for (int cf = 0; cf < static_cast<int>(handles_.size()); ++cf) {
    ASSERT_OK(Flush(cf));
  }
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  EXPECT_EQ(live_wal_bytes(), dbfull()->TEST_wals_total_size());
  EXPECT_LT(dbfull()->TEST_wals_total_size(), before / 4);

  // A limit no partition reaches alone, that every partition reaches
  // together: the write that takes the sum past it switches.
  GC31Counter switches("DBImpl::SwitchMemtable:AfterCreateWAL");
  const uint64_t limit =
      dbfull()->TEST_wals_total_size() + 3 * value.size() + value.size() / 2;
  ASSERT_OK(db_->SetDBOptions({{"max_total_wal_size", std::to_string(limit)}}));
  for (size_t p = 0; p < kPartitions; ++p) {
    ASSERT_OK(Put(cf_for_partition_[p], "over" + std::to_string(p), value));
  }
  EXPECT_EQ(0, switches.Get());
  ASSERT_GT(dbfull()->TEST_wals_total_size(), limit);
  ASSERT_OK(Put(cf_for_partition_[0], "trigger", "v"));
  EXPECT_EQ(1, switches.Get());
}

TEST_P(DBWritePartitionTest, UnpublishedSyncGroupWaiterNeverMissesWake) {
  // GC31 Q3 (A3): the end of a sync group takes mutex_ and signals only if a
  // waiter announced itself. A memtable switch (Flush), and then LockWAL(),
  // waits for a sync group whose fsync is held; between the waiter's check
  // and its wait (WaitForUnpublishedSyncGroups:Wait) the fsync is released
  // and the group ends. The waiter announced itself before its check, so the
  // ender signals it, and it returns.
  // Neuter: the ender never signals (skip the announcement check); the
  // waiter hangs and RunWithin fails.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (const bool lock_wal : {false, true}) {
    SCOPED_TRACE(lock_wal ? "LockWAL" : "Flush");
    GC30Write s;
    GC28Hold hold_fsync;
    hold_fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
                   GC28Partition(0));
    s.Start([&] {
      return Put(cf_for_partition_[0], lock_wal ? "s2" : "s1", "v", sync_wo);
    });
    ASSERT_TRUE(hold_fsync.WaitHeld());
    ASSERT_EQ(1U, dbfull()->TEST_UnpublishedSyncGroups());
    std::atomic<bool> in_window{false};
    std::atomic<bool> ended_in_window{false};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WaitForUnpublishedSyncGroups:Wait", [&](void*) {
          if (in_window.exchange(true)) {
            return;
          }
          hold_fsync.Release();
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds(10);
          while (dbfull()->TEST_UnpublishedSyncGroups() != 0 &&
                 std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }
          ended_in_window = dbfull()->TEST_UnpublishedSyncGroups() == 0;
        });
    ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
      if (!lock_wal) {
        return Flush(cf_for_partition_[1]);
      }
      Status st = db_->LockWAL();
      if (st.ok()) {
        st = db_->UnlockWAL();
      }
      return st;
    }));
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WaitForUnpublishedSyncGroups:Wait");
    EXPECT_TRUE(in_window.load());
    EXPECT_TRUE(ended_in_window.load());
    ASSERT_OK(s.Join());
  }
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s1"));
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s2"));
}

TEST_P(DBWritePartitionTest, EndSyncGroupWithoutWaiterTakesNoDbMutex) {
  // GC31 Q3 (A3): with no memtable switch or other waiter, ending a sync
  // group takes no DB mutex: 20 sequential sync writes never reach the
  // signalling branch of EndUnpublishedSyncGroup.
  // Neuter: always signal; the count is 20.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  GC31Counter signals("DBImpl::EndUnpublishedSyncGroup:Signal");
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int i = 0; i < 20; ++i) {
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions], "s" + std::to_string(i),
                  "v", sync_wo));
  }
  EXPECT_EQ(0, signals.Get());
  EXPECT_EQ(0U, dbfull()->TEST_UnpublishedSyncGroups());
}

TEST_P(DBWritePartitionTest, GateSyncTakesNoAllPartitionLock) {
  // GC31 Q4 (A4, A5): a gate sync reads the horizon under one partition's
  // mutex and finishes each claimed partition under its own: after a clean
  // sync write, ten sync writes take all N WAL mutexes zero times.
  // Neuter: take all N around the marking.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.statistics = CreateDBStatistics();
  options.write_buffer_size = 64 << 20;
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf_for_partition_[0], "clean", "v", sync_wo));
  ASSERT_OK(options.statistics->Reset());
  for (int i = 0; i < 10; ++i) {
    ASSERT_OK(
        Put(cf_for_partition_[i % kPartitions], "a" + std::to_string(i), "v"));
    ASSERT_OK(Put(cf_for_partition_[(i + 1) % kPartitions],
                  "s" + std::to_string(i), "v", sync_wo));
  }
  HistogramData all_wait;
  options.statistics->histogramData(WAL_WRITE_MUTEX_ALL_WAIT_MICROS, &all_wait);
  EXPECT_EQ(0U, all_wait.count);
  HistogramData fsynced;
  options.statistics->histogramData(WAL_SYNC_FSYNCED_PARTITIONS, &fsynced);
  EXPECT_EQ(10U, fsynced.count);
}

TEST_P(DBWritePartitionTest, AppendProceedsWhileGateSyncMarksOtherPartition) {
  // GC31 Q4 (A5): a gate sync finishes each claimed partition under that
  // partition's mutex alone. Held inside partition 0's finish, it leaves
  // partition 2 free: an async write there binds its WAL and appends.
  // Neuter: take all N around the marking; the append waits for the finish.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  GC30Write s;
  GC30Write w;
  GC28Hold finishing;
  finishing.Arm("DBImpl::SyncWalsFromCompletionGate:FinishPartition",
                GC28Partition(0));
  s.Start([&] { return Put(cf_for_partition_[0], "s", "v", sync_wo); });
  ASSERT_TRUE(finishing.WaitHeld());
  std::atomic<std::thread::id> w_id;
  GC28Hold w_appended;
  w_appended.Arm("DBImpl::WriteToWAL:AfterAddRecord", GC28OnThread(&w_id),
                 /*hold=*/false);
  w.Start([&] {
    w_id.store(std::this_thread::get_id());
    return Put(cf_for_partition_[2], "w", "v");
  });
  EXPECT_TRUE(w_appended.WaitHits(1, std::chrono::seconds(10)));
  finishing.Release();
  ASSERT_OK(s.Join());
  ASSERT_OK(w.Join());
  EXPECT_EQ("v", Get(cf_for_partition_[0], "s"));
  EXPECT_EQ("v", Get(cf_for_partition_[2], "w"));
}

TEST_P(DBWritePartitionTest, GateSyncMarkingRacesKeepAckedWrites) {
  // GC31 Q4 (A5) races: a gate sync claiming partitions 0 and 2 is held at
  // its entry, then between its two finishes (inside partition 2's), while
  // a forced rotation (a flush, whose background job runs SyncClosedWals and
  // FindObsoleteFiles) and a SyncWAL() start. Once released, everything
  // completes; after a power loss that drops every unsynced byte, a reopen
  // in kAbsoluteConsistency reads every acknowledged write the syncs
  // covered, and the MANIFEST records no WAL as synced beyond what was
  // fsynced (the reopen verifies the tracked WAL sizes).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Defer close_before_env([this] { Close(); });
  for (const char* point :
       {"DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex",
        "DBImpl::SyncWalsFromCompletionGate:FinishPartition"}) {
    SCOPED_TRACE(point);
    Options options = IndexedWalOptions(CurrentOptions());
    options.env = fault_env.get();
    options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
    Close();
    ASSERT_OK(DestroyDB(dbname_, options));
    Reopen(options);
    OpenWithPartitions(kPartitions, options);
    WriteOptions sync_wo;
    sync_wo.sync = true;
    std::vector<std::pair<int, std::string>> acked;
    {
      GC30Write s;
      GC30Write flush;
      GC30Write sync_wal;
      GC28Hold held;
      const bool between = std::strstr(point, "FinishPartition") != nullptr;
      held.Arm(point, between ? GC28Partition(2) : nullptr);
      ASSERT_OK(Put(cf_for_partition_[0], "a0", "v"));
      ASSERT_OK(Put(cf_for_partition_[2], "a2", "v"));
      s.Start([&] { return Put(cf_for_partition_[2], "s", "v", sync_wo); });
      ASSERT_TRUE(held.WaitHeld());
      flush.Start([&] {
        FlushOptions fo;
        fo.wait = false;
        return db_->Flush(fo, handles_[cf_for_partition_[1]]);
      });
      sync_wal.Start([&] { return db_->SyncWAL(); });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      held.Release();
      ASSERT_OK(s.Join());
      ASSERT_OK(flush.Join());
      ASSERT_OK(sync_wal.Join());
    }
    acked.emplace_back(cf_for_partition_[0], "a0");
    acked.emplace_back(cf_for_partition_[2], "a2");
    acked.emplace_back(cf_for_partition_[2], "s");
    ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
    ASSERT_OK(Put(cf_for_partition_[3], "b3", "v", sync_wo));
    acked.emplace_back(cf_for_partition_[3], "b3");
    for (size_t p = 0; p < kPartitions; ++p) {
      for (const auto& marks : dbfull()->TEST_WalSyncMarks(p)) {
        EXPECT_FALSE(marks.second) << "partition " << p;
      }
    }
    Close();
    ASSERT_OK(fault_env->DropUnsyncedFileData());
    ReopenWithPartitions(kPartitions, options);
    for (const auto& [cf, key] : acked) {
      EXPECT_EQ("v", Get(cf, key)) << key;
    }
    Close();
  }
}

TEST_P(DBWritePartitionTest, GateSyncDoesNotWaitBehindParkedMarker) {
  // GC31 Q6 (A7, D-GC31e (i)): no marker waits for another marker's sync
  // while holding a lock or a mark other syncs need. A one-family DB at N=4
  // writes only on partition r. A flush's obsolete-file cleanup closes every
  // partition's obsolete WAL in ascending order, marking each while it
  // closes it, and the close of partition q's (q above r) is held. A user
  // SyncWAL() marks the partitions below q, r included, finds q marked,
  // releases its marks and parks on q. A sync write on partition r runs its
  // gate sync and returns meanwhile.
  // (A flush's SyncClosedWals cannot be the held marker here: with coupled
  // rotation every partition has a closed WAL, and SyncClosedWals marks
  // every one, r's included.)
  // Neuter: wait while holding the marks below the parked partition (D-GC31e
  // (ii)), or restore the global mark mutex; the sync write waits for the
  // held close.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  WriteBatch probe;
  ASSERT_OK(WriteBatchInternal::Put(&probe, 0, "key", "value"));
  size_t r = kPartitions;
  ASSERT_OK(DBImpl::TEST_WalPartitionForWrite(probe, kPartitions, &r));
  ASSERT_LT(r + 1, kPartitions);
  const size_t q = r + 1;
  auto fs = std::make_shared<GC31BlockingCloseFS>(env_->GetFileSystem());
  std::unique_ptr<Env> env(NewCompositeEnv(fs));
  Defer close_before_env([this] { Close(); });
  Options options = CurrentOptions();
  options.env = env.get();
  options.wal_partitions = kPartitions;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  options.avoid_flush_during_shutdown = true;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
  DestroyAndReopen(options);
  ASSERT_EQ(kPartitions, dbfull()->TEST_NumWalPartitions());
  SyncPoint::GetInstance()->EnableProcessing();
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put("a", "v"));
  fs->BlockClose(dbfull()->TEST_CurrentWalNumber(q));
  Defer unblock([&] { fs->Unblock(); });
  {
    // The flush rotates now and cleans up once the pools run. A sync write
    // in between syncs every closed WAL (each stays open), so later gate
    // syncs need none of them.
    GC31BlockedPools pools(env_);
    FlushOptions no_wait;
    no_wait.wait = false;
    ASSERT_OK(db_->Flush(no_wait));
    ASSERT_OK(Put("r0", "v", sync_wo));
  }
  ASSERT_TRUE(fs->WaitBlocked());
  ASSERT_EQ(1U, dbfull()->TEST_WalSyncMarks(r).size());
  ASSERT_TRUE(dbfull()->TEST_WalSyncMarks(q).front().second);
  GC30Write sync_wal;
  GC30Write r_write;
  GC28Hold parked;
  parked.Arm("DBImpl::ClaimWalPartitionForSyncWalImpl:Wait", nullptr,
             /*hold=*/false);
  sync_wal.Start([&] { return db_->SyncWAL(); });
  ASSERT_TRUE(parked.WaitHits(1));
  r_write.Start([&] { return Put("r", "v", sync_wo); });
  EXPECT_TRUE(r_write.WaitDone(std::chrono::seconds(10)));
  EXPECT_FALSE(sync_wal.Done());
  fs->Unblock();
  ASSERT_OK(r_write.Join());
  ASSERT_OK(sync_wal.Join());
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable());
  for (size_t p = 0; p < kPartitions; ++p) {
    for (const auto& marks : dbfull()->TEST_WalSyncMarks(p)) {
      EXPECT_FALSE(marks.second) << "partition " << p;
    }
  }
  EXPECT_EQ("v", Get("r"));
}

TEST_P(DBWritePartitionTest, MarkersRetryWithoutDeadlock) {
  // GC31 Q6 (A7, D-GC31e (i)): markers with overlapping claims, in each
  // order, both finish. (a) SyncWAL() marks every partition and is held in
  // its fsync; a sync write needing partition 2 finds it marked and waits
  // holding nothing. (b) A sync write claiming partitions 0 and 2 is held in
  // partition 0's fsync; SyncWAL() finds partition 0 marked and waits
  // holding nothing. Then SyncWAL(), FlushWAL(true) and sync writes on every
  // partition race; every one finishes.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  OpenWithPartitions(kPartitions, options);
  WriteOptions sync_wo;
  sync_wo.sync = true;
  {
    GC30Write sync_wal;
    GC30Write s;
    GC28Hold held;
    ASSERT_OK(Put(cf_for_partition_[2], "a", "v"));
    held.Arm("DBImpl::SyncWalImpl:BeforeSyncWAL");
    sync_wal.Start([&] { return db_->SyncWAL(); });
    ASSERT_TRUE(held.WaitHeld());
    s.Start([&] { return Put(cf_for_partition_[2], "s", "v", sync_wo); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(s.Done());
    held.Release();
    ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
      Status st = sync_wal.Join();
      return st.ok() ? s.Join() : st;
    }));
  }
  {
    GC30Write s;
    GC30Write sync_wal;
    GC28Hold held;
    ASSERT_OK(Put(cf_for_partition_[0], "b", "v"));
    held.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
             GC28Partition(0));
    s.Start([&] { return Put(cf_for_partition_[2], "t", "v", sync_wo); });
    ASSERT_TRUE(held.WaitHeld());
    GC28Hold parked;
    parked.Arm("DBImpl::ClaimWalPartitionForSyncWalImpl:Wait", nullptr,
               /*hold=*/false);
    sync_wal.Start([&] { return db_->SyncWAL(); });
    ASSERT_TRUE(parked.WaitHits(1));
    held.Release();
    ASSERT_OK(RunWithin(std::chrono::seconds(30), [&] {
      Status st = s.Join();
      return st.ok() ? sync_wal.Join() : st;
    }));
  }
  ASSERT_OK(RunWithin(std::chrono::seconds(60), [&] {
    std::vector<port::Thread> threads;
    std::atomic<bool> failed{false};
    for (size_t p = 0; p < kPartitions; ++p) {
      threads.emplace_back([&, p] {
        for (int i = 0; i < 50; ++i) {
          if (!Put(cf_for_partition_[p], "x" + std::to_string(i), "v",
                   i % 2 == 0 ? sync_wo : WriteOptions())
                   .ok()) {
            failed = true;
          }
        }
      });
    }
    threads.emplace_back([&] {
      for (int i = 0; i < 50; ++i) {
        if (!db_->SyncWAL().ok() || !db_->FlushWAL(/*sync=*/true).ok()) {
          failed = true;
        }
      }
    });
    for (auto& thread : threads) {
      thread.join();
    }
    return failed.load() ? Status::Corruption("a marker failed") : Status::OK();
  }));
  for (size_t p = 0; p < kPartitions; ++p) {
    for (const auto& marks : dbfull()->TEST_WalSyncMarks(p)) {
      EXPECT_FALSE(marks.second) << "partition " << p;
    }
  }
}

namespace {
// A FileSystem whose WAL appends block, while armed, on data containing
// "blockme", until Unblock(). Works without sync points (release builds).
class GC31BlockingAppendFS : public FileSystemWrapper {
 public:
  explicit GC31BlockingAppendFS(const std::shared_ptr<FileSystem>& target)
      : FileSystemWrapper(target) {}
  static const char* kClassName() { return "GC31BlockingAppendFS"; }
  const char* Name() const override { return kClassName(); }
  IOStatus NewWritableFile(const std::string& fname,
                           const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, file_opts, result, dbg);
    if (s.ok() && fname.size() > 4 &&
        fname.compare(fname.size() - 4, 4, ".log") == 0) {
      result->reset(new File(std::move(*result), this));
    }
    return s;
  }
  bool WaitBlocked() {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::seconds(30),
                        [this] { return blocked_; });
  }
  void Unblock() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  class File : public FSWritableFileOwnerWrapper {
   public:
    File(std::unique_ptr<FSWritableFile>&& t, GC31BlockingAppendFS* fs)
        : FSWritableFileOwnerWrapper(std::move(t)), fs_(fs) {}
    IOStatus Append(const Slice& data, const IOOptions& options,
                    IODebugContext* dbg) override {
      fs_->MaybeBlock(data);
      return FSWritableFileOwnerWrapper::Append(data, options, dbg);
    }
    IOStatus Append(const Slice& data, const IOOptions& options,
                    const DataVerificationInfo& info,
                    IODebugContext* dbg) override {
      fs_->MaybeBlock(data);
      return FSWritableFileOwnerWrapper::Append(data, options, info, dbg);
    }

   private:
    GC31BlockingAppendFS* const fs_;
  };
  void MaybeBlock(const Slice& data) {
    if (data.ToString().find("blockme") == std::string::npos) {
      return;
    }
    std::unique_lock<std::mutex> lock(mu_);
    blocked_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
  }
  std::mutex mu_;
  std::condition_variable cv_;
  bool blocked_ = false;
  bool released_ = false;
};

// Runs, at `num_partitions`, a write group {leader, follower} on the default
// column family whose follower's batch (a few thousand puts, then one for a
// dropped column family) fails its insert last, so the follower runs the
// group's exit. The group forms behind a write whose WAL append blocks.
// Returns the leader's and the follower's statuses.
void GC31FollowerExitGroup(DBTestBase* test, size_t num_partitions,
                           Status* leader_status, Status* follower_status) {
  auto fs = std::make_shared<GC31BlockingAppendFS>(test->env_->GetFileSystem());
  std::unique_ptr<Env> env(NewCompositeEnv(fs));
  // DestroyAndReopen destroys with the last options, whose env goes away.
  Defer close_before_env([test] {
    test->Close();
    test->last_options_.env = test->env_;
  });
  Options options = test->CurrentOptions();
  options.env = env.get();
  options.wal_partitions = static_cast<uint32_t>(num_partitions);
  if (num_partitions > 1) {
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    options.track_and_verify_wals_in_manifest = true;
  }
  options.allow_concurrent_memtable_write = true;
  test->DestroyAndReopen(options);
  DB* db = test->db_.get();
  ColumnFamilyHandle* dropped = nullptr;
  ASSERT_OK(db->CreateColumnFamily(options, "dropped", &dropped));
  const uint32_t dropped_id = dropped->GetID();
  ASSERT_OK(db->DropColumnFamily(dropped));
  ASSERT_OK(db->DestroyColumnFamilyHandle(dropped));

  GC30Write blocker;
  GC30Write leader;
  GC30Write follower;
  Defer unblock([&] { fs->Unblock(); });
  blocker.Start([&] { return db->Put(WriteOptions(), "a", "blockme"); });
  ASSERT_TRUE(fs->WaitBlocked());
  leader.Start([&] { return db->Put(WriteOptions(), "leader", "v"); });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  follower.Start([&] {
    WriteBatch batch;
    for (int i = 0; i < 4000; ++i) {
      EXPECT_OK(batch.Put("f" + std::to_string(i), "v"));
    }
    EXPECT_OK(WriteBatchInternal::Put(&batch, dropped_id, "lost", "v"));
    return db->Write(WriteOptions(), &batch);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fs->Unblock();
  ASSERT_OK(blocker.Join());
  *leader_status = leader.Join();
  *follower_status = follower.Join();
  test->Close();
}

}  // namespace

TEST_P(DBWritePartitionTest, N1FollowerExitMatchesUpstream) {
  // GC31 Q11: at one WAL partition the follower exit is upstream's. A
  // follower's insert fails (a dropped column family) and the follower runs
  // the group's exit: upstream's assertion that the group succeeded fires in
  // debug builds, and in release builds the leader returns OK while the
  // follower returns the failure.
  // Neuter: GC30's unconditional hand-off of a failed group status to the
  // leader; no assertion fires (debug), the leader returns InvalidArgument
  // (release).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("single-queue path only");
    return;
  }
#ifdef NDEBUG
  Status leader_status;
  Status follower_status;
  GC31FollowerExitGroup(this, 1, &leader_status, &follower_status);
  EXPECT_OK(leader_status);
  EXPECT_TRUE(follower_status.IsInvalidArgument())
      << follower_status.ToString();
#else
#ifdef GTEST_FLAG_SET
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#else
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif
  EXPECT_DEATH(
      {
        Status leader_status;
        Status follower_status;
        GC31FollowerExitGroup(this, 1, &leader_status, &follower_status);
      },
      "write_group->status\\.ok\\(\\)");
#endif  // NDEBUG
}

TEST_P(DBWritePartitionTest, NGreaterThan1FollowerExitHandsFailureToLeader) {
  // GC31 Q11: above one partition GC30's follower exit stays: the same group,
  // whose follower's insert fails and whose exit the follower runs, fails as
  // one, the leader included.
  // Neuter: upstream's follower exit at every N; the leader returns OK (and
  // debug builds assert).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("partitions=" + std::to_string(partitions));
    Status leader_status;
    Status follower_status;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::ValidateOptions:AllowWalPartitions",
        [](void* arg) { *static_cast<bool*>(arg) = true; });
    SyncPoint::GetInstance()->EnableProcessing();
    GC31FollowerExitGroup(this, partitions, &leader_status, &follower_status);
    EXPECT_TRUE(leader_status.IsInvalidArgument()) << leader_status.ToString();
    EXPECT_TRUE(follower_status.IsInvalidArgument())
        << follower_status.ToString();
  }
}

TEST_P(DBWritePartitionTest, FailedSyncDropKeepsBelowFenceRangeTombstone) {
  // GC31 Q13 (e): ported from Muse's GC30 candidate (pwal/ab/gc30-muse-PR34
  // @ ef4defc379c3), adapted to this base. A flush dropping a failed range
  // above a fence keeps the range tombstones below it. S, acknowledged below
  // the fence, deletes "f" via a range tombstone; F, a sync write above it
  // with its own tombstone, fails its gate sync and is dropped. After Resume
  // and a flush, "f" stays deleted (the below-fence tombstone persisted),
  // "z" reads back, and F's key is absent everywhere, SSTs included.
  // Neuter: drop the below-fence tombstone filter in the recovery flush.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  WriteOptions sync_wo;
  sync_wo.sync = true;
  ASSERT_OK(Put(cf, "f", "value_f"));
  {
    WriteBatch batch;
    ASSERT_OK(batch.DeleteRange(handles_[cf], "a", "m"));
    ASSERT_OK(batch.Put(handles_[cf], "z", "value_z"));
    ASSERT_OK(db_->Write(sync_wo, &batch));
  }
  // F fails its gate sync once, above the tombstone.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
        IOStatus error = IOStatus::IOError("injected gate sync error");
        error.SetRetryable(true);
        *static_cast<IOStatus*>(arg) = error;
      });
  {
    WriteBatch batch;
    ASSERT_OK(batch.DeleteRange(handles_[cf], "p", "t"));
    ASSERT_OK(batch.Put(handles_[cf], "g", "value_g"));
    const Status s = db_->Write(sync_wo, &batch);
    ASSERT_TRUE(s.IsIOError()) << s.ToString();
  }
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
  ASSERT_OK(dbfull()->Resume());
  ASSERT_OK(Flush(cf));
  EXPECT_EQ("NOT_FOUND", Get(cf, "f"));
  EXPECT_EQ("value_z", Get(cf, "z"));
  EXPECT_EQ("NOT_FOUND", Get(cf, "g"));
  EXPECT_FALSE(GC30LiveSstHasKey(db_.get(), handles_[cf], "g"));
  Close();
}

TEST_P(DBWritePartitionTest, ManualWalFlushSyncWriteSurvivesCrash) {
  // manual_wal_flush with sync writes at N > 1: async A on partition 0 and B
  // on partition 2 sit in their WAL buffers until sync S on partition 1 runs
  // its gate sync, which flushes and fsyncs every partition. A crash that
  // drops every unsynced byte keeps A, B and S; a later async write, never
  // flushed, is lost, proving the crash dropped unsynced data.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
  EXPECT_OK(batch.Put(handles_[cf_for_partition_[0]], "key", "value"));
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
    if (!wait_for(held_in_append)) {
      // Stop before the stall begins: it would block `held` short of its
      // append until the stall ends, after held.join().
      {
        std::lock_guard<std::mutex> lock(mu);
        release_held = true;
        cv.notify_all();
      }
      held.join();
      SyncPoint::GetInstance()->DisableProcessing();
      SyncPoint::GetInstance()->ClearAllCallBacks();
      FAIL() << "the held write never reached its append";
    }
    port::Thread queued([&] {
      ASSERT_TRUE(
          Put(cf_for_partition_[1], "queued", "v", no_slowdown).IsIncomplete());
    });
    EXPECT_TRUE(wait_for(queued_linked));

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
      EXPECT_TRUE(wbm->ShouldStall());
      EXPECT_FALSE(wbm->ShouldFlush());
    }
    port::Thread stalled(
        [&] { ASSERT_OK(Put(cf_for_partition_[0], "stalled", "v")); });
    EXPECT_TRUE(wait_for(stall_begun));

    for (size_t p : {size_t{0}, size_t{2}, size_t{3}}) {
      EXPECT_TRUE(
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
    EXPECT_TRUE(
        Put(cf_for_partition_[1], "ns1", "v", no_slowdown).IsIncomplete());
    EXPECT_EQ("v", Get(cf_for_partition_[1], "held"));
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "queued"));
    for (size_t p = 0; p < kPartitions; ++p) {
      EXPECT_EQ("NOT_FOUND",
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
  // The callbacks capture this frame's locals.
  Defer clear_callbacks([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });
  int exited = 0;
  auto note_exit = [&] {
    std::lock_guard<std::mutex> lock(mu);
    ++exited;
    cv.notify_all();
  };
  bool reserved = false;
  // On a timeout, releases L0's hold, ends a stall still in force, and
  // reports the failure once the `started` leaders have exited; a leader
  // that stays stalled cannot be joined, so then it aborts with the reason
  // instead of hanging.
  auto wait_until = [&](const std::function<bool()>& pred, const char* what,
                        int started) {
    {
      std::unique_lock<std::mutex> lock(mu);
      if (cv.wait_for(lock, std::chrono::seconds(30), pred)) {
        return true;
      }
      release_l0 = true;
      cv.notify_all();
    }
    if (reserved) {
      reserved = false;
      wbm->FreeMem(kBufferSize);
    }
    std::unique_lock<std::mutex> lock(mu);
    if (!cv.wait_for(lock, std::chrono::seconds(10),
                     [&] { return exited == started; })) {
      fprintf(stderr, "%s\n", what);
      std::abort();
    }
    ADD_FAILURE() << what;
    return false;
  };

  wbm->ReserveMem(kBufferSize);
  wbm->ScheduleFreeMem(kBufferSize);
  reserved = true;
  port::Thread l0([&] {
    Defer exit_note(note_exit);
    hold_this_writer = true;
    ASSERT_OK(Put(cf_for_partition_[0], "l0", "v"));
    hold_this_writer = false;
    ++done;
    cv.notify_all();
  });
  if (!wait_until([&] { return before_block == 1; }, "L0 never stalled", 1)) {
    l0.join();
    return;
  }
  // The stall ends (L0 is signaled), then a new one begins before L0
  // rechecks.
  wbm->FreeMem(kBufferSize);
  wbm->ReserveMem(kBufferSize);
  wbm->ScheduleFreeMem(kBufferSize);
  port::Thread l1([&] {
    Defer exit_note(note_exit);
    ASSERT_OK(Put(cf_for_partition_[1], "l1", "v"));
    ++done;
    cv.notify_all();
  });
  auto join_both = [&] {
    l0.join();
    l1.join();
  };
  if (!wait_until([&] { return before_block == 2; }, "L1 never stalled", 2)) {
    join_both();
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    release_l0 = true;
    cv.notify_all();
  }
  // L0 either passed on its own wakeup or is waiting again; only then end
  // the second stall, so a lost wakeup cannot be masked by timing.
  if (!wait_until([&] { return done.load() >= 1 || l0_blocked_again; },
                  "L0 neither passed nor stalled again", 2)) {
    join_both();
    return;
  }
  reserved = false;
  wbm->FreeMem(kBufferSize);
  if (!wait_until([&] { return done.load() == 2; },
                  "a stalled leader was never woken", 2)) {
    join_both();
    return;
  }
  join_both();
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
        EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30), [&] {
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
      EXPECT_EQ(0U, done.load());
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
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
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
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
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
  EXPECT_OK(DestroyDB(checkpoint_dir, options));
  Checkpoint* checkpoint = nullptr;
  EXPECT_OK(Checkpoint::Create(db_.get(), &checkpoint));
  EXPECT_OK(checkpoint->CreateCheckpoint(checkpoint_dir,
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
      EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_held; }));
    }
    port::Thread b_thread([&] {
      b_status = Put(cf_b, "key_b", "value_b");
      b_returned.store(true);
    });
    // B parks in its completion gate only after its memtable insert.
    EXPECT_TRUE(WaitForDurabilityWaiters(1));
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
      EXPECT_OK(it->status());
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

// A sync write's record lands and its entry goes into the memtable (sync
// writers insert before their fsync), then recording the synced WAL in the
// MANIFEST fails with no background error set. This pins
// FailUnpublishedSyncRange's branch for a failed range that may be in a
// memtable: it sets the background error, pins the range and fences it.
// Without that branch the range would stay pending in the applied prefix,
// and a later write on the other partition would park for good in its
// completion gate (the M2 prefix-oracle hang, seed 20261023); with it, that
// write returns the background error and neither write is visible.
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
  // GC30 (D-GC30d (ii)): the sync write was in the memtable before its
  // fsync, so its failed MANIFEST record fences its range, with a background
  // error, instead of publishing it as a hole. B, above it, returns that
  // error instead of waiting forever, and neither write is visible.
  ASSERT_NOK(b_status);
  EXPECT_EQ(dbfull()->TEST_GetBGError().ToString(), b_status.ToString());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "b"));
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "a"));
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
    EXPECT_OK(writer.Open(sst));
    EXPECT_OK(writer.Put(key, "ingested"));
    EXPECT_OK(writer.Finish());
    EXPECT_OK(db_->IngestExternalFile(handles_[cf_for_partition_[p]], {sst},
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
    EXPECT_TRUE(cv.wait_until(lock, deadline, [&] { return a_held; }));
  }
  port::Thread b_thread(
      [&] { Put(cf_b, "key_b", "v").PermitUncheckedError(); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  EXPECT_LT(db_->GetLatestSequenceNumber(), seq_a);

  // D-G32 (a): at N > 1 GetUpdatesSince refuses before reading any file, so
  // no change stream exists to run past A.
  {
    std::unique_ptr<TransactionLogIterator> iter;
    const Status s = db_->GetUpdatesSince(first, &iter);
    EXPECT_TRUE(s.IsNotSupported()) << s.ToString();
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
  // Review-M3 T7: A never landed, and B, above it, is not recovered across
  // the hole.
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "key_a"));
  EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "key_b"));
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
      : FileSystemWrapper(base) {
    error_.PermitUncheckedError();
  }
  static const char* kClassName() { return "FailingWalFS"; }
  const char* Name() const override { return kClassName(); }

  // How much of a failing append reaches the file before the error returns.
  enum class Landing { kNothing, kWhole, kTorn };

  // `appends_before_failure` appends to the WAL still succeed once armed.
  void Arm(uint64_t wal_number, const IOStatus& error,
           Landing landing = Landing::kNothing,
           int appends_before_failure = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    wal_number_ = wal_number;
    error_ = error;
    error_.PermitUncheckedError();
    landing_ = landing;
    appends_before_failure_ = appends_before_failure;
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
      IOStatus s = fs_->CheckAppend(number_, &landing);
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
      IOStatus s = fs_->CheckAppend(number_, &landing);
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
  IOStatus CheckAppend(uint64_t number, Landing* landing) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (number == wal_number_ && appends_before_failure_ > 0) {
        --appends_before_failure_;
        return IOStatus::OK();
      }
    }
    return Check(number, landing);
  }

  std::mutex mu_;
  uint64_t wal_number_ = 0;
  IOStatus error_;
  Landing landing_ = Landing::kNothing;
  int appends_before_failure_ = 0;
};

class DBWalAppendFailureTest : public DBWritePartitionTest {
 public:
  // Clears a flag a failed earlier test may have left set (see Writer).
  DBWalAppendFailureTest()
      : fail_fs_(std::make_shared<FailingWalFS>(FileSystem::Default())),
        fail_env_(NewCompositeEnv(fail_fs_)) {
    test_holds_abandoned = false;
  }
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
  // instead of hanging: nothing could release a write parked for good. A
  // Writer a failed assertion leaves unjoined is joined by its destructor
  // the same way, so the failure is reported instead of std::terminate; the
  // destructor first abandons every test hold (each waits through
  // WaitForReleaseOrAbandon), so a held write goes on.
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
    ~Writer() {
      if (thread_.joinable()) {
        test_holds_abandoned = true;
        Join().PermitUncheckedError();
      }
    }
    // Whether the write returned within `timeout`; it can still be joined.
    bool WaitDone(std::chrono::seconds timeout) {
      std::unique_lock<std::mutex> lock(mu_);
      return cv_.wait_for(lock, timeout, [&] { return done_; });
    }
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

  // Holds the next WAL append at its start until ReleaseHeldAppend(), or
  // until a failed assertion abandons the test's holds (see Writer).
  void HoldNextAppend() {
    {
      std::lock_guard<std::mutex> lock(hold_mu_);
      hold_armed_ = true;
      held_ = false;
      release_ = false;
    }
    test_holds_abandoned = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [this](void*) {
          std::unique_lock<std::mutex> lock(hold_mu_);
          if (!hold_armed_) {
            return;
          }
          hold_armed_ = false;
          held_ = true;
          hold_cv_.notify_all();
          WaitForReleaseOrAbandon(lock, hold_cv_, [this] { return release_; });
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
  // refuses paranoid_checks=false there and names both options. Without the
  // test hook N>1 is still refused first; N=1 is unaffected.
  Options options = CurrentOptions();
  options.wal_partitions = 2;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  options.paranoid_checks = false;
  Destroy(options);
  Status s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  EXPECT_NE(nullptr, std::strstr(s.getState(), "not supported yet"))
      << s.ToString();

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
  s = TryReopen(options);
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
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
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
          WaitForReleaseOrAbandon(lock, cv, [&] { return follower_linked; });
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
  // The sync write appends to partition 0; its gate fsync fails on
  // partition 1's file, which holds a record below it (GC28 Q1: a sync
  // fsyncs only the partitions it depends on), written before the fault
  // armed.
  ExpectPartitionOneQuarantined(this, FailureOptions(), [this] {
    const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(1);
    fail_fs_->Disarm();
    EXPECT_OK(Put(cf_for_partition_[1], "below", "value"));
    fail_fs_->Arm(failed_wal, FileScoped("injected"));
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

  // Review-M3 T7: the failed write is absent while writes are stopped.
  // Resume() recovers from a file-scoped WAL error above one partition
  // (GC32), and the write stays absent after it.
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "failed"));
  ASSERT_OK(dbfull()->Resume());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "failed"));
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ("value", Get(cf_b, "k" + std::to_string(i)));
  }
  // Its record landed whole before the injected failure, but the resume
  // retired the WAL holding it, so the reopen that follows a successful
  // Resume() does not recover it either (D-1's reopen rule covers only a
  // reopen without one).
  ReopenWithPartitions(2, options);
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "failed"));
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ("value", Get(cf_b, "k" + std::to_string(i)));
  }
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
  // with WAL tracking in the MANIFEST: N=1 with that tracking stays stopped
  // until reopened. Above one partition the tracking no longer refuses
  // (GC32, D-F1a (i)), but a configuration the guard still refuses (D-F1c),
  // here manual_wal_flush, stays stopped exactly as N=1 with tracking does,
  // with the same status. The failed range is never released and writes
  // fail instead of hanging.
  Options options = FailureOptions();
  options.wal_partitions = 1;
  DestroyAndReopen(options);
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), FileScoped("injected"));
  ASSERT_NOK(Put("failed", "value"));
  fail_fs_->Disarm();
  const Status resume_1 = dbfull()->Resume();

  options.manual_wal_flush = true;
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  ASSERT_OK(dbfull()->FlushWAL(/*sync=*/false));
  ASSERT_OK(Put(cf_for_partition_[1], "failed", "value"));
  fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), FileScoped("injected"));
  ASSERT_NOK(dbfull()->FlushWAL(/*sync=*/false));
  fail_fs_->Disarm();
  const Status resume_n = dbfull()->Resume();
  ASSERT_TRUE(resume_1.IsNotSupported()) << resume_1.ToString();
  EXPECT_EQ(resume_1.ToString(), resume_n.ToString());
  for (size_t p = 0; p < 2; ++p) {
    EXPECT_NOK(PutWithin(cf_for_partition_[p], "stopped"));
  }
  options.manual_wal_flush = false;
  ReopenWithPartitions(2, options);
  ExpectWritesResumeOnEveryPartition(this, options);
}

// GC32 (F1): a file-scoped WAL write error at N > 1 resumes live. Recovery
// rotates every partition, atomically flushes every live column family and
// retires every WAL at or below the failed one, in memory and in the
// MANIFEST; the failed write and every write fenced above it stay gone.
// Every case runs at N in {2, 8}, failing on partition 0 and on the last
// partition, whose WAL is numbered above every other partition's.

// Defined with the gate-failure tests below.
bool LiveSstHasKey(DBWalAppendFailureTest* test, const Options& options, int cf,
                   const std::string& key);

struct GC32Case {
  size_t partitions;
  size_t failing;
  // The partition a write fenced above the failed one goes to.
  size_t other() const { return failing == 0 ? partitions - 1 : 0; }
  std::string Name() const {
    return "partitions=" + std::to_string(partitions) +
           " failing=" + std::to_string(failing);
  }
};

std::vector<GC32Case> GC32Cases() { return {{2, 0}, {2, 1}, {8, 0}, {8, 7}}; }

// Swaps the file system the failure options run on (FailureOptions() takes
// fail_env_), keeping a FailingWalFS on top. Call with no DB open on the old
// one.
void GC32UseFileSystem(DBWalAppendFailureTest* test,
                       const std::shared_ptr<FileSystem>& base) {
  test->Close();
  // DestroyAndReopen destroys with the last options, whose env goes away.
  test->last_options_.env = test->env_;
  test->fail_fs_ = std::make_shared<FailingWalFS>(base);
  test->fail_env_ = NewCompositeEnv(test->fail_fs_);
}

void GC32Open(DBWalAppendFailureTest* test, const Options& options,
              size_t partitions) {
  test->DestroyAndReopen(options);
  test->OpenWithPartitions(partitions, options);
}

void GC32PutOnEvery(DBWalAppendFailureTest* test, const std::string& prefix,
                    const WriteOptions& write_options = WriteOptions()) {
  for (size_t p = 0; p < test->cf_for_partition_.size(); ++p) {
    ASSERT_OK(test->Put(test->cf_for_partition_[p], prefix + std::to_string(p),
                        "value", write_options));
  }
}

void GC32ExpectOnEvery(DBWalAppendFailureTest* test, const std::string& prefix,
                       const std::string& expected) {
  for (size_t p = 0; p < test->cf_for_partition_.size(); ++p) {
    EXPECT_EQ(expected,
              test->Get(test->cf_for_partition_[p], prefix + std::to_string(p)))
        << prefix << p;
  }
}

// Every write succeeds on every partition, promptly.
void GC32PutOnEveryWithin(DBWalAppendFailureTest* test,
                          const std::string& prefix) {
  for (size_t p = 0; p < test->cf_for_partition_.size(); ++p) {
    ASSERT_OK(test->PutWithin(test->cf_for_partition_[p],
                              prefix + std::to_string(p)));
  }
}

// The highest WAL index any WAL file in `dir` bears.
uint64_t GC32MaxWalIndexOnDisk(Env* env, const std::string& dir) {
  std::vector<std::string> children;
  EXPECT_OK(env->GetChildren(dir, &children));
  uint64_t max_index = 0;
  for (const std::string& child : children) {
    uint64_t number = 0;
    FileType type;
    if (!ParseFileName(child, &number, &type) || type != kWalFile) {
      continue;
    }
    std::unique_ptr<SequentialFileReader> file_reader;
    if (!SequentialFileReader::Create(env->GetFileSystem(),
                                      LogFileName(dir, number), FileOptions(),
                                      &file_reader, /*dbg=*/nullptr,
                                      /*rate_limiter=*/nullptr)
             .ok()) {
      continue;
    }
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, number);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
      max_index = std::max(max_index, reader.GetLastReadWALIndex());
    }
  }
  return max_index;
}

// The lowest WAL index borne by a WAL file in `dir` numbered above
// `above_wal`, or zero.
uint64_t GC32MinWalIndexAbove(Env* env, const std::string& dir,
                              uint64_t above_wal) {
  std::vector<std::string> children;
  EXPECT_OK(env->GetChildren(dir, &children));
  uint64_t min_index = 0;
  for (const std::string& child : children) {
    uint64_t number = 0;
    FileType type;
    if (!ParseFileName(child, &number, &type) || type != kWalFile ||
        number <= above_wal) {
      continue;
    }
    std::unique_ptr<SequentialFileReader> file_reader;
    EXPECT_OK(SequentialFileReader::Create(
        env->GetFileSystem(), LogFileName(dir, number), FileOptions(),
        &file_reader, /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, number);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch)) {
      const uint64_t index = reader.GetLastReadWALIndex();
      if (index != 0 && (min_index == 0 || index < min_index)) {
        min_index = index;
      }
    }
  }
  return min_index;
}

// Waits for auto-resume to report success, through the callback
// GC32ExpectAutoResume installed.
class GC32AutoResume {
 public:
  GC32AutoResume() {
    SyncPoint::GetInstance()->SetCallBack(
        "RecoverFromRetryableBGIOError:RecoverSuccess", [this](void*) {
          std::lock_guard<std::mutex> lock(mu_);
          recovered_ = true;
          cv_.notify_all();
        });
  }
  ~GC32AutoResume() {
    SyncPoint::GetInstance()->ClearCallBack(
        "RecoverFromRetryableBGIOError:RecoverSuccess");
  }
  bool Wait() {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, std::chrono::seconds(30),
                        [this] { return recovered_; });
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool recovered_ = false;
};

TEST_P(DBWalAppendFailureTest, FileScopedFailureResumesLive) {
  // GC32 Q2, Q3 (F1 R2, R3; D-F1a (i), D-F1b (i)): after a file-scoped append
  // failure, Resume() (and, separately, auto-resume) succeeds with the failed
  // file still faulty; every partition's current WAL is numbered above the
  // failed one and the cutoff is cleared; writes then succeed on every
  // partition, promptly; a reopen sees every acknowledged write and not the
  // failed one.
  // Neuter: restore the tracking member of the guard at N > 1; Resume()
  // returns NotSupported (auto-resume never reports success).
  for (const GC32Case& c : GC32Cases()) {
    for (const bool auto_resume : {false, true}) {
      SCOPED_TRACE(c.Name() + " auto_resume=" + std::to_string(auto_resume));
      Options options = FailureOptions();
      if (auto_resume) {
        options.max_bgerror_resume_count = 2;
        options.bgerror_resume_retry_interval = 10000;
      }
      GC32Open(this, options, c.partitions);
      GC32PutOnEvery(this, "before");
      const int cf = cf_for_partition_[c.failing];
      const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
      std::unique_ptr<GC32AutoResume> auto_resumed;
      if (auto_resume) {
        auto_resumed.reset(new GC32AutoResume());
      }
      fail_fs_->Arm(failed_wal, FileScoped("injected"));
      ASSERT_NOK(Put(cf, "failed", "value"));
      if (auto_resume) {
        ASSERT_TRUE(auto_resumed->Wait());
        auto_resumed.reset();
      } else {
        EXPECT_EQ(failed_wal, dbfull()->TEST_WalRecoveryCutoff());
        EXPECT_EQ("NOT_FOUND", Get(cf, "failed"));
        ASSERT_OK(dbfull()->Resume());
      }
      ASSERT_OK(dbfull()->TEST_GetBGError());
      EXPECT_EQ(0U, dbfull()->TEST_WalRecoveryCutoff());
      for (size_t p = 0; p < c.partitions; ++p) {
        EXPECT_GT(dbfull()->TEST_CurrentWalNumber(p), failed_wal) << p;
      }
      GC32PutOnEveryWithin(this, "after");
      fail_fs_->Disarm();
      EXPECT_EQ("NOT_FOUND", Get(cf, "failed"));
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectOnEvery(this, "before", "value");
      GC32ExpectOnEvery(this, "after", "value");
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[c.failing], "failed"));
    }
  }
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeStillRefusedWhereUnsupported) {
  // GC32 Q2 (D-F1a (i), D-F1c): a file-scoped failure still leaves the DB
  // stopped, with the base's NotSupported message, at N=1 with MANIFEST WAL
  // tracking, and at N=2 with manual_wal_flush or WAL_ttl_seconds (members of
  // the guard Open accepts above one partition). N=1 without tracking keeps
  // upstream's live resume.
  // Neuter: drop a member from the guard; its case resumes.
  const std::string refusal =
      "Not implemented: file-scoped WAL write error recovery is not supported "
      "with manual WAL flush, 2PC, two write queues, pipelined or unordered "
      "writes, WAL recycling, retention or tracking, or blob direct write";
  {
    SCOPED_TRACE("N=1 tracking");
    Options options = FailureOptions();
    options.wal_partitions = 1;
    DestroyAndReopen(options);
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), FileScoped("injected"));
    ASSERT_NOK(Put("failed", "value"));
    fail_fs_->Disarm();
    EXPECT_EQ(refusal, dbfull()->Resume().ToString());
    EXPECT_NOK(Writer([this] { return Put("stopped", "value"); }).Join());
  }
  {
    SCOPED_TRACE("N=1 without tracking");
    Options options = FailureOptions();
    options.wal_partitions = 1;
    options.track_and_verify_wals_in_manifest = false;
    DestroyAndReopen(options);
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(0), FileScoped("injected"));
    ASSERT_NOK(Put("failed", "value"));
    fail_fs_->Disarm();
    ASSERT_OK(dbfull()->Resume());
    ASSERT_OK(Writer([this] { return Put("after", "value"); }).Join());
    EXPECT_EQ("NOT_FOUND", Get("failed"));
  }
  for (const bool manual_wal_flush : {true, false}) {
    SCOPED_TRACE(manual_wal_flush ? "N=2 manual_wal_flush"
                                  : "N=2 WAL_ttl_seconds");
    Options options = FailureOptions();
    if (manual_wal_flush) {
      options.manual_wal_flush = true;
    } else {
      options.WAL_ttl_seconds = 1000;
    }
    GC32Open(this, options, 2);
    const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(1);
    fail_fs_->Arm(failed_wal, FileScoped("injected"));
    if (manual_wal_flush) {
      ASSERT_OK(Put(cf_for_partition_[1], "failed", "value"));
      ASSERT_NOK(dbfull()->FlushWAL(/*sync=*/false));
    } else {
      ASSERT_NOK(Put(cf_for_partition_[1], "failed", "value"));
    }
    fail_fs_->Disarm();
    EXPECT_EQ(refusal, dbfull()->Resume().ToString());
    EXPECT_EQ(failed_wal, dbfull()->TEST_WalRecoveryCutoff());
    for (size_t p = 0; p < 2; ++p) {
      EXPECT_NOK(PutWithin(cf_for_partition_[p], "stopped"));
    }
  }
}

TEST_P(DBWalAppendFailureTest,
       FileScopedResumeChecksEveryPartitionsReplacement) {
  // GC32 Q3 (F1 R3): the replacement check after the flush covers every
  // partition. A test hook makes the last partition's current WAL look
  // unreplaced: Resume() fails with Corruption, the cutoff and the
  // background error stay, and writes keep failing on every partition.
  // Neuter: check partition 0 only; the resume succeeds.
  for (const GC32Case& c : GC32Cases()) {
    SCOPED_TRACE(c.Name());
    Options options = FailureOptions();
    GC32Open(this, options, c.partitions);
    GC32PutOnEvery(this, "before");
    const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
    fail_fs_->Arm(failed_wal, FileScoped("injected"));
    ASSERT_NOK(Put(cf_for_partition_[c.failing], "failed", "value"));
    fail_fs_->Disarm();
    const size_t last = c.partitions - 1;
    bool checked_last = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::ResumeImpl:ReplacementWal", [&](void* arg) {
          auto* replacement = static_cast<std::pair<size_t, uint64_t>*>(arg);
          if (replacement->first == last) {
            replacement->second = failed_wal;
            checked_last = true;
          }
        });
    const Status resume = dbfull()->Resume();
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::ResumeImpl:ReplacementWal");
    EXPECT_TRUE(checked_last);
    EXPECT_TRUE(resume.IsCorruption()) << resume.ToString();
    EXPECT_NOK(dbfull()->TEST_GetBGError());
    EXPECT_EQ(failed_wal, dbfull()->TEST_WalRecoveryCutoff());
    for (size_t p = 0; p < c.partitions; ++p) {
      EXPECT_NOK(PutWithin(cf_for_partition_[p], "stopped"));
    }
  }
}

// Fails A (on c.failing) file-scoped with `landing` while it is held in its
// append under B (on c.other()), which appended and inserted above it and
// parks in its completion gate; both fail. Every partition first takes an
// acknowledged synced write "before<p>". Returns the last sequence allocated
// once both failed: the top of the fence.
SequenceNumber GC32FailUnderParkedWrite(DBWalAppendFailureTest* test,
                                        const GC32Case& c,
                                        FailingWalFS::Landing landing) {
  WriteOptions sync_write;
  sync_write.sync = true;
  GC32PutOnEvery(test, "before", sync_write);
  const int cf_a = test->cf_for_partition_[c.failing];
  const int cf_b = test->cf_for_partition_[c.other()];
  test->HoldNextAppend();
  DBWalAppendFailureTest::Writer a([&] { return test->Put(cf_a, "A", "v"); });
  EXPECT_TRUE(test->WaitUntilHeld());
  const size_t parked_before = test->dbfull()->TEST_WalDurabilityWaiters();
  DBWalAppendFailureTest::Writer b([&] { return test->Put(cf_b, "B", "v"); });
  EXPECT_TRUE(test->WaitForDurabilityWaiters(parked_before + 1));
  test->fail_fs_->Arm(test->dbfull()->TEST_CurrentWalNumber(c.failing),
                      DBWalAppendFailureTest::FileScoped("injected"), landing);
  test->ReleaseHeldAppend();
  EXPECT_NOK(a.Join());
  EXPECT_NOK(b.Join());
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  return test->dbfull()->GetVersionSet()->LastAllocatedSequence();
}

// A and B are not readable through Get or an iterator, and no live SST holds
// them; every acknowledged "before<p>" is.
void GC32ExpectFencedGone(DBWalAppendFailureTest* test, const Options& options,
                          const GC32Case& c, bool check_ssts) {
  GC32ExpectOnEvery(test, "before", "value");
  const int cf_a = test->cf_for_partition_[c.failing];
  const int cf_b = test->cf_for_partition_[c.other()];
  EXPECT_EQ("NOT_FOUND", test->Get(cf_a, "A"));
  EXPECT_EQ("NOT_FOUND", test->Get(cf_b, "B"));
  for (const int cf : {cf_a, cf_b}) {
    std::unique_ptr<Iterator> iter(
        test->db_->NewIterator(ReadOptions(), test->handles_[cf]));
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      EXPECT_NE("A", iter->key().ToString());
      EXPECT_NE("B", iter->key().ToString());
    }
    EXPECT_OK(iter->status());
  }
  if (check_ssts) {
    EXPECT_FALSE(LiveSstHasKey(test, options, cf_a, "A"));
    EXPECT_FALSE(LiveSstHasKey(test, options, cf_b, "B"));
  }
}

const std::vector<WALRecoveryMode>& GC32RecoveryModes() {
  static const std::vector<WALRecoveryMode> modes = {
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kSkipAnyCorruptedRecords};
  return modes;
}

const char* GC32LandingName(FailingWalFS::Landing landing) {
  switch (landing) {
    case FailingWalFS::Landing::kNothing:
      return "nothing";
    case FailingWalFS::Landing::kWhole:
      return "whole";
    case FailingWalFS::Landing::kTorn:
      return "torn";
  }
  return "?";
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeDropsFencedWrites) {
  // GC32 Q4 (F1 R4): A fails file-scoped on its partition; B, on another
  // partition, allocated above A, appended into a healthy WAL and inserted
  // before the failure, then failed at its gate. For every landing of A's
  // record (absent, torn, whole): after Resume() neither A nor B is readable
  // or in a live SST; the MANIFEST reserves the top of the fence (a reopen
  // straight after the resume reports a last sequence at or above every
  // sequence allocated before it); a reopen in every recovery mode shows
  // neither; a write after the resume and after the reopen takes a sequence
  // above the fence top and a WAL index above every index allocated before
  // the resume.
  // Neuter: reserve only the failed group's range; the reopen reports a last
  // sequence below B's.
  for (const GC32Case& c : {GC32Case{2, 0}, GC32Case{8, 7}}) {
    for (const FailingWalFS::Landing landing :
         {FailingWalFS::Landing::kNothing, FailingWalFS::Landing::kTorn,
          FailingWalFS::Landing::kWhole}) {
      SCOPED_TRACE(c.Name() + " landing=" + GC32LandingName(landing));
      Options options = FailureOptions();
      GC32Open(this, options, c.partitions);
      const SequenceNumber fence_top =
          GC32FailUnderParkedWrite(this, c, landing);
      const uint64_t max_index_before =
          GC32MaxWalIndexOnDisk(Env::Default(), dbname_);
      fail_fs_->Disarm();
      ASSERT_OK(dbfull()->Resume());
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      // Straight after the resume, before any write: only the MANIFEST can
      // carry the reservation.
      options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
      ReopenWithPartitions(c.partitions, options);
      EXPECT_GE(db_->GetLatestSequenceNumber(), fence_top);
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      const uint64_t last_wal = dbfull()->TEST_CurrentWalNumber(0) - 1;
      GC32PutOnEveryWithin(this, "after");
      EXPECT_GT(db_->GetLatestSequenceNumber(), fence_top + c.partitions - 1);
      const uint64_t min_index_after =
          GC32MinWalIndexAbove(Env::Default(), dbname_, last_wal);
      EXPECT_GT(min_index_after, max_index_before);
      for (const WALRecoveryMode mode : GC32RecoveryModes()) {
        SCOPED_TRACE(static_cast<int>(mode));
        options.wal_recovery_mode = mode;
        ReopenWithPartitions(c.partitions, options);
        GC32ExpectFencedGone(this, options, c, /*check_ssts=*/false);
        GC32ExpectOnEvery(this, "after", "value");
      }
    }
  }
}

TEST_P(DBWalAppendFailureTest, FailedGateSyncFileScopedResumeReservesFenceTop) {
  // GC32 Q4 (F1 R4): a sync write on one partition whose gate fsync fails
  // file-scoped on another partition's WAL (which holds an acknowledged
  // record below it) records no sequence range in the recovery context.
  // Resume() still reserves the top of the fence in the MANIFEST: a reopen
  // straight after it reports a last sequence at or above the failed sync
  // write's. The failed sync write, whose record landed whole in a healthy
  // WAL and which went into the memtable with its group, is absent after the
  // resume and in every recovery mode; the acknowledged record below it is
  // present.
  // Neuter: reserve nothing when the context carries no sequence range; the
  // reopen reports a last sequence below the failed sync write's.
  for (const GC32Case& c : GC32Cases()) {
    SCOPED_TRACE(c.Name());
    Options options = FailureOptions();
    GC32Open(this, options, c.partitions);
    WriteOptions sync_write;
    sync_write.sync = true;
    GC32PutOnEvery(this, "before", sync_write);
    const int cf_f = cf_for_partition_[c.failing];
    const int cf_s = cf_for_partition_[c.other()];
    const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
    ASSERT_OK(Put(cf_f, "below", "value"));
    fail_fs_->Arm(failed_wal, FileScoped("injected"));
    ASSERT_NOK(Put(cf_s, "synced", "value", sync_write));
    EXPECT_EQ(failed_wal, dbfull()->TEST_WalRecoveryCutoff());
    const SequenceNumber fence_top =
        dbfull()->GetVersionSet()->LastAllocatedSequence();
    ASSERT_OK(dbfull()->Resume());
    fail_fs_->Disarm();
    EXPECT_EQ("NOT_FOUND", Get(cf_s, "synced"));
    EXPECT_EQ("value", Get(cf_f, "below"));
    EXPECT_FALSE(LiveSstHasKey(this, options, cf_s, "synced"));
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    ReopenWithPartitions(c.partitions, options);
    EXPECT_GE(db_->GetLatestSequenceNumber(), fence_top);
    GC32PutOnEveryWithin(this, "after");
    for (const WALRecoveryMode mode : GC32RecoveryModes()) {
      SCOPED_TRACE(static_cast<int>(mode));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectOnEvery(this, "before", "value");
      GC32ExpectOnEvery(this, "after", "value");
      EXPECT_EQ("value", Get(cf_for_partition_[c.failing], "below"));
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[c.other()], "synced"));
    }
  }
}

// Watches the WALs numbered at or below a bound: counts every open of one
// for reading or rewriting, and every sync of one, and optionally fails them
// (a file-scoped fault that leaves the file unusable for good). Sits above
// the FailingWalFS, so it sees the calls that one fails too.
class GC32WalWatchFS : public FileSystemWrapper {
 public:
  explicit GC32WalWatchFS(const std::shared_ptr<FileSystem>& base)
      : FileSystemWrapper(base) {}
  static const char* kClassName() { return "GC32WalWatchFS"; }
  const char* Name() const override { return kClassName(); }

  void Watch(uint64_t through, bool fail) {
    std::lock_guard<std::mutex> lock(mu_);
    through_ = through;
    fail_ = fail;
    touched_.clear();
  }
  // The watched WALs touched since Watch(), one entry per call.
  std::vector<uint64_t> Touched() {
    std::lock_guard<std::mutex> lock(mu_);
    return touched_;
  }

  IOStatus NewSequentialFile(const std::string& fname, const FileOptions& opts,
                             std::unique_ptr<FSSequentialFile>* result,
                             IODebugContext* dbg) override {
    IOStatus s = Touch(fname);
    return s.ok() ? target()->NewSequentialFile(fname, opts, result, dbg) : s;
  }
  IOStatus NewRandomAccessFile(const std::string& fname,
                               const FileOptions& opts,
                               std::unique_ptr<FSRandomAccessFile>* result,
                               IODebugContext* dbg) override {
    IOStatus s = Touch(fname);
    return s.ok() ? target()->NewRandomAccessFile(fname, opts, result, dbg) : s;
  }
  IOStatus NewRandomRWFile(const std::string& fname, const FileOptions& opts,
                           std::unique_ptr<FSRandomRWFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = Touch(fname);
    return s.ok() ? target()->NewRandomRWFile(fname, opts, result, dbg) : s;
  }
  IOStatus ReopenWritableFile(const std::string& fname, const FileOptions& opts,
                              std::unique_ptr<FSWritableFile>* result,
                              IODebugContext* dbg) override {
    IOStatus s = Touch(fname);
    return s.ok() ? target()->ReopenWritableFile(fname, opts, result, dbg) : s;
  }
  IOStatus SyncFile(const std::string& fname, const FileOptions& file_opts,
                    const IOOptions& io_opts, bool use_fsync,
                    IODebugContext* dbg) override {
    IOStatus s = Touch(fname);
    return s.ok()
               ? target()->SyncFile(fname, file_opts, io_opts, use_fsync, dbg)
               : s;
  }
  IOStatus NewWritableFile(const std::string& fname, const FileOptions& opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override {
    IOStatus s = target()->NewWritableFile(fname, opts, result, dbg);
    uint64_t number = 0;
    if (s.ok() && WalNumber(fname, &number)) {
      result->reset(new WalFile(std::move(*result), this, number));
    }
    return s;
  }

 private:
  class WalFile : public FSWritableFileOwnerWrapper {
   public:
    WalFile(std::unique_ptr<FSWritableFile>&& file, GC32WalWatchFS* fs,
            uint64_t number)
        : FSWritableFileOwnerWrapper(std::move(file)),
          fs_(fs),
          number_(number) {}
    IOStatus Sync(const IOOptions& options, IODebugContext* dbg) override {
      IOStatus s = fs_->Touch(number_);
      return s.ok() ? target()->Sync(options, dbg) : s;
    }
    IOStatus Fsync(const IOOptions& options, IODebugContext* dbg) override {
      IOStatus s = fs_->Touch(number_);
      return s.ok() ? target()->Fsync(options, dbg) : s;
    }
    IOStatus RangeSync(uint64_t offset, uint64_t nbytes,
                       const IOOptions& options, IODebugContext* dbg) override {
      IOStatus s = fs_->Touch(number_);
      return s.ok() ? target()->RangeSync(offset, nbytes, options, dbg) : s;
    }

   private:
    GC32WalWatchFS* const fs_;
    const uint64_t number_;
  };

  static bool WalNumber(const std::string& fname, uint64_t* number) {
    FileType type;
    return ParseFileName(fname.substr(fname.find_last_of('/') + 1), number,
                         &type) &&
           type == kWalFile;
  }
  IOStatus Touch(const std::string& fname) {
    uint64_t number = 0;
    return WalNumber(fname, &number) ? Touch(number) : IOStatus::OK();
  }
  IOStatus Touch(uint64_t number) {
    std::lock_guard<std::mutex> lock(mu_);
    if (number == 0 || number > through_) {
      return IOStatus::OK();
    }
    touched_.push_back(number);
    if (!fail_) {
      return IOStatus::OK();
    }
    IOStatus s = IOStatus::IOError("unusable WAL");
    s.SetScope(IOStatus::IOErrorScope::kIOErrorScopeFile);
    return s;
  }

  std::mutex mu_;
  uint64_t through_ = 0;
  bool fail_ = false;
  std::vector<uint64_t> touched_;
};

// As GC32UseFileSystem, with a GC32WalWatchFS on top, which it returns.
std::shared_ptr<GC32WalWatchFS> GC32UseWatchedFileSystem(
    DBWalAppendFailureTest* test, const std::shared_ptr<FileSystem>& base) {
  test->Close();
  // DestroyAndReopen destroys with the last options, whose env goes away.
  test->last_options_.env = test->env_;
  test->fail_fs_ = std::make_shared<FailingWalFS>(base);
  auto watch_fs = std::make_shared<GC32WalWatchFS>(test->fail_fs_);
  test->fail_env_ = NewCompositeEnv(watch_fs);
  return watch_fs;
}

// As ReopenWithPartitions, returning the open's status.
Status GC32TryReopen(DBWalAppendFailureTest* test, size_t num_partitions,
                     Options options) {
  std::vector<std::string> cf_names;
  Status s = DB::ListColumnFamilies(options, test->dbname_, &cf_names);
  if (!s.ok()) {
    return s;
  }
  options.wal_partitions = static_cast<uint32_t>(num_partitions);
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  return test->TryReopenWithColumnFamilies(cf_names, options);
}

// Drops every unsynced byte, as a power loss would, and reopens nothing.
void GC32Crash(DBWalAppendFailureTest* test,
               const std::shared_ptr<FaultInjectionTestFS>& fault_fs) {
  fault_fs->SetFilesystemActive(false);
  test->Close();
  ASSERT_OK(fault_fs->DropUnsyncedFileData());
  fault_fs->SetFilesystemActive(true);
  test->fail_fs_->Disarm();
}

std::shared_ptr<FaultInjectionTestFS> GC32NewFaultFS() {
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(FileSystem::Default());
  // Before the open: files capture it when created.
  fault_fs->SetInjectUnsyncedDataLoss(true);
  return fault_fs;
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeKeepsManifestWalSetConsistent) {
  // GC32 Q5 (F1 R5): once Resume() returns OK, the MANIFEST WAL set lists no
  // WAL at or below the failed one, nor any WAL of the failed generation, on
  // any partition: every WAL it tracks is alive. A crash straight after the
  // resume, with file deletions disabled across it (the retired WAL files
  // still on disk), and a crash after new writes once they were purged, both
  // reopen in every recovery mode with no missing WAL or size mismatch, every
  // acknowledged write and not the failed one.
  // Neuter: drop DeleteWalsBefore from the flush's edit (memtable_list.cc);
  // the WAL set keeps the retired WALs.
  for (const GC32Case& c : GC32Cases()) {
    for (const bool deletions_disabled : {true, false}) {
      SCOPED_TRACE(c.Name() +
                   " deletions_disabled=" + std::to_string(deletions_disabled));
      const std::shared_ptr<FaultInjectionTestFS> fault_fs = GC32NewFaultFS();
      GC32UseFileSystem(this, fault_fs);
      Options options = FailureOptions();
      GC32Open(this, options, c.partitions);
      WriteOptions sync_write;
      sync_write.sync = true;
      GC32PutOnEvery(this, "before", sync_write);
      const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
      std::vector<uint64_t> old_wals;
      for (const std::vector<uint64_t>& alive : AliveWalNumbers()) {
        old_wals.insert(old_wals.end(), alive.begin(), alive.end());
      }
      fail_fs_->Arm(failed_wal, FileScoped("injected"));
      ASSERT_NOK(Put(cf_for_partition_[c.failing], "failed", "value"));
      if (deletions_disabled) {
        ASSERT_OK(db_->DisableFileDeletions());
      }
      ASSERT_OK(dbfull()->Resume());
      std::set<uint64_t> alive_after;
      for (const std::vector<uint64_t>& alive : AliveWalNumbers()) {
        alive_after.insert(alive.begin(), alive.end());
      }
      const auto& wals = dbfull()->GetVersionSet()->GetWalSet().GetWals();
      for (const auto& wal : wals) {
        EXPECT_GT(wal.first, failed_wal);
        EXPECT_EQ(1U, alive_after.count(wal.first)) << wal.first;
      }
      // No writer through the failed WAL is still open, on any partition.
      for (size_t p = 0; p < c.partitions; ++p) {
        for (const auto& mark : dbfull()->TEST_WalSyncMarks(p)) {
          EXPECT_GT(mark.first, failed_wal) << p;
        }
      }
      for (const uint64_t old_wal : old_wals) {
        EXPECT_EQ(0U, wals.count(old_wal)) << old_wal;
        const Status exists =
            Env::Default()->FileExists(LogFileName(dbname_, old_wal));
        if (deletions_disabled) {
          // Their files, and their place in the WAL size accounting, stay
          // until deletions resume.
          EXPECT_OK(exists) << old_wal;
        } else {
          EXPECT_EQ(0U, alive_after.count(old_wal)) << old_wal;
          EXPECT_TRUE(exists.IsNotFound()) << old_wal;
        }
      }
      if (!deletions_disabled) {
        GC32PutOnEvery(this, "after", sync_write);
      }
      GC32Crash(this, fault_fs);
      for (const WALRecoveryMode mode : GC32RecoveryModes()) {
        SCOPED_TRACE(static_cast<int>(mode));
        options.wal_recovery_mode = mode;
        ReopenWithPartitions(c.partitions, options);
        GC32ExpectOnEvery(this, "before", "value");
        GC32ExpectOnEvery(this, "after",
                          deletions_disabled ? "NOT_FOUND" : "value");
        EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[c.failing], "failed"));
      }
      Close();
    }
  }
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeKeepsWalIndexContinuity) {
  // GC32 Q6 (F1 R6): for every landing of A's record (absent, torn, whole),
  // a crash straight after Resume(), and a crash after synced writes on every
  // partition, both reopen in point-in-time and absolute consistency modes
  // with every acknowledged write, neither A nor B, and no unexplained hole:
  // the indices allocated before the resume are explained by the replacement
  // WALs' predecessor spans over the retired WALs, which no recovery reads,
  // so a record that landed whole there is not recovered and raises no
  // bears-records-and-a-void-cover error.
  // This rests on the flush's retirement, not on how the resume explains a
  // retired WAL's burned index: explaining it as the retryable path does
  // (landed whole: no void; otherwise a void in the replacement WAL) still
  // passes here, and fails FileScopedResumeNeedsNoIoOnRetiredWal instead.
  for (const GC32Case& c : {GC32Case{2, 0}, GC32Case{8, 7}}) {
    for (const FailingWalFS::Landing landing :
         {FailingWalFS::Landing::kNothing, FailingWalFS::Landing::kTorn,
          FailingWalFS::Landing::kWhole}) {
      for (const bool writes_after : {false, true}) {
        SCOPED_TRACE(c.Name() + " landing=" + GC32LandingName(landing) +
                     " writes_after=" + std::to_string(writes_after));
        const std::shared_ptr<FaultInjectionTestFS> fault_fs = GC32NewFaultFS();
        GC32UseFileSystem(this, fault_fs);
        Options options = FailureOptions();
        GC32Open(this, options, c.partitions);
        GC32FailUnderParkedWrite(this, c, landing);
        ASSERT_OK(dbfull()->Resume());
        WriteOptions sync_write;
        sync_write.sync = true;
        if (writes_after) {
          GC32PutOnEvery(this, "after", sync_write);
        }
        GC32Crash(this, fault_fs);
        for (const WALRecoveryMode mode :
             {WALRecoveryMode::kPointInTimeRecovery,
              WALRecoveryMode::kAbsoluteConsistency}) {
          SCOPED_TRACE(static_cast<int>(mode));
          options.wal_recovery_mode = mode;
          ReopenWithPartitions(c.partitions, options);
          GC32ExpectFencedGone(this, options, c, /*check_ssts=*/false);
          GC32ExpectOnEvery(this, "after",
                            writes_after ? "value" : "NOT_FOUND");
          GC32PutOnEvery(this, "post" + std::to_string(static_cast<int>(mode)),
                         sync_write);
        }
        Close();
      }
    }
  }
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeNeedsNoIoOnRetiredWal) {
  // GC32 Q6 (F1 R6): the failed WAL stays unusable for good: once the append
  // failed, every open of it for reading or rewriting, every sync of it and
  // every append to it fails. Resume() still returns OK for every landing of
  // A's record, without touching it, and a reopen in absolute consistency
  // mode shows every acknowledged write and neither A nor B.
  // Neuter: let the resume read back, zero or sync a WAL it retires (as the
  // retryable path's explanation does); Resume() fails on the unusable file
  // and the watch records the attempt.
  for (const GC32Case& c : GC32Cases()) {
    for (const FailingWalFS::Landing landing :
         {FailingWalFS::Landing::kNothing, FailingWalFS::Landing::kTorn,
          FailingWalFS::Landing::kWhole}) {
      SCOPED_TRACE(c.Name() + " landing=" + GC32LandingName(landing));
      const std::shared_ptr<GC32WalWatchFS> watch_fs =
          GC32UseWatchedFileSystem(this, FileSystem::Default());
      Options options = FailureOptions();
      GC32Open(this, options, c.partitions);
      const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
      GC32FailUnderParkedWrite(this, c, landing);
      watch_fs->Watch(failed_wal, /*fail=*/true);
      const Status resume = dbfull()->Resume();
      const std::vector<uint64_t> touched = watch_fs->Touched();
      watch_fs->Watch(0, /*fail=*/false);
      ASSERT_OK(resume);
      std::vector<uint64_t> touched_failed;
      for (const uint64_t wal : touched) {
        if (wal == failed_wal) {
          touched_failed.push_back(wal);
        }
      }
      EXPECT_TRUE(touched_failed.empty()) << touched_failed.size();
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      GC32PutOnEveryWithin(this, "after");
      fail_fs_->Disarm();
      options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/false);
      GC32ExpectOnEvery(this, "after", "value");
      Close();
    }
  }
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeRacesSyncWalAndSyncWrite) {
  // GC32 Q7 (F1 R7): at each step of the resume (after the cutoff
  // extension, before the flush, after the flush, before the retire loop,
  // inside it, after it, before ClearBGError), a SyncWAL(), a FlushWAL(true)
  // and a sync write on another partition start. No sync ever touches a WAL
  // at or below the failed one (watched at the file system, below every
  // sync path: SyncWAL, FlushWAL, the gate sync and SyncClosedWals); every
  // call returns within the Writer bound (a stopped-DB error, or OK once the
  // resume admitted it); the resume succeeds and writes go on.
  // Where the resume holds no lock (before the flush, before the retire
  // loop), the calls are joined there, so they run to completion against the
  // quarantine and fail with the background error.
  // Neuters: clear the cutoff before the flush; the flush's own sync of the
  // closed WALs reaches the quarantined writer and the resume fails. Clear
  // it before the retire loop; the SyncWAL() joined there reports success
  // while the DB is still stopped.
  // Every WAL the MANIFEST tracks with a synced size is at least that long on
  // disk: before the resume, where the calls are joined before the flush,
  // and after the resume once a rotation and SyncWAL() record the next WALs'
  // sizes. The WALs before the failure are rotated out and synced first, so
  // the MANIFEST tracks their sizes until the recovery flush makes them
  // obsolete; each check compares at least one WAL.
  // Neuter: record a closed WAL's synced size one byte beyond its synced
  // size; the check before the resume fails. Neuter: skip that rotation and
  // sync; the checks compare no WAL.
  struct Step {
    const char* sync_point;
    bool join_inside;
    // Before the recovery flush: the rotated WALs' synced sizes are tracked.
    bool before_flush;
  };
  const std::vector<Step> steps = {
      {"DBImpl::ResumeImpl:AfterSyncSeq", false, true},
      {"DBImpl::FlushAllColumnFamilies:BeforeAtomicFlushMemTables", true, true},
      {"DBImpl::ResumeImpl:AfterFlush", false, false},
      {"DBImpl::ResumeImpl:BeforeRetire", true, false},
      {"DBImpl::ResumeImpl:RetireWal", false, false},
      {"DBImpl::ResumeImpl:AfterRetire", false, false},
      {"DBImpl::ResumeImpl:BeforeClearBGError", false, false},
  };
  for (const GC32Case& c : GC32Cases()) {
    for (const Step& step : steps) {
      SCOPED_TRACE(c.Name() + " step=" + step.sync_point);
      const std::shared_ptr<GC32WalWatchFS> watch_fs =
          GC32UseWatchedFileSystem(this, FileSystem::Default());
      Options options = FailureOptions();
      GC32Open(this, options, c.partitions);
      GC32PutOnEvery(this, "before");
      ASSERT_OK(dbfull()->TEST_SwitchMemtable());
      ASSERT_OK(dbfull()->SyncWAL());
      auto check_synced_sizes = [&]() {
        size_t checked = 0;
        for (const auto& [number, meta] :
             dbfull()->GetVersionSet()->GetWalSet().GetWals()) {
          if (meta.HasSyncedSize()) {
            uint64_t size = 0;
            EXPECT_OK(env_->GetFileSize(LogFileName(dbname_, number), &size));
            EXPECT_GE(size, meta.GetSyncedSizeInBytes()) << "WAL " << number;
            ++checked;
          }
        }
        return checked;
      };
      const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
      fail_fs_->Arm(failed_wal, FileScoped("injected"));
      ASSERT_NOK(Put(cf_for_partition_[c.failing], "failed", "value"));
      watch_fs->Watch(failed_wal, /*fail=*/false);
      EXPECT_GT(check_synced_sizes(), 0U);
      std::vector<uint64_t> synced;
      std::mutex synced_mu;
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::SyncWalImpl:BeforeSyncWAL", [&](void* arg) {
            std::lock_guard<std::mutex> lock(synced_mu);
            synced.push_back(static_cast<log::Writer*>(arg)->get_log_number());
          });
      std::vector<std::unique_ptr<Writer>> calls;
      std::atomic<bool> fired{false};
      const int cf_other = cf_for_partition_[c.other()];
      SyncPoint::GetInstance()->SetCallBack(step.sync_point, [&](void*) {
        if (fired.exchange(true)) {
          return;
        }
        calls.emplace_back(new Writer([this] { return dbfull()->SyncWAL(); }));
        calls.emplace_back(
            new Writer([this] { return dbfull()->FlushWAL(/*sync=*/true); }));
        calls.emplace_back(new Writer([this, cf_other] {
          WriteOptions sync_write;
          sync_write.sync = true;
          return Put(cf_other, "racing", "value", sync_write);
        }));
        if (step.join_inside) {
          for (auto& call : calls) {
            EXPECT_NOK(call->Join());
          }
          calls.clear();
          if (step.before_flush) {
            EXPECT_GT(check_synced_sizes(), 0U);
          }
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
      });
      const Status resume = dbfull()->Resume();
      SyncPoint::GetInstance()->ClearCallBack(step.sync_point);
      for (auto& call : calls) {
        call->Join().PermitUncheckedError();
      }
      calls.clear();
      SyncPoint::GetInstance()->ClearCallBack(
          "DBImpl::SyncWalImpl:BeforeSyncWAL");
      ASSERT_OK(resume);
      EXPECT_TRUE(fired.load());
      const std::vector<uint64_t> touched = watch_fs->Touched();
      watch_fs->Watch(0, /*fail=*/false);
      EXPECT_TRUE(touched.empty())
          << touched.size() << " first " << (touched.empty() ? 0 : touched[0]);
      for (const uint64_t wal : synced) {
        EXPECT_GT(wal, failed_wal);
      }
      fail_fs_->Disarm();
      GC32PutOnEveryWithin(this, "after");
      // SyncWAL() records synced sizes of closed WALs only.
      ASSERT_OK(dbfull()->TEST_SwitchMemtable());
      ASSERT_OK(dbfull()->SyncWAL());
      EXPECT_GT(check_synced_sizes(), 0U);
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectOnEvery(this, "before", "value");
      GC32ExpectOnEvery(this, "after", "value");
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[c.failing], "failed"));
      Close();
    }
  }
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeCrashAtEachStepFailsClosed) {
  // GC32 Q8 (F1 R8): a crash (every unsynced byte dropped, the file system
  // frozen at that instant) between each pair of the resume's steps:
  // quarantine; rotation; the flush's SST before its MANIFEST commit (which
  // also retires the WALs in the MANIFEST); the in-memory retirement; the
  // explanation and the reservation; the release. Every crash reopens in
  // point-in-time mode, and in absolute consistency mode where the hole the
  // failed append left is explained, with every acknowledged (synced) write
  // and neither A (absent) nor B.
  // Neuter: release the failed ranges before the flush; the flush keeps B,
  // and a crash after its commit recovers it.
  const std::vector<const char*> steps = {
      "DBImpl::ResumeImpl:AfterSyncSeq",
      "DBImpl::FlushAllColumnFamilies:BeforeAtomicFlushMemTables",
      "FlushJob::WriteLevel0Table:s",
      "DBImpl::ResumeImpl:AfterFlush",
      "DBImpl::ResumeImpl:AfterRetire",
      "DBImpl::ResumeImpl:BeforeClearBGError",
      "ErrorHandler::ClearBGError:BeforeClearStopped",
  };
  for (const GC32Case& c : {GC32Case{2, 0}, GC32Case{2, 1}, GC32Case{8, 7}}) {
    for (const char* step : steps) {
      SCOPED_TRACE(c.Name() + " step=" + step);
      const std::shared_ptr<FaultInjectionTestFS> fault_fs = GC32NewFaultFS();
      GC32UseFileSystem(this, fault_fs);
      Options options = FailureOptions();
      GC32Open(this, options, c.partitions);
      GC32FailUnderParkedWrite(this, c, FailingWalFS::Landing::kNothing);
      std::atomic<bool> crashed{false};
      SyncPoint::GetInstance()->SetCallBack(step, [&](void*) {
        if (!crashed.exchange(true)) {
          fault_fs->SetFilesystemActive(false);
        }
      });
      dbfull()->Resume().PermitUncheckedError();
      SyncPoint::GetInstance()->ClearCallBack(step);
      EXPECT_TRUE(crashed.load());
      GC32Crash(this, fault_fs);
      options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/false);
      Close();
      options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
      const Status absolute = GC32TryReopen(this, c.partitions, options);
      if (absolute.ok()) {
        GC32ExpectFencedGone(this, options, c, /*check_ssts=*/false);
      } else {
        EXPECT_TRUE(absolute.IsCorruption()) << absolute.ToString();
      }
      Close();
    }
  }
}

TEST_P(DBWalAppendFailureTest, FailedFileScopedResumeKeepsFailingLaterWrites) {
  // GC32 Q9 (F1 R9): a file-scoped resume that fails at its reservation's
  // MANIFEST write, at its flush, at its retirement record (the flush's
  // atomic MANIFEST commit, which advances every log number and deletes the
  // retired WALs from the WAL set; reported failed after it reached the
  // MANIFEST), or at its replacement check leaves the hard background error,
  // the quarantine cutoff, the open fences and the failed ranges in place:
  // writes keep failing on every partition, promptly, and A and B stay
  // invisible. The next resume succeeds, writes go on, and a reopen shows every
  // acknowledged write and neither A nor B. A reopen instead of the next
  // resume, after the failed retirement record (which reached the MANIFEST),
  // opens in every recovery mode with every acknowledged write and neither A
  // nor B. Neuter: release the failed ranges before the flush; the failed
  // resume leaves no open fence.
  enum class Failure {
    kReservation,
    kFlush,
    kRetirementRecord,
    kRetirementRecordThenReopen,
    kReplacement
  };
  for (const GC32Case& c : {GC32Case{2, 1}, GC32Case{8, 0}}) {
    for (const Failure failure :
         {Failure::kReservation, Failure::kFlush, Failure::kRetirementRecord,
          Failure::kRetirementRecordThenReopen, Failure::kReplacement}) {
      SCOPED_TRACE(c.Name() +
                   " failure=" + std::to_string(static_cast<int>(failure)));
      Options options = FailureOptions();
      GC32Open(this, options, c.partitions);
      const uint64_t failed_wal = dbfull()->TEST_CurrentWalNumber(c.failing);
      GC32FailUnderParkedWrite(this, c, FailingWalFS::Landing::kNothing);
      std::atomic<bool> injected{false};
      std::atomic<bool> reservation_armed{false};
      std::atomic<bool> commit_armed{false};
      const char* sync_point = nullptr;
      switch (failure) {
        case Failure::kFlush:
          sync_point = "FlushJob::WriteLevel0Table:s";
          SyncPoint::GetInstance()->SetCallBack(sync_point, [&](void* arg) {
            if (!injected.exchange(true)) {
              *static_cast<Status*>(arg) = Retryable("injected flush error");
            }
          });
          break;
        case Failure::kRetirementRecord:
        case Failure::kRetirementRecordThenReopen:
          // Only an atomic group writes more than one edit in this resume:
          // the flush's commit.
          sync_point = "VersionSet::ProcessManifestWrites:AfterSyncManifest";
          SyncPoint::GetInstance()->SetCallBack(
              "VersionSet::ProcessManifestWrites:BeforeWriteLastVersionEdit:0",
              [&](void*) { commit_armed.store(true); });
          SyncPoint::GetInstance()->SetCallBack(sync_point, [&](void* arg) {
            if (commit_armed.exchange(false) && !injected.exchange(true)) {
              *static_cast<IOStatus*>(arg) = Retryable("injected");
            }
          });
          break;
        case Failure::kReplacement:
          sync_point = "DBImpl::ResumeImpl:ReplacementWal";
          SyncPoint::GetInstance()->SetCallBack(sync_point, [&](void* arg) {
            auto* replacement = static_cast<std::pair<size_t, uint64_t>*>(arg);
            if (replacement->first == 0 && !injected.exchange(true)) {
              replacement->second = failed_wal;
            }
          });
          break;
        case Failure::kReservation:
          // The first MANIFEST write after the quarantine is the
          // reservation's.
          sync_point = "VersionSet::ProcessManifestWrites:AfterSyncManifest";
          SyncPoint::GetInstance()->SetCallBack(
              "DBImpl::ResumeImpl:AfterSyncSeq",
              [&](void*) { reservation_armed.store(true); });
          SyncPoint::GetInstance()->SetCallBack(sync_point, [&](void* arg) {
            if (reservation_armed.exchange(false) && !injected.exchange(true)) {
              *static_cast<IOStatus*>(arg) = Retryable("injected");
            }
          });
          break;
      }
      const Status resume = dbfull()->Resume();
      SyncPoint::GetInstance()->ClearCallBack(sync_point);
      SyncPoint::GetInstance()->ClearCallBack(
          "DBImpl::ResumeImpl:AfterSyncSeq");
      SyncPoint::GetInstance()->ClearCallBack(
          "VersionSet::ProcessManifestWrites:BeforeWriteLastVersionEdit:0");
      EXPECT_TRUE(injected.load());
      ASSERT_NOK(resume);
      const Status bg = dbfull()->TEST_GetBGError();
      ASSERT_NOK(bg);
      EXPECT_EQ(Status::Severity::kHardError, bg.severity()) << bg.ToString();
      EXPECT_EQ(failed_wal, dbfull()->TEST_WalRecoveryCutoff());
      bool open_fence = false;
      for (const auto& fence : dbfull()->TEST_WalGateFences()) {
        open_fence |= fence.second == kMaxSequenceNumber;
      }
      EXPECT_TRUE(open_fence);
      for (size_t p = 0; p < c.partitions; ++p) {
        EXPECT_EQ(bg.ToString(),
                  PutWithin(cf_for_partition_[p], "stopped").ToString());
      }
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      fail_fs_->Disarm();
      if (failure == Failure::kRetirementRecordThenReopen) {
        for (const WALRecoveryMode mode : GC32RecoveryModes()) {
          SCOPED_TRACE(static_cast<int>(mode));
          options.wal_recovery_mode = mode;
          ReopenWithPartitions(c.partitions, options);
          GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
        }
        continue;
      }
      ASSERT_OK(dbfull()->Resume());
      EXPECT_EQ(0U, dbfull()->TEST_WalRecoveryCutoff());
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      GC32PutOnEveryWithin(this, "after");
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      GC32ExpectOnEvery(this, "after", "value");
    }
  }
}

TEST_P(DBWalAppendFailureTest, FailedSyncLedGroupReleasesEveryRegisteredRange) {
  // GC32 Q10 (GC31 Q13(a)): a sync-led group with async writers registers one
  // durability range but two applied ranges (the async writers' leading
  // range, then the sync writers'). W0 (async) is held in its append on
  // partition 1 while L (sync), A1 and A2 (async) queue behind it, so L leads
  // the group [L, A1, A2]. Its failure is released as the ranges were
  // registered, for a retryable and for a file-scoped error:
  // - the group's append fails: both applied ranges failed and both release;
  //   L, A1 and A2 fail and stay absent;
  // - its gate sync fails: only the sync range failed and releases; A1 and
  //   A2, published as the group exited, stay readable.
  // After Resume() no tracker holds a range of the group: the applied tracker
  // is idle and no writer is parked, the durability tracker is idle once
  // quiet, every later write on every partition publishes promptly, and the
  // published sequence covers every allocated one.
  // At the base the append case fails with a retryable error: Resume()
  // completes the whole group's range in the applied tracker, which matches
  // neither registered range (a debug assertion; in release the two ranges
  // stay and the next write never returns).
  // Neuter: release only the recorded range; the next write never returns.
  // Neuter: release only the sync writers' applied range of a failed append;
  // the applied tracker still holds the async writers' range after Resume().
  enum class Shape { kAppend, kGateSync };
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    for (const Shape shape : {Shape::kAppend, Shape::kGateSync}) {
      for (const bool file_scoped : {false, true}) {
        SCOPED_TRACE("partitions=" + std::to_string(partitions) + " shape=" +
                     (shape == Shape::kAppend ? "append" : "gate sync") +
                     " file_scoped=" + std::to_string(file_scoped));
        const IOStatus error =
            file_scoped ? FileScoped("injected") : Retryable("injected");
        Options options = FailureOptions();
        GC32Open(this, options, partitions);
        const int cf = cf_for_partition_[1];
        WriteOptions sync_write;
        sync_write.sync = true;
        ASSERT_OK(Put(cf, "below", "v", sync_write));

        std::mutex mu;
        std::condition_variable cv;
        bool hold_append = true;
        bool w0_in_append = false;
        bool release_w0 = false;
        size_t waiting = 0;
        SyncPoint::GetInstance()->SetCallBack(
            "DBImpl::WriteToWAL:log_entry", [&](void*) {
              std::unique_lock<std::mutex> lock(mu);
              if (hold_append) {
                hold_append = false;
                w0_in_append = true;
                cv.notify_all();
                WaitForReleaseOrAbandon(lock, cv, [&] { return release_w0; });
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
        auto wait_for = [&](const std::function<bool()>& pred) {
          std::unique_lock<std::mutex> lock(mu);
          return cv.wait_for(lock, std::chrono::seconds(30), pred);
        };
        Writer w0([&] { return Put(cf, "w0", "v"); });
        ASSERT_TRUE(wait_for([&] { return w0_in_append; }));
        Writer l([&] { return Put(cf, "l", "v", sync_write); });
        ASSERT_TRUE(wait_for([&] { return waiting == 1; }));
        Writer a1([&] { return Put(cf, "a1", "v"); });
        ASSERT_TRUE(wait_for([&] { return waiting == 2; }));
        Writer a2([&] { return Put(cf, "a2", "v"); });
        ASSERT_TRUE(wait_for([&] { return waiting == 3; }));
        // W0's append is the next to add a record, its group's the one after.
        std::atomic<int> appends{0};
        std::atomic<bool> gate_sync_failed{false};
        if (shape == Shape::kAppend) {
          SyncPoint::GetInstance()->SetCallBack(
              "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
                if (appends.fetch_add(1) == 1) {
                  *static_cast<IOStatus*>(arg) = error;
                }
              });
        } else {
          SyncPoint::GetInstance()->SetCallBack(
              "DBImpl::SyncWalsFromCompletionGate:AfterGateSync",
              [&](void* arg) {
                if (!gate_sync_failed.exchange(true)) {
                  *static_cast<IOStatus*>(arg) = error;
                }
              });
        }
        {
          std::lock_guard<std::mutex> lock(mu);
          release_w0 = true;
          cv.notify_all();
        }
        const Status w0_status = w0.Join();
        const Status l_status = l.Join();
        const Status a1_status = a1.Join();
        const Status a2_status = a2.Join();
        SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
        SyncPoint::GetInstance()->ClearCallBack(
            "WriteThread::JoinBatchGroup:BeganWaiting");
        SyncPoint::GetInstance()->ClearCallBack(
            "DBImpl::WriteToWAL:AfterAddRecord");
        SyncPoint::GetInstance()->ClearCallBack(
            "DBImpl::SyncWalsFromCompletionGate:AfterGateSync");
        ASSERT_OK(w0_status);
        ASSERT_NOK(l_status);
        if (shape == Shape::kAppend) {
          EXPECT_EQ(2, appends.load());
          EXPECT_NOK(a1_status);
          EXPECT_NOK(a2_status);
        } else {
          EXPECT_TRUE(gate_sync_failed.load());
          EXPECT_OK(a1_status);
          EXPECT_OK(a2_status);
        }
        ASSERT_NOK(dbfull()->TEST_GetBGError());

        ASSERT_OK(dbfull()->Resume());
        for (size_t p = 0; p < partitions; ++p) {
          EXPECT_EQ(WalDurabilityTracker::kIdle, AppliedPendingLow(p)) << p;
        }
        EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
        GC32PutOnEveryWithin(this, "after");
        for (size_t p = 0; p < partitions; ++p) {
          EXPECT_EQ(WalDurabilityTracker::kIdle,
                    dbfull()->TEST_WalDurabilityPendingLow(p))
              << p;
        }
        EXPECT_EQ(dbfull()->GetVersionSet()->LastAllocatedSequence(),
                  db_->GetLatestSequenceNumber());
        EXPECT_EQ("v", Get(cf, "below"));
        EXPECT_EQ("v", Get(cf, "w0"));
        EXPECT_EQ("NOT_FOUND", Get(cf, "l"));
        const std::string async_expected =
            shape == Shape::kAppend ? "NOT_FOUND" : "v";
        EXPECT_EQ(async_expected, Get(cf, "a1"));
        EXPECT_EQ(async_expected, Get(cf, "a2"));
        ReopenWithPartitions(partitions, options);
        GC32ExpectOnEvery(this, "after", "value");
        EXPECT_EQ("v", Get(cf_for_partition_[1], "w0"));
        EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "l"));
        EXPECT_EQ(async_expected, Get(cf_for_partition_[1], "a1"));
        EXPECT_EQ(async_expected, Get(cf_for_partition_[1], "a2"));
      }
    }
  }
}

namespace {
// The number of immutable memtables of column family `cf` (a handle index).
uint64_t GC33Immutable(DBTestBase* test, ColumnFamilyHandle* cf) {
  uint64_t immutable = 0;
  EXPECT_TRUE(test->dbfull()->GetIntProperty(
      cf, DB::Properties::kNumImmutableMemTable, &immutable));
  return immutable;
}
}  // namespace

TEST_P(DBWritePartitionTest, WbmFlushQuiesceSkippedWhenNothingToSwitch) {
  // GC33 Q3: above one partition a write-buffer-manager flush stops every
  // partition only when it will switch a memtable or request a flush. The
  // manager's memory is reserved past its limit, so ShouldFlush() stays
  // true for every write group, and the blocked pools hold every flush. The
  // first writes switch each partition's column family once; from then on
  // each is flushing and nothing is selectable, so 200 writes from 8
  // threads switch nothing and quiesce no partition. Once the pools run and
  // the flushes finish, the next write's flush acts, once. Fails at the
  // base, where every write group that reaches the manager's branch
  // quiesces.
  // Neuter: drop the pre-check; every such group quiesces.
  // Neuter: let the pre-check never find a switch; the refill switches
  // nothing.
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 8;
  constexpr int kThreads = 8;
  constexpr int kWritesPerThread = 25;
  constexpr size_t kBufferSize = 1 << 20;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 20;
  options.avoid_flush_during_shutdown = true;
  options.write_buffer_manager =
      std::make_shared<WriteBufferManager>(kBufferSize);
  OpenWithPartitions(kPartitions, options);
  GC31BlockedPools pools(env_);
  options.write_buffer_manager->ReserveMem(4 * kBufferSize);
  ASSERT_TRUE(options.write_buffer_manager->ShouldFlush());
  GC31Counter quiesces("DBImpl::WaitForPendingWrites:BeforeBlock");
  auto switched = [&]() {
    uint64_t total = 0;
    for (const int cf : cf_for_partition_) {
      total += std::min<uint64_t>(GC33Immutable(this, handles_[cf]), 1);
    }
    return total;
  };
  for (int i = 0; switched() < kPartitions; ++i) {
    ASSERT_LT(i, 1000);
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "fill" + std::to_string(i), "v"));
  }
  std::vector<uint64_t> immutable_before;
  for (ColumnFamilyHandle* handle : handles_) {
    immutable_before.push_back(GC33Immutable(this, handle));
  }
  quiesces.Reset();
  GC31Counter switches("DBImpl::SwitchMemtable:AfterCreateWAL");
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kWritesPerThread; ++i) {
        ASSERT_OK(Put(cf_for_partition_[t % kPartitions],
                      "t" + std::to_string(t) + "_" + std::to_string(i), "v"));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(0, quiesces.Get());
  EXPECT_EQ(0, switches.Get());
  for (size_t i = 0; i < handles_.size(); ++i) {
    EXPECT_EQ(immutable_before[i], GC33Immutable(this, handles_[i])) << i;
  }
  EXPECT_EQ("v", Get(cf_for_partition_[kThreads - 1], "t7_24"));

  // Once the pools run, the held flushes finish; the reservation keeps the
  // manager past its limit, so the next write's flush acts, once.
  pools.Release();
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  quiesces.Reset();
  for (int i = 0; switches.Get() == 0; ++i) {
    ASSERT_LT(i, 1000);
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "refill" + std::to_string(i), "v"));
  }
  EXPECT_EQ(1, quiesces.Get());
  EXPECT_EQ(1, switches.Get());
  options.write_buffer_manager->ScheduleFreeMem(4 * kBufferSize);
  options.write_buffer_manager->FreeMem(4 * kBufferSize);
}

TEST_P(DBWritePartitionTest, WbmFlushDecidedAgainAfterGateCloses) {
  // GC33 Q3, the decision after closing: the manager's memory is reserved
  // past its limit, every partition's column family but partition 1's is
  // flushing (the blocked pools hold the flushes), and partition 1's holds
  // one record, so the next manager flush switches it. D and A both find
  // that the flush acts. D closes the switch gate first and waits for C,
  // held past the gate; A then waits for the gate. Once C is
  // let go D switches partition 1's column family and opens the gate. A
  // closes it, finds nothing selectable any more, and opens it again
  // without draining any partition: one quiesce in all, D's.
  // Neuter: decide only before closing; A drains too (two quiesces).
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  constexpr size_t kPartitions = 4;
  constexpr size_t kBufferSize = 1 << 20;
  Options options = CurrentOptions();
  options.write_buffer_size = 64 << 20;
  options.avoid_flush_during_shutdown = true;
  options.write_buffer_manager =
      std::make_shared<WriteBufferManager>(kBufferSize);
  OpenWithPartitions(kPartitions, options);
  GC31BlockedPools pools(env_);
  options.write_buffer_manager->ReserveMem(4 * kBufferSize);
  ColumnFamilyHandle* const target = handles_[cf_for_partition_[1]];
  auto others_switched = [&]() {
    for (size_t p = 0; p < kPartitions; ++p) {
      if (p != 1 && GC33Immutable(this, handles_[cf_for_partition_[p]]) == 0) {
        return false;
      }
    }
    return true;
  };
  for (int i = 0; !others_switched(); ++i) {
    ASSERT_LT(i, 1000);
    const size_t p = std::vector<size_t>{0, 2, 3}[i % 3];
    ASSERT_OK(Put(cf_for_partition_[p], "fill" + std::to_string(i), "v"));
  }
  ASSERT_EQ(0, GC33Immutable(this, target));

  GC30Write c;
  GC30Write d;
  GC30Write a;
  // C is held past the gate, before it takes a sequence, so the next write
  // does not wait for it.
  std::atomic<std::thread::id> c_id;
  GC28Hold c_held;
  c_held.Arm("DBImpl::WriteImpl:BeforeLeaderEnters", GC28OnThread(&c_id));
  c.Start([&] {
    c_id.store(std::this_thread::get_id());
    return Put(cf_for_partition_[2], "c", "v");
  });
  ASSERT_TRUE(c_held.WaitHeld());
  // Nothing is selectable when this write checks, and it leaves partition
  // 1's column family flushable.
  ASSERT_OK(Put(cf_for_partition_[1], "y", "v"));
  ASSERT_EQ(0, GC33Immutable(this, target));
  GC31Counter quiesces("DBImpl::WaitForPendingWrites:BeforeBlock");
  GC28Hold d_waits;
  d_waits.Arm("DBImpl::CloseMemTableSwitchGate:WaitForLeaders", nullptr,
              /*hold=*/false);
  d.Start([&] { return Put(cf_for_partition_[0], "d", "v"); });
  ASSERT_TRUE(d_waits.WaitHits(1));
  std::atomic<std::thread::id> a_id;
  GC28Hold a_decided;
  a_decided.Arm("DBImpl::QuiescedWriteBufferManagerFlush:WouldAct",
                GC28OnThread(&a_id), /*hold=*/false);
  a.Start([&] {
    a_id.store(std::this_thread::get_id());
    return Put(cf_for_partition_[3], "a", "v");
  });
  ASSERT_TRUE(a_decided.WaitHits(1));
  // A now waits for the gate D closed.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(a.Done());
  c_held.Release();
  ASSERT_OK(c.Join());
  ASSERT_OK(d.Join());
  ASSERT_OK(a.Join());
  EXPECT_EQ(1, GC33Immutable(this, target));
  EXPECT_EQ(1, quiesces.Get());
  EXPECT_EQ("v", Get(cf_for_partition_[3], "a"));
  options.write_buffer_manager->ScheduleFreeMem(4 * kBufferSize);
  options.write_buffer_manager->FreeMem(4 * kBufferSize);
}

TEST_P(DBWalAppendFailureTest, FileScopedResumeStragglerAboveReservation) {
  // A straggler S on c.other() passed its stopped-DB check before A failed
  // file-scoped on c.failing, and is held as its group's leader before it
  // allocates. The resume persists its sequence reservation, then, before
  // its flush, releases S: S allocates above the reservation and before the
  // failed ranges are released. S fails at its fence, and stays invisible
  // after the resume, after the next writes and after a reopen in every
  // recovery mode; every later write takes a sequence above S's, in the same
  // instance and after a reopen before any further write, and the published
  // sequence covers every allocated one.
  // Neuter: let the recovery flush keep entries above the open fence; S is
  // readable after the resume and its live SST holds it. Neuter: drop the
  // reservation the release persists; a reopen recovers the sequence below
  // S's, and the next write reuses S's sequence.
  for (const GC32Case& c : GC32Cases()) {
    SCOPED_TRACE(c.Name());
    Options options = FailureOptions();
    GC32Open(this, options, c.partitions);
    WriteOptions sync_write;
    sync_write.sync = true;
    GC32PutOnEvery(this, "before", sync_write);
    const int cf_a = cf_for_partition_[c.failing];
    const int cf_s = cf_for_partition_[c.other()];

    std::mutex mu;
    std::condition_variable cv;
    bool armed = true;
    bool held = false;
    bool release = false;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteImpl:BeforeLeaderEnters", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (!armed) {
            return;
          }
          armed = false;
          held = true;
          cv.notify_all();
          WaitForReleaseOrAbandon(lock, cv, [&] { return release; });
        });
    Writer straggler([&] { return Put(cf_s, "S", "v"); });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return held; }));
    }
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(c.failing),
                  FileScoped("injected"));
    EXPECT_NOK(Put(cf_a, "A", "v"));
    fail_fs_->Disarm();
    EXPECT_NOK(dbfull()->TEST_GetBGError());

    SequenceNumber reserved = 0;
    SequenceNumber straggler_top = 0;
    Status straggler_status;
    std::atomic<bool> fired{false};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::FlushAllColumnFamilies:BeforeAtomicFlushMemTables",
        [&](void*) {
          if (fired.exchange(true)) {
            return;
          }
          reserved = dbfull()->GetVersionSet()->LastAllocatedSequence();
          {
            std::lock_guard<std::mutex> lock(mu);
            release = true;
            cv.notify_all();
          }
          straggler_status = straggler.Join();
          straggler_top = dbfull()->GetVersionSet()->LastAllocatedSequence();
        });
    const Status resume = dbfull()->Resume();
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::FlushAllColumnFamilies:BeforeAtomicFlushMemTables");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WriteImpl:BeforeLeaderEnters");
    if (!fired.load()) {
      // S is still held: release and join it before failing.
      {
        std::lock_guard<std::mutex> lock(mu);
        release = true;
        cv.notify_all();
      }
      straggler.Join().PermitUncheckedError();
    }
    ASSERT_TRUE(fired.load());
    EXPECT_NOK(straggler_status);
    EXPECT_GT(straggler_top, reserved);
    ASSERT_OK(resume);
    EXPECT_EQ(dbfull()->GetVersionSet()->LastAllocatedSequence(),
              db_->GetLatestSequenceNumber());
    EXPECT_EQ("NOT_FOUND", Get(cf_s, "S"));
    GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
    EXPECT_FALSE(LiveSstHasKey(this, options, cf_s, "S"));
    // S's sequences are not issued again after a reopen before any further
    // write, in any recovery mode: the release persisted them.
    const WALRecoveryMode resume_mode = options.wal_recovery_mode;
    for (const WALRecoveryMode mode : GC32RecoveryModes()) {
      SCOPED_TRACE(static_cast<int>(mode));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(c.partitions, options);
      EXPECT_GE(db_->GetLatestSequenceNumber(), straggler_top);
    }
    options.wal_recovery_mode = resume_mode;
    ReopenWithPartitions(c.partitions, options);
    ASSERT_OK(Put(cf_for_partition_[c.other()], "first", "v"));
    EXPECT_GT(db_->GetLatestSequenceNumber(), straggler_top);
    GC32PutOnEveryWithin(this, "after");
    EXPECT_GT(db_->GetLatestSequenceNumber(), straggler_top);
    EXPECT_EQ("NOT_FOUND", Get(cf_s, "S"));
    for (const WALRecoveryMode mode : GC32RecoveryModes()) {
      SCOPED_TRACE(static_cast<int>(mode));
      options.wal_recovery_mode = mode;
      ReopenWithPartitions(c.partitions, options);
      GC32ExpectFencedGone(this, options, c, /*check_ssts=*/true);
      GC32ExpectOnEvery(this, "after", "value");
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[c.other()], "S"));
      EXPECT_GT(db_->GetLatestSequenceNumber(), straggler_top);
    }
    Close();
  }
}

TEST_P(DBWalAppendFailureTest,
       FileScopedResumeHoldsAllocationUntilFencesClose) {
  // From its last sequence reservation to the release that closes the gate
  // fences, a file-scoped recovery holds allocation back. An allocation
  // arriving then (here right before ClearBGError) waits, and lands above
  // every fence the release closed, instead of inside one and above every
  // persisted reservation, where a reopen would issue its sequence again.
  // Neuter: drop the hold (HoldSequencesForWalRecoveryRelease leaves
  // sequence_reservation_held_ clear); the allocation does not wait, and the
  // release closes a fence at its sequence.
  for (const GC32Case& c : GC32Cases()) {
    SCOPED_TRACE(c.Name());
    Options options = FailureOptions();
    GC32Open(this, options, c.partitions);
    WriteOptions sync_write;
    sync_write.sync = true;
    GC32PutOnEvery(this, "before", sync_write);
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(c.failing),
                  FileScoped("injected"));
    ASSERT_NOK(Put(cf_for_partition_[c.failing], "A", "v"));
    fail_fs_->Disarm();
    ASSERT_NOK(dbfull()->TEST_GetBGError());

    std::mutex mu;
    std::condition_variable cv;
    bool allocation_waiting = false;
    bool allocation_done = false;
    SequenceNumber held_at = 0;
    SequenceNumber before = 0;
    IOStatus allocated;
    std::unique_ptr<port::Thread> allocator;
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::AllocateSequenceAndWALIndex:WaitForReservationRelease",
        [&](void*) {
          std::lock_guard<std::mutex> lock(mu);
          allocation_waiting = true;
          cv.notify_all();
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::ResumeImpl:BeforeClearBGError", [&](void*) {
          held_at = dbfull()->GetVersionSet()->LastAllocatedSequence();
          allocator.reset(new port::Thread([&] {
            uint64_t wal_index = 0;
            allocated = dbfull()->TEST_AllocateSequenceAndWALIndex(
                1, /*write_wal=*/false, &before, &wal_index);
            std::lock_guard<std::mutex> lock(mu);
            allocation_done = true;
            cv.notify_all();
          }));
          std::unique_lock<std::mutex> lock(mu);
          cv.wait_for(lock, std::chrono::seconds(30),
                      [&] { return allocation_waiting || allocation_done; });
        });
    const Status resume = dbfull()->Resume();
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::ResumeImpl:BeforeClearBGError");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::AllocateSequenceAndWALIndex:WaitForReservationRelease");
    ASSERT_NE(nullptr, allocator);
    allocator->join();
    ASSERT_OK(resume);
    ASSERT_OK(allocated);
    EXPECT_TRUE(allocation_waiting);
    EXPECT_EQ(held_at, before);
    const auto fences = dbfull()->TEST_WalGateFences();
    ASSERT_FALSE(fences.empty());
    for (const auto& fence : fences) {
      EXPECT_LE(fence.second, before);
    }
    GC32PutOnEveryWithin(this, "after");
    Close();
  }
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

// GC34 Q2 (D-GC34a (i)): above one partition a memtable-insert failure, and
// a gate failure once the range is inserted, each install a fence that pins
// every later write on every partition until recovery. A listener that
// suppresses every error is told of them but cannot suppress them: the
// background error is recorded, and later writes fail with it before their
// append. At N=1 the listener still suppresses the insert failure and the DB
// goes on; a gate failure with nothing inserted (the hole leg) fails only its
// own write.
// Neuter: drop the fence sites' flag in ErrorHandler; GetBGError() stays OK
// and later writes fail with the stale insert or gate status.
TEST_P(DBWalAppendFailureTest, GC34SuppressedFenceFailureStopsDb) {
  auto listener = std::make_shared<SuppressAllBgErrorsListener>();
  Options options = FailureOptions();
  options.listeners.push_back(listener);
  // A batch naming a column family that does not exist, routed to partition
  // 0 of `partitions`.
  auto missing_cf_write = [&](size_t partitions) {
    uint32_t missing_cf = static_cast<uint32_t>(handles_.size()) + 100;
    while (true) {
      WriteBatch batch;
      EXPECT_OK(WriteBatchInternal::Put(&batch, missing_cf, "key", "value"));
      size_t partition = partitions;
      EXPECT_OK(
          DBImpl::TEST_WalPartitionForWrite(batch, partitions, &partition));
      if (partition == 0) {
        return db_->Write(WriteOptions(), &batch);
      }
      ++missing_cf;
    }
  };
  WriteOptions sync_wo;
  sync_wo.sync = true;
  std::atomic<bool> fail_record{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalsFromCompletionGate:AfterManifestRecord", [&](void* arg) {
        if (fail_record.exchange(false)) {
          *static_cast<Status*>(arg) =
              Status::Aborted("injected MANIFEST record failure");
        }
      });
  Defer clear_callback([] {
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::SyncWalsFromCompletionGate:AfterManifestRecord");
  });
  // Reopens at N=2 (the same column families, so the same routing), failing
  // the check instead of the test.
  auto reopen = [&] {
    Options reopen_options = options;
    reopen_options.wal_partitions = 2;
    std::vector<std::string> cf_names;
    EXPECT_OK(DB::ListColumnFamilies(reopen_options, dbname_, &cf_names));
    const Status s = TryReopenWithColumnFamilies(cf_names, reopen_options);
    EXPECT_OK(s);
    return s.ok();
  };

  {
    SCOPED_TRACE("N=1 insert failure");
    options.wal_partitions = 1;
    DestroyAndReopen(options);
    listener->calls.store(0);
    const Status failed = missing_cf_write(1);
    EXPECT_TRUE(failed.IsInvalidArgument()) << failed.ToString();
    EXPECT_LE(1, listener->calls.load());
    EXPECT_OK(dbfull()->TEST_GetBGError());
    EXPECT_OK(Put("next", "value"));
  }
  {
    SCOPED_TRACE("N=2 insert failure");
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    listener->calls.store(0);
    const Status failed = missing_cf_write(2);
    EXPECT_TRUE(failed.IsInvalidArgument()) << failed.ToString();
    EXPECT_LE(1, listener->calls.load());
    const Status bg = dbfull()->TEST_GetBGError();
    EXPECT_NOK(bg);
    EXPECT_EQ(bg.ToString(),
              PutWithin(cf_for_partition_[1], "later").ToString());
    if (reopen()) {
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "later"));
    }
  }
  {
    SCOPED_TRACE("N=2 gate failure after insert");
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    ASSERT_OK(Put(cf_for_partition_[1], "acked", "v", sync_wo));
    listener->calls.store(0);
    fail_record.store(true);
    const Status failed = Put(cf_for_partition_[1], "s", "v", sync_wo);
    EXPECT_FALSE(fail_record.load());
    EXPECT_TRUE(failed.IsAborted()) << failed.ToString();
    EXPECT_LE(1, listener->calls.load());
    const Status bg = dbfull()->TEST_GetBGError();
    EXPECT_NOK(bg);
    EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[1], "s"));
    EXPECT_EQ(bg.ToString(),
              PutWithin(cf_for_partition_[0], "later").ToString());
    if (reopen()) {
      EXPECT_EQ("NOT_FOUND", Get(cf_for_partition_[0], "later"));
      EXPECT_EQ("v", Get(cf_for_partition_[1], "acked"));
    }
  }
  {
    SCOPED_TRACE("N=2 gate failure with nothing inserted");
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    listener->calls.store(0);
    fail_record.store(true);
    WriteBatch batch;
    ASSERT_OK(batch.Put(handles_[cf_for_partition_[1]], "wal_only", "v"));
    const Status failed = WalOnlyWrite(sync_wo, &batch);
    EXPECT_FALSE(fail_record.load());
    EXPECT_TRUE(failed.IsAborted()) << failed.ToString();
    EXPECT_OK(dbfull()->TEST_GetBGError());
    EXPECT_OK(PutWithin(cf_for_partition_[0], "later"));
    EXPECT_EQ("value", Get(cf_for_partition_[0], "later"));
  }
}

// GC34 Q3 (D-GC34b (i)): W, an async write on partition 1, appended,
// inserted and was marked applied at group exit, and parks in its gate behind
// A, a sync write on partition 0 held in its gate fsync. Shutdown
// (CancelAllBackgroundWork) wakes W, which keeps waiting: once A's fsync
// completes and A publishes, W is covered and returns OK, visible in the
// session and after reopen. Before GC34 W returned ShutdownInProgress, yet
// became visible once A published (D-1).
// Neuter: restore the unconditional shutdown leg in WaitForWalDurability; W
// returns ShutdownInProgress and is visible.
TEST_P(DBWalAppendFailureTest,
       GC34ShutdownWakeKeepsStatusAndVisibilityTogether) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  GC28Hold parks;
  parks.Arm("DBImpl::WaitForWalDurability:BeforePark", nullptr,
            /*hold=*/false);
  GC28Hold wakes;
  wakes.Arm("DBImpl::WaitForWalDurability:AfterWake", nullptr,
            /*hold=*/false);
  GC28Hold fsync;
  fsync.Arm("DBImpl::SyncWalPartitionFiles:BeforeSyncPartition",
            [](void* arg) { return *static_cast<size_t*>(arg) == 0; });
  Writer a([&] { return Put(cf_a, "a", "value", sync_wo); });
  ASSERT_TRUE(fsync.WaitHeld());
  Writer w([&] { return Put(cf_b, "w", "value"); });
  ASSERT_TRUE(parks.WaitHits(1));
  wakes.ResetHits();
  CancelAllBackgroundWork(db_.get(), /*wait=*/false);
  // The shutdown wake reached W (nothing else wakes it while A is held).
  ASSERT_TRUE(wakes.WaitHits(1));
  fsync.Release();
  const Status a_status = a.Join();
  const Status w_status = w.Join();
  EXPECT_OK(a_status);
  EXPECT_OK(w_status);
  EXPECT_EQ("value", Get(cf_b, "w"));
  EXPECT_EQ("value", Get(cf_a, "a"));
  ReopenWithPartitions(2, options);
  EXPECT_EQ("value", Get(cf_for_partition_[1], "w"));
  EXPECT_EQ("value", Get(cf_for_partition_[0], "a"));
}

// GC34 Q3, the not-applied leg: A, a sync write on partition 0, appended
// and inserted, waits in its gate for X, an async write below it on
// partition 1 held before its append. A's range is not applied (a sync range
// publishes only once fsynced), and A would still sync, record and publish
// after the wait, so shutdown fails A with ShutdownInProgress and its gate
// fences the range: A is never visible in the session, while X, released,
// completes. (A's record reached the WAL, so a reopen may recover it, as
// after any failed sync; see wal_partitions in options.h.)
// Neuter: pass applied=true from RunCompletionGate's written wait; A keeps
// waiting for X, so "A kept waiting through shutdown" fails.
TEST_P(DBWalAppendFailureTest, GC34ShutdownFailsUnappliedWaiter) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  const int cf_a = cf_for_partition_[0];
  const int cf_b = cf_for_partition_[1];
  WriteOptions sync_wo;
  sync_wo.sync = true;
  GC28Hold parks;
  parks.Arm("DBImpl::WaitForWalDurability:BeforePark", nullptr,
            /*hold=*/false);
  Defer release([this] { ReleaseHeldAppend(); });
  HoldNextAppend();
  Writer x([&] { return Put(cf_b, "x", "value"); });
  EXPECT_TRUE(WaitUntilHeld());
  Writer a([&] { return Put(cf_a, "a", "value", sync_wo); });
  EXPECT_TRUE(parks.WaitHits(1));
  CancelAllBackgroundWork(db_.get(), /*wait=*/false);
  // Bounded, and X is released either way, so a waiter that ignores
  // shutdown fails the test instead of aborting in Join().
  EXPECT_TRUE(a.WaitDone(std::chrono::seconds(10)))
      << "A kept waiting through shutdown";
  ReleaseHeldAppend();
  const Status a_status = a.Join();
  const Status x_status = x.Join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  EXPECT_TRUE(a_status.IsShutdownInProgress()) << a_status.ToString();
  EXPECT_OK(x_status);
  EXPECT_EQ("NOT_FOUND", Get(cf_a, "a"));
  EXPECT_EQ("value", Get(cf_b, "x"));
}

// Q2 (N-2-4): A is held in its append on partition 0 and W parks above it on
// partition 1; with `background_error`, C on partition 1 then fails its
// append above W, which leaves W parked (D-N71 (b)). Close() runs on another
// thread and its shutdown marker wakes W, whose range is applied, so W keeps
// waiting (GC34 Q3, D-GC34b (i)): once A, released, publishes, W is covered
// and returns OK (C's fence lies above it), and Close() waits for every
// parked write to leave the gate before it frees it. Close() is held right
// after that wait until A has finished: a write still running when teardown
// starts is a write concurrent with Close(). After reopen A and W are there
// and C is not.
void ExpectCloseWakesParkedWrite(DBWalAppendFailureTest* test,
                                 bool background_error) {
  Options options = test->FailureOptions();
  test->DestroyAndReopen(options);
  test->OpenWithPartitions(2, options);
  const int cf_a = test->cf_for_partition_[0];
  const int cf_b = test->cf_for_partition_[1];
  std::mutex mu;
  std::condition_variable cv;
  int parks = 0;
  bool closing = false;
  int wakes_while_closing = 0;
  bool close_drained = false;
  bool resume_close = false;
  // Reported under `mu`, so everything the parked write did before parking
  // (reading the handle Close() destroys) happens before Close() starts.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForWalDurability:BeforePark", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        ++parks;
        cv.notify_all();
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WaitForWalDurability:AfterWake", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        if (closing) {
          ++wakes_while_closing;
          cv.notify_all();
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CloseHelper:WalDurabilityWaitersLeft", [&](void*) {
        std::unique_lock<std::mutex> lock(mu);
        close_drained = true;
        cv.notify_all();
        WaitForReleaseOrAbandon(lock, cv, [&] { return resume_close; });
      });
  // Releases the held append and Close() on every exit, so that no thread
  // is left parked on them, and clears the callbacks that capture locals.
  Defer release([&] {
    test->ReleaseHeldAppend();
    {
      std::lock_guard<std::mutex> lock(mu);
      resume_close = true;
      cv.notify_all();
    }
  });
  Defer clear_callbacks([] {
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::CloseHelper:WalDurabilityWaitersLeft");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WaitForWalDurability:AfterWake");
    SyncPoint::GetInstance()->ClearCallBack(
        "DBImpl::WaitForWalDurability:BeforePark");
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  });
  auto wait_for = [&](const std::function<bool()>& done) {
    std::unique_lock<std::mutex> lock(mu);
    return cv.wait_for(lock, std::chrono::seconds(30), done);
  };
  test->HoldNextAppend();
  DBWalAppendFailureTest::Writer held(
      [&] { return test->Put(cf_a, "held", "value"); });
  EXPECT_TRUE(test->WaitUntilHeld());
  DBWalAppendFailureTest::Writer parked(
      [&] { return test->Put(cf_b, "parked", "value"); });
  EXPECT_TRUE(wait_for([&] { return parks >= 1; }));
  EXPECT_TRUE(test->WaitForDurabilityWaiters(1));
  if (background_error) {
    test->fail_fs_->Arm(test->dbfull()->TEST_CurrentWalNumber(1),
                        DBWalAppendFailureTest::Retryable("injected"));
    // Bounded: if W were stuck behind held A, this write would queue behind
    // it, and A is released only below.
    EXPECT_NOK(test->PutWithin(cf_b, "failed"));
    test->fail_fs_->Disarm();
    EXPECT_NOK(test->dbfull()->TEST_GetBGError());
    // The background error woke W, which parked again.
    EXPECT_TRUE(wait_for([&] { return parks >= 2; }));
    EXPECT_TRUE(test->WaitForDurabilityWaiters(1));
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    closing = true;
  }
  DBWalAppendFailureTest::Writer closer([&] {
    test->Close();
    return Status::OK();
  });
  // The shutdown marker woke W (nothing else can while A is held).
  EXPECT_TRUE(wait_for([&] { return wakes_while_closing >= 1; }));
  test->ReleaseHeldAppend();
  const Status parked_status = parked.Join();
  const Status held_status = held.Join();
  EXPECT_TRUE(wait_for([&] { return close_drained; }));
  {
    std::lock_guard<std::mutex> lock(mu);
    resume_close = true;
    cv.notify_all();
  }
  EXPECT_OK(closer.Join());

  EXPECT_OK(parked_status);
  ASSERT_OK(held_status);
  test->ReopenWithPartitions(2, options);
  EXPECT_EQ("value", test->Get(test->cf_for_partition_[0], "held"));
  EXPECT_EQ("value", test->Get(test->cf_for_partition_[1], "parked"));
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
          WaitForReleaseOrAbandon(lock, cv, [&] { return release_b; });
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
        WaitForReleaseOrAbandon(lock, cv, [&] { return release; });
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
        WaitForReleaseOrAbandon(lock, cv, [&] { return release_a; });
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
        WaitForReleaseOrAbandon(lock, cv, [&] { return release_b; });
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
        WaitForReleaseOrAbandon(lock, cv, [&] { return release_a; });
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
          WaitForReleaseOrAbandon(lock, cv,
                                  [&] { return release_retry_flush; });
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

// GC24 Q4: more than one partition refuses not persisting user-defined
// timestamps, the only setting that strips them in a flush, so the stripping
// flush path never runs above one partition. With concurrent memtable writes
// the refusal names wal_partitions and persist_user_defined_timestamps;
// without them, Open refuses allow_concurrent_memtable_write = false first.
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
    EXPECT_NE(
        std::string::npos,
        s.ToString().find(concurrent ? "wal_partitions > 1 is not supported "
                                       "with persist_user_defined_timestamps"
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

// A failed gate sync fails a zero-count write allocated above it. Z, an async
// WAL-only write that consumes no sequence, allocates right after B, so its
// empty range sits at B's last sequence + 1. B (sync, on partition 1) is
// held inside its gate fsync and Z (on partition 0) inside its append; B's
// fsync fails, then Z is released and fails with B's error: B's open fence,
// installed at B's first sequence, holds Z's gate (target B's last
// sequence), with no exemption for an empty range.
// Ported from GC25's closed Muse candidate.
// Neuter: Z's async completion gate returns OK at once for a group that
// consumes no sequence (an empty-range exemption); Z keeps its OK.
TEST_P(DBWalAppendFailureTest, FailedGateSyncFailsZeroCountWriteAbove) {
  Options options = FailureOptions();
  OpenWithPartitions(2, options);
  PutBeforeFailure(this);
  const int cf_b = cf_for_partition_[1];
  const uint64_t wal1 = dbfull()->TEST_CurrentWalNumber(1);
  std::mutex mu;
  std::condition_variable cv;
  bool b_in_fsync = false;
  bool release_b = false;
  bool z_in_append = false;
  bool release_z = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition", [&](void* arg) {
        if (*static_cast<size_t*>(arg) != 1) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        if (b_in_fsync) {
          return;
        }
        b_in_fsync = true;
        cv.notify_all();
        WaitForReleaseOrAbandon(lock, cv, [&] { return release_b; });
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        if (!static_cast<Slice*>(arg)->ends_with("marker_z")) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
        z_in_append = true;
        cv.notify_all();
        WaitForReleaseOrAbandon(lock, cv, [&] { return release_z; });
      });
  WriteOptions sync_wo;
  sync_wo.sync = true;
  Writer b([&] { return Put(cf_b, "key_b", "value_b", sync_wo); });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return b_in_fsync; }));
  }
  // B is inserted but not published while its fsync runs.
  const SequenceNumber b_last = db_->GetLatestSequenceNumber();
  // A LogData-only batch routes to partition 0.
  Writer z([&] {
    WriteBatch marker;
    Status s = marker.PutLogData("marker_z");
    return s.ok() ? db_->Write(WriteOptions(), &marker) : s;
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return z_in_append; }));
  }
  fail_fs_->Arm(wal1, Retryable("injected sync error"));
  {
    std::lock_guard<std::mutex> lock(mu);
    release_b = true;
    cv.notify_all();
  }
  const Status b_status = b.Join();
  fail_fs_->Disarm();
  // The fences while Z is still held; Z is released and joined before the
  // checks.
  const auto fences = dbfull()->TEST_WalGateFences();
  {
    std::lock_guard<std::mutex> lock(mu);
    release_z = true;
    cv.notify_all();
  }
  const Status z_status = z.Join();
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SyncWalPartitionFiles:BeforeSyncPartition");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  ASSERT_NOK(b_status);
  EXPECT_TRUE(b_status.IsIOError());
  ASSERT_EQ(1U, fences.size());
  EXPECT_EQ(b_last + 1, fences[0].first);
  EXPECT_EQ(kMaxSequenceNumber, fences[0].second);
  ASSERT_NOK(z_status);
  EXPECT_TRUE(z_status.IsIOError());
  EXPECT_EQ("NOT_FOUND", Get(cf_b, "key_b"));
  ASSERT_OK(dbfull()->Resume());
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
    EXPECT_TRUE(
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
    EXPECT_TRUE(cv.wait_for(lock, std::chrono::seconds(30),
                            [&] { return b1_waiting; }));
  }
  port::Thread b2_thread([&] { ASSERT_OK(Put(cf_b, "b2", "value")); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));

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

// GC27 Q2 (former M4-F1.1, GC25 carry-over; AMB-M4-23): above one partition
// a sync group leaves its write queue before its sync writers are fsynced,
// inserted and published (D-9). LockWAL() enters every queue and then waits
// for such a group, so once it returns nothing becomes visible and
// LastSequence() stays put until UnlockWAL(). A sync write S is held ahead of
// its gate fsync while another thread calls LockWAL(); S is released once
// LockWAL() is waiting (or, at the base, has returned). Whatever LockWAL()
// saw on return is what the DB shows until UnlockWAL(): S's key and the last
// sequence. Neuter: drop the wait from LockWAL(); it returns while S is held,
// and S then publishes under the lock.
TEST_P(DBWritePartitionTest, LockWalWaitsForPendingSyncGroup) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("N=" + std::to_string(partitions));
    Options options = CurrentOptions();
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[1];
    ASSERT_OK(Put(cf, "before", "v"));

    std::mutex mu;
    std::condition_variable cv;
    bool held = false;
    bool release = false;
    std::atomic<bool> lock_waiting{false};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (held) {
            return;
          }
          held = true;
          cv.notify_all();
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return release; });
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WaitForUnpublishedSyncGroups:Wait",
        [&](void*) { lock_waiting = true; });
    SyncPoint::GetInstance()->EnableProcessing();

    port::Thread sync_writer([&] {
      WriteOptions sync_wo;
      sync_wo.sync = true;
      ASSERT_OK(db_->Put(sync_wo, handles_[cf], "s", "v"));
    });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return held; }));
    }
    std::atomic<bool> locked{false};
    std::atomic<bool> check{false};
    SequenceNumber seq_at_lock = 0;
    std::string s_at_lock;
    SequenceNumber seq_before_unlock = 0;
    std::string s_before_unlock;
    // UnlockWAL() must come from the thread that called LockWAL().
    port::Thread locker([&] {
      ASSERT_OK(db_->LockWAL());
      seq_at_lock = db_->GetLatestSequenceNumber();
      s_at_lock = Get(cf, "s");
      locked = true;
      while (!check) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      seq_before_unlock = db_->GetLatestSequenceNumber();
      s_before_unlock = Get(cf, "s");
      ASSERT_OK(db_->UnlockWAL());
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!lock_waiting && !locked &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(lock_waiting || locked);
    {
      std::lock_guard<std::mutex> lock(mu);
      release = true;
      cv.notify_all();
    }
    sync_writer.join();
    check = true;
    locker.join();
    // Under the lock: exactly what LockWAL() returned to.
    EXPECT_EQ(seq_at_lock, seq_before_unlock);
    EXPECT_EQ(s_at_lock, s_before_unlock);
    EXPECT_EQ("v", Get(cf, "s"));
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  }
}

// LockWalWaitsForPendingSyncGroup for CreateColumnFamilyWithImport: the
// import holds every write queue, and the job may raise the last allocated
// sequence to the imported files' largest seqno, so it first waits for every
// sync group that left its queue unpublished. A sync write S is held ahead of
// its gate fsync while another thread imports a column family; the import
// waits (DBImpl::WaitForUnpublishedSyncGroups:Wait) and has not returned
// while S is held, then completes once S is released. S and the imported
// key both read back. Neuter: drop the wait from
// DBImpl::CreateColumnFamilyWithImport; the import runs beside the held
// group and trips AssertLastSequenceCoversAllocated (debug), or returns
// before S is released.
TEST_P(DBWritePartitionTest, ImportWaitsForPendingSyncGroup) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("N=" + std::to_string(partitions));
    Options options = CurrentOptions();
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int export_cf = cf_for_partition_[0];
    const int cf = cf_for_partition_[1];
    ASSERT_OK(Put(export_cf, "key", "value"));
    ASSERT_OK(Flush(export_cf));
    Checkpoint* checkpoint = nullptr;
    ASSERT_OK(Checkpoint::Create(db_.get(), &checkpoint));
    std::unique_ptr<Checkpoint> checkpoint_guard(checkpoint);
    const std::string export_dir = dbname_ + "/export";
    ASSERT_OK(DestroyDir(env_, export_dir));
    ExportImportFilesMetaData* metadata = nullptr;
    ASSERT_OK(checkpoint->ExportColumnFamily(handles_[export_cf], export_dir,
                                             &metadata));
    std::unique_ptr<ExportImportFilesMetaData> metadata_guard(metadata);

    std::mutex mu;
    std::condition_variable cv;
    bool held = false;
    bool release = false;
    std::atomic<bool> import_waiting{false};
    std::atomic<bool> imported_done{false};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (held) {
            return;
          }
          held = true;
          cv.notify_all();
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return release; });
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WaitForUnpublishedSyncGroups:Wait",
        [&](void*) { import_waiting = true; });
    SyncPoint::GetInstance()->EnableProcessing();

    Status sync_status;
    port::Thread sync_writer([&] {
      WriteOptions sync_wo;
      sync_wo.sync = true;
      sync_status = db_->Put(sync_wo, handles_[cf], "s", "v");
    });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return held; }));
    }
    Status import_status;
    ColumnFamilyHandle* imported = nullptr;
    port::Thread importer([&] {
      import_status = db_->CreateColumnFamilyWithImport(
          options, "imported", ImportColumnFamilyOptions(), *metadata,
          &imported);
      imported_done = true;
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!import_waiting && !imported_done &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(import_waiting);
    EXPECT_FALSE(imported_done);
    {
      std::lock_guard<std::mutex> lock(mu);
      release = true;
      cv.notify_all();
    }
    sync_writer.join();
    importer.join();
    std::unique_ptr<ColumnFamilyHandle> imported_guard(imported);
    ASSERT_OK(sync_status);
    ASSERT_OK(import_status);
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    EXPECT_EQ("v", Get(cf, "s"));
    std::string value;
    ASSERT_OK(db_->Get(ReadOptions(), imported, "key", &value));
    EXPECT_EQ("value", value);
  }
}

// GC27 Q5 (a) (D-3, L3): an opener with track_and_verify_wals classifies the
// recovery set like any other, and merges one written with more than one
// partition, logging one warning that names the option; it verifies
// predecessor WAL info only on a single chain. At one partition, under both
// usages: an interleaved two-partition set opens with every key, and a
// non-interleaved one whose point-in-time cut is declared on disk opens
// without the cut record. A one-partition indexed chain missing its middle
// WAL still fails on the predecessor check. At the base the option forced
// file-order replay: "sequence number is being set backwards" on the
// interleaved set, and the cut record replayed on the other. Neuter: select
// the sequential path whenever the option is set (as before). (With MANIFEST
// tracking off, the single chain's missing WAL is reported by the predecessor
// check as a mismatched predecessor log number.)
TEST_P(DBWritePartitionTest, TrackAndVerifyWalsOpenerMergesPartitionedSet) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  auto reopen_verifying = [&](Options options, PartitionWALUsage usage,
                              WALRecoveryMode mode) {
    options.partition_wal_usage = usage;
    options.track_and_verify_wals = true;
    options.wal_recovery_mode = mode;
    ReopenWithPartitions(1, options);
  };
  auto warned = [&]() {
    std::string log;
    EXPECT_OK(ReadFileToString(env_, dbname_ + "/LOG", &log));
    return log.find(
               "track_and_verify_wals: the recovery set was written "
               "with wal_partitions > 1") != std::string::npos;
  };
  for (const PartitionWALUsage usage :
       {PartitionWALUsage::kNone,
        PartitionWALUsage::kPartitionByColumnFamily}) {
    SCOPED_TRACE("usage=" + std::to_string(static_cast<int>(usage)));
    {
      SCOPED_TRACE("interleaved");
      Options options = IndexedWalOptions(CurrentOptions());
      DestroyAndReopen(options);
      OpenWithPartitions(2, options);
      const std::vector<int> cfs = cf_for_partition_;
      for (int i = 0; i < 6; ++i) {
        ASSERT_OK(Put(cfs[i % 2], "k" + std::to_string(i), "v"));
      }
      reopen_verifying(options, usage, WALRecoveryMode::kPointInTimeRecovery);
      for (int i = 0; i < 6; ++i) {
        EXPECT_EQ("v", Get(cfs[i % 2], "k" + std::to_string(i))) << i;
      }
      Close();
      EXPECT_TRUE(warned());
    }
    {
      SCOPED_TRACE("non-interleaved, declared cut");
      Options options = IndexedWalOptions(CurrentOptions());
      DestroyAndReopen(options);
      OpenWithPartitions(2, options);
      const std::vector<int> cfs = cf_for_partition_;
      ASSERT_OK(Put(cfs[0], "a0", "v"));
      ASSERT_OK(Put(cfs[1], "b0", "v"));
      const uint64_t lost_wal = dbfull()->TEST_CurrentWalNumber(0);
      uint64_t lost_begin = 0;
      ASSERT_OK(env_->GetFileSize(LogFileName(dbname_, lost_wal), &lost_begin));
      ASSERT_OK(Put(cfs[0], "lost", "v"));
      ASSERT_OK(Put(cfs[1], "above", "v"));
      Close();
      ASSERT_OK(
          test::TruncateFile(env_, LogFileName(dbname_, lost_wal), lost_begin));
      options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
      ReopenWithPartitions(2, options);
      ASSERT_EQ("NOT_FOUND", Get(cfs[1], "above"));
      Close();
      reopen_verifying(options, usage,
                       WALRecoveryMode::kTolerateCorruptedTailRecords);
      EXPECT_EQ("v", Get(cfs[0], "a0"));
      EXPECT_EQ("v", Get(cfs[1], "b0"));
      EXPECT_EQ("NOT_FOUND", Get(cfs[0], "lost"));
      EXPECT_EQ("NOT_FOUND", Get(cfs[1], "above"));
      Close();
      EXPECT_TRUE(warned());
    }
  }
  {
    SCOPED_TRACE("single chain");
    Options options = IndexedWalOptions(CurrentOptions());
    options.track_and_verify_wals = true;
    options.track_and_verify_wals_in_manifest = false;
    DestroyAndReopen(options);
    OpenWithPartitions(1, options);
    for (int round = 0; round < 3; ++round) {
      if (round > 0) {
        ASSERT_OK(dbfull()->TEST_SwitchMemtable(
            static_cast_with_check<ColumnFamilyHandleImpl>(
                handles_[cf_for_partition_[0]])
                ->cfd()));
      }
      ASSERT_OK(Put(cf_for_partition_[0], "r" + std::to_string(round), "v"));
    }
    const uint64_t middle = AliveWalNumbers()[0][1];
    Close();
    ASSERT_OK(env_->DeleteFile(LogFileName(dbname_, middle)));
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
    const Status s = TryReopenWithColumnFamilies(cf_names, options);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    // The predecessor check, not the MANIFEST's (off here), names the gap.
    EXPECT_NE(std::string::npos,
              s.ToString().find("Mismatched predecessor log number"))
        << s.ToString();
  }
}

// Fails NewSequentialFile for one file, as an unreadable WAL would.
class FailOpenFS : public FileSystemWrapper {
 public:
  FailOpenFS(const std::shared_ptr<FileSystem>& base, std::string fname)
      : FileSystemWrapper(base), fname_(std::move(fname)) {}
  static const char* kClassName() { return "FailOpenFS"; }
  const char* Name() const override { return kClassName(); }
  IOStatus NewSequentialFile(const std::string& fname, const FileOptions& opts,
                             std::unique_ptr<FSSequentialFile>* result,
                             IODebugContext* dbg) override {
    if (fname == fname_) {
      return IOStatus::IOError("injected open failure");
    }
    return target()->NewSequentialFile(fname, opts, result, dbg);
  }

 private:
  const std::string fname_;
};

// GC27 Q5 (d) (L13, K4; D-GC27b): a WAL whose head cannot be read is of an
// unknown partition, not partition 0. A DB written at two partitions has its
// partition 1 WAL's first record damaged, or the file unreadable. RepairDB
// with wal_partitions > 1 refuses with NotSupported naming the WAL and leaves
// the DB unchanged. At the base the head read returned partition 0, and
// RepairDB converted the set. Neuter: read an unreadable head as partition 0
// (as before). GetUpdatesSince at a one-partition opener of the set (the
// MANIFEST guard marks it indexed) with its partition 1 WALs' heads damaged
// fails too, but it already did at the base: reading each WAL's first
// sequence reports the damage first (a pin; WalManager's refusal of an
// unknown head is argued, not tested).
TEST_P(DBWritePartitionTest, RepairRefusesDamagedHeadOfPartitionedWal) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  auto db_files = [this] {
    std::vector<std::string> children;
    EXPECT_OK(env_->GetChildren(dbname_, &children));
    std::map<std::string, std::string> files;
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (ParseFileName(child, &number, &type) && type != kInfoLogFile &&
          type != kDBLockFile) {
        EXPECT_OK(ReadFileToString(env_, dbname_ + "/" + child, &files[child]));
      }
    }
    return files;
  };
  auto damage_head = [this](const std::string& fname) {
    std::string contents;
    ASSERT_OK(ReadFileToString(env_, fname, &contents));
    ASSERT_FALSE(contents.empty());
    contents[0] ^= 0x5a;
    ASSERT_OK(WriteStringToFile(env_, contents, fname, /*should_sync=*/true));
  };
  for (const bool unreadable : {false, true}) {
    SCOPED_TRACE(unreadable ? "unreadable" : "damaged head");
    Options options = IndexedWalOptions(CurrentOptions());
    DestroyAndReopen(options);
    OpenWithPartitions(2, options);
    ASSERT_OK(Put(cf_for_partition_[0], "k0", "v"));
    ASSERT_OK(Put(cf_for_partition_[1], "k1", "v"));
    ASSERT_OK(Put(0, "default_key", "v"));
    const uint64_t p1_wal = dbfull()->TEST_CurrentWalNumber(1);
    Close();
    const std::string p1_fname = LogFileName(dbname_, p1_wal);
    std::shared_ptr<FailOpenFS> fail_fs;
    std::unique_ptr<Env> fail_env;
    if (unreadable) {
      fail_fs = std::make_shared<FailOpenFS>(env_->GetFileSystem(), p1_fname);
      fail_env = NewCompositeEnv(fail_fs);
      options.env = fail_env.get();
    } else {
      damage_head(p1_fname);
    }
    const std::map<std::string, std::string> before = db_files();
    std::vector<std::string> cf_names;
    ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
    std::vector<ColumnFamilyDescriptor> column_families;
    for (const std::string& name : cf_names) {
      column_families.emplace_back(name, options);
    }
    options.wal_partitions = 2;
    const Status s = RepairDB(dbname_, options, column_families);
    ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
    EXPECT_NE(std::string::npos,
              s.ToString().find("WAL " + std::to_string(p1_wal)))
        << s.ToString();
    EXPECT_EQ(before, db_files());
  }

  // GetUpdatesSince over archived partition 1 WALs with damaged heads.
  Options options = IndexedWalOptions(CurrentOptions());
  options.WAL_ttl_seconds = 1000;
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  ASSERT_OK(Put(cf_for_partition_[0], "k0", "v"));
  ASSERT_OK(Put(cf_for_partition_[1], "k1", "v"));
  FlushAllColumnFamilies();
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
  ReopenWithPartitions(1, options);
  ASSERT_OK(Put(cf_for_partition_[0], "k2", "v"));
  int damaged = 0;
  for (const std::string& dir : {dbname_, ArchivalDirectory(dbname_)}) {
    std::vector<std::string> children;
    ASSERT_OK(env_->GetChildren(dir, &children));
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      if (!ParseFileName(child, &number, &type) || type != kWalFile) {
        continue;
      }
      uint32_t partition = 0;
      ASSERT_OK(log::ReadWALPartition(env_->GetFileSystem().get(),
                                      dir + "/" + child, FileOptions(), number,
                                      &partition));
      if (partition != 0) {
        damage_head(dir + "/" + child);
        ++damaged;
      }
    }
  }
  ASSERT_GT(damaged, 0);
  std::unique_ptr<TransactionLogIterator> iter;
  const Status s = db_->GetUpdatesSince(1, &iter);
  ASSERT_NOK(s);
  EXPECT_EQ(nullptr, iter);
}

// Keys a tailing iterator shows from the first, after a fresh seek.
static std::vector<std::string> TailingKeys(Iterator* it) {
  std::vector<std::string> keys;
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    keys.push_back(it->key().ToString());
  }
  EXPECT_OK(it->status());
  return keys;
}

// GC27 Q4 (D-GC27a): above one partition an async write in its own group
// above a pending sync write inserts into the memtable before its completion
// gate waits for the sync write, and fails when that write's fsync fails. A
// tailing iterator reads the memtables at kMaxSequenceNumber; it is now
// bounded by the published sequence on every visibility check, so it never
// shows the failed write: not while it is inserted and parked, not after the
// failure, before or after Resume(), whether created before the failure, new,
// or opened through NewIterators() (a tailing iterator does not support
// Refresh()). It still shows writes as they publish. Neuter: tail at
// kMaxSequenceNumber with no read callback (as before); "zebra" shows while
// parked and before Resume.
TEST_P(DBWritePartitionTest,
       TailingIteratorNeverShowsFailedWriteAboveFailedSync) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("N=" + std::to_string(partitions));
    Options options = CurrentOptions();
    options.paranoid_checks = true;
    options.max_bgerror_resume_count = 0;
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf = cf_for_partition_[0];
    ASSERT_OK(Put(cf, "f", "value_f"));
    ReadOptions tailing;
    tailing.tailing = true;
    std::unique_ptr<Iterator> early(db_->NewIterator(tailing, handles_[cf]));
    const std::vector<std::string> kOnlyF = {"f"};
    ASSERT_EQ(kOnlyF, TailingKeys(early.get()));

    std::mutex mu;
    std::condition_variable cv;
    bool held = false;
    bool release = false;
    std::atomic<bool> fail_sync{false};
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::CompletionGate:BeforeCoalescedSync", [&](void*) {
          std::unique_lock<std::mutex> lock(mu);
          if (held) {
            return;
          }
          held = true;
          cv.notify_all();
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return release; });
        });
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::SyncWalsFromCompletionGate:AfterGateSync", [&](void* arg) {
          if (fail_sync.exchange(false)) {
            IOStatus error = IOStatus::IOError("injected gate sync error");
            error.SetRetryable(true);
            *static_cast<IOStatus*>(arg) = error;
          }
        });
    SyncPoint::GetInstance()->EnableProcessing();
    Status s_status;
    port::Thread s_thread([&] {
      WriteOptions sync_wo;
      sync_wo.sync = true;
      s_status = db_->Put(sync_wo, handles_[cf], "s", "value_s");
    });
    {
      std::unique_lock<std::mutex> lock(mu);
      EXPECT_TRUE(
          cv.wait_for(lock, std::chrono::seconds(30), [&] { return held; }));
    }
    Status c_status;
    port::Thread c_thread([&] { c_status = Put(cf, "zebra", "value_zebra"); });
    EXPECT_TRUE(WaitForDurabilityWaiters(1));
    // Inserted, parked, not published.
    {
      std::unique_ptr<Iterator> fresh(db_->NewIterator(tailing, handles_[cf]));
      EXPECT_EQ(kOnlyF, TailingKeys(fresh.get()));
      EXPECT_EQ(kOnlyF, TailingKeys(early.get()));
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      fail_sync = true;
      release = true;
      cv.notify_all();
    }
    s_thread.join();
    c_thread.join();
    ASSERT_TRUE(s_status.IsIOError()) << s_status.ToString();
    ASSERT_TRUE(c_status.IsIOError()) << c_status.ToString();
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();

    auto check = [&](const std::string& phase) {
      SCOPED_TRACE(phase);
      EXPECT_EQ(kOnlyF, TailingKeys(early.get()));
      std::unique_ptr<Iterator> fresh(db_->NewIterator(tailing, handles_[cf]));
      EXPECT_EQ(kOnlyF, TailingKeys(fresh.get()));
      std::vector<Iterator*> iterators;
      ASSERT_OK(db_->NewIterators(tailing, {handles_[cf]}, &iterators));
      ASSERT_EQ(1U, iterators.size());
      EXPECT_EQ(kOnlyF, TailingKeys(iterators[0]));
      delete iterators[0];
    };
    check("before resume");
    ASSERT_OK(dbfull()->Resume());
    check("after resume");
    // Published writes show without a refresh.
    ASSERT_OK(Put(cf, "after", "value_after"));
    const std::vector<std::string> kWithAfter = {"after", "f"};
    EXPECT_EQ(kWithAfter, TailingKeys(early.get()));
    early.reset();
  }
}

// The failed-append half of TailingIteratorNeverShowsFailedWriteAboveFailedSync
// (D-1): a write on partition 0 appends above a write whose append on
// partition 1 fails, and parks in its completion gate, inserted. No tailing
// iterator shows it, while parked or after the failure, before or after
// Resume(). Neuter: as there.
TEST_P(DBWalAppendFailureTest,
       TailingIteratorNeverShowsFailedWriteAboveFailedAppend) {
  if (GetParam()) {
    return;
  }
  for (const size_t partitions : {size_t{2}, size_t{8}}) {
    SCOPED_TRACE("N=" + std::to_string(partitions));
    Options options = FailureOptions();
    options.disable_auto_compactions = true;
    DestroyAndReopen(options);
    OpenWithPartitions(partitions, options);
    const int cf_a = cf_for_partition_[0];
    const int cf_b = cf_for_partition_[1];
    ASSERT_OK(Put(cf_a, "f", "value_f"));
    ReadOptions tailing;
    tailing.tailing = true;
    std::unique_ptr<Iterator> early(db_->NewIterator(tailing, handles_[cf_a]));
    const std::vector<std::string> kOnlyF = {"f"};
    ASSERT_EQ(kOnlyF, TailingKeys(early.get()));

    HoldNextAppend();
    Writer held_writer([&] { return Put(cf_b, "failed", "value"); });
    ASSERT_TRUE(WaitUntilHeld());
    const size_t parked_before = dbfull()->TEST_WalDurabilityWaiters();
    Writer above_writer([&] { return Put(cf_a, "zebra", "value"); });
    ASSERT_TRUE(WaitForDurabilityWaiters(parked_before + 1));
    {
      std::unique_ptr<Iterator> fresh(
          db_->NewIterator(tailing, handles_[cf_a]));
      EXPECT_EQ(kOnlyF, TailingKeys(fresh.get()));
      EXPECT_EQ(kOnlyF, TailingKeys(early.get()));
    }
    fail_fs_->Arm(dbfull()->TEST_CurrentWalNumber(1), Retryable("injected"));
    ReleaseHeldAppend();
    ASSERT_NOK(held_writer.Join());
    ASSERT_NOK(above_writer.Join());
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
    fail_fs_->Disarm();
    auto check = [&](const std::string& phase) {
      SCOPED_TRACE(phase);
      EXPECT_EQ(kOnlyF, TailingKeys(early.get()));
      std::unique_ptr<Iterator> fresh(
          db_->NewIterator(tailing, handles_[cf_a]));
      EXPECT_EQ(kOnlyF, TailingKeys(fresh.get()));
    };
    check("before resume");
    ASSERT_OK(dbfull()->Resume());
    check("after resume");
    early.reset();
  }
}

// GC27 Q4: the corrected sibling of
// FailedSyncRangeDeletionHiddenUntilResumeDropsIt without its tailing
// allowance. That test lets a tailing iterator show "zebra", the async write
// that parked behind the failed sync range deletion, before Resume(); a
// tailing iterator is now bounded by the published sequence, so it shows
// neither "point" nor "zebra", and the range deletion never hides "f",
// before Resume() (an iterator created before the failure and a fresh one)
// and after it.
// Neuter: the tailing DBIter at kMaxSequenceNumber (no
// PublishedSequenceReadCallback): "zebra" shows before Resume().
TEST_P(DBWritePartitionTest,
       FailedSyncRangeDeletionHiddenFromTailingIteratorBeforeResume) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  Options options = CurrentOptions();
  options.paranoid_checks = true;
  options.max_bgerror_resume_count = 0;
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);
  OpenWithPartitions(2, options);
  const int cf = cf_for_partition_[0];
  ASSERT_OK(Put(cf, "f", "value_f"));
  ReadOptions tailing_ro;
  tailing_ro.tailing = true;
  std::unique_ptr<Iterator> early(db_->NewIterator(tailing_ro, handles_[cf]));
  ASSERT_EQ(std::vector<std::string>{"f"}, TailingKeys(early.get()));

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
    EXPECT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(30), [&] { return f_in_sync; }));
  }
  Status zebra_status;
  port::Thread zebra_thread(
      [&] { zebra_status = Put(cf, "zebra", "value_zebra"); });
  EXPECT_TRUE(WaitForDurabilityWaiters(1));
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_syncs = 1;
    release_f = true;
    cv.notify_all();
  }
  f_thread.join();
  zebra_thread.join();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_TRUE(w_status.IsIOError()) << w_status.ToString();
  ASSERT_TRUE(zebra_status.IsIOError()) << zebra_status.ToString();

  const std::vector<std::string> kOnlyF = {"f"};
  {
    SCOPED_TRACE("before resume");
    std::unique_ptr<Iterator> fresh(db_->NewIterator(tailing_ro, handles_[cf]));
    EXPECT_EQ(kOnlyF, TailingKeys(fresh.get()));
    EXPECT_EQ(kOnlyF, TailingKeys(early.get()));
  }
  early.reset();
  ASSERT_OK(dbfull()->Resume());
  ASSERT_OK(Flush(cf));
  {
    SCOPED_TRACE("after resume");
    std::unique_ptr<Iterator> fresh(db_->NewIterator(tailing_ro, handles_[cf]));
    EXPECT_EQ(kOnlyF, TailingKeys(fresh.get()));
  }
  Close();
}

// GC27 Q7: ported from GC26's losing candidate (Muse, babe10dd6620), with
// credit; adaptations are listed in the summary.
// GC26 Q4 (D-10): at N=1 indexed, a record-boundary loss with records above
// is a hole: tolerate and absolute fail, skip skips it and applies the
// records above and below, and point-in-time opens the prefix below it.
// Neuter: SetSkipUnexplainedHoles(false); skip mode fails on the hole.
TEST_P(DBWritePartitionTest, SkipModeSkipsBoundaryLossAtOnePartition) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(1, options);
  const int cf = cf_for_partition_[0];
  ASSERT_OK(Put(cf, "kept", "value"));
  ASSERT_OK(Put(cf, "hole", "value"));
  const uint64_t old_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(dbfull()->TEST_SwitchMemtable(
      static_cast_with_check<ColumnFamilyHandleImpl>(handles_[cf])->cfd()));
  ASSERT_OK(Put(cf, "above", "value"));
  Close();
  // Keep the head and the first record; drop the second at its boundary:
  // LastRecordOffset is the start of the last returned record.
  const std::string lost_wal = LogFileName(dbname_, old_wal);
  uint64_t second_record = 0;
  {
    std::unique_ptr<SequentialFileReader> file_reader;
    ASSERT_OK(SequentialFileReader::Create(
        env_->GetFileSystem(), lost_wal, FileOptions(), &file_reader,
        /*dbg=*/nullptr, /*rate_limiter=*/nullptr));
    log::Reader reader(/*info_log=*/nullptr, std::move(file_reader),
                       /*reporter=*/nullptr, /*checksum=*/true, 0);
    std::string scratch;
    Slice record;
    while (reader.ReadRecord(&record, &scratch) &&
           reader.GetLastReadWALIndex() != 2) {
    }
    ASSERT_EQ(2U, reader.GetLastReadWALIndex());
    second_record = reader.LastRecordOffset();
  }
  ASSERT_GT(second_record, 0U);
  ASSERT_OK(test::TruncateFile(env_, lost_wal, second_record));

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = 1;
  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kAbsoluteConsistency}) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    options.wal_recovery_mode = recovery_mode;
    const Status s = TryReopenWithColumnFamilies(cf_names, options);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find("unexplained hole"))
        << s.ToString();
  }
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  EXPECT_EQ("value", Get(cf, "kept"));
  EXPECT_EQ("NOT_FOUND", Get(cf, "hole"));
  EXPECT_EQ("value", Get(cf, "above"));
  Close();
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  EXPECT_EQ("value", Get(cf, "kept"));
  EXPECT_EQ("NOT_FOUND", Get(cf, "hole"));
  EXPECT_EQ("NOT_FOUND", Get(cf, "above"));
  Close();
}

// GC27 Q7: ported from GC26's losing candidate (Muse, babe10dd6620), with
// credit; adaptations are listed in the summary.
// GC26 Q4 (D-10): at N=1 indexed, a torn non-last WAL whose only record is
// torn is a hole, like a boundary loss: tolerate and absolute fail, skip
// skips it and applies the records above it, and point-in-time opens the
// prefix below it. (A torn tail behind intact records instead ends the
// wired modes with the read error itself.)
// Neuter: SetSkipUnexplainedHoles(false); skip mode fails on the hole.
TEST_P(DBWritePartitionTest, SkipModeSkipsTornTailAtOnePartition) {
  if (GetParam()) {
    ROCKSDB_GTEST_BYPASS("N > 1 rejects two_write_queues");
    return;
  }
  Options options = IndexedWalOptions(CurrentOptions());
  OpenWithPartitions(1, options);
  const int cf = cf_for_partition_[0];
  ASSERT_OK(Put(cf, "hole", "value"));
  const uint64_t old_wal = dbfull()->TEST_CurrentWalNumber(0);
  ASSERT_OK(dbfull()->TEST_SwitchMemtable(
      static_cast_with_check<ColumnFamilyHandleImpl>(handles_[cf])->cfd()));
  ASSERT_OK(Put(cf, "above", "value"));
  Close();
  // Tear the last byte off the only record.
  const std::string lost_wal = LogFileName(dbname_, old_wal);
  uint64_t size = 0;
  ASSERT_OK(env_->GetFileSize(lost_wal, &size));
  ASSERT_GT(size, 1U);
  ASSERT_OK(test::TruncateFile(env_, lost_wal, size - 1));

  std::vector<std::string> cf_names;
  ASSERT_OK(DB::ListColumnFamilies(options, dbname_, &cf_names));
  options.wal_partitions = 1;
  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kAbsoluteConsistency}) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    options.wal_recovery_mode = recovery_mode;
    const Status s = TryReopenWithColumnFamilies(cf_names, options);
    ASSERT_TRUE(s.IsCorruption()) << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find("unexplained hole"))
        << s.ToString();
  }
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  EXPECT_EQ("NOT_FOUND", Get(cf, "hole"));
  EXPECT_EQ("value", Get(cf, "above"));
  Close();
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ASSERT_OK(TryReopenWithColumnFamilies(cf_names, options));
  EXPECT_EQ("NOT_FOUND", Get(cf, "hole"));
  EXPECT_EQ("NOT_FOUND", Get(cf, "above"));
  Close();
}

// GC27 Q7: ported from GC26's losing candidate (Muse, babe10dd6620), with
// credit; adaptations are listed in the summary.
// The N>1 column-family options through SetOptions: max_successive_merges is
// the one mutable option of the set, so its runtime change reaches the N>1
// guard and is refused by it; the rest are immutable and refuse with the
// generic not-changeable error before any guard runs.
// Neuter: drop the N>1 max_successive_merges check in
// ColumnFamilyData::ValidateOptions; the change is accepted.
TEST_F(DBWriteTestUnparameterized, WalPartitionsSetOptionsRuntimeConflicts) {
  Options options = CurrentOptions();
  options.wal_partitions = 2;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  options.avoid_flush_during_shutdown = true;
  Destroy(options);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(TryReopen(options));
  ColumnFamilyHandle* cf = db_->DefaultColumnFamily();
  const Status merges = db_->SetOptions(cf, {{"max_successive_merges", "10"}});
  EXPECT_TRUE(merges.IsInvalidArgument()) << merges.ToString();
  EXPECT_NE(std::string::npos, merges.ToString().find("wal_partitions"))
      << merges.ToString();
  EXPECT_NE(std::string::npos, merges.ToString().find("max_successive_merges"))
      << merges.ToString();
  for (const std::pair<const char*, const char*>& opt :
       {std::make_pair("inplace_update_support", "true"),
        std::make_pair("enable_blob_direct_write", "true"),
        std::make_pair("persist_user_defined_timestamps", "false")}) {
    SCOPED_TRACE(opt.first);
    const Status s = db_->SetOptions(cf, {{opt.first, opt.second}});
    EXPECT_TRUE(s.IsInvalidArgument()) << s.ToString();
    EXPECT_NE(std::string::npos, s.ToString().find("Option not changeable"))
        << s.ToString();
  }
  // memtable_factory takes no string value through this API, so there is no
  // runtime change to refuse; CreateColumnFamily above covers its guard.
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
  // GC27 Q6: the failed value is `failed_value_size` bytes, and
  // `appends_before_failure` of its WAL's appends still succeed once armed.
  // GC27 Q7: with `pin_with_file_deletions_`, file deletions stay disabled
  // from the failure on, which keeps the failed WAL on disk for Resume's
  // read-back without a prepare.
  void ResumeAfterLandingThenCrash(size_t num_partitions, bool two_pc,
                                   FailingWalFS::Landing landing,
                                   size_t failed_value_size = 1,
                                   int appends_before_failure = 0) {
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

    fail_fs_->Arm(failed_wal, Retryable("injected"), landing,
                  appends_before_failure);
    ASSERT_NOK(db->Put(WriteOptions(), handles[cf_for_partition_[newest]],
                       "failed", std::string(failed_value_size, 'v')));
    fail_fs_->Disarm();
    const Status bg = db_impl->TEST_GetBGError();
    ASSERT_EQ(Status::Severity::kHardError, bg.severity()) << bg.ToString();
    if (pin_with_file_deletions_) {
      ASSERT_OK(db->DisableFileDeletions());
    }
    const bool pinned = two_pc || pin_with_file_deletions_;
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
    EXPECT_EQ(pinned, env_->FileExists(LogFileName(dbname_, failed_wal)).ok());
    EXPECT_EQ(pinned && landing == FailingWalFS::Landing::kWhole,
              info_log.find("landed whole in WAL #" +
                            std::to_string(failed_wal)) != std::string::npos);
    EXPECT_EQ(pinned && (landing == FailingWalFS::Landing::kTorn ||
                         appends_before_failure > 0),
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
    if (pin_with_file_deletions_) {
      ASSERT_OK(db->EnableFileDeletions());
    }
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

  bool pin_with_file_deletions_ = false;
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

// GC27 Q6 (former M4-F1.3, GC26 carry-over): a failed record larger than
// the WAL writer's buffer (writable_file_max_buffer_size, 1 MB) flushes
// partway before its append fails, so the writer's flushed size at the
// failure, which Resume used as the torn copy's start, lies inside it. Resume
// now zeroes from the end of the last whole record its read-back returned.
// With a prepare keeping the WAL live, a 3 MB value whose first 1 MB flush
// lands: Resume() succeeds, and a crash image opens in all four modes with
// every acknowledged write and the prepare. At the base the record's first
// fragments stayed, and tolerate (and absolute) failed with a checksum
// mismatch. Neuter: zero from the flushed size (as before).
TEST_P(DBWalLandedAppendTest,
       GC27LargeTornRecordFlushedPartwayOpensInEveryMode) {
  ASSERT_NO_FATAL_FAILURE(ResumeAfterLandingThenCrash(
      /*num_partitions=*/2, /*two_pc=*/true, FailingWalFS::Landing::kNothing,
      /*failed_value_size=*/3 << 20, /*appends_before_failure=*/1));
}

// GC27 Q7: ported from GC26's losing candidate (Muse, babe10dd6620), with
// credit: GC26ResumeExplainsOnlyWhatDidNotLand without 2PC. File deletions,
// disabled from the failure on, keep the failed WAL on disk in place of a
// prepare, so Resume's read-back runs the landed-whole and torn paths, and
// every crash image opens in all four modes with every acknowledged write.
// Without a prepare, Resume's flush retires the failed WAL below the recovery
// set, so the images open either way; what fails under each neuter is the
// read-back's verdict in the LOG. Neuters: ExplainBurnedWALIndices voids
// every burned index (ignore `landed`): no "landed whole"; no zeroing of the
// torn copy: no "landed torn".
TEST_P(DBWalLandedAppendTest,
       GC26ResumeExplainsOnlyWhatDidNotLandWithoutTwoPC) {
  using Landing = FailingWalFS::Landing;
  pin_with_file_deletions_ = true;
  for (const size_t num_partitions : {size_t{2}, size_t{8}}) {
    for (const Landing landing :
         {Landing::kWhole, Landing::kTorn, Landing::kNothing}) {
      SCOPED_TRACE("wal_partitions=" + std::to_string(num_partitions) +
                   " landing=" + std::to_string(static_cast<int>(landing)));
      ASSERT_NO_FATAL_FAILURE(ResumeAfterLandingThenCrash(
          num_partitions, /*two_pc=*/false, landing));
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
       Status::kNotSupported,
       "wal_partitions > 1 is not supported with "
       "persist_user_defined_timestamps = false"},
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
