//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

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
#include <thread>
#include <vector>

#include "db/db_impl/wal_durability_tracker.h"
#include "db/db_test_util.h"
#include "db/log_writer.h"
#include "db/write_batch_internal.h"
#include "db/write_callback.h"
#include "db/write_thread.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/utilities/checkpoint.h"
#include "rocksdb/utilities/options_util.h"
#include "test_util/sync_point.h"
#include "util/random.h"
#include "util/string_util.h"
#include "utilities/fault_injection_env.h"
#include "utilities/fault_injection_fs.h"

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

  // Handle index of a column family routed to each partition.
  std::vector<int> cf_for_partition_;
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

TEST_P(DBWritePartitionTest, RejectsSeqPerBatchTwoQueues) {
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

TEST_P(DBWritePartitionTest, SyncWriteSyncsAgainOnlyAfterWait) {
  // M2.6: a sync write's leadership-time fsync covers only what was written
  // before it. When predecessors land after it, the gate syncs again once the
  // whole prefix is written; when nothing was outstanding, the second sync is
  // skipped. SyncWAL runs only on the post-wait path (leadership syncs inline
  // on the standard path, through SyncWAL on the two-queues path), so its
  // count observes the mechanism directly.
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

  std::atomic<int> sync_wal_count{0};
  SyncPoint::GetInstance()->SetCallBack("DBImpl::SyncWAL:Begin",
                                        [&](void*) { ++sync_wal_count; });

  // Phase 1: uncontended sync writes never wait, so no post-wait sync runs.
  // Standard path: leadership syncs inline, so zero SyncWAL calls. Two-queues
  // path: exactly one SyncWAL per put (leadership only); any post-wait sync
  // would show up as extra calls, which also validates the snapshot runs
  // after the append.
  WriteOptions sync_wo;
  sync_wo.sync = true;
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(cf_for_partition_[i % kPartitions],
                  "uncontended" + std::to_string(i), "value", sync_wo));
  }
  const int uncontended_count = sync_wal_count.load();
  EXPECT_EQ(GetParam() ? 3 : 0, uncontended_count);

  // Phase 2 is standard-path only. With two write queues a sync write takes
  // all N WAL mutexes in PreprocessWrite and MarkLogsSynced, so any in-flight
  // predecessor append is mutex-serialized before B's snapshot and the
  // snapshot always covers: the contended interleaving (predecessor allocated
  // but pre-append at the snapshot) is reachable in production only through
  // the pre-append race and cannot be scripted deterministically. (Holding A
  // mid-append instead deadlocks the test, since B-sync then blocks on A's
  // partition mutex before ever appending.) The gate and post-wait sync are
  // shared code covered below, and the two-queues gate wait itself is covered
  // by CompletionWaitsForPriorPartitionAppend.
  if (GetParam()) {
    SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWAL:Begin");
    return;
  }

  // Phase 2: B (sync) waits on held A (async), then syncs once more.
  std::mutex mu;
  std::condition_variable cv;
  bool first_append = true;
  bool a_in_append = false;
  bool release_a = false;
  bool b_appended = false;
  bool b_took_snapshot = false;
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
  // B's leadership sync starts after its pre-sync snapshot; releasing A
  // earlier could let A complete before that snapshot, which then covers A.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex", [&](void*) {
        std::lock_guard<std::mutex> lock(mu);
        if (!release_a) {
          b_took_snapshot = true;
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
    EXPECT_TRUE(cv.wait_until(lock, deadline,
                              [&] { return b_appended && b_took_snapshot; }));
  }
  EXPECT_EQ(0, sync_wal_count.load());
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
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWAL:Begin");

  ASSERT_OK(b_status);
  EXPECT_TRUE(b_returned.load());
  EXPECT_EQ(1, sync_wal_count.load());
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
  // write still fsyncs all four, and takes all N WAL mutexes.
  ASSERT_OK(Put(cf_for_partition_[0], "k2", "v"));
  ASSERT_OK(Put(cf_for_partition_[0], "k3", "v", sync_wo));
  EXPECT_EQ(1U, count(WAL_SYNC_FSYNCED_PARTITIONS));
  EXPECT_EQ(kPartitions, sum(WAL_SYNC_FSYNCED_PARTITIONS));
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
  // R4 (second half): for sync writes the gate's outcome includes the
  // post-wait sync. B waits on held A, then its post-wait SyncWAL fails; B
  // reports the sync error. Standard path only: here no leadership-time
  // SyncWAL exists, so arming the failure from the start can only hit the
  // post-wait sync. (The two-queues leadership sync is covered by the counts
  // in SyncWriteSyncsAgainOnlyAfterWait; the gate itself is shared code.)
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
  // A sync=false and B's inline leadership sync never reach SyncWAL, so the
  // only SyncWAL in this test is B's post-wait one.
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalImpl:AfterSyncWAL", [&](void* arg) {
        auto* wal_and_status =
            static_cast<std::pair<log::Writer*, IOStatus*>*>(arg);
        *wal_and_status->second = IOStatus::IOError("injected sync error");
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
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWalImpl:AfterSyncWAL");

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
      "DBImpl::PreprocessWrite:TakesSyncMarkMutex", [&](void*) { ++pre_mark; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex",
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
      "DBImpl::PreprocessWrite:TakesSyncMarkMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex");
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
      "DBImpl::PreprocessWrite:TakesSyncMarkMutex", [&](void*) { ++pre_mark; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex",
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
      "DBImpl::PreprocessWrite:TakesSyncMarkMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex");
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
      "DBImpl::PreprocessWrite:TakesSyncMarkMutex", [&](void*) { ++pre_mark; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex",
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
      "DBImpl::PreprocessWrite:TakesSyncMarkMutex");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::WriteGroupToWAL:TakesPartitionSyncMutex");
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
  // A lets B reach its post-wait sync, which fails: W must wake with the
  // background error while M is still held, proving the wake came from
  // B's sync failure and not from M. M then fails too.
  // Standard path only (as in PostWaitSyncErrorFailsSyncWrite): with two
  // write queues the leadership sync itself goes through SyncWAL, so the
  // failure cannot be aimed at the post-wait sync alone.
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
  // No leadership-time SyncWAL exists on this path, so the only SyncWAL is
  // B's post-wait one. Hold it until W installed its gate and parked, so
  // W is provably parked when B's sync fails.
  SyncPoint::GetInstance()->SetCallBack("DBImpl::SyncWAL:Begin", [&](void*) {
    ++sync_wal_begins;
    {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait_until(lock, deadline, [&] { return gate_installs.load() >= 3; });
    }
    const auto park_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (dbfull()->TEST_WalDurabilityWaiters() != 1 &&
           std::chrono::steady_clock::now() < park_deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncWalImpl:AfterSyncWAL", [&](void* arg) {
        auto* wal_and_status =
            static_cast<std::pair<log::Writer*, IOStatus*>*>(arg);
        *wal_and_status->second = IOStatus::IOError("injected sync error");
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError",
      [&](void*) { ++bg_error_wakes; });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ResumeImpl:SyncLastSequenceWithAllocated",
      [&](void*) { ++resume_syncs; });
  SyncPoint::GetInstance()->SetCallBack("DBImpl::ResumeImpl:AfterSyncSeq",
                                        [&](void*) { ++resume_fences; });

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
    w_status = Put(cf_a, "key_w", "value_w");
    w_returned.store(true);
  });
  {
    std::lock_guard<std::mutex> lock(mu);
    release_a = true;
    cv.notify_all();
  }
  // W wakes with B's sync failure while M is still held.
  const auto w_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!w_returned.load() && std::chrono::steady_clock::now() < w_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(w_returned.load());
  EXPECT_FALSE(m_returned.load());
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_m_append = true;
    release_m = true;
    cv.notify_all();
  }
  a_thread.join();
  b_thread.join();
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
  // Exactly one SyncWAL ran: B's post-wait one. One notification per
  // failure: B's post-wait sync (classified inside SyncWalImpl), then M's
  // append.
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
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWAL:Begin");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::SyncWalImpl:AfterSyncWAL");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::ResumeImpl:SyncLastSequenceWithAllocated");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::ResumeImpl:AfterSyncSeq");
}

TEST_P(DBWritePartitionTest, DurabilityWaitErrorBeforeAdvanceFailsWaiter) {
  // Invariant 4, error-first order: W parks behind held A; the background
  // error publishes while A is still held. W must wake promptly with the
  // error rather than waiting for A, with no lost wake and no leaked
  // waiter-count. A, released afterwards, still completes OK: its own
  // range is covered, and coverage never depended on the error.
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
  dbfull()->TEST_SetBGError(IOStatus::IOError("injected race error"),
                            BackgroundErrorReason::kWriteCallback);
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

  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
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
  // compaction then fails at output verification. W must wake with the
  // compaction error while A is still held, proving the wake came from
  // the publication and not from the frontier (nor from any timer: the
  // park has none). CompactFiles is used because CompactRange and Flush
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
  // Armed before the compaction starts, so it cannot finish uninjected.
  {
    std::lock_guard<std::mutex> lock(mu);
    fail_compaction = true;
  }
  port::Thread compact_thread([&] {
    compact_status = db_->CompactFiles(CompactionOptions(), handles_[cf_a],
                                       l0_files, /*output_level=*/1);
  });
  // W wakes with the compaction error while A is still held: the wake
  // came from the error publication, not from the frontier.
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
  compact_thread.join();
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:log_entry");
  SyncPoint::GetInstance()->ClearCallBack("DBImpl::WriteToWAL:AfterAddRecord");
  SyncPoint::GetInstance()->ClearCallBack(
      "CompactionJob::Run():AfterVerifyOutputFiles");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(compact_status);
  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
  ASSERT_OK(a_status);
  EXPECT_EQ(1, compaction_verifies);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(w_status.ToString(), dbfull()->TEST_GetBGError().ToString());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, LeadershipSyncFailureWakesParkedGate) {
  // Q1, leadership-time sync failure: A is held open on partition 0; W
  // appends on partition 1 and parks in its gate; C (sync) then fails
  // exactly its leadership fsync. W must wake with that error while A is
  // still held. Standard path only: with two write queues a sync write
  // takes all N WAL mutexes in PreprocessWrite, so C would block on A's
  // held append before ever reaching its sync. The two-queue leadership
  // sync still funnels through the same SetBGError transition the
  // two-queue append-failure run already covers; only the funnel
  // (WriteStatusCheck) differs, and it is pre-existing code.
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
  bool fail_leadership_sync = false;
  int leadership_syncs = 0;
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
      "DBImpl::WriteGroupToWAL:AfterLeadershipSync", [&](void* arg) {
        std::lock_guard<std::mutex> lock(mu);
        if (fail_leadership_sync) {
          fail_leadership_sync = false;
          ++leadership_syncs;
          *static_cast<IOStatus*>(arg) =
              IOStatus::IOError("injected leadership sync error");
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
    fail_leadership_sync = true;
  }
  WriteOptions sync_options;
  sync_options.sync = true;
  port::Thread c_thread([&] {
    c_status = db_->Put(sync_options, handles_[cf_w], "key_c", "value_c");
  });
  // W wakes with the sync error while A is still held.
  const auto w_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!w_returned.load() && std::chrono::steady_clock::now() < w_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(w_returned.load());
  c_thread.join();
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
      "DBImpl::WriteGroupToWAL:AfterLeadershipSync");
  SyncPoint::GetInstance()->ClearCallBack(
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_NOK(c_status);
  EXPECT_TRUE(c_status.IsIOError());
  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
  ASSERT_OK(a_status);
  EXPECT_EQ(1, leadership_syncs);
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_TRUE(dbfull()->TEST_WalDurabilityBGError());
  EXPECT_EQ(w_status.ToString(), dbfull()->TEST_GetBGError().ToString());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, SuppressedErrorLeavesGateUndisturbed) {
  // Q1: errors that never become the background error wake nothing, stick
  // nothing, and leave a parked gate undisturbed. W parks behind held A;
  // a compaction retryable (soft, never stored) and a listener-suppressed
  // error are published; neither fires the wake nor sets the flag, and W
  // stays parked. Repeating the suppressed error without the listener
  // then lands it, proving the listener did the suppressing.
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
      "DBImpl::SignalWalDurabilityWaitersOnBGError");

  ASSERT_OK(a_status);
  ASSERT_NOK(w_status);
  EXPECT_TRUE(w_status.IsIOError());
  EXPECT_EQ(1, bg_error_wakes.load());
  EXPECT_EQ(0U, dbfull()->TEST_WalDurabilityWaiters());
  dbfull()->Resume().PermitUncheckedError();
}

TEST_P(DBWritePartitionTest, CheckThenParkRaceWakesWaiter) {
  // Q1, the check-then-park window: an error published between the
  // waiter's predicate check and its park is never missed. W parks behind
  // held A; the BeforePark hook holds W after its check with the park
  // mutex held; the error is published while W is held there (its signal
  // blocks on the mutex W holds); W is then released and must fail with
  // the error instead of sleeping past it. There is no timer to save a
  // missed wake, so a lost signal would hang this test.
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
  bool w_at_park = false;
  bool release_w = false;
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
  // W has checked (flag clear then) and is held before its park with the
  // park mutex held. Publish the error from another thread: its signal
  // blocks on that mutex until W is released.
  port::Thread setter_thread([&] {
    dbfull()->TEST_SetBGError(IOStatus::IOError("park-window error"),
                              BackgroundErrorReason::kWriteCallback);
  });
  // Release W only once the flag is published: the setter's signal is
  // then all that is left to wake W, and it must still reach W. The bound
  // only turns a flag that never appears into a reported failure; W is
  // released either way so the test can finish.
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
  setter_thread.join();
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

// G-07 opener gap: read-only and secondary opens apply the same N > 1 option
// checks as a read-write open, with the same status and message. Every
// DBOptions check runs for them, including the write-only ones. Neuter: drop
// the check from the read-only open; its rows open.
TEST_P(DBWritePartitionTest, ReadOnlyAndSecondaryApplyPartitionChecks) {
  if (GetParam()) {
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

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
