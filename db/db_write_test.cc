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

#include "db/db_test_util.h"
#include "db/write_batch_internal.h"
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
  std::atomic<bool> blocked_append_done{false};
  std::atomic<bool> switched_after_append{false};
  std::atomic<bool> switched{false};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* arg) {
        if (!static_cast<Slice*>(arg)->ends_with("blocked")) {
          return;
        }
        std::unique_lock<std::mutex> lock(mu);
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

  port::Thread blocked_writer(
      [&] { ASSERT_OK(Put(blocked_cf, "key", "blocked")); });
  // Fills the switching partition's memtable while the blocked append is
  // still open. A concurrent insert schedules the flush on the memtable's
  // next insert, and the write after that switches.
  ASSERT_OK(Put(switching_cf, "fill", std::string(128 << 10, 'x')));
  for (int i = 0; i < 3 && !switched.load(); ++i) {
    ASSERT_OK(Put(switching_cf, "switch", "value"));
  }
  blocked_writer.join();

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

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
