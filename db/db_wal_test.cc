//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include <algorithm>
#include <map>
#include <mutex>
#include <numeric>

#include "db/db_test_util.h"
#include "db/db_with_timestamp_test_util.h"
#include "db/log_reader.h"
#include "db/log_writer.h"
#include "options/options_helper.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/file_system.h"
#include "rocksdb/utilities/checkpoint.h"
#include "rocksdb/utilities/ldb_cmd.h"
#include "rocksdb/wal_filter.h"
#include "test_util/sync_point.h"
#include "util/crc32c.h"
#include "util/defer.h"
#include "util/udt_util.h"
#include "utilities/fault_injection_env.h"
#include "utilities/fault_injection_fs.h"

#if defined(OS_LINUX)
#include <sys/statfs.h>
#endif

#if defined(OS_LINUX) && !defined(BTRFS_SUPER_MAGIC)
#define BTRFS_SUPER_MAGIC 0x9123683E
#endif

#if defined(OS_LINUX) && !defined(TMPFS_MAGIC)
#define TMPFS_MAGIC 0x01021994
#endif

#if defined(OS_LINUX) && !defined(OVERLAYFS_SUPER_MAGIC)
#define OVERLAYFS_SUPER_MAGIC 0x794c7630
#endif

#if defined(OS_LINUX) && !defined(ZFS_SUPER_MAGIC)
#define ZFS_SUPER_MAGIC 0x2fc12fc1
#endif

namespace ROCKSDB_NAMESPACE {
class DBWALTestBase : public DBTestBase {
 protected:
  explicit DBWALTestBase(const std::string& dir_name)
      : DBTestBase(dir_name, /*env_do_fsync=*/true) {}

  std::vector<uint64_t> ListWalNumbers() {
    std::vector<uint64_t> wal_numbers;
    std::vector<std::string> files;
    EXPECT_OK(env_->GetChildren(dbname_, &files));
    for (const auto& file : files) {
      uint64_t number = 0;
      FileType type = kWalFile;
      if (ParseFileName(file, &number, &type) && type == kWalFile) {
        wal_numbers.push_back(number);
      }
    }
    std::sort(wal_numbers.begin(), wal_numbers.end());
    return wal_numbers;
  }

  void CreateEmptyWal(uint64_t wal_number) {
    std::unique_ptr<WritableFile> file;
    ASSERT_OK(env_->NewWritableFile(LogFileName(dbname_, wal_number), &file,
                                    EnvOptions()));
    ASSERT_OK(file->Close());
  }

  void CreateWalWithContents(uint64_t wal_number, const Slice& contents) {
    std::unique_ptr<WritableFile> file;
    ASSERT_OK(env_->NewWritableFile(LogFileName(dbname_, wal_number), &file,
                                    EnvOptions()));
    ASSERT_OK(file->Append(contents));
    ASSERT_OK(file->Close());
  }

#if defined(ROCKSDB_PLATFORM_POSIX)
 public:
#if defined(ROCKSDB_FALLOCATE_PRESENT)
  bool IsFallocateSupported() {
    // Test fallocate support of running file system.
    // Skip this test if fallocate is not supported.
    std::string fname_test_fallocate = dbname_ + "/preallocate_testfile";
    int fd = -1;
    do {
      fd = open(fname_test_fallocate.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    } while (fd < 0 && errno == EINTR);
    assert(fd > 0);
    int alloc_status = fallocate(fd, 0, 0, 1);
    int err_number = errno;
    close(fd);
    assert(env_->DeleteFile(fname_test_fallocate) == Status::OK());
    if (err_number == ENOSYS || err_number == EOPNOTSUPP) {
      fprintf(stderr, "Skipped preallocated space check: %s\n",
              errnoStr(err_number).c_str());
      return false;
    }
    assert(alloc_status == 0);
    return true;
  }
#endif  // ROCKSDB_FALLOCATE_PRESENT

  uint64_t GetAllocatedFileSize(std::string file_name) {
    struct stat sbuf;
    int err = stat(file_name.c_str(), &sbuf);
    assert(err == 0);
    return sbuf.st_blocks * 512;
  }

#if defined(ROCKSDB_FALLOCATE_PRESENT)
  bool ShouldSkipAllocationCheck(const std::string& file_name) {
    (void)file_name;
#if defined(OS_LINUX)
    struct statfs fs_stat;
    if (statfs(file_name.c_str(), &fs_stat) == 0) {
      if (fs_stat.f_type ==
          static_cast<decltype(fs_stat.f_type)>(BTRFS_SUPER_MAGIC)) {
        return true;
      } else if (fs_stat.f_type ==
                 static_cast<decltype(fs_stat.f_type)>(ZFS_SUPER_MAGIC)) {
        return true;
      } else if (fs_stat.f_type ==
                 static_cast<decltype(fs_stat.f_type)>(TMPFS_MAGIC)) {
        return true;
      } else if (fs_stat.f_type ==
                 static_cast<decltype(fs_stat.f_type)>(OVERLAYFS_SUPER_MAGIC)) {
        return true;
      }
    }
#endif
    return false;
  }
#endif  // ROCKSDB_FALLOCATE_PRESENT
#endif  // ROCKSDB_PLATFORM_POSIX
};

class DBWALTest : public DBWALTestBase {
 public:
  DBWALTest() : DBWALTestBase("/db_wal_test") {}
};

// A SpecialEnv enriched to give more insight about deleted files
class EnrichedSpecialEnv : public SpecialEnv {
 public:
  explicit EnrichedSpecialEnv(Env* base) : SpecialEnv(base) {}
  Status NewSequentialFile(const std::string& f,
                           std::unique_ptr<SequentialFile>* r,
                           const EnvOptions& soptions) override {
    InstrumentedMutexLock l(&env_mutex_);
    if (f == skipped_wal) {
      deleted_wal_reopened = true;
      if (IsWAL(f) && largest_deleted_wal.size() != 0 &&
          f.compare(largest_deleted_wal) <= 0) {
        gap_in_wals = true;
      }
    }
    return SpecialEnv::NewSequentialFile(f, r, soptions);
  }
  Status DeleteFile(const std::string& fname) override {
    if (IsWAL(fname)) {
      deleted_wal_cnt++;
      InstrumentedMutexLock l(&env_mutex_);
      // If this is the first WAL, remember its name and skip deleting it. We
      // remember its name partly because the application might attempt to
      // delete the file again.
      if (skipped_wal.size() != 0 && skipped_wal != fname) {
        if (largest_deleted_wal.size() == 0 ||
            largest_deleted_wal.compare(fname) < 0) {
          largest_deleted_wal = fname;
        }
      } else {
        skipped_wal = fname;
        return Status::OK();
      }
    }
    return SpecialEnv::DeleteFile(fname);
  }
  bool IsWAL(const std::string& fname) {
    // printf("iswal %s\n", fname.c_str());
    return fname.compare(fname.size() - 3, 3, "log") == 0;
  }

  InstrumentedMutex env_mutex_;
  // the wal whose actual delete was skipped by the env
  std::string skipped_wal;
  // the largest WAL that was requested to be deleted
  std::string largest_deleted_wal;
  // number of WALs that were successfully deleted
  std::atomic<size_t> deleted_wal_cnt = {0};
  // the WAL whose delete from fs was skipped is reopened during recovery
  std::atomic<bool> deleted_wal_reopened = {false};
  // whether a gap in the WALs was detected during recovery
  std::atomic<bool> gap_in_wals = {false};
};

class DBWALTestWithEnrichedEnv : public DBTestBase {
 public:
  DBWALTestWithEnrichedEnv()
      : DBTestBase("db_wal_test", /*env_do_fsync=*/true) {
    enriched_env_ = new EnrichedSpecialEnv(env_->target());
    auto options = CurrentOptions();
    options.env = enriched_env_;
    options.allow_2pc = true;
    Reopen(options);
    delete env_;
    // to be deleted by the parent class
    env_ = enriched_env_;
  }

 protected:
  EnrichedSpecialEnv* enriched_env_;
};

// Test that the recovery would successfully avoid the gaps between the logs.
// One known scenario that could cause this is that the application issue the
// WAL deletion out of order. For the sake of simplicity in the test, here we
// create the gap by manipulating the env to skip deletion of the first WAL but
// not the ones after it.
TEST_F(DBWALTestWithEnrichedEnv, SkipDeletedWALs) {
  auto options = last_options_;
  // To cause frequent WAL deletion
  options.write_buffer_size = 128;
  Reopen(options);

  SyncPoint::GetInstance()->LoadDependency(
      {{"DBImpl::PurgeObsoleteFiles:End",
        "DBWALTestWithEnrichedEnv.SkipDeletedWALs:AfterFlush"}});
  SyncPoint::GetInstance()->EnableProcessing();

  WriteOptions writeOpt = WriteOptions();
  for (int i = 0; i < 128 * 5; i++) {
    ASSERT_OK(dbfull()->Put(writeOpt, "foo", "v1"));
  }
  FlushOptions fo;
  fo.wait = true;
  ASSERT_OK(db_->Flush(fo));

  TEST_SYNC_POINT("DBWALTestWithEnrichedEnv.SkipDeletedWALs:AfterFlush");

  // some wals are deleted
  ASSERT_NE(0, enriched_env_->deleted_wal_cnt);
  // but not the first one
  ASSERT_NE(0, enriched_env_->skipped_wal.size());

  // Test that the WAL that was not deleted will be skipped during recovery
  options = last_options_;
  Reopen(options);
  ASSERT_FALSE(enriched_env_->deleted_wal_reopened);
  ASSERT_FALSE(enriched_env_->gap_in_wals);

  SyncPoint::GetInstance()->DisableProcessing();
}

TEST_F(DBWALTest, WAL) {
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());
    WriteOptions writeOpt = WriteOptions();
    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "foo", "v1"));
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v1"));

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    ASSERT_EQ("v1", Get(1, "foo"));
    ASSERT_EQ("v1", Get(1, "bar"));

    writeOpt.disableWAL = false;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v2"));
    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "foo", "v2"));

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    // Both value's should be present.
    ASSERT_EQ("v2", Get(1, "bar"));
    ASSERT_EQ("v2", Get(1, "foo"));

    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v3"));
    writeOpt.disableWAL = false;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "foo", "v3"));

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    // again both values should be present.
    ASSERT_EQ("v3", Get(1, "foo"));
    ASSERT_EQ("v3", Get(1, "bar"));
  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, RollLog) {
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());
    ASSERT_OK(Put(1, "foo", "v1"));
    ASSERT_OK(Put(1, "baz", "v5"));

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    for (int i = 0; i < 10; i++) {
      ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    }
    ASSERT_OK(Put(1, "foo", "v4"));
    for (int i = 0; i < 10; i++) {
      ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    }
  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, SyncWALNotBlockWrite) {
  Options options = CurrentOptions();
  options.max_write_buffer_number = 4;
  DestroyAndReopen(options);

  ASSERT_OK(Put("foo1", "bar1"));
  ASSERT_OK(Put("foo5", "bar5"));

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency({
      {"WritableFileWriter::SyncWithoutFlush:1",
       "DBWALTest::SyncWALNotBlockWrite:1"},
      {"DBWALTest::SyncWALNotBlockWrite:2",
       "WritableFileWriter::SyncWithoutFlush:2"},
  });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  ROCKSDB_NAMESPACE::port::Thread thread([&]() { ASSERT_OK(db_->SyncWAL()); });

  TEST_SYNC_POINT("DBWALTest::SyncWALNotBlockWrite:1");
  ASSERT_OK(Put("foo2", "bar2"));
  ASSERT_OK(Put("foo3", "bar3"));
  FlushOptions fo;
  fo.wait = false;
  ASSERT_OK(db_->Flush(fo));
  ASSERT_OK(Put("foo4", "bar4"));

  TEST_SYNC_POINT("DBWALTest::SyncWALNotBlockWrite:2");

  thread.join();

  ASSERT_EQ(Get("foo1"), "bar1");
  ASSERT_EQ(Get("foo2"), "bar2");
  ASSERT_EQ(Get("foo3"), "bar3");
  ASSERT_EQ(Get("foo4"), "bar4");
  ASSERT_EQ(Get("foo5"), "bar5");
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

TEST_F(DBWALTest, SyncWALNotWaitWrite) {
  ASSERT_OK(Put("foo1", "bar1"));
  ASSERT_OK(Put("foo3", "bar3"));

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency({
      {"SpecialEnv::SpecialWalFile::Append:1",
       "DBWALTest::SyncWALNotWaitWrite:1"},
      {"DBWALTest::SyncWALNotWaitWrite:2",
       "SpecialEnv::SpecialWalFile::Append:2"},
  });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();

  ROCKSDB_NAMESPACE::port::Thread thread(
      [&]() { ASSERT_OK(Put("foo2", "bar2")); });
  // Moving this to SyncWAL before the actual fsync
  // TEST_SYNC_POINT("DBWALTest::SyncWALNotWaitWrite:1");
  ASSERT_OK(db_->SyncWAL());
  // Moving this to SyncWAL after actual fsync
  // TEST_SYNC_POINT("DBWALTest::SyncWALNotWaitWrite:2");

  thread.join();

  ASSERT_EQ(Get("foo1"), "bar1");
  ASSERT_EQ(Get("foo2"), "bar2");
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

// A flush's SyncClosedWals that marks nothing, because partition 0 holds
// only the current WAL, must leave the current WAL's mark to the sync that
// set it: SyncWAL or a sync=true write's leadership sync (param 0), which
// then succeeds or fails (param 1). Finishing that mark from SyncClosedWals
// unpins the WAL under the owner's fsync and trips the owner's
// FinishSync() assertion.
class DBWALSyncMarkOwnershipTest
    : public DBWALTest,
      public testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(DBWALSyncMarkOwnershipTest, SyncClosedWalsLeavesOtherSyncsMark) {
  const bool leader = std::get<0>(GetParam());
  const bool fail = std::get<1>(GetParam());
  Options options = CurrentOptions();
  options.env = env_;
  CreateAndReopenWithCF({"pikachu"}, options);

  // Flushing cf 0 rotates the WAL, and its SyncClosedWals syncs, closes and
  // drops the old one, leaving the empty current WAL alone in partition 0
  // while cf 1 still has unflushed data. Flushing cf 1 then switches no WAL.
  ASSERT_OK(Put(1, "k1", "v1"));
  ASSERT_OK(Put(0, "k0", "v0"));
  ASSERT_OK(Flush(0));
  const uint64_t cur = dbfull()->TEST_GetCurrentLogNumber();
  using Marks = std::vector<std::pair<uint64_t, bool>>;
  ASSERT_EQ((Marks{{cur, false}}), dbfull()->TEST_WalSyncMarks(0));

  // The owner parks holding its mark, right after its claim (SyncWAL) or
  // after its WAL append and before its leadership fsync (sync=true write),
  // until the flush has finished; the flush's SyncClosedWals starts only
  // once the owner is parked. A failing owner fails right after the park.
  SyncPoint::GetInstance()->LoadDependency({
      {"DBWALSyncMarkOwnershipTest:OwnerParked",
       "DBImpl::SyncClosedWals:Start"},
      {"DBWALSyncMarkOwnershipTest:FlushChecked",
       "DBWALSyncMarkOwnershipTest:ReleaseOwner"},
  });
  std::atomic<bool> parked{false};
  auto park = [&]() {
    if (parked.exchange(true)) {
      return false;
    }
    TEST_SYNC_POINT("DBWALSyncMarkOwnershipTest:OwnerParked");
    TEST_SYNC_POINT("DBWALSyncMarkOwnershipTest:ReleaseOwner");
    return true;
  };
  if (leader) {
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:AfterAddRecord", [&](void* arg) {
          if (park() && fail) {
            *static_cast<IOStatus*>(arg) = IOStatus::IOError("injected");
          }
        });
  } else {
    SyncPoint::GetInstance()->SetCallBack("DBWALTest::SyncWALNotWaitWrite:1",
                                          [&](void*) { park(); });
    if (fail) {
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::SyncWalImpl:AfterSyncWAL", [&](void* arg) {
            auto* wal_and_status =
                static_cast<std::pair<log::Writer*, IOStatus*>*>(arg);
            if (wal_and_status->first->get_log_number() == cur) {
              *wal_and_status->second = IOStatus::IOError("injected");
            }
          });
    }
  }
  SyncPoint::GetInstance()->EnableProcessing();

  FlushOptions no_wait;
  no_wait.wait = false;
  ASSERT_OK(db_->Flush(no_wait, handles_[1]));
  Status owner_status;
  port::Thread owner([&]() {
    if (leader) {
      WriteOptions sync_wo;
      sync_wo.sync = true;
      owner_status = Put(0, "k2", "v2", sync_wo);
    } else {
      owner_status = db_->SyncWAL();
    }
  });
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[1]));
  ASSERT_EQ(cur, dbfull()->TEST_GetCurrentLogNumber());
  EXPECT_EQ((Marks{{cur, true}}), dbfull()->TEST_WalSyncMarks(0));
  TEST_SYNC_POINT("DBWALSyncMarkOwnershipTest:FlushChecked");
  owner.join();

  EXPECT_TRUE(parked.load());
  if (fail) {
    EXPECT_TRUE(owner_status.IsIOError()) << owner_status.ToString();
  } else {
    EXPECT_OK(owner_status);
  }
  // The owner finished or cancelled its own mark.
  EXPECT_EQ((Marks{{cur, false}}), dbfull()->TEST_WalSyncMarks(0));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
}

INSTANTIATE_TEST_CASE_P(DBWALSyncMarkOwnershipTest, DBWALSyncMarkOwnershipTest,
                        testing::Combine(testing::Bool(), testing::Bool()));

TEST_F(DBWALTest, SyncClosedWalsEmptiedByWaitLeavesLaterMark) {
  // A flush's SyncClosedWals finds a closed WAL ahead of the current WAL and
  // waits for the SyncWAL syncing it. That SyncWAL reclaims it, so after the
  // wait the partition holds only the current WAL and SyncClosedWals marks
  // nothing. A second SyncWAL then marks the current WAL, and SyncClosedWals
  // must leave that mark alone when it finishes.
  Options options = CurrentOptions();
  options.env = env_;
  // Lets a sync reclaim a fully synced inactive WAL.
  options.background_close_inactive_wals = true;
  CreateAndReopenWithCF({"pikachu"}, options);
  ASSERT_OK(Put(1, "k1", "v1"));
  ASSERT_OK(Put(0, "k0", "v0"));
  const uint64_t old_wal = dbfull()->TEST_GetCurrentLogNumber();

  SyncPoint::GetInstance()->LoadDependency({
      {"DBImpl::ClaimWalPartitionForSyncWalImpl:Wait",
       "SyncClosedWalsEmptiedByWait:ClosedWalsWaiting"},
      {"SyncClosedWalsEmptiedByWait:ReleaseFirst",
       "SyncClosedWalsEmptiedByWait:FirstReleased"},
      {"SyncClosedWalsEmptiedByWait:ClosedWalsParked",
       "SyncClosedWalsEmptiedByWait:StartSecond"},
      {"SyncClosedWalsEmptiedByWait:SecondParked",
       "SyncClosedWalsEmptiedByWait:ReleaseClosedWals"},
      {"SyncClosedWalsEmptiedByWait:FlushChecked",
       "SyncClosedWalsEmptiedByWait:SecondReleased"},
  });
  std::atomic<int> syncwal_claims{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBWALTest::SyncWALNotWaitWrite:1", [&](void*) {
        if (syncwal_claims.fetch_add(1) == 0) {
          TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:FirstReleased");
        } else {
          TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:SecondParked");
          TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:SecondReleased");
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::SyncClosedWals:BeforeReLock", [&](void*) {
        TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:ClosedWalsParked");
        TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:ReleaseClosedWals");
      });
  SyncPoint::GetInstance()->EnableProcessing();

  // The first SyncWAL marks the old WAL and parks; flushing cf 0 then
  // rotates, and the flush's SyncClosedWals waits on the marked old WAL.
  port::Thread first([&]() { ASSERT_OK(db_->SyncWAL()); });
  FlushOptions no_wait;
  no_wait.wait = false;
  ASSERT_OK(db_->Flush(no_wait, handles_[0]));
  const uint64_t cur = dbfull()->TEST_GetCurrentLogNumber();
  ASSERT_GT(cur, old_wal);
  TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:ClosedWalsWaiting");
  TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:ReleaseFirst");
  first.join();

  // SyncClosedWals woke to the current WAL alone and parked before its
  // finish; the second SyncWAL marks the current WAL and parks.
  TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:StartSecond");
  using Marks = std::vector<std::pair<uint64_t, bool>>;
  ASSERT_EQ((Marks{{cur, false}}), dbfull()->TEST_WalSyncMarks(0));
  port::Thread second([&]() { ASSERT_OK(db_->SyncWAL()); });
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[0]));
  EXPECT_EQ((Marks{{cur, true}}), dbfull()->TEST_WalSyncMarks(0));
  TEST_SYNC_POINT("SyncClosedWalsEmptiedByWait:FlushChecked");
  second.join();

  EXPECT_EQ((Marks{{cur, false}}), dbfull()->TEST_WalSyncMarks(0));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST_F(DBWALTest, Recover) {
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());
    ASSERT_OK(Put(1, "foo", "v1"));
    ASSERT_OK(Put(1, "baz", "v5"));

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    ASSERT_EQ("v1", Get(1, "foo"));
    ASSERT_EQ("v1", Get(1, "foo"));
    ASSERT_EQ("v5", Get(1, "baz"));
    ASSERT_OK(Put(1, "bar", "v2"));
    ASSERT_OK(Put(1, "foo", "v3"));

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    ASSERT_EQ("v3", Get(1, "foo"));
    ASSERT_OK(Put(1, "foo", "v4"));
    ASSERT_EQ("v4", Get(1, "foo"));
    ASSERT_EQ("v2", Get(1, "bar"));
    ASSERT_EQ("v5", Get(1, "baz"));
  } while (ChangeWalOptions());
}

class DBWALTestWithTimestamp
    : public DBBasicTestWithTimestampBase,
      public testing::WithParamInterface<test::UserDefinedTimestampTestMode> {
 public:
  DBWALTestWithTimestamp()
      : DBBasicTestWithTimestampBase("db_wal_test_with_timestamp") {}

  Status CreateAndReopenWithTs(const std::vector<std::string>& cfs,
                               const Options& ts_options, bool persist_udt,
                               bool avoid_flush_during_recovery = false) {
    Options default_options = CurrentOptions();
    default_options.allow_concurrent_memtable_write =
        persist_udt ? true : false;
    DestroyAndReopen(default_options);
    CreateColumnFamilies(cfs, ts_options);
    return ReopenColumnFamiliesWithTs(cfs, ts_options, persist_udt,
                                      avoid_flush_during_recovery);
  }

  Status ReopenColumnFamiliesWithTs(const std::vector<std::string>& cfs,
                                    Options ts_options, bool persist_udt,
                                    bool avoid_flush_during_recovery = false) {
    Options default_options = CurrentOptions();
    default_options.create_if_missing = false;
    default_options.allow_concurrent_memtable_write =
        persist_udt ? true : false;
    default_options.avoid_flush_during_recovery = avoid_flush_during_recovery;
    ts_options.create_if_missing = false;

    std::vector<Options> cf_options(cfs.size(), ts_options);
    std::vector<std::string> cfs_plus_default = cfs;
    cfs_plus_default.insert(cfs_plus_default.begin(), kDefaultColumnFamilyName);
    cf_options.insert(cf_options.begin(), default_options);
    Close();
    return TryReopenWithColumnFamilies(cfs_plus_default, cf_options);
  }

  Status Put(uint32_t cf, const Slice& key, const Slice& ts,
             const Slice& value) {
    WriteOptions write_opts;
    return db_->Put(write_opts, handles_[cf], key, ts, value);
  }

  void CheckGet(const ReadOptions& read_opts, uint32_t cf, const Slice& key,
                const std::string& expected_value,
                const std::string& expected_ts) {
    std::string actual_value;
    std::string actual_ts;
    ASSERT_OK(
        db_->Get(read_opts, handles_[cf], key, &actual_value, &actual_ts));
    ASSERT_EQ(expected_value, actual_value);
    ASSERT_EQ(expected_ts, actual_ts);
  }
};

TEST_P(DBWALTestWithTimestamp, RecoverAndNoFlush) {
  // Set up the option that enables user defined timestmp size.
  std::string ts1;
  PutFixed64(&ts1, 1);
  Options ts_options;
  ts_options.create_if_missing = true;
  ts_options.comparator = test::BytewiseComparatorWithU64TsWrapper();
  // Test that user-defined timestamps are recovered from WAL regardless of
  // the value of this flag because UDTs are saved in WAL nonetheless.
  // We however need to explicitly disable flush during recovery by setting
  // `avoid_flush_during_recovery=true` so that we can avoid timestamps getting
  // stripped when the `persist_user_defined_timestamps` flag is false, so that
  // all written timestamps are available for testing user-defined time travel
  // read.
  bool persist_udt = test::ShouldPersistUDT(GetParam());
  ts_options.persist_user_defined_timestamps = persist_udt;
  bool avoid_flush_during_recovery = true;

  std::string full_history_ts_low;
  ReadOptions read_opts;
  do {
    Slice ts_slice = ts1;
    read_opts.timestamp = &ts_slice;
    ASSERT_OK(CreateAndReopenWithTs({"pikachu"}, ts_options, persist_udt,
                                    avoid_flush_during_recovery));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"), 0U);
    ASSERT_OK(Put(1, "foo", ts1, "v1"));
    ASSERT_OK(Put(1, "baz", ts1, "v5"));

    ASSERT_OK(ReopenColumnFamiliesWithTs({"pikachu"}, ts_options, persist_udt,
                                         avoid_flush_during_recovery));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"), 0U);
    // Do a timestamped read with ts1 after second reopen.
    CheckGet(read_opts, 1, "foo", "v1", ts1);
    CheckGet(read_opts, 1, "baz", "v5", ts1);

    // Write more value versions for key "foo" and "bar" before and after second
    // reopen.
    std::string ts2;
    PutFixed64(&ts2, 2);
    ASSERT_OK(Put(1, "bar", ts2, "v2"));
    ASSERT_OK(Put(1, "foo", ts2, "v3"));

    ASSERT_OK(ReopenColumnFamiliesWithTs({"pikachu"}, ts_options, persist_udt,
                                         avoid_flush_during_recovery));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"), 0U);
    std::string ts3;
    PutFixed64(&ts3, 3);
    ASSERT_OK(Put(1, "foo", ts3, "v4"));

    // All the key value pairs available for read:
    // "foo" -> [(ts1, "v1"), (ts2, "v3"), (ts3, "v4")]
    // "bar" -> [(ts2, "v2")]
    // "baz" -> [(ts1, "v5")]
    // Do a timestamped read with ts1 after third reopen.
    // read_opts.timestamp is set to ts1 for below reads
    CheckGet(read_opts, 1, "foo", "v1", ts1);
    std::string value;
    ASSERT_TRUE(db_->Get(read_opts, handles_[1], "bar", &value).IsNotFound());
    CheckGet(read_opts, 1, "baz", "v5", ts1);

    // Do a timestamped read with ts2 after third reopen.
    ts_slice = ts2;
    // read_opts.timestamp is set to ts2 for below reads.
    CheckGet(read_opts, 1, "foo", "v3", ts2);
    CheckGet(read_opts, 1, "bar", "v2", ts2);
    CheckGet(read_opts, 1, "baz", "v5", ts1);

    // Do a timestamped read with ts3 after third reopen.
    ts_slice = ts3;
    // read_opts.timestamp is set to ts3 for below reads.
    CheckGet(read_opts, 1, "foo", "v4", ts3);
    CheckGet(read_opts, 1, "bar", "v2", ts2);
    CheckGet(read_opts, 1, "baz", "v5", ts1);
    ASSERT_OK(db_->GetFullHistoryTsLow(handles_[1], &full_history_ts_low));
    ASSERT_TRUE(full_history_ts_low.empty());
  } while (ChangeWalOptions());
}

TEST_P(DBWALTestWithTimestamp, RecoverAndFlush) {
  // Set up the option that enables user defined timestamp size.
  std::string min_ts;
  std::string write_ts;
  PutFixed64(&min_ts, 0);
  PutFixed64(&write_ts, 1);
  Options ts_options;
  ts_options.create_if_missing = true;
  ts_options.comparator = test::BytewiseComparatorWithU64TsWrapper();
  bool persist_udt = test::ShouldPersistUDT(GetParam());
  ts_options.persist_user_defined_timestamps = persist_udt;

  std::string smallest_ukey_without_ts = "baz";
  std::string largest_ukey_without_ts = "foo";

  ASSERT_OK(CreateAndReopenWithTs({"pikachu"}, ts_options, persist_udt));
  // No flush, no sst files, because of no data.
  ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"), 0U);
  ASSERT_OK(Put(1, largest_ukey_without_ts, write_ts, "v1"));
  ASSERT_OK(Put(1, smallest_ukey_without_ts, write_ts, "v5"));

  ASSERT_OK(ReopenColumnFamiliesWithTs({"pikachu"}, ts_options, persist_udt));
  // Memtable recovered from WAL flushed because `avoid_flush_during_recovery`
  // defaults to false, created one L0 file.
  ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"), 1U);

  std::vector<std::vector<FileMetaData>> level_to_files;
  dbfull()->TEST_GetFilesMetaData(handles_[1], &level_to_files);
  std::string full_history_ts_low;
  ASSERT_OK(db_->GetFullHistoryTsLow(handles_[1], &full_history_ts_low));
  ASSERT_GT(level_to_files.size(), 1);
  // L0 only has one SST file.
  ASSERT_EQ(level_to_files[0].size(), 1);
  auto meta = level_to_files[0][0];
  if (persist_udt) {
    ASSERT_EQ(smallest_ukey_without_ts + write_ts, meta.smallest.user_key());
    ASSERT_EQ(largest_ukey_without_ts + write_ts, meta.largest.user_key());
    ASSERT_TRUE(full_history_ts_low.empty());
  } else {
    ASSERT_EQ(smallest_ukey_without_ts + min_ts, meta.smallest.user_key());
    ASSERT_EQ(largest_ukey_without_ts + min_ts, meta.largest.user_key());
    std::string effective_cutoff;
    Slice write_ts_slice = write_ts;
    GetFullHistoryTsLowFromU64CutoffTs(&write_ts_slice, &effective_cutoff);
    ASSERT_EQ(effective_cutoff, full_history_ts_low);
  }
}

// Param 0: test mode for the user-defined timestamp feature
INSTANTIATE_TEST_CASE_P(
    P, DBWALTestWithTimestamp,
    ::testing::Values(
        test::UserDefinedTimestampTestMode::kStripUserDefinedTimestamp,
        test::UserDefinedTimestampTestMode::kNormal));

TEST_F(DBWALTestWithTimestamp, EnableDisableUDT) {
  Options options;
  options.create_if_missing = true;
  options.comparator = BytewiseComparator();
  bool avoid_flush_during_recovery = true;
  ASSERT_OK(CreateAndReopenWithTs({"pikachu"}, options, true /* persist_udt */,
                                  avoid_flush_during_recovery));

  ASSERT_OK(db_->Put(WriteOptions(), handles_[1], "foo", "v1"));
  ASSERT_OK(db_->Put(WriteOptions(), handles_[1], "baz", "v5"));

  options.comparator = test::BytewiseComparatorWithU64TsWrapper();
  options.persist_user_defined_timestamps = false;
  // Test handle timestamp size inconsistency in WAL when enabling user-defined
  // timestamps.
  ASSERT_OK(ReopenColumnFamiliesWithTs({"pikachu"}, options,
                                       false /* persist_udt */,
                                       avoid_flush_during_recovery));

  std::string ts;
  PutFixed64(&ts, 0);
  Slice ts_slice = ts;
  ReadOptions read_opts;
  read_opts.timestamp = &ts_slice;
  // Pre-existing entries are treated as if they have the min timestamp.
  CheckGet(read_opts, 1, "foo", "v1", ts);
  CheckGet(read_opts, 1, "baz", "v5", ts);
  ts.clear();
  PutFixed64(&ts, 1);
  ASSERT_OK(db_->Put(WriteOptions(), handles_[1], "foo", ts, "v2"));
  ASSERT_OK(db_->Put(WriteOptions(), handles_[1], "baz", ts, "v6"));
  CheckGet(read_opts, 1, "foo", "v2", ts);
  CheckGet(read_opts, 1, "baz", "v6", ts);

  options.comparator = BytewiseComparator();
  // Open the column family again with the UDT feature disabled. Test handle
  // timestamp size inconsistency in WAL when disabling user-defined timestamps
  ASSERT_OK(ReopenColumnFamiliesWithTs({"pikachu"}, options,
                                       true /* persist_udt */,
                                       avoid_flush_during_recovery));
  ASSERT_EQ("v2", Get(1, "foo"));
  ASSERT_EQ("v6", Get(1, "baz"));
}

TEST_F(DBWALTest, RecoverWithTableHandle) {
  do {
    Options options = CurrentOptions();
    options.create_if_missing = true;
    options.disable_auto_compactions = true;
    options.avoid_flush_during_recovery = false;
    DestroyAndReopen(options);
    CreateAndReopenWithCF({"pikachu"}, options);

    ASSERT_OK(Put(1, "foo", "v1"));
    ASSERT_OK(Put(1, "bar", "v2"));
    ASSERT_OK(Flush(1));
    ASSERT_OK(Put(1, "foo", "v3"));
    ASSERT_OK(Put(1, "bar", "v4"));
    ASSERT_OK(Flush(1));
    ASSERT_OK(Put(1, "big", std::string(100, 'a')));

    options = CurrentOptions();
    const int kSmallMaxOpenFiles = 13;
    if (option_config_ == kDBLogDir) {
      // Use this option to check not preloading files
      // Set the max open files to be small enough so no preload will
      // happen.
      options.max_open_files = kSmallMaxOpenFiles;
      // RocksDB sanitize max open files to at least 20. Modify it back.
      ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
          "SanitizeOptions::AfterChangeMaxOpenFiles", [&](void* arg) {
            int* max_open_files = static_cast<int*>(arg);
            *max_open_files = kSmallMaxOpenFiles;
          });

    } else if (option_config_ == kWalDirAndMmapReads) {
      // Use this option to check always loading all files.
      options.max_open_files = 100;
    } else {
      options.max_open_files = -1;
    }
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();

    std::vector<std::vector<FileMetaData>> files;
    dbfull()->TEST_GetFilesMetaData(handles_[1], &files);
    size_t total_files = 0;
    for (const auto& level : files) {
      total_files += level.size();
    }
    ASSERT_EQ(total_files, 3);
    for (const auto& level : files) {
      for (const auto& file : level) {
        if (options.max_open_files == kSmallMaxOpenFiles) {
          ASSERT_TRUE(file.fd.pinned_reader.Get() == nullptr);
        } else {
          ASSERT_TRUE(file.fd.pinned_reader.Get() != nullptr);
        }
      }
    }
  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, RecoverWithBlob) {
  // Write a value that's below the prospective size limit for blobs and another
  // one that's above. Note that blob files are not actually enabled at this
  // point.
  constexpr uint64_t min_blob_size = 10;

  constexpr char short_value[] = "short";
  static_assert(sizeof(short_value) - 1 < min_blob_size,
                "short_value too long");

  constexpr char long_value[] = "long_value";
  static_assert(sizeof(long_value) - 1 >= min_blob_size,
                "long_value too short");

  ASSERT_OK(Put("key1", short_value));
  ASSERT_OK(Put("key2", long_value));

  // There should be no files just yet since we haven't flushed.
  {
    VersionSet* const versions = dbfull()->GetVersionSet();
    ASSERT_NE(versions, nullptr);

    ColumnFamilyData* const cfd = versions->GetColumnFamilySet()->GetDefault();
    ASSERT_NE(cfd, nullptr);

    Version* const current = cfd->current();
    ASSERT_NE(current, nullptr);

    const VersionStorageInfo* const storage_info = current->storage_info();
    ASSERT_NE(storage_info, nullptr);

    ASSERT_EQ(storage_info->num_non_empty_levels(), 0);
    ASSERT_TRUE(storage_info->GetBlobFiles().empty());
  }

  // Reopen the database with blob files enabled. A new table file/blob file
  // pair should be written during recovery.
  Options options;
  options.enable_blob_files = true;
  options.min_blob_size = min_blob_size;
  options.avoid_flush_during_recovery = false;
  options.disable_auto_compactions = true;
  options.env = env_;

  Reopen(options);

  ASSERT_EQ(Get("key1"), short_value);
  ASSERT_EQ(Get("key2"), long_value);

  VersionSet* const versions = dbfull()->GetVersionSet();
  ASSERT_NE(versions, nullptr);

  ColumnFamilyData* const cfd = versions->GetColumnFamilySet()->GetDefault();
  ASSERT_NE(cfd, nullptr);

  Version* const current = cfd->current();
  ASSERT_NE(current, nullptr);

  const VersionStorageInfo* const storage_info = current->storage_info();
  ASSERT_NE(storage_info, nullptr);

  const auto& l0_files = storage_info->LevelFiles(0);
  ASSERT_EQ(l0_files.size(), 1);

  const FileMetaData* const table_file = l0_files[0];
  ASSERT_NE(table_file, nullptr);

  const auto& blob_files = storage_info->GetBlobFiles();
  ASSERT_EQ(blob_files.size(), 1);

  const auto& blob_file = blob_files.front();
  ASSERT_NE(blob_file, nullptr);

  ASSERT_EQ(table_file->smallest.user_key(), "key1");
  ASSERT_EQ(table_file->largest.user_key(), "key2");
  ASSERT_EQ(table_file->fd.smallest_seqno, 1);
  ASSERT_EQ(table_file->fd.largest_seqno, 2);
  ASSERT_EQ(table_file->oldest_blob_file_number,
            blob_file->GetBlobFileNumber());

  ASSERT_EQ(blob_file->GetTotalBlobCount(), 1);

  const InternalStats* const internal_stats = cfd->internal_stats();
  ASSERT_NE(internal_stats, nullptr);

  const auto& compaction_stats = internal_stats->TEST_GetCompactionStats();
  ASSERT_FALSE(compaction_stats.empty());
  ASSERT_EQ(compaction_stats[0].bytes_written, table_file->fd.GetFileSize());
  ASSERT_EQ(compaction_stats[0].bytes_written_blob,
            blob_file->GetTotalBlobBytes());
  ASSERT_EQ(compaction_stats[0].num_output_files, 1);
  ASSERT_EQ(compaction_stats[0].num_output_files_blob, 1);

  const uint64_t* const cf_stats_value = internal_stats->TEST_GetCFStatsValue();
  ASSERT_EQ(cf_stats_value[InternalStats::BYTES_FLUSHED],
            compaction_stats[0].bytes_written +
                compaction_stats[0].bytes_written_blob);
}

TEST_F(DBWALTest, RecoverWithBlobMultiSST) {
  // Write several large (4 KB) values without flushing. Note that blob files
  // are not actually enabled at this point.
  std::string large_value(1 << 12, 'a');

  constexpr int num_keys = 64;

  for (int i = 0; i < num_keys; ++i) {
    ASSERT_OK(Put(Key(i), large_value));
  }

  // There should be no files just yet since we haven't flushed.
  {
    VersionSet* const versions = dbfull()->GetVersionSet();
    ASSERT_NE(versions, nullptr);

    ColumnFamilyData* const cfd = versions->GetColumnFamilySet()->GetDefault();
    ASSERT_NE(cfd, nullptr);

    Version* const current = cfd->current();
    ASSERT_NE(current, nullptr);

    const VersionStorageInfo* const storage_info = current->storage_info();
    ASSERT_NE(storage_info, nullptr);

    ASSERT_EQ(storage_info->num_non_empty_levels(), 0);
    ASSERT_TRUE(storage_info->GetBlobFiles().empty());
  }

  // Reopen the database with blob files enabled and write buffer size set to a
  // smaller value. Multiple table files+blob files should be written and added
  // to the Version during recovery.
  Options options;
  options.write_buffer_size = 1 << 16;  // 64 KB
  options.enable_blob_files = true;
  options.avoid_flush_during_recovery = false;
  options.disable_auto_compactions = true;
  options.env = env_;

  Reopen(options);

  for (int i = 0; i < num_keys; ++i) {
    ASSERT_EQ(Get(Key(i)), large_value);
  }

  VersionSet* const versions = dbfull()->GetVersionSet();
  ASSERT_NE(versions, nullptr);

  ColumnFamilyData* const cfd = versions->GetColumnFamilySet()->GetDefault();
  ASSERT_NE(cfd, nullptr);

  Version* const current = cfd->current();
  ASSERT_NE(current, nullptr);

  const VersionStorageInfo* const storage_info = current->storage_info();
  ASSERT_NE(storage_info, nullptr);

  const auto& l0_files = storage_info->LevelFiles(0);
  ASSERT_GT(l0_files.size(), 1);

  const auto& blob_files = storage_info->GetBlobFiles();
  ASSERT_GT(blob_files.size(), 1);

  ASSERT_EQ(l0_files.size(), blob_files.size());
}

TEST_F(DBWALTest, WALWithChecksumHandoff) {
#ifndef ROCKSDB_ASSERT_STATUS_CHECKED
  if (mem_env_ || encrypted_env_) {
    ROCKSDB_GTEST_SKIP("Test requires non-mem or non-encrypted environment");
    return;
  }
  std::shared_ptr<FaultInjectionTestFS> fault_fs(
      new FaultInjectionTestFS(FileSystem::Default()));
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  // Close the DB before the local Env is destroyed if an ASSERT exits early.
  Defer close_db_on_exit([this]() { Close(); });
  do {
    Options options = CurrentOptions();

    options.checksum_handoff_file_types.Add(FileType::kWalFile);
    options.env = fault_fs_env.get();
    fault_fs->SetChecksumHandoffFuncType(ChecksumType::kCRC32c);

    CreateAndReopenWithCF({"pikachu"}, options);
    WriteOptions writeOpt = WriteOptions();
    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "foo", "v1"));
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v1"));

    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    ASSERT_EQ("v1", Get(1, "foo"));
    ASSERT_EQ("v1", Get(1, "bar"));

    writeOpt.disableWAL = false;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v2"));
    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "foo", "v2"));

    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    // Both value's should be present.
    ASSERT_EQ("v2", Get(1, "bar"));
    ASSERT_EQ("v2", Get(1, "foo"));

    writeOpt.disableWAL = true;
    // This put, data is persisted by Flush
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v3"));
    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    writeOpt.disableWAL = false;
    // Data is persisted in the WAL
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "zoo", "v3"));
    ASSERT_OK(dbfull()->SyncWAL());
    // The hash does not match, write fails
    fault_fs->SetChecksumHandoffFuncType(ChecksumType::kxxHash);
    writeOpt.disableWAL = false;
    ASSERT_NOK(dbfull()->Put(writeOpt, handles_[1], "foo", "v3"));

    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    // Due to the write failure, Get should not find
    ASSERT_NE("v3", Get(1, "foo"));
    ASSERT_EQ("v3", Get(1, "zoo"));
    ASSERT_EQ("v3", Get(1, "bar"));

    fault_fs->SetChecksumHandoffFuncType(ChecksumType::kCRC32c);
    // Each write will be similated as corrupted.
    fault_fs->IngestDataCorruptionBeforeWrite();
    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v4"));
    writeOpt.disableWAL = false;
    ASSERT_NOK(dbfull()->Put(writeOpt, handles_[1], "foo", "v4"));
    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    ASSERT_NE("v4", Get(1, "foo"));
    ASSERT_NE("v4", Get(1, "bar"));
    fault_fs->NoDataCorruptionBeforeWrite();

    fault_fs->SetChecksumHandoffFuncType(ChecksumType::kNoChecksum);
    // The file system does not provide checksum method and verification.
    writeOpt.disableWAL = true;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "bar", "v5"));
    writeOpt.disableWAL = false;
    ASSERT_OK(dbfull()->Put(writeOpt, handles_[1], "foo", "v5"));
    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    ASSERT_EQ("v5", Get(1, "foo"));
    ASSERT_EQ("v5", Get(1, "bar"));

    Destroy(options);
  } while (ChangeWalOptions());
#endif  // ROCKSDB_ASSERT_STATUS_CHECKED
}

TEST_F(DBWALTest, LockWal) {
  do {
    Options options = CurrentOptions();
    options.create_if_missing = true;
    DestroyAndReopen(options);

    ASSERT_OK(Put("foo", "v"));
    ASSERT_OK(Put("bar", "v"));

    ASSERT_OK(db_->LockWAL());
    // Verify writes are stopped
    WriteOptions wopts;
    wopts.no_slowdown = true;
    Status s = db_->Put(wopts, "foo", "dontcare");
    ASSERT_TRUE(s.IsIncomplete());
    {
      VectorLogPtr wals;
      ASSERT_OK(db_->GetSortedWalFiles(wals));
      ASSERT_FALSE(wals.empty());
    }
    port::Thread worker([&]() {
      Status tmp_s = db_->Flush(FlushOptions());
      ASSERT_OK(tmp_s);
    });
    FlushOptions flush_opts;
    flush_opts.wait = false;
    s = db_->Flush(flush_opts);
    ASSERT_TRUE(s.IsTryAgain());
    ASSERT_OK(db_->UnlockWAL());
    ASSERT_OK(db_->Put(WriteOptions(), "foo", "dontcare"));

    worker.join();
  } while (ChangeWalOptions());
}

class DBRecoveryTestBlobError
    : public DBWALTest,
      public testing::WithParamInterface<std::string> {
 public:
  DBRecoveryTestBlobError() : sync_point_(GetParam()) {}

  std::string sync_point_;
};

INSTANTIATE_TEST_CASE_P(DBRecoveryTestBlobError, DBRecoveryTestBlobError,
                        ::testing::ValuesIn(std::vector<std::string>{
                            "BlobFileBuilder::WriteBlobToFile:AddRecord",
                            "BlobFileBuilder::WriteBlobToFile:AppendFooter"}));

TEST_P(DBRecoveryTestBlobError, RecoverWithBlobError) {
  // Write a value. Note that blob files are not actually enabled at this point.
  ASSERT_OK(Put("key", "blob"));

  // Reopen with blob files enabled but make blob file writing fail during
  // recovery.
  SyncPoint::GetInstance()->SetCallBack(sync_point_, [this](void* arg) {
    Status* const s = static_cast<Status*>(arg);
    assert(s);

    (*s) = Status::IOError(sync_point_);
  });
  SyncPoint::GetInstance()->EnableProcessing();

  Options options;
  options.enable_blob_files = true;
  options.avoid_flush_during_recovery = false;
  options.disable_auto_compactions = true;
  options.env = env_;

  ASSERT_NOK(TryReopen(options));

  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();

  // Make sure the files generated by the failed recovery have been deleted.
  std::vector<std::string> files;
  ASSERT_OK(env_->GetChildren(dbname_, &files));
  for (const auto& file : files) {
    uint64_t number = 0;
    FileType type = kTableFile;

    if (!ParseFileName(file, &number, &type)) {
      continue;
    }

    ASSERT_NE(type, kTableFile);
    ASSERT_NE(type, kBlobFile);
  }
}

TEST_F(DBWALTest, IgnoreRecoveredLog) {
  std::string backup_logs = dbname_ + "/backup_logs";

  do {
    // delete old files in backup_logs directory
    ASSERT_OK(env_->CreateDirIfMissing(backup_logs));
    std::vector<std::string> old_files;
    ASSERT_OK(env_->GetChildren(backup_logs, &old_files));
    for (auto& file : old_files) {
      ASSERT_OK(env_->DeleteFile(backup_logs + "/" + file));
    }
    Options options = CurrentOptions();
    options.create_if_missing = true;
    options.merge_operator = MergeOperators::CreateUInt64AddOperator();
    options.wal_dir = dbname_ + "/logs";
    DestroyAndReopen(options);

    // fill up the DB
    std::string one;
    std::string two;
    PutFixed64(&one, 1);
    PutFixed64(&two, 2);
    ASSERT_OK(db_->Merge(WriteOptions(), Slice("foo"), Slice(one)));
    ASSERT_OK(db_->Merge(WriteOptions(), Slice("foo"), Slice(one)));
    ASSERT_OK(db_->Merge(WriteOptions(), Slice("bar"), Slice(one)));

    // copy the logs to backup
    std::vector<std::string> logs;
    ASSERT_OK(env_->GetChildren(options.wal_dir, &logs));
    for (auto& log : logs) {
      CopyFile(options.wal_dir + "/" + log, backup_logs + "/" + log);
    }

    // recover the DB
    Reopen(options);
    ASSERT_EQ(two, Get("foo"));
    ASSERT_EQ(one, Get("bar"));
    Close();

    // copy the logs from backup back to wal dir
    for (auto& log : logs) {
      CopyFile(backup_logs + "/" + log, options.wal_dir + "/" + log);
    }
    // this should ignore the log files, recovery should not happen again
    // if the recovery happens, the same merge operator would be called twice,
    // leading to incorrect results
    Reopen(options);
    ASSERT_EQ(two, Get("foo"));
    ASSERT_EQ(one, Get("bar"));
    Close();
    Destroy(options);
    Reopen(options);
    Close();

    // copy the logs from backup back to wal dir
    ASSERT_OK(env_->CreateDirIfMissing(options.wal_dir));
    for (auto& log : logs) {
      CopyFile(backup_logs + "/" + log, options.wal_dir + "/" + log);
    }
    // assert that we successfully recovered only from logs, even though we
    // destroyed the DB
    Reopen(options);
    ASSERT_EQ(two, Get("foo"));
    ASSERT_EQ(one, Get("bar"));

    // Recovery will fail if DB directory doesn't exist.
    Destroy(options);
    // copy the logs from backup back to wal dir
    ASSERT_OK(env_->CreateDirIfMissing(options.wal_dir));
    for (auto& log : logs) {
      CopyFile(backup_logs + "/" + log, options.wal_dir + "/" + log);
      // we won't be needing this file no more
      ASSERT_OK(env_->DeleteFile(backup_logs + "/" + log));
    }
    Status s = TryReopen(options);
    ASSERT_NOK(s);
    Destroy(options);
  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, RecoveryWithEmptyLog) {
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());
    ASSERT_OK(Put(1, "foo", "v1"));
    ASSERT_OK(Put(1, "foo", "v2"));
    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    ASSERT_OK(Put(1, "foo", "v3"));
    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    ASSERT_EQ("v3", Get(1, "foo"));
  } while (ChangeWalOptions());
}

#if !(defined NDEBUG) || !defined(OS_WIN)
TEST_F(DBWALTest, PreallocateBlock) {
  Options options = CurrentOptions();
  options.write_buffer_size = 10 * 1000 * 1000;
  options.max_total_wal_size = 0;

  size_t expected_preallocation_size = static_cast<size_t>(
      options.write_buffer_size + options.write_buffer_size / 10);

  DestroyAndReopen(options);

  std::atomic<int> called(0);
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "DBTestWalFile.GetPreallocationStatus", [&](void* arg) {
        ASSERT_TRUE(arg != nullptr);
        size_t preallocation_size = *(static_cast<size_t*>(arg));
        ASSERT_EQ(expected_preallocation_size, preallocation_size);
        called.fetch_add(1);
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Put("", ""));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("", ""));
  Close();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ASSERT_EQ(2, called.load());

  options.max_total_wal_size = 1000 * 1000;
  expected_preallocation_size = static_cast<size_t>(options.max_total_wal_size);
  Reopen(options);
  called.store(0);
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "DBTestWalFile.GetPreallocationStatus", [&](void* arg) {
        ASSERT_TRUE(arg != nullptr);
        size_t preallocation_size = *(static_cast<size_t*>(arg));
        ASSERT_EQ(expected_preallocation_size, preallocation_size);
        called.fetch_add(1);
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Put("", ""));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("", ""));
  Close();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ASSERT_EQ(2, called.load());

  options.db_write_buffer_size = 800 * 1000;
  expected_preallocation_size =
      static_cast<size_t>(options.db_write_buffer_size);
  Reopen(options);
  called.store(0);
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "DBTestWalFile.GetPreallocationStatus", [&](void* arg) {
        ASSERT_TRUE(arg != nullptr);
        size_t preallocation_size = *(static_cast<size_t*>(arg));
        ASSERT_EQ(expected_preallocation_size, preallocation_size);
        called.fetch_add(1);
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Put("", ""));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("", ""));
  Close();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ASSERT_EQ(2, called.load());

  expected_preallocation_size = 700 * 1000;
  std::shared_ptr<WriteBufferManager> write_buffer_manager =
      std::make_shared<WriteBufferManager>(static_cast<uint64_t>(700 * 1000));
  options.write_buffer_manager = write_buffer_manager;
  Reopen(options);
  called.store(0);
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "DBTestWalFile.GetPreallocationStatus", [&](void* arg) {
        ASSERT_TRUE(arg != nullptr);
        size_t preallocation_size = *(static_cast<size_t*>(arg));
        ASSERT_EQ(expected_preallocation_size, preallocation_size);
        called.fetch_add(1);
      });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Put("", ""));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("", ""));
  Close();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ASSERT_EQ(2, called.load());
}
#endif  // !(defined NDEBUG) || !defined(OS_WIN)

TEST_F(DBWALTest, FullPurgePreservesRecycledLog) {
  // For github issue #1303
  for (int i = 0; i < 2; ++i) {
    Options options = CurrentOptions();
    options.create_if_missing = true;
    options.recycle_log_file_num = 2;
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    if (i != 0) {
      options.wal_dir = alternative_wal_dir_;
    }

    DestroyAndReopen(options);
    ASSERT_OK(Put("foo", "v1"));
    VectorLogPtr log_files;
    ASSERT_OK(dbfull()->GetSortedWalFiles(log_files));
    ASSERT_GT(log_files.size(), 0);
    ASSERT_OK(Flush());

    // Now the original WAL is in log_files[0] and should be marked for
    // recycling.
    // Verify full purge cannot remove this file.
    JobContext job_context(0);
    dbfull()->TEST_LockMutex();
    dbfull()->FindObsoleteFiles(&job_context, true /* force */);
    dbfull()->TEST_UnlockMutex();
    dbfull()->PurgeObsoleteFiles(job_context);
    job_context.Clean();

    if (i == 0) {
      ASSERT_OK(
          env_->FileExists(LogFileName(dbname_, log_files[0]->LogNumber())));
    } else {
      ASSERT_OK(env_->FileExists(
          LogFileName(alternative_wal_dir_, log_files[0]->LogNumber())));
    }
  }
}

TEST_F(DBWALTest, FullPurgePreservesLogPendingReuse) {
  // Ensures full purge cannot delete a WAL while it's in the process of being
  // recycled. In particular, we force the full purge after a file has been
  // chosen for reuse, but before it has been renamed.
  for (int i = 0; i < 2; ++i) {
    Options options = CurrentOptions();
    options.recycle_log_file_num = 1;
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    if (i != 0) {
      options.wal_dir = alternative_wal_dir_;
    }
    DestroyAndReopen(options);

    // The first flush creates a second log so writes can continue before the
    // flush finishes.
    ASSERT_OK(Put("foo", "bar"));
    ASSERT_OK(Flush());

    // The second flush can recycle the first log. Sync points enforce the
    // full purge happens after choosing the log to recycle and before it is
    // renamed.
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency({
        {"DBImpl::CreateWAL:BeforeReuseWritableFile1",
         "DBWALTest::FullPurgePreservesLogPendingReuse:PreFullPurge"},
        {"DBWALTest::FullPurgePreservesLogPendingReuse:PostFullPurge",
         "DBImpl::CreateWAL:BeforeReuseWritableFile2"},
    });
    ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
    ROCKSDB_NAMESPACE::port::Thread thread([&]() {
      TEST_SYNC_POINT(
          "DBWALTest::FullPurgePreservesLogPendingReuse:PreFullPurge");
      ASSERT_OK(db_->EnableFileDeletions());
      TEST_SYNC_POINT(
          "DBWALTest::FullPurgePreservesLogPendingReuse:PostFullPurge");
    });
    ASSERT_OK(Put("foo", "bar"));
    ASSERT_OK(Flush());
    thread.join();
  }
}

TEST_F(DBWALTest, GetSortedWalFiles) {
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());
    VectorLogPtr log_files;
    ASSERT_OK(dbfull()->GetSortedWalFiles(log_files));
    ASSERT_EQ(0, log_files.size());

    ASSERT_OK(Put(1, "foo", "v1"));
    ASSERT_OK(dbfull()->GetSortedWalFiles(log_files));
    ASSERT_EQ(1, log_files.size());
  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, GetCurrentWalFile) {
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());

    std::unique_ptr<LogFile>* bad_log_file = nullptr;
    ASSERT_NOK(dbfull()->GetCurrentWalFile(bad_log_file));

    std::unique_ptr<LogFile> log_file;
    ASSERT_OK(dbfull()->GetCurrentWalFile(&log_file));

    // nothing has been written to the log yet
    ASSERT_EQ(log_file->StartSequence(), 0);
    ASSERT_EQ(log_file->SizeFileBytes(), 0);
    ASSERT_EQ(log_file->Type(), kAliveLogFile);
    ASSERT_GT(log_file->LogNumber(), 0);

    // add some data and verify that the file size actually moves foward
    ASSERT_OK(Put(0, "foo", "v1"));
    ASSERT_OK(Put(0, "foo2", "v2"));
    ASSERT_OK(Put(0, "foo3", "v3"));

    ASSERT_OK(dbfull()->GetCurrentWalFile(&log_file));

    ASSERT_EQ(log_file->StartSequence(), 0);
    ASSERT_GT(log_file->SizeFileBytes(), 0);
    ASSERT_EQ(log_file->Type(), kAliveLogFile);
    ASSERT_GT(log_file->LogNumber(), 0);

    // force log files to cycle and add some more data, then check if
    // log number moves forward

    ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    for (int i = 0; i < 10; i++) {
      ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
    }

    ASSERT_OK(Put(0, "foo4", "v4"));
    ASSERT_OK(Put(0, "foo5", "v5"));
    ASSERT_OK(Put(0, "foo6", "v6"));

    ASSERT_OK(dbfull()->GetCurrentWalFile(&log_file));

    ASSERT_EQ(log_file->StartSequence(), 0);
    ASSERT_GT(log_file->SizeFileBytes(), 0);
    ASSERT_EQ(log_file->Type(), kAliveLogFile);
    ASSERT_GT(log_file->LogNumber(), 0);

  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, RecoveryWithLogDataForSomeCFs) {
  // Test for regression of WAL cleanup missing files that don't contain data
  // for every column family.
  do {
    CreateAndReopenWithCF({"pikachu"}, CurrentOptions());
    ASSERT_OK(Put(1, "foo", "v1"));
    ASSERT_OK(Put(1, "foo", "v2"));
    uint64_t earliest_log_nums[2];
    for (int i = 0; i < 2; ++i) {
      if (i > 0) {
        ReopenWithColumnFamilies({"default", "pikachu"}, CurrentOptions());
      }
      VectorLogPtr log_files;
      ASSERT_OK(dbfull()->GetSortedWalFiles(log_files));
      if (log_files.size() > 0) {
        earliest_log_nums[i] = log_files[0]->LogNumber();
      } else {
        earliest_log_nums[i] = std::numeric_limits<uint64_t>::max();
      }
    }
    // Check at least the first WAL was cleaned up during the recovery.
    ASSERT_LT(earliest_log_nums[0], earliest_log_nums[1]);
  } while (ChangeWalOptions());
}

TEST_F(DBWALTest, RecoverWithLargeLog) {
  do {
    {
      Options options = CurrentOptions();
      CreateAndReopenWithCF({"pikachu"}, options);
      ASSERT_OK(Put(1, "big1", std::string(200000, '1')));
      ASSERT_OK(Put(1, "big2", std::string(200000, '2')));
      ASSERT_OK(Put(1, "small3", std::string(10, '3')));
      ASSERT_OK(Put(1, "small4", std::string(10, '4')));
      ASSERT_EQ(NumTableFilesAtLevel(0, 1), 0);
    }

    // Make sure that if we re-open with a small write buffer size that
    // we flush table files in the middle of a large log file.
    Options options;
    options.write_buffer_size = 100000;
    options = CurrentOptions(options);
    ReopenWithColumnFamilies({"default", "pikachu"}, options);
    ASSERT_EQ(NumTableFilesAtLevel(0, 1), 3);
    ASSERT_EQ(std::string(200000, '1'), Get(1, "big1"));
    ASSERT_EQ(std::string(200000, '2'), Get(1, "big2"));
    ASSERT_EQ(std::string(10, '3'), Get(1, "small3"));
    ASSERT_EQ(std::string(10, '4'), Get(1, "small4"));
    ASSERT_GT(NumTableFilesAtLevel(0, 1), 1);
  } while (ChangeWalOptions());
}

// In https://reviews.facebook.net/D20661 we change
// recovery behavior: previously for each log file each column family
// memtable was flushed, even it was empty. Now it's changed:
// we try to create the smallest number of table files by merging
// updates from multiple logs
TEST_F(DBWALTest, RecoverCheckFileAmountWithSmallWriteBuffer) {
  Options options = CurrentOptions();
  options.write_buffer_size = 5000000;
  CreateAndReopenWithCF({"pikachu", "dobrynia", "nikitich"}, options);

  // Since we will reopen DB with smaller write_buffer_size,
  // each key will go to new SST file
  ASSERT_OK(Put(1, Key(10), DummyString(1000000)));
  ASSERT_OK(Put(1, Key(10), DummyString(1000000)));
  ASSERT_OK(Put(1, Key(10), DummyString(1000000)));
  ASSERT_OK(Put(1, Key(10), DummyString(1000000)));

  ASSERT_OK(Put(3, Key(10), DummyString(1)));
  // Make 'dobrynia' to be flushed and new WAL file to be created
  ASSERT_OK(Put(2, Key(10), DummyString(7500000)));
  ASSERT_OK(Put(2, Key(1), DummyString(1)));
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[2]));
  {
    auto tables = ListTableFiles(env_, dbname_);
    ASSERT_EQ(tables.size(), static_cast<size_t>(1));
    // Make sure 'dobrynia' was flushed: check sst files amount
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "dobrynia"),
              static_cast<uint64_t>(1));
  }
  // New WAL file
  ASSERT_OK(Put(1, Key(1), DummyString(1)));
  ASSERT_OK(Put(1, Key(1), DummyString(1)));
  ASSERT_OK(Put(3, Key(10), DummyString(1)));
  ASSERT_OK(Put(3, Key(10), DummyString(1)));
  ASSERT_OK(Put(3, Key(10), DummyString(1)));

  options.write_buffer_size = 4096;
  options.arena_block_size = 4096;
  ReopenWithColumnFamilies({"default", "pikachu", "dobrynia", "nikitich"},
                           options);
  {
    // No inserts => default is empty
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "default"),
              static_cast<uint64_t>(0));
    // First 4 keys goes to separate SSTs + 1 more SST for 2 smaller keys
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"),
              static_cast<uint64_t>(5));
    // 1 SST for big key + 1 SST for small one
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "dobrynia"),
              static_cast<uint64_t>(2));
    // 1 SST for all keys
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "nikitich"),
              static_cast<uint64_t>(1));
  }
}

// In https://reviews.facebook.net/D20661 we change
// recovery behavior: previously for each log file each column family
// memtable was flushed, even it wasn't empty. Now it's changed:
// we try to create the smallest number of table files by merging
// updates from multiple logs
TEST_F(DBWALTest, RecoverCheckFileAmount) {
  Options options = CurrentOptions();
  options.write_buffer_size = 100000;
  options.arena_block_size = 4 * 1024;
  options.avoid_flush_during_recovery = false;
  CreateAndReopenWithCF({"pikachu", "dobrynia", "nikitich"}, options);

  ASSERT_OK(Put(0, Key(1), DummyString(1)));
  ASSERT_OK(Put(1, Key(1), DummyString(1)));
  ASSERT_OK(Put(2, Key(1), DummyString(1)));

  // Make 'nikitich' memtable to be flushed
  ASSERT_OK(Put(3, Key(10), DummyString(1002400)));
  ASSERT_OK(Put(3, Key(1), DummyString(1)));
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[3]));
  // 4 memtable are not flushed, 1 sst file
  {
    auto tables = ListTableFiles(env_, dbname_);
    ASSERT_EQ(tables.size(), static_cast<size_t>(1));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "nikitich"),
              static_cast<uint64_t>(1));
  }
  // Memtable for 'nikitich' has flushed, new WAL file has opened
  // 4 memtable still not flushed

  // Write to new WAL file
  ASSERT_OK(Put(0, Key(1), DummyString(1)));
  ASSERT_OK(Put(1, Key(1), DummyString(1)));
  ASSERT_OK(Put(2, Key(1), DummyString(1)));

  // Fill up 'nikitich' one more time
  ASSERT_OK(Put(3, Key(10), DummyString(1002400)));
  // make it flush
  ASSERT_OK(Put(3, Key(1), DummyString(1)));
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[3]));
  // There are still 4 memtable not flushed, and 2 sst tables
  ASSERT_OK(Put(0, Key(1), DummyString(1)));
  ASSERT_OK(Put(1, Key(1), DummyString(1)));
  ASSERT_OK(Put(2, Key(1), DummyString(1)));

  {
    auto tables = ListTableFiles(env_, dbname_);
    ASSERT_EQ(tables.size(), static_cast<size_t>(2));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "nikitich"),
              static_cast<uint64_t>(2));
  }

  ReopenWithColumnFamilies({"default", "pikachu", "dobrynia", "nikitich"},
                           options);
  {
    std::vector<uint64_t> table_files = ListTableFiles(env_, dbname_);
    // Check, that records for 'default', 'dobrynia' and 'pikachu' from
    // first, second and third WALs  went to the same SST.
    // So, there is 6 SSTs: three  for 'nikitich', one for 'default', one for
    // 'dobrynia', one for 'pikachu'
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "default"),
              static_cast<uint64_t>(1));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "nikitich"),
              static_cast<uint64_t>(3));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "dobrynia"),
              static_cast<uint64_t>(1));
    ASSERT_EQ(GetNumberOfSstFilesForColumnFamily(db_.get(), "pikachu"),
              static_cast<uint64_t>(1));
  }
}

TEST_F(DBWALTest, SyncMultipleLogs) {
  const uint64_t kNumBatches = 2;
  const int kBatchSize = 1000;

  Options options = CurrentOptions();
  options.create_if_missing = true;
  options.write_buffer_size = 4096;
  Reopen(options);

  WriteBatch batch;
  WriteOptions wo;
  wo.sync = true;

  for (uint64_t b = 0; b < kNumBatches; b++) {
    batch.Clear();
    for (int i = 0; i < kBatchSize; i++) {
      ASSERT_OK(batch.Put(Key(i), DummyString(128)));
    }

    ASSERT_OK(dbfull()->Write(wo, &batch));
  }

  ASSERT_OK(dbfull()->SyncWAL());
}

TEST_F(DBWALTest, DISABLED_RecycleMultipleWalsCrash) {
  Options options = CurrentOptions();
  options.max_write_buffer_number = 5;
  options.track_and_verify_wals_in_manifest = true;
  options.max_bgerror_resume_count = 0;  // manual resume
  options.recycle_log_file_num = 3;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;

  // Disable truncating recycled WALs to new size in posix env
  // (approximating a crash)
  SyncPoint::GetInstance()->SetCallBack(
      "PosixWritableFile::Close",
      [](void* arg) { *(static_cast<size_t*>(arg)) = 0; });
  SyncPoint::GetInstance()->EnableProcessing();

  // Re-open with desired options
  DestroyAndReopen(options);
  Defer closer([this]() { Close(); });

  // Ensure WAL recycling wasn't sanitized away
  ASSERT_EQ(db_->GetOptions().recycle_log_file_num,
            options.recycle_log_file_num);

  // Prepare external files for later ingestion
  std::string sst_files_dir = dbname_ + "/sst_files/";
  ASSERT_OK(DestroyDir(env_, sst_files_dir));
  ASSERT_OK(env_->CreateDir(sst_files_dir));
  std::string external_file1 = sst_files_dir + "file1.sst";
  {
    SstFileWriter sst_file_writer(EnvOptions(), options);
    ASSERT_OK(sst_file_writer.Open(external_file1));
    ASSERT_OK(sst_file_writer.Put("external1", "ex1"));
    ExternalSstFileInfo file_info;
    ASSERT_OK(sst_file_writer.Finish(&file_info));
  }
  std::string external_file2 = sst_files_dir + "file2.sst";
  {
    SstFileWriter sst_file_writer(EnvOptions(), options);
    ASSERT_OK(sst_file_writer.Open(external_file2));
    ASSERT_OK(sst_file_writer.Put("external2", "ex2"));
    ExternalSstFileInfo file_info;
    ASSERT_OK(sst_file_writer.Finish(&file_info));
  }

  // Populate some WALs to be recycled such that there will be extra data
  // from an old incarnation of the WAL on recovery
  ASSERT_OK(db_->PauseBackgroundWork());
  ASSERT_OK(Put("ignore1", Random::GetTLSInstance()->RandomString(500)));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_OK(Put("ignore2", Random::GetTLSInstance()->RandomString(500)));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_OK(db_->ContinueBackgroundWork());
  ASSERT_OK(Flush());
  ASSERT_OK(Put("ignore3", Random::GetTLSInstance()->RandomString(500)));
  ASSERT_OK(Flush());

  // Verify expected log files (still there for recycling)
  std::vector<FileAttributes> files;
  int log_count = 0;
  ASSERT_OK(options.env->GetChildrenFileAttributes(dbname_, &files));
  for (const auto& f : files) {
    if (EndsWith(f.name, ".log")) {
      EXPECT_GT(f.size_bytes, 500);
      ++log_count;
    }
  }
  EXPECT_EQ(log_count, 3);

  // (Re-used recipe) Generate two inactive WALs and one active WAL, with a
  // gap in sequence numbers to interfere with recovery
  ASSERT_OK(db_->PauseBackgroundWork());
  ASSERT_OK(Put("key1", "val1"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_OK(Put("key2", "val2"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  // Need a gap in sequence numbers, so e.g. ingest external file
  // with an open snapshot
  {
    ManagedSnapshot snapshot(db_.get());
    ASSERT_OK(
        db_->IngestExternalFile({external_file1}, IngestExternalFileOptions()));
  }
  ASSERT_OK(Put("key3", "val3"));
  ASSERT_OK(db_->SyncWAL());
  // Need an SST file that is logically after that WAL, so that dropping WAL
  // data is not a valid point in time.
  {
    ManagedSnapshot snapshot(db_.get());
    ASSERT_OK(
        db_->IngestExternalFile({external_file2}, IngestExternalFileOptions()));
  }

  // Approximate a crash, with respect to recycled WAL data extending past
  // the end of the current WAL data (see SyncPoint callback above)
  Close();

  // Verify recycled log files haven't been truncated
  files.clear();
  log_count = 0;
  ASSERT_OK(options.env->GetChildrenFileAttributes(dbname_, &files));
  for (const auto& f : files) {
    if (EndsWith(f.name, ".log")) {
      EXPECT_GT(f.size_bytes, 500);
      ++log_count;
    }
  }
  EXPECT_EQ(log_count, 3);

  // Verify no data loss after reopen.
  Reopen(options);
  EXPECT_EQ("val1", Get("key1"));
  EXPECT_EQ("val2", Get("key2"));  // Passes because of adjacent seqnos
  EXPECT_EQ("ex1", Get("external1"));
  EXPECT_EQ("val3", Get("key3"));  // <- ONLY FAILURE! (Not a point in time)
  EXPECT_EQ("ex2", Get("external2"));

  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST_F(DBWALTest, SyncWalPartialFailure) {
  class MyTestFileSystem : public FileSystemWrapper {
   public:
    explicit MyTestFileSystem(std::shared_ptr<FileSystem> base)
        : FileSystemWrapper(std::move(base)) {}

    static const char* kClassName() { return "MyTestFileSystem"; }
    const char* Name() const override { return kClassName(); }
    IOStatus NewWritableFile(const std::string& fname,
                             const FileOptions& file_opts,
                             std::unique_ptr<FSWritableFile>* result,
                             IODebugContext* dbg) override {
      IOStatus s = target()->NewWritableFile(fname, file_opts, result, dbg);
      if (s.ok()) {
        *result =
            std::make_unique<MyTestWritableFile>(std::move(*result), *this);
      }
      return s;
    }

    Atomic<uint32_t> syncs_before_failure_{UINT32_MAX};

   protected:
    class MyTestWritableFile : public FSWritableFileOwnerWrapper {
     public:
      MyTestWritableFile(std::unique_ptr<FSWritableFile>&& file,
                         MyTestFileSystem& fs)
          : FSWritableFileOwnerWrapper(std::move(file)), fs_(fs) {}

      IOStatus Sync(const IOOptions& options, IODebugContext* dbg) override {
        int prev_val = fs_.syncs_before_failure_.FetchSub(1);
        if (prev_val == 0) {
          return IOStatus::IOError("fault");
        } else {
          return target()->Sync(options, dbg);
        }
      }

     protected:
      MyTestFileSystem& fs_;
    };
  };

  Options options = CurrentOptions();
  options.max_write_buffer_number = 4;
  options.track_and_verify_wals_in_manifest = true;
  options.max_bgerror_resume_count = 0;  // manual resume

  auto custom_fs =
      std::make_shared<MyTestFileSystem>(options.env->GetFileSystem());
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(custom_fs));
  options.env = fault_fs_env.get();
  Reopen(options);
  Defer closer([this]() { Close(); });

  // This is the simplest way to get
  // * one inactive WAL, synced
  // * one inactive WAL, not synced, and
  // * one active WAL, not synced
  // with a single thread, to exercise as much logic as we reasonably can.
  ASSERT_OK(db_->PauseBackgroundWork());
  ASSERT_OK(Put("key1", "val1"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put("key2", "val2"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_OK(Put("key3", "val3"));

  // Allow 1 of the WALs to sync, but another won't
  custom_fs->syncs_before_failure_.Store(1);
  ASSERT_NOK(db_->SyncWAL());

  // Stuck in this state. (This could previously cause a segfault.)
  ASSERT_NOK(db_->SyncWAL());

  // Can't Resume because WAL write failure is considered non-recoverable,
  // regardless of the IOStatus itself. (Can/should be fixed?)
  ASSERT_NOK(db_->Resume());

  // Verify no data loss after reopen.
  // Also Close() could previously crash in this state.
  Reopen(options);
  ASSERT_EQ("val1", Get("key1"));
  ASSERT_EQ("val2", Get("key2"));
  ASSERT_EQ("val3", Get("key3"));
}

// Github issue 1339. Prior the fix we read sequence id from the first log to
// a local variable, then keep increase the variable as we replay logs,
// ignoring actual sequence id of the records. This is incorrect if some writes
// come with WAL disabled.
TEST_F(DBWALTest, PartOfWritesWithWALDisabled) {
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Options options = CurrentOptions();
  options.env = fault_env.get();
  options.disable_auto_compactions = true;
  WriteOptions wal_on;
  WriteOptions wal_off;
  wal_on.sync = true;
  wal_on.disableWAL = false;
  wal_off.disableWAL = true;
  CreateAndReopenWithCF({"dummy"}, options);
  ASSERT_OK(Put(1, "dummy", "d1", wal_on));  // seq id 1
  ASSERT_OK(Put(1, "dummy", "d2", wal_off));
  ASSERT_OK(Put(1, "dummy", "d3", wal_off));
  ASSERT_OK(Put(0, "key", "v4", wal_on));  // seq id 4
  ASSERT_OK(Flush(0));
  ASSERT_OK(Put(0, "key", "v5", wal_on));  // seq id 5
  ASSERT_EQ("v5", Get(0, "key"));
  ASSERT_OK(dbfull()->FlushWAL(false));
  // Simulate a crash.
  fault_env->SetFilesystemActive(false);
  Close();
  fault_env->ResetState();
  ReopenWithColumnFamilies({"default", "dummy"}, options);
  // Prior to the fix, we may incorrectly recover "v5" with sequence id = 3.
  ASSERT_EQ("v5", Get(0, "key"));
  // Destroy DB before destruct fault_env.
  Destroy(options);
}

//
// Test WAL recovery for the various modes available
//
class RecoveryTestHelper {
 public:
  // Number of WAL files to generate
  static constexpr int kWALFilesCount = 10;
  // Starting number for the WAL file name like 00010.log
  static constexpr int kWALFileOffset = 10;
  // Keys to be written per WAL file
  static constexpr int kKeysPerWALFile = 133;
  // Size of the value
  static constexpr int kValueSize = 96;

  // Create WAL files with values filled in
  static void FillData(DBWALTestBase* test, const Options& options,
                       const size_t wal_count, size_t* count) {
    // Calling internal functions requires sanitized options.
    Options sanitized_options = SanitizeOptions(test->dbname_, options);
    const ImmutableDBOptions db_options(sanitized_options);

    *count = 0;

    std::shared_ptr<Cache> table_cache = NewLRUCache(50, 0);
    FileOptions file_options;
    WriteBufferManager write_buffer_manager(db_options.db_write_buffer_size);

    std::unique_ptr<VersionSet> versions;
    std::unique_ptr<WalManager> wal_manager;
    WriteController write_controller;

    versions.reset(new VersionSet(
        test->dbname_, &db_options, MutableDBOptions{options}, file_options,
        table_cache.get(), &write_buffer_manager, &write_controller,
        /*block_cache_tracer=*/nullptr,
        /*io_tracer=*/nullptr, /*db_id=*/"", /*db_session_id=*/"",
        options.daily_offpeak_time_utc,
        /*error_handler=*/nullptr, /*read_only=*/false));

    wal_manager.reset(
        new WalManager(db_options, file_options, /*io_tracer=*/nullptr));

    std::unique_ptr<log::Writer> current_log_writer;

    for (size_t j = kWALFileOffset; j < wal_count + kWALFileOffset; j++) {
      uint64_t current_log_number = j;
      std::string fname = LogFileName(test->dbname_, current_log_number);
      std::unique_ptr<WritableFileWriter> file_writer;
      ASSERT_OK(WritableFileWriter::Create(db_options.env->GetFileSystem(),
                                           fname, file_options, &file_writer,
                                           nullptr));
      log::Writer* log_writer =
          new log::Writer(std::move(file_writer), current_log_number,
                          db_options.recycle_log_file_num > 0, false,
                          db_options.wal_compression);
      ASSERT_OK(log_writer->AddCompressionTypeRecord(WriteOptions()));
      current_log_writer.reset(log_writer);

      WriteBatch batch;
      for (int i = 0; i < kKeysPerWALFile; i++) {
        std::string key = "key" + std::to_string((*count)++);
        std::string value = test->DummyString(kValueSize);
        ASSERT_NE(current_log_writer.get(), nullptr);
        uint64_t seq = versions->LastSequence() + 1;
        batch.Clear();
        ASSERT_OK(batch.Put(key, value));
        WriteBatchInternal::SetSequence(&batch, seq);
        ASSERT_OK(current_log_writer->AddRecord(
            WriteOptions(), WriteBatchInternal::Contents(&batch)));
        versions->SetLastAllocatedSequence(seq);
        versions->SetLastPublishedSequence(seq);
        versions->SetLastSequence(seq);
      }
    }
  }

  // Recreate and fill the store with some data
  static size_t FillData(DBWALTestBase* test, Options* options) {
    options->create_if_missing = true;
    test->DestroyAndReopen(*options);
    test->Close();

    size_t count = 0;
    FillData(test, *options, kWALFilesCount, &count);
    return count;
  }

  // Read back all the keys we wrote and return the number of keys found
  static size_t GetData(DBWALTestBase* test) {
    size_t count = 0;
    for (size_t i = 0; i < kWALFilesCount * kKeysPerWALFile; i++) {
      if (test->Get("key" + std::to_string(i)) != "NOT_FOUND") {
        ++count;
      }
    }
    return count;
  }

  // Manuall corrupt the specified WAL
  static void CorruptWAL(DBWALTestBase* test, const Options& options,
                         const double off, const double len,
                         const int wal_file_id, const bool trunc = false) {
    Env* env = options.env;
    std::string fname = LogFileName(test->dbname_, wal_file_id);
    uint64_t size;
    ASSERT_OK(env->GetFileSize(fname, &size));
    ASSERT_GT(size, 0);
#ifdef OS_WIN
    // Windows disk cache behaves differently. When we truncate
    // the original content is still in the cache due to the original
    // handle is still open. Generally, in Windows, one prohibits
    // shared access to files and it is not needed for WAL but we allow
    // it to induce corruption at various tests.
    test->Close();
#endif
    if (trunc) {
      ASSERT_OK(
          test::TruncateFile(env, fname, static_cast<uint64_t>(size * off)));
    } else {
      ASSERT_OK(test::CorruptFile(env, fname, static_cast<int>(size * off + 8),
                                  static_cast<int>(size * len), false));
    }
  }
};

TEST_F(DBWALTest, TrackAndVerifyWALsRecycleWAL) {
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  options.recycle_log_file_num = 1;
  options.track_and_verify_wals = true;

  DestroyAndReopen(options);

  ASSERT_OK(Put("key_ignore", "wal_to_recycle"));
  ASSERT_OK(Put("key_ignore1", "wal_to_recycle"));
  ASSERT_OK(Put("key_ignore2", "wal_to_recycle"));
  ASSERT_OK(Flush());

  ASSERT_OK(Put("key_ignore", "wal_to_recycle"));
  ASSERT_OK(Put("key_ignore1", "wal_to_recycle"));
  ASSERT_OK(Put("key_ignore2", "wal_to_recycle"));
  ASSERT_OK(Flush());

  // Stop background flush to avoid deleting any WAL
  options.env->SetBackgroundThreads(1, Env::HIGH);
  test::SleepingBackgroundTask sleeping_task;
  options.env->Schedule(&test::SleepingBackgroundTask::DoSleepTask,
                        &sleeping_task, Env::Priority::HIGH);

  // Recycle the first WAL
  ASSERT_OK(Put("key1", "old_value"));
  // Recycle the second WAL
  ASSERT_OK(dbfull()->TEST_SwitchWAL());
  ASSERT_OK(Put("key1", "new_value"));

  // Create a WAL hole on sequence number by truncating the first WAL to 0 byte
  VectorWalPtr log_files;
  ASSERT_OK(db_->GetSortedWalFiles(log_files));
  ASSERT_EQ(log_files.size(), 2);
  std::string log_name = LogFileName(dbname_, log_files.front()->LogNumber());
  Close();
  // Drop `Put("key1", "old_value")` in the first WAL
  ASSERT_OK(test::TruncateFile(options.env, log_name, 0 /* new_length */));

  {
    Status s = DB::Open(options, dbname_, &db_);

    ASSERT_OK(s);
  }

  ASSERT_EQ("wal_to_recycle", Get("key_ignore2"));
  ASSERT_EQ("NOT_FOUND", Get("key1"));

  Close();
}

class DBWALTrackAndVerifyWALsWithParamsTest
    : public DBWALTestBase,
      public ::testing::WithParamInterface<WALRecoveryMode> {
 public:
  DBWALTrackAndVerifyWALsWithParamsTest()
      : DBWALTestBase("/db_wal_track_and_verify_wals_with_params_test") {}
};

INSTANTIATE_TEST_CASE_P(
    DBWALTrackAndVerifyWALsWithParamsTest,
    DBWALTrackAndVerifyWALsWithParamsTest,
    ::testing::Values(WALRecoveryMode::kTolerateCorruptedTailRecords,
                      WALRecoveryMode::kAbsoluteConsistency,
                      WALRecoveryMode::kPointInTimeRecovery,
                      WALRecoveryMode::kSkipAnyCorruptedRecords));

TEST_P(DBWALTrackAndVerifyWALsWithParamsTest, Basic) {
  Options options = CurrentOptions();
  options.avoid_flush_during_shutdown = true;
  options.track_and_verify_wals = true;
  options.wal_recovery_mode = GetParam();

  // Stop background flush to avoid deleting any WAL
  options.env->SetBackgroundThreads(1, Env::HIGH);
  test::SleepingBackgroundTask sleeping_task;
  options.env->Schedule(&test::SleepingBackgroundTask::DoSleepTask,
                        &sleeping_task, Env::Priority::HIGH);

  for (int i = 0; i < 5; i++) {
    DestroyAndReopen(options);

    ASSERT_OK(Put("key1", "old_value"));
    SequenceNumber last_seqno_recorded_in_fist_wal =
        dbfull()->GetLatestSequenceNumber();

    ASSERT_OK(dbfull()->TEST_SwitchWAL());
    ASSERT_OK(Put("key1", "new_value"));

    VectorWalPtr log_files;
    ASSERT_OK(db_->GetSortedWalFiles(log_files));
    ASSERT_EQ(log_files.size(), 2);
    uint64_t first_log_number = log_files.front()->LogNumber();
    std::string first_log_name = LogFileName(dbname_, first_log_number);
    std::string second_log_name =
        LogFileName(dbname_, log_files.back()->LogNumber());

    if (i == 0) {
      // Delete the obsolete WAL and verify it will not be seen as a WAL hole
      sleeping_task.WakeUp();
      sleeping_task.WaitUntilDone();
      ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());
      // Stop background flush to avoid deleting any WAL
      sleeping_task.Reset();
      options.env->Schedule(&test::SleepingBackgroundTask::DoSleepTask,
                            &sleeping_task, Env::Priority::HIGH);
      Close();
    } else if (i == 1) {
      // Create a WAL hole on WAL number by deleting the first WAL and verify
      // the hole will be detected
      Close();
      ASSERT_OK(options.env->DeleteFile(first_log_name));
    } else if (i == 2) {
      // Create a WAL hole on sequence number by truncating the first WAL and
      // verify the hole will be detected
      Close();
      ASSERT_OK(
          test::TruncateFile(options.env, first_log_name, 0 /* new_length */));
    } else if (i == 3) {
      // Create a WAL hole on size difference by truncating the first WAL and
      // mocking a correct sequence number to force triggering corruption based
      // on size instead of sequence number and verify the hole will be detected
      Close();
      ASSERT_OK(
          test::TruncateFile(options.env, first_log_name, 0 /* new_length */));
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::UpdatePredecessorWALInfo", [&](void* arg) {
            std::pair<uint64_t, SequenceNumber*>* pair =
                static_cast<std::pair<uint64_t, SequenceNumber*>*>(arg);
            if (pair->first == first_log_number) {
              *(pair->second) = last_seqno_recorded_in_fist_wal;
            }
          });
      SyncPoint::GetInstance()->EnableProcessing();
    } else if (i == 4) {
      // Delete all wals and verify opening a DB with no WAL will be detected
      Close();
      ASSERT_OK(options.env->DeleteFile(first_log_name));
      ASSERT_OK(options.env->DeleteFile(second_log_name));
    }

    Status s;
    {
      s = DB::Open(options, dbname_, &db_);
    }

    if (i == 0) {
      ASSERT_OK(s);
      ASSERT_EQ("new_value", Get("key1"));
      continue;
    } else if (i == 3) {
      SyncPoint::GetInstance()->ClearAllCallBacks();
      SyncPoint::GetInstance()->DisableProcessing();
    } else if (i == 4) {
      ASSERT_TRUE(s.IsCorruption());
      ASSERT_TRUE(
          s.ToString().find("Opening an existing DB with no WAL files") !=
          std::string::npos);
      Close();
      continue;
    }

    if (options.wal_recovery_mode == WALRecoveryMode::kPointInTimeRecovery) {
      ASSERT_OK(s);
      ASSERT_EQ("NOT_FOUND", Get("key1"));
    } else if (options.wal_recovery_mode ==
                   WALRecoveryMode::kAbsoluteConsistency ||
               options.wal_recovery_mode ==
                   WALRecoveryMode::kTolerateCorruptedTailRecords) {
      ASSERT_TRUE(s.IsCorruption());
      std::string msg;
      if (i == 1) {
        msg = "Missing WAL";
      } else if (i == 2) {
        msg = "Mismatched last sequence number recorded in the WAL";
      } else if (i == 3) {
        msg = "Mismatched size of the WAL";
      }
      ASSERT_TRUE(s.ToString().find(msg) != std::string::npos);
    } else {
      ASSERT_OK(s);
      ASSERT_EQ("new_value", Get("key1"));
    }

    Close();
  }
}

class DBWALTestWithParams
    : public DBWALTestBase,
      public ::testing::WithParamInterface<
          std::tuple<bool, int, int, CompressionType, bool>> {
 public:
  DBWALTestWithParams() : DBWALTestBase("/db_wal_test_with_params") {}
};

INSTANTIATE_TEST_CASE_P(
    Wal, DBWALTestWithParams,
    ::testing::Combine(::testing::Bool(), ::testing::Range(0, 4, 1),
                       ::testing::Range(RecoveryTestHelper::kWALFileOffset,
                                        RecoveryTestHelper::kWALFileOffset +
                                            RecoveryTestHelper::kWALFilesCount,
                                        1),
                       ::testing::Values(CompressionType::kNoCompression,
                                         CompressionType::kZSTD),
                       ::testing::Bool()));

class DBWALTestWithParamsVaryingRecoveryMode
    : public DBWALTestBase,
      public ::testing::WithParamInterface<
          std::tuple<bool, int, int, WALRecoveryMode, CompressionType>> {
 public:
  DBWALTestWithParamsVaryingRecoveryMode()
      : DBWALTestBase("/db_wal_test_with_params_mode") {}
};

INSTANTIATE_TEST_CASE_P(
    Wal, DBWALTestWithParamsVaryingRecoveryMode,
    ::testing::Combine(
        ::testing::Bool(), ::testing::Range(0, 4, 1),
        ::testing::Range(RecoveryTestHelper::kWALFileOffset,
                         RecoveryTestHelper::kWALFileOffset +
                             RecoveryTestHelper::kWALFilesCount,
                         1),
        ::testing::Values(WALRecoveryMode::kTolerateCorruptedTailRecords,
                          WALRecoveryMode::kAbsoluteConsistency,
                          WALRecoveryMode::kPointInTimeRecovery,
                          WALRecoveryMode::kSkipAnyCorruptedRecords),
        ::testing::Values(CompressionType::kNoCompression,
                          CompressionType::kZSTD)));

// Test scope:
// - We expect to open the data store when there is incomplete trailing writes
// at the end of any of the logs
// - We do not expect to open the data store for corruption
TEST_P(DBWALTestWithParams, kTolerateCorruptedTailRecords) {
  bool trunc = std::get<0>(GetParam());  // Corruption style
  // Corruption offset position
  int corrupt_offset = std::get<1>(GetParam());
  int wal_file_id = std::get<2>(GetParam());  // WAL file

  // Fill data for testing
  Options options = CurrentOptions();
  options.track_and_verify_wals = std::get<4>(GetParam());
  const size_t row_count = RecoveryTestHelper::FillData(this, &options);
  // test checksum failure or parsing
  RecoveryTestHelper::CorruptWAL(this, options, corrupt_offset * .3,
                                 /*len%=*/.1, wal_file_id, trunc);

  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  if (trunc) {
    options.create_if_missing = false;
    ASSERT_OK(TryReopen(options));
    const size_t recovered_row_count = RecoveryTestHelper::GetData(this);
    ASSERT_TRUE(corrupt_offset == 0 || recovered_row_count > 0);
    ASSERT_LT(recovered_row_count, row_count);
  } else {
    ASSERT_NOK(TryReopen(options));
  }
}

// Test scope:
// We don't expect the data store to be opened if there is any corruption
// (leading, middle or trailing -- incomplete writes or corruption)
TEST_P(DBWALTestWithParams, kAbsoluteConsistency) {
  // Verify clean slate behavior
  Options options = CurrentOptions();
  options.track_and_verify_wals = std::get<4>(GetParam());
  const size_t row_count = RecoveryTestHelper::FillData(this, &options);
  options.create_if_missing = false;
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ(RecoveryTestHelper::GetData(this), row_count);

  bool trunc = std::get<0>(GetParam());  // Corruption style
  // Corruption offset position
  int corrupt_offset = std::get<1>(GetParam());
  int wal_file_id = std::get<2>(GetParam());  // WAL file
  // WAL compression type
  CompressionType compression_type = std::get<3>(GetParam());
  options.wal_compression = compression_type;

  if (trunc && corrupt_offset == 0) {
    return;
  }

  // fill with new date
  RecoveryTestHelper::FillData(this, &options);
  // corrupt the wal
  RecoveryTestHelper::CorruptWAL(this, options, corrupt_offset * .33,
                                 /*len%=*/.1, wal_file_id, trunc);
  // verify
  options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
  options.create_if_missing = false;
  ASSERT_NOK(TryReopen(options));
}

// Test scope:
// We don't expect the data store to be opened if there is any inconsistency
// between WAL and SST files
TEST_F(DBWALTest, kPointInTimeRecoveryCFConsistency) {
  Options options = CurrentOptions();
  options.avoid_flush_during_recovery = true;

  // Create DB with multiple column families.
  CreateAndReopenWithCF({"one", "two"}, options);
  ASSERT_OK(Put(1, "key1", "val1"));
  ASSERT_OK(Put(2, "key2", "val2"));

  // Record the offset at this point
  Env* env = options.env;
  uint64_t wal_file_id = dbfull()->TEST_LogfileNumber();
  std::string fname = LogFileName(dbname_, wal_file_id);
  uint64_t offset_to_corrupt;
  ASSERT_OK(env->GetFileSize(fname, &offset_to_corrupt));
  ASSERT_GT(offset_to_corrupt, 0);

  ASSERT_OK(Put(1, "key3", "val3"));
  // Corrupt WAL at location of key3
  ASSERT_OK(test::CorruptFile(env, fname, static_cast<int>(offset_to_corrupt),
                              4, false));
  ASSERT_OK(Put(2, "key4", "val4"));
  ASSERT_OK(Put(1, "key5", "val5"));
  ASSERT_OK(Flush(2));

  // PIT recovery & verify
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ASSERT_NOK(TryReopenWithColumnFamilies({"default", "one", "two"}, options));
}

TEST_F(DBWALTest, RaceInstallFlushResultsWithWalObsoletion) {
  Options options = CurrentOptions();
  options.env = env_;
  options.track_and_verify_wals_in_manifest = true;
  // The following make sure there are two bg flush threads.
  options.max_background_jobs = 8;

  DestroyAndReopen(options);

  const std::string cf1_name("cf1");
  CreateAndReopenWithCF({cf1_name}, options);
  assert(handles_.size() == 2);

  {
    dbfull()->TEST_LockMutex();
    ASSERT_LE(2, dbfull()->GetBGJobLimits().max_flushes);
    dbfull()->TEST_UnlockMutex();
  }

  ASSERT_OK(dbfull()->PauseBackgroundWork());

  ASSERT_OK(db_->Put(WriteOptions(), handles_[1], "foo", "value"));
  ASSERT_OK(db_->Put(WriteOptions(), "foo", "value"));

  ASSERT_OK(dbfull()->TEST_FlushMemTable(
      /*wait=*/false, /*allow_write_stall=*/true, handles_[1]));

  ASSERT_OK(db_->Put(WriteOptions(), "foo", "value"));

  ASSERT_OK(dbfull()->TEST_FlushMemTable(
      /*wait=*/false, /*allow_write_stall=*/true, handles_[0]));

  bool called = false;
  std::atomic<int> bg_flush_threads{0};
  std::atomic<bool> wal_synced{false};
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::BackgroundCallFlush:start", [&](void* /*arg*/) {
        int cur = bg_flush_threads.load();
        int desired = cur + 1;
        if (cur > 0 ||
            !bg_flush_threads.compare_exchange_strong(cur, desired)) {
          while (!wal_synced.load()) {
            // Wait until the other bg flush thread finishes committing WAL sync
            // operation to the MANIFEST.
          }
        }
      });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::FlushMemTableToOutputFile:CommitWal:1",
      [&](void* /*arg*/) { wal_synced.store(true); });
  // This callback will be called when the first bg flush thread reaches the
  // point before entering the MANIFEST write queue after flushing the SST
  // file.
  // The purpose of the sync points here is to ensure both bg flush threads
  // finish computing `min_wal_number_to_keep` before any of them updates the
  // `log_number` for the column family that's being flushed.
  SyncPoint::GetInstance()->SetCallBack(
      "MemTableList::TryInstallMemtableFlushResults:AfterComputeMinWalToKeep",
      [&](void* /*arg*/) {
        dbfull()->mutex()->AssertHeld();
        if (!called) {
          // We are the first bg flush thread in the MANIFEST write queue.
          // We set up the dependency between sync points for two threads that
          // will be executing the same code.
          // For the interleaving of events, see
          // https://github.com/facebook/rocksdb/pull/9715.
          // bg flush thread1 will release the db mutex while in the MANIFEST
          // write queue. In the meantime, bg flush thread2 locks db mutex and
          // computes the min_wal_number_to_keep (before thread1 writes to
          // MANIFEST thus before cf1->log_number is updated). Bg thread2 joins
          // the MANIFEST write queue afterwards and bg flush thread1 proceeds
          // with writing to MANIFEST.
          called = true;
          SyncPoint::GetInstance()->LoadDependency({
              {"VersionSet::LogAndApply:WriteManifestStart",
               "DBWALTest::RaceInstallFlushResultsWithWalObsoletion:BgFlush2"},
              {"DBWALTest::RaceInstallFlushResultsWithWalObsoletion:BgFlush2",
               "VersionSet::LogAndApply:WriteManifest"},
          });
        } else {
          // The other bg flush thread has already been in the MANIFEST write
          // queue, and we are after.
          TEST_SYNC_POINT(
              "DBWALTest::RaceInstallFlushResultsWithWalObsoletion:BgFlush2");
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();

  ASSERT_OK(dbfull()->ContinueBackgroundWork());

  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[0]));
  ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(handles_[1]));

  ASSERT_TRUE(called);

  Close();

  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();

  std::unique_ptr<DB> db1;
  Status s = DB::OpenForReadOnly(options, dbname_, &db1);
  ASSERT_OK(s);
  assert(db1);
}

TEST_F(DBWALTest, FixSyncWalOnObseletedWalWithNewManifestCausingMissingWAL) {
  Options options = CurrentOptions();
  // Small size to force manifest creation
  options.max_manifest_file_size = 1;
  options.max_manifest_space_amp_pct = 0;
  options.track_and_verify_wals_in_manifest = true;
  DestroyAndReopen(options);

  // Accumulate memtable m1 and create the 1st wal (i.e, 4.log)
  ASSERT_OK(Put(Key(1), ""));
  ASSERT_OK(Put(Key(2), ""));
  ASSERT_OK(Put(Key(3), ""));

  const std::string wal_file_path = db_->GetName() + "/000004.log";

  // Coerce the following sequence of events:
  // (1) Flush() marks 4.log to be obsoleted, 8.log to be the latest (i.e,
  // active) log and release the lock
  // (2) SyncWAL() proceeds with the lock. It
  // creates a new manifest and syncs all the inactive wals before the latest
  // (i.e, active log), which is 4.log. Note that SyncWAL() is not aware of the
  // fact that 4.log has marked as to be obseleted. Such wal
  // sync will then add a WAL addition record of 4.log to the new manifest
  // without any special treatment. Prior to the fix, there is no WAL deletion
  // record to offset it. (3) BackgroundFlush() will eventually purge 4.log.

  bool wal_synced = false;
  SyncPoint::GetInstance()->SetCallBack(
      "FindObsoleteFiles::PostMutexUnlock", [&](void*) {
        ASSERT_OK(env_->FileExists(wal_file_path));
        uint64_t pre_sync_wal_manifest_no =
            dbfull()->TEST_Current_Manifest_FileNo();
        ASSERT_OK(db_->SyncWAL());
        uint64_t post_sync_wal_manifest_no =
            dbfull()->TEST_Current_Manifest_FileNo();
        bool new_manifest_created =
            post_sync_wal_manifest_no == pre_sync_wal_manifest_no + 1;
        ASSERT_TRUE(new_manifest_created);
        wal_synced = true;
      });

  SyncPoint::GetInstance()->EnableProcessing();

  ASSERT_OK(Flush());
  ASSERT_OK(dbfull()->TEST_WaitForBackgroundWork());

  ASSERT_TRUE(wal_synced);
  // BackgroundFlush() purged 4.log
  // because the memtable associated with the WAL was flushed and new WAL was
  // created (i.e, 8.log)
  ASSERT_TRUE(env_->FileExists(wal_file_path).IsNotFound());

  SyncPoint::GetInstance()->ClearAllCallBacks();
  SyncPoint::GetInstance()->DisableProcessing();

  // To verify the corruption of "Missing WAL with log number: 4" under
  // `options.track_and_verify_wals_in_manifest = true` is fixed.
  //
  // Before the fix, `db_->SyncWAL()` will sync and record WAL addtion of the
  // obseleted WAL 4.log in a new manifest without any special treament.
  // This will result in missing-wal corruption in DB::Reopen().
  Status s = TryReopen(options);
  EXPECT_OK(s);
}

// Test scope:
// - We expect to open data store under all circumstances
// - We expect only data upto the point where the first error was encountered
TEST_P(DBWALTestWithParams, kPointInTimeRecovery) {
  const int maxkeys =
      RecoveryTestHelper::kWALFilesCount * RecoveryTestHelper::kKeysPerWALFile;

  bool trunc = std::get<0>(GetParam());  // Corruption style
  // Corruption offset position
  int corrupt_offset = std::get<1>(GetParam());
  int wal_file_id = std::get<2>(GetParam());  // WAL file
  // WAL compression type
  CompressionType compression_type = std::get<3>(GetParam());

  // Fill data for testing
  Options options = CurrentOptions();
  options.track_and_verify_wals = std::get<4>(GetParam());
  options.wal_compression = compression_type;
  const size_t row_count = RecoveryTestHelper::FillData(this, &options);

  // Corrupt the wal
  // The offset here was 0.3 which cuts off right at the end of a
  // valid fragment after wal zstd compression checksum is enabled,
  // so changed the value to 0.33.
  RecoveryTestHelper::CorruptWAL(this, options, corrupt_offset * .33,
                                 /*len%=*/.1, wal_file_id, trunc);

  // Verify
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  options.create_if_missing = false;
  ASSERT_OK(TryReopen(options));

  // Probe data for invariants
  size_t recovered_row_count = RecoveryTestHelper::GetData(this);
  ASSERT_LT(recovered_row_count, row_count);

  // Verify a prefix of keys were recovered. But not in the case of full WAL
  // truncation, because we have no way to know there was a corruption when
  // truncation happened on record boundaries (preventing recovery holes in
  // that case requires using `track_and_verify_wals_in_manifest`).
  if (!trunc || corrupt_offset != 0) {
    bool expect_data = true;
    for (size_t k = 0; k < maxkeys; ++k) {
      bool found = Get("key" + std::to_string(k)) != "NOT_FOUND";
      if (expect_data && !found) {
        expect_data = false;
      }
      ASSERT_EQ(found, expect_data);
    }
  }

  const size_t min = RecoveryTestHelper::kKeysPerWALFile *
                     (wal_file_id - RecoveryTestHelper::kWALFileOffset);
  ASSERT_GE(recovered_row_count, min);
  if (!trunc && corrupt_offset != 0) {
    const size_t max = RecoveryTestHelper::kKeysPerWALFile *
                       (wal_file_id - RecoveryTestHelper::kWALFileOffset + 1);
    ASSERT_LE(recovered_row_count, max);
  }
}

// Test scope:
// - We expect to open the data store under all scenarios
// - We expect to have recovered records past the corruption zone
TEST_P(DBWALTestWithParams, kSkipAnyCorruptedRecords) {
  bool trunc = std::get<0>(GetParam());  // Corruption style
  // Corruption offset position
  int corrupt_offset = std::get<1>(GetParam());
  int wal_file_id = std::get<2>(GetParam());  // WAL file
  // WAL compression type
  CompressionType compression_type = std::get<3>(GetParam());

  // Fill data for testing
  Options options = CurrentOptions();
  options.track_and_verify_wals = std::get<4>(GetParam());
  options.wal_compression = compression_type;
  const size_t row_count = RecoveryTestHelper::FillData(this, &options);

  // Corrupt the WAL
  RecoveryTestHelper::CorruptWAL(this, options, corrupt_offset * .3,
                                 /*len%=*/.1, wal_file_id, trunc);

  // Verify behavior
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  options.create_if_missing = false;
  ASSERT_OK(TryReopen(options));

  // Probe data for invariants
  size_t recovered_row_count = RecoveryTestHelper::GetData(this);
  ASSERT_LT(recovered_row_count, row_count);

  if (!trunc) {
    ASSERT_TRUE(corrupt_offset != 0 || recovered_row_count > 0);
  }
}

TEST_F(DBWALTest, AvoidFlushDuringRecovery) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.avoid_flush_during_recovery = false;

  // Test with flush after recovery.
  Reopen(options);
  ASSERT_OK(Put("foo", "v1"));
  ASSERT_OK(Put("bar", "v2"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo", "v3"));
  ASSERT_OK(Put("bar", "v4"));
  ASSERT_EQ(1, TotalTableFiles());
  // Reopen DB. Check if WAL logs flushed.
  Reopen(options);
  ASSERT_EQ("v3", Get("foo"));
  ASSERT_EQ("v4", Get("bar"));
  ASSERT_EQ(2, TotalTableFiles());

  // Test without flush after recovery.
  options.avoid_flush_during_recovery = true;
  DestroyAndReopen(options);
  ASSERT_OK(Put("foo", "v5"));
  ASSERT_OK(Put("bar", "v6"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo", "v7"));
  ASSERT_OK(Put("bar", "v8"));
  ASSERT_EQ(1, TotalTableFiles());
  // Reopen DB. WAL logs should not be flushed this time.
  Reopen(options);
  ASSERT_EQ("v7", Get("foo"));
  ASSERT_EQ("v8", Get("bar"));
  ASSERT_EQ(1, TotalTableFiles());

  // Force flush with allow_2pc.
  options.avoid_flush_during_recovery = true;
  options.allow_2pc = true;
  ASSERT_OK(Put("foo", "v9"));
  ASSERT_OK(Put("bar", "v10"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("foo", "v11"));
  ASSERT_OK(Put("bar", "v12"));
  Reopen(options);
  ASSERT_EQ("v11", Get("foo"));
  ASSERT_EQ("v12", Get("bar"));
  ASSERT_EQ(3, TotalTableFiles());
}

TEST_F(DBWALTest, WalCleanupAfterAvoidFlushDuringRecovery) {
  // Verifies WAL files that were present during recovery, but not flushed due
  // to avoid_flush_during_recovery, will be considered for deletion at a later
  // stage. We check at least one such file is deleted during Flush().
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.avoid_flush_during_recovery = true;
  Reopen(options);

  ASSERT_OK(Put("foo", "v1"));
  Reopen(options);
  for (int i = 0; i < 2; ++i) {
    if (i > 0) {
      // Flush() triggers deletion of obsolete tracked files
      ASSERT_OK(Flush());
    }
    VectorLogPtr log_files;
    ASSERT_OK(dbfull()->GetSortedWalFiles(log_files));
    if (i == 0) {
      ASSERT_GT(log_files.size(), 0);
    } else {
      ASSERT_EQ(0, log_files.size());
    }
  }
}

TEST_F(DBWALTest, RecoverWithoutFlush) {
  Options options = CurrentOptions();
  options.avoid_flush_during_recovery = true;
  options.create_if_missing = false;
  options.disable_auto_compactions = true;
  options.write_buffer_size = 64 * 1024 * 1024;

  size_t count = RecoveryTestHelper::FillData(this, &options);
  auto validateData = [this, count]() {
    for (size_t i = 0; i < count; i++) {
      ASSERT_NE(Get("key" + std::to_string(i)), "NOT_FOUND");
    }
  };
  Reopen(options);
  validateData();
  // Insert some data without flush
  ASSERT_OK(Put("foo", "foo_v1"));
  ASSERT_OK(Put("bar", "bar_v1"));
  Reopen(options);
  validateData();
  ASSERT_EQ(Get("foo"), "foo_v1");
  ASSERT_EQ(Get("bar"), "bar_v1");
  // Insert again and reopen
  ASSERT_OK(Put("foo", "foo_v2"));
  ASSERT_OK(Put("bar", "bar_v2"));
  Reopen(options);
  validateData();
  ASSERT_EQ(Get("foo"), "foo_v2");
  ASSERT_EQ(Get("bar"), "bar_v2");
  // manual flush and insert again
  ASSERT_OK(Flush());
  ASSERT_EQ(Get("foo"), "foo_v2");
  ASSERT_EQ(Get("bar"), "bar_v2");
  ASSERT_OK(Put("foo", "foo_v3"));
  ASSERT_OK(Put("bar", "bar_v3"));
  Reopen(options);
  validateData();
  ASSERT_EQ(Get("foo"), "foo_v3");
  ASSERT_EQ(Get("bar"), "bar_v3");
}

TEST_F(DBWALTest, RecoverWithoutFlushMultipleCF) {
  const std::string kSmallValue = "v";
  const std::string kLargeValue = DummyString(1024);
  Options options = CurrentOptions();
  options.avoid_flush_during_recovery = true;
  options.create_if_missing = false;
  options.disable_auto_compactions = true;

  auto countWalFiles = [this]() {
    VectorLogPtr log_files;
    if (!dbfull()->GetSortedWalFiles(log_files).ok()) {
      return size_t{0};
    }
    return log_files.size();
  };

  // Create DB with multiple column families and multiple log files.
  CreateAndReopenWithCF({"one", "two"}, options);
  ASSERT_OK(Put(0, "key1", kSmallValue));
  ASSERT_OK(Put(1, "key2", kLargeValue));
  ASSERT_OK(Flush(1));
  ASSERT_EQ(1, countWalFiles());
  ASSERT_OK(Put(0, "key3", kSmallValue));
  ASSERT_OK(Put(2, "key4", kLargeValue));
  ASSERT_OK(Flush(2));
  ASSERT_EQ(2, countWalFiles());

  // Reopen, insert and flush.
  options.db_write_buffer_size = 64 * 1024 * 1024;
  ReopenWithColumnFamilies({"default", "one", "two"}, options);
  ASSERT_EQ(Get(0, "key1"), kSmallValue);
  ASSERT_EQ(Get(1, "key2"), kLargeValue);
  ASSERT_EQ(Get(0, "key3"), kSmallValue);
  ASSERT_EQ(Get(2, "key4"), kLargeValue);
  // Insert more data.
  ASSERT_OK(Put(0, "key5", kLargeValue));
  ASSERT_OK(Put(1, "key6", kLargeValue));
  ASSERT_EQ(3, countWalFiles());
  ASSERT_OK(Flush(1));
  ASSERT_OK(Put(2, "key7", kLargeValue));
  ASSERT_OK(dbfull()->FlushWAL(false));
  ASSERT_EQ(4, countWalFiles());

  // Reopen twice and validate.
  for (int i = 0; i < 2; i++) {
    ReopenWithColumnFamilies({"default", "one", "two"}, options);
    ASSERT_EQ(Get(0, "key1"), kSmallValue);
    ASSERT_EQ(Get(1, "key2"), kLargeValue);
    ASSERT_EQ(Get(0, "key3"), kSmallValue);
    ASSERT_EQ(Get(2, "key4"), kLargeValue);
    ASSERT_EQ(Get(0, "key5"), kLargeValue);
    ASSERT_EQ(Get(1, "key6"), kLargeValue);
    ASSERT_EQ(Get(2, "key7"), kLargeValue);
    ASSERT_EQ(4, countWalFiles());
  }
}

// In this test we are trying to do the following:
//   1. Create a DB with corrupted WAL log;
//   2. Open with avoid_flush_during_recovery = true;
//   3. Append more data without flushing, which creates new WAL log.
//   4. Open again. See if it can correctly handle previous corruption.
TEST_P(DBWALTestWithParamsVaryingRecoveryMode,
       RecoverFromCorruptedWALWithoutFlush) {
  const int kAppendKeys = 100;
  Options options = CurrentOptions();
  options.avoid_flush_during_recovery = true;
  options.create_if_missing = false;
  options.disable_auto_compactions = true;
  options.write_buffer_size = 64 * 1024 * 1024;

  auto getAll = [this]() {
    std::vector<std::pair<std::string, std::string>> data;
    ReadOptions ropt;
    Iterator* iter = dbfull()->NewIterator(ropt);
    for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
      data.emplace_back(iter->key().ToString(), iter->value().ToString());
    }
    EXPECT_OK(iter->status());
    delete iter;
    return data;
  };

  bool trunc = std::get<0>(GetParam());  // Corruption style
  // Corruption offset position
  int corrupt_offset = std::get<1>(GetParam());
  int wal_file_id = std::get<2>(GetParam());  // WAL file
  WALRecoveryMode recovery_mode = std::get<3>(GetParam());
  // WAL compression type
  CompressionType compression_type = std::get<4>(GetParam());

  options.wal_recovery_mode = recovery_mode;
  options.wal_compression = compression_type;
  // Create corrupted WAL
  RecoveryTestHelper::FillData(this, &options);
  RecoveryTestHelper::CorruptWAL(this, options, corrupt_offset * .3,
                                 /*len%=*/.1, wal_file_id, trunc);
  // Skip the test if DB won't open.
  if (!TryReopen(options).ok()) {
    ASSERT_TRUE(options.wal_recovery_mode ==
                    WALRecoveryMode::kAbsoluteConsistency ||
                (!trunc && options.wal_recovery_mode ==
                               WALRecoveryMode::kTolerateCorruptedTailRecords));
    return;
  }
  ASSERT_OK(TryReopen(options));
  // Append some more data.
  for (int k = 0; k < kAppendKeys; k++) {
    std::string key = "extra_key" + std::to_string(k);
    std::string value = DummyString(RecoveryTestHelper::kValueSize);
    ASSERT_OK(Put(key, value));
  }
  // Save data for comparison.
  auto data = getAll();
  // Reopen. Verify data.
  ASSERT_OK(TryReopen(options));
  auto actual_data = getAll();
  ASSERT_EQ(data, actual_data);
}

// Tests that total log size is recovered if we set
// avoid_flush_during_recovery=true.
// Flush should trigger if max_total_wal_size is reached.
TEST_F(DBWALTest, RestoreTotalLogSizeAfterRecoverWithoutFlush) {
  auto test_listener = std::make_shared<FlushCounterListener>();
  test_listener->expected_flush_reason = FlushReason::kWalFull;

  constexpr size_t kKB = 1024;
  constexpr size_t kMB = 1024 * 1024;
  Options options = CurrentOptions();
  options.avoid_flush_during_recovery = true;
  options.max_total_wal_size = 1 * kMB;
  options.listeners.push_back(test_listener);
  // Have to open DB in multi-CF mode to trigger flush when
  // max_total_wal_size is reached.
  CreateAndReopenWithCF({"one"}, options);
  // Write some keys and we will end up with one log file which is slightly
  // smaller than 1MB.
  std::string value_100k(100 * kKB, 'v');
  std::string value_300k(300 * kKB, 'v');
  ASSERT_OK(Put(0, "foo", "v1"));
  for (int i = 0; i < 9; i++) {
    ASSERT_OK(Put(1, "key" + std::to_string(i), value_100k));
  }
  // Get log files before reopen.
  VectorLogPtr log_files_before;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_before));
  ASSERT_EQ(1, log_files_before.size());
  uint64_t log_size_before = log_files_before[0]->SizeFileBytes();
  ASSERT_GT(log_size_before, 900 * kKB);
  ASSERT_LT(log_size_before, 1 * kMB);
  ReopenWithColumnFamilies({"default", "one"}, options);
  // Write one more value to make log larger than 1MB.
  ASSERT_OK(Put(1, "bar", value_300k));
  // Get log files again. A new log file will be opened.
  VectorLogPtr log_files_after_reopen;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_after_reopen));
  ASSERT_EQ(2, log_files_after_reopen.size());
  ASSERT_EQ(log_files_before[0]->LogNumber(),
            log_files_after_reopen[0]->LogNumber());
  ASSERT_GT(log_files_after_reopen[0]->SizeFileBytes() +
                log_files_after_reopen[1]->SizeFileBytes(),
            1 * kMB);
  // Write one more key to trigger flush.
  ASSERT_OK(Put(0, "foo", "v2"));
  for (auto* h : handles_) {
    ASSERT_OK(dbfull()->TEST_WaitForFlushMemTable(h));
  }
  // Flushed two column families.
  ASSERT_EQ(2, test_listener->count.load());
}

#if defined(ROCKSDB_PLATFORM_POSIX)
#if defined(ROCKSDB_FALLOCATE_PRESENT)
// Tests that we will truncate the preallocated space of the last log from
// previous.
TEST_F(DBWALTest, TruncateLastLogAfterRecoverWithoutFlush) {
  constexpr size_t kKB = 1024;
  Options options = CurrentOptions();
  options.env = env_;
  options.avoid_flush_during_recovery = true;
  if (mem_env_) {
    ROCKSDB_GTEST_SKIP("Test requires non-mem environment");
    return;
  }
  if (!IsFallocateSupported()) {
    return;
  }

  DestroyAndReopen(options);
  size_t preallocated_size =
      dbfull()->TEST_GetWalPreallocateBlockSize(options.write_buffer_size);
  ASSERT_OK(Put("foo", "v1"));
  VectorLogPtr log_files_before;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_before));
  ASSERT_EQ(1, log_files_before.size());
  auto& file_before = log_files_before[0];
  ASSERT_LT(file_before->SizeFileBytes(), 1 * kKB);
  // The log file has preallocated space.
  {
    std::string fname = dbname_ + file_before->PathName();
    uint64_t allocated = GetAllocatedFileSize(fname);
    if (ShouldSkipAllocationCheck(fname)) {
      fprintf(stderr,
              "Skipping preallocation check on this filesystem for %s\n",
              fname.c_str());
    } else if (allocated < preallocated_size) {
      fprintf(stderr,
              "Warning: allocated size (%lu) less than preallocated size (%lu) "
              "for %s, skipping check. This may indicate the filesystem does "
              "not support preallocation or does not report it.\n",
              allocated, preallocated_size, fname.c_str());
    } else {
      ASSERT_GE(allocated, preallocated_size);
    }
  }
  Reopen(options);
  VectorLogPtr log_files_after;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_after));
  ASSERT_EQ(1, log_files_after.size());
  ASSERT_LT(log_files_after[0]->SizeFileBytes(), 1 * kKB);
  // The preallocated space should be truncated.
  {
    std::string fname = dbname_ + file_before->PathName();
    if (!ShouldSkipAllocationCheck(fname)) {
      ASSERT_LT(GetAllocatedFileSize(fname), preallocated_size);
    }
  }
}
// Tests that we will truncate the preallocated space of the last log from
// previous.
TEST_F(DBWALTest, TruncateLastLogAfterRecoverWithFlush) {
  constexpr size_t kKB = 1024;
  Options options = CurrentOptions();
  options.env = env_;
  options.avoid_flush_during_recovery = false;
  options.avoid_flush_during_shutdown = true;
  if (mem_env_) {
    ROCKSDB_GTEST_SKIP("Test requires non-mem environment");
    return;
  }
  if (!IsFallocateSupported()) {
    return;
  }

  DestroyAndReopen(options);
  size_t preallocated_size =
      dbfull()->TEST_GetWalPreallocateBlockSize(options.write_buffer_size);
  ASSERT_OK(Put("foo", "v1"));
  VectorLogPtr log_files_before;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_before));
  ASSERT_EQ(1, log_files_before.size());
  auto& file_before = log_files_before[0];
  ASSERT_LT(file_before->SizeFileBytes(), 1 * kKB);
  {
    std::string fname = dbname_ + file_before->PathName();
    uint64_t allocated = GetAllocatedFileSize(fname);
    if (ShouldSkipAllocationCheck(fname)) {
      fprintf(stderr,
              "Skipping preallocation check on this filesystem for %s\n",
              fname.c_str());
    } else if (allocated < preallocated_size) {
      fprintf(stderr,
              "Warning: allocated size (%lu) less than preallocated size (%lu) "
              "for %s, skipping check. This may indicate the filesystem does "
              "not support preallocation or does not report it.\n",
              allocated, preallocated_size, fname.c_str());
    } else {
      ASSERT_GE(allocated, preallocated_size);
    }
  }
  // The log file has preallocated space.
  Close();

  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBImpl::PurgeObsoleteFiles:Begin",
        "DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterRecover"},
       {"DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterTruncate",
        "DBImpl::DeleteObsoleteFileImpl::BeforeDeletion"}});
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  port::Thread reopen_thread([&]() { Reopen(options); });

  TEST_SYNC_POINT(
      "DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterRecover");
  // After the flush during Open, the log file should get deleted.  However,
  // if  the process is in a crash loop, the log file may not get
  // deleted and thte preallocated space will keep accumulating. So we need
  // to ensure it gets trtuncated.
  {
    std::string fname = dbname_ + file_before->PathName();
    if (!ShouldSkipAllocationCheck(fname)) {
      EXPECT_LT(GetAllocatedFileSize(fname), preallocated_size);
    }
  }
  TEST_SYNC_POINT(
      "DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterTruncate");
  reopen_thread.join();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
}

TEST_F(DBWALTest, TruncateLastLogAfterRecoverWALEmpty) {
  Options options = CurrentOptions();
  options.env = env_;
  options.avoid_flush_during_recovery = false;
  if (mem_env_ || encrypted_env_) {
    ROCKSDB_GTEST_SKIP("Test requires non-mem/non-encrypted  environment");
    return;
  }
  if (!IsFallocateSupported()) {
    return;
  }

  DestroyAndReopen(options);
  size_t preallocated_size =
      dbfull()->TEST_GetWalPreallocateBlockSize(options.write_buffer_size);
  Close();
  std::vector<std::string> filenames;
  std::string last_log;
  uint64_t last_log_num = 0;
  ASSERT_OK(env_->GetChildren(dbname_, &filenames));
  for (const auto& fname : filenames) {
    uint64_t number;
    FileType type;
    if (ParseFileName(fname, &number, &type, nullptr)) {
      if (type == kWalFile && number > last_log_num) {
        last_log = fname;
      }
    }
  }
  ASSERT_NE(last_log, "");
  last_log = dbname_ + '/' + last_log;
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->LoadDependency(
      {{"DBImpl::PurgeObsoleteFiles:Begin",
        "DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterRecover"},
       {"DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterTruncate",
        "DBImpl::DeleteObsoleteFileImpl::BeforeDeletion"}});
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->SetCallBack(
      "PosixWritableFile::Close",
      [](void* arg) { *(static_cast<size_t*>(arg)) = 0; });
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->EnableProcessing();
  // Preallocate space for the empty log file. This could happen if WAL data
  // was buffered in memory and the process crashed.
  std::unique_ptr<WritableFile> log_file;
  ASSERT_OK(env_->ReopenWritableFile(last_log, &log_file, EnvOptions()));
  log_file->SetPreallocationBlockSize(preallocated_size);
  log_file->PrepareWrite(0, 4096);
  log_file.reset();

  {
    uint64_t allocated = GetAllocatedFileSize(last_log);
    if (ShouldSkipAllocationCheck(last_log)) {
      fprintf(stderr,
              "Skipping preallocation check on this filesystem for %s\n",
              last_log.c_str());
    } else if (allocated < preallocated_size) {
      fprintf(stderr,
              "Warning: allocated size (%lu) less than preallocated size (%lu) "
              "for %s, skipping check. This may indicate the filesystem does "
              "not support preallocation or does not report it.\n",
              allocated, preallocated_size, last_log.c_str());
    } else {
      ASSERT_GE(allocated, preallocated_size);
    }
  }

  port::Thread reopen_thread([&]() { Reopen(options); });

  TEST_SYNC_POINT(
      "DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterRecover");
  // The preallocated space should be truncated.
  if (!ShouldSkipAllocationCheck(last_log)) {
    EXPECT_LT(GetAllocatedFileSize(last_log), preallocated_size);
  }
  TEST_SYNC_POINT(
      "DBWALTest::TruncateLastLogAfterRecoverWithFlush:AfterTruncate");
  reopen_thread.join();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->DisableProcessing();
  ROCKSDB_NAMESPACE::SyncPoint::GetInstance()->ClearAllCallBacks();
}

TEST_F(DBWALTest, ReadOnlyRecoveryNoTruncate) {
  constexpr size_t kKB = 1024;
  Options options = CurrentOptions();
  options.env = env_;
  options.avoid_flush_during_recovery = true;
  if (mem_env_) {
    ROCKSDB_GTEST_SKIP("Test requires non-mem environment");
    return;
  }
  if (!IsFallocateSupported()) {
    return;
  }

  // create DB and close with file truncate disabled
  std::atomic_bool enable_truncate{false};

  SyncPoint::GetInstance()->SetCallBack("PosixWritableFile::Close",
                                        [&](void* arg) {
                                          if (!enable_truncate) {
                                            *(static_cast<size_t*>(arg)) = 0;
                                          }
                                        });
  SyncPoint::GetInstance()->EnableProcessing();

  DestroyAndReopen(options);
  size_t preallocated_size =
      dbfull()->TEST_GetWalPreallocateBlockSize(options.write_buffer_size);
  ASSERT_OK(Put("foo", "v1"));
  VectorLogPtr log_files_before;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_before));
  ASSERT_EQ(1, log_files_before.size());
  auto& file_before = log_files_before[0];
  ASSERT_LT(file_before->SizeFileBytes(), 1 * kKB);
  // The log file has preallocated space.
  std::string fname = dbname_ + file_before->PathName();
  auto db_size = GetAllocatedFileSize(fname);
  {
    if (ShouldSkipAllocationCheck(fname)) {
      fprintf(stderr,
              "Skipping preallocation check on this filesystem for %s\n",
              fname.c_str());
    } else if (db_size < preallocated_size) {
      fprintf(stderr,
              "Warning: allocated size (%lu) less than preallocated size (%lu) "
              "for %s, skipping check. This may indicate the filesystem does "
              "not support preallocation or does not report it.\n",
              db_size, preallocated_size, fname.c_str());
    } else {
      ASSERT_GE(db_size, preallocated_size);
    }
  }
  Close();

  // enable truncate and open DB as readonly, the file should not be truncated
  // and DB size is not changed.
  enable_truncate = true;
  ASSERT_OK(ReadOnlyReopen(options));
  VectorLogPtr log_files_after;
  ASSERT_OK(dbfull()->GetSortedWalFiles(log_files_after));
  ASSERT_EQ(1, log_files_after.size());
  ASSERT_LT(log_files_after[0]->SizeFileBytes(), 1 * kKB);
  ASSERT_EQ(log_files_after[0]->PathName(), file_before->PathName());
  // The preallocated space should NOT be truncated.
  // the DB size is almost the same.
  {
    std::string fname2 = dbname_ + file_before->PathName();
    if (!ShouldSkipAllocationCheck(fname2)) {
      ASSERT_NEAR(GetAllocatedFileSize(fname2), db_size, db_size / 100);
    }
  }
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
}
#endif  // ROCKSDB_FALLOCATE_PRESENT
#endif  // ROCKSDB_PLATFORM_POSIX

// Reports a configurable preallocation capability and counts
// ReopenWritableFile() calls on WAL files.
class PreallocationTestFs : public FileSystemWrapper {
 public:
  PreallocationTestFs(const std::shared_ptr<FileSystem>& target,
                      bool preallocation_supported)
      : FileSystemWrapper(target),
        preallocation_supported_(preallocation_supported) {}

  const char* Name() const override { return "PreallocationTestFs"; }

  bool IsPreallocationSupported() override { return preallocation_supported_; }

  IOStatus ReopenWritableFile(const std::string& fname,
                              const FileOptions& file_opts,
                              std::unique_ptr<FSWritableFile>* result,
                              IODebugContext* dbg) override {
    uint64_t number;
    FileType type;
    if (ParseFileName(fname.substr(fname.find_last_of('/') + 1), &number,
                      &type) &&
        type == kWalFile) {
      wal_reopen_count_.FetchAddRelaxed(1);
    }
    return target()->ReopenWritableFile(fname, file_opts, result, dbg);
  }

  void SetPreallocationSupported(bool supported) {
    preallocation_supported_ = supported;
  }

  int GetWalReopenCount() const { return wal_reopen_count_.LoadRelaxed(); }

  void ResetWalReopenCount() { wal_reopen_count_.StoreRelaxed(0); }

 private:
  bool preallocation_supported_;
  RelaxedAtomic<int> wal_reopen_count_{0};
};

// Tests that the last WAL is reopened for truncation during recovery only if
// the FileSystem reports that it supports preallocation.
TEST_F(DBWALTest, SkipWalTruncateIfPreallocationNotSupported) {
  // The Env must outlive the loop since DestroyAndReopen() destroys the
  // previous DB using the previous iteration's options.
  auto test_fs = std::make_shared<PreallocationTestFs>(
      env_->GetFileSystem(), /*preallocation_supported=*/true);
  std::unique_ptr<Env> test_env(NewCompositeEnv(test_fs));
  for (bool preallocation_supported : {true, false}) {
    // avoid_flush_during_recovery = true restores the WAL as alive, while
    // false flushes the recovered data and leaves the WAL obsolete.
    for (bool avoid_flush_during_recovery : {true, false}) {
      SCOPED_TRACE(
          "preallocation_supported=" + std::to_string(preallocation_supported) +
          " avoid_flush_during_recovery=" +
          std::to_string(avoid_flush_during_recovery));
      Options options = CurrentOptions();
      options.env = test_env.get();
      options.avoid_flush_during_recovery = avoid_flush_during_recovery;
      options.avoid_flush_during_shutdown = true;
      test_fs->SetPreallocationSupported(preallocation_supported);
      DestroyAndReopen(options);
      ASSERT_OK(Put("foo", "v1"));
      Close();

      test_fs->ResetWalReopenCount();
      Reopen(options);
      ASSERT_EQ(preallocation_supported ? 1 : 0, test_fs->GetWalReopenCount());
      ASSERT_EQ("v1", Get("foo"));
      Close();
    }
  }
}

// Tests that a read-only open never reopens a WAL for truncation, even if the
// FileSystem supports preallocation.
TEST_F(DBWALTest, ReadOnlyOpenSkipsWalTruncate) {
  Options options = CurrentOptions();
  auto test_fs = std::make_shared<PreallocationTestFs>(
      env_->GetFileSystem(), /*preallocation_supported=*/true);
  std::unique_ptr<Env> test_env(NewCompositeEnv(test_fs));
  options.env = test_env.get();
  options.avoid_flush_during_shutdown = true;
  DestroyAndReopen(options);
  ASSERT_OK(Put("foo", "v1"));
  Close();

  ASSERT_OK(ReadOnlyReopen(options));
  ASSERT_EQ("v1", Get("foo"));
  Close();
  ASSERT_EQ(0, test_fs->GetWalReopenCount());

  // The same WAL is truncated by a read-write open.
  Reopen(options);
  ASSERT_EQ(1, test_fs->GetWalReopenCount());
  Close();
}

TEST_F(DBWALTest, WalInManifestButNotInSortedWals) {
  Options options = CurrentOptions();
  options.track_and_verify_wals_in_manifest = true;
  options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;

  // Build a way to make wal files selectively go missing
  bool wals_go_missing = false;
  struct MissingWalFs : public FileSystemWrapper {
    MissingWalFs(const std::shared_ptr<FileSystem>& t,
                 bool* _wals_go_missing_flag)
        : FileSystemWrapper(t), wals_go_missing_flag(_wals_go_missing_flag) {}
    bool* wals_go_missing_flag;
    IOStatus GetChildren(const std::string& dir, const IOOptions& io_opts,
                         std::vector<std::string>* r,
                         IODebugContext* dbg) override {
      IOStatus s = target_->GetChildren(dir, io_opts, r, dbg);
      if (s.ok() && *wals_go_missing_flag) {
        for (size_t i = 0; i < r->size();) {
          if (EndsWith(r->at(i), ".log")) {
            r->erase(r->begin() + i);
          } else {
            ++i;
          }
        }
      }
      return s;
    }
    const char* Name() const override { return "MissingWalFs"; }
  };
  auto my_fs =
      std::make_shared<MissingWalFs>(env_->GetFileSystem(), &wals_go_missing);
  std::unique_ptr<Env> my_env(NewCompositeEnv(my_fs));
  options.env = my_env.get();

  CreateAndReopenWithCF({"blah"}, options);

  // Currently necessary to get a WAL tracked in manifest; see
  // https://github.com/facebook/rocksdb/issues/10080
  ASSERT_OK(Put(0, "x", "y"));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Put(1, "x", "y"));
  ASSERT_OK(db_->SyncWAL());
  ASSERT_OK(Flush(1));

  ASSERT_FALSE(dbfull()->GetVersionSet()->GetWalSet().GetWals().empty());
  std::vector<std::unique_ptr<LogFile>> wals;
  ASSERT_OK(db_->GetSortedWalFiles(wals));
  wals_go_missing = true;
  ASSERT_NOK(db_->GetSortedWalFiles(wals));
  wals_go_missing = false;
  Close();
}

TEST_F(DBWALTest, WalTermTest) {
  Options options = CurrentOptions();
  options.env = env_;
  CreateAndReopenWithCF({"pikachu"}, options);

  ASSERT_OK(Put(1, "foo", "bar"));

  WriteOptions wo;
  wo.sync = true;
  wo.disableWAL = false;

  WriteBatch batch;
  ASSERT_OK(batch.Put("foo", "bar"));
  batch.MarkWalTerminationPoint();
  ASSERT_OK(batch.Put("foo2", "bar2"));

  ASSERT_OK(dbfull()->Write(wo, &batch));

  // make sure we can re-open it.
  ASSERT_OK(TryReopenWithColumnFamilies({"default", "pikachu"}, options));
  ASSERT_EQ("bar", Get(1, "foo"));
  ASSERT_EQ("NOT_FOUND", Get(1, "foo2"));
}

TEST_F(DBWALTest, GetCompressedWalsAfterSync) {
  if (db_->GetOptions().wal_compression == kNoCompression) {
    ROCKSDB_GTEST_BYPASS("stream compression not present");
    return;
  }
  Options options = GetDefaultOptions();
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  options.create_if_missing = true;
  options.env = env_;
  options.avoid_flush_during_recovery = true;
  options.track_and_verify_wals_in_manifest = true;
  // Enable WAL compression so that the newly-created WAL will be non-empty
  // after DB open, even if point-in-time WAL recovery encounters no
  // corruption.
  options.wal_compression = kZSTD;
  DestroyAndReopen(options);

  // Write something to memtable and WAL so that wal_empty_ will be false after
  // next DB::Open().
  ASSERT_OK(Put("a", "v"));

  Reopen(options);

  // New WAL is created, thanks to !wal_empty_.
  ASSERT_OK(dbfull()->TEST_SwitchWAL());

  ASSERT_OK(Put("b", "v"));

  ASSERT_OK(db_->SyncWAL());

  VectorLogPtr wals;
  Status s = dbfull()->GetSortedWalFiles(wals);
  ASSERT_OK(s);
}

TEST_F(DBWALTest, EmptyWalReopenTest) {
  Options options = CurrentOptions();
  options.env = env_;
  CreateAndReopenWithCF({"pikachu"}, options);

  // make sure we can re-open it.
  ASSERT_OK(TryReopenWithColumnFamilies({"default", "pikachu"}, options));

  {
    std::vector<std::string> files;
    int num_wal_files = 0;
    ASSERT_OK(env_->GetChildren(dbname_, &files));
    for (const auto& file : files) {
      uint64_t number = 0;
      FileType type = kWalFile;
      if (ParseFileName(file, &number, &type) && type == kWalFile) {
        num_wal_files++;
      }
    }

    ASSERT_EQ(num_wal_files, 1);
  }
}

TEST_F(DBWALTest, AsyncWalPrecreateWaitsForPendingFile) {
  Options options = CurrentOptions();
  options.env = env_;
  options.statistics = CreateDBStatistics();
  options.async_wal_precreate = true;
  options.recycle_log_file_num = 0;
  options.write_buffer_size = 1024;
  options.max_write_buffer_number = 4;

  // Goal: cover the race where foreground WAL rotation reaches the switch
  // before the background precreate publishes its result. The sync points hold
  // the async worker after it has created the file but before it marks it
  // ready, then verify the foreground writer waits instead of allocating a
  // higher WAL number.
  SyncPoint::GetInstance()->LoadDependency({
      {"DBImpl::BGWorkAsyncWALPrecreate:BeforePublish",
       "DBWALTest::AsyncWalPrecreateWaitsForPendingFile:WorkerBlocked"},
      {"DBWALTest::AsyncWalPrecreateWaitsForPendingFile:AllowPublish",
       "DBImpl::BGWorkAsyncWALPrecreate:Publish"},
      {"DBImpl::WaitForAsyncWALPrecreate:BeforeWait",
       "DBWALTest::AsyncWalPrecreateWaitsForPendingFile:ForegroundWaiting"},
  });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  Reopen(options);
  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateWaitsForPendingFile:WorkerBlocked");

  ASSERT_OK(Put("fill", "value"));
  Status writer_status;
  port::Thread writer(
      [&]() { writer_status = dbfull()->TEST_SwitchMemtable(); });

  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateWaitsForPendingFile:ForegroundWaiting");
  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateWaitsForPendingFile:AllowPublish");
  writer.join();
  ASSERT_OK(writer_status);
  ASSERT_OK(Put("trigger", "value"));
  ASSERT_EQ("value", Get("trigger"));
  ASSERT_EQ(1, options.statistics->getTickerCount(WAL_PRECREATE_HIT));
  ASSERT_EQ(1, options.statistics->getTickerCount(WAL_PRECREATE_WAITED));
  ASSERT_EQ(0, options.statistics->getTickerCount(WAL_PRECREATE_MISS));
}

TEST_F(DBWALTest, AsyncWalPrecreateConsumesReadyFile) {
  Options options = CurrentOptions();
  options.env = env_;
  options.statistics = CreateDBStatistics();
  options.async_wal_precreate = true;
  options.recycle_log_file_num = 0;
  options.write_buffer_size = 1024;
  options.max_write_buffer_number = 4;

  // Goal: exercise the no-wait consumption path. The background task publishes
  // a prepared future WAL before the foreground memtable switch, and the switch
  // must install exactly that reserved WAL number instead of allocating
  // another.
  SyncPoint::GetInstance()->LoadDependency({
      {"DBImpl::BGWorkAsyncWALPrecreate:Done",
       "DBWALTest::AsyncWalPrecreateConsumesReadyFile:PrecreateDone"},
  });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  Reopen(options);
  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateConsumesReadyFile:PrecreateDone");

  const std::vector<uint64_t> wal_numbers = ListWalNumbers();
  ASSERT_GE(wal_numbers.size(), 2);
  const uint64_t prepared_wal_number = wal_numbers.back();
  ASSERT_NE(prepared_wal_number, dbfull()->TEST_GetCurrentLogNumber());

  ASSERT_OK(Put("fill", "value"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_EQ(prepared_wal_number, dbfull()->TEST_GetCurrentLogNumber());
  ASSERT_EQ(1, options.statistics->getTickerCount(WAL_PRECREATE_HIT));
  ASSERT_EQ(0, options.statistics->getTickerCount(WAL_PRECREATE_MISS));
  ASSERT_EQ(0, options.statistics->getTickerCount(WAL_PRECREATE_WAITED));
}

TEST_F(DBWALTest, AsyncWalPrecreateRecoveryToleratesActualFutureWal) {
  Options options = CurrentOptions();
  options.env = env_;
  options.async_wal_precreate = true;
  options.recycle_log_file_num = 0;
  options.track_and_verify_wals = true;
  options.track_and_verify_wals_in_manifest = true;
  options.avoid_flush_during_shutdown = true;
  options.avoid_flush_during_recovery = true;

  // Goal: cover the actual async precreation path, not a hand-forged empty WAL.
  // The background task opens the future WAL, clean close releases the
  // unpublished writer and leaves the empty future WAL behind, and recovery
  // must tolerate that file while replaying the preceding WAL with real user
  // data.
  SyncPoint::GetInstance()->LoadDependency({
      {"DBImpl::BGWorkAsyncWALPrecreate:Done",
       "DBWALTest::AsyncWalPrecreateRecoveryToleratesActualFutureWal:"
       "PrecreateDone"},
  });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  DestroyAndReopen(options);
  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateRecoveryToleratesActualFutureWal:"
      "PrecreateDone");

  const std::vector<uint64_t> wal_numbers = ListWalNumbers();
  ASSERT_GE(wal_numbers.size(), 2);
  const uint64_t future_wal_number = wal_numbers.back();
  uint64_t future_wal_size = 0;
  ASSERT_OK(env_->GetFileSize(LogFileName(dbname_, future_wal_number),
                              &future_wal_size));
  ASSERT_EQ(0, future_wal_size);

  ASSERT_OK(Put("key", "value"));
  Close();

  ASSERT_OK(TryReopen(options));
  ASSERT_EQ("value", Get("key"));
  ASSERT_OK(Put("after-recovery", "value2"));
  ASSERT_EQ("value2", Get("after-recovery"));
}

TEST_F(DBWALTest, AsyncWalPrecreateDeletesFileOnStartFailure) {
  Options options = CurrentOptions();
  options.env = env_;
  options.statistics = CreateDBStatistics();
  options.async_wal_precreate = true;
  options.recycle_log_file_num = 0;
  options.write_buffer_size = 1024;
  options.max_write_buffer_number = 4;

  // Goal: force failure after consuming a prepared WAL but before it becomes a
  // logical WAL. The failed, unpublished file must be deleted and its writer
  // destroyed outside mutex_.
  SyncPoint::GetInstance()->LoadDependency({
      {"DBImpl::BGWorkAsyncWALPrecreate:Done",
       "DBWALTest::AsyncWalPrecreateDeletesFileOnStartFailure:PrecreateDone"},
  });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  Reopen(options);
  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateDeletesFileOnStartFailure:PrecreateDone");
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::StartWALFile:AfterCompressionTypeRecord", [](void* arg) {
        *static_cast<IOStatus*>(arg) =
            IOStatus::IOError("injected async WAL start failure");
      });
  const std::vector<uint64_t> wal_numbers = ListWalNumbers();
  ASSERT_GE(wal_numbers.size(), 2);
  const uint64_t prepared_wal_number = wal_numbers.back();

  ASSERT_OK(Put("fill", "value"));
  ASSERT_NOK(dbfull()->TEST_SwitchMemtable());
  ASSERT_TRUE(
      env_->FileExists(LogFileName(dbname_, prepared_wal_number)).IsNotFound());
  ASSERT_EQ(1, options.statistics->getTickerCount(WAL_PRECREATE_HIT));
}

TEST_F(DBWALTest, AsyncWalPrecreateFailureFallsBackToSyncCreation) {
  Options options = CurrentOptions();
  options.env = env_;
  options.statistics = CreateDBStatistics();
  options.async_wal_precreate = true;
  options.recycle_log_file_num = 0;
  options.write_buffer_size = 1024;
  options.max_write_buffer_number = 4;

  // Goal: fail the background precreate file open after normal DB open has
  // created the current WAL. The foreground switch should observe no prepared
  // WAL, create one synchronously, and expose both failure and miss counters.
  SyncPoint::GetInstance()->LoadDependency({
      {"DBImpl::BGWorkAsyncWALPrecreate:Done",
       "DBWALTest::AsyncWalPrecreateFailureFallsBackToSyncCreation:"
       "PrecreateDone"},
  });
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::Open:AfterDeleteFiles",
      [&](void*) { env_->non_writable_count_ = 1; });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([&] {
    env_->non_writable_count_ = 0;
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  Reopen(options);
  TEST_SYNC_POINT(
      "DBWALTest::AsyncWalPrecreateFailureFallsBackToSyncCreation:"
      "PrecreateDone");
  ASSERT_EQ(1, options.statistics->getTickerCount(WAL_PRECREATE_FAILED));

  ASSERT_OK(Put("fill", "value"));
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_EQ(1, options.statistics->getTickerCount(WAL_PRECREATE_MISS));
}

TEST_F(DBWALTest, ReadOnlyErrorIfWalFileExistsIgnoresEmptyWals) {
  Options options = CurrentOptions();
  options.env = env_;
  DestroyAndReopen(options);
  Close();

  // Goal: preserve error_if_wal_file_exists for WALs containing data while
  // allowing empty future WALs left by async precreation. This creates empty
  // WAL files directly so the test isolates the read-only open contract.
  const std::vector<uint64_t> existing_wal_numbers = ListWalNumbers();
  const uint64_t empty_wal_number =
      existing_wal_numbers.empty() ? 1 : existing_wal_numbers.back() + 1;
  CreateEmptyWal(empty_wal_number);

  DBOptions db_options(options);
  std::vector<ColumnFamilyDescriptor> column_families;
  column_families.emplace_back(kDefaultColumnFamilyName,
                               ColumnFamilyOptions(options));
  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> read_only_db;
  ASSERT_OK(DB::OpenForReadOnly(db_options, dbname_, column_families, &handles,
                                &read_only_db,
                                true /* error_if_wal_file_exists */));
  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(read_only_db->DestroyColumnFamilyHandle(handle));
  }
  read_only_db.reset();
  handles.clear();

  CreateWalWithContents(empty_wal_number + 1, Slice("not empty"));
  ASSERT_NOK(DB::OpenForReadOnly(db_options, dbname_, column_families, &handles,
                                 &read_only_db,
                                 true /* error_if_wal_file_exists */));
}

TEST_F(DBWALTest, RecoveryToleratesEmptyFutureWal) {
  Options options = CurrentOptions();
  options.env = env_;
  options.track_and_verify_wals = true;
  options.track_and_verify_wals_in_manifest = true;
  options.avoid_flush_during_recovery = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("key", "value"));
  Close();

  const std::vector<uint64_t> wal_numbers = ListWalNumbers();
  ASSERT_FALSE(wal_numbers.empty());
  const uint64_t future_wal_number = wal_numbers.back() + 1;
  CreateEmptyWal(future_wal_number);

  // Goal: model a crash after async WAL precreation but before consumption.
  // Recovery should process the zero-byte future WAL, mark its file number
  // used, and keep replaying the real preceding WAL without reporting a WAL
  // chain or MANIFEST-tracking error.
  ASSERT_OK(TryReopen(options));
  ASSERT_EQ("value", Get("key"));
  ASSERT_OK(Put("after-recovery", "value2"));
  ASSERT_EQ("value2", Get("after-recovery"));
}

TEST_F(DBWALTest, RecoveryFlushSwitchWALOnEmptyMemtable) {
  Options options = CurrentOptions();
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(FileSystem::Default());
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  options.env = fault_fs_env.get();
  options.avoid_flush_during_shutdown = true;
  DestroyAndReopen(options);

  // Make sure the memtable switch in recovery flush happened after test checks
  // the memtable is empty.
  SyncPoint::GetInstance()->LoadDependency(
      {{"DBWALTest.RecoveryFlushSwitchWALOnEmptyMemtable:"
        "AfterCheckMemtableEmpty",
        "RecoverFromRetryableBGIOError:BeforeStart"}});
  SyncPoint::GetInstance()->EnableProcessing();
  fault_fs->SetThreadLocalErrorContext(
      FaultInjectionIOType::kMetadataWrite, 7 /* seed*/, 1 /* one_in */,
      true /* retryable */, false /* has_data_loss*/);
  fault_fs->EnableThreadLocalErrorInjection(
      FaultInjectionIOType::kMetadataWrite);

  WriteOptions wo;
  wo.sync = true;
  Status s = Put("k", "old_v", wo);
  ASSERT_TRUE(s.IsIOError());
  // To verify the key is not in memtable nor SST
  ASSERT_TRUE(static_cast<ColumnFamilyHandleImpl*>(db_->DefaultColumnFamily())
                  ->cfd()
                  ->mem()
                  ->IsEmpty());
  ASSERT_EQ("NOT_FOUND", Get("k"));
  TEST_SYNC_POINT(
      "DBWALTest.RecoveryFlushSwitchWALOnEmptyMemtable:"
      "AfterCheckMemtableEmpty");
  SyncPoint::GetInstance()->DisableProcessing();

  fault_fs->DisableThreadLocalErrorInjection(
      FaultInjectionIOType::kMetadataWrite);

  // Keep trying write until recovery of the previous IO error finishes
  while (!s.ok()) {
    options.env->SleepForMicroseconds(1000);
    s = Put("k", "new_v");
  }

  // If recovery flush didn't switch WAL, we will end up having two duplicate
  // WAL entries with same seqno and same key that violate assertion during WAL
  // recovery and fail DB reopen
  options.avoid_flush_during_recovery = false;
  Reopen(options);

  ASSERT_EQ("new_v", Get("k"));
  Destroy(options);
}

TEST_F(DBWALTest, WALWriteErrorNoRecovery) {
  Options options = CurrentOptions();
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(FileSystem::Default());
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  options.env = fault_fs_env.get();
  options.manual_wal_flush = true;
  DestroyAndReopen(options);
  fault_fs->SetThreadLocalErrorContext(
      FaultInjectionIOType::kWrite, 7 /* seed*/, 1 /* one_in */,
      true /* retryable */, false /* has_data_loss*/);
  fault_fs->EnableThreadLocalErrorInjection(FaultInjectionIOType::kWrite);

  ASSERT_OK(Put("k", "v"));
  Status s;
  s = db_->FlushWAL(false);
  ASSERT_TRUE(s.IsIOError());
  s = dbfull()->TEST_GetBGError();
  ASSERT_EQ(s.severity(), Status::Severity::kFatalError);
  ASSERT_FALSE(dbfull()->TEST_IsRecoveryInProgress());
  fault_fs->DisableThreadLocalErrorInjection(FaultInjectionIOType::kWrite);
  Destroy(options);
}

TEST_F(DBWALTest, PartitionWALUsageOpenValidation) {
  Options options = CurrentOptions();
  options.create_if_missing = true;

  for (PartitionWALUsage usage :
       {PartitionWALUsage::kNone,
        PartitionWALUsage::kPartitionByColumnFamily}) {
    options.partition_wal_usage = usage;
    ASSERT_OK(TryReopen(options));
  }
}

// DB-level coverage for `DBOptions::partition_wal_usage`: what DBImpl actually
// writes to the WAL and that those WALs replay through real recovery.
class DBWALIndexTest : public DBWALTestBase {
 public:
  DBWALIndexTest() : DBWALTestBase("/db_wal_index_test") {}

 protected:
  struct WALRecordMetadata {
    SequenceNumber sequence;
    uint64_t wal_index;
  };

  static constexpr int kNumKeys = 16;

  Options WALIndexOptions(PartitionWALUsage usage) {
    Options options = CurrentOptions();
    options.partition_wal_usage = usage;
    // Keep the written WAL on disk across Close()/Reopen() so it can be both
    // inspected and replayed.
    options.avoid_flush_during_shutdown = true;
    options.avoid_flush_during_recovery = true;
    return options;
  }

  void WriteKeys() {
    for (int i = 0; i < kNumKeys; i++) {
      ASSERT_OK(Put(Key(i), "value" + std::to_string(i)));
    }
  }

  // Writes through a protected WriteBatch so that a corrupted log entry is
  // caught by WriteBatch::VerifyChecksum inside WriteToWAL -- a failure that
  // happens before any append and so leaves the WAL writer usable.
  Status WriteProtectedBatch(const std::string& key, const std::string& value) {
    WriteBatch batch(/*reserved_bytes=*/0, /*max_bytes=*/0,
                     /*protection_bytes_per_key=*/8, /*default_cf_ts_sz=*/0);
    Status s = batch.Put(key, value);
    if (!s.ok()) {
      return s;
    }
    return db_->Write(WriteOptions(), &batch);
  }

  void VerifyKeys() {
    for (int i = 0; i < kNumKeys; i++) {
      ASSERT_EQ("value" + std::to_string(i), Get(Key(i)));
    }
  }

  uint64_t LatestWALNumber() {
    const std::vector<uint64_t> wal_numbers = ListWalNumbers();
    EXPECT_FALSE(wal_numbers.empty());
    return wal_numbers.empty() ? 0 : wal_numbers.back();
  }

  std::string WALPath(uint64_t wal_number) {
    return LogFileName(dbname_, wal_number);
  }

  struct PhysicalRecord {
    uint8_t type;
    size_t header_offset;
    size_t header_size;
    size_t payload_size;

    size_t payload_offset() const { return header_offset + header_size; }
  };

  static std::vector<PhysicalRecord> ScanPhysicalRecords(
      const std::string& contents) {
    std::vector<PhysicalRecord> records;
    size_t offset = 0;
    while (offset + log::kHeaderSize <= contents.size()) {
      const size_t block_remaining =
          log::kBlockSize - (offset % log::kBlockSize);
      if (block_remaining < static_cast<size_t>(log::kHeaderSize)) {
        offset += block_remaining;
        continue;
      }
      const uint8_t type = static_cast<uint8_t>(contents[offset + 6]);
      if (type == log::kZeroType) {
        break;
      }
      const size_t payload_size =
          static_cast<uint8_t>(contents[offset + 4]) |
          (static_cast<size_t>(static_cast<uint8_t>(contents[offset + 5]))
           << 8);
      const size_t header_size = log::IsRecyclableRecordType(type)
                                     ? log::kRecyclableHeaderSize
                                     : log::kHeaderSize;
      records.push_back({type, offset, header_size, payload_size});
      offset += header_size + payload_size;
    }
    return records;
  }

  std::string ReadWALContents(const std::string& wal_path) {
    std::string contents;
    EXPECT_OK(ReadFileToString(env_, wal_path, &contents));
    return contents;
  }

  void WriteWALContents(const std::string& wal_path,
                        const std::string& contents) {
    std::unique_ptr<WritableFile> file;
    ASSERT_OK(env_->NewWritableFile(wal_path, &file, EnvOptions()));
    ASSERT_OK(file->Append(contents));
    ASSERT_OK(file->Close());
  }

  static void FixRecordChecksum(const PhysicalRecord& record,
                                std::string* contents) {
    const uint32_t crc =
        crc32c::Value(&(*contents)[record.header_offset + 6],
                      record.header_size - 6 + record.payload_size);
    EncodeFixed32(&(*contents)[record.header_offset], crc32c::Mask(crc));
  }

  std::optional<PhysicalRecord> FindLastDataRecord(
      const std::string& contents) {
    std::optional<PhysicalRecord> found;
    for (const PhysicalRecord& record : ScanPhysicalRecords(contents)) {
      if (log::IsWALIndexRecordType(record.type)) {
        found = record;
      }
    }
    return found;
  }

  void CorruptLastDataRecordPayload(const std::string& wal_path) {
    std::string contents = ReadWALContents(wal_path);
    const std::optional<PhysicalRecord> last = FindLastDataRecord(contents);
    ASSERT_TRUE(last.has_value());
    ASSERT_GT(last->payload_size, 2U);
    contents[last->payload_offset() + 2] ^= 0xff;
    WriteWALContents(wal_path, contents);
  }

  void TruncateLastDataRecord(const std::string& wal_path) {
    std::string contents = ReadWALContents(wal_path);
    const std::optional<PhysicalRecord> last = FindLastDataRecord(contents);
    ASSERT_TRUE(last.has_value());
    ASSERT_GT(last->payload_size, 0U);
    contents.resize(last->payload_offset() + last->payload_size - 1);
    WriteWALContents(wal_path, contents);
  }

  void ShiftWALSequencesBackOne(const std::string& wal_path) {
    std::string contents = ReadWALContents(wal_path);
    for (const PhysicalRecord& record : ScanPhysicalRecords(contents)) {
      if (!log::IsWALIndexRecordType(record.type)) {
        continue;
      }
      ASSERT_GE(record.payload_size,
                static_cast<size_t>(log::kWALIndexSize + sizeof(uint64_t)));
      char* sequence = &contents[record.payload_offset() + log::kWALIndexSize];
      const uint64_t original = DecodeFixed64(sequence);
      ASSERT_GT(original, 0U);
      EncodeFixed64(sequence, original - 1);
      FixRecordChecksum(record, &contents);
    }
    WriteWALContents(wal_path, contents);
  }

  void OverwriteLastDataRecordWALIndex(const std::string& wal_path,
                                       uint64_t wal_index) {
    std::string contents = ReadWALContents(wal_path);
    const std::optional<PhysicalRecord> last = FindLastDataRecord(contents);
    ASSERT_TRUE(last.has_value());
    ASSERT_GE(last->payload_size, static_cast<size_t>(log::kWALIndexSize));
    EncodeFixed64(&contents[last->payload_offset()], wal_index);
    FixRecordChecksum(*last, &contents);
    WriteWALContents(wal_path, contents);
  }

  static std::optional<PhysicalRecord> FindDataRecordByOrdinal(
      const std::string& contents, size_t ordinal) {
    size_t seen = 0;
    for (const PhysicalRecord& record : ScanPhysicalRecords(contents)) {
      if (!log::IsWALIndexRecordType(record.type)) {
        continue;
      }
      if (seen == ordinal) {
        return record;
      }
      ++seen;
    }
    return std::nullopt;
  }

  // Rewrites the wal_index prefix of the `ordinal`-th data record (0-based),
  // fixing its checksum so the record stays readable at its new position.
  void OverwriteDataRecordWALIndexAt(const std::string& wal_path,
                                     size_t ordinal, uint64_t wal_index) {
    std::string contents = ReadWALContents(wal_path);
    const std::optional<PhysicalRecord> record =
        FindDataRecordByOrdinal(contents, ordinal);
    ASSERT_TRUE(record.has_value());
    ASSERT_GE(record->payload_size, static_cast<size_t>(log::kWALIndexSize));
    EncodeFixed64(&contents[record->payload_offset()], wal_index);
    FixRecordChecksum(*record, &contents);
    WriteWALContents(wal_path, contents);
  }

  // Rewrites the type byte of the `ordinal`-th data record (0-based) to
  // an unknown type with a fixed checksum. Unlike a checksum failure (which
  // drops the rest of the block), the reader reports this record and reads
  // on, so the loss is bounded on both sides.
  void CorruptDataRecordTypeAt(const std::string& wal_path, size_t ordinal) {
    std::string contents = ReadWALContents(wal_path);
    const std::optional<PhysicalRecord> record =
        FindDataRecordByOrdinal(contents, ordinal);
    ASSERT_TRUE(record.has_value());
    contents[record->header_offset + 6] = static_cast<char>(50);
    FixRecordChecksum(*record, &contents);
    WriteWALContents(wal_path, contents);
  }

  // Flips a payload byte of the `ordinal`-th data record (0-based) without
  // fixing its checksum, so the reader reports it and skips to the next
  // record.
  void CorruptDataRecordPayloadAt(const std::string& wal_path, size_t ordinal) {
    std::string contents = ReadWALContents(wal_path);
    const std::optional<PhysicalRecord> record =
        FindDataRecordByOrdinal(contents, ordinal);
    ASSERT_TRUE(record.has_value());
    ASSERT_GT(record->payload_size, 2U);
    contents[record->payload_offset() + 2] ^= 0xff;
    WriteWALContents(wal_path, contents);
  }

  // Adds `delta` to the wal_index prefix of every data record, fixing
  // checksums. Sequences stay put, so the merge witness still sees them in
  // order while the file starts wherever the test wants.
  void ShiftWALIndices(const std::string& wal_path, uint64_t delta) {
    std::string contents = ReadWALContents(wal_path);
    size_t shifted = 0;
    for (const PhysicalRecord& record : ScanPhysicalRecords(contents)) {
      if (!log::IsWALIndexRecordType(record.type)) {
        continue;
      }
      ASSERT_GE(record.payload_size, static_cast<size_t>(log::kWALIndexSize));
      char* prefix = &contents[record.payload_offset()];
      EncodeFixed64(prefix, DecodeFixed64(prefix) + delta);
      FixRecordChecksum(record, &contents);
      ++shifted;
    }
    ASSERT_GT(shifted, 0U);
    WriteWALContents(wal_path, contents);
  }

  // Walks the physical block layout of `wal_path`, calling `visit` with each
  // record's type byte and payload. Goes below log::Reader on purpose: the
  // reader consumes WAL-index metadata records without exposing them.
  template <typename Visitor>
  void ForEachPhysicalRecord(const std::string& wal_path, Visitor visit) {
    const std::string contents = ReadWALContents(wal_path);
    for (const PhysicalRecord& record : ScanPhysicalRecords(contents)) {
      visit(record.type, Slice(contents.data() + record.payload_offset(),
                               record.payload_size));
    }
  }

  // Physical record type bytes in `wal_path`, in file order.
  std::vector<uint8_t> ReadWALRecordTypes(const std::string& wal_path) {
    std::vector<uint8_t> types;
    ForEachPhysicalRecord(wal_path, [&types](uint8_t type, Slice /*payload*/) {
      types.push_back(type);
    });
    return types;
  }

  // Ranges declared by void records in `wal_path`, in file order. Each is the
  // closed range of wal_index values that record covers.
  std::vector<std::pair<uint64_t, uint64_t>> ReadWALVoidRanges(
      const std::string& wal_path) {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ForEachPhysicalRecord(wal_path, [&ranges](uint8_t type, Slice payload) {
      if (type != log::kWALIndexVoidType &&
          type != log::kRecyclableWALIndexVoidType) {
        return;
      }
      EXPECT_EQ(log::kWALIndexVoidPayloadSize, payload.size());
      if (payload.size() < log::kWALIndexVoidPayloadSize) {
        return;
      }
      ranges.emplace_back(DecodeFixed64(payload.data()),
                          DecodeFixed64(payload.data() + log::kWALIndexSize));
    });
    return ranges;
  }

  std::vector<std::pair<uint64_t, uint64_t>> ReadWALSupersessions(
      const std::string& wal_path) {
    std::vector<std::pair<uint64_t, uint64_t>> supersessions;
    ForEachPhysicalRecord(
        wal_path, [&supersessions](uint8_t type, Slice payload) {
          if (type != log::kWALIndexSupersessionType &&
              type != log::kRecyclableWALIndexSupersessionType) {
            return;
          }
          EXPECT_EQ(log::kWALIndexSupersessionPayloadSize, payload.size());
          if (payload.size() != log::kWALIndexSupersessionPayloadSize) {
            return;
          }
          supersessions.emplace_back(
              DecodeFixed64(payload.data()),
              DecodeFixed64(payload.data() + log::kWALIndexSize));
        });
    return supersessions;
  }

  // wal_index of every logical record in `wal_path`, in replay order. Zero for
  // a record that was written without an index.
  // A reader over `wal_path` that reads as `wal_number`, so recyclable
  // records and supersession records resolve against the right file.
  std::unique_ptr<log::Reader> OpenWALReader(const std::string& wal_path,
                                             uint64_t wal_number) {
    std::unique_ptr<FSSequentialFile> file;
    EXPECT_OK(env_->GetFileSystem()->NewSequentialFile(wal_path, FileOptions(),
                                                       &file, /*dbg=*/nullptr));
    std::unique_ptr<SequentialFileReader> file_reader(
        new SequentialFileReader(std::move(file), wal_path));
    return std::make_unique<log::Reader>(
        /*info_log=*/nullptr, std::move(file_reader), /*reporter=*/nullptr,
        /*checksum=*/true, wal_number);
  }

  std::vector<uint64_t> ReadWALIndices(const std::string& wal_path,
                                       uint64_t wal_number) {
    std::unique_ptr<log::Reader> reader = OpenWALReader(wal_path, wal_number);
    std::vector<uint64_t> wal_indices;
    std::string scratch;
    Slice record;
    while (reader->ReadRecord(&record, &scratch)) {
      wal_indices.push_back(reader->GetLastReadWALIndex());
    }
    return wal_indices;
  }

  std::vector<WALRecordMetadata> ReadWALRecordMetadata(
      const std::string& wal_path, uint64_t wal_number) {
    std::unique_ptr<log::Reader> reader = OpenWALReader(wal_path, wal_number);
    std::vector<WALRecordMetadata> metadata;
    std::string scratch;
    Slice record;
    while (reader->ReadRecord(&record, &scratch)) {
      WriteBatch batch;
      const Status status = WriteBatchInternal::SetContents(&batch, record);
      EXPECT_OK(status);
      if (!status.ok()) {
        continue;
      }
      metadata.push_back({WriteBatchInternal::Sequence(&batch),
                          reader->GetLastReadWALIndex()});
    }
    return metadata;
  }

  void WriteKeyRange(int start, int count) {
    for (int i = 0; i < count; ++i) {
      ASSERT_OK(Put(Key(start + i), "value" + std::to_string(start + i)));
    }
  }

  // Two WALs where the first is damaged at its tail. Point-in-time recovery
  // stops inside the first and never reaches the second, so both files end up
  // with an obsolete suffix while the second one stays intact on disk.
  void SetupPointInTimeSkippedSecondWAL(const Options& options,
                                        uint64_t* first_wal,
                                        uint64_t* second_wal) {
    assert(first_wal != nullptr);
    assert(second_wal != nullptr);
    DestroyAndReopen(options);
    WriteKeyRange(0, 6);
    *first_wal = LatestWALNumber();
    Close();
    Reopen(options);
    WriteKeyRange(6, 4);
    *second_wal = LatestWALNumber();
    ASSERT_NE(*first_wal, *second_wal);
    Close();
    CorruptLastDataRecordPayload(WALPath(*first_wal));
  }

  // Three WALs: keys 0-3 at indices 1-4, key 4 alone at index 5, then keys
  // 10 and 11 at 6 and 7. Returns the WAL holding key 4.
  uint64_t SetupLoneRecordBetweenWALs(const Options& options) {
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    Close();
    Reopen(options);
    WriteKeyRange(4, 1);
    const uint64_t lone_wal = LatestWALNumber();
    Close();
    Reopen(options);
    WriteKeyRange(10, 2);
    EXPECT_NE(lone_wal, LatestWALNumber());
    Close();
    return lone_wal;
  }

  // Opens a byte copy of the DB directory with partition_wal_usage=kNone, so a
  // test can hold merged recovery to the sequential path's outcome on the same
  // files. The copy is destroyed before returning.
  Status OpenSequentialCopy(Options options,
                            const std::vector<std::string>& cfs = {}) {
    const std::string copy_dir = dbname_ + "_sequential_copy";
    DestroyDB(copy_dir, options).PermitUncheckedError();
    EXPECT_OK(env_->CreateDirIfMissing(copy_dir));
    std::vector<std::string> children;
    EXPECT_OK(env_->GetChildren(dbname_, &children));
    for (const std::string& child : children) {
      std::string contents;
      if (ReadFileToString(env_, dbname_ + "/" + child, &contents).ok()) {
        EXPECT_OK(WriteStringToFile(env_, contents, copy_dir + "/" + child));
      }
    }
    options.partition_wal_usage = PartitionWALUsage::kNone;
    options.create_if_missing = false;
    std::vector<ColumnFamilyDescriptor> descriptors = {
        {kDefaultColumnFamilyName, options}};
    for (const std::string& cf : cfs) {
      descriptors.emplace_back(cf, options);
    }
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    const Status status =
        DB::Open(options, copy_dir, descriptors, &handles, &db);
    for (ColumnFamilyHandle* handle : handles) {
      EXPECT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    db.reset();
    EXPECT_OK(DestroyDB(copy_dir, options));
    return status;
  }

  // Approximates a crash under WAL recycling: a recycled WAL keeps the bytes
  // of its previous use past what its new incarnation wrote.
  static void KeepRecycledWALTails(bool keep) {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    if (keep) {
      SyncPoint::GetInstance()->SetCallBack(
          "PosixWritableFile::Close",
          [](void* arg) { *static_cast<size_t*>(arg) = 0; });
      SyncPoint::GetInstance()->EnableProcessing();
    }
  }

  // Log number in a recyclable record's header.
  static uint32_t RecyclableRecordLogNumber(const std::string& contents,
                                            const PhysicalRecord& record) {
    return DecodeFixed32(contents.data() + record.header_offset + 7);
  }
};

// Stops recovery after a configured number of records for WAL-index cut tests.
class StopReplayAfterRecordsWalFilter : public WalFilter {
 public:
  explicit StopReplayAfterRecordsWalFilter(int stop_after)
      : stop_after_(stop_after) {}

  static const char* kClassName() { return "StopReplayAfterRecordsWalFilter"; }
  const char* Name() const override { return kClassName(); }

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

// Drops every record of one WAL and stops replay at the first record of a
// later WAL. The records are ignored, or with `report_corrupted` reported
// corrupted, which arms a point-in-time stop instead.
class IgnoreWALWalFilter : public WalFilter {
 public:
  IgnoreWALWalFilter(uint64_t dropped_wal, bool report_corrupted)
      : dropped_wal_(dropped_wal), report_corrupted_(report_corrupted) {}

  const char* Name() const override { return "IgnoreWALWalFilter"; }

  WalProcessingOption LogRecordFound(unsigned long long log_number,
                                     const std::string& /*log_file_name*/,
                                     const WriteBatch& /*batch*/,
                                     WriteBatch* /*new_batch*/,
                                     bool* /*batch_changed*/) override {
    if (log_number < dropped_wal_) {
      return WalProcessingOption::kContinueProcessing;
    }
    if (log_number > dropped_wal_) {
      return report_corrupted_ ? WalProcessingOption::kContinueProcessing
                               : WalProcessingOption::kStopReplay;
    }
    return report_corrupted_ ? WalProcessingOption::kCorruptedRecord
                             : WalProcessingOption::kIgnoreCurrentRecord;
  }

 private:
  const uint64_t dropped_wal_;
  const bool report_corrupted_;
};

// Makes one WAL unreadable while leaving directory enumeration unchanged.
class FailWALOpenFileSystem : public FileSystemWrapper {
 public:
  FailWALOpenFileSystem(const std::shared_ptr<FileSystem>& base,
                        std::string blocked_path)
      : FileSystemWrapper(base), blocked_path_(std::move(blocked_path)) {}

  static const char* kClassName() { return "FailWALOpenFileSystem"; }
  const char* Name() const override { return kClassName(); }

  IOStatus NewSequentialFile(const std::string& fname,
                             const FileOptions& file_opts,
                             std::unique_ptr<FSSequentialFile>* result,
                             IODebugContext* dbg) override {
    if (fname == blocked_path_) {
      return IOStatus::IOError("injected WAL open failure", fname);
    }
    return FileSystemWrapper::NewSequentialFile(fname, file_opts, result, dbg);
  }

 private:
  const std::string blocked_path_;
};

// Counts recovery opens of one WAL without changing the underlying I/O.
class CountingWALOpenFileSystem : public FileSystemWrapper {
 public:
  CountingWALOpenFileSystem(const std::shared_ptr<FileSystem>& base,
                            std::string counted_path)
      : FileSystemWrapper(base), counted_path_(std::move(counted_path)) {}

  static const char* kClassName() { return "CountingWALOpenFileSystem"; }
  const char* Name() const override { return kClassName(); }

  IOStatus NewSequentialFile(const std::string& fname,
                             const FileOptions& file_opts,
                             std::unique_ptr<FSSequentialFile>* result,
                             IODebugContext* dbg) override {
    if (fname == counted_path_) {
      ++open_count_;
    }
    return FileSystemWrapper::NewSequentialFile(fname, file_opts, result, dbg);
  }

  int open_count() const { return open_count_; }

 private:
  const std::string counted_path_;
  int open_count_ = 0;
};

// Captures every recovery log line with the level it was emitted at, so a
// test can assert both that a message appeared at the intended level and that
// it did not also appear at a different one. Recording the level alongside
// the text is what lets one helper serve the routing assertions as well as
// the operator-facing warnings.
class RecordingLogger : public Logger {
 public:
  using Logger::Logv;
  void Logv(InfoLogLevel log_level, const char* format, va_list ap) override {
    char buf[4096];
    vsnprintf(buf, sizeof(buf), format, ap);
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back({log_level, buf});
  }

  int count(InfoLogLevel level, const std::string& substr) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(
        std::count_if(entries_.begin(), entries_.end(), [&](const Entry& e) {
          return e.level == level &&
                 e.message.find(substr) != std::string::npos;
        }));
  }

  bool saw(InfoLogLevel level, const std::string& substr) const {
    return count(level, substr) > 0;
  }

 private:
  struct Entry {
    InfoLogLevel level;
    std::string message;
  };

  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
};

TEST_F(DBWALIndexTest, TwoWriteQueuesIsRejected) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.two_write_queues = true;
  const Status s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
}

// The M1.4 gates at one indexed partition (the table above
// DBImpl::ValidateOptions): each option opens, and every key written comes
// back from an indexed WAL after a clean reopen.
class DBWALIndexOnePartitionGateTest : public DBWALIndexTest {
 protected:
  void ExpectIndexedWALRoundTrip(const Options& options) {
    ASSERT_EQ(1U, options.wal_partitions);
    DestroyAndReopen(options);
    WriteKeys();
    if (options.manual_wal_flush) {
      ASSERT_OK(db_->FlushWAL(/*sync=*/false));
    }
    const uint64_t wal_number = LatestWALNumber();
    Close();
    const std::vector<uint8_t> types = ReadWALRecordTypes(WALPath(wal_number));
    ASSERT_FALSE(types.empty());
    EXPECT_TRUE(types.front() == log::kWALIndexMarkerType ||
                types.front() == log::kRecyclableWALIndexMarkerType)
        << static_cast<int>(types.front());
    Reopen(options);
    VerifyKeys();
    Close();
    Destroy(options);
  }
};

TEST_F(DBWALIndexOnePartitionGateTest, ManualWALFlush) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.manual_wal_flush = true;
  ExpectIndexedWALRoundTrip(options);
}

// The keys land in a recycled WAL: the flushes free the first WAL, and the
// WAL switch at the second flush reuses it.
TEST_F(DBWALIndexOnePartitionGateTest, RecycleLogFileNum) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  options.recycle_log_file_num = 1;
  DestroyAndReopen(options);
  std::atomic<int> reused{0};
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CreateWAL:BeforeReuseWritableFile1",
      [&](void* /*arg*/) { reused.fetch_add(1); });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Put("before0", "value"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("before1", "value"));
  ASSERT_OK(Flush());
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_GT(reused.load(), 0);
  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  Close();
  const std::vector<uint8_t> types = ReadWALRecordTypes(WALPath(wal_number));
  ASSERT_FALSE(types.empty());
  EXPECT_EQ(log::kRecyclableWALIndexMarkerType, types.front());
  Reopen(options);
  ASSERT_EQ(1U, dbfull()->GetDBOptions().recycle_log_file_num);
  VerifyKeys();
  EXPECT_EQ("value", Get("before0"));
  EXPECT_EQ("value", Get("before1"));
  Close();
  Destroy(options);
}

TEST_F(DBWALIndexOnePartitionGateTest, UnorderedWrite) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.unordered_write = true;
  ExpectIndexedWALRoundTrip(options);
}

TEST_F(DBWALIndexOnePartitionGateTest, PipelinedWrite) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.enable_pipelined_write = true;
  ExpectIndexedWALRoundTrip(options);
}

// two_write_queues is the one M1.4 gate that keys on usage at N=1.
TEST_F(DBWALIndexOnePartitionGateTest, TwoWriteQueues) {
  Options options = WALIndexOptions(PartitionWALUsage::kNone);
  options.two_write_queues = true;
  DestroyAndReopen(options);
  Close();
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  const Status s = TryReopen(options);
  ASSERT_TRUE(s.IsNotSupported()) << s.ToString();
  ASSERT_STREQ("partition_wal_usage is not supported with two_write_queues. ",
               s.getState());
}

TEST_F(DBWALIndexTest, SequenceAndWALIndexAllocationIsAtomic) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  constexpr int kThreads = 8;
  constexpr int kAllocationsPerThread = 100;
  constexpr size_t kTotalAllocations = kThreads * kAllocationsPerThread;
  const SequenceNumber first_sequence = db_->GetLatestSequenceNumber() + 1;
  std::vector<std::vector<WALRecordMetadata>> per_thread(kThreads);
  std::vector<port::Thread> threads;
  threads.reserve(kThreads);
  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    per_thread[thread_index].reserve(kAllocationsPerThread);
    threads.emplace_back([&, thread_index]() {
      for (int allocation_index = 0; allocation_index < kAllocationsPerThread;
           ++allocation_index) {
        SequenceNumber last_sequence_before = 0;
        uint64_t wal_index = 0;
        const IOStatus status = dbfull()->TEST_AllocateSequenceAndWALIndex(
            /*sequence_count=*/1, /*write_wal=*/true, &last_sequence_before,
            &wal_index);
        EXPECT_OK(status);
        if (!status.ok()) {
          return;
        }
        per_thread[thread_index].push_back(
            {last_sequence_before + 1, wal_index});
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::vector<WALRecordMetadata> allocations;
  allocations.reserve(kTotalAllocations);
  for (const auto& thread_allocations : per_thread) {
    ASSERT_EQ(static_cast<size_t>(kAllocationsPerThread),
              thread_allocations.size());
    allocations.insert(allocations.end(), thread_allocations.begin(),
                       thread_allocations.end());
  }
  std::sort(allocations.begin(), allocations.end(),
            [](const WALRecordMetadata& lhs, const WALRecordMetadata& rhs) {
              return lhs.sequence < rhs.sequence;
            });
  ASSERT_EQ(kTotalAllocations, allocations.size());
  for (size_t index = 0; index < allocations.size(); ++index) {
    EXPECT_EQ(first_sequence + index, allocations[index].sequence);
    EXPECT_EQ(log::kWALIndexStartNumber + index, allocations[index].wal_index);
  }
}

TEST_F(DBWALIndexTest, DisableWALDoesNotConsumeWALIndex) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  ASSERT_OK(Put("key1", "value1"));
  const SequenceNumber first_sequence = db_->GetLatestSequenceNumber();

  WriteOptions disable_wal;
  disable_wal.disableWAL = true;
  ASSERT_OK(db_->Put(disable_wal, "key2", "value2"));
  const SequenceNumber sequence_after_no_wal = db_->GetLatestSequenceNumber();
  ASSERT_GT(sequence_after_no_wal, first_sequence);

  ASSERT_OK(Put("key3", "value3"));
  const SequenceNumber third_sequence = db_->GetLatestSequenceNumber();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  const std::vector<WALRecordMetadata> metadata =
      ReadWALRecordMetadata(wal_path, wal_number);
  ASSERT_EQ(2U, metadata.size());
  EXPECT_EQ(first_sequence, metadata[0].sequence);
  EXPECT_EQ(log::kWALIndexStartNumber, metadata[0].wal_index);
  EXPECT_EQ(third_sequence, metadata[1].sequence);
  EXPECT_EQ(log::kWALIndexStartNumber + 1, metadata[1].wal_index);
}

TEST_F(DBWALIndexTest, IngestExternalFileDoesNotConsumeWALIndex) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  const std::string external_file = dbname_ + "/external.sst";
  SstFileWriter sst_file_writer(EnvOptions(), options);
  ASSERT_OK(sst_file_writer.Open(external_file));
  ASSERT_OK(sst_file_writer.Put("key2", "value2"));
  ExternalSstFileInfo file_info;
  ASSERT_OK(sst_file_writer.Finish(&file_info));

  ASSERT_OK(Put("key1", "value1"));
  const SequenceNumber first_sequence = db_->GetLatestSequenceNumber();
  {
    ManagedSnapshot snapshot(db_.get());
    ASSERT_OK(
        db_->IngestExternalFile({external_file}, IngestExternalFileOptions()));
  }
  const SequenceNumber sequence_after_ingest = db_->GetLatestSequenceNumber();
  ASSERT_GT(sequence_after_ingest, first_sequence);

  ASSERT_OK(Put("key3", "value3"));
  const SequenceNumber third_sequence = db_->GetLatestSequenceNumber();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  const std::vector<WALRecordMetadata> metadata =
      ReadWALRecordMetadata(wal_path, wal_number);
  ASSERT_EQ(2U, metadata.size());
  EXPECT_EQ(first_sequence, metadata[0].sequence);
  EXPECT_EQ(log::kWALIndexStartNumber, metadata[0].wal_index);
  EXPECT_EQ(third_sequence, metadata[1].sequence);
  EXPECT_EQ(log::kWALIndexStartNumber + 1, metadata[1].wal_index);
}

TEST_F(DBWALIndexTest, IndexedWALReplaysThroughRecovery) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  WriteKeys();
  Reopen(options);

  VerifyKeys();
  // A second recovery replays the same indexed WAL again.
  Reopen(options);
  VerifyKeys();

  Destroy(options);
}

// A read-only opener replays an indexed WAL, whatever its own usage.
TEST_F(DBWALIndexTest, ReadOnlyOpenReadsIndexedWAL) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  WriteKeys();
  Close();
  ASSERT_GT(test::CountIndexedWALFiles(env_, dbname_), 0);
  for (const PartitionWALUsage usage :
       {PartitionWALUsage::kPartitionByColumnFamily,
        PartitionWALUsage::kNone}) {
    SCOPED_TRACE(static_cast<int>(usage));
    Options read_only = options;
    read_only.partition_wal_usage = usage;
    ASSERT_OK(ReadOnlyReopen(read_only));
    VerifyKeys();
    Close();
  }
  Destroy(options);
}

// Records each batch's sequence number as recovery hands it to the filter.
class RecordSequencesWalFilter : public WalFilter {
 public:
  const char* Name() const override { return "RecordSequencesWalFilter"; }

  WalProcessingOption LogRecordFound(unsigned long long /*log_number*/,
                                     const std::string& /*log_file_name*/,
                                     const WriteBatch& batch,
                                     WriteBatch* /*new_batch*/,
                                     bool* /*batch_changed*/) override {
    sequences_.push_back(WriteBatchInternal::Sequence(&batch));
    return WalProcessingOption::kContinueProcessing;
  }

  const std::vector<SequenceNumber>& sequences() const { return sequences_; }

 private:
  std::vector<SequenceNumber> sequences_;
};

// WalFilter sees every record of two indexed WALs exactly once, in sequence
// order.
TEST_F(DBWALIndexTest, WalFilterSeesEveryIndexedRecordOnce) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  WriteKeys();
  Reopen(options);
  for (int i = 0; i < kNumKeys; ++i) {
    ASSERT_OK(Put("more" + std::to_string(i), "value"));
  }
  Close();
  ASSERT_GE(test::CountIndexedWALFiles(env_, dbname_), 2);

  RecordSequencesWalFilter filter;
  options.wal_filter = &filter;
  Reopen(options);
  std::vector<SequenceNumber> expected(2 * kNumKeys);
  std::iota(expected.begin(), expected.end(), SequenceNumber{1});
  EXPECT_EQ(expected, filter.sequences());
  VerifyKeys();
  Close();
  options.wal_filter = nullptr;
  Destroy(options);
}

// The index counter restarts at kWALIndexStartNumber on every open, so WALs
// left live by different processes reuse the same values. Nothing consumes the
// index yet, so this is a documented limitation rather than a bug -- pinned
// here because plain writes after a reopen reach it, and a future fix should
// have to change this test deliberately.
// An append that fails after the index was allocated leaves the index burned.
// It must be declared with a void record, so a later gap check reads it as a
// deliberate hole rather than as data that was written and lost.
//
// The failure is injected by corrupting the log entry, which fails
// VerifyChecksum before anything is appended. That matters: it is a real
// non-poisoning failure, so the writer is still usable and the cover can be
// written. See UncoveredWALIndexFailsSubsequentWrites for the IO-error case.
TEST_F(DBWALIndexTest, BurnedWALIndexIsCovered) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [](void* arg) {
        Slice* const log_entry = static_cast<Slice*>(arg);
        // Flip a byte past the sequence/count header so the batch fails its
        // own checksum inside WriteToWAL, before log::Writer::AddRecord.
        char* const data = const_cast<char*>(log_entry->data());
        data[log_entry->size() - 1] ^= 0xff;
      });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_NOK(WriteProtectedBatch(Key(kNumKeys), "burned"));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  Close();

  // The burned index is the one after the last record that did land.
  const uint64_t burned = log::kWALIndexStartNumber + kNumKeys;
  const std::vector<std::pair<uint64_t, uint64_t>> expected = {
      {burned, burned}};
  EXPECT_EQ(expected, ReadWALVoidRanges(wal_path));

  Destroy(options);
}

// The union of what data records carry and what void records cover is exactly
// the contiguous run the allocator handed out: nothing allocated goes missing.
TEST_F(DBWALIndexTest, NoWALIndexIsLostWhenAppendFails) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [](void* arg) {
        Slice* const log_entry = static_cast<Slice*>(arg);
        char* const data = const_cast<char*>(log_entry->data());
        data[log_entry->size() - 1] ^= 0xff;
      });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_NOK(WriteProtectedBatch(Key(kNumKeys), "burned"));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  Close();

  std::vector<uint64_t> accounted = ReadWALIndices(wal_path, wal_number);
  for (const auto& [lo, hi] : ReadWALVoidRanges(wal_path)) {
    for (uint64_t i = lo; i <= hi; i++) {
      accounted.push_back(i);
    }
  }
  std::sort(accounted.begin(), accounted.end());

  std::vector<uint64_t> expected;
  expected.reserve(kNumKeys + 1);
  for (int i = 0; i <= kNumKeys; i++) {
    expected.push_back(log::kWALIndexStartNumber + i);
  }
  EXPECT_EQ(expected, accounted);

  Destroy(options);
}

// The other failure shape: a real IO error. It poisons the WritableFileWriter,
// so the cover cannot be written either and the index stays uncovered. The
// requirement then is that nothing above the burned index is ever
// acknowledged, which the poisoned writer enforces -- every later append fails
// through MaybeHandleSeenFileWriterError.
TEST_F(DBWALIndexTest, UncoveredWALIndexFailsSubsequentWrites) {
  // Layer the injector over the fixture's own file system, not the default
  // one: the assertions below read the WAL back through `env_`, and under
  // ENCRYPTED_ENV (or MEM_ENV) a WAL written straight to the default file
  // system is not readable through it.
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(env_->GetFileSystem());
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.env = fault_fs_env.get();
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);

  fault_fs->SetThreadLocalErrorContext(
      FaultInjectionIOType::kWrite, 7 /* seed */, 1 /* one_in */,
      true /* retryable */, false /* has_data_loss */);
  fault_fs->EnableThreadLocalErrorInjection(FaultInjectionIOType::kWrite);

  // The append fails, and so does the cover: same poisoned writer.
  ASSERT_NOK(Put(Key(kNumKeys), "burned"));
  // Nothing above the burned index may be acknowledged afterwards.
  ASSERT_NOK(Put(Key(kNumKeys + 1), "after"));

  fault_fs->DisableThreadLocalErrorInjection(FaultInjectionIOType::kWrite);
  // Still refused with injection off, so the guarantee does not depend on the
  // fault staying armed: the failed WAL write has already put the DB in
  // read-only mode, on top of the poisoned writer.
  ASSERT_NOK(Put(Key(kNumKeys + 2), "later"));

  // No data record above the burned index reached the WAL. The run stops at
  // the last index that actually landed.
  const std::vector<uint64_t> indices = ReadWALIndices(wal_path, wal_number);
  std::vector<uint64_t> expected;
  expected.reserve(kNumKeys);
  for (int i = 0; i < kNumKeys; i++) {
    expected.push_back(log::kWALIndexStartNumber + i);
  }
  EXPECT_EQ(expected, indices);

  Close();
  Destroy(options);
}

// Indexing off means no void records, the same way it means no marker and no
// indexed data types: the WAL stays byte-for-byte vanilla.
// A sync that fails after the append burned nothing: the record is in the
// file, so a void over its index would trail the reader's order mark and fail
// every open that rejects corruption. The write still fails.
TEST_F(DBWALIndexTest, SyncFailureAfterAppendWritesNoCover) {
  enum class Failure { kLeadershipSync, kPipelinedLeadershipSync, kWalDirSync };
  auto fault_fs = std::make_shared<FaultInjectionTestFS>(env_->GetFileSystem());
  std::unique_ptr<Env> fault_fs_env(NewCompositeEnv(fault_fs));
  for (const Failure failure :
       {Failure::kLeadershipSync, Failure::kPipelinedLeadershipSync,
        Failure::kWalDirSync}) {
    SCOPED_TRACE(static_cast<int>(failure));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.env = fault_fs_env.get();
    options.enable_pipelined_write =
        failure == Failure::kPipelinedLeadershipSync;
    DestroyAndReopen(options);
    // Unsynced, so the synced write below is the first to sync the new WAL's
    // directory.
    WriteKeys();
    const uint64_t wal_number = LatestWALNumber();

    if (failure == Failure::kWalDirSync) {
      fault_fs->SetThreadLocalErrorContext(
          FaultInjectionIOType::kMetadataWrite, 7 /* seed */, 1 /* one_in */,
          false /* retryable */, false /* has_data_loss */);
      fault_fs->EnableThreadLocalErrorInjection(
          FaultInjectionIOType::kMetadataWrite);
    } else {
      SyncPoint::GetInstance()->SetCallBack(
          "DBImpl::WriteGroupToWAL:AfterLeadershipSync", [](void* arg) {
            *static_cast<IOStatus*>(arg) = IOStatus::IOError("injected");
          });
      SyncPoint::GetInstance()->EnableProcessing();
    }
    WriteOptions sync_write;
    sync_write.sync = true;
    ASSERT_NOK(Put(Key(kNumKeys), "synced", sync_write));
    fault_fs->DisableThreadLocalErrorInjection(
        FaultInjectionIOType::kMetadataWrite);
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    ASSERT_NOK(dbfull()->TEST_GetBGError());
    Close();

    std::vector<uint64_t> expected;
    for (int i = 0; i <= kNumKeys; i++) {
      expected.push_back(log::kWALIndexStartNumber + i);
    }
    EXPECT_EQ(expected, ReadWALIndices(WALPath(wal_number), wal_number));
    EXPECT_TRUE(ReadWALVoidRanges(WALPath(wal_number)).empty());

    for (const WALRecoveryMode mode :
         {WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kAbsoluteConsistency}) {
      SCOPED_TRACE(static_cast<int>(mode));
      options.wal_recovery_mode = mode;
      Reopen(options);
      VerifyKeys();
      EXPECT_EQ("synced", Get(Key(kNumKeys)));
      Close();
    }
    Destroy(options);
  }
}

TEST_F(DBWALIndexTest, DefaultUsageWritesNoVoidRecords) {
  Options options = WALIndexOptions(PartitionWALUsage::kNone);
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [](void* arg) {
        Slice* const log_entry = static_cast<Slice*>(arg);
        char* const data = const_cast<char*>(log_entry->data());
        data[log_entry->size() - 1] ^= 0xff;
      });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_NOK(WriteProtectedBatch(Key(kNumKeys), "burned"));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  Close();

  for (const uint8_t type : ReadWALRecordTypes(wal_path)) {
    EXPECT_NE(log::kWALIndexVoidType, type);
    EXPECT_NE(log::kRecyclableWALIndexVoidType, type);
  }
  EXPECT_TRUE(ReadWALVoidRanges(wal_path).empty());

  Destroy(options);
}

TEST_F(DBWALIndexTest, RecoveryModesContinueWALIndexAcrossLiveWALs) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);

    WriteKeys();
    const uint64_t first_wal = LatestWALNumber();
    Reopen(options);
    WriteKeys();
    const uint64_t second_wal = LatestWALNumber();
    ASSERT_NE(first_wal, second_wal);
    Close();

    std::vector<uint64_t> first_expected;
    std::vector<uint64_t> second_expected;
    for (int i = 0; i < kNumKeys; ++i) {
      first_expected.push_back(log::kWALIndexStartNumber + i);
      second_expected.push_back(log::kWALIndexStartNumber + kNumKeys + i);
    }
    EXPECT_EQ(first_expected, ReadWALIndices(WALPath(first_wal), first_wal));
    EXPECT_EQ(second_expected, ReadWALIndices(WALPath(second_wal), second_wal));
    Destroy(options);
  }
}

TEST_F(DBWALIndexTest, ToleratedTailSeedsFromLastReplayedRecord) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  DestroyAndReopen(options);
  WriteKeyRange(0, 6);
  const uint64_t old_wal = LatestWALNumber();
  Close();
  TruncateLastDataRecord(WALPath(old_wal));

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(6), "value6"));
  Close();

  const std::vector<uint64_t> expected = {6};
  EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
  const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
      {old_wal, 6}};
  EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
  Destroy(options);
}

TEST_F(DBWALIndexTest, ReplayedVoidRangeAdvancesRecoverySeed) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [](void* arg) {
        Slice* const log_entry = static_cast<Slice*>(arg);
        char* const data = const_cast<char*>(log_entry->data());
        data[log_entry->size() - 1] ^= 0xff;
      });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_NOK(WriteProtectedBatch(Key(4), "burned"));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  Close();

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(5), "value5"));
  Close();

  const std::vector<uint64_t> expected = {6};
  EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
  Destroy(options);
}

TEST_F(DBWALIndexTest, PointInTimeSeedsAtCutAndSupersedesSkippedWALs) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  uint64_t first_wal = 0;
  uint64_t second_wal = 0;
  SetupPointInTimeSkippedSecondWAL(options, &first_wal, &second_wal);

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(10), "value10"));
  Close();

  const std::vector<uint64_t> expected = {6, 7};
  EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
  const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
      {first_wal, 6}, {second_wal, 7}};
  EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
  Destroy(options);
}

// The markers recovery writes do not just exist, they resolve the reuse they
// exist for. The skipped WAL keeps its records intact on disk, so the index
// recovery hands out again is present in two files at once: a reader of the
// new WAL must call the old copy obsolete and the new one live.
TEST_F(DBWALIndexTest, RecoveryMarkersResolveReusedIndices) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  uint64_t first_wal = 0;
  uint64_t second_wal = 0;
  SetupPointInTimeSkippedSecondWAL(options, &first_wal, &second_wal);
  const std::vector<uint64_t> skipped_indices =
      ReadWALIndices(WALPath(second_wal), second_wal);
  ASSERT_FALSE(skipped_indices.empty());
  const uint64_t reused = skipped_indices.front();

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(10), "value10"));
  Close();

  // The collision is real: the same wal_index is readable in both files.
  const std::vector<uint64_t> new_indices =
      ReadWALIndices(WALPath(new_wal), new_wal);
  ASSERT_NE(new_indices.end(),
            std::find(new_indices.begin(), new_indices.end(), reused));

  std::unique_ptr<log::Reader> reader =
      OpenWALReader(WALPath(new_wal), new_wal);
  std::string scratch;
  Slice record;
  while (reader->ReadRecord(&record, &scratch)) {
  }
  EXPECT_TRUE(reader->IsWALIndexSuperseded(second_wal, reused));
  EXPECT_FALSE(reader->IsWALIndexSuperseded(new_wal, reused));
  // The first WAL replayed up to its cut, so only the suffix above it went.
  EXPECT_TRUE(reader->IsWALIndexSuperseded(first_wal, 6));
  EXPECT_FALSE(reader->IsWALIndexSuperseded(first_wal, 5));

  Destroy(options);
}

// The reader swallows a corrupt record under kSkipAnyCorruptedRecords, and
// that is the contract. A failed file read is not a corrupt record: it leaves
// the rest of the WAL unseen, so the cut recovery would compute is unknown
// and the open has to fail rather than seed low.
TEST_F(DBWALIndexTest, ReadIOErrorFailsRecoveryUnderSkipMode) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  Close();

  bool recovering_wal = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::RecoverLogFiles:BeforeReadWal",
      [&](void* /*arg*/) { recovering_wal = true; });
  SyncPoint::GetInstance()->SetCallBack(
      "LogReader::ReadMore:AfterReadFile", [&](void* arg) {
        if (recovering_wal) {
          ASSERT_NE(nullptr, arg);
          *static_cast<Status*>(arg) = Status::IOError("injected read error");
        }
      });
  SyncPoint::GetInstance()->EnableProcessing();
  const Status status = TryReopen(options);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();

  ASSERT_TRUE(status.IsIOError()) << status.ToString();
  Destroy(options);
}

TEST_F(DBWALIndexTest, PointInTimeResumeSupersedesOnlyStoppedWAL) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  DestroyAndReopen(options);
  WriteKeyRange(0, 6);
  const uint64_t first_wal = LatestWALNumber();
  Close();
  Reopen(options);
  WriteKeyRange(6, 4);
  const uint64_t second_wal = LatestWALNumber();
  ASSERT_NE(first_wal, second_wal);
  Close();

  CorruptLastDataRecordPayload(WALPath(first_wal));
  ShiftWALSequencesBackOne(WALPath(second_wal));

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(10), "value10"));
  Close();

  const std::vector<uint64_t> expected = {11, 12};
  EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
  const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
      {first_wal, 6}};
  EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
  Destroy(options);
}

TEST_F(DBWALIndexTest, SkipCorruptionSeedsAfterLaterReplayedWAL) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  DestroyAndReopen(options);
  WriteKeyRange(0, 6);
  const uint64_t first_wal = LatestWALNumber();
  Close();
  Reopen(options);
  WriteKeyRange(6, 4);
  const uint64_t second_wal = LatestWALNumber();
  ASSERT_NE(first_wal, second_wal);
  Close();
  CorruptLastDataRecordPayload(WALPath(first_wal));

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(10), "value10"));
  Close();

  const std::vector<uint64_t> expected = {11};
  EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
  EXPECT_TRUE(ReadWALSupersessions(WALPath(new_wal)).empty());
  Destroy(options);
}

TEST_F(DBWALIndexTest, WALIndexAtTypeLimitFailsRecovery) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  // A single record: the merge sees a 1-to-Max jump as a pristine hole and
  // fails on the hole before the seed can exhaust, so the counter
  // exhaustion this test names needs a hole-free file.
  WriteKeyRange(0, 1);
  const uint64_t wal_number = LatestWALNumber();
  Close();

  OverwriteLastDataRecordWALIndex(WALPath(wal_number),
                                  std::numeric_limits<uint64_t>::max());
  const Status status = TryReopen(options);
  ASSERT_TRUE(status.IsCorruption()) << status.ToString();
  EXPECT_NE(std::string::npos, status.ToString().find("type limit"));
  Destroy(options);
}

TEST_F(DBWALIndexTest, WalFilterStopWritesSupersessionAtReplayCut) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  WriteKeyRange(0, 6);
  const uint64_t first_wal = LatestWALNumber();
  Close();
  Reopen(options);
  WriteKeyRange(6, 4);
  const uint64_t second_wal = LatestWALNumber();
  ASSERT_NE(first_wal, second_wal);
  Close();

  StopReplayAfterRecordsWalFilter filter(/*stop_after=*/8);
  options.wal_filter = &filter;
  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(10), "value10"));
  Close();
  options.wal_filter = nullptr;

  const std::vector<uint64_t> expected = {9};
  EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
  const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
      {second_wal, 9}};
  EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
  Destroy(options);
}

// After a filter or point-in-time stop, a record the drive emitted but
// dropped sits at the next index in a WAL that finished clean. The stop must
// cut that WAL too, or the next write reissues the index and the open after
// that finds two live records at it. The reopen after the stop also holds the
// recovery dummy and an empty batch, which must not trip the end-of-recovery
// check.
TEST_F(DBWALIndexTest, StopCutsCleanWALsHoldingDroppedRecords) {
  for (const bool point_in_time : {false, true}) {
    SCOPED_TRACE(point_in_time);
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    const uint64_t lone_wal = SetupLoneRecordBetweenWALs(options);
    // The point-in-time case drops key 4 as corrupted, arming the stop; key
    // 10 does not continue the sequence, so replay stops there.
    IgnoreWALWalFilter filter(lone_wal, point_in_time);
    options.wal_filter = &filter;
    Reopen(options);
    WriteBatch empty_batch;
    ASSERT_OK(db_->Write(WriteOptions(), &empty_batch));
    ASSERT_OK(Put(Key(20), "value20"));
    Close();

    options.wal_filter = nullptr;
    Reopen(options);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
    }
    EXPECT_EQ("NOT_FOUND", Get(Key(4)));
    EXPECT_EQ("NOT_FOUND", Get(Key(10)));
    EXPECT_EQ("value20", Get(Key(20)));
    Close();
    Destroy(options);
  }
}

// The end-of-recovery check refuses an open whose seed would reissue a live
// index, at that open rather than the next. Reached by switching off the cut
// of clean WALs holding dropped records.
TEST_F(DBWALIndexTest, LiveIndexAtOrAboveSeedFailsTheOpen) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  const uint64_t lone_wal = SetupLoneRecordBetweenWALs(options);
  IgnoreWALWalFilter filter(lone_wal, /*report_corrupted=*/false);
  options.wal_filter = &filter;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ProcessLogFilesMerged:CutPastSeed",
      [](void* arg) { *static_cast<bool*>(arg) = false; });
  SyncPoint::GetInstance()->EnableProcessing();
  const Status status = TryReopen(options);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_TRUE(status.IsCorruption()) << status.ToString();
  EXPECT_NE(std::string::npos,
            status.ToString().find("WAL " + std::to_string(lone_wal) +
                                   " holds live wal_index 5"))
      << status.ToString();
  Destroy(options);
}

TEST_F(DBWALIndexTest, UnopenableIndexedWALFailsRecovery) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.paranoid_checks = false;
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  // Over the fixture's file system, not the default one: the WAL above was
  // written through `env_`, and under ENCRYPTED_ENV a default-file-system
  // reopen would fail on unreadable bytes instead of on the injected open
  // error this test is about.
  auto fail_fs =
      std::make_shared<FailWALOpenFileSystem>(env_->GetFileSystem(), wal_path);
  std::unique_ptr<Env> fail_fs_env(NewCompositeEnv(fail_fs));
  options.env = fail_fs_env.get();
  const Status status = TryReopen(options);
  Close();
  ASSERT_NOK(status);
  Destroy(options);
}

// A tolerated tail is the normal state after a crash, and a read-only open
// allocates no wal_index, so it has nothing to supersede and no WAL to record
// a supersession in. It must open and serve the records below the cut.
TEST_F(DBWALIndexTest, ReadOnlyOpenToleratesStopWithoutSupersession) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  Close();
  TruncateLastDataRecord(WALPath(wal_number));

  std::unique_ptr<DB> read_only_db;
  ASSERT_OK(DB::OpenForReadOnly(options, dbname_, &read_only_db));
  ASSERT_NE(nullptr, read_only_db);
  std::string value;
  ASSERT_OK(read_only_db->Get(ReadOptions(), Key(0), &value));
  EXPECT_EQ("value0", value);
  read_only_db.reset();
  Destroy(options);
}

TEST_F(DBWALIndexTest, SupersessionWriteFailureFailsOpen) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  Close();
  TruncateLastDataRecord(WALPath(wal_number));

  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteWALIndexSupersessions:io_s", [](void* arg) {
        *static_cast<IOStatus*>(arg) =
            IOStatus::IOError("injected supersession write failure");
      });
  SyncPoint::GetInstance()->EnableProcessing();
  const Status status = TryReopen(options);
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();

  ASSERT_NOK(status);
  Destroy(options);
}

// Recovery does not retire a superseded WAL, so the next open reads it again
// and redeclares the same cut. The repeat must be stable: the seed still
// clears the indices the previous open reissued in the WAL it wrote, and the
// supersession names only the file that stopped early, never that WAL.
TEST_F(DBWALIndexTest, SecondOpenAfterSupersessionRedeclaresTheSameCut) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  DestroyAndReopen(options);
  WriteKeyRange(0, 6);
  const uint64_t stopped_wal = LatestWALNumber();
  Close();
  TruncateLastDataRecord(WALPath(stopped_wal));

  Reopen(options);
  const uint64_t first_new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(6), "value6"));
  Close();
  const std::vector<std::pair<uint64_t, uint64_t>> expected_supersessions = {
      {stopped_wal, 6}};
  ASSERT_EQ(expected_supersessions,
            ReadWALSupersessions(WALPath(first_new_wal)));
  const std::vector<uint64_t> first_new_indices = {6};
  ASSERT_EQ(first_new_indices,
            ReadWALIndices(WALPath(first_new_wal), first_new_wal));

  // The stopped WAL is still in the recovery set, so this open repeats the
  // whole decision on top of a WAL that already holds index 6.
  Reopen(options);
  const uint64_t second_new_wal = LatestWALNumber();
  ASSERT_NE(first_new_wal, second_new_wal);
  ASSERT_OK(Put(Key(7), "value7"));
  Close();

  const std::vector<uint64_t> second_new_indices = {7};
  EXPECT_EQ(second_new_indices,
            ReadWALIndices(WALPath(second_new_wal), second_new_wal));
  EXPECT_EQ(expected_supersessions,
            ReadWALSupersessions(WALPath(second_new_wal)));

  Reopen(options);
  ASSERT_EQ("value0", Get(Key(0)));
  ASSERT_EQ("value4", Get(Key(4)));
  ASSERT_EQ("value6", Get(Key(6)));
  ASSERT_EQ("value7", Get(Key(7)));
  Destroy(options);
}

TEST_F(DBWALIndexTest, DefaultUsageWritesNoIndexedRecords) {
  ASSERT_EQ(PartitionWALUsage::kNone, Options().partition_wal_usage);

  Options options = WALIndexOptions(PartitionWALUsage::kNone);
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  const std::vector<uint8_t> types = ReadWALRecordTypes(wal_path);
  ASSERT_FALSE(types.empty());
  for (const uint8_t type : types) {
    EXPECT_FALSE(log::IsWALIndexRecordType(type));
    EXPECT_NE(log::kWALIndexMarkerType, type);
    EXPECT_NE(log::kRecyclableWALIndexMarkerType, type);
  }

  const std::vector<uint64_t> wal_indices =
      ReadWALIndices(wal_path, wal_number);
  ASSERT_EQ(static_cast<size_t>(kNumKeys), wal_indices.size());
  for (const uint64_t wal_index : wal_indices) {
    EXPECT_EQ(0U, wal_index);
  }

  Reopen(options);
  VerifyKeys();

  Destroy(options);
}

TEST_F(DBWALIndexTest, IndexedWALCarriesWALIndexOnDisk) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  const std::vector<uint8_t> types = ReadWALRecordTypes(wal_path);
  ASSERT_FALSE(types.empty());
  // The marker is written when the WAL file is started, so it precedes every
  // data record and identifies the file as indexed to older readers.
  EXPECT_EQ(log::kWALIndexMarkerType, types.front());
  int num_indexed = 0;
  for (size_t i = 1; i < types.size(); i++) {
    EXPECT_TRUE(log::IsWALIndexRecordType(types[i]));
    num_indexed++;
  }
  EXPECT_EQ(kNumKeys, num_indexed);

  std::vector<uint64_t> expected_wal_indices;
  expected_wal_indices.reserve(kNumKeys);
  for (int i = 0; i < kNumKeys; i++) {
    expected_wal_indices.push_back(log::kWALIndexStartNumber + i);
  }
  EXPECT_EQ(expected_wal_indices, ReadWALIndices(wal_path, wal_number));

  Reopen(options);
  VerifyKeys();

  Destroy(options);
}

TEST_F(DBWALIndexTest, SwitchedWALCarriesWALIndexOnDisk) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);

  WriteKeys();
  const uint64_t old_wal_number = LatestWALNumber();
  ASSERT_OK(dbfull()->TEST_SwitchWAL());
  const uint64_t new_wal_number = LatestWALNumber();
  ASSERT_GT(new_wal_number, old_wal_number);

  WriteKeys();
  ASSERT_EQ(new_wal_number, LatestWALNumber());
  const std::string wal_path = WALPath(new_wal_number);
  Close();

  const std::vector<uint8_t> types = ReadWALRecordTypes(wal_path);
  ASSERT_FALSE(types.empty());
  EXPECT_EQ(log::kWALIndexMarkerType, types.front());
  int num_indexed = 0;
  for (size_t i = 1; i < types.size(); i++) {
    EXPECT_TRUE(log::IsWALIndexRecordType(types[i]));
    num_indexed++;
  }
  EXPECT_EQ(kNumKeys, num_indexed);

  // The counter is DB-wide, so the switched WAL continues where the old one
  // stopped rather than restarting.
  std::vector<uint64_t> expected_wal_indices;
  expected_wal_indices.reserve(kNumKeys);
  for (int i = 0; i < kNumKeys; i++) {
    expected_wal_indices.push_back(log::kWALIndexStartNumber + kNumKeys + i);
  }
  EXPECT_EQ(expected_wal_indices, ReadWALIndices(wal_path, new_wal_number));

  Destroy(options);
}

TEST_F(DBWALIndexTest, ExhaustedWALIndexFailsOnlyWALWrites) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  dbfull()->TEST_SetNextWALIndex(std::numeric_limits<uint64_t>::max());

  // Must precede the WAL write: its failure sets a background error that
  // rejects every later write.
  WriteOptions disable_wal;
  disable_wal.disableWAL = true;
  ASSERT_OK(db_->Put(disable_wal, "key1", "value1"));

  const Status s = Put("key2", "value2");
  ASSERT_TRUE(s.IsCorruption()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("WAL index exhausted"))
      << s.ToString();
}

// A point-in-time cut leaves a recovery dummy (count 0, reusing the cut
// sequence) ahead of the reissued records in the new WAL. The next open
// replays the dummy, so the merge witness sees the cut sequence twice and
// must accept the repeat: an empty batch consumes an index but no sequence.
TEST_F(DBWALIndexTest, SecondOpenAfterPointInTimeCutReplaysDummy) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  uint64_t first_wal = 0;
  uint64_t second_wal = 0;
  SetupPointInTimeSkippedSecondWAL(options, &first_wal, &second_wal);

  Reopen(options);
  const uint64_t new_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(10), "value10"));
  Close();

  // The dummy shares its sequence with the reissued record: the repeat the
  // witness must accept.
  const std::vector<WALRecordMetadata> metadata =
      ReadWALRecordMetadata(WALPath(new_wal), new_wal);
  ASSERT_EQ(2U, metadata.size());
  EXPECT_EQ(6U, metadata[0].wal_index);
  EXPECT_EQ(7U, metadata[1].wal_index);
  EXPECT_EQ(metadata[0].sequence, metadata[1].sequence);

  // The second open replays the dummy and the reissue instead of failing
  // the witness on the repeated sequence.
  Reopen(options);
  EXPECT_EQ("value0", Get(Key(0)));
  EXPECT_EQ("value10", Get(Key(10)));
  const uint64_t second_new_wal = LatestWALNumber();
  Close();

  // The cut is redeclared, not extended. The stopped WAL keeps the first
  // open's declaration (it finished clean, so nothing redeclares it); the
  // skipped WAL's cut is restated.
  const std::vector<std::pair<uint64_t, uint64_t>> redeclared = {
      {second_wal, 7}};
  EXPECT_EQ(redeclared, ReadWALSupersessions(WALPath(second_new_wal)));

  // A third open under skip re-trusts the discard declaration instead of
  // stopping at the superseded-only frontier.
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  Reopen(options);
  EXPECT_EQ("value0", Get(Key(0)));
  EXPECT_EQ("value10", Get(Key(10)));
  const uint64_t skip_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(11), "value11"));
  Close();
  EXPECT_TRUE(ReadWALSupersessions(WALPath(skip_wal)).empty());
  // The seed clears the explained-skipped suffix as well as the replayed run.
  const std::vector<uint64_t> skip_expected = {11};
  EXPECT_EQ(skip_expected, ReadWALIndices(WALPath(skip_wal), skip_wal));
  Destroy(options);
}

// A checksum failure drops the rest of the reader's block, so mid-file
// corruption reads as a torn tail: the member ends early with the loss
// reported. Skip mode replays across it without a cut (the member finished
// clean); point-in-time cuts at it; the strict modes fail on it.
TEST_F(DBWALIndexTest, MidFileCorruptionMapsAcrossModes) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 6);
    const uint64_t first_wal = LatestWALNumber();
    Reopen(options);
    WriteKeyRange(6, 4);
    const uint64_t second_wal = LatestWALNumber();
    ASSERT_NE(first_wal, second_wal);
    Close();
    // Destroys index 3 (key 2) with readable records on both sides.
    CorruptDataRecordPayloadAt(WALPath(first_wal), 2);

    if (recovery_mode == WALRecoveryMode::kSkipAnyCorruptedRecords) {
      Reopen(options);
      EXPECT_EQ("value0", Get(Key(0)));
      EXPECT_EQ("value1", Get(Key(1)));
      EXPECT_EQ("value6", Get(Key(6)));
      std::string missing;
      for (int key : {2, 3, 4, 5}) {
        EXPECT_TRUE(db_->Get(ReadOptions(), Key(key), &missing).IsNotFound())
            << "key " << key;
      }
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      // The torn tail leaves no cut; the seed clears the replayed run.
      EXPECT_TRUE(ReadWALSupersessions(WALPath(new_wal)).empty());
      const std::vector<uint64_t> expected = {11};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    if (recovery_mode == WALRecoveryMode::kPointInTimeRecovery) {
      Reopen(options);
      EXPECT_EQ("value0", Get(Key(0)));
      EXPECT_EQ("value1", Get(Key(1)));
      std::string missing;
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(2), &missing).IsNotFound());
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(6), &missing).IsNotFound());
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
          {first_wal, 3}, {second_wal, 7}};
      EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
      // The recovery dummy takes the cut index ahead of the reissued Put.
      const std::vector<uint64_t> expected = {3, 4};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    const Status status = TryReopen(options);
    EXPECT_TRUE(status.IsCorruption()) << status.ToString();
    Destroy(options);
  }
}

// A record lost to an unknown type is bounded: the member reads on past it
// and the merge skips exactly the lost index. Skip mode replays across;
// point-in-time cuts at it; the strict modes fail on it.
TEST_F(DBWALIndexTest, SwallowedTypeCorruptionReplaysAcrossInSkipMode) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 6);
    const uint64_t first_wal = LatestWALNumber();
    Reopen(options);
    WriteKeyRange(6, 4);
    const uint64_t second_wal = LatestWALNumber();
    ASSERT_NE(first_wal, second_wal);
    Close();
    // Destroys index 3 (key 2) with readable records on both sides.
    CorruptDataRecordTypeAt(WALPath(first_wal), 2);

    if (recovery_mode == WALRecoveryMode::kSkipAnyCorruptedRecords) {
      Reopen(options);
      EXPECT_EQ("value0", Get(Key(0)));
      EXPECT_EQ("value1", Get(Key(1)));
      EXPECT_EQ("value3", Get(Key(3)));
      EXPECT_EQ("value9", Get(Key(9)));
      std::string missing;
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(2), &missing).IsNotFound());
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      // The swallowed index leaves no cut; the seed clears the replayed run.
      EXPECT_TRUE(ReadWALSupersessions(WALPath(new_wal)).empty());
      const std::vector<uint64_t> expected = {11};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    if (recovery_mode == WALRecoveryMode::kPointInTimeRecovery) {
      Reopen(options);
      EXPECT_EQ("value0", Get(Key(0)));
      EXPECT_EQ("value1", Get(Key(1)));
      std::string missing;
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(2), &missing).IsNotFound());
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(6), &missing).IsNotFound());
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
          {first_wal, 3}, {second_wal, 7}};
      EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
      const std::vector<uint64_t> expected = {3, 4};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    const Status status = TryReopen(options);
    EXPECT_TRUE(status.IsCorruption()) << status.ToString();
    Destroy(options);
  }
}

// A torn tail with records above it in another file is not a tail: the hole
// has data on both sides. The strict modes fail on it (an open the
// sequential path used to succeed); point-in-time cuts and resumes;
// skip replays across it, cutting the torn member.
TEST_F(DBWALIndexTest, TailGapWithDataAboveIsNotATail) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    const uint64_t first_wal = LatestWALNumber();
    Reopen(options);
    WriteKeyRange(4, 2);
    const uint64_t second_wal = LatestWALNumber();
    ASSERT_NE(first_wal, second_wal);
    Reopen(options);
    WriteKeyRange(6, 2);
    const uint64_t third_wal = LatestWALNumber();
    ASSERT_NE(second_wal, third_wal);
    Close();
    // Tears index 6 (key 5) off the middle WAL; indices 7 and 8 survive
    // above it.
    TruncateLastDataRecord(WALPath(second_wal));

    if (recovery_mode == WALRecoveryMode::kTolerateCorruptedTailRecords) {
      const Status status = TryReopen(options);
      ASSERT_TRUE(status.IsCorruption()) << status.ToString();
      EXPECT_NE(std::string::npos,
                status.ToString().find("cannot replay across the gap"));
      Destroy(options);
      continue;
    }
    if (recovery_mode == WALRecoveryMode::kAbsoluteConsistency) {
      const Status status = TryReopen(options);
      ASSERT_TRUE(status.IsCorruption()) << status.ToString();
      EXPECT_NE(std::string::npos,
                status.ToString().find("truncated record body"));
      Destroy(options);
      continue;
    }
    if (recovery_mode == WALRecoveryMode::kPointInTimeRecovery) {
      Reopen(options);
      EXPECT_EQ("value0", Get(Key(0)));
      EXPECT_EQ("value4", Get(Key(4)));
      std::string missing;
      for (int key : {5, 6, 7}) {
        EXPECT_TRUE(db_->Get(ReadOptions(), Key(key), &missing).IsNotFound())
            << "key " << key;
      }
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
          {second_wal, 6}, {third_wal, 7}};
      EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
      const std::vector<uint64_t> expected = {6, 7};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    Reopen(options);
    EXPECT_EQ("value0", Get(Key(0)));
    EXPECT_EQ("value4", Get(Key(4)));
    EXPECT_EQ("value6", Get(Key(6)));
    EXPECT_EQ("value7", Get(Key(7)));
    std::string missing;
    EXPECT_TRUE(db_->Get(ReadOptions(), Key(5), &missing).IsNotFound());
    const uint64_t new_wal = LatestWALNumber();
    ASSERT_OK(Put(Key(10), "value10"));
    Close();
    const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
        {second_wal, 6}};
    EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
    const std::vector<uint64_t> expected = {9};
    EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
    Destroy(options);
  }
}

// A burn cover with live records above it: the merge skips the covered
// index and replays both sides, in every mode. The burn bricks the writer,
// so the records above land in the next WAL after a reopen -- the only shape
// production can produce. Includes a user-written empty batch, which reuses
// its sequence like the recovery dummy and pins the witness's
// non-decreasing check from the write side.
TEST_F(DBWALIndexTest, VoidWithRecordsAboveReplaysInAllModes) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [](void* arg) {
          Slice* const log_entry = static_cast<Slice*>(arg);
          char* const data = const_cast<char*>(log_entry->data());
          data[log_entry->size() - 1] ^= 0xff;
        });
    SyncPoint::GetInstance()->EnableProcessing();
    ASSERT_NOK(WriteProtectedBatch(Key(4), "burned"));
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    const uint64_t burned_wal = LatestWALNumber();
    const std::vector<std::pair<uint64_t, uint64_t>> voids = {{5, 5}};
    EXPECT_EQ(voids, ReadWALVoidRanges(WALPath(burned_wal)));
    Close();

    Reopen(options);
    WriteKeyRange(4, 2);
    WriteBatch empty_batch;
    ASSERT_OK(db_->Write(WriteOptions(), &empty_batch));
    WriteKeyRange(6, 1);
    Close();

    Reopen(options);
    for (int i = 0; i <= 6; ++i) {
      EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
    }
    const uint64_t new_wal = LatestWALNumber();
    ASSERT_OK(Put(Key(10), "value10"));
    Close();
    EXPECT_TRUE(ReadWALSupersessions(WALPath(new_wal)).empty());
    const std::vector<uint64_t> expected = {10};
    EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
    Destroy(options);
  }
}

// A supersession in the new WAL's head explains the old file's obsolete
// suffix: the second open emits the reissued records, skips the abandoned
// ones, and declares no new cut, in every mode.
TEST_F(DBWALIndexTest, SupersededAbsenceExplainedInAllModes) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 6);
    const uint64_t first_wal = LatestWALNumber();
    Close();
    Reopen(options);
    WriteKeyRange(6, 4);
    const uint64_t second_wal = LatestWALNumber();
    ASSERT_NE(first_wal, second_wal);
    Close();

    StopReplayAfterRecordsWalFilter filter(/*stop_after=*/8);
    options.wal_filter = &filter;
    Reopen(options);
    options.wal_filter = nullptr;
    const uint64_t new_wal = LatestWALNumber();
    ASSERT_OK(Put(Key(10), "value10"));
    Close();
    const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
        {second_wal, 9}};
    EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));

    Reopen(options);
    for (int i = 0; i <= 7; ++i) {
      EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
    }
    EXPECT_EQ("value10", Get(Key(10)));
    std::string missing;
    EXPECT_TRUE(db_->Get(ReadOptions(), Key(8), &missing).IsNotFound());
    EXPECT_TRUE(db_->Get(ReadOptions(), Key(9), &missing).IsNotFound());
    const uint64_t newest_wal = LatestWALNumber();
    ASSERT_OK(Put(Key(12), "value12"));
    Close();
    EXPECT_TRUE(ReadWALSupersessions(WALPath(newest_wal)).empty());
    // The seed clears the explained-skipped index as well as the replayed run.
    const std::vector<uint64_t> expected = {11};
    EXPECT_EQ(expected, ReadWALIndices(WALPath(newest_wal), newest_wal));
    Destroy(options);
  }
}

// Indices below the merge's derived lower bound are never questioned: a
// file starting at 11 recovers its records in every mode with no cut.
TEST_F(DBWALIndexTest, ShiftedIndexStartRecoversInAllModes) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    const uint64_t wal_number = LatestWALNumber();
    Close();
    ShiftWALIndices(WALPath(wal_number), 10);

    Reopen(options);
    for (int i = 0; i <= 3; ++i) {
      EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
    }
    const uint64_t new_wal = LatestWALNumber();
    ASSERT_OK(Put(Key(10), "value10"));
    Close();
    EXPECT_TRUE(ReadWALSupersessions(WALPath(new_wal)).empty());
    const std::vector<uint64_t> expected = {15};
    EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
    Destroy(options);
  }
}

// A hole with live records above it and no explanation anywhere stops the
// stream: point-in-time opens at the cut, every other mode fails.
TEST_F(DBWALIndexTest, PristineHoleFailsExceptPointInTime) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    const uint64_t wal_number = LatestWALNumber();
    Close();
    // Moves index 3 to 30, leaving a pristine hole with index 4 above it.
    OverwriteDataRecordWALIndexAt(WALPath(wal_number), 2, 30);

    if (recovery_mode == WALRecoveryMode::kPointInTimeRecovery) {
      Reopen(options);
      EXPECT_EQ("value0", Get(Key(0)));
      EXPECT_EQ("value1", Get(Key(1)));
      std::string missing;
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(2), &missing).IsNotFound());
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(3), &missing).IsNotFound());
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
          {wal_number, 3}};
      EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
      const std::vector<uint64_t> expected = {3, 4};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    const Status status = TryReopen(options);
    ASSERT_TRUE(status.IsCorruption()) << status.ToString();
    EXPECT_NE(std::string::npos, status.ToString().find("unexplained hole"));
    Destroy(options);
  }
}

// Damage before a file's first readable record leaves the lower bound
// unattested: recovery fails in every mode rather than absorbing the loss.
TEST_F(DBWALIndexTest, DamagedHeadFailsInAllModes) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    const uint64_t wal_number = LatestWALNumber();
    Close();
    CorruptDataRecordPayloadAt(WALPath(wal_number), 0);

    const Status status = TryReopen(options);
    ASSERT_TRUE(status.IsCorruption()) << status.ToString();
    EXPECT_NE(std::string::npos, status.ToString().find("head damaged"));
    Destroy(options);
  }
}

// A newest WAL that holds its marker and then a torn write, with no data
// record surviving, has nothing above where it stopped: a lost tail, not a
// damaged head. Every mode that tolerates a tail opens it read-only and
// read-write with every flushed key, and absolute consistency fails exactly as
// the sequential path does on the same bytes.
TEST_F(DBWALIndexTest, DataLessTornNewestWALIsATail) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const bool torn_record : {false, true}) {
    for (const WALRecoveryMode recovery_mode : recovery_modes) {
      SCOPED_TRACE(std::string(torn_record ? "torn record" : "torn header") +
                   " mode " + std::to_string(static_cast<int>(recovery_mode)));
      Options options =
          WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
      options.wal_recovery_mode = recovery_mode;
      DestroyAndReopen(options);
      WriteKeys();
      ASSERT_OK(Flush());
      const uint64_t wal_number = LatestWALNumber();
      if (torn_record) {
        ASSERT_OK(Put(Key(kNumKeys), "torn"));
      }
      Close();
      const std::string wal_path = WALPath(wal_number);
      if (torn_record) {
        // A header and part of the first data record's payload.
        TruncateLastDataRecord(wal_path);
      } else {
        // A header cut short right after the marker.
        WriteWALContents(wal_path, ReadWALContents(wal_path) + "abc");
      }

      const Status sequential = OpenSequentialCopy(options);
      if (recovery_mode == WALRecoveryMode::kAbsoluteConsistency) {
        const Status status = TryReopen(options);
        ASSERT_TRUE(status.IsCorruption()) << status.ToString();
        EXPECT_EQ(sequential.ToString(), status.ToString());
        EXPECT_EQ(std::string::npos, status.ToString().find("head damaged"));
        Destroy(options);
        continue;
      }
      ASSERT_OK(sequential);
      ASSERT_OK(ReadOnlyReopen(options));
      VerifyKeys();
      Close();
      Reopen(options);
      VerifyKeys();
      EXPECT_EQ("NOT_FOUND", Get(Key(kNumKeys)));
      Destroy(options);
    }
  }
}

// The normal crash state under recycling: the newest WAL reuses a recycled
// file and holds its new marker, then a record from the file's previous use.
// Point-in-time recovery stops at the old record before any new data and opens
// read-only and read-write with every flushed key, as the sequential path
// does.
TEST_F(DBWALIndexTest, RecycledWALWithOnlyOldRecordsIsATail) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  options.recycle_log_file_num = 1;
  KeepRecycledWALTails(true);
  DestroyAndReopen(options);
  ASSERT_OK(Put(Key(0), "value0"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put(Key(1), "value1"));
  ASSERT_OK(Flush());
  const uint64_t wal_number = LatestWALNumber();
  Close();
  KeepRecycledWALTails(false);

  const std::string contents = ReadWALContents(WALPath(wal_number));
  const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
  ASSERT_GE(records.size(), 2U);
  EXPECT_EQ(log::kRecyclableWALIndexMarkerType, records[0].type);
  EXPECT_NE(wal_number, RecyclableRecordLogNumber(contents, records[1]));

  ASSERT_OK(OpenSequentialCopy(options));
  ASSERT_OK(ReadOnlyReopen(options));
  EXPECT_EQ("value0", Get(Key(0)));
  EXPECT_EQ("value1", Get(Key(1)));
  Close();
  Reopen(options);
  EXPECT_EQ("value0", Get(Key(0)));
  EXPECT_EQ("value1", Get(Key(1)));
  Destroy(options);
}

// The recycled shape with a report before the old-record stop: the bytes
// after the marker are dropped as corrupt, then the next block starts with a
// record from the file's previous use. Either the first new record is torn
// over old bytes, or the marker does not line up with the old records and the
// reader parses old payload as a header. Point-in-time opens both read-only
// and read-write, as the sequential path does, because the dropped bytes hold
// no header of the current log. When they do, valid new records were lost
// with the corrupt one and the head stays damaged.
TEST_F(DBWALIndexTest, RecycledWALDroppingOnlyOldBytesIsATail) {
  enum class Shape { kTornFirstRecord, kMisalignedMarker, kNewRecordsDropped };
  for (const Shape shape : {Shape::kTornFirstRecord, Shape::kMisalignedMarker,
                            Shape::kNewRecordsDropped}) {
    SCOPED_TRACE(static_cast<int>(shape));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    options.recycle_log_file_num = 1;
    KeepRecycledWALTails(true);
    DestroyAndReopen(options);
    // Spans two blocks, so the recycled file's second block starts with an
    // old fragment.
    ASSERT_OK(Put(Key(0), std::string(40000, 'v')));
    ASSERT_OK(Flush());
    ASSERT_OK(Put(Key(1), "value1"));
    ASSERT_OK(Flush());
    const uint64_t wal_number = LatestWALNumber();
    const int new_records = shape == Shape::kTornFirstRecord     ? 1
                            : shape == Shape::kNewRecordsDropped ? 3
                                                                 : 0;
    for (int i = 0; i < new_records; ++i) {
      ASSERT_OK(Put(Key(2 + i), "new"));
    }
    Close();
    KeepRecycledWALTails(false);

    const std::string wal_path = WALPath(wal_number);
    std::string contents = ReadWALContents(wal_path);
    const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
    ASSERT_GE(records.size(), 2U);
    ASSERT_EQ(log::kRecyclableWALIndexMarkerType, records[0].type);
    if (shape == Shape::kMisalignedMarker) {
      // Old payload where the reader expects the next header.
      ASSERT_NE(wal_number, RecyclableRecordLogNumber(contents, records[1]));
      contents.replace(records[1].header_offset, log::kRecyclableHeaderSize,
                       log::kRecyclableHeaderSize, 'v');
      WriteWALContents(wal_path, contents);
    } else {
      CorruptDataRecordPayloadAt(wal_path, 0);
    }

    if (shape == Shape::kNewRecordsDropped) {
      Status status = TryReopen(options);
      ASSERT_TRUE(status.IsCorruption()) << status.ToString();
      EXPECT_NE(std::string::npos, status.ToString().find("head damaged"));
      status = ReadOnlyReopen(options);
      ASSERT_TRUE(status.IsCorruption()) << status.ToString();
      Destroy(options);
      continue;
    }
    ASSERT_OK(OpenSequentialCopy(options));
    ASSERT_OK(ReadOnlyReopen(options));
    EXPECT_EQ("value1", Get(Key(1)));
    Close();
    Reopen(options);
    EXPECT_EQ(std::string(40000, 'v'), Get(Key(0)));
    EXPECT_EQ("value1", Get(Key(1)));
    EXPECT_EQ("NOT_FOUND", Get(Key(2)));
    Destroy(options);
  }
}

// A recycled WAL holding only its marker and old records, reopened under a
// mode that stops there silently: tolerate and absolute consistency open it
// as the sequential path does. Recycling is off in those modes, so only a
// mode change reaches this.
TEST_F(DBWALIndexTest, RecycledOldRecordStopOpensInStrictModes) {
  for (const WALRecoveryMode recovery_mode :
       {WALRecoveryMode::kTolerateCorruptedTailRecords,
        WALRecoveryMode::kAbsoluteConsistency}) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    options.recycle_log_file_num = 1;
    KeepRecycledWALTails(true);
    DestroyAndReopen(options);
    ASSERT_OK(Put(Key(0), "value0"));
    ASSERT_OK(Flush());
    ASSERT_OK(Put(Key(1), "value1"));
    ASSERT_OK(Flush());
    const uint64_t wal_number = LatestWALNumber();
    Close();
    KeepRecycledWALTails(false);
    const std::string contents = ReadWALContents(WALPath(wal_number));
    const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
    ASSERT_GE(records.size(), 2U);
    ASSERT_NE(wal_number, RecyclableRecordLogNumber(contents, records[1]));

    options.wal_recovery_mode = recovery_mode;
    ASSERT_OK(OpenSequentialCopy(options));
    ASSERT_OK(ReadOnlyReopen(options));
    EXPECT_EQ("value1", Get(Key(1)));
    Close();
    Reopen(options);
    EXPECT_EQ("value0", Get(Key(0)));
    EXPECT_EQ("value1", Get(Key(1)));
    Destroy(options);
  }
}

// A recycled member that holds new data and then stops at a checksum
// mismatch matches the sequential path: absolute fails both, tolerate opens
// both with every key up to the stop. Ported from D126149797.
TEST_F(DBWALIndexTest, RecycledChecksumStopWithDataMatchesSequential) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  options.recycle_log_file_num = 1;
  KeepRecycledWALTails(true);
  DestroyAndReopen(options);
  ASSERT_OK(Put(Key(0), "value0"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put(Key(1), "value1"));
  ASSERT_OK(Flush());
  const uint64_t wal_number = LatestWALNumber();
  // Save a parseable old record: the real write below overwrites the old
  // head in place, leaving misaligned old bytes behind.
  const std::string before = ReadWALContents(WALPath(wal_number));
  const std::vector<PhysicalRecord> before_records =
      ScanPhysicalRecords(before);
  ASSERT_GE(before_records.size(), 2U);
  ASSERT_EQ(log::kRecyclableWALIndexMarkerType, before_records[0].type);
  ASSERT_NE(wal_number, RecyclableRecordLogNumber(before, before_records[1]));
  const std::string first_old = before.substr(
      before_records[1].header_offset,
      before_records[1].header_size + before_records[1].payload_size);

  // A real new record in the recycled newest WAL, so the member holds data.
  ASSERT_OK(Put(Key(2), "value2"));
  Close();
  KeepRecycledWALTails(false);

  const std::string contents = ReadWALContents(WALPath(wal_number));
  const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
  ASSERT_EQ(2U, records.size());
  ASSERT_EQ(log::kRecyclableWALIndexMarkerType, records[0].type);
  ASSERT_EQ(wal_number, RecyclableRecordLogNumber(contents, records[1]));
  const size_t marker_end = records[1].header_offset;
  const std::string real_new = contents.substr(
      marker_end, records[1].header_size + records[1].payload_size);

  // The old record relabeled with this log's number, checksum repaired, then
  // its payload corrupted: a current-log record that fails its checksum.
  std::string corrupt = first_old;
  const PhysicalRecord old_record = ScanPhysicalRecords(first_old).at(0);
  EncodeFixed32(&corrupt[old_record.header_offset + 7],
                static_cast<uint32_t>(wal_number));
  FixRecordChecksum(old_record, &corrupt);
  corrupt[old_record.payload_offset() + 2] ^= 0xff;
  // Marker, the real record, the corrupt one, then the old tail at the next
  // block: the report in block 0 ends at an old-record stop.
  std::string spliced = contents.substr(0, marker_end) + real_new + corrupt;
  ASSERT_LT(spliced.size(), log::kBlockSize);
  spliced.append(log::kBlockSize - spliced.size(), '\0');
  spliced += first_old;
  WriteWALContents(WALPath(wal_number), spliced);

  options.wal_recovery_mode = WALRecoveryMode::kAbsoluteConsistency;
  ASSERT_TRUE(OpenSequentialCopy(options).IsCorruption());
  ASSERT_TRUE(TryReopen(options).IsCorruption());

  options.wal_recovery_mode = WALRecoveryMode::kTolerateCorruptedTailRecords;
  ASSERT_OK(OpenSequentialCopy(options));
  ASSERT_OK(TryReopen(options));
  EXPECT_EQ("value0", Get(Key(0)));
  EXPECT_EQ("value1", Get(Key(1)));
  EXPECT_EQ("value2", Get(Key(2)));
  Destroy(options);
}

// A data-less lost tail is still a loss: when the WAL after it starts above
// the merge's lower bound, nothing in the index stream shows the hole, so
// point-in-time recovery arms its stop at the next WAL and resumes only if the
// sequence continues, exactly as the sequential path does across files. Here
// the lost record's sequence never returns, so the later WAL is not replayed.
TEST_F(DBWALIndexTest, DataLessLossStopsPointInTimeAtTheNextWAL) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  DestroyAndReopen(options);
  WriteKeys();
  ASSERT_OK(Flush());
  const uint64_t lost_wal = LatestWALNumber();
  ASSERT_OK(Put(Key(kNumKeys), "lost"));
  Reopen(options);
  ASSERT_OK(Put(Key(kNumKeys + 1), "after"));
  Close();
  // The first WAL loses its only record after the second was written.
  TruncateLastDataRecord(WALPath(lost_wal));

  ASSERT_OK(OpenSequentialCopy(options));
  Reopen(options);
  VerifyKeys();
  EXPECT_EQ("NOT_FOUND", Get(Key(kNumKeys)));
  EXPECT_EQ("NOT_FOUND", Get(Key(kNumKeys + 1)));
  Destroy(options);
}

// The merged path's copy of DBWALTest.kPointInTimeRecoveryCFConsistency: a
// point-in-time stop is attributed to the WAL that reported the corruption, so
// the column-family consistency check rejects a column family flushed past it,
// as the sequential path does. Without that flush the same corruption opens.
TEST_F(DBWALIndexTest, PointInTimeStopChecksColumnFamilyConsistency) {
  for (const bool flush_past_corruption : {true, false}) {
    SCOPED_TRACE(flush_past_corruption ? "flushed past" : "not flushed");
    Options options = CurrentOptions();
    options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
    options.avoid_flush_during_recovery = true;
    DestroyAndReopen(options);
    CreateAndReopenWithCF({"one", "two"}, options);
    ASSERT_OK(Put(1, "key1", "val1"));
    ASSERT_OK(Put(2, "key2", "val2"));
    const std::string fname =
        LogFileName(dbname_, dbfull()->TEST_LogfileNumber());
    uint64_t offset_to_corrupt = 0;
    ASSERT_OK(env_->GetFileSize(fname, &offset_to_corrupt));
    ASSERT_GT(offset_to_corrupt, 0);
    ASSERT_OK(Put(1, "key3", "val3"));
    ASSERT_OK(test::CorruptFile(env_, fname,
                                static_cast<int>(offset_to_corrupt), 4,
                                /*verify_checksum=*/false));
    ASSERT_OK(Put(2, "key4", "val4"));
    ASSERT_OK(Put(1, "key5", "val5"));
    if (flush_past_corruption) {
      ASSERT_OK(Flush(2));
    }
    Close();

    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    const Status sequential = OpenSequentialCopy(options, {"one", "two"});
    const Status status =
        TryReopenWithColumnFamilies({"default", "one", "two"}, options);
    EXPECT_EQ(sequential.ok(), status.ok()) << sequential.ToString();
    if (flush_past_corruption) {
      ASSERT_TRUE(status.IsCorruption()) << status.ToString();
      EXPECT_NE(std::string::npos,
                status.ToString().find("Column family inconsistency"));
    } else {
      ASSERT_OK(status);
      EXPECT_EQ("val1", Get(1, "key1"));
      EXPECT_EQ("val2", Get(2, "key2"));
      EXPECT_EQ("NOT_FOUND", Get(1, "key3"));
    }
    Destroy(options);
  }
}

// Two trailing WALs both stopped and nothing later passed the second: the
// first ends in a reported loss, the second lost its only record. Merged
// point-in-time names the first for the column-family check, stricter than
// the sequential path, which names the last stopping WAL. A column family
// flushed between them is rejected here and accepted sequentially.
TEST_F(DBWALIndexTest, TwoTrailingStopsAttributeToTheEarlierWAL) {
  Options options = CurrentOptions();
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.avoid_flush_during_recovery = true;
  DestroyAndReopen(options);
  CreateAndReopenWithCF({"one", "two"}, options);
  ASSERT_OK(Put(1, "key1", "val1"));
  ASSERT_OK(Put(2, "key2", "val2"));
  ASSERT_OK(Put(1, "key3", "val3"));
  const uint64_t first_wal = dbfull()->TEST_LogfileNumber();
  // Switches to the second WAL, which column family two's log number names.
  ASSERT_OK(Flush(2));
  const uint64_t second_wal = dbfull()->TEST_LogfileNumber();
  ASSERT_NE(first_wal, second_wal);
  ASSERT_OK(Put(1, "key4", "val4"));
  Close();
  CorruptLastDataRecordPayload(WALPath(first_wal));
  TruncateLastDataRecord(WALPath(second_wal));

  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  ASSERT_OK(OpenSequentialCopy(options, {"one", "two"}));
  const Status status =
      TryReopenWithColumnFamilies({"default", "one", "two"}, options);
  ASSERT_TRUE(status.IsCorruption()) << status.ToString();
  EXPECT_NE(std::string::npos,
            status.ToString().find("Column family inconsistency"));
  Destroy(options);
}

// A recycled WAL ends at a record from its previous use after its new data.
// When a later WAL's records continue the stream past it, the old record is no
// point-in-time stop, and the column-family check accepts a column family
// flushed after it. When the recycled WAL is the last with data and the column
// family was flushed past it, the stop stands and the check fires. Both match
// the sequential path.
TEST_F(DBWALIndexTest, RecycledOldRecordStopsOnlyAtTheStreamEnd) {
  for (const bool trailing : {false, true}) {
    SCOPED_TRACE(trailing ? "trailing" : "passed");
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
    options.recycle_log_file_num = 2;
    KeepRecycledWALTails(true);
    DestroyAndReopen(options);
    CreateAndReopenWithCF({"x"}, options);
    // Equal-sized records across incarnations, so the recycled WAL's new
    // records end on a record boundary of its previous use.
    for (int i = 0; i < 8; ++i) {
      ASSERT_OK(Put(Key(i), "old"));
      if (i % 4 == 3) {
        ASSERT_OK(Flush());
      }
    }
    const uint64_t recycled_wal = LatestWALNumber();
    if (trailing) {
      // A column family record one value byte shorter keeps the size.
      ASSERT_OK(Put(1, Key(8), "ol"));
      ASSERT_OK(Put(Key(9), "new"));
      ASSERT_OK(Flush(1));
      // The WAL the flush switched to stays empty and is trimmed on close.
      KeepRecycledWALTails(false);
    } else {
      ASSERT_OK(Put(Key(8), "new"));
      ASSERT_OK(Put(Key(9), "new"));
    }
    Close();
    KeepRecycledWALTails(false);
    const std::string contents = ReadWALContents(WALPath(recycled_wal));
    const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
    ASSERT_GE(records.size(), 4U);
    EXPECT_EQ(recycled_wal, RecyclableRecordLogNumber(contents, records[2]));
    EXPECT_NE(recycled_wal, RecyclableRecordLogNumber(contents, records[3]));

    if (trailing) {
      const Status sequential = OpenSequentialCopy(options, {"x"});
      const Status status =
          TryReopenWithColumnFamilies({"default", "x"}, options);
      EXPECT_TRUE(sequential.IsCorruption()) << sequential.ToString();
      ASSERT_TRUE(status.IsCorruption()) << status.ToString();
      EXPECT_NE(std::string::npos,
                status.ToString().find("Column family inconsistency"));
      Destroy(options);
      continue;
    }
    ASSERT_OK(OpenSequentialCopy(options, {"x"}));
    ReopenWithColumnFamilies({"default", "x"}, options);
    ASSERT_OK(Put(Key(10), "value10"));
    ASSERT_OK(Put(1, Key(11), "value11"));
    ASSERT_OK(Flush(1));
    Close();

    ASSERT_OK(OpenSequentialCopy(options, {"x"}));
    ReopenWithColumnFamilies({"default", "x"}, options);
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ("old", Get(Key(i))) << "key " << i;
    }
    EXPECT_EQ("new", Get(Key(8)));
    EXPECT_EQ("new", Get(Key(9)));
    EXPECT_EQ("value10", Get(Key(10)));
    EXPECT_EQ("value11", Get(1, Key(11)));
    Destroy(options);
  }
}

// Every truncating merged recovery leaves one warning naming the cut and each
// superseded WAL with its first obsolete index, and every corruption warning
// names the WAL it came from. A recovery that discards nothing adds neither.
TEST_F(DBWALIndexTest, TruncatingRecoveryLogsWALAndCut) {
  const std::string truncated = "Merged WAL recovery truncated";
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;

  // A pristine hole: index 3 moved to 30 stops point-in-time at index 3.
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  Close();
  OverwriteDataRecordWALIndexAt(WALPath(wal_number), 2, 30);
  std::shared_ptr<RecordingLogger> logger = std::make_shared<RecordingLogger>();
  options.info_log = logger;
  Reopen(options);
  EXPECT_EQ(1, logger->count(InfoLogLevel::WARN_LEVEL, truncated));
  EXPECT_TRUE(logger->saw(InfoLogLevel::WARN_LEVEL, "cut at wal_index 3 "));
  EXPECT_TRUE(logger->saw(InfoLogLevel::WARN_LEVEL,
                          "superseded WALs: #" + std::to_string(wal_number) +
                              " from wal_index 3;"));

  // A destroyed record mid-file: the reader's warning names its WAL, and the
  // cut supersedes both WALs.
  options.info_log = nullptr;
  DestroyAndReopen(options);
  WriteKeyRange(0, 6);
  const uint64_t first_wal = LatestWALNumber();
  Reopen(options);
  WriteKeyRange(6, 4);
  const uint64_t second_wal = LatestWALNumber();
  Close();
  CorruptDataRecordPayloadAt(WALPath(first_wal), 2);
  logger = std::make_shared<RecordingLogger>();
  options.info_log = logger;
  Reopen(options);
  EXPECT_TRUE(
      logger->saw(InfoLogLevel::WARN_LEVEL, WALPath(first_wal) + ": dropping"));
  EXPECT_EQ(1, logger->count(InfoLogLevel::WARN_LEVEL, truncated));
  EXPECT_TRUE(logger->saw(
      InfoLogLevel::WARN_LEVEL,
      "superseded WALs: #" + std::to_string(first_wal) + " from wal_index 3 #" +
          std::to_string(second_wal) + " from wal_index 7;"));
  Close();

  // A recovery that discards nothing warns about nothing.
  options.info_log = nullptr;
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  Close();
  logger = std::make_shared<RecordingLogger>();
  options.info_log = logger;
  Reopen(options);
  EXPECT_EQ(0, logger->count(InfoLogLevel::WARN_LEVEL, truncated));
  EXPECT_EQ(0, logger->count(InfoLogLevel::WARN_LEVEL, "dropping"));
  Destroy(options);
}

// The truncation summary's counts and its scope over several WALs: a
// skip-mode tail gap and a WAL-filter stop each warn once, with the tail-gap
// count and the discarded lower bound, and a clean recovery warns nothing and
// logs one completion line rather than one per WAL.
TEST_F(DBWALIndexTest, TruncationSummaryCountsAndCleanCompletion) {
  const std::string truncated = "Merged WAL recovery truncated";
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  uint64_t first_wal = 0;
  uint64_t second_wal = 0;
  // Keys 0-3 at indices 1-4 in the first WAL, keys 4-7 at 5-8 in the second.
  auto write_two_wals = [&]() {
    options.info_log = nullptr;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    first_wal = LatestWALNumber();
    Reopen(options);
    WriteKeyRange(4, 4);
    second_wal = LatestWALNumber();
    Close();
  };
  std::shared_ptr<RecordingLogger> logger;
  auto reopen_logged = [&]() {
    logger = std::make_shared<RecordingLogger>();
    options.info_log = logger;
    Reopen(options);
  };

  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  write_two_wals();
  reopen_logged();
  EXPECT_EQ(0, logger->count(InfoLogLevel::WARN_LEVEL, truncated));
  EXPECT_EQ(1,
            logger->count(InfoLogLevel::INFO_LEVEL, "Recovered merged logs"));
  EXPECT_TRUE(logger->saw(InfoLogLevel::INFO_LEVEL,
                          "Recovered merged logs #" +
                              std::to_string(first_wal) + " to #" +
                              std::to_string(second_wal) + " next seq #9"));
  EXPECT_EQ(0, logger->count(InfoLogLevel::INFO_LEVEL, "Recovered to log #"));
  Close();

  // The first WAL's last record, index 4, is torn.
  options.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  write_two_wals();
  TruncateLastDataRecord(WALPath(first_wal));
  reopen_logged();
  EXPECT_EQ(1, logger->count(InfoLogLevel::WARN_LEVEL, truncated));
  EXPECT_TRUE(logger->saw(InfoLogLevel::WARN_LEVEL,
                          "1 tail-gap indices from wal_index 4,"));
  EXPECT_TRUE(
      logger->saw(InfoLogLevel::WARN_LEVEL, "discarded at least 1 records"));
  Close();

  // The filter stops at key 6, index 7, which it drops.
  StopReplayAfterRecordsWalFilter filter(/*stop_after=*/6);
  options.wal_recovery_mode = WALRecoveryMode::kPointInTimeRecovery;
  write_two_wals();
  options.wal_filter = &filter;
  reopen_logged();
  options.wal_filter = nullptr;
  EXPECT_EQ(1, logger->count(InfoLogLevel::WARN_LEVEL, truncated));
  EXPECT_TRUE(logger->saw(InfoLogLevel::WARN_LEVEL,
                          "cut at wal_index 7 (0: no cut), 0 tail-gap"));
  EXPECT_TRUE(logger->saw(InfoLogLevel::WARN_LEVEL,
                          "superseded WALs: #" + std::to_string(second_wal) +
                              " from wal_index 7;"));
  EXPECT_TRUE(
      logger->saw(InfoLogLevel::WARN_LEVEL, "discarded at least 1 records"));
  Close();
  Destroy(options);
}

// One drive exercising both directions: a burn cover skipped mid-stream and
// a pristine hole stopping it. Point-in-time opens at the cut; the rest fail.
TEST_F(DBWALIndexTest, MixedVoidAndHoleMapsPerMode) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 2);
    SyncPoint::GetInstance()->SetCallBack(
        "DBImpl::WriteToWAL:log_entry", [](void* arg) {
          Slice* const log_entry = static_cast<Slice*>(arg);
          char* const data = const_cast<char*>(log_entry->data());
          data[log_entry->size() - 1] ^= 0xff;
        });
    SyncPoint::GetInstance()->EnableProcessing();
    ASSERT_NOK(WriteProtectedBatch(Key(2), "burned"));
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
    Close();

    Reopen(options);
    WriteKeyRange(2, 3);
    const uint64_t wal_number = LatestWALNumber();
    Close();
    // Moves index 6 to 60, leaving a pristine hole above the void.
    OverwriteDataRecordWALIndexAt(WALPath(wal_number), 2, 60);

    if (recovery_mode == WALRecoveryMode::kPointInTimeRecovery) {
      Reopen(options);
      for (int i = 0; i <= 3; ++i) {
        EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
      }
      std::string missing;
      EXPECT_TRUE(db_->Get(ReadOptions(), Key(4), &missing).IsNotFound());
      const uint64_t new_wal = LatestWALNumber();
      ASSERT_OK(Put(Key(10), "value10"));
      Close();
      const std::vector<std::pair<uint64_t, uint64_t>> supersessions = {
          {wal_number, 6}};
      EXPECT_EQ(supersessions, ReadWALSupersessions(WALPath(new_wal)));
      const std::vector<uint64_t> expected = {6, 7};
      EXPECT_EQ(expected, ReadWALIndices(WALPath(new_wal), new_wal));
      Destroy(options);
      continue;
    }
    const Status status = TryReopen(options);
    ASSERT_TRUE(status.IsCorruption()) << status.ToString();
    EXPECT_NE(std::string::npos, status.ToString().find("unexplained hole"));
    Destroy(options);
  }
}

// Enabling the option leaves every pre-existing WAL unindexed. The first
// reopen meets an all-unindexed set; after more writes the second meets a
// true mixed set. Both route sequential and replay every key.
TEST_F(DBWALIndexTest, MixedRecoverySetOpensAfterEnablingWALIndex) {
  Options none = WALIndexOptions(PartitionWALUsage::kNone);
  DestroyAndReopen(none);
  WriteKeyRange(0, 4);
  const uint64_t legacy_wal = LatestWALNumber();
  Close();
  for (const uint8_t type : ReadWALRecordTypes(WALPath(legacy_wal))) {
    EXPECT_FALSE(log::IsWALIndexRecordType(type));
    EXPECT_NE(log::kWALIndexMarkerType, type);
  }

  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  Reopen(options);
  for (int i = 0; i <= 3; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  WriteKeyRange(4, 4);
  const uint64_t indexed_wal = LatestWALNumber();
  ASSERT_NE(legacy_wal, indexed_wal);
  Close();

  Reopen(options);
  for (int i = 0; i <= 7; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  for (const uint64_t wal_index :
       ReadWALIndices(WALPath(indexed_wal), indexed_wal)) {
    EXPECT_NE(0U, wal_index);
  }
  Destroy(options);
}

// The documented rollback: opening with kNone while indexed WALs survive.
// The pre-pass is skipped and the sequential path replays the survivors;
// new writes are byte-identical legacy records.
TEST_F(DBWALIndexTest, IndexedWALsOpenAfterDisablingWALIndex) {
  Options indexed =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(indexed);
  WriteKeyRange(0, 4);
  Close();

  Options none = WALIndexOptions(PartitionWALUsage::kNone);
  Reopen(none);
  for (int i = 0; i <= 3; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  WriteKeyRange(4, 4);
  const uint64_t new_wal = LatestWALNumber();
  Close();
  const std::vector<uint8_t> types = ReadWALRecordTypes(WALPath(new_wal));
  ASSERT_FALSE(types.empty());
  for (const uint8_t type : types) {
    EXPECT_FALSE(log::IsWALIndexRecordType(type));
    EXPECT_NE(log::kWALIndexMarkerType, type);
    EXPECT_NE(log::kRecyclableWALIndexMarkerType, type);
  }
  Destroy(none);
}

// Runs `ldb check_wal_index` against a closed DB, returning its exit code
// and appending its standard output to `output`.
static int RunCheckWALIndex(const std::string& db, const Options& options,
                            std::string* output) {
  const std::string db_arg = "--db=" + db;
  const char* argv[] = {"./ldb", db_arg.c_str(), "check_wal_index"};
  testing::internal::CaptureStdout();
  const int code = LDBCommandRunner::RunCommand(3, argv, options, LDBOptions(),
                                                /*column_families=*/nullptr);
  *output += testing::internal::GetCapturedStdout();
  return code;
}

// The line `ldb check_wal_index` printed for the WAL named `file`, or "".
static std::string CheckWALIndexLine(const std::string& output,
                                     const std::string& file) {
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find("/" + file + ":") != std::string::npos) {
      return line;
    }
  }
  return "";
}

static uint64_t ManifestWALIndexGuard(Env* env, const std::string& db) {
  uint64_t guard = 0;
  EXPECT_OK(
      VersionSet::ReadWALIndexGuard(db, env->GetFileSystem().get(), &guard));
  return guard;
}

constexpr uint64_t kGuardAnyWAL = std::numeric_limits<uint64_t>::max();

// The guard against older binaries is synced into the MANIFEST before the
// first indexed record reaches the WAL, and survives losing every unsynced
// byte.
TEST_F(DBWALIndexTest, WALIndexGuardIsDurableBeforeFirstIndexedWrite) {
  std::unique_ptr<FaultInjectionTestEnv> fault_env(
      new FaultInjectionTestEnv(env_));
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.env = fault_env.get();
  DestroyAndReopen(options);
  std::vector<uint64_t> guard_at_append;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::WriteToWAL:log_entry", [&](void* /*arg*/) {
        guard_at_append.push_back(ManifestWALIndexGuard(env_, dbname_));
      });
  SyncPoint::GetInstance()->EnableProcessing();
  ASSERT_OK(Put(Key(0), "value0"));
  SyncPoint::GetInstance()->DisableProcessing();
  SyncPoint::GetInstance()->ClearAllCallBacks();
  ASSERT_EQ(1U, guard_at_append.size());
  EXPECT_EQ(kGuardAnyWAL, guard_at_append[0]);

  Close();
  ASSERT_OK(fault_env->DropUnsyncedFileData());
  EXPECT_EQ(kGuardAnyWAL, ManifestWALIndexGuard(env_, dbname_));
  Reopen(options);
  Close();
  options.env = env_;
  Destroy(options);
}

// Every new MANIFEST carries the guard while it is set: here one per write.
TEST_F(DBWALIndexTest, WALIndexGuardSurvivesManifestRollover) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  options.max_manifest_file_size = 1;
  options.max_manifest_space_amp_pct = 0;
  DestroyAndReopen(options);
  const uint64_t first_manifest = dbfull()->TEST_Current_Manifest_FileNo();
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK(Put(Key(i), "value" + std::to_string(i)));
    ASSERT_OK(Flush());
  }
  EXPECT_NE(first_manifest, dbfull()->TEST_Current_Manifest_FileNo());
  EXPECT_EQ(kGuardAnyWAL, ManifestWALIndexGuard(env_, dbname_));
  Reopen(options);
  EXPECT_EQ(kGuardAnyWAL, ManifestWALIndexGuard(env_, dbname_));
  Close();
  Destroy(options);
}

// A DB that never enables indexing writes no guard, in any MANIFEST.
TEST_F(DBWALIndexTest, NeverIndexedDBWritesNoWALIndexGuard) {
  Options options = WALIndexOptions(PartitionWALUsage::kNone);
  options.max_manifest_file_size = 1;
  options.max_manifest_space_amp_pct = 0;
  DestroyAndReopen(options);
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(Put(Key(i), "value" + std::to_string(i)));
    ASSERT_OK(Flush());
    EXPECT_EQ(0U, ManifestWALIndexGuard(env_, dbname_));
    Reopen(options);
  }
  EXPECT_EQ(0U, ManifestWALIndexGuard(env_, dbname_));
  Close();
  Destroy(options);
}

// A kNone open whose recovery flushes the indexed WALs' contents
// (avoid_flush_during_recovery=false) leaves none of them live, so it clears
// the guard itself, while the obsolete files are still on disk.
TEST_F(DBWALIndexTest, KNoneOpenThatFlushesIndexedWALsClearsTheGuard) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  Close();
  options.partition_wal_usage = PartitionWALUsage::kNone;
  options.avoid_flush_during_recovery = false;
  Reopen(options);
  EXPECT_EQ(0U, ManifestWALIndexGuard(env_, dbname_));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  Close();
  Destroy(options);
}

// `ldb check_wal_index` reports an empty WAL and a damaged one as such, never
// as unindexed: a torn marker hides whether the file is indexed.
TEST_F(DBWALIndexTest, CheckWALIndexReportsEmptyAndDamagedWALs) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  Close();
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  const std::string file = wal_path.substr(wal_path.rfind('/') + 1);
  std::string contents = ReadWALContents(wal_path);
  const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
  ASSERT_EQ(1U, records.size());
  ASSERT_EQ(log::kWALIndexMarkerType, records[0].type);

  std::string output;
  EXPECT_NE(0, RunCheckWALIndex(dbname_, options, &output));
  EXPECT_EQ("live WAL " + wal_path + ": indexed",
            CheckWALIndexLine(output, file));

  contents[0] ^= 0x1;
  WriteWALContents(wal_path, contents);
  output.clear();
  EXPECT_NE(0, RunCheckWALIndex(dbname_, options, &output));
  EXPECT_NE(std::string::npos,
            CheckWALIndexLine(output, file)
                .find(": damaged, may hide WAL index records"))
      << output;

  WriteWALContents(wal_path, "");
  output.clear();
  // Still fails: the MANIFEST keeps the guard for a DB that indexed.
  EXPECT_NE(0, RunCheckWALIndex(dbname_, options, &output));
  EXPECT_EQ("live WAL " + wal_path + ": empty",
            CheckWALIndexLine(output, file));
  EXPECT_NE(std::string::npos, output.find("WAL index guard: set, any WAL"));
  Destroy(options);
}

// The rollback procedure in PartitionWALUsage's comment, end to end: set
// kNone, flush every column family, let the indexed WALs go, purge the
// archive, reopen once more, then require that `ldb check_wal_index` finds
// nothing and the guard is clear. With an archive the check fails until the
// archive is purged. A checkpoint taken before the rollback keeps the guard,
// so an older binary refuses it; one taken after does not.
TEST_F(DBWALIndexTest, RollbackProcedureLeavesNoIndexedWAL) {
  std::vector<CompressionType> compressions = {kNoCompression};
  if (StreamingCompressionTypeSupported(kZSTD)) {
    compressions.push_back(kZSTD);
  }
  const std::string checkpoint_before = dbname_ + "_checkpoint_before";
  const std::string checkpoint_after = dbname_ + "_checkpoint_after";
  auto checkpoint = [this](const std::string& dir) {
    ASSERT_OK(DestroyDB(dir, CurrentOptions()));
    Checkpoint* raw = nullptr;
    ASSERT_OK(Checkpoint::Create(db_.get(), &raw));
    std::unique_ptr<Checkpoint> owned(raw);
    // No flush: the checkpoint copies the live WALs.
    ASSERT_OK(
        owned->CreateCheckpoint(dir, /*log_size_for_flush=*/kGuardAnyWAL));
  };
  for (const CompressionType compression : compressions) {
    for (const bool archive : {false, true}) {
      SCOPED_TRACE(std::to_string(static_cast<int>(compression)) + " " +
                   std::to_string(archive));
      Options indexed =
          WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
      indexed.wal_compression = compression;
      indexed.WAL_ttl_seconds = archive ? 1000 : 0;
      DestroyAndReopen(indexed);
      CreateAndReopenWithCF({"one"}, indexed);
      ASSERT_OK(Put(0, "key0", "value0"));
      ASSERT_OK(Put(1, "key1", "value1"));
      checkpoint(checkpoint_before);
      Close();
      const std::vector<uint64_t> indexed_wals = ListWalNumbers();
      ASSERT_FALSE(indexed_wals.empty());
      const std::string indexed_path = WALPath(indexed_wals.back());
      const std::string indexed_file =
          indexed_path.substr(indexed_path.rfind('/') + 1);
      std::string output;
      EXPECT_NE(0, RunCheckWALIndex(dbname_, indexed, &output));
      EXPECT_EQ("live WAL " + indexed_path + ": indexed",
                CheckWALIndexLine(output, indexed_file))
          << output;

      // 1. Set kNone. The indexed WALs stay live until the flush below
      // (avoid_flush_during_recovery), and so does the guard.
      Options none = indexed;
      none.partition_wal_usage = PartitionWALUsage::kNone;
      ReopenWithColumnFamilies({"default", "one"}, none);
      EXPECT_EQ("value0", Get(0, "key0"));
      EXPECT_EQ("value1", Get(1, "key1"));
      ASSERT_OK(Put(0, "key2", "value2"));
      const uint64_t none_wal = dbfull()->TEST_LogfileNumber();
      const std::vector<uint8_t> none_types =
          ReadWALRecordTypes(WALPath(none_wal));
      ASSERT_FALSE(none_types.empty());
      for (const uint8_t type : none_types) {
        EXPECT_FALSE(log::IsWALIndexRecordType(type));
        EXPECT_FALSE(log::IsWALIndexMetadataRecordType(type));
      }
      EXPECT_EQ(indexed_wals.back(), ManifestWALIndexGuard(env_, dbname_));

      // 2. Flush every column family. 3. The indexed WALs are deleted, or
      // moved to the archive.
      ASSERT_OK(Flush(0));
      ASSERT_OK(Flush(1));
      Close();
      for (const uint64_t wal_number : indexed_wals) {
        EXPECT_TRUE(env_->FileExists(WALPath(wal_number)).IsNotFound());
        EXPECT_EQ(
            archive,
            env_->FileExists(ArchivedLogFileName(dbname_, wal_number)).ok());
      }
      output.clear();
      EXPECT_NE(0, RunCheckWALIndex(dbname_, none, &output));
      EXPECT_EQ(archive
                    ? "archived WAL " +
                          ArchivedLogFileName(dbname_, indexed_wals.back()) +
                          ": indexed"
                    : "",
                CheckWALIndexLine(output, indexed_file))
          << output;
      if (archive) {
        // 4. Wait out WAL_ttl_seconds; the reopen below purges the archive.
        env_->SetMockSleep();
        env_->MockSleepForSeconds(2 *
                                  static_cast<int64_t>(none.WAL_ttl_seconds));
      }

      // The guard clears at the first open that finds no indexed WAL live.
      ReopenWithColumnFamilies({"default", "one"}, none);
      checkpoint(checkpoint_after);
      Close();

      // 5. The check finds nothing. 6. The guard is clear.
      output.clear();
      EXPECT_EQ(0, RunCheckWALIndex(dbname_, none, &output)) << output;
      EXPECT_NE(std::string::npos,
                output.find("No WAL index records and no WAL index guard"));
      EXPECT_EQ(0U, ManifestWALIndexGuard(env_, dbname_));
      EXPECT_EQ(0U, ManifestWALIndexGuard(env_, checkpoint_after));
      EXPECT_EQ(kGuardAnyWAL, ManifestWALIndexGuard(env_, checkpoint_before));
      output.clear();
      EXPECT_NE(0, RunCheckWALIndex(checkpoint_before, indexed, &output));
      // What an older binary reads: no remaining WAL holds a WAL index type.
      for (const uint64_t wal_number : ListWalNumbers()) {
        for (const uint8_t type : ReadWALRecordTypes(WALPath(wal_number))) {
          EXPECT_FALSE(log::IsWALIndexRecordType(type)) << wal_number;
          EXPECT_FALSE(log::IsWALIndexMetadataRecordType(type)) << wal_number;
        }
      }

      // 7. Every key is still there.
      ReopenWithColumnFamilies({"default", "one"}, none);
      EXPECT_EQ("value0", Get(0, "key0"));
      EXPECT_EQ("value1", Get(1, "key1"));
      EXPECT_EQ("value2", Get(0, "key2"));
      Close();
      ASSERT_OK(DestroyDB(checkpoint_before, CurrentOptions()));
      ASSERT_OK(DestroyDB(checkpoint_after, CurrentOptions()));
      Destroy(none);
    }
  }
}

// RunCheckWALIndex with one more argument, appending standard error to
// `output` after standard output.
static int RunCheckWALIndexWithArg(const std::string& db,
                                   const std::string& arg,
                                   const Options& options,
                                   std::string* output) {
  const std::string db_arg = "--db=" + db;
  const char* argv[] = {"./ldb", db_arg.c_str(), "check_wal_index",
                        arg.c_str()};
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int code = LDBCommandRunner::RunCommand(4, argv, options, LDBOptions(),
                                                /*column_families=*/nullptr);
  const std::string err = testing::internal::GetCapturedStderr();
  *output += testing::internal::GetCapturedStdout() + err;
  return code;
}

// `ldb check_wal_index` scans the wal_dir the OPTIONS file names, or the one
// --wal_dir names, and fails rather than scanning the DB directory when that
// directory does not exist. Rolling back per the procedure makes it pass.
TEST_F(DBWALIndexTest, CheckWALIndexScansTheConfiguredWALDir) {
  const std::string wal_dir = dbname_ + "_wal";
  const std::string missing_dir = dbname_ + "_no_such_wal_dir";
  Options indexed =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  indexed.wal_dir = wal_dir;
  DestroyAndReopen(indexed);
  WriteKeyRange(0, 4);
  const std::string wal_path =
      LogFileName(wal_dir, dbfull()->TEST_LogfileNumber());
  const std::string file = wal_path.substr(wal_path.rfind('/') + 1);
  Close();

  // Opening ldb with the DB's own options, as the OPTIONS file holds them.
  std::string output;
  EXPECT_NE(0, RunCheckWALIndex(dbname_, indexed, &output));
  EXPECT_EQ("live WAL " + wal_path + ": indexed",
            CheckWALIndexLine(output, file))
      << output;
  output.clear();
  EXPECT_NE(0, RunCheckWALIndexWithArg(dbname_, "--wal_dir=" + wal_dir, indexed,
                                       &output));
  EXPECT_EQ("live WAL " + wal_path + ": indexed",
            CheckWALIndexLine(output, file))
      << output;
  output.clear();
  EXPECT_NE(0, RunCheckWALIndexWithArg(dbname_, "--wal_dir=" + missing_dir,
                                       indexed, &output));
  EXPECT_NE(std::string::npos,
            output.find("WAL directory " + missing_dir + " does not exist"))
      << output;

  Options none = indexed;
  none.partition_wal_usage = PartitionWALUsage::kNone;
  Reopen(none);
  ASSERT_OK(Flush());
  Close();
  Reopen(none);
  Close();
  output.clear();
  EXPECT_EQ(0, RunCheckWALIndex(dbname_, none, &output)) << output;
  EXPECT_NE(std::string::npos,
            output.find("No WAL index records and no WAL index guard"));
  output.clear();
  EXPECT_EQ(0, RunCheckWALIndexWithArg(dbname_, "--wal_dir=" + wal_dir, none,
                                       &output))
      << output;

  // The configured WAL directory is gone, as with an unmounted volume: the
  // check fails instead of finding no WAL in the DB directory.
  std::vector<std::string> children;
  ASSERT_OK(env_->GetChildren(wal_dir, &children));
  for (const std::string& child : children) {
    if (child != "." && child != "..") {
      ASSERT_OK(env_->DeleteFile(wal_dir + "/" + child));
    }
  }
  ASSERT_OK(env_->DeleteDir(wal_dir));
  output.clear();
  EXPECT_NE(
      0, RunCheckWALIndexWithArg(dbname_, "--try_load_options", none, &output));
  EXPECT_NE(std::string::npos,
            output.find("WAL directory " + wal_dir + " does not exist"))
      << output;
  EXPECT_EQ(std::string::npos, output.find("No WAL index records")) << output;
  ASSERT_OK(env_->CreateDirIfMissing(wal_dir));
  Destroy(none);
}

// RunCheckWALIndex with more arguments, appending standard error to `output`
// after standard output.
static int RunCheckWALIndexWithArgs(const std::string& db,
                                    const std::vector<std::string>& args,
                                    const Options& options,
                                    std::string* output) {
  const std::string db_arg = "--db=" + db;
  std::vector<const char*> argv = {"./ldb", db_arg.c_str(), "check_wal_index"};
  for (const std::string& arg : args) {
    argv.push_back(arg.c_str());
  }
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int code = LDBCommandRunner::RunCommand(
      static_cast<int>(argv.size()), argv.data(), options, LDBOptions(),
      /*column_families=*/nullptr);
  const std::string err = testing::internal::GetCapturedStderr();
  *output += testing::internal::GetCapturedStdout() + err;
  return code;
}

// Without --wal_dir, `ldb check_wal_index` finds the WAL directory in the
// OPTIONS file. When OPTIONS loading is disabled or the file cannot be read,
// it fails and suggests --wal_dir, rather than scanning the DB directory: here
// that would miss the indexed WAL archived under the configured wal_dir, whose
// guard is already clear, and pass. Explicit OPTIONS load credit: D126218376.
TEST_F(DBWALIndexTest, CheckWALIndexFailsWithoutTheOptionsFile) {
  const std::string wal_dir = dbname_ + "_wal";
  Options indexed =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  indexed.wal_dir = wal_dir;
  indexed.WAL_ttl_seconds = 1000;
  DestroyAndReopen(indexed);
  WriteKeyRange(0, 4);
  const uint64_t indexed_wal = dbfull()->TEST_LogfileNumber();
  Options none = indexed;
  none.partition_wal_usage = PartitionWALUsage::kNone;
  Reopen(none);
  ASSERT_OK(Flush());
  Close();
  Reopen(none);
  Close();
  const std::string archived = ArchivedLogFileName(wal_dir, indexed_wal);
  ASSERT_OK(env_->FileExists(archived));
  ASSERT_EQ(0U, ManifestWALIndexGuard(env_, dbname_));
  // What ldb is given when its caller does not know the DB's wal_dir.
  Options unknown = none;
  unknown.wal_dir = "";

  std::string output;
  EXPECT_NE(0, RunCheckWALIndexWithArgs(dbname_, {}, unknown, &output));
  EXPECT_NE(std::string::npos, output.find("archived WAL " + archived))
      << output;
  output.clear();
  EXPECT_NE(0, RunCheckWALIndexWithArgs(dbname_, {"--try_load_options=false"},
                                        unknown, &output));
  EXPECT_NE(std::string::npos, output.find("OPTIONS loading is disabled"))
      << output;
  EXPECT_NE(std::string::npos, output.find("--wal_dir")) << output;

  std::vector<std::string> children;
  ASSERT_OK(env_->GetChildren(dbname_, &children));
  std::vector<std::string> options_files;
  for (const std::string& child : children) {
    uint64_t number = 0;
    FileType type;
    if (ParseFileName(child, &number, &type) && type == kOptionsFile) {
      options_files.push_back(child);
      ASSERT_OK(env_->RenameFile(dbname_ + "/" + child,
                                 dbname_ + "/hidden_" + child));
    }
  }
  ASSERT_FALSE(options_files.empty());
  output.clear();
  EXPECT_NE(0, RunCheckWALIndexWithArgs(dbname_, {}, unknown, &output));
  EXPECT_NE(std::string::npos, output.find("Failed to load OPTIONS")) << output;
  EXPECT_NE(std::string::npos, output.find("--wal_dir")) << output;
  const std::string garbage = OptionsFileName(dbname_, 999999);
  ASSERT_OK(WriteStringToFile(env_, "not an OPTIONS file", garbage));
  output.clear();
  EXPECT_NE(0, RunCheckWALIndexWithArgs(dbname_, {}, unknown, &output));
  EXPECT_NE(std::string::npos, output.find("Failed to load OPTIONS")) << output;
  ASSERT_OK(env_->DeleteFile(garbage));

  // With --wal_dir all work as before, including with a corrupt OPTIONS file
  // (ported from D126568250): they find the archived WAL, failing because it
  // is indexed rather than over OPTIONS, and pass once it is purged.
  for (const bool purged : {false, true}) {
    SCOPED_TRACE(purged);
    if (purged) {
      ASSERT_OK(env_->DeleteFile(archived));
    }
    for (const std::vector<std::string>& args :
         {std::vector<std::string>{"--wal_dir=" + wal_dir},
          std::vector<std::string>{"--wal_dir=" + wal_dir,
                                   "--try_load_options=false"},
          std::vector<std::string>{"--wal_dir=" + wal_dir, "corrupt"}}) {
      const bool corrupt = args.back() == "corrupt";
      if (corrupt) {
        ASSERT_OK(WriteStringToFile(env_, "not an OPTIONS file", garbage));
      }
      output.clear();
      const int code = RunCheckWALIndexWithArgs(
          dbname_, corrupt ? std::vector<std::string>{args.front()} : args,
          unknown, &output);
      if (corrupt) {
        ASSERT_OK(env_->DeleteFile(garbage));
        EXPECT_EQ(std::string::npos, output.find("OPTIONS")) << output;
      }
      if (purged) {
        EXPECT_EQ(0, code) << output;
        EXPECT_NE(std::string::npos,
                  output.find("No WAL index records and no WAL index guard"))
            << output;
      } else {
        EXPECT_NE(0, code) << output;
        EXPECT_NE(std::string::npos, output.find("archived WAL " + archived))
            << output;
      }
    }
  }
  for (const std::string& file : options_files) {
    ASSERT_OK(
        env_->RenameFile(dbname_ + "/hidden_" + file, dbname_ + "/" + file));
  }
  Destroy(none);
}

// kNone must take the established sequential recovery path directly. Counting
// the actual WAL open makes an accidentally enabled classification pass add a
// second open and fail this test.
TEST_F(DBWALIndexTest, KNoneOpensEachWALOnceDuringRecovery) {
  Options options = WALIndexOptions(PartitionWALUsage::kNone);
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  auto counting_fs = std::make_shared<CountingWALOpenFileSystem>(
      env_->GetFileSystem(), wal_path);
  std::unique_ptr<Env> counting_env(NewCompositeEnv(counting_fs));
  options.env = counting_env.get();
  Reopen(options);
  EXPECT_EQ(1, counting_fs->open_count());
  for (int i = 0; i <= 3; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  Destroy(options);
}

// Pins the shape of the classification pass: every live WAL is visited
// exactly once. A second visit to the same file would make recovery cost
// super-linear in the WAL count, and would also break the single-pull premise
// the head-integrity reads depend on. Keyed by WAL number so a repeat visit
// is distinguishable from an extra file being drawn into the pass.
//
// Scope: the hook fires once per loop iteration, so this pins visits, not
// reads. That one visit pulls exactly one record is a property of the loop
// body itself, not something this test can observe.
TEST_F(DBWALIndexTest, PrePassExaminesEachLiveWALOnce) {
  Options options =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(options);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  Close();

  std::map<uint64_t, int> fires;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CheckWALFilesCarryWALIndex:PerFile",
      [&](void* arg) { ++fires[*static_cast<uint64_t*>(arg)]; });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  Reopen(options);

  ASSERT_EQ(1U, fires.size());
  EXPECT_EQ(1, fires[wal_number]);
  for (int i = 0; i <= 3; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  Destroy(options);
}

// Simulate the exact condition the merge-side gate backstops: classification
// sees an indexed file, but the merge receives an unindexed member. This uses
// a real DB recovery and must fail before that member can be replayed.
//
// Ordering dependency: the swap is driven from the :AfterClassification hook,
// which fires once every file has been classified but before the merge opens
// any of them. That window is what makes the substitution invisible to
// classification and visible only to the merge. If the hook were ever moved
// inside the classification loop, the pre-pass would see the legacy contents
// itself and reject them, and this test would still pass -- but it would be
// exercising the classification path, not the merge backstop it is named for.
TEST_F(DBWALIndexTest, MergeBackstopRejectsUnindexedWALAfterClassification) {
  Options none = WALIndexOptions(PartitionWALUsage::kNone);
  DestroyAndReopen(none);
  WriteKeyRange(0, 4);
  const uint64_t legacy_wal = LatestWALNumber();
  Close();
  const std::string legacy_contents = ReadWALContents(WALPath(legacy_wal));
  Destroy(none);

  Options indexed =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  DestroyAndReopen(indexed);
  WriteKeyRange(0, 4);
  const uint64_t indexed_wal = LatestWALNumber();
  const std::string indexed_path = WALPath(indexed_wal);
  Close();

  bool replaced_after_classification = false;
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::CheckWALFilesCarryWALIndex:AfterClassification",
      [&](void* /*arg*/) {
        WriteWALContents(indexed_path, legacy_contents);
        replaced_after_classification = true;
      });
  SyncPoint::GetInstance()->EnableProcessing();
  Defer cleanup_sync_point([] {
    SyncPoint::GetInstance()->DisableProcessing();
    SyncPoint::GetInstance()->ClearAllCallBacks();
  });

  const Status status = TryReopen(indexed);
  ASSERT_TRUE(replaced_after_classification);
  ASSERT_TRUE(status.IsCorruption()) << status.ToString();
  EXPECT_NE(std::string::npos,
            status.ToString().find(std::to_string(indexed_wal)));
  EXPECT_NE(std::string::npos,
            status.ToString().find(
                "an unindexed file cannot be ordered in the merge"));
  Destroy(indexed);
}

// A damaged marker-less head with unindexed data is undecidable from the WAL:
// recovery fails open, but operators need a warning when investigating loss.
TEST_F(DBWALIndexTest, DamagedUnindexedHeadLogsWarning) {
  Options none = WALIndexOptions(PartitionWALUsage::kNone);
  DestroyAndReopen(none);
  WriteKeyRange(0, 4);
  const uint64_t wal_number = LatestWALNumber();
  const std::string wal_path = WALPath(wal_number);
  Close();

  std::string contents = ReadWALContents(wal_path);
  const std::vector<PhysicalRecord> records = ScanPhysicalRecords(contents);
  const auto first_data = std::find_if(
      records.begin(), records.end(), [](const PhysicalRecord& record) {
        return record.type == log::kFullType ||
               record.type == log::kRecyclableFullType;
      });
  ASSERT_NE(records.end(), first_data);
  contents[first_data->header_offset + 6] = static_cast<char>(50);
  FixRecordChecksum(*first_data, &contents);
  WriteWALContents(wal_path, contents);

  auto logger = std::make_shared<RecordingLogger>();
  Options indexed =
      WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
  indexed.wal_recovery_mode = WALRecoveryMode::kSkipAnyCorruptedRecords;
  indexed.info_log = logger;
  Reopen(indexed);

  const std::string undecidable =
      "cannot distinguish a pre-option WAL from a lost marker";
  EXPECT_EQ(1, logger->count(InfoLogLevel::WARN_LEVEL, undecidable));
  EXPECT_EQ(0, logger->count(InfoLogLevel::INFO_LEVEL, undecidable));
  EXPECT_TRUE(
      logger->saw(InfoLogLevel::WARN_LEVEL, std::to_string(wal_number)));
  // The warning only matters if the file actually took the path it warns
  // about, so pin the routing decision itself rather than trusting that the
  // warning implies it.
  EXPECT_TRUE(
      logger->saw(InfoLogLevel::INFO_LEVEL, "Recovering WALs sequentially"));
  EXPECT_EQ("NOT_FOUND", Get(Key(0)));
  for (int i = 1; i <= 3; ++i) {
    EXPECT_EQ("value" + std::to_string(i), Get(Key(i))) << "key " << i;
  }
  Destroy(indexed);
}

// A marker prepended to a legacy WAL promises indices the file never
// carries. The batches stay valid, so nothing masks the contradiction and
// recovery fails on the marker in every mode.
TEST_F(DBWALIndexTest, MarkerWithoutIndicesFailsRecovery) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options none = WALIndexOptions(PartitionWALUsage::kNone);
    DestroyAndReopen(none);
    WriteKeyRange(0, 4);
    const uint64_t wal_number = LatestWALNumber();
    Close();

    std::string marker(log::kHeaderSize, '\0');
    marker[6] = static_cast<char>(log::kWALIndexMarkerType);
    const uint32_t crc = crc32c::Mask(crc32c::Value(&marker[6], 1));
    EncodeFixed32(&marker[0], crc);
    WriteWALContents(WALPath(wal_number),
                     marker + ReadWALContents(WALPath(wal_number)));

    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    const Status status = TryReopen(options);
    ASSERT_TRUE(status.IsCorruption()) << status.ToString();
    EXPECT_NE(std::string::npos, status.ToString().find("carry no wal_index"));
    Destroy(options);
  }
}

// Excising the marker from an indexed WAL leaves indexed data with no marker
// over an intact head. The pre-pass fails the open before replay in every
// mode, naming the file.
TEST_F(DBWALIndexTest, IndexedDataWithoutMarkerFailsRecovery) {
  const std::vector<WALRecoveryMode> recovery_modes = {
      WALRecoveryMode::kTolerateCorruptedTailRecords,
      WALRecoveryMode::kAbsoluteConsistency,
      WALRecoveryMode::kPointInTimeRecovery,
      WALRecoveryMode::kSkipAnyCorruptedRecords,
  };
  for (const WALRecoveryMode recovery_mode : recovery_modes) {
    SCOPED_TRACE(static_cast<int>(recovery_mode));
    Options options =
        WALIndexOptions(PartitionWALUsage::kPartitionByColumnFamily);
    options.wal_recovery_mode = recovery_mode;
    DestroyAndReopen(options);
    WriteKeyRange(0, 4);
    const uint64_t wal_number = LatestWALNumber();
    Close();

    std::string contents = ReadWALContents(WALPath(wal_number));
    bool excised = false;
    for (const PhysicalRecord& record : ScanPhysicalRecords(contents)) {
      if (record.type == log::kWALIndexMarkerType ||
          record.type == log::kRecyclableWALIndexMarkerType) {
        contents.erase(record.header_offset,
                       record.header_size + record.payload_size);
        excised = true;
        break;
      }
    }
    ASSERT_TRUE(excised);
    WriteWALContents(WALPath(wal_number), contents);

    const Status status = TryReopen(options);
    ASSERT_TRUE(status.IsCorruption()) << status.ToString();
    EXPECT_NE(std::string::npos, status.ToString().find("before replay"));
    EXPECT_NE(std::string::npos,
              status.ToString().find(std::to_string(wal_number)));
    Destroy(options);
  }
}

TEST_F(DBWALTest, SingleWalPartition) {
  // The WAL containers are per-partition; with wal_partitions > 1 rejected
  // at Open there is exactly one partition throughout the WAL lifecycle.
  Options options = CurrentOptions();
  options.create_if_missing = true;
  DestroyAndReopen(options);
  ASSERT_EQ(1, dbfull()->TEST_NumWalPartitions());

  ASSERT_OK(Put("foo", "v1"));
  ASSERT_EQ(1, dbfull()->TEST_NumWalPartitions());

  // Roll the WAL and flush; the partition count never changes.
  ASSERT_OK(dbfull()->TEST_SwitchMemtable());
  ASSERT_EQ(1, dbfull()->TEST_NumWalPartitions());
  ASSERT_OK(Flush());
  ASSERT_EQ(1, dbfull()->TEST_NumWalPartitions());

  Reopen(options);
  ASSERT_EQ(1, dbfull()->TEST_NumWalPartitions());
  ASSERT_EQ("v1", Get("foo"));
}

TEST_F(DBWALTest, SplitWalMutexConcurrentManualFlush) {
  // The WAL write mutex is split per partition: the single-partition append
  // path (WriteToWAL under manual_wal_flush) must exclude the spanning
  // FlushWAL on the written partition, the single-partition resolve step
  // (PreprocessWrite) must exclude concurrent WAL bookkeeping, and the
  // spanning manual-flush surface (FlushWAL, WALBufferIsEmpty,
  // LockWAL/UnlockWAL) must stay correct under concurrent writes. Hammer
  // both sides; TSan watches for a missed exclusion. Runs with
  // two_write_queues off and on.
  for (bool two_write_queues : {false, true}) {
    Options options = CurrentOptions();
    options.create_if_missing = true;
    options.manual_wal_flush = true;
    options.two_write_queues = two_write_queues;
    // Small buffer: force WAL rotations so obsolete-file cleanup runs
    // against the resolve step.
    options.write_buffer_size = 8192;
    DestroyAndReopen(options);

    constexpr int kWriterThreads = 4;
    constexpr int kWritesPerThread = 200;
    constexpr int kTotalWrites = kWriterThreads * kWritesPerThread;
    std::atomic<int> puts_done{0};
    std::vector<port::Thread> writers;
    for (int t = 0; t < kWriterThreads; ++t) {
      writers.emplace_back([&, t]() {
        for (int i = 0; i < kWritesPerThread; ++i) {
          ASSERT_OK(Put(Key(t * kWritesPerThread + i), "v"));
          ++puts_done;
        }
      });
    }
    // Flush surface across the whole write phase, with periodic flushes so
    // rotated WALs become obsolete while writes are still resolving.
    int rounds = 0;
    bool lock_cycled = false;
    while (puts_done.load() < kTotalWrites) {
      ASSERT_OK(db_->FlushWAL(false));
      (void)dbfull()->WALBufferIsEmpty();
      if (++rounds % 100 == 0) {
        ASSERT_OK(Flush());
      }
      if (!lock_cycled && puts_done.load() >= kTotalWrites / 2) {
        // Writes stall for the hold, then proceed after UnlockWAL.
        ASSERT_OK(db_->LockWAL());
        ASSERT_OK(db_->UnlockWAL());
        lock_cycled = true;
      }
    }
    for (auto& w : writers) {
      w.join();
    }

    ASSERT_OK(db_->FlushWAL(true));
    for (int t = 0; t < kWriterThreads; ++t) {
      for (int i = 0; i < kWritesPerThread; ++i) {
        ASSERT_EQ("v", Get(Key(t * kWritesPerThread + i)));
      }
    }
    ASSERT_OK(db_->LockWAL());
    ASSERT_OK(db_->UnlockWAL());
    Destroy(options);
  }
}

TEST_F(DBWALTest, MemtableInsertFailureTruncatesBoundWal) {
  // A write whose WAL append lands but whose memtable insert fails marks the
  // WAL it bound to for truncation back to its pre-append size, applied when
  // the WAL writer is cleared on close.
  Options options = CurrentOptions();
  options.create_if_missing = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("good", "v"));
  const std::string wal_path =
      LogFileName(dbname_, dbfull()->TEST_GetCurrentLogNumber());
  uint64_t size_before = 0;
  ASSERT_OK(env_->GetFileSize(wal_path, &size_before));

  // No column family 99 exists, so the insert fails after the append.
  WriteBatch batch;
  ASSERT_OK(WriteBatchInternal::Put(&batch, 99, "bad", "v"));
  ASSERT_TRUE(db_->Write(WriteOptions(), &batch).IsInvalidArgument());
  uint64_t size_after_append = 0;
  ASSERT_OK(env_->GetFileSize(wal_path, &size_after_append));
  ASSERT_GT(size_after_append, size_before);

  Close();
  uint64_t size_after_close = 0;
  ASSERT_OK(env_->GetFileSize(wal_path, &size_after_close));
  ASSERT_EQ(size_before, size_after_close);

  Reopen(options);
  ASSERT_EQ("v", Get("good"));
}
}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
