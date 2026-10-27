//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// Indexed wal_partitions > 1 as a production build opens it: no sync point,
// no testing-only option, nothing from the debug-only test harness, so this
// file also builds and runs in release (DEBUG_LEVEL=0) builds.

#include <memory>
#include <string>
#include <vector>

#include "file/filename.h"
#include "port/stack_trace.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/options.h"
#include "test_util/testharness.h"
#include "utilities/fault_injection_env.h"

namespace ROCKSDB_NAMESPACE {

class DBWalPartitionsOpenTest : public testing::Test,
                                public testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_CASE_P(CrashOrCleanClose, DBWalPartitionsOpenTest,
                        testing::Bool());

// Indexed N=4 opens with wal_partitions and partition_wal_usage alone. Writes
// spread over several column families, so several partitions' WALs are live,
// with every fifth write synced, then an async tail after the last sync write.
// With the parameter set, the DB is closed and every byte that was never
// synced is dropped (a crash): every write up to and including the last
// acknowledged sync write comes back, part of the tail is lost, and what comes
// back is a prefix of the write order. A clean close and reopen recovers every
// write. Neuters: restore the wal_partitions > 1 gate, and Open returns
// NotSupported; treat a middle synced key as missing, and the prefix check
// fails.
TEST_P(DBWalPartitionsOpenTest, IndexedOpensWithoutHookAndRecoversPrefix) {
  const bool crash = GetParam();
  constexpr uint32_t kPartitions = 4;
  constexpr int kColumnFamilies = 8;
  constexpr int kSyncedWrites = 200;
  constexpr int kAsyncTail = 10;
  constexpr int kWrites = kSyncedWrites + kAsyncTail;
  constexpr int kSyncEvery = 5;

  auto env = std::make_unique<FaultInjectionTestEnv>(Env::Default());
  const std::string dbname = test::PerThreadDBPath(
      env.get(), crash ? "wal_partitions_open_crash" : "wal_partitions_open");

  Options options;
  options.env = env.get();
  options.create_if_missing = true;
  options.create_missing_column_families = true;
  options.wal_partitions = kPartitions;
  options.partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options.track_and_verify_wals_in_manifest = true;
  // Leave the memtables unflushed so recovery has to replay the WALs.
  options.avoid_flush_during_shutdown = true;
  ASSERT_OK(DestroyDB(dbname, options));

  std::vector<ColumnFamilyDescriptor> cf_descs;
  cf_descs.emplace_back(kDefaultColumnFamilyName, options);
  for (int cf = 1; cf < kColumnFamilies; ++cf) {
    cf_descs.emplace_back("cf" + std::to_string(cf), options);
  }
  auto key = [](int i) { return "key" + std::to_string(i); };

  int last_synced = -1;
  {
    std::vector<ColumnFamilyHandle*> handles;
    std::unique_ptr<DB> db;
    ASSERT_OK(DB::Open(options, dbname, cf_descs, &handles, &db));
    ASSERT_EQ(kPartitions, db->GetDBOptions().wal_partitions);
    for (int i = 0; i < kWrites; ++i) {
      WriteOptions write_options;
      write_options.sync =
          i < kSyncedWrites && i % kSyncEvery == kSyncEvery - 1;
      ASSERT_OK(db->Put(write_options, handles[i % kColumnFamilies], key(i),
                        "value" + std::to_string(i)));
      if (write_options.sync) {
        last_synced = i;
      }
    }
    // Several partitions hold records.
    std::vector<std::string> children;
    ASSERT_OK(env->GetChildren(dbname, &children));
    int live_wals = 0;
    for (const std::string& child : children) {
      uint64_t number = 0;
      FileType type;
      uint64_t size = 0;
      if (ParseFileName(child, &number, &type) && type == kWalFile &&
          env->GetFileSize(dbname + "/" + child, &size).ok() && size > 0) {
        ++live_wals;
      }
    }
    ASSERT_GE(live_wals, 2);
    for (ColumnFamilyHandle* handle : handles) {
      ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
    }
    ASSERT_OK(db->Close());
  }
  ASSERT_EQ(kSyncedWrites - 1, last_synced);
  if (crash) {
    ASSERT_OK(env->DropUnsyncedFileData());
  }

  std::vector<ColumnFamilyHandle*> handles;
  std::unique_ptr<DB> db;
  ASSERT_OK(DB::Open(options, dbname, cf_descs, &handles, &db));
  int recovered = 0;
  for (int i = 0; i < kWrites; ++i) {
    std::string value;
    const Status s =
        db->Get(ReadOptions(), handles[i % kColumnFamilies], key(i), &value);
    if (s.ok()) {
      // Present: everything below it is present too.
      ASSERT_EQ(recovered, i) << "write " << i << " recovered above a hole";
      ASSERT_EQ("value" + std::to_string(i), value);
      ++recovered;
    } else {
      ASSERT_TRUE(s.IsNotFound()) << s.ToString();
    }
  }
  if (crash) {
    ASSERT_GE(recovered, last_synced + 1);
    ASSERT_LT(recovered, kWrites) << "the crash dropped no unsynced tail write";
  } else {
    ASSERT_EQ(kWrites, recovered);
  }
  for (ColumnFamilyHandle* handle : handles) {
    ASSERT_OK(db->DestroyColumnFamilyHandle(handle));
  }
  ASSERT_OK(db->Close());
  db.reset();
  ASSERT_OK(DestroyDB(dbname, options));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
