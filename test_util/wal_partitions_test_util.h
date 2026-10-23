//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "db/db_impl/db_impl.h"
#include "db/write_batch_internal.h"
#include "rocksdb/options.h"
#include "rocksdb/status.h"
#ifndef NDEBUG
#include "test_util/sync_point.h"
#endif

namespace ROCKSDB_NAMESPACE {
namespace test {

// Applies `wal_partitions` to `options` with the settings N > 1 requires:
// indexed WALs, MANIFEST WAL tracking, and concurrent memtable writes (each
// partition's leader inserts concurrently), plus the GC13 opt-in flag so the
// open succeeds in opt builds, where the sync point below compiles out. In
// debug builds it also installs the `AllowWalPartitions` hook. At N == 1 the
// same settings apply (indexing on); the open succeeds with or without them.
inline void EnableWalPartitionsForTest(Options* options, uint32_t partitions) {
  options->wal_partitions = partitions;
  options->partition_wal_usage = PartitionWALUsage::kPartitionByColumnFamily;
  options->track_and_verify_wals_in_manifest = true;
  options->allow_concurrent_memtable_write = true;
  options->experimental_allow_wal_partitions_for_testing = true;
#ifndef NDEBUG
  SyncPoint::GetInstance()->SetCallBack(
      "DBImpl::ValidateOptions:AllowWalPartitions",
      [](void* arg) { *static_cast<bool*>(arg) = true; });
  SyncPoint::GetInstance()->EnableProcessing();
#endif
}

// Every WAL recovery mode, for tests that reopen a copy in each one.
inline std::vector<WALRecoveryMode> AllWalRecoveryModes() {
  return {WALRecoveryMode::kTolerateCorruptedTailRecords,
          WALRecoveryMode::kAbsoluteConsistency,
          WALRecoveryMode::kPointInTimeRecovery,
          WALRecoveryMode::kSkipAnyCorruptedRecords};
}

// Computes column families covering every WAL partition: appends "cf<id>"
// names for ids 1..K to `cf_names`, choosing K so that every partition has a
// family routing to it, and sets `(*cf_for_partition)[p]` to the id (which is
// also the handle index when the families are created in order after default)
// whose writes route to partition p.
inline Status WalPartitionCoveringColumnFamilies(
    size_t num_partitions, std::vector<std::string>* cf_names,
    std::vector<int>* cf_for_partition) {
  cf_names->clear();
  cf_for_partition->assign(num_partitions, 0);
  std::vector<bool> covered(num_partitions, false);
  size_t num_covered = 0;
  for (uint32_t cf_id = 1; num_covered < num_partitions; ++cf_id) {
    WriteBatch batch;
    Status s = WriteBatchInternal::Put(&batch, cf_id, "key", "value");
    if (!s.ok()) {
      return s;
    }
    size_t partition = num_partitions;
    s = DBImpl::TEST_WalPartitionForWrite(batch, num_partitions, &partition);
    if (!s.ok()) {
      return s;
    }
    cf_names->push_back("cf" + std::to_string(cf_id));
    if (!covered[partition]) {
      covered[partition] = true;
      (*cf_for_partition)[partition] = static_cast<int>(cf_id);
      ++num_covered;
    }
  }
  return Status::OK();
}

}  // namespace test
}  // namespace ROCKSDB_NAMESPACE
