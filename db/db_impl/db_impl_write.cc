//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <optional>
#include <thread>
#include <unordered_map>

#include "db/blob/blob_file_partition_manager.h"
#include "db/blob/blob_write_batch_transformer.h"
#include "db/db_impl/db_impl.h"
#include "db/error_handler.h"
#include "db/event_helpers.h"
#include "db/wide/wide_columns_helper.h"
#include "file/filename.h"
#include "logging/logging.h"
#include "memtable/wbwi_memtable.h"
#include "monitoring/perf_context_imp.h"
#include "options/options_helper.h"
#include "port/port.h"
#include "test_util/sync_point.h"
#include "util/cast_util.h"
#include "util/coding.h"
#include "util/hash.h"
#include "util/hash_containers.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

namespace {

constexpr uint64_t kWalColumnFamilyRoutingSeed = 0x6a09e667f3bcc909ULL;
constexpr uint64_t kWalXidRoutingSeed = 0xbb67ae8584caa73bULL;

class BlobWriteRollbackGuard {
 public:
  explicit BlobWriteRollbackGuard(
      const std::vector<BlobWriteBatchTransformer::RollbackInfo>*
          rollback_infos,
      Logger* info_log)
      : rollback_infos_(rollback_infos), info_log_(info_log) {}
  BlobWriteRollbackGuard(const BlobWriteRollbackGuard&) = delete;
  BlobWriteRollbackGuard& operator=(const BlobWriteRollbackGuard&) = delete;
  BlobWriteRollbackGuard(BlobWriteRollbackGuard&&) = delete;
  BlobWriteRollbackGuard& operator=(BlobWriteRollbackGuard&&) = delete;

  ~BlobWriteRollbackGuard() {
    if (!active_ || rollback_infos_ == nullptr) {
      return;
    }

    for (const auto& rollback_info : *rollback_infos_) {
      if (rollback_info.partition_mgr == nullptr) {
        continue;
      }
      Status rollback_s = rollback_info.partition_mgr->MarkBlobWriteAsGarbage(
          rollback_info.file_number, rollback_info.count, rollback_info.bytes);
      if (!rollback_s.ok()) {
        if (info_log_ != nullptr) {
          ROCKS_LOG_ERROR(
              info_log_,
              "Failed to rollback blob direct-write garbage accounting for "
              "file #%" PRIu64 " (%" PRIu64 " blobs, %" PRIu64 " bytes): %s",
              rollback_info.file_number, rollback_info.count,
              rollback_info.bytes, rollback_s.ToString().c_str());
        }
        rollback_s.PermitUncheckedError();
      }
    }
  }

  void Dismiss() { active_ = false; }

 private:
  const std::vector<BlobWriteBatchTransformer::RollbackInfo>* rollback_infos_;
  Logger* info_log_;
  bool active_ = true;
};

class PutEntityFastPathWriteCallback final : public WriteCallback {
 public:
  Status Callback(DB* /*db*/) override { return Status::OK(); }

  bool AllowWriteBatching() override { return false; }
};

}  // namespace

size_t DBImpl::WalPartitionForWrite(
    const WriteBatch& batch,
    const DeferredPutEntityBatch* deferred_put_entities,
    size_t num_partitions) {
  assert(num_partitions > 0);
  // Performance shortcut, not a neuter: with one partition the route is 0
  // whether or not this returns early (hash % 1 == 0), so removing it
  // changes no route. It only skips the record scan and hash, keeping
  // the N=1 write path byte- and cycle-identical.
  if (num_partitions == 1) {
    return 0;
  }

  // The route reads only the lowest column family id, else the lowest XID,
  // so a scan of the record headers suffices: no handler dispatch, no set,
  // and it stops at column family 0, below which nothing routes. The tags
  // are the ones WriteBatch::Iterate hands a ColumnFamilyCollector (column
  // families) and the commit, rollback and end-prepare markers (XIDs). A
  // record the scan cannot parse ends it: the route stays deterministic
  // (what was read before it, else partition 0), and the write path reports
  // the corruption exactly as with one partition.
  std::optional<uint32_t> column_family_id;
  bool has_xid = false;
  Slice xid;
  Slice input = WriteBatchInternal::Contents(&batch);
  if (input.size() >= WriteBatchInternal::kHeader) {
    input.remove_prefix(WriteBatchInternal::kHeader);
  } else {
    input.clear();
  }
  Slice key;
  Slice value;
  Slice blob;
  while (!input.empty() && column_family_id.value_or(1) != 0) {
    char tag = 0;
    uint32_t record_column_family = 0;
    Slice record_xid;
    if (!ReadRecordFromWriteBatch(&input, &tag, &record_column_family, &key,
                                  &value, &blob, &record_xid,
                                  /*write_unix_time=*/nullptr)
             .ok()) {
      break;
    }
    switch (tag) {
      case kTypeColumnFamilyValue:
      case kTypeValue:
      case kTypeColumnFamilyDeletion:
      case kTypeDeletion:
      case kTypeColumnFamilySingleDeletion:
      case kTypeSingleDeletion:
      case kTypeColumnFamilyRangeDeletion:
      case kTypeRangeDeletion:
      case kTypeColumnFamilyMerge:
      case kTypeMerge:
      case kTypeColumnFamilyBlobIndex:
      case kTypeBlobIndex:
      case kTypeColumnFamilyWideColumnEntity:
      case kTypeWideColumnEntity:
      case kTypeColumnFamilyValuePreferredSeqno:
      case kTypeValuePreferredSeqno:
        column_family_id =
            std::min(column_family_id.value_or(record_column_family),
                     record_column_family);
        break;
      case kTypeEndPrepareXID:
      case kTypeCommitXID:
      case kTypeCommitXIDAndTimestamp:
      case kTypeRollbackXID:
        if (!has_xid || record_xid.compare(xid) < 0) {
          xid = record_xid;
          has_xid = true;
        }
        break;
      default:
        break;
    }
  }
  if (deferred_put_entities != nullptr) {
    for (const auto& op : deferred_put_entities->ops) {
      column_family_id = std::min(
          column_family_id.value_or(op.column_family_id), op.column_family_id);
    }
  }

  uint64_t hash = 0;
  if (column_family_id.has_value()) {
    char encoded_id[sizeof(uint32_t)];
    EncodeFixed32(encoded_id, column_family_id.value());
    hash = Hash64(encoded_id, sizeof(encoded_id), kWalColumnFamilyRoutingSeed);
  } else if (has_xid) {
    hash = Hash64(xid.data(), xid.size(), kWalXidRoutingSeed);
  } else {
    return 0;
  }
  return static_cast<size_t>(hash % num_partitions);
}

#ifndef NDEBUG
Status DBImpl::TEST_WalPartitionForWrite(
    const WriteBatch& batch, size_t num_partitions, size_t* wal_partition,
    const std::vector<uint32_t>* deferred_cf_ids) {
  DeferredPutEntityBatch deferred_storage;
  const DeferredPutEntityBatch* deferred_put_entities = nullptr;
  if (deferred_cf_ids != nullptr) {
    for (uint32_t cf_id : *deferred_cf_ids) {
      DeferredPutEntityBatch::Op op;
      op.column_family_id = cf_id;
      deferred_storage.ops.push_back(std::move(op));
    }
    deferred_put_entities = &deferred_storage;
  }
  if (num_partitions == 0 || wal_partition == nullptr) {
    return Status::InvalidArgument("WAL routing requires a partition");
  }
  *wal_partition =
      WalPartitionForWrite(batch, deferred_put_entities, num_partitions);
  return Status::OK();
}
#endif

// Convenience methods
Status DBImpl::Put(const WriteOptions& o, ColumnFamilyHandle* column_family,
                   const Slice& key, const Slice& val) {
  const Status s = FailIfCfHasTs(column_family);
  if (!s.ok()) {
    return s;
  }
  return DB::Put(o, column_family, key, val);
}

Status DBImpl::Put(const WriteOptions& o, ColumnFamilyHandle* column_family,
                   const Slice& key, const Slice& ts, const Slice& val) {
  const Status s = FailIfTsMismatchCf(column_family, ts);
  if (!s.ok()) {
    return s;
  }
  return DB::Put(o, column_family, key, ts, val);
}

Status DBImpl::PutEntity(const WriteOptions& options,
                         ColumnFamilyHandle* column_family, const Slice& key,
                         const WideColumns& columns) {
  const Status s = FailIfCfHasTs(column_family);
  if (!s.ok()) {
    return s;
  }

  return PutEntityFastPath(options, column_family, key, columns);
}

Status DBImpl::PutEntity(const WriteOptions& options, const Slice& key,
                         const AttributeGroups& attribute_groups) {
  if (attribute_groups.empty()) {
    return Status::InvalidArgument(
        "Cannot call this method with empty attribute groups");
  }

  for (const AttributeGroup& ag : attribute_groups) {
    const Status s = FailIfCfHasTs(ag.column_family());
    if (!s.ok()) {
      return s;
    }
  }
  return PutEntityFastPath(options, key, attribute_groups);
}

Status DBImpl::Merge(const WriteOptions& o, ColumnFamilyHandle* column_family,
                     const Slice& key, const Slice& val) {
  const Status s = FailIfCfHasTs(column_family);
  if (!s.ok()) {
    return s;
  }
  auto cfh = static_cast_with_check<ColumnFamilyHandleImpl>(column_family);
  if (!cfh->cfd()->ioptions().merge_operator) {
    return Status::NotSupported("Provide a merge_operator when opening DB");
  } else {
    return DB::Merge(o, column_family, key, val);
  }
}

Status DBImpl::Merge(const WriteOptions& o, ColumnFamilyHandle* column_family,
                     const Slice& key, const Slice& ts, const Slice& val) {
  const Status s = FailIfTsMismatchCf(column_family, ts);
  if (!s.ok()) {
    return s;
  }
  return DB::Merge(o, column_family, key, ts, val);
}

Status DBImpl::Delete(const WriteOptions& write_options,
                      ColumnFamilyHandle* column_family, const Slice& key) {
  const Status s = FailIfCfHasTs(column_family);
  if (!s.ok()) {
    return s;
  }
  return DB::Delete(write_options, column_family, key);
}

Status DBImpl::Delete(const WriteOptions& write_options,
                      ColumnFamilyHandle* column_family, const Slice& key,
                      const Slice& ts) {
  const Status s = FailIfTsMismatchCf(column_family, ts);
  if (!s.ok()) {
    return s;
  }
  return DB::Delete(write_options, column_family, key, ts);
}

Status DBImpl::SingleDelete(const WriteOptions& write_options,
                            ColumnFamilyHandle* column_family,
                            const Slice& key) {
  const Status s = FailIfCfHasTs(column_family);
  if (!s.ok()) {
    return s;
  }
  return DB::SingleDelete(write_options, column_family, key);
}

Status DBImpl::SingleDelete(const WriteOptions& write_options,
                            ColumnFamilyHandle* column_family, const Slice& key,
                            const Slice& ts) {
  const Status s = FailIfTsMismatchCf(column_family, ts);
  if (!s.ok()) {
    return s;
  }
  return DB::SingleDelete(write_options, column_family, key, ts);
}

Status DBImpl::DeleteRange(const WriteOptions& write_options,
                           ColumnFamilyHandle* column_family,
                           const Slice& begin_key, const Slice& end_key) {
  const Status s = FailIfCfHasTs(column_family);
  if (!s.ok()) {
    return s;
  }
  return DB::DeleteRange(write_options, column_family, begin_key, end_key);
}

Status DBImpl::DeleteRange(const WriteOptions& write_options,
                           ColumnFamilyHandle* column_family,
                           const Slice& begin_key, const Slice& end_key,
                           const Slice& ts) {
  const Status s = FailIfTsMismatchCf(column_family, ts);
  if (!s.ok()) {
    return s;
  }
  return DB::DeleteRange(write_options, column_family, begin_key, end_key, ts);
}

void DBImpl::SetRecoverableStatePreReleaseCallback(
    PreReleaseCallback* callback) {
  recoverable_state_pre_release_callback_.reset(callback);
}

Status DBImpl::Write(const WriteOptions& write_options, WriteBatch* my_batch) {
  Status s;
  if (write_options.protection_bytes_per_key > 0) {
    s = WriteBatchInternal::UpdateProtectionInfo(
        my_batch, write_options.protection_bytes_per_key);
  }
  if (s.ok()) {
    s = WriteImpl(write_options, my_batch, /*callback=*/nullptr,
                  /*user_write_cb=*/nullptr,
                  /*wal_used=*/nullptr);
  }
  return s;
}

Status DBImpl::WriteWithCallback(const WriteOptions& write_options,
                                 WriteBatch* my_batch, WriteCallback* callback,
                                 UserWriteCallback* user_write_cb) {
  if (callback != nullptr && NumWalPartitions() > 1) {
    // The callback runs on its write group's leader, and each partition has
    // its own leader, so callbacks on different partitions run concurrently.
    return Status::NotSupported(
        "WriteWithCallback with a WriteCallback is not supported with "
        "wal_partitions > 1");
  }
  Status s;
  if (write_options.protection_bytes_per_key > 0) {
    s = WriteBatchInternal::UpdateProtectionInfo(
        my_batch, write_options.protection_bytes_per_key);
  }
  if (s.ok()) {
    s = WriteImpl(write_options, my_batch, callback, user_write_cb);
  }
  return s;
}

Status DBImpl::WriteWithCallback(const WriteOptions& write_options,
                                 WriteBatch* my_batch,
                                 UserWriteCallback* user_write_cb) {
  Status s;
  if (write_options.protection_bytes_per_key > 0) {
    s = WriteBatchInternal::UpdateProtectionInfo(
        my_batch, write_options.protection_bytes_per_key);
  }
  if (s.ok()) {
    s = WriteImpl(write_options, my_batch, /*callback=*/nullptr, user_write_cb);
  }
  return s;
}

Status DBImpl::IngestWriteBatchWithIndex(
    const WriteOptions& write_options,
    std::shared_ptr<WriteBatchWithIndex> wbwi) {
  if (!wbwi) {
    return Status::InvalidArgument("Batch is nullptr!");
  }
  if (!write_options.disableWAL) {
    return Status::NotSupported(
        "IngestWriteBatchWithIndex does not support disableWAL=true");
  }
  if (HasAnyBlobDirectWriteColumnFamily()) {
    return Status::NotSupported(
        "IngestWriteBatchWithIndex is not supported with "
        "enable_blob_direct_write");
  }
  Status s;
  if (write_options.protection_bytes_per_key > 0) {
    s = WriteBatchInternal::UpdateProtectionInfo(
        wbwi->GetWriteBatch(), write_options.protection_bytes_per_key);
  }
  if (s.ok()) {
    WriteBatch dummy_empty_batch;
    s = WriteImpl(
        write_options, /*updates=*/&dummy_empty_batch, /*callback=*/nullptr,
        /*user_write_cb=*/nullptr, /*log_used=*/nullptr, /*log_ref=*/0,
        /*disable_memtable=*/false, /*seq_used=*/nullptr,
        /*batch_cnt=*/0, /*pre_release_callback=*/nullptr,
        /*post_memtable_callback=*/nullptr, /*wbwi=*/wbwi);
  }
  return s;
}

Status DBImpl::IngestWBWIAsMemtable(
    std::shared_ptr<WriteBatchWithIndex> wbwi,
    const WBWIMemTable::SeqnoRange& assigned_seqno, uint64_t min_prep_log,
    SequenceNumber last_seqno_after_ingest, bool memtable_updated,
    bool ingest_wbwi_for_commit, bool ignore_missing_cf) {
  // Rejected in WriteImpl above one partition: the switch below assumes
  // no other leader is inserting.
  assert(NumWalPartitions() == 1);
  // Keys in new memtable have seqno > last_seqno_after_ingest >= keys in wbwi.
  assert(assigned_seqno.upper_bound <= last_seqno_after_ingest);
  // Keys in the current memtable have seqno <= LastSequence() < keys in wbwi.
  assert(assigned_seqno.lower_bound > versions_->LastSequence());
  autovector<ReadOnlyMemTable*> memtables;
  autovector<ColumnFamilyData*> cfds;
  InstrumentedMutexLock lock(&mutex_);
  ColumnFamilySet* cf_set = versions_->GetColumnFamilySet();

  // Create WBWIMemTables
  for (const auto& [cf_id, stat] : wbwi->GetCFStats()) {
    ColumnFamilyData* cfd = cf_set->GetColumnFamily(cf_id);
    if (!cfd) {
      if (ignore_missing_cf) {
        continue;
      }
      for (auto mem : memtables) {
        mem->Unref();
        delete mem;
      }
      for (auto cfd_ptr : cfds) {
        cfd_ptr->UnrefAndTryDelete();
      }
      Status s = Status::InvalidArgument(
          "Invalid column family id from WriteBatchWithIndex: " +
          std::to_string(cf_id));
      if (memtable_updated) {
        s = Status::Corruption(
            "Part of the write batch is applied. Memtable is in an "
            "inconsistent state due to invalid column family. " +
            s.ToString());
        error_handler_.SetBGError(s, BackgroundErrorReason::kMemTable);
      } else if (ingest_wbwi_for_commit) {
        s = Status::Corruption(
            "Commit marker is durable in WAL, but publishing committed WBWI "
            "data to memtable failed due to invalid column family. This DB "
            "instance cannot safely resume; close and reopen the DB for WAL "
            "recovery. " +
            s.ToString());
        error_handler_.SetBGError(s, BackgroundErrorReason::kMemTable);
      }

      return s;
    }
    WBWIMemTable* wbwi_memtable =
        new WBWIMemTable(wbwi, cfd->user_comparator(), cf_id, &cfd->ioptions(),
                         &cfd->GetLatestMutableCFOptions(), stat);
    wbwi_memtable->Ref();
    wbwi_memtable->AssignSequenceNumbers(assigned_seqno);
    // This is needed to keep the WAL that contains Prepare alive until
    // committed data in this memtable is persisted.
    wbwi_memtable->SetMinPrepLog(min_prep_log);
    memtables.push_back(wbwi_memtable);
    cfd->Ref();
    cfds.push_back(cfd);
  }

  autovector<ColumnFamilyData*> cfds_for_atomic_flush;
  if (immutable_db_options_.atomic_flush) {
    SelectColumnFamiliesForAtomicFlush(&cfds_for_atomic_flush);
    for (auto cfd : cfds_for_atomic_flush) {
      bool found = false;
      for (auto existing_cfd : cfds) {
        if (existing_cfd == cfd) {
          found = true;
          break;
        }
      }
      if (!found) {
        cfd->Ref();
        cfds.push_back(cfd);
      }
    }
  }

  // Stop writes to the DB by entering both write threads
  AllPartitionsWriter nonmem_w(this, /*nonmem=*/true);
  if (two_write_queues_) {
    nonmem_w.EnterUnbatched();
  }
  WaitForPendingWrites();

  // Switch memtable and add WBWIMemTables
  Status s;
  const bool defer_wbm_accounting_refresh =
      immutable_db_options_.atomic_flush && cfds.size() > 1;
  bool switch_attempted = false;
  for (size_t i = 0; i < cfds.size(); ++i) {
    WriteContext write_context;
    switch_attempted = true;
    // TODO: not switch on empty memtable, may need to update metadata
    //   like NextLogNumber(), earliest_seqno and memtable id.
    if (i < memtables.size()) {
      s = defer_wbm_accounting_refresh
              ? SwitchMemtableImpl(cfds[i], &write_context, memtables[i],
                                   last_seqno_after_ingest,
                                   false /* refresh_wbm_accounting */)
              : SwitchMemtable(cfds[i], &write_context, memtables[i],
                               last_seqno_after_ingest);
    } else {
      s = defer_wbm_accounting_refresh
              ? SwitchMemtableImpl(cfds[i], &write_context, nullptr, 0,
                                   false /* refresh_wbm_accounting */)
              : SwitchMemtable(cfds[i], &write_context);
    }
    if (!s.ok()) {
      // SwitchMemtable() can only fail if a new WAL is to be created, this
      // should only happen for the first call to SwitchMemtable(). log will
      // be empty and no new WAL is created for the rest of the calls.
      assert(i == 0);
      if (i != 0 || memtable_updated) {
        // escalate error to non-recoverable
        s = Status::Corruption(
            "Part of the write batch is applied. Memtable is in an "
            "inconsistent state due to SwitchMemtable failure. " +
            s.ToString());
        error_handler_.SetBGError(s, BackgroundErrorReason::kMemTable);
      } else if (ingest_wbwi_for_commit) {
        s = Status::Corruption(
            "Commit marker is durable in WAL, but publishing committed WBWI "
            "data to memtable failed due to SwitchMemtable failure. This DB "
            "instance cannot safely resume; close and reopen the DB for WAL "
            "recovery. " +
            s.ToString());
        error_handler_.SetBGError(s, BackgroundErrorReason::kMemTable);
      } else {
        // SwitchMemtable() already sets appropriate bg error
      }
      for (size_t j = i; j < memtables.size(); j++) {
        memtables[j]->Unref();
        delete memtables[j];
      }
      break;
    }
  }
  if (defer_wbm_accounting_refresh && switch_attempted) {
    UpdateWBMAccountingAfterMemtableSwitch(true /* refresh_now */);
  }
  for (size_t i = 0; i < cfds.size(); ++i) {
    if (cfds[i]->UnrefAndTryDelete()) {
      cfds[i] = nullptr;
    }
  }

  // exit the second queue before returning
  if (two_write_queues_) {
    nonmem_w.ExitUnbatched();
  }
  if (s.ok()) {
    // Trigger flushes for the new immutable memtables.
    for (const auto cfd : cfds) {
      if (cfd == nullptr) {
        continue;
      }
      cfd->imm()->FlushRequested();
      if (!immutable_db_options_.atomic_flush) {
        FlushRequest flush_req;
        // TODO: a new flush reason for ingesting memtable
        GenerateFlushRequest({cfd}, FlushReason::kExternalFileIngestion,
                             &flush_req);
        EnqueuePendingFlush(flush_req);
      }
    }
    if (immutable_db_options_.atomic_flush) {
      AssignAtomicFlushSeq(cfds);
      FlushRequest flush_req;
      flush_req.atomic_flush = true;
      GenerateFlushRequest(cfds, FlushReason::kExternalFileIngestion,
                           &flush_req);
      EnqueuePendingFlush(flush_req);
    }
    MaybeScheduleFlushOrCompaction();
  }
  return s;
}

struct DBImpl::BlobDirectWriteContext {
  struct Entry {
    // Hold a referenced SuperVersion so BDW uses the same published mutable
    // option snapshot as the rest of the DB while keeping the CFD alive.
    SuperVersion* super_version = nullptr;
    BlobFilePartitionManager* partition_mgr = nullptr;
    BlobDirectWriteSettings settings;
  };

  explicit BlobDirectWriteContext(DBImpl* db_impl) : db_impl_(db_impl) {}
  BlobDirectWriteContext(const BlobDirectWriteContext&) = delete;
  BlobDirectWriteContext& operator=(const BlobDirectWriteContext&) = delete;
  BlobDirectWriteContext(BlobDirectWriteContext&&) = delete;
  BlobDirectWriteContext& operator=(BlobDirectWriteContext&&) = delete;

  ~BlobDirectWriteContext() { Release(); }

  BlobDirectWriteSettings GetSettings(uint32_t cf_id,
                                      bool lookup_from_write_thread) {
    return GetOrCreate(cf_id, lookup_from_write_thread).settings;
  }

  BlobFilePartitionManager* GetPartitionManager(uint32_t cf_id,
                                                bool lookup_from_write_thread) {
    return GetOrCreate(cf_id, lookup_from_write_thread).partition_mgr;
  }

  void Release() {
    if (entries_.empty()) {
      return;
    }

    for (auto& entry : entries_) {
      if (entry.second.super_version != nullptr) {
        db_impl_->CleanupSuperVersion(entry.second.super_version);
      }
    }
    entries_.clear();
  }

  UnorderedSet<BlobFilePartitionManager*> touched_managers;
  std::vector<BlobWriteBatchTransformer::RollbackInfo> rollback_infos;

 private:
  static BlobDirectWriteSettings BuildSettings(const SuperVersion* sv) {
    assert(sv != nullptr);

    BlobDirectWriteSettings settings;
    settings.enable_blob_direct_write =
        sv->cfd->ioptions().enable_blob_direct_write;
    settings.min_blob_size = sv->mutable_cf_options.min_blob_size;
    settings.compression_type = sv->mutable_cf_options.blob_compression_type;
    settings.compression_opts = sv->mutable_cf_options.blob_compression_opts;
    settings.blob_cache = sv->cfd->ioptions().blob_cache.get();
    settings.prepopulate_blob_cache =
        sv->mutable_cf_options.prepopulate_blob_cache;
    return settings;
  }

  SuperVersion* AcquireReferencedSuperVersion(uint32_t cf_id,
                                              bool lookup_from_write_thread) {
    if (lookup_from_write_thread) {
      auto* cfd =
          db_impl_->versions_->GetColumnFamilySet()->GetColumnFamily(cf_id);
      if (cfd != nullptr) {
        return cfd->GetReferencedSuperVersion(db_impl_);
      }
      return nullptr;
    }

    ColumnFamilyData* cfd = nullptr;
    {
      InstrumentedMutexLock lock(&db_impl_->mutex_);
      cfd = db_impl_->versions_->GetColumnFamilySet()->GetColumnFamily(cf_id);
      if (cfd != nullptr) {
        // Hold the CFD long enough to acquire a referenced SuperVersion after
        // dropping DB mutex. This reuses the existing SuperVersion publication
        // path for mutable options rather than maintaining a separate BDW
        // settings snapshot. A future attached-WriteBatch path could pin the
        // needed CFDs up front and avoid this DB mutex lookup entirely.
        cfd->Ref();
      }
    }
    if (cfd == nullptr) {
      return nullptr;
    }

    SuperVersion* sv = cfd->GetReferencedSuperVersion(db_impl_);
    {
      InstrumentedMutexLock lock(&db_impl_->mutex_);
      cfd->UnrefAndTryDelete();
    }
    return sv;
  }

  Entry& GetOrCreate(uint32_t cf_id, bool lookup_from_write_thread) {
    auto entry_it = entries_.find(cf_id);
    if (entry_it != entries_.end()) {
      return entry_it->second;
    }

    Entry entry;
    if (SuperVersion* sv =
            AcquireReferencedSuperVersion(cf_id, lookup_from_write_thread)) {
      entry.super_version = sv;
      entry.partition_mgr = sv->cfd->blob_partition_manager();
      if (entry.partition_mgr != nullptr) {
        entry.settings = BuildSettings(sv);
      }
    }

    return entries_.emplace(cf_id, std::move(entry)).first->second;
  }

  DBImpl* db_impl_;
  std::unordered_map<uint32_t, Entry> entries_;
};

Status DBImpl::MaybeTransformBatchForBlobDirectWrite(
    const WriteOptions& write_options, WriteBatch** batch,
    bool should_write_to_memtable, bool lookup_from_write_thread,
    WriteBatch* transformed_storage,
    BlobDirectWriteContext* blob_direct_write_ctx) {
  assert(batch != nullptr);
  assert(transformed_storage != nullptr);
  assert(blob_direct_write_ctx != nullptr);

  if (!should_write_to_memtable || *batch == nullptr ||
      (!(*batch)->HasPut() && !(*batch)->HasPutEntity()) ||
      (*batch)->HasMerge()) {
    return Status::OK();
  }

  // Quick check: skip the (potentially expensive) batch iteration when no
  // column family has blob direct write enabled. This also avoids Iterate()
  // returning an error on a corrupted batch before the WAL verification path
  // has a chance to detect it.
  if (!HasAnyBlobDirectWriteColumnFamily()) {
    return Status::OK();
  }

  std::vector<BlobFilePartitionManager*> batch_managers;
  std::vector<BlobWriteBatchTransformer::RollbackInfo> batch_rollbacks;
  bool transformed = false;
  transformed_storage->Clear();

  auto settings_provider = [&](uint32_t cf_id) -> BlobDirectWriteSettings {
    return blob_direct_write_ctx->GetSettings(cf_id, lookup_from_write_thread);
  };
  auto partition_mgr_provider =
      [&](uint32_t cf_id) -> BlobFilePartitionManager* {
    return blob_direct_write_ctx->GetPartitionManager(cf_id,
                                                      lookup_from_write_thread);
  };
  Status s = BlobWriteBatchTransformer::TransformBatch(
      write_options, *batch, transformed_storage, partition_mgr_provider,
      settings_provider, &transformed, &batch_managers, &batch_rollbacks);
  if (!s.ok()) {
    return s;
  }
  if (transformed) {
    *batch = transformed_storage;
    blob_direct_write_ctx->touched_managers.insert(batch_managers.begin(),
                                                   batch_managers.end());
    blob_direct_write_ctx->rollback_infos.insert(
        blob_direct_write_ctx->rollback_infos.end(), batch_rollbacks.begin(),
        batch_rollbacks.end());
  }
  return Status::OK();
}

Status DBImpl::AppendPreprocessedPutEntityToBatch(
    const WriteOptions& write_options, WriteBatch* batch,
    ColumnFamilyHandle* column_family, const Slice& key,
    const WideColumns& columns, BlobDirectWriteContext* blob_direct_write_ctx) {
  assert(batch != nullptr);

  Status s;
  uint32_t cf_id = 0;
  size_t ts_sz = 0;
  std::tie(s, cf_id, ts_sz) =
      WriteBatchInternal::GetColumnFamilyIdAndTimestampSize(batch,
                                                            column_family);
  if (s.ok()) {
    assert(ts_sz == 0);

    WideColumns sorted_columns(columns);
    WideColumnsHelper::SortColumns(sorted_columns);

    return AppendSortedPutEntityToBatch(write_options, batch, cf_id, key,
                                        sorted_columns, blob_direct_write_ctx);
  }
  return s;
}

Status DBImpl::AppendSortedPutEntityToBatch(
    const WriteOptions& write_options, WriteBatch* batch, uint32_t cf_id,
    const Slice& key, const WideColumns& sorted_columns,
    BlobDirectWriteContext* blob_direct_write_ctx) {
  assert(batch != nullptr);

  Status s;
  BlobDirectWriteSettings settings;
  BlobFilePartitionManager* partition_mgr = nullptr;
  std::vector<BlobWriteBatchTransformer::RollbackInfo>* rollback_infos =
      nullptr;
  if (blob_direct_write_ctx != nullptr) {
    settings = blob_direct_write_ctx->GetSettings(
        cf_id, /*lookup_from_write_thread=*/false);
    partition_mgr = blob_direct_write_ctx->GetPartitionManager(
        cf_id, /*lookup_from_write_thread=*/false);
    rollback_infos = &blob_direct_write_ctx->rollback_infos;
  }

  std::string entity;
  bool transformed = false;
  s = BlobWriteBatchTransformer::MaybePreprocessWideColumns(
      write_options, cf_id, key, sorted_columns, partition_mgr, settings,
      /*serialize_inline_entity=*/true, &entity, &transformed, rollback_infos);
  if (s.ok()) {
    if (transformed && blob_direct_write_ctx != nullptr) {
      blob_direct_write_ctx->touched_managers.insert(partition_mgr);
    }

    return WriteBatchInternal::PutEntitySerialized(batch, cf_id, key, entity);
  }
  return s;
}

Status DBImpl::PutEntityFastPath(const WriteOptions& write_options,
                                 ColumnFamilyHandle* column_family,
                                 const Slice& key, const WideColumns& columns) {
  const ColumnFamilyHandle* const default_cf = DefaultColumnFamily();
  assert(default_cf);

  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);

  WriteBatch batch(/*reserved_bytes=*/0, /*max_bytes=*/0,
                   write_options.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  std::unique_ptr<WriteBatch> trace_batch;
  bool needs_trace_batch = false;
  {
    InstrumentedMutexLock lock(&trace_mutex_);
    needs_trace_batch = tracer_ != nullptr;
  }
  if (needs_trace_batch) {
    // Keep tracing on the original logical `PutEntity()` op. The applied batch
    // can stay empty here and be materialized later inside `WriteImpl()`.
    trace_batch = std::make_unique<WriteBatch>(
        /*reserved_bytes=*/0, /*max_bytes=*/0,
        /*protection_bytes_per_key=*/0, default_cf_ucmp->timestamp_size());
    Status trace_s = trace_batch->PutEntity(column_family, key, columns);
    if (!trace_s.ok()) {
      return trace_s;
    }
  }

  auto* cfh = static_cast_with_check<ColumnFamilyHandleImpl>(column_family);
  const bool needs_deferred_transform =
      cfh->cfd()->blob_partition_manager() != nullptr;
  std::optional<DeferredPutEntityBatch> deferred_put_entities;
  if (needs_deferred_transform) {
    WideColumns sorted_columns(columns);
    WideColumnsHelper::SortColumns(sorted_columns);
    deferred_put_entities.emplace();
    deferred_put_entities->ops.push_back(
        {cfh->cfd()->GetID(), key.ToString(), std::move(sorted_columns)});
  } else {
    Status append_s = AppendPreprocessedPutEntityToBatch(
        write_options, &batch, column_family, key, columns,
        /*blob_direct_write_ctx=*/nullptr);
    if (!append_s.ok()) {
      return append_s;
    }
  }

  Status write_s = WritePreprocessedPutEntityBatch(
      write_options, &batch, trace_batch ? trace_batch.get() : nullptr,
      deferred_put_entities ? &deferred_put_entities.value() : nullptr);
  return write_s;
}

Status DBImpl::PutEntityFastPath(const WriteOptions& write_options,
                                 const Slice& key,
                                 const AttributeGroups& attribute_groups) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);

  WriteBatch batch(/*reserved_bytes=*/0, /*max_bytes=*/0,
                   write_options.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  std::unique_ptr<WriteBatch> trace_batch;
  bool needs_trace_batch = false;
  {
    InstrumentedMutexLock lock(&trace_mutex_);
    needs_trace_batch = tracer_ != nullptr;
  }
  if (needs_trace_batch) {
    // Keep tracing on the original logical `PutEntity()` op. The applied batch
    // can stay empty here and be materialized later inside `WriteImpl()`.
    trace_batch = std::make_unique<WriteBatch>(
        /*reserved_bytes=*/0, /*max_bytes=*/0,
        /*protection_bytes_per_key=*/0, default_cf_ucmp->timestamp_size());
    Status trace_s = trace_batch->PutEntity(key, attribute_groups);
    if (!trace_s.ok()) {
      return trace_s;
    }
  }

  bool needs_deferred_transform = false;
  for (const AttributeGroup& ag : attribute_groups) {
    auto* cfh =
        static_cast_with_check<ColumnFamilyHandleImpl>(ag.column_family());
    if (cfh->cfd()->blob_partition_manager() != nullptr) {
      needs_deferred_transform = true;
      break;
    }
  }

  std::optional<DeferredPutEntityBatch> deferred_put_entities;
  if (needs_deferred_transform) {
    deferred_put_entities.emplace();
  }

  for (const AttributeGroup& ag : attribute_groups) {
    auto* cfh =
        static_cast_with_check<ColumnFamilyHandleImpl>(ag.column_family());
    if (deferred_put_entities.has_value()) {
      WideColumns sorted_columns(ag.columns());
      WideColumnsHelper::SortColumns(sorted_columns);
      deferred_put_entities->ops.push_back(
          {cfh->cfd()->GetID(), key.ToString(), std::move(sorted_columns)});
      continue;
    }

    Status append_s = AppendPreprocessedPutEntityToBatch(
        write_options, &batch, ag.column_family(), key, ag.columns(),
        /*blob_direct_write_ctx=*/nullptr);
    if (!append_s.ok()) {
      return append_s;
    }
  }

  Status write_s = WritePreprocessedPutEntityBatch(
      write_options, &batch, trace_batch ? trace_batch.get() : nullptr,
      deferred_put_entities ? &deferred_put_entities.value() : nullptr);
  return write_s;
}

Status DBImpl::WritePreprocessedPutEntityBatch(
    const WriteOptions& write_options, WriteBatch* batch,
    WriteBatch* trace_batch,
    const DeferredPutEntityBatch* deferred_put_entities) {
  Status s;
  if (deferred_put_entities == nullptr &&
      write_options.protection_bytes_per_key > 0) {
    s = WriteBatchInternal::UpdateProtectionInfo(
        batch, write_options.protection_bytes_per_key);
  }
  if (s.ok()) {
    PutEntityFastPathWriteCallback no_batching_callback;
    WriteCallback* callback =
        deferred_put_entities != nullptr ? &no_batching_callback : nullptr;
    // Pass the logical trace batch separately since `batch` may be rebuilt in
    // `WriteImpl()` after `PreprocessWrite()` selects the target generation.
    s = WriteImpl(write_options, batch, callback,
                  /*user_write_cb=*/nullptr,
                  /*wal_used=*/nullptr, /*log_ref=*/0,
                  /*disable_memtable=*/false, /*seq_used=*/nullptr,
                  /*batch_cnt=*/0, /*pre_release_callback=*/nullptr,
                  /*post_memtable_callback=*/nullptr,
                  /*wbwi=*/nullptr, trace_batch,
                  /*skip_blob_direct_write_transform=*/true,
                  deferred_put_entities);
  }
  return s;
}

Status DBImpl::SyncBlobDirectWriteManagers(
    const WriteOptions& write_options,
    const BlobDirectWriteContext& blob_direct_write_ctx) {
  for (auto* mgr : blob_direct_write_ctx.touched_managers) {
    Status blob_s = write_options.sync ? mgr->SyncAllOpenFiles(write_options)
                                       : mgr->FlushAllOpenFiles(write_options);
    if (!blob_s.ok()) {
      return blob_s;
    }
  }

  if (!blob_direct_write_ctx.touched_managers.empty()) {
    // Tests can inject a post-transform failure after blob bytes are durable
    // enough to require manifest-time garbage accounting.
    Status blob_s;
    TEST_SYNC_POINT_CALLBACK("DBImpl::WriteImpl:AfterBlobDirectWrite", &blob_s);
    if (!blob_s.ok()) {
      return blob_s;
    }
  }

  return Status::OK();
}

Status DBImpl::WriteImpl(
    const WriteOptions& write_options, WriteBatch* my_batch,
    WriteCallback* callback, UserWriteCallback* user_write_cb,
    uint64_t* wal_used, uint64_t log_ref, bool disable_memtable,
    uint64_t* seq_used, size_t batch_cnt,
    PreReleaseCallback* pre_release_callback,
    PostMemTableCallback* post_memtable_callback,
    std::shared_ptr<WriteBatchWithIndex> wbwi, WriteBatch* trace_batch_override,
    bool skip_blob_direct_write_transform,
    const DBImpl::DeferredPutEntityBatch* deferred_put_entities) {
  assert(!seq_per_batch_ || batch_cnt != 0);
  assert(my_batch == nullptr || my_batch->Count() == 0 ||
         write_options.protection_bytes_per_key == 0 ||
         write_options.protection_bytes_per_key ==
             my_batch->GetProtectionBytesPerKey());
  if (my_batch == nullptr) {
    return Status::InvalidArgument("Batch is nullptr!");
  } else if (!disable_memtable &&
             WriteBatchInternal::TimestampsUpdateNeeded(*my_batch)) {
    // If writing to memtable, then we require the caller to set/update the
    // timestamps for the keys in the write batch.
    // Otherwise, it means we are just writing to the WAL, and we allow
    // timestamps unset for the keys in the write batch. This can happen if we
    // use TransactionDB with write-committed policy, and we currently do not
    // support user-defined timestamp with other policies.
    // In the prepare phase, a transaction can write the batch to the WAL
    // without inserting to memtable. The keys in the batch do not have to be
    // assigned timestamps because they will be used only during recovery if
    // there is a commit marker which includes their commit timestamp.
    return Status::InvalidArgument("write batch must have timestamp(s) set");
  } else if (write_options.rate_limiter_priority != Env::IO_TOTAL &&
             write_options.rate_limiter_priority != Env::IO_USER) {
    return Status::InvalidArgument(
        "WriteOptions::rate_limiter_priority only allows "
        "Env::IO_TOTAL and Env::IO_USER due to implementation constraints");
  } else if (write_options.rate_limiter_priority != Env::IO_TOTAL &&
             (write_options.disableWAL || manual_wal_flush_)) {
    return Status::InvalidArgument(
        "WriteOptions::rate_limiter_priority currently only supports "
        "rate-limiting automatic WAL flush, which requires "
        "`WriteOptions::disableWAL` and "
        "`DBOptions::manual_wal_flush` both set to false");
  } else if (write_options.protection_bytes_per_key != 0 &&
             write_options.protection_bytes_per_key != 8) {
    return Status::InvalidArgument(
        "`WriteOptions::protection_bytes_per_key` must be zero or eight");
  } else if (write_options.disableWAL &&
             immutable_db_options_.recycle_log_file_num > 0 &&
             !(two_write_queues_ && disable_memtable)) {
    // Corruption detection in recycled WALs relies on sequential sequence
    // numbers, but WritePreparedTxnDB uses disableWAL internally for split
    // writes
    return Status::InvalidArgument(
        "WriteOptions::disableWAL option is not supported if "
        "DBOptions::recycle_log_file_num > 0");
  }
  const size_t wal_partition = WalPartitionForWrite(
      *my_batch, deferred_put_entities, NumWalPartitions());
  WriteBatch* trace_batch =
      trace_batch_override != nullptr ? trace_batch_override : my_batch;
  // Non-preserved-order tracing runs before the write joins a group, so choose
  // the logical batch here. Preserved-order tracing later uses
  // `Writer::trace_batch` on each grouped writer.

  // TODO: this use of operator bool on `tracer_` can avoid unnecessary lock
  // grabs but does not seem thread-safe.
  if (tracer_) {
    InstrumentedMutexLock lock(&trace_mutex_);
    if (tracer_ && !tracer_->IsWriteOrderPreserved()) {
      // We don't have to preserve write order so can trace anywhere. It's more
      // efficient to trace here than to add latency to a phase of the log/apply
      // pipeline.
      // TODO: maybe handle the tracing status?
      tracer_->Write(trace_batch).PermitUncheckedError();
    }
  }
  if (write_options.sync && write_options.disableWAL) {
    return Status::InvalidArgument("Sync writes has to enable WAL.");
  }
  if (two_write_queues_ && immutable_db_options_.enable_pipelined_write) {
    return Status::NotSupported(
        "pipelined_writes is not compatible with concurrent prepares");
  }
  if (seq_per_batch_ && immutable_db_options_.enable_pipelined_write) {
    // TODO(yiwu): update pipeline write with seq_per_batch and batch_cnt
    return Status::NotSupported(
        "pipelined_writes is not compatible with seq_per_batch");
  }
  if (immutable_db_options_.unordered_write &&
      immutable_db_options_.enable_pipelined_write) {
    return Status::NotSupported(
        "pipelined_writes is not compatible with unordered_write");
  }
  if (immutable_db_options_.enable_pipelined_write &&
      post_memtable_callback != nullptr) {
    return Status::NotSupported(
        "pipelined write currently does not honor post_memtable_callback");
  }
  if (seq_per_batch_ && post_memtable_callback != nullptr) {
    return Status::NotSupported(
        "seq_per_batch currently does not honor post_memtable_callback");
  }
  if (my_batch->HasDeleteRange() && immutable_db_options_.row_cache) {
    return Status::NotSupported(
        "DeleteRange is not compatible with row cache.");
  }
  // Whether the WBWI is from transaction commit or a direct write
  // (IngestWriteBatchWithIndex())
  bool ingest_wbwi_for_commit = false;
  if (wbwi) {
    if (my_batch->HasCommit()) {
      ingest_wbwi_for_commit = true;
      assert(log_ref);
    } else {
      // Only supports disableWAL for directly ingesting WBWI for now.
      assert(write_options.disableWAL);
    }
    assert(!callback);
    if (immutable_db_options_.unordered_write) {
      return Status::NotSupported(
          "Ingesting WriteBatch does not support unordered_write");
    }
    if (immutable_db_options_.enable_pipelined_write) {
      return Status::NotSupported(
          "Ingesting WriteBatch does not support pipelined_write");
    }
    if (!wbwi->GetOverwriteKey()) {
      return Status::NotSupported(
          "WriteBatchWithIndex ingestion requires overwrite_key=true");
    }
    // Ingestion switches memtables, which assumes no other leader is
    // inserting; at N>1 other partitions' leaders may be. This rejects
    // both direct ingestion (IngestWriteBatchWithIndex) and commit-time
    // ingestion.
    if (NumWalPartitions() > 1) {
      return Status::NotSupported(
          "WriteBatchWithIndex ingestion is not supported with "
          "wal_partitions > 1");
    }
  }
  // Otherwise IsLatestPersistentState optimization does not make sense
  assert(!WriteBatchInternal::IsLatestPersistentState(my_batch) ||
         disable_memtable);

  if (write_options.low_pri) {
    Status s = ThrottleLowPriWritesIfNeeded(write_options, my_batch);
    if (!s.ok()) {
      return s;
    }
  }

  if (two_write_queues_ && disable_memtable) {
    AssignOrder assign_order =
        seq_per_batch_ ? kDoAssignOrder : kDontAssignOrder;
    // Otherwise it is WAL-only Prepare batches in WriteCommitted policy and
    // they don't consume sequence.
    return WriteImplWALOnly(
        &NonmemWriteThread(wal_partition), write_options, my_batch, my_batch,
        callback, user_write_cb, wal_used, log_ref, seq_used, batch_cnt,
        pre_release_callback, assign_order, kDontPublishLastSeq,
        disable_memtable, wal_partition);
  }

  const bool maybe_use_blob_direct_write = wbwi == nullptr &&
                                           !skip_blob_direct_write_transform &&
                                           HasAnyBlobDirectWriteColumnFamily();
  const bool maybe_use_deferred_put_entity =
      deferred_put_entities != nullptr && wbwi == nullptr &&
      HasAnyBlobDirectWriteColumnFamily();
  std::optional<WriteBatch> transformed_batch_storage;
  std::optional<std::vector<WriteBatch>> transformed_write_group_batches;
  std::optional<BlobDirectWriteContext> blob_direct_write_ctx;
  std::optional<BlobWriteRollbackGuard> blob_write_rollback_guard;
  if (maybe_use_blob_direct_write || maybe_use_deferred_put_entity) {
    blob_direct_write_ctx.emplace(this);
    blob_write_rollback_guard.emplace(&blob_direct_write_ctx->rollback_infos,
                                      immutable_db_options_.info_log.get());
  }
  auto finish_write = [&](Status s) {
    if (s.ok() && blob_write_rollback_guard.has_value()) {
      blob_write_rollback_guard->Dismiss();
    }
    return s;
  };
  if (UNLIKELY(maybe_use_blob_direct_write) &&
      (immutable_db_options_.unordered_write ||
       immutable_db_options_.enable_pipelined_write)) {
    transformed_batch_storage.emplace();
    Status blob_s = MaybeTransformBatchForBlobDirectWrite(
        write_options, &my_batch, !disable_memtable,
        /*lookup_from_write_thread=*/false, &transformed_batch_storage.value(),
        &blob_direct_write_ctx.value());
    if (!blob_s.ok()) {
      return blob_s;
    }
    blob_s = SyncBlobDirectWriteManagers(write_options,
                                         blob_direct_write_ctx.value());
    if (!blob_s.ok()) {
      return blob_s;
    }
  }

  if (immutable_db_options_.unordered_write) {
    const size_t sub_batch_cnt = batch_cnt != 0
                                     ? batch_cnt
                                     // every key is a sub-batch consuming a seq
                                     : WriteBatchInternal::Count(my_batch);
    uint64_t seq = 0;
    // Use a write thread to i) optimize for WAL write, ii) publish last
    // sequence in in increasing order, iii) call pre_release_callback serially
    Status status = WriteImplWALOnly(
        &PartitionWriteThread(wal_partition), write_options, my_batch,
        trace_batch, callback, user_write_cb, wal_used, log_ref, &seq,
        sub_batch_cnt, pre_release_callback, kDoAssignOrder, kDoPublishLastSeq,
        disable_memtable, wal_partition);
    TEST_SYNC_POINT("DBImpl::WriteImpl:UnorderedWriteAfterWriteWAL");
    if (!status.ok()) {
      return status;
    }
    if (seq_used) {
      *seq_used = seq;
    }
    if (!disable_memtable) {
      TEST_SYNC_POINT("DBImpl::WriteImpl:BeforeUnorderedWriteMemtable");
      status = UnorderedWriteMemtable(write_options, my_batch, callback,
                                      log_ref, seq, sub_batch_cnt);
    }
    return finish_write(status);
  }

  if (immutable_db_options_.enable_pipelined_write) {
    return finish_write(PipelinedWriteImpl(
        write_options, my_batch, trace_batch, wal_partition, callback,
        user_write_cb, wal_used, log_ref, disable_memtable, seq_used));
  }

  PERF_TIMER_GUARD(write_pre_and_post_process_time);
  WriteThread::Writer w(write_options, my_batch, callback, user_write_cb,
                        log_ref, disable_memtable, batch_cnt,
                        pre_release_callback, post_memtable_callback,
                        /*_ingest_wbwi=*/wbwi != nullptr, trace_batch);
  w.wal_partition = wal_partition;
  WriteThread& write_thread = PartitionWriteThread(wal_partition);
  StopWatch write_sw(immutable_db_options_.clock, stats_, DB_WRITE);

  {
    // Time blocked waiting to join a write batch group: the queueing that
    // write-partition routing could relieve.
    StopWatch join_sw(immutable_db_options_.clock, stats_,
                      JOIN_BATCH_GROUP_WAIT_MICROS);
    write_thread.JoinBatchGroup(&w);
  }
  if (w.state == WriteThread::STATE_PARALLEL_MEMTABLE_CALLER) {
    write_thread.SetMemWritersEachStride(&w);
  }
  if (w.state == WriteThread::STATE_PARALLEL_MEMTABLE_WRITER) {
    // we are a non-leader in a parallel group

    if (w.ShouldWriteToMemtable()) {
      PERF_TIMER_STOP(write_pre_and_post_process_time);
      PERF_TIMER_FOR_WAIT_GUARD(write_memtable_time);

      ColumnFamilyMemTablesImpl column_family_memtables(
          versions_->GetColumnFamilySet());
      w.status = WriteBatchInternal::InsertInto(
          &w, w.sequence, &column_family_memtables, &flush_scheduler_,
          &trim_history_scheduler_,
          write_options.ignore_missing_column_families, 0 /*log_number*/, this,
          true /*concurrent_memtable_writes*/, seq_per_batch_, w.batch_cnt,
          batch_per_txn_, write_options.memtable_insert_hint_per_batch);

      PERF_TIMER_START(write_pre_and_post_process_time);
    }

    if (write_thread.CompleteParallelMemTableWriter(&w)) {
      // we're responsible for exit batch group
      // TODO(myabandeh): propagate status to write_group
      auto last_sequence = w.write_group->last_sequence;
      for (auto* tmp_w : *(w.write_group)) {
        assert(tmp_w);
        if (tmp_w->post_memtable_callback && !w.write_group->completion_gate) {
          Status tmp_s =
              (*tmp_w->post_memtable_callback)(last_sequence, disable_memtable);
          // TODO: propagate the execution status of post_memtable_callback to
          // caller.
          assert(tmp_s.ok());
        }
      }
      if (w.status.ok()) {  // Don't publish a partial batch write
        PublishLastSequence(w.wal_partition, w.write_group->first_sequence,
                            last_sequence, !w.disable_wal);
      } else {
        HandleMemTableInsertFailure(w.status, w.write_group->first_sequence);
      }
      ExitMemTableSwitchGate();
      write_thread.ExitAsBatchGroupFollower(&w);
    }
    assert(w.state == WriteThread::STATE_COMPLETED);
    // STATE_COMPLETED conditional below handles exit
  }
  if (w.state == WriteThread::STATE_COMPLETED) {
    if (wal_used != nullptr) {
      *wal_used = w.wal_used;
    }
    if (seq_used != nullptr) {
      *seq_used = w.sequence;
    }
    // write is complete and leader has updated sequence
    return finish_write(w.FinalStatus());
  }
  // else we are the leader of the write batch group
  assert(w.state == WriteThread::STATE_GROUP_LEADER);
  Status status;
  // Once reaches this point, the current writer "w" will try to do its write
  // job.  It may also pick up some of the remaining writers in the "writers_"
  // when it finds suitable, and finish them in the same write batch.
  // This is how a write job could be done by the other writer.
  WriteContext write_context;
  // FIXME: also check disableWAL like others?
  // Above one partition a sync write fsyncs from its completion gate, once
  // its group has published (SyncWalsFromCompletionGate), never as leader.
  WalContext wal_context(write_options.sync && NumWalPartitions() == 1);
  WriteThread::WriteGroup write_group;
  bool in_parallel_group = false;
  uint64_t last_sequence = kMaxSequenceNumber;
  // The group's range is registered for N>1 publication, and a step after
  // its WAL append failed before anything was inserted (see the exit).
  bool registered_applied_range = false;
  bool release_unapplied_range = false;

  assert(!two_write_queues_ || !disable_memtable);
  {
    // With concurrent writes we do preprocess only in the write thread that
    // also does write to memtable to avoid sync issue on shared data structure
    // with the other thread

    // PreprocessWrite does its own perf timing.
    PERF_TIMER_STOP(write_pre_and_post_process_time);

    status = PreprocessWrite(write_options, w.wal_partition, &wal_context,
                             &write_context);
    if (!two_write_queues_) {
      // Assign it after ::PreprocessWrite since the sequence might advance
      // inside it by WriteRecoverableState
      last_sequence = versions_->LastSequence();
    }

    PERF_TIMER_START(write_pre_and_post_process_time);
  }

  // Add to log and apply to memtable.  We can release the lock
  // during this phase since &w is currently responsible for logging
  // and protects against concurrent loggers and concurrent writes
  // into memtables

  TEST_SYNC_POINT("DBImpl::WriteImpl:BeforeLeaderEnters");
  last_batch_group_size_[wal_partition].StoreRelaxed(
      write_thread.EnterAsBatchGroupLeader(&w, &write_group));
  if (wbwi) {
    assert(write_group.size == 1);
  }

  IOStatus io_s;
  Status pre_release_cb_status;
  size_t seq_inc = 0;
  uint64_t wal_write_number = 0;
  if (status.ok()) {
    if (deferred_put_entities != nullptr) {
      my_batch->Clear();
      for (const auto& op : deferred_put_entities->ops) {
        status = AppendSortedPutEntityToBatch(
            write_options, my_batch, op.column_family_id, op.key,
            op.sorted_columns,
            blob_direct_write_ctx ? &blob_direct_write_ctx.value() : nullptr);
        if (!status.ok()) {
          break;
        }
      }
      if (status.ok() && write_options.protection_bytes_per_key > 0) {
        status = WriteBatchInternal::UpdateProtectionInfo(
            my_batch, write_options.protection_bytes_per_key);
      }
      if (status.ok() && blob_direct_write_ctx.has_value()) {
        status = SyncBlobDirectWriteManagers(write_options,
                                             blob_direct_write_ctx.value());
      }
    }

    if (UNLIKELY(maybe_use_blob_direct_write) &&
        !immutable_db_options_.unordered_write &&
        !immutable_db_options_.enable_pipelined_write) {
      // PreprocessWrite() may switch memtables in order to schedule pending
      // flushes. Defer the blob direct-write transformation until after that
      // point so the blob file generation matches the memtable that will store
      // the transformed blob index.
      transformed_write_group_batches.emplace();
      transformed_write_group_batches->reserve(write_group.size);
      for (auto* writer : write_group) {
        assert(writer != nullptr);
        if (!writer->ShouldWriteToMemtable()) {
          continue;
        }
        transformed_write_group_batches->emplace_back();
        status = MaybeTransformBatchForBlobDirectWrite(
            write_options, &writer->batch, /*should_write_to_memtable=*/true,
            /*lookup_from_write_thread=*/true,
            &transformed_write_group_batches->back(),
            &blob_direct_write_ctx.value());
        if (!status.ok()) {
          break;
        }
      }
      if (status.ok()) {
        status = SyncBlobDirectWriteManagers(write_options,
                                             blob_direct_write_ctx.value());
      }
    }
  }

  if (status.ok()) {
    // Rules for when we can update the memtable concurrently
    // 1. supported by memtable
    // 2. Puts are not okay if inplace_update_support
    // 3. Merges are not okay
    //
    // Rules 1..2 are enforced by checking the options
    // during startup (CheckConcurrentWritesSupported), so if
    // options.allow_concurrent_memtable_write is true then they can be
    // assumed to be true.  Rule 3 is checked for each batch.  We could
    // relax rules 2 if we could prevent write batches from referring
    // more than once to a particular key.
    // With more than one WAL partition, other partitions' leaders insert into
    // the same memtables, so even a single-writer group inserts concurrently.
    // Rule 3 is lifted there: a Merge batch that left the parallel path
    // would insert non-concurrently while other leaders insert into the
    // same memtable. Concurrent Merge inserts are safe with
    // max_successive_merges == 0, which Open requires at N>1 (as with
    // unordered_write); the successive-merge counting is the only
    // concurrent-unsafe part of the merge path.
    bool parallel = immutable_db_options_.allow_concurrent_memtable_write &&
                    (write_group.size > 1 || NumWalPartitions() > 1);
    size_t total_count = 0;
    size_t valid_batches = 0;
    size_t total_byte_size = 0;
    size_t pre_release_callback_cnt = 0;
    for (auto* writer : write_group) {
      assert(writer);
      if (writer->CheckCallback(this)) {
        valid_batches += writer->batch_cnt;
        if (writer->ShouldWriteToMemtable()) {
          total_count += WriteBatchInternal::Count(writer->batch);
          total_byte_size = WriteBatchInternal::AppendedByteSize(
              total_byte_size, WriteBatchInternal::ByteSize(writer->batch));
          if (NumWalPartitions() == 1) {
            parallel = parallel && !writer->batch->HasMerge();
          }
        }
        if (writer->pre_release_callback) {
          pre_release_callback_cnt++;
        }
      }
    }
    // At N>1 every memtable insert is concurrent (other leaders may be
    // inserting into the same memtable): the non-parallel path below is
    // unreachable, and a regression that revives it fails here loudly.
    assert(parallel || NumWalPartitions() == 1);
    // TODO: this use of operator bool on `tracer_` can avoid unnecessary lock
    // grabs but does not seem thread-safe.
    if (tracer_) {
      InstrumentedMutexLock lock(&trace_mutex_);
      if (tracer_ && tracer_->IsWriteOrderPreserved()) {
        for (auto* writer : write_group) {
          if (writer->CallbackFailed()) {
            continue;
          }
          // TODO: maybe handle the tracing status?
          if (wbwi && !ingest_wbwi_for_commit) {
            // for transaction write, tracer only needs the commit marker which
            // is in writer->batch
            tracer_->Write(wbwi->GetWriteBatch()).PermitUncheckedError();
          } else {
            tracer_->Write(writer->trace_batch).PermitUncheckedError();
          }
        }
      }
    }
    // Note about seq_per_batch_: either disableWAL is set for the entire write
    // group or not. In either case we inc seq for each write batch with no
    // failed callback. This means that there could be a batch with
    // disable_memtable in between; although we do not write this batch to
    // memtable it still consumes a seq. Otherwise, if !seq_per_batch_, we inc
    // the seq per valid written key to mem.
    seq_inc = seq_per_batch_ ? valid_batches : total_count;
    if (wbwi) {
      // Reserve sequence numbers for the ingested memtable. We need to reserve
      // at lease this amount for recovery. During recovery,
      // transactions do not commit by ingesting WBWI. The sequence number
      // associated with the commit entry in WAL is used as the starting
      // sequence number for inserting into memtable. We need to reserve
      // enough sequence numbers here (at least the number of operations
      // in write batch) to assign to memtable entries for this transaction.
      // This prevents updates in different transactions from using out-of-order
      // sequence numbers or the same key+seqno.
      //
      // WBWI ingestion requires not grouping writes, so we don't need to
      // consider incrementing sequence number for WBWI from other writers.
      seq_inc += wbwi->GetWriteBatch()->Count();
    }

    const bool concurrent_update = two_write_queues_ || NumWalPartitions() > 1;
    // Update stats while we are an exclusive group leader, so we know
    // that nobody else can be writing to these particular stats.
    // We're optimistic, updating the stats before we successfully
    // commit.  That lets us release our leader status early.
    auto stats = default_cf_internal_stats_;
    stats->AddDBStats(InternalStats::kIntStatsNumKeysWritten, total_count,
                      concurrent_update);
    RecordTick(stats_, NUMBER_KEYS_WRITTEN, total_count);
    stats->AddDBStats(InternalStats::kIntStatsBytesWritten, total_byte_size,
                      concurrent_update);
    RecordTick(stats_, BYTES_WRITTEN, total_byte_size);
    stats->AddDBStats(InternalStats::kIntStatsWriteDoneBySelf, 1,
                      concurrent_update);
    RecordTick(stats_, WRITE_DONE_BY_SELF);
    auto write_done_by_other = write_group.size - 1;
    if (write_done_by_other > 0) {
      stats->AddDBStats(InternalStats::kIntStatsWriteDoneByOther,
                        write_done_by_other, concurrent_update);
      RecordTick(stats_, WRITE_DONE_BY_OTHER, write_done_by_other);
    }
    RecordInHistogram(stats_, BYTES_PER_WRITE, total_byte_size);
    // One sample per write group: the group-size distribution, and the number
    // of compatible writers the size cap turned away.
    RecordInHistogram(stats_, WRITERS_PER_WRITE_GROUP, write_group.size);
    RecordInHistogram(stats_, WRITE_GROUP_CAP_EXCLUDED,
                      write_group.num_cap_excluded);

    if (write_options.disableWAL) {
      has_unpersisted_data_.store(true, std::memory_order_relaxed);
    }

    PERF_TIMER_STOP(write_pre_and_post_process_time);

    uint64_t wal_index = 0;
    if (!two_write_queues_) {
      // With more than one WAL partition, leaders allocate concurrently, so
      // the LastSequence() read above no longer reserves anything.
      if (status.ok() &&
          (log::WALIndexEnabled(immutable_db_options_.partition_wal_usage) ||
           NumWalPartitions() > 1)) {
        // Overwrites the last_sequence read after PreprocessWrite above; that
        // read still serves the kNone path, which never reaches the allocator.
        // The two agree because the allocator catches last_allocated_sequence_
        // up to LastSequence() before reserving.
        // Register only above one partition: at N==1 nothing consumes
        // the tracker (the gate and its snapshots are all N>1-gated), so
        // an indexed N=1 write allocates its index without entering it.
        io_s = AllocateSequenceAndWALIndex(
            seq_inc, !write_options.disableWAL, &last_sequence, &wal_index,
            wal_context.wal_partition,
            /*register_durability=*/NumWalPartitions() > 1);
        status = io_s;
        registered_applied_range = io_s.ok() && NumWalPartitions() > 1 &&
                                   (!write_options.disableWAL || seq_inc > 0);
      }
      if (status.ok() && !write_options.disableWAL) {
        assert(wal_context.wal_file_number_size);
        wal_write_number = wal_context.wal_file_number_size->number;
        wal_context.prev_size = wal_context.writer->file()->GetFileSize();
        PERF_TIMER_GUARD(write_wal_time);
        // Paired with the Allocate above: only complete what was
        // registered. At N==1 this skips the completion's mutex
        // acquisition: nothing at N==1 reads the tracker's answer.
        const bool complete_durability = NumWalPartitions() > 1;
        io_s = WriteGroupToWAL(write_group, wal_context, wal_used,
                               last_sequence + 1, wal_index, seq_inc,
                               complete_durability);
        if (!io_s.ok()) {
          // The index was consumed; if its record never landed, cover it
          // before the error escapes, so the hole is declared rather than
          // looking like data that was written and lost. Keep the write
          // error: it is what the caller has to see.
          const IOStatus cover_io_s =
              CoverBurnedWALIndex(write_options, wal_context.writer, wal_index);
          cover_io_s.PermitUncheckedError();
        }
      }
    } else {
      if (status.ok() && !write_options.disableWAL) {
        PERF_TIMER_GUARD(write_wal_time);
        // LastAllocatedSequence is increased inside WriteToWAL under
        // the selected partition's WalWriteMutex to ensure ordered events
        // in WAL
        io_s = ConcurrentWriteGroupToWAL(write_group, wal_used, &last_sequence,
                                         seq_inc);
      } else {
        // Otherwise we inc seq number for memtable writes
        last_sequence = versions_->FetchAddLastAllocatedSequence(seq_inc);
      }
    }
    status = io_s;
    assert(last_sequence != kMaxSequenceNumber);
    const SequenceNumber current_sequence = last_sequence + 1;
    last_sequence += seq_inc;
    // Seqno assigned to this write are [current_sequence, last_sequence]

    if (wal_context.need_wal_sync) {
      VersionEdit synced_wals;
      LockWalWriteMutexes();
      if (status.ok()) {
        MarkLogsSynced(wal_context.wal_sync_up_to,
                       wal_context.need_wal_dir_sync, &synced_wals,
                       wal_context.wal_sync_claimed);
      } else {
        MarkLogsNotSynced(wal_context.wal_sync_up_to,
                          wal_context.wal_sync_claimed);
      }
      UnlockWalWriteMutexes();
      if (status.ok() && synced_wals.IsWalAddition()) {
        InstrumentedMutexLock l(&mutex_);
        // TODO: plumb Env::IOActivity, Env::IOPriority
        const ReadOptions read_options;
        status = ApplyWALToManifest(read_options, write_options, &synced_wals);
      }

      // Requesting sync with two_write_queues_ is expected to be very rare. We
      // hence provide a simple implementation that is not necessarily
      // efficient.
      if (status.ok() && two_write_queues_) {
        if (manual_wal_flush_) {
          status = FlushWAL(true);
        } else {
          status = SyncWAL();
        }
      }
    }

    // PreReleaseCallback is called after WAL write and before memtable write
    if (status.ok()) {
      SequenceNumber next_sequence = current_sequence;
      size_t index = 0;
      // Note: the logic for advancing seq here must be consistent with the
      // logic in WriteBatchInternal::InsertInto(write_group...) as well as
      // with WriteBatchInternal::InsertInto(write_batch...) that is called on
      // the merged batch during recovery from the WAL.
      for (auto* writer : write_group) {
        if (writer->CallbackFailed()) {
          continue;
        }
        writer->sequence = next_sequence;
        if (writer->pre_release_callback) {
          Status ws = writer->pre_release_callback->Callback(
              writer->sequence, disable_memtable, writer->wal_used, index++,
              pre_release_callback_cnt);
          if (!ws.ok()) {
            status = pre_release_cb_status = ws;
            break;
          }
        }
        if (seq_per_batch_) {
          assert(writer->batch_cnt);
          next_sequence += writer->batch_cnt;
        } else if (writer->ShouldWriteToMemtable()) {
          next_sequence += WriteBatchInternal::Count(writer->batch);
        }
      }
    }

    // Completion gate: this group's WAL record is registered and its append
    // succeeded, so followers must not complete until the frontier covers the
    // group (sequence + count - 1, which is last_sequence by now).
    // Installed before LaunchParallelMemTableWriters so the last worker out
    // runs it too; ExitAsBatchGroupLeader runs it after promoting the next
    // leader. disableWAL groups never registered for durability and get the
    // publication-only gate below; a failed append stays pinned and must
    // never be waited on. At one partition the first condition is false and
    // the exit path is untouched.
    if (NumWalPartitions() > 1 && !write_options.disableWAL && io_s.ok()) {
      TEST_SYNC_POINT_CALLBACK("DBImpl::InstallCompletionGate", &last_sequence);
      const SequenceNumber gate_low = current_sequence;
      const SequenceNumber gate_target = last_sequence;
      const bool gate_sync = write_options.sync;
      const Env::IOPriority gate_io_priority =
          write_group.leader->rate_limiter_priority;
      write_group.completion_gate = [this, gate_low, gate_target, gate_sync,
                                     gate_io_priority]() {
        // D-G11 (c): a sync write fsyncs every partition here, after its
        // group published, so no write on any partition waits for it. An
        // async write never syncs; its gate waits only for every record at
        // or below it to be written.
        SequenceNumber pre_sync_safe_point = 0;
        uint64_t sync_micros = 0;
        if (gate_sync) {
          const uint64_t sync_start = WalPartitionTimerStart();
          Status s = RunCoalescedGateSync(
              gate_low, gate_target, &pre_sync_safe_point,
              [&]() { return SyncWalsFromCompletionGate(gate_io_priority); });
          sync_micros += WalPartitionTimerElapsed(sync_start);
          if (!s.ok()) {
            return s;
          }
        }
        return RunCompletionGate(gate_low, gate_target, gate_sync,
                                 pre_sync_safe_point, sync_micros);
      };
    } else if (NumWalPartitions() > 1 && seq_inc > 0 && io_s.ok()) {
      // A disableWAL group has nothing to make durable, but its sequences
      // become visible only once every range below them is applied. Waiting
      // for that keeps OK meaning read-your-writes.
      const SequenceNumber gate_target = last_sequence;
      write_group.completion_gate = [this, gate_target]() {
        return WaitForWalDurability(gate_target, /*wal_record=*/false);
      };
    }
    bool has_post_memtable_callback = false;
    if (write_group.completion_gate) {
      for (auto* writer : write_group) {
        has_post_memtable_callback |= writer->post_memtable_callback != nullptr;
      }
    }
    if (has_post_memtable_callback) {
      // A post-memtable callback (CommitAndTryCreateSnapshot's timestamped
      // snapshot) runs once the gate has published every sequence at or below
      // the group's, never from inside the group; the exits below skip it.
      // Once the group publishes, a higher snapshot, or a flush that drops a
      // version the callback's snapshot must see, can come before the
      // callback, so reserve that snapshot's place in snapshots_ now: the
      // group's range is unapplied, so no published sequence reaches it yet.
      // Released with the group, after the callback has created its own.
      int64_t unix_time = 0;
      immutable_db_options_.clock->GetCurrentTime(&unix_time)
          .PermitUncheckedError();  // Ignore error
      SnapshotImpl* reserved = new SnapshotImpl;
      {
        InstrumentedMutexLock l(&mutex_);
        // A reservation, not a user snapshot: user-visible snapshot counts
        // and listings skip it, while compaction and flush still see it.
        snapshots_.New(reserved, last_sequence, unix_time,
                       /*is_write_conflict_boundary=*/true,
                       /*ts=*/std::numeric_limits<uint64_t>::max(),
                       /*is_write_reservation=*/true);
      }
      std::shared_ptr<const Snapshot> reserved_snapshot(
          reserved, [this](const Snapshot* s) { ReleaseSnapshot(s); });
      std::function<Status()> gate = std::move(write_group.completion_gate);
      const WriteThread::WriteGroup* const group = &write_group;
      const SequenceNumber group_last_sequence = last_sequence;
      write_group.completion_gate = [gate = std::move(gate), group,
                                     group_last_sequence, disable_memtable,
                                     reserved_snapshot]() {
        Status s = gate();
        if (s.ok()) {
          for (auto* writer : *group) {
            if (writer->post_memtable_callback) {
              Status tmp_s = (*writer->post_memtable_callback)(
                  group_last_sequence, disable_memtable);
              assert(tmp_s.ok());
            }
          }
        }
        return s;
      };
    }

    release_unapplied_range =
        registered_applied_range && io_s.ok() && !status.ok();

    if (status.ok()) {
      PERF_TIMER_FOR_WAIT_GUARD(write_memtable_time);

      if (!parallel) {
        // w.sequence will be set inside InsertInto
        w.status = WriteBatchInternal::InsertInto(
            write_group, current_sequence, column_family_memtables_.get(),
            &flush_scheduler_, &trim_history_scheduler_,
            write_options.ignore_missing_column_families,
            0 /*recovery_log_number*/, this, seq_per_batch_, batch_per_txn_);
      } else {
        write_group.first_sequence = current_sequence;
        write_group.last_sequence = last_sequence;
        if (NumWalPartitions() > 1) {
          // A failed insert is recorded under the leader's state mutex
          // (CompleteParallelMemTableWriter), which exists only once the
          // leader has waited. Above one partition every group takes this
          // path, a lone writer that never waited included.
          w.CreateMutex();
        }
        write_thread.LaunchParallelMemTableWriters(&write_group);
        in_parallel_group = true;

        // Each parallel follower is doing each own writes. The leader should
        // also do its own.
        if (w.ShouldWriteToMemtable()) {
          ColumnFamilyMemTablesImpl column_family_memtables(
              versions_->GetColumnFamilySet());
          assert(w.sequence == current_sequence);
          w.status = WriteBatchInternal::InsertInto(
              &w, w.sequence, &column_family_memtables, &flush_scheduler_,
              &trim_history_scheduler_,
              write_options.ignore_missing_column_families, 0 /*log_number*/,
              this, true /*concurrent_memtable_writes*/, seq_per_batch_,
              w.batch_cnt, batch_per_txn_,
              write_options.memtable_insert_hint_per_batch);
        }
      }
      if (seq_used != nullptr) {
        *seq_used = w.sequence;
      }
    }
  }
  PERF_TIMER_START(write_pre_and_post_process_time);

  if (!io_s.ok()) {
    // Check WriteToWAL status
    TEST_SYNC_POINT_CALLBACK("DBImpl::WriteImpl:BeforeWALIOStatusCheck", &io_s);
    WALIOStatusCheck(io_s,
                     wal_context.sync_failed_wal_number != 0
                         ? wal_context.sync_failed_wal_number
                         : wal_write_number,
                     last_sequence, last_sequence + 1 - seq_inc,
                     registered_applied_range ? wal_context.wal_partition
                                              : kNoPinnedWalRange);
  }
  if (!w.CallbackFailed()) {
    if (!io_s.ok()) {
      assert(pre_release_cb_status.ok());
    } else {
      WriteStatusCheck(pre_release_cb_status);
    }
  } else {
    assert(pre_release_cb_status.ok());
  }

  bool should_exit_batch_group = true;
  if (in_parallel_group) {
    // CompleteParallelWorker returns true if this thread should
    // handle exit, false means somebody else did
    should_exit_batch_group = write_thread.CompleteParallelMemTableWriter(&w);
  }
  if (wbwi && status.ok() && w.status.ok()) {
    uint32_t wbwi_count = wbwi->GetWriteBatch()->Count();
    // skip empty batch case
    if (wbwi_count) {
      // w.batch contains (potentially empty) commit time batch updates,
      // only ingest wbwi if w.batch is applied to memtable successfully
      uint32_t memtable_update_count = w.batch->Count();
      // Seqno assigned to this write are [last_seq + 1 - seq_inc, last_seq].
      // seq_inc includes w.batch (memtable updates) and wbwi
      // w.batch gets first `memtable_update_count` sequence numbers.
      // wbwi gets the rest `wbwi_count` sequence numbers.
      assert(seq_inc == memtable_update_count + wbwi_count);
      assert(wbwi_count > 0);
      assert(last_sequence != kMaxSequenceNumber);
      SequenceNumber lb = last_sequence + 1 - wbwi_count;
      SequenceNumber ub = last_sequence;
      if (two_write_queues_) {
        assert(ub <= versions_->LastAllocatedSequence());
      }
      status = IngestWBWIAsMemtable(
          wbwi, {/*lower_bound=*/lb, /*upper_bound=*/ub},
          /*min_prep_log=*/log_ref, last_sequence,
          /*memtable_updated=*/memtable_update_count > 0,
          ingest_wbwi_for_commit, write_options.ignore_missing_column_families);
      RecordTick(stats_, NUMBER_WBWI_INGEST);
    }
  }

  if (should_exit_batch_group) {
    if (status.ok()) {
      for (auto* tmp_w : write_group) {
        assert(tmp_w);
        if (tmp_w->post_memtable_callback && !write_group.completion_gate) {
          Status tmp_s =
              (*tmp_w->post_memtable_callback)(last_sequence, disable_memtable);
          // TODO: propagate the execution status of post_memtable_callback to
          // caller.
          assert(tmp_s.ok());
        }
      }
      // Note: if we are to resume after non-OK statuses we need to revisit how
      // we react to non-OK statuses here.
      if (w.status.ok()) {  // Don't publish a partial batch write
        TEST_SYNC_POINT(
            "DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:pause");
        TEST_SYNC_POINT(
            "DBImpl::WriteImpl:AfterWBWIIngestBeforeSetLastSequence:resume");
        PublishLastSequence(w.wal_partition, last_sequence + 1 - seq_inc,
                            last_sequence, !write_options.disableWAL);
      }
    } else if (release_unapplied_range) {
      // The record landed but a later step failed (recording the sync in the
      // MANIFEST), so the group inserted nothing. Mark its range applied, as
      // one partition does when the next group publishes past it: left
      // pending, it would hold every higher range on every partition, and
      // their completion gates, for good. It publishes no data.
      PublishLastSequence(w.wal_partition, last_sequence + 1 - seq_inc,
                          last_sequence, !write_options.disableWAL);
    }
    if (!w.status.ok()) {
      if (wal_context.prev_size < SIZE_MAX) {
        // The partition this write bound to, not CurrentWal(), which is
        // partition 0 only.
        const size_t bound_partition = wal_context.wal_partition;
        WalWriteMutexLock l(&WalWriteMutex(bound_partition));
        LogWriterNumber& current_wal = CurrentWal(bound_partition);
        if (current_wal.number == wal_context.wal_file_number_size->number) {
          current_wal.SetAttemptTruncateSize(wal_context.prev_size);
        }
      }
      HandleMemTableInsertFailure(w.status, last_sequence + 1 - seq_inc);
    }
    ExitMemTableSwitchGate();
    write_thread.ExitAsBatchGroupLeader(write_group, status);
  }

  if (status.ok()) {
    status = w.FinalStatus();
  }
  return finish_write(status);
}

Status DBImpl::PipelinedWriteImpl(const WriteOptions& write_options,
                                  WriteBatch* my_batch, WriteBatch* trace_batch,
                                  size_t wal_partition, WriteCallback* callback,
                                  UserWriteCallback* user_write_cb,
                                  uint64_t* wal_used, uint64_t log_ref,
                                  bool disable_memtable, uint64_t* seq_used) {
  PERF_TIMER_GUARD(write_pre_and_post_process_time);
  StopWatch write_sw(immutable_db_options_.clock, stats_, DB_WRITE);

  WriteContext write_context;

  WriteThread::Writer w(write_options, my_batch, callback, user_write_cb,
                        log_ref, disable_memtable, /*_batch_cnt=*/0,
                        /*_pre_release_callback=*/nullptr,
                        /*_post_memtable_callback=*/nullptr,
                        /*_ingest_wbwi=*/false, trace_batch);
  w.wal_partition = wal_partition;
  WriteThread& write_thread = PartitionWriteThread(wal_partition);
  {
    // Time blocked waiting to join a write batch group: the queueing that
    // write-partition routing could relieve.
    StopWatch join_sw(immutable_db_options_.clock, stats_,
                      JOIN_BATCH_GROUP_WAIT_MICROS);
    write_thread.JoinBatchGroup(&w);
  }
  TEST_SYNC_POINT("DBImplWrite::PipelinedWriteImpl:AfterJoinBatchGroup");
  if (w.state == WriteThread::STATE_GROUP_LEADER) {
    WriteThread::WriteGroup wal_write_group;
    if (w.callback && !w.callback->AllowWriteBatching()) {
      write_thread.WaitForMemTableWriters();
    }
    WalContext wal_context(!write_options.disableWAL && write_options.sync);
    // PreprocessWrite does its own perf timing.
    PERF_TIMER_STOP(write_pre_and_post_process_time);
    w.status = PreprocessWrite(write_options, w.wal_partition, &wal_context,
                               &write_context);
    PERF_TIMER_START(write_pre_and_post_process_time);

    // This can set non-OK status if callback fail.
    last_batch_group_size_[wal_partition].StoreRelaxed(
        write_thread.EnterAsBatchGroupLeader(&w, &wal_write_group));
    const bool wal_index_enabled =
        log::WALIndexEnabled(immutable_db_options_.partition_wal_usage);
    // At N==1 nothing consumes the tracker, so neither the registration
    // nor its paired completion below enters it.
    const bool track_durability = wal_index_enabled && NumWalPartitions() > 1;
    // With WAL index the sequence block is reserved jointly with the index,
    // which needs total_count -- not known until the loop below has run. So
    // that loop numbers writers from zero and they are shifted onto the
    // reserved block once it is known, leaving the loop itself unchanged.
    SequenceNumber current_sequence =
        wal_index_enabled
            ? 0
            : write_thread.UpdateLastSequence(versions_->LastSequence()) + 1;
    size_t total_count = 0;
    size_t total_byte_size = 0;
    uint64_t wal_index = 0;

    if (w.status.ok()) {
      SequenceNumber next_sequence = current_sequence;
      for (auto* writer : wal_write_group) {
        assert(writer);
        if (writer->CheckCallback(this)) {
          if (writer->ShouldWriteToMemtable()) {
            writer->sequence = next_sequence;
            size_t count = WriteBatchInternal::Count(writer->batch);
            total_byte_size = WriteBatchInternal::AppendedByteSize(
                total_byte_size, WriteBatchInternal::ByteSize(writer->batch));
            next_sequence += count;
            total_count += count;
          }
        }
      }
      if (wal_index_enabled) {
        SequenceNumber last_sequence_before = 0;
        w.status = AllocateSequenceAndWALIndex(
            total_count, !write_options.disableWAL, &last_sequence_before,
            &wal_index, w.wal_partition,
            /*register_durability=*/track_durability);
        if (w.status.ok()) {
          current_sequence = last_sequence_before + 1;
          for (auto* writer : wal_write_group) {
            if (!writer->CallbackFailed() && writer->ShouldWriteToMemtable()) {
              writer->sequence += current_sequence;
            }
          }
        } else {
          // The shift never ran, so every writer still holds an offset from
          // zero rather than a sequence. The WAL write and UpdateLastSequence
          // below are both skipped on this path, but leave a value that cannot
          // be mistaken for a real sequence if anything else reaches it.
          for (auto* writer : wal_write_group) {
            writer->sequence = kMaxSequenceNumber;
          }
        }
      }
      // TODO: this use of operator bool on `tracer_` can avoid unnecessary lock
      // grabs but does not seem thread-safe.
      if (tracer_) {
        InstrumentedMutexLock lock(&trace_mutex_);
        if (tracer_ != nullptr && tracer_->IsWriteOrderPreserved()) {
          for (auto* writer : wal_write_group) {
            if (writer->CallbackFailed()) {
              // When optimisitc txn conflict checking fails, we should
              // not record to trace.
              continue;
            }
            // TODO: maybe handle the tracing status?
            tracer_->Write(writer->trace_batch).PermitUncheckedError();
          }
        }
      }
      if (w.disable_wal) {
        has_unpersisted_data_.store(true, std::memory_order_relaxed);
      }
      // Only the indexed path needs the guard: the allocation above can have
      // set w.status, leaving current_sequence at 0 with no block to publish.
      // Without WAL index this is unreachable, so the kNone path keeps its
      // unconditional update.
      if (!wal_index_enabled || w.status.ok()) {
        write_thread.UpdateLastSequence(current_sequence + total_count - 1);
      }
    }

    auto stats = default_cf_internal_stats_;
    stats->AddDBStats(InternalStats::kIntStatsNumKeysWritten, total_count);
    RecordTick(stats_, NUMBER_KEYS_WRITTEN, total_count);
    stats->AddDBStats(InternalStats::kIntStatsBytesWritten, total_byte_size);
    RecordTick(stats_, BYTES_WRITTEN, total_byte_size);
    RecordInHistogram(stats_, BYTES_PER_WRITE, total_byte_size);
    // One sample per write group: the group-size distribution, and the number
    // of compatible writers the size cap turned away.
    RecordInHistogram(stats_, WRITERS_PER_WRITE_GROUP, wal_write_group.size);
    RecordInHistogram(stats_, WRITE_GROUP_CAP_EXCLUDED,
                      wal_write_group.num_cap_excluded);

    PERF_TIMER_STOP(write_pre_and_post_process_time);

    IOStatus io_s;
    io_s.PermitUncheckedError();  // Allow io_s to be uninitialized

    // Nothing between the allocation above and the append below can set
    // w.status, so an allocated index always reaches the append and its cover.
    assert(wal_index == 0 || w.status.ok());
    if (w.status.ok() && !write_options.disableWAL) {
      PERF_TIMER_GUARD(write_wal_time);
      stats->AddDBStats(InternalStats::kIntStatsWriteDoneBySelf, 1);
      RecordTick(stats_, WRITE_DONE_BY_SELF, 1);
      if (wal_write_group.size > 1) {
        stats->AddDBStats(InternalStats::kIntStatsWriteDoneByOther,
                          wal_write_group.size - 1);
        RecordTick(stats_, WRITE_DONE_BY_OTHER, wal_write_group.size - 1);
      }
      assert(wal_context.wal_file_number_size);
      // Paired with the registration above.
      io_s = WriteGroupToWAL(wal_write_group, wal_context, wal_used,
                             current_sequence, wal_index, total_count,
                             /*complete_durability=*/track_durability);
      if (!io_s.ok()) {
        // As in WriteImpl: declare the burned index before the error escapes.
        const IOStatus cover_io_s =
            CoverBurnedWALIndex(write_options, wal_context.writer, wal_index);
        cover_io_s.PermitUncheckedError();
      }
      w.status = io_s;
    }

    if (!io_s.ok()) {
      // Check WriteToWAL status
      WALIOStatusCheck(io_s);
    } else if (!w.CallbackFailed()) {
      WriteStatusCheck(w.status);
    }

    VersionEdit synced_wals;
    if (wal_context.need_wal_sync) {
      AllWalWriteMutexesLock l(this);
      if (w.status.ok()) {
        MarkLogsSynced(wal_context.wal_sync_up_to,
                       wal_context.need_wal_dir_sync, &synced_wals,
                       wal_context.wal_sync_claimed);
      } else {
        MarkLogsNotSynced(wal_context.wal_sync_up_to,
                          wal_context.wal_sync_claimed);
      }
    }
    if (w.status.ok() && synced_wals.IsWalAddition()) {
      InstrumentedMutexLock l(&mutex_);
      // TODO: plumb Env::IOActivity, Env::IOPriority
      const ReadOptions read_options;
      w.status = ApplyWALToManifest(read_options, write_options, &synced_wals);
    }
    write_thread.ExitAsBatchGroupLeader(wal_write_group, w.status);
    // Only now is the group linked as a memtable writer, which a switch waits
    // for.
    ExitMemTableSwitchGate();
  }

  // NOTE: the memtable_write_group is declared before the following
  // `if` statement because its lifetime needs to be longer
  // that the inner context  of the `if` as a reference to it
  // may be used further below within the outer _write_thread
  WriteThread::WriteGroup memtable_write_group;

  if (w.state == WriteThread::STATE_MEMTABLE_WRITER_LEADER) {
    PERF_TIMER_FOR_WAIT_GUARD(write_memtable_time);
    assert(w.ShouldWriteToMemtable());
    write_thread.EnterAsMemTableWriter(&w, &memtable_write_group);
    if (memtable_write_group.size > 1 &&
        immutable_db_options_.allow_concurrent_memtable_write) {
      write_thread.LaunchParallelMemTableWriters(&memtable_write_group);
    } else {
      memtable_write_group.status = WriteBatchInternal::InsertInto(
          memtable_write_group, w.sequence, column_family_memtables_.get(),
          &flush_scheduler_, &trim_history_scheduler_,
          write_options.ignore_missing_column_families, 0 /*log_number*/, this,
          seq_per_batch_, batch_per_txn_);
      if (memtable_write_group.status
              .ok()) {  // Don't publish a partial batch write
        versions_->SetLastSequence(memtable_write_group.last_sequence);
      } else {
        HandleMemTableInsertFailure(memtable_write_group.status, w.sequence);
      }
      write_thread.ExitAsMemTableWriter(&w, memtable_write_group);
    }
  } else {
    // NOTE: the memtable_write_group is never really used,
    // so we need to set its status to pass ASSERT_STATUS_CHECKED
    memtable_write_group.status.PermitUncheckedError();
  }
  if (w.state == WriteThread::STATE_PARALLEL_MEMTABLE_CALLER) {
    write_thread.SetMemWritersEachStride(&w);
  }
  if (w.state == WriteThread::STATE_PARALLEL_MEMTABLE_WRITER) {
    PERF_TIMER_STOP(write_pre_and_post_process_time);
    PERF_TIMER_FOR_WAIT_GUARD(write_memtable_time);

    assert(w.ShouldWriteToMemtable());
    ColumnFamilyMemTablesImpl column_family_memtables(
        versions_->GetColumnFamilySet());
    w.status = WriteBatchInternal::InsertInto(
        &w, w.sequence, &column_family_memtables, &flush_scheduler_,
        &trim_history_scheduler_, write_options.ignore_missing_column_families,
        0 /*log_number*/, this, true /*concurrent_memtable_writes*/,
        false /*seq_per_batch*/, 0 /*batch_cnt*/, true /*batch_per_txn*/,
        write_options.memtable_insert_hint_per_batch);

    PERF_TIMER_STOP(write_memtable_time);
    PERF_TIMER_START(write_pre_and_post_process_time);

    if (write_thread.CompleteParallelMemTableWriter(&w)) {
      if (w.status.ok()) {  // Don't publish a partial batch write
        versions_->SetLastSequence(w.write_group->last_sequence);
      } else {
        HandleMemTableInsertFailure(w.status, w.write_group->first_sequence);
      }
      write_thread.ExitAsMemTableWriter(&w, *w.write_group);
    }
  }
  if (seq_used != nullptr) {
    *seq_used = w.sequence;
  }

  assert(w.state == WriteThread::STATE_COMPLETED);
  return w.FinalStatus();
}

Status DBImpl::UnorderedWriteMemtable(const WriteOptions& write_options,
                                      WriteBatch* my_batch,
                                      WriteCallback* callback, uint64_t log_ref,
                                      SequenceNumber seq,
                                      const size_t sub_batch_cnt) {
  PERF_TIMER_GUARD(write_pre_and_post_process_time);
  StopWatch write_sw(immutable_db_options_.clock, stats_, DB_WRITE);

  WriteThread::Writer w(write_options, my_batch, callback,
                        /*user_write_cb=*/nullptr, log_ref,
                        false /*disable_memtable*/);

  if (w.CheckCallback(this) && w.ShouldWriteToMemtable()) {
    w.sequence = seq;
    size_t total_count = WriteBatchInternal::Count(my_batch);
    InternalStats* stats = default_cf_internal_stats_;
    stats->AddDBStats(InternalStats::kIntStatsNumKeysWritten, total_count);
    RecordTick(stats_, NUMBER_KEYS_WRITTEN, total_count);

    PERF_TIMER_STOP(write_pre_and_post_process_time);
    PERF_TIMER_FOR_WAIT_GUARD(write_memtable_time);

    ColumnFamilyMemTablesImpl column_family_memtables(
        versions_->GetColumnFamilySet());
    w.status = WriteBatchInternal::InsertInto(
        &w, w.sequence, &column_family_memtables, &flush_scheduler_,
        &trim_history_scheduler_, write_options.ignore_missing_column_families,
        0 /*log_number*/, this, true /*concurrent_memtable_writes*/,
        seq_per_batch_, sub_batch_cnt, true /*batch_per_txn*/,
        write_options.memtable_insert_hint_per_batch);
    if (write_options.disableWAL) {
      has_unpersisted_data_.store(true, std::memory_order_relaxed);
    }

    PERF_TIMER_START(write_pre_and_post_process_time);
  }

  size_t pending_cnt = pending_memtable_writes_.fetch_sub(1) - 1;
  if (pending_cnt == 0) {
    // switch_cv_ waits until pending_memtable_writes_ = 0. Locking its mutex
    // before notify ensures that cv is in waiting state when it is notified
    // thus not missing the update to pending_memtable_writes_ even though it is
    // not modified under the mutex.
    std::lock_guard<std::mutex> lck(switch_mutex_);
    switch_cv_.notify_all();
  }
  WriteStatusCheck(w.status);

  if (!w.FinalStatus().ok()) {
    return w.FinalStatus();
  }
  return Status::OK();
}

// The 2nd write queue. If enabled it will be used only for WAL-only writes.
// This is the only queue that updates LastPublishedSequence which is only
// applicable in a two-queue setting.
Status DBImpl::WriteImplWALOnly(
    WriteThread* write_thread, const WriteOptions& write_options,
    WriteBatch* my_batch, WriteBatch* trace_batch, WriteCallback* callback,
    UserWriteCallback* user_write_cb, uint64_t* wal_used,
    const uint64_t log_ref, uint64_t* seq_used, const size_t sub_batch_cnt,
    PreReleaseCallback* pre_release_callback, const AssignOrder assign_order,
    const PublishLastSeq publish_last_seq, const bool disable_memtable,
    size_t wal_partition) {
  PERF_TIMER_GUARD(write_pre_and_post_process_time);
  WriteThread::Writer w(write_options, my_batch, callback, user_write_cb,
                        log_ref, disable_memtable, sub_batch_cnt,
                        pre_release_callback,
                        /*_post_memtable_callback=*/nullptr,
                        /*_ingest_wbwi=*/false, trace_batch);
  w.wal_partition = wal_partition;
  StopWatch write_sw(immutable_db_options_.clock, stats_, DB_WRITE);

  {
    // Time blocked waiting to join a write batch group: the queueing that
    // write-partition routing could relieve.
    StopWatch join_sw(immutable_db_options_.clock, stats_,
                      JOIN_BATCH_GROUP_WAIT_MICROS);
    write_thread->JoinBatchGroup(&w);
  }
  assert(w.state != WriteThread::STATE_PARALLEL_MEMTABLE_WRITER);
  if (w.state == WriteThread::STATE_COMPLETED) {
    if (wal_used != nullptr) {
      *wal_used = w.wal_used;
    }
    if (seq_used != nullptr) {
      *seq_used = w.sequence;
    }
    return w.FinalStatus();
  }
  // else we are the leader of the write batch group
  assert(w.state == WriteThread::STATE_GROUP_LEADER);

  if (publish_last_seq == kDoPublishLastSeq) {
    // Currently we only use kDoPublishLastSeq in unordered_write
    assert(immutable_db_options_.unordered_write);

    // TODO(myabandeh): Make preliminary checks thread-safe so we could do them
    // without paying the cost of obtaining the mutex.
    WalContext wal_context;
    WriteContext write_context;
    Status status = PreprocessWrite(write_options, w.wal_partition,
                                    &wal_context, &write_context);
    WriteStatusCheckOnLocked(status);

    if (!status.ok()) {
      WriteThread::WriteGroup write_group;
      write_thread->EnterAsBatchGroupLeader(&w, &write_group);
      ExitMemTableSwitchGate();
      write_thread->ExitAsBatchGroupLeader(write_group, status);
      return status;
    }
  } else {
    PERF_TIMER_STOP(write_pre_and_post_process_time);
    PERF_TIMER_FOR_WAIT_GUARD(write_delay_time);
    InstrumentedMutexLock lock(&mutex_);
    Status status =
        DelayWrite(/*num_bytes=*/0ull, *write_thread, write_options);
    PERF_TIMER_STOP(write_delay_time);
    PERF_TIMER_START(write_pre_and_post_process_time);
    if (!status.ok()) {
      WriteThread::WriteGroup write_group;
      write_thread->EnterAsBatchGroupLeader(&w, &write_group);
      write_thread->ExitAsBatchGroupLeader(write_group, status);
      return status;
    }
  }

  WriteThread::WriteGroup write_group;
  uint64_t last_sequence;
  write_thread->EnterAsBatchGroupLeader(&w, &write_group);

  size_t pre_release_callback_cnt = 0;
  size_t total_byte_size = 0;
  for (auto* writer : write_group) {
    assert(writer);
    if (writer->CheckCallback(this)) {
      total_byte_size = WriteBatchInternal::AppendedByteSize(
          total_byte_size, WriteBatchInternal::ByteSize(writer->batch));
      if (writer->pre_release_callback) {
        pre_release_callback_cnt++;
      }
    }
  }

  // Note: no need to update last_batch_group_size_ here since the batch writes
  // to WAL only
  // TODO: this use of operator bool on `tracer_` can avoid unnecessary lock
  // grabs but does not seem thread-safe.
  if (tracer_) {
    InstrumentedMutexLock lock(&trace_mutex_);
    if (tracer_ != nullptr && tracer_->IsWriteOrderPreserved()) {
      for (auto* writer : write_group) {
        if (writer->CallbackFailed()) {
          continue;
        }
        // TODO: maybe handle the tracing status?
        tracer_->Write(writer->trace_batch).PermitUncheckedError();
      }
    }
  }

  const bool concurrent_update = true;
  // Update stats while we are an exclusive group leader, so we know
  // that nobody else can be writing to these particular stats.
  // We're optimistic, updating the stats before we successfully
  // commit.  That lets us release our leader status early.
  auto stats = default_cf_internal_stats_;
  stats->AddDBStats(InternalStats::kIntStatsBytesWritten, total_byte_size,
                    concurrent_update);
  RecordTick(stats_, BYTES_WRITTEN, total_byte_size);
  stats->AddDBStats(InternalStats::kIntStatsWriteDoneBySelf, 1,
                    concurrent_update);
  RecordTick(stats_, WRITE_DONE_BY_SELF);
  auto write_done_by_other = write_group.size - 1;
  if (write_done_by_other > 0) {
    stats->AddDBStats(InternalStats::kIntStatsWriteDoneByOther,
                      write_done_by_other, concurrent_update);
    RecordTick(stats_, WRITE_DONE_BY_OTHER, write_done_by_other);
  }
  RecordInHistogram(stats_, BYTES_PER_WRITE, total_byte_size);
  // One sample per write group: the group-size distribution, and the number
  // of compatible writers the size cap turned away.
  RecordInHistogram(stats_, WRITERS_PER_WRITE_GROUP, write_group.size);
  RecordInHistogram(stats_, WRITE_GROUP_CAP_EXCLUDED,
                    write_group.num_cap_excluded);

  PERF_TIMER_STOP(write_pre_and_post_process_time);

  PERF_TIMER_GUARD(write_wal_time);
  // LastAllocatedSequence is increased inside WriteToWAL under
  // the selected partition's WalWriteMutex to ensure ordered events in WAL
  size_t seq_inc = 0 /* total_count */;
  if (assign_order == kDoAssignOrder) {
    size_t total_batch_cnt = 0;
    for (auto* writer : write_group) {
      assert(writer->batch_cnt || !seq_per_batch_);
      if (!writer->CallbackFailed()) {
        total_batch_cnt += writer->batch_cnt;
      }
    }
    seq_inc = total_batch_cnt;
  }
  Status status;
  // Snapshot for the completion gate's conditional post-wait sync, read
  // before the leadership-time sync below (sync writes, partitioned only).
  SequenceNumber pre_sync_safe_point = 0;
  if (!write_options.disableWAL) {
    IOStatus io_s = ConcurrentWriteGroupToWAL(write_group, wal_used,
                                              &last_sequence, seq_inc);
    status = io_s;
    // last_sequence may not be set if there is an error
    // This error checking and return is moved up to avoid using uninitialized
    // last_sequence.
    if (!io_s.ok()) {
      WALIOStatusCheck(io_s);
      if (publish_last_seq == kDoPublishLastSeq) {
        ExitMemTableSwitchGate();
      }
      write_thread->ExitAsBatchGroupLeader(write_group, status);
      return status;
    }
  } else if (log::WALIndexEnabled(immutable_db_options_.partition_wal_usage)) {
    uint64_t wal_index = 0;
    IOStatus io_s = AllocateSequenceAndWALIndex(
        seq_inc, /*write_wal=*/false, &last_sequence, &wal_index, wal_partition,
        /*register_durability=*/true);
    // write_wal=false, so no index is consumed and there is nothing that could
    // burn -- no cover is needed on this path.
    assert(wal_index == 0);
    status = io_s;
    if (!io_s.ok()) {
      if (publish_last_seq == kDoPublishLastSeq) {
        ExitMemTableSwitchGate();
      }
      write_thread->ExitAsBatchGroupLeader(write_group, status);
      return status;
    }
  } else {
    // Otherwise we inc seq number to do solely the seq allocation
    last_sequence = versions_->FetchAddLastAllocatedSequence(seq_inc);
  }

  size_t memtable_write_cnt = 0;
  auto curr_seq = last_sequence + 1;
  for (auto* writer : write_group) {
    if (writer->CallbackFailed()) {
      continue;
    }
    writer->sequence = curr_seq;
    if (assign_order == kDoAssignOrder) {
      assert(writer->batch_cnt || !seq_per_batch_);
      curr_seq += writer->batch_cnt;
    }
    if (!writer->disable_memtable) {
      memtable_write_cnt++;
    }
    // else seq advances only by memtable writes
  }
  if (status.ok() && write_options.sync) {
    assert(!write_options.disableWAL);
    if (NumWalPartitions() > 1) {
      // Read after our append completed and before this sync starts, so the
      // gate can skip its own sync when this one already covers it.
      pre_sync_safe_point = wal_durability_tracker_->SafePoint();
    }
    // Requesting sync with two_write_queues_ is expected to be very rare. We
    // hance provide a simple implementation that is not necessarily efficient.
    if (manual_wal_flush_) {
      status = FlushWAL(true);
    } else {
      status = SyncWAL();
    }
  }
  PERF_TIMER_START(write_pre_and_post_process_time);

  if (!w.CallbackFailed()) {
    WriteStatusCheck(status);
  }
  if (status.ok()) {
    size_t index = 0;
    for (auto* writer : write_group) {
      if (!writer->CallbackFailed() && writer->pre_release_callback) {
        assert(writer->sequence != kMaxSequenceNumber);
        Status ws = writer->pre_release_callback->Callback(
            writer->sequence, disable_memtable, writer->wal_used, index++,
            pre_release_callback_cnt);
        if (!ws.ok()) {
          status = ws;
          break;
        }
      }
    }
  }
  if (publish_last_seq == kDoPublishLastSeq) {
    PublishLastSequence(wal_partition, last_sequence + 1,
                        last_sequence + seq_inc, !write_options.disableWAL);
    // Currently we only use kDoPublishLastSeq in unordered_write
    assert(immutable_db_options_.unordered_write);
  }
  if (immutable_db_options_.unordered_write && status.ok()) {
    pending_memtable_writes_ += memtable_write_cnt;
  }
  if (publish_last_seq == kDoPublishLastSeq) {
    // This group's memtable writes follow it out of the group, but they are
    // already counted in pending_memtable_writes_, which a switch waits on.
    ExitMemTableSwitchGate();
  }
  // Completion gate, as in WriteImpl. Reaching here without disableWAL means
  // the append succeeded (the error path above returned early), so the range
  // is registered and completed; the target is sequence + count - 1.
  if (NumWalPartitions() > 1 && !write_options.disableWAL) {
    TEST_SYNC_POINT_CALLBACK("DBImpl::InstallCompletionGate", &last_sequence);
    const SequenceNumber gate_low = last_sequence + 1;
    const SequenceNumber gate_target = last_sequence + seq_inc;
    const bool gate_sync = write_options.sync;
    write_group.completion_gate = [this, gate_low, gate_target, gate_sync,
                                   pre_sync_safe_point]() {
      return RunCompletionGate(gate_low, gate_target, gate_sync,
                               pre_sync_safe_point,
                               /*sync_micros=*/0);
    };
  }
  write_thread->ExitAsBatchGroupLeader(write_group, status);
  if (status.ok()) {
    status = w.FinalStatus();
  }
  if (seq_used != nullptr) {
    *seq_used = w.sequence;
  }
  return status;
}

void DBImpl::WriteStatusCheckOnLocked(const Status& status) {
  // Is setting bg_error_ enough here?  This will at least stop
  // compaction and fail any further writes.
  InstrumentedMutexLock l(&mutex_);
  assert(!status.IsIOFenced() || !error_handler_.GetBGError().ok());
  if (immutable_db_options_.paranoid_checks && !status.ok() &&
      !status.IsBusy() && !status.IsIncomplete()) {
    // Maybe change the return status to void?
    error_handler_.SetBGError(status, BackgroundErrorReason::kWriteCallback);
  }
}

void DBImpl::WriteStatusCheck(const Status& status) {
  // Is setting bg_error_ enough here?  This will at least stop
  // compaction and fail any further writes.
  assert(!status.IsIOFenced() || !error_handler_.GetBGError().ok());
  if (immutable_db_options_.paranoid_checks && !status.ok() &&
      !status.IsBusy() && !status.IsIncomplete()) {
    mutex_.Lock();
    // Maybe change the return status to void?
    error_handler_.SetBGError(status, BackgroundErrorReason::kWriteCallback);
    mutex_.Unlock();
  }
}

void DBImpl::WALIOStatusCheck(const IOStatus& io_status,
                              uint64_t failed_wal_number,
                              SequenceNumber failed_wal_sequence,
                              SequenceNumber failed_wal_sequence_start,
                              size_t pinned_wal_partition) {
  // Is setting bg_error_ enough here?  This will at least stop
  // compaction and fail any further writes.
  // Above one partition every failure stops the DB (D-1), Busy and Incomplete
  // included: continuing would leave the failed range pinned below every
  // later write. Open requires paranoid_checks there.
  if ((immutable_db_options_.paranoid_checks && !io_status.ok() &&
       ((!io_status.IsBusy() && !io_status.IsIncomplete()) ||
        NumWalPartitions() > 1)) ||
      io_status.IsIOFenced()) {
    mutex_.Lock();
    if (pinned_wal_partition != kNoPinnedWalRange) {
      // Under the same mutex_ hold as the error below, so recovery from an
      // earlier error cannot release this range before the error is set.
      assert(NumWalPartitions() > 1);
      failed_wal_ranges_.push_back(
          {pinned_wal_partition, failed_wal_sequence_start,
           std::max(failed_wal_sequence_start, failed_wal_sequence)});
    }
    // Maybe change the return status to void?
    if (io_status.IsIOError() && !io_status.IsIOFenced()) {
      DBRecoverContext context;
      context.failed_wal_sequence = failed_wal_sequence;
      context.failed_wal_sequence_start = failed_wal_sequence_start;
      const uint64_t resolved_failed_wal_number =
          failed_wal_number != 0 ? failed_wal_number : cur_wal_number_;
      if (io_status.GetScope() == IOStatus::IOErrorScope::kIOErrorScopeFile) {
        context.failed_wal_number = resolved_failed_wal_number;
        if (failed_wal_sequence_start != 0 &&
            failed_wal_sequence_start <= failed_wal_sequence) {
          // Successful live resumption deliberately omits this write from the
          // database even if the physical WAL record is complete. Keep a
          // DB-lifetime fence so TransactionLogIterator does not expose it.
          wal_manager_.RecordIndeterminateWriteSequence(
              resolved_failed_wal_number, failed_wal_sequence_start,
              failed_wal_sequence);
        }
        ExtendWALRecoveryCutoff(context.failed_wal_number);
      }
      error_handler_.SetBGError(io_status,
                                BackgroundErrorReason::kWriteCallback,
                                /*wal_related=*/true, context);
    } else {
      error_handler_.SetBGError(io_status,
                                BackgroundErrorReason::kWriteCallback,
                                /*wal_related=*/true);
    }
    if (pinned_wal_partition != kNoPinnedWalRange) {
      // The background error is published (no listener can suppress it
      // here, see ErrorHandler::HandleKnownErrors); fail every gate at or
      // above the failed range with it.
      Status bg_error = error_handler_.GetBGError();
      assert(!bg_error.ok());
      InstallWalGateFence(failed_wal_sequence_start,
                          bg_error.ok() ? Status(io_status) : bg_error,
                          /*released_by_recovery=*/true);
    }
    mutex_.Unlock();
  } else {
    // Reset-and-continue would reset partition 0 whichever partition failed;
    // above one partition every failure takes the branch above instead.
    assert(NumWalPartitions() == 1);
    // Force writable file to be continue writable. No caller holds a WAL
    // write mutex, so taking partition 0's cannot deadlock or violate the
    // lock order.
    WalWriteMutexLock wal_lock(&WalWriteMutex(0));
    CurrentWal().writer->file()->reset_seen_error();
  }
}

void DBImpl::ExtendWALRecoveryCutoff(uint64_t failed_wal_number) {
  assert(failed_wal_number != 0);
  uint64_t current = wal_recovery_cutoff_.load(std::memory_order_relaxed);
  while (current < failed_wal_number &&
         !wal_recovery_cutoff_.compare_exchange_weak(
             current, failed_wal_number, std::memory_order_release,
             std::memory_order_relaxed)) {
  }
}

void DBImpl::HandleMemTableInsertFailure(const Status& status,
                                         SequenceNumber first_sequence) {
  assert(!status.ok());
  // A non-OK status on memtable insert indicates that the state implied by the
  // WAL has diverged from the in-memory state.  This could be because of a
  // corrupt write_batch (very bad), or because the client specified an invalid
  // column family and didn't specify ignore_missing_column_families.
  mutex_.Lock();
  assert(!error_handler_.IsBGWorkStopped());
  error_handler_.SetBGError(status, BackgroundErrorReason::kMemTable);
  if (NumWalPartitions() > 1) {
    Status bg_error = error_handler_.GetBGError();
    InstallWalGateFence(first_sequence, bg_error.ok() ? status : bg_error,
                        /*released_by_recovery=*/false);
  }
  mutex_.Unlock();
}

Status DBImpl::PreprocessWrite(const WriteOptions& write_options,
                               size_t wal_partition, WalContext* wal_context,
                               WriteContext* write_context) {
  assert(write_context != nullptr && wal_context != nullptr);
  Status status;

  if (error_handler_.IsDBStopped()) {
    InstrumentedMutexLock l(&mutex_);
    status = error_handler_.GetBGError();
  }

  PERF_TIMER_GUARD(write_scheduling_flushes_compactions_time);

  if (UNLIKELY(status.ok() &&
               wals_total_size_.LoadRelaxed() > GetMaxTotalWalSize())) {
    assert(versions_);
    InstrumentedMutexLock l(&mutex_);
    const ColumnFamilySet* const column_families =
        versions_->GetColumnFamilySet();
    assert(column_families);
    size_t num_cfs = column_families->NumberOfColumnFamilies();
    assert(num_cfs >= 1);
    if (num_cfs > 1) {
      CloseMemTableSwitchGate();
      WaitForPendingWrites();
      status = SwitchWAL(write_context);
      OpenMemTableSwitchGate();
    }
  }

  if (UNLIKELY(status.ok() && write_buffer_manager_->ShouldFlush())) {
    if (write_buffer_manager_->TryAcquireLocalFlush()) {
      InstrumentedMutexLock l(&mutex_);
      CloseMemTableSwitchGate();
      WaitForPendingWrites();
      status = HandleWriteBufferManagerFlush(write_context);
      OpenMemTableSwitchGate();
    }
  }

  if (UNLIKELY(status.ok() && !trim_history_scheduler_.Empty())) {
    InstrumentedMutexLock l(&mutex_);
    status = TrimMemtableHistory(write_context);
  }

  if (UNLIKELY(status.ok() && !flush_scheduler_.Empty())) {
    InstrumentedMutexLock l(&mutex_);
    CloseMemTableSwitchGate();
    WaitForPendingWrites();
    status = ScheduleFlushes(write_context);
    OpenMemTableSwitchGate();
  }

  PERF_TIMER_STOP(write_scheduling_flushes_compactions_time);
  PERF_TIMER_GUARD(write_pre_and_post_process_time);

  if (UNLIKELY(status.ok() && (write_controller_.IsStopped() ||
                               write_controller_.NeedsDelay()))) {
    PERF_TIMER_STOP(write_pre_and_post_process_time);
    PERF_TIMER_FOR_WAIT_GUARD(write_delay_time);
    // We don't know size of curent batch so that we always use the size
    // for previous one. It might create a fairness issue that expiration
    // might happen for smaller writes but larger writes can go through.
    // Can optimize it if it is an issue.
    InstrumentedMutexLock l(&mutex_);
    status = DelayWrite(last_batch_group_size_[wal_partition].LoadRelaxed(),
                        PartitionWriteThread(wal_partition), write_options);
    PERF_TIMER_START(write_pre_and_post_process_time);
  }

  // If memory usage exceeded beyond a certain threshold,
  // write_buffer_manager_->ShouldStall() returns true to all threads writing to
  // all DBs and writers will be stalled.
  // It does soft checking because WriteBufferManager::buffer_limit_ has already
  // exceeded at this point so no new write (including current one) will go
  // through until memory usage is decreased.
  if (UNLIKELY(status.ok() && write_buffer_manager_->ShouldStall())) {
    default_cf_internal_stats_->AddDBStats(
        InternalStats::kIntStatsWriteBufferManagerLimitStopsCounts, 1,
        true /* concurrent */);
    if (write_options.no_slowdown) {
      status = Status::Incomplete("Write stall");
    } else {
      InstrumentedMutexLock l(&mutex_);
      status = WriteBufferManagerStallWrites(wal_partition, write_context);
    }
  }
  // Enter before binding a WAL, so another partition's rotation cannot
  // replace it between the bind and the append. Every caller exits once its
  // group has exited.
  EnterMemTableSwitchGate();
  if (status.ok() && wal_context->need_wal_sync) {
    // Wait until the parallel syncs are finished. Any sync process has to sync
    // the front log too so it is enough to check the status of front()
    // We do a while loop since WalSyncCV(p) is signalled when any sync of
    // that partition is finished
    // Note: there does not seem to be a reason to wait for parallel sync at
    // this early step but it is not important since parallel sync (SyncWAL) and
    // need_wal_sync are usually not used together.
    {
      // wal_dir_synced_ is written under the rotated partition's mutex or
      // all N, so it is read under all N. Only the sync path acts on
      // need_wal_dir_sync (every use is guarded by need_wal_sync), so
      // non-sync writes skip this section.
      AllWalWriteMutexesLock all_locks(this);
      wal_context->need_wal_dir_sync =
          wal_context->need_wal_dir_sync && !wal_dir_synced_;
      wal_context->wal_sync_up_to = cur_wal_number_;
    }
    // Claim every partition for this sync write, each under only its own
    // mutex: a Wait() must never hold a second partition mutex (see the LOCK
    // ORDER note). The selected partition is claimed last so that its claim
    // and the SelectWalForWrite() bind below happen under one hold of its
    // mutex: a rotation landing between them would bind an unmarked WAL the
    // completion does not finish. The completion finishes only partitions
    // recorded in wal_sync_claimed, against wal_sync_up_to.
    // Only one partition gets here: above one, a sync write syncs from its
    // completion gate instead (SyncWalsFromCompletionGate), which takes the
    // mark mutex. With one partition there is nothing to interleave, so this
    // path takes nothing here, exactly as before the mutex existed.
    assert(NumWalPartitions() == 1);
    wal_context->wal_sync_claimed.resize(NumWalPartitions());
    for (size_t p = 0; p < NumWalPartitions(); ++p) {
      if (p == wal_partition) {
        continue;
      }
      WalWriteMutexLock l(&WalWriteMutex(p));
      wal_context->wal_sync_claimed[p] = ClaimWalPartitionForLeadershipSync(
          p, &wal_context->wal_sync_unsynced_partitions);
    }
    {
      WalWriteMutexLock l(&WalWriteMutex(wal_partition));
      wal_context->wal_sync_claimed[wal_partition] =
          ClaimWalPartitionForLeadershipSync(
              wal_partition, &wal_context->wal_sync_unsynced_partitions);
      SelectWalForWrite(wal_partition, wal_context);
    }
  } else {
    wal_context->need_wal_sync = false;
    // Non-sync writes bind here holding exactly one partition's mutex, so
    // the resolve step no longer serializes across partitions.
    WalWriteMutexLock l(&WalWriteMutex(wal_partition));
    SelectWalForWrite(wal_partition, wal_context);
  }

  return status;
}

Status DBImpl::MergeBatch(const WriteThread::WriteGroup& write_group,
                          WriteBatch* tmp_batch, WriteBatch** merged_batch,
                          size_t* write_with_wal,
                          WriteBatch** to_be_cached_state) {
  assert(write_with_wal != nullptr);
  assert(tmp_batch != nullptr);
  assert(*to_be_cached_state == nullptr);
  *write_with_wal = 0;
  auto* leader = write_group.leader;
  assert(!leader->disable_wal);  // Same holds for all in the batch group
  if (write_group.size == 1 && !leader->CallbackFailed() &&
      leader->batch->GetWalTerminationPoint().is_cleared()) {
    // we simply write the first WriteBatch to WAL if the group only
    // contains one batch, that batch should be written to the WAL,
    // and the batch is not wanting to be truncated
    *merged_batch = leader->batch;
    if (WriteBatchInternal::IsLatestPersistentState(*merged_batch)) {
      *to_be_cached_state = *merged_batch;
    }
    *write_with_wal = 1;
  } else {
    // WAL needs all of the batches flattened into a single batch.
    // We could avoid copying here with an iov-like AddRecord
    // interface
    *merged_batch = tmp_batch;
    for (auto writer : write_group) {
      if (!writer->CallbackFailed()) {
        Status s = WriteBatchInternal::Append(*merged_batch, writer->batch,
                                              /*WAL_only*/ true);
        if (!s.ok()) {
          tmp_batch->Clear();
          return s;
        }
        if (WriteBatchInternal::IsLatestPersistentState(writer->batch)) {
          // We only need to cache the last of such write batch
          *to_be_cached_state = writer->batch;
        }
        (*write_with_wal)++;
      }
    }
  }
  // return merged_batch;
  return Status::OK();
}

IOStatus DBImpl::AllocateSequenceAndWALIndex(
    size_t sequence_count, bool write_wal, SequenceNumber* last_sequence_before,
    uint64_t* wal_index, size_t wal_partition, bool register_durability) {
  const bool wal_index_enabled =
      log::WALIndexEnabled(immutable_db_options_.partition_wal_usage);
  assert(wal_index_enabled || NumWalPartitions() > 1);
  assert(last_sequence_before != nullptr);
  assert(wal_index != nullptr);

  InstrumentedMutexLock lock(&sequence_wal_index_mutex_);
  if (wal_index_enabled && write_wal &&
      next_wal_index_ == std::numeric_limits<uint64_t>::max()) {
    return IOStatus::Corruption("WAL index exhausted");
  }

  // Not defensive: this catch-up is what makes the allocator agree with the
  // LastSequence() reads it replaces. WriteRecoverableState and external file
  // ingestion advance last_sequence_ directly, without going through the
  // allocator, so last_allocated_sequence_ can trail it. Skipping the catch-up
  // would hand out sequences that have already been published.
  const SequenceNumber last_sequence = versions_->LastSequence();
  if (versions_->LastAllocatedSequence() < last_sequence) {
    versions_->SetLastAllocatedSequence(last_sequence);
  }
  *last_sequence_before =
      versions_->FetchAddLastAllocatedSequence(sequence_count);
  *wal_index = wal_index_enabled && write_wal ? next_wal_index_++ : 0;
  if (register_durability && write_wal) {
    // A WAL-only marker with count 0 is stamped with last_allocated + 1
    // without consuming it; it shares its value with the next real write and
    // both entries pin the minimum until both are durable.
    const SequenceNumber low = *last_sequence_before + 1;
    const SequenceNumber high =
        sequence_count == 0 ? low : *last_sequence_before + sequence_count;
    assert(wal_partition < NumWalPartitions());
    wal_durability_tracker_->Register(wal_partition, low, high);
  }
  // Every N>1 range PublishLastSequence will publish: the ranges registered
  // above, plus disableWAL ranges, which have no record but must not become
  // visible above a range still being inserted. Same section as the
  // allocation, so no frontier scan can miss an allocated range.
  if (register_durability && NumWalPartitions() > 1 &&
      (write_wal || sequence_count > 0)) {
    const SequenceNumber low = *last_sequence_before + 1;
    const SequenceNumber high =
        sequence_count == 0 ? low : *last_sequence_before + sequence_count;
    applied_tracker_->Register(wal_partition, low, high);
  }
  return IOStatus::OK();
}

void DBImpl::PublishAppliedPrefix(size_t wal_partition,
                                  SequenceNumber first_sequence,
                                  SequenceNumber last_sequence,
                                  bool wal_record) {
  assert(NumWalPartitions() > 1);
  if (!wal_record && first_sequence > last_sequence) {
    // Consumed no sequence and wrote no record: never registered.
    return;
  }
  const SequenceNumber high =
      first_sequence > last_sequence ? first_sequence : last_sequence;
  InstrumentedMutexLock l(&sequence_wal_index_mutex_);
  applied_tracker_->Complete(wal_partition, first_sequence, high);
  PublishAppliedFrontierLocked();
}

void DBImpl::PublishAppliedFrontierLocked() {
  sequence_wal_index_mutex_.AssertHeld();
  // A record that consumes no sequence is tracked at the next unallocated
  // one, which the idle safe point may then report; never publish past what
  // was allocated.
  SequenceNumber frontier = std::min(applied_tracker_->SafePoint(),
                                     versions_->LastAllocatedSequence());
  // Nor into an open gate fence: no write there is ever acknowledged. A
  // failed append or insert leaves its range unapplied anyway, but a failed
  // gate sync fences a range that already applied (FenceFailedGateSync).
  if (wal_gate_fence_high_.load(std::memory_order_relaxed) ==
      kMaxSequenceNumber) {
    const SequenceNumber fence_low = OpenWalGateFenceLowLocked();
    assert(fence_low > 0);
    frontier = std::min(frontier, fence_low - 1);
  }
#ifndef NDEBUG
  // The applied prefix is WAL-durable: no range with a record at or below
  // it is still pending its append.
  for (size_t p = 0; p < NumWalPartitions(); ++p) {
    assert(wal_durability_tracker_->PendingLow(p) > frontier);
  }
#endif
  versions_->SetLastSequenceAtLeast(frontier);
  SignalWalDurabilityWaiters();
}

void DBImpl::AssertLastSequenceCoversAllocated() {
#ifndef NDEBUG
  mutex_.AssertHeld();
  if (NumWalPartitions() > 1 && error_handler_.GetBGError().ok()) {
    assert(versions_->LastSequence() >= versions_->LastAllocatedSequence());
  }
#endif
}

IOStatus DBImpl::CoverBurnedWALIndex(const WriteOptions& write_options,
                                     log::Writer* log_writer,
                                     uint64_t wal_index) {
  if (wal_index == 0) {
    // disableWAL writes and the sequence-only path never take an index.
    return IOStatus::OK();
  }
  assert(log::WALIndexEnabled(immutable_db_options_.partition_wal_usage));
  assert(log_writer != nullptr);
  if (wal_index <= log_writer->last_wal_index_recorded()) {
    // The record reached the file and only a later sync failed, so nothing
    // was burned. A void here would trail the reader's index order mark.
    return IOStatus::OK();
  }

  IOStatus io_s =
      log_writer->AddWALIndexVoidRecord(write_options, wal_index, wal_index);
  if (!io_s.ok()) {
    // No retry loop, no blocking, no swallowing. The index stays uncovered, so
    // nothing above it may be acknowledged.
    //
    // What enforces that is the *append* failure that brought us here, not
    // this one: every call site reaches the cover only with a non-OK append
    // status, and it is that status the caller propagates and turns into a
    // background error, putting the DB in read-only mode. This status is
    // deliberately dropped by all three callers. A future path that covers an
    // index after a *successful* append would inherit none of that protection
    // and must establish its own.
    //
    // With one WAL at a time there is a second, independent guard: the IO
    // error that usually fails the cover has already poisoned the
    // WritableFileWriter, so every later append fails through
    // MaybeHandleSeenFileWriterError. Log whether that one applied.
    ROCKS_LOG_ERROR(
        immutable_db_options_.info_log,
        "Failed to cover burned WAL index %" PRIu64
        "; it is allocated but absent from the WAL (writer already failed: "
        "%d): %s",
        wal_index, static_cast<int>(log_writer->file()->seen_error()),
        io_s.ToString().c_str());
  }
  return io_s;
}

// When two_write_queues_ is disabled, this function is called from the only
// write thread. Otherwise this must be called holding the selected
// partition's WalWriteMutex.
void DBImpl::SelectWalForWrite(size_t wal_partition, WalContext* wal_context) {
  assert(wal_context != nullptr);

  wal_context->wal_partition = wal_partition;
  wal_context->writer = CurrentWal(wal_partition).writer;
  wal_context->wal_file_number_size =
      std::addressof(CurrentAliveWal(wal_partition));

  assert(wal_context->writer->get_log_number() ==
         wal_context->wal_file_number_size->number);
}

IOStatus DBImpl::WriteToWAL(const WriteBatch& merged_batch,
                            const WriteOptions& write_options,
                            WalContext& wal_context, uint64_t* wal_used,
                            uint64_t* log_size, SequenceNumber sequence,
                            uint64_t wal_index, size_t sequence_count,
                            bool wal_mutex_held) {
  assert(log_size != nullptr);
  assert(wal_context.writer != nullptr);
  assert(wal_context.wal_file_number_size != nullptr);

  log::Writer* log_writer = wal_context.writer;
  WalFileNumberSize& wal_file_number_size = *wal_context.wal_file_number_size;

  Slice log_entry = WriteBatchInternal::Contents(&merged_batch);
  TEST_SYNC_POINT_CALLBACK("DBImpl::WriteToWAL:log_entry", &log_entry);
  auto s = merged_batch.VerifyChecksum();
  if (!s.ok()) {
    return status_to_io_status(std::move(s));
  }
  *log_size = log_entry.size();
  // Concurrent callers hold the selected partition's WalWriteMutex across
  // this call. Otherwise the mutex is taken here: it guards the file
  // records below against a concurrent FlushWAL by the application
  // (manual_wal_flush_) and, at N > 1, against a concurrent leadership
  // sync flushing this file's buffers, and it guards the size bookkeeping
  // after the append against the WAL-mutex readers of entry sizes. The take
  // is deliberately after the log_entry hook so a test holding the append
  // open there pins no mutex. At N==1 the take applies only under manual
  // flush without two write queues, exactly as before the partitioned path
  // existed: a default N=1 write takes no WAL mutex here, and its
  // bookkeeping below is serialized instead by write-thread leadership, or
  // for Open's post-recovery marker by single-threaded Open under mutex_
  // (see alive_wal_files_).
  const bool take_lock =
      !wal_mutex_held && (NumWalPartitions() > 1 ||
                          UNLIKELY(manual_wal_flush_ && !two_write_queues_));
  if (take_lock) {
    TEST_SYNC_POINT("DBImpl::WriteToWAL:TakesWalMutex");
    StopWatch lock_sw(immutable_db_options_.clock, WalPartitionStats(),
                      WAL_WRITE_MUTEX_WAIT_MICROS);
    WalWriteMutex(wal_context.wal_partition).Lock();
  } else if (wal_mutex_held) {
    WalWriteMutex(wal_context.wal_partition).AssertHeld();
  }
  IOStatus io_s = log_writer->MaybeAddUserDefinedTimestampSizeRecord(
      write_options, versions_->GetColumnFamiliesTimestampSizeForRecord());
  TEST_SYNC_POINT_CALLBACK(
      "DBImpl::WriteToWAL:AfterMaybeAddUserDefinedTimestampSizeRecord", &io_s);
  if (!io_s.ok()) {
    if (take_lock) {
      WalWriteMutex(wal_context.wal_partition).Unlock();
    }
    return io_s;
  }
  // Sequence order equals byte order within a file: no record may start at
  // or below a sequence an earlier record in the same file consumed.
  assert(sequence > log_writer->last_sequence_consumed());
  io_s = log_writer->AddRecord(write_options, log_entry, sequence, wal_index);
  TEST_SYNC_POINT_CALLBACK("DBImpl::WriteToWAL:AfterAddRecord", &io_s);
#ifndef NDEBUG
  WalAppendOutcome append_outcome{wal_context.wal_partition, sequence,
                                  sequence_count, io_s.ok(),
                                  log_writer->get_log_number()};
  TEST_SYNC_POINT_CALLBACK("DBImpl::WriteToWAL:AppendOutcome", &append_outcome);
#endif
#ifndef NDEBUG
  if (io_s.ok() && sequence_count > 0) {
    log_writer->set_last_sequence_consumed(sequence + sequence_count - 1);
  }
#else
  (void)sequence_count;
#endif

  // The WAL mutex, when held, stays held across the bookkeeping: the size
  // accumulates under it, matching its WAL-mutex readers. A default N==1
  // write holds none and relies on leadership, or for Open's post-recovery
  // marker on single-threaded Open (see alive_wal_files_). After
  // the unlock only the entry's immutable number may be touched through
  // wal_file_number_size (see CurrentAliveWal's escape rule).
  if (wal_used != nullptr) {
    *wal_used = wal_file_number_size.number;
  }
  wals_total_size_.FetchAddRelaxed(log_entry.size());
  wal_file_number_size.AddSize(*log_size);
  WalEmpty(wal_context.wal_partition) = false;

  if (take_lock) {
    WalWriteMutex(wal_context.wal_partition).Unlock();
  }

  return io_s;
}

Status DBImpl::WaitForWalDurability(SequenceNumber target, bool wal_record) {
  StopWatch wait_sw(immutable_db_options_.clock, WalPartitionStats(),
                    WAL_DURABILITY_WAIT_MICROS);
  // Publication implies durability for a range with a WAL record; checking
  // both keeps a LastSequence() jump (ingestion) from skipping a pending one.
  auto covered = [this, target, wal_record]() {
    return versions_->LastSequence() >= target &&
           (!wal_record || wal_durability_tracker_->SafePoint() >= target);
  };
  // Fast path: atomic loads, no blocking. On the partitioned path this is
  // the whole gate whenever predecessors already landed. Covered is not
  // enough: recovery covers a range that failed, and everything fenced above
  // it, without making it durable.
  if (covered()) {
    return WalGateFenceStatus(target);
  }

  // Below the fast path: spin, then block, mirroring
  // WriteThread::AwaitState (pause loop, adaptive yield loop, blocking wait),
  // with the durability frontier as the predicate instead of a writer state.
  PERF_TIMER_FOR_WAIT_GUARD(write_thread_wait_nanos);

  for (uint32_t tries = 0; tries < 200; ++tries) {
    if (covered()) {
      return WalGateFenceStatus(target);
    }
    port::AsmVolatilePause();
  }

  static WriteThread::AdaptationContext wwd_ctx("WaitForWalDurability");
  const size_t kMaxSlowYieldsWhileSpinning = 3;
  auto& yield_credit = wwd_ctx.value;
  bool update_ctx = false;
  bool would_spin_again = false;
  const int sampling_base = 256;
  const uint64_t max_yield_usec =
      immutable_db_options_.enable_write_thread_adaptive_yield
          ? immutable_db_options_.write_thread_max_yield_usec
          : 0;

  if (max_yield_usec > 0) {
    update_ctx = Random::GetTLSInstance()->OneIn(sampling_base);

    if (update_ctx || yield_credit.load(std::memory_order_relaxed) >= 0) {
      auto spin_begin = std::chrono::steady_clock::now();
      size_t slow_yield_count = 0;

      auto iter_begin = spin_begin;
      while ((iter_begin - spin_begin) <=
             std::chrono::microseconds(max_yield_usec)) {
        std::this_thread::yield();

        if (covered()) {
          would_spin_again = true;
          break;
        }

        auto now = std::chrono::steady_clock::now();
        if (now == iter_begin ||
            now - iter_begin >=
                std::chrono::microseconds(
                    immutable_db_options_.write_thread_slow_yield_usec)) {
          ++slow_yield_count;
          if (slow_yield_count >= kMaxSlowYieldsWhileSpinning) {
            update_ctx = true;
            break;
          }
        }
        iter_begin = now;
      }
    }
  }

  Status s;
  if (covered()) {
    s = WalGateFenceStatus(target);
  } else {
    // Park until the frontier covers the target, a fence holds it, or
    // shutdown begins; every exit is checked under the mutex that each of
    // their signalers holds, so none can slip between the check and the park.
    // Any other background error only wakes the waiter to recheck: a group
    // whose append succeeded is acknowledged once covered, as with one WAL.
    // mutex_ is never held together with it (leaf rule). The wait has no
    // timeout: a missed wake would hang, loudly, instead of hiding behind a
    // poll.
    InstrumentedMutexLock l(&sequence_wal_index_mutex_);
    wal_durability_waiters_.fetch_add(1, std::memory_order_relaxed);
    while (true) {
      s = WalGateFenceStatusLocked(target);
      if (!s.ok() || covered()) {
        break;
      }
      if (shutting_down_.load(std::memory_order_acquire)) {
        s = Status::ShutdownInProgress("parked write");
        break;
      }
      TEST_SYNC_POINT("DBImpl::WaitForWalDurability:BeforePark");
      wal_durability_cv_.Wait();
    }
    if (wal_durability_waiters_.fetch_sub(1, std::memory_order_relaxed) == 1 &&
        shutting_down_.load(std::memory_order_acquire)) {
      // CloseHelper waits on this cv for the last waiter to leave.
      wal_durability_cv_.SignalAll();
    }
  }

  if (update_ctx) {
    auto v = yield_credit.load(std::memory_order_relaxed);
    v = v - (v / 1024) + (would_spin_again ? 1 : -1) * 131072;
    yield_credit.store(v, std::memory_order_relaxed);
  }
  return s;
}

void DBImpl::SignalWalDurabilityWaitersOnBGError() {
  // Called with mutex_ held from ErrorHandler on each publication that
  // stores an error or repeats the recorded severity, whatever its source
  // (WAL, flush, compaction, MANIFEST). Woken gates recheck their exits and
  // park again unless a fence now holds them. Safe with no waiters; takes no
  // WAL or sync-mark mutex (mutex_ sits above sequence_wal_index_mutex_; see
  // the LOCK ORDER note).
  mutex_.AssertHeld();
  wal_durability_bg_error_.store(true, std::memory_order_release);
  InstrumentedMutexLock l(&sequence_wal_index_mutex_);
  wal_durability_cv_.SignalAll();
  TEST_SYNC_POINT("DBImpl::SignalWalDurabilityWaitersOnBGError");
}

void DBImpl::ClearWalDurabilityBGError() {
  mutex_.AssertHeld();
  wal_durability_bg_error_.store(false, std::memory_order_release);
  bool closed_fence = false;
  {
    // Before the release below covers them: every sequence allocated so far
    // above a failed range belongs to a write that already failed, or will
    // fail at its gate. Its memtable entries went with recovery's flush
    // (UnreleasedWalGateFenceLow), and the WAL iterator stops before its
    // record too. Those records may lie in any partition's WAL, so the gap
    // names the lowest WAL number and stops every iterator.
    InstrumentedMutexLock l(&sequence_wal_index_mutex_);
    const SequenceNumber last_allocated = versions_->LastAllocatedSequence();
    for (WalGateFence& fence : wal_gate_fences_) {
      if (fence.released_by_recovery && fence.high == kMaxSequenceNumber) {
        fence.high = std::max(fence.low, last_allocated);
        wal_manager_.RecordIndeterminateWriteSequence(/*wal_number=*/1,
                                                      fence.low, fence.high);
        closed_fence = true;
      }
    }
    SequenceNumber high = 0;
    for (const WalGateFence& fence : wal_gate_fences_) {
      high = std::max(high, fence.high);
    }
    wal_gate_fence_high_.store(high, std::memory_order_release);
  }
  // Durability first: the applied frontier this publishes must stay below
  // every range still pending its append. The trackers feed only the gates
  // and publication; recovery and the indeterminate-write fence still see the
  // released range as a hole.
  for (const FailedWalRange& range : failed_wal_ranges_) {
    {
      InstrumentedMutexLock l(&sequence_wal_index_mutex_);
      wal_durability_tracker_->Release(range.partition, range.low, range.high);
    }
    PublishAppliedPrefix(range.partition, range.low, range.high,
                         /*wal_record=*/true);
  }
  failed_wal_ranges_.clear();
  if (closed_fence) {
    // A failed gate sync fences ranges that already applied, so closing its
    // fence above may have freed the frontier without any range to release.
    InstrumentedMutexLock l(&sequence_wal_index_mutex_);
    PublishAppliedFrontierLocked();
  }
}

void DBImpl::InstallWalGateFence(SequenceNumber low, const Status& status,
                                 bool released_by_recovery) {
  mutex_.AssertHeld();
  assert(NumWalPartitions() > 1);
  assert(!status.ok());
  InstrumentedMutexLock l(&sequence_wal_index_mutex_);
  wal_gate_fences_.push_back(
      {low, kMaxSequenceNumber, status, released_by_recovery});
  wal_gate_fence_high_.store(kMaxSequenceNumber, std::memory_order_release);
  if (low < wal_gate_fence_low_.load(std::memory_order_relaxed)) {
    wal_gate_fence_low_.store(low, std::memory_order_release);
  }
  wal_durability_cv_.SignalAll();
}

Status DBImpl::WalGateFenceStatusLocked(SequenceNumber target) const {
  sequence_wal_index_mutex_.AssertHeld();
  for (const WalGateFence& fence : wal_gate_fences_) {
    if (fence.low <= target && target <= fence.high) {
      return fence.status;
    }
  }
  return Status::OK();
}

SequenceNumber DBImpl::UnreleasedWalGateFenceLow() {
  mutex_.AssertHeld();
  if (NumWalPartitions() == 1) {
    return kMaxSequenceNumber;
  }
  InstrumentedMutexLock l(&sequence_wal_index_mutex_);
  return OpenWalGateFenceLowLocked();
}

SequenceNumber DBImpl::OpenWalGateFenceLowLocked() const {
  sequence_wal_index_mutex_.AssertHeld();
  SequenceNumber low = kMaxSequenceNumber;
  for (const WalGateFence& fence : wal_gate_fences_) {
    if (fence.high == kMaxSequenceNumber) {
      low = std::min(low, fence.low);
    }
  }
  return low;
}

Status DBImpl::RunCompletionGate(SequenceNumber low, SequenceNumber target,
                                 bool sync, SequenceNumber pre_sync_safe_point,
                                 uint64_t sync_micros) {
  TEST_SYNC_POINT("DBImpl::RunCompletionGate:BeforeWait");
  Status s = WaitForWalDurability(target, /*wal_record=*/true);
  if (!s.ok()) {
    return s;
  }
  if (sync && pre_sync_safe_point < target) {
    // The second threshold. The first fsync covered only what was written
    // before it (everything pre_sync_safe_point covers, by the snapshot's
    // position after our append and before the fsync). Predecessors
    // that landed after it are covered by the wait above but not by that
    // fsync, so sync again now that the whole prefix is written: this fsync
    // covers every record at or below target, restoring what a single WAL
    // gives for free (a sync write flushing all preceding async writes).
    // Waiting on a tracked "synced" frontier instead would hang: async
    // predecessors never publish one, so nothing would advance it past them.
    // Manual flush buffers in user space, so flush there like the write path.
    const uint64_t sync_start = WalPartitionTimerStart();
    SequenceNumber covered = 0;
    s = RunCoalescedGateSync(low, target, &covered, [this]() {
      return manual_wal_flush_ ? FlushWAL(true) : SyncWAL();
    });
    sync_micros += WalPartitionTimerElapsed(sync_start);
    // No WALIOStatusCheck here: SyncWAL/FlushWAL already route an I/O
    // failure to it (which wakes other parked gates); the coalescer fences
    // the group's range, and this path propagates the outcome to the group.
  }
  if (sync && WalPartitionTimersEnabled()) {
    RecordTimeToHistogram(stats_, WAL_SYNC_BARRIER_MICROS, sync_micros);
  }
  return s;
}

IOStatus DBImpl::WriteGroupToWAL(const WriteThread::WriteGroup& write_group,
                                 WalContext& wal_context, uint64_t* wal_used,
                                 SequenceNumber sequence, uint64_t wal_index,
                                 size_t sequence_count,
                                 bool complete_durability) {
  IOStatus io_s;
  assert(!two_write_queues_);
  assert(!write_group.leader->disable_wal);
  assert(wal_context.wal_file_number_size != nullptr);

  WalFileNumberSize& wal_file_number_size = *wal_context.wal_file_number_size;
  const bool need_wal_sync = wal_context.need_wal_sync;
  const bool need_wal_dir_sync = wal_context.need_wal_dir_sync;
  // Same holds for all in the batch group
  size_t write_with_wal = 0;
  WriteBatch* to_be_cached_state = nullptr;
  WriteBatch* merged_batch;
  WriteBatch& merge_buffer = WalMergeBuffer(wal_context.wal_partition);
  io_s =
      status_to_io_status(MergeBatch(write_group, &merge_buffer, &merged_batch,
                                     &write_with_wal, &to_be_cached_state));
  if (UNLIKELY(!io_s.ok())) {
    return io_s;
  }

  if (merged_batch == write_group.leader->batch) {
    write_group.leader->wal_used = wal_file_number_size.number;
  } else if (write_with_wal > 1) {
    for (auto writer : write_group) {
      writer->wal_used = wal_file_number_size.number;
    }
  }

  WriteBatchInternal::SetSequence(merged_batch, sequence);

  uint64_t log_size;

  // TODO: plumb Env::IOActivity, Env::IOPriority
  WriteOptions write_options;
  write_options.rate_limiter_priority =
      write_group.leader->rate_limiter_priority;
  const uint64_t append_start = WalPartitionTimerStart();
  io_s = WriteToWAL(*merged_batch, write_options, wal_context, wal_used,
                    &log_size, sequence, wal_index, sequence_count,
                    /*wal_mutex_held=*/false);
  if (io_s.ok() && complete_durability) {
    // write() completed (not fsync): the range is durable at sync=false.
    // A failed append stays pinned per the burned-index rule.
    const SequenceNumber high =
        sequence_count == 0 ? sequence : sequence + sequence_count - 1;
    InstrumentedMutexLock seq_lock(&sequence_wal_index_mutex_);
    wal_durability_tracker_->Complete(wal_context.wal_partition, sequence,
                                      high);
    if (append_start != 0) {
      RecordTimeToHistogram(stats_, WAL_PARTITION_COMPLETION_MICROS,
                            WalPartitionTimerElapsed(append_start));
    }
    SignalWalDurabilityWaiters();
  }
  if (to_be_cached_state) {
    TEST_SYNC_POINT("DBImpl::WriteGroupToWAL:BeforeCacheRecoverableState");
    // Leaders of different partitions get here concurrently and in no
    // particular sequence order, so keep the highest-sequence state.
    InstrumentedMutexLock state_lock(&cached_recoverable_state_mutex_);
    if (cached_recoverable_state_empty_ ||
        sequence > cached_recoverable_state_seq_) {
      cached_recoverable_state_ = *to_be_cached_state;
      cached_recoverable_state_seq_ = sequence;
      cached_recoverable_state_empty_ = false;
    }
  }

  if (io_s.ok() && need_wal_sync) {
    // Only one partition syncs here; above one, the completion gate does
    // (SyncWalsFromCompletionGate).
    assert(NumWalPartitions() == 1);
    StopWatch sw(immutable_db_options_.clock, stats_, WAL_FILE_SYNC_MICROS);
    // It's safe to sync partition p's WALs with unlocked mutex_ here because:
    //  - we've set getting_synced=true for all logs,
    //    so other threads won't retire them while we're here,
    //  - only writer thread can install a current WAL, and we're in
    //    writer thread, so no one will install one,
    //  - as long as other threads don't modify the deque, it's safe to read
    //    from std::deque from multiple threads concurrently.
    uint64_t fsynced_partitions = 0;
    io_s = SyncEveryWalPartition(write_options,
                                 &wal_context.sync_failed_wal_number,
                                 &fsynced_partitions);
    // Fires after the leadership-time fsync above, before the directory
    // sync: tests override io_s here to fail exactly this sync, which then
    // follows the normal WAL error path (WALIOStatusCheck).
    TEST_SYNC_POINT_CALLBACK("DBImpl::WriteGroupToWAL:AfterLeadershipSync",
                             &io_s);

    if (io_s.ok() && need_wal_dir_sync) {
      // We only sync WAL directory the first time WAL syncing is
      // requested, so that in case users never turn on WAL sync,
      // we can avoid the disk I/O in the write code path.
      io_s = directories_.GetWalDir()->FsyncWithDirOptions(
          IOOptions(), nullptr,
          DirFsyncOptions(DirFsyncOptions::FsyncReason::kNewFileSynced));
    }
  }

  if (merged_batch == &merge_buffer) {
    // WriteToWAL's borrowed Slice is no longer live, so this partition can
    // safely reuse the backing storage for its next write group.
    merge_buffer.Clear();
  }
  if (io_s.ok()) {
    // Leaders of other partitions update the same stats concurrently.
    const bool concurrent = NumWalPartitions() > 1;
    auto stats = default_cf_internal_stats_;
    if (need_wal_sync) {
      stats->AddDBStats(InternalStats::kIntStatsWalFileSynced, 1, concurrent);
      RecordTick(stats_, WAL_FILE_SYNCED);
    }
    stats->AddDBStats(InternalStats::kIntStatsWalFileBytes, log_size,
                      concurrent);
    RecordTick(stats_, WAL_FILE_BYTES, log_size);
    stats->AddDBStats(InternalStats::kIntStatsWriteWithWal, write_with_wal,
                      concurrent);
    RecordTick(stats_, WRITE_WITH_WAL, write_with_wal);
    for (auto* writer : write_group) {
      if (!writer->CallbackFailed()) {
        writer->CheckPostWalWriteCallback();
      }
    }
  }
  return io_s;
}

IOStatus DBImpl::SyncEveryWalPartition(const WriteOptions& write_options,
                                       uint64_t* failed_wal_number,
                                       uint64_t* fsynced_partitions) {
  // Sync operation should work with locked WAL write mutexes, because:
  //   when DBOptions.manual_wal_flush_ is set,
  //   FlushWAL function will be invoked by another thread.
  //   if without locked WAL write mutexes, the log file may get data
  //   corruption.
  // At N > 1 the mutexes are needed regardless of manual_wal_flush_: the
  // sync below flushes every live WAL's user-space buffers, which is not
  // safe against a concurrent AddRecord, and every other partition's
  // leader appends concurrently. Each partition's files are synced under
  // its own WAL mutex, taken one at a time in index order; threads
  // parked on WalSyncCV wait with the mutex released, so a sync holding
  // marks can always take them.
  IOStatus io_s;
  const bool needs_locking = manual_wal_flush_ && !two_write_queues_;
  if (UNLIKELY(needs_locking)) {
    LockWalWriteMutexes();
  }
  // At N==1 without manual flush the sync runs with no WAL mutex, as
  // before the partitioned path existed: the single leader cannot race
  // itself. (needs_locking, the pre-existing manual-flush take, is
  // unchanged.)
  const bool take_partition_locks = !needs_locking && NumWalPartitions() > 1;
  if (take_partition_locks) {
    TEST_SYNC_POINT("DBImpl::SyncEveryWalPartition:TakesPartitionSyncMutex");
  }
  for (size_t p = 0; p < NumWalPartitions(); ++p) {
    // Per-partition here: with manual_wal_flush_, needs_locking already
    // holds all N across the whole loop.
    if (take_partition_locks) {
      WalWriteMutex(p).Lock();
    }
    io_s = SyncWalPartitionFiles(p, write_options, failed_wal_number);
    if (take_partition_locks) {
      WalWriteMutex(p).Unlock();
    }
    if (!io_s.ok()) {
      break;
    }
    ++*fsynced_partitions;
  }
  if (UNLIKELY(needs_locking)) {
    UnlockWalWriteMutexes();
  }
  return io_s;
}

Status DBImpl::RunCoalescedGateSync(SequenceNumber low, SequenceNumber target,
                                    SequenceNumber* covered,
                                    const std::function<Status()>& sync) {
  assert(NumWalPartitions() > 1);
  {
    InstrumentedMutexLock l(&wal_gate_sync_mutex_);
    while (true) {
      if (wal_gate_synced_upto_ >= target) {
        // A sync that completed after `target` was written covers it.
        *covered = wal_gate_synced_upto_;
        return Status::OK();
      }
      if (!wal_gate_sync_running_) {
        break;
      }
      // Group commit: the sync that starts after this one finishes covers
      // every record written meanwhile, so the writes that queue here share
      // it instead of each fsyncing every partition in turn.
      TEST_SYNC_POINT("DBImpl::RunCoalescedGateSync:BeforeWait");
      wal_gate_sync_cv_.Wait();
    }
    wal_gate_sync_running_ = true;
  }
  // Read before the sync claims anything: every record at or below it is
  // written, so the sync covers it.
  const SequenceNumber covers = wal_durability_tracker_->SafePoint();
  const Status s = sync();
  if (!s.ok()) {
    // Before any waiter can start its own sync: one that succeeds must still
    // find this fence at its gate when it lies above this group.
    FenceFailedGateSync(low);
  }
  {
    InstrumentedMutexLock l(&wal_gate_sync_mutex_);
    wal_gate_sync_running_ = false;
    if (s.ok()) {
      wal_gate_synced_upto_ = std::max(wal_gate_synced_upto_, covers);
    }
    wal_gate_sync_cv_.SignalAll();
  }
  *covered = covers;
  return s;
}

void DBImpl::FenceFailedGateSync(SequenceNumber low) {
  assert(NumWalPartitions() > 1);
  InstrumentedMutexLock l(&mutex_);
  const Status bg_error = error_handler_.GetBGError();
  if (bg_error.ok()) {
    // Recovery already ran (its flush kept the group), or the failure never
    // became the background error; an open fence would then never close.
    return;
  }
  // As for a failed append: the group, and everything above it, never becomes
  // visible (PublishAppliedFrontierLocked), recovery's flush drops it, and
  // every gate in the fence returns the background error.
  InstallWalGateFence(low, bg_error, /*released_by_recovery=*/true);
}

Status DBImpl::SyncWalsFromCompletionGate(
    Env::IOPriority rate_limiter_priority) {
  assert(NumWalPartitions() > 1);
  bool need_wal_dir_sync = false;
  uint64_t up_to = 0;
  {
    // wal_dir_synced_ is written under the rotated partition's mutex or
    // all N, so it is read under all N; cur_wal_number_ likewise.
    AllWalWriteMutexesLock all_locks(this);
    need_wal_dir_sync = !wal_dir_synced_;
    up_to = cur_wal_number_;
  }
  // Claim every partition, each under only its own mutex: a Wait() must
  // never hold a second partition mutex (see the LOCK ORDER note). Marking
  // is serialized across all syncs (see wal_sync_mark_mutex_): the waits
  // below are only for markers that already released it, which finish
  // without it. The marks keep every claimed WAL from retiring until
  // MarkLogsSynced below; a WAL a rotation installs meanwhile is numbered
  // above up_to and is synced but not recorded.
  autovector<uint8_t, 1> claimed;
  uint32_t unsynced_partitions = 0;
  {
    TEST_SYNC_POINT("DBImpl::SyncWalsFromCompletionGate:TakesSyncMarkMutex");
    InstrumentedMutexLock mark_lock(&wal_sync_mark_mutex_);
    claimed.resize(NumWalPartitions());
    for (size_t p = 0; p < NumWalPartitions(); ++p) {
      WalWriteMutexLock l(&WalWriteMutex(p));
      claimed[p] = ClaimWalPartitionForLeadershipSync(p, &unsynced_partitions);
    }
  }

  WriteOptions write_options;
  write_options.rate_limiter_priority = rate_limiter_priority;
  uint64_t failed_wal_number = 0;
  uint64_t fsynced_partitions = 0;
  IOStatus io_s;
  {
    StopWatch sw(immutable_db_options_.clock, stats_, WAL_FILE_SYNC_MICROS);
    io_s = SyncEveryWalPartition(write_options, &failed_wal_number,
                                 &fsynced_partitions);
    // Tests override io_s here to fail exactly this sync, which then
    // follows the WAL error path below.
    TEST_SYNC_POINT_CALLBACK("DBImpl::SyncWalsFromCompletionGate:AfterGateSync",
                             &io_s);
    if (io_s.ok() && need_wal_dir_sync) {
      io_s = directories_.GetWalDir()->FsyncWithDirOptions(
          IOOptions(), nullptr,
          DirFsyncOptions(DirFsyncOptions::FsyncReason::kNewFileSynced));
    }
  }
  RecordInHistogram(stats_, WAL_SYNC_FSYNCED_PARTITIONS, fsynced_partitions);
  RecordInHistogram(stats_, WAL_SYNC_UNSYNCED_PARTITIONS, unsynced_partitions);

  VersionEdit synced_wals;
  {
    AllWalWriteMutexesLock all_locks(this);
    if (io_s.ok()) {
      MarkLogsSynced(up_to, need_wal_dir_sync, &synced_wals, claimed);
    } else {
      MarkLogsNotSynced(up_to, claimed);
    }
  }
  if (!io_s.ok()) {
    // The group is already inserted, so, as for a failed SyncWAL(), only the
    // WAL is named here: the failure stops the DB (D-1) and the write reports
    // it. Nothing is pinned, since the append completed; the coalescer then
    // fences the group's range (FenceFailedGateSync).
    WALIOStatusCheck(io_s, failed_wal_number);
    return io_s;
  }
  default_cf_internal_stats_->AddDBStats(InternalStats::kIntStatsWalFileSynced,
                                         1, /*concurrent=*/true);
  RecordTick(stats_, WAL_FILE_SYNCED);
  if (synced_wals.IsWalAddition()) {
    InstrumentedMutexLock l(&mutex_);
    const ReadOptions read_options;
    return ApplyWALToManifest(read_options, write_options, &synced_wals);
  }
  return Status::OK();
}

IOStatus DBImpl::ConcurrentWriteGroupToWAL(
    const WriteThread::WriteGroup& write_group, uint64_t* wal_used,
    SequenceNumber* last_sequence, size_t seq_inc) {
  IOStatus io_s;

  assert(two_write_queues_ || immutable_db_options_.unordered_write);
  assert(!write_group.leader->disable_wal);
  // Same holds for all in the batch group
  WriteBatch tmp_batch;
  size_t write_with_wal = 0;
  WriteBatch* to_be_cached_state = nullptr;
  WriteBatch* merged_batch;
  io_s = status_to_io_status(MergeBatch(write_group, &tmp_batch, &merged_batch,
                                        &write_with_wal, &to_be_cached_state));
  if (UNLIKELY(!io_s.ok())) {
    return io_s;
  }

  // We need to lock since the partition's WAL deques might change
  // concurrently. Exactly one partition's mutex: the whole group carries the
  // route its leader resolved before group formation.
  const size_t wal_partition = write_group.leader->wal_partition;
  WalWriteMutexType& wal_mutex = WalWriteMutex(wal_partition);
  wal_mutex.Lock();

  WalContext wal_context;
  SelectWalForWrite(wal_partition, &wal_context);

  WalFileNumberSize& wal_file_number_size = *wal_context.wal_file_number_size;

  if (merged_batch == write_group.leader->batch) {
    write_group.leader->wal_used = wal_file_number_size.number;
  } else if (write_with_wal > 1) {
    for (auto writer : write_group) {
      writer->wal_used = wal_file_number_size.number;
    }
  }
  uint64_t wal_index = 0;
  // At N==1 nothing consumes the tracker, so neither path below enters
  // it; the mutex acquisitions go away with the registration.
  const bool track_durability = NumWalPartitions() > 1;
  if (log::WALIndexEnabled(immutable_db_options_.partition_wal_usage)) {
    io_s = AllocateSequenceAndWALIndex(
        seq_inc, /*write_wal=*/true, last_sequence, &wal_index, wal_partition,
        /*register_durability=*/track_durability);
    if (!io_s.ok()) {
      wal_mutex.Unlock();
      return io_s;
    }
  } else if (track_durability) {
    // No-index path: bracket the allocation with durability registration so
    // no later scan can miss the outstanding range. sequence_wal_index_mutex_
    // is a leaf below the WAL write mutex already held here.
    InstrumentedMutexLock seq_lock(&sequence_wal_index_mutex_);
    *last_sequence = versions_->FetchAddLastAllocatedSequence(seq_inc);
    const SequenceNumber low = *last_sequence + 1;
    const SequenceNumber high = seq_inc == 0 ? low : *last_sequence + seq_inc;
    wal_durability_tracker_->Register(wal_partition, low, high);
  } else {
    *last_sequence = versions_->FetchAddLastAllocatedSequence(seq_inc);
  }
  auto sequence = *last_sequence + 1;
  WriteBatchInternal::SetSequence(merged_batch, sequence);

  uint64_t log_size;

  // TODO: plumb Env::IOActivity, Env::IOPriority
  WriteOptions write_options;
  write_options.rate_limiter_priority =
      write_group.leader->rate_limiter_priority;
  io_s = WriteToWAL(*merged_batch, write_options, wal_context, wal_used,
                    &log_size, sequence, wal_index, seq_inc,
                    /*wal_mutex_held=*/true);
  if (!io_s.ok()) {
    // Not reachable today: this path needs two_write_queues or
    // unordered_write, and Open() rejects the first alongside WAL indexing.
    // Covered anyway rather than asserted, so enabling either combination
    // later cannot silently reintroduce an uncovered hole. Still under the
    // selected partition's WalWriteMutex, which is where WAL appends on this
    // path belong.
    const IOStatus cover_io_s =
        CoverBurnedWALIndex(write_options, wal_context.writer, wal_index);
    cover_io_s.PermitUncheckedError();
  } else if (track_durability) {
    // write() completed (not fsync). Still under the partition's WAL write
    // mutex; sequence_wal_index_mutex_ is a leaf below it. Paired with the
    // registration above: only complete what was registered, so at N==1
    // this completion (and its mutex acquisition) goes away with it.
    const SequenceNumber high =
        seq_inc == 0 ? sequence : sequence + seq_inc - 1;
    InstrumentedMutexLock seq_lock(&sequence_wal_index_mutex_);
    wal_durability_tracker_->Complete(wal_partition, sequence, high);
    SignalWalDurabilityWaiters();
  }
  if (to_be_cached_state) {
    cached_recoverable_state_ = *to_be_cached_state;
    cached_recoverable_state_empty_ = false;
  }
  wal_mutex.Unlock();

  if (io_s.ok()) {
    const bool concurrent = true;
    auto stats = default_cf_internal_stats_;
    stats->AddDBStats(InternalStats::kIntStatsWalFileBytes, log_size,
                      concurrent);
    RecordTick(stats_, WAL_FILE_BYTES, log_size);
    stats->AddDBStats(InternalStats::kIntStatsWriteWithWal, write_with_wal,
                      concurrent);
    RecordTick(stats_, WRITE_WITH_WAL, write_with_wal);
    for (auto* writer : write_group) {
      if (!writer->CallbackFailed()) {
        writer->CheckPostWalWriteCallback();
      }
    }
  }
  return io_s;
}

Status DBImpl::WriteRecoverableState() {
  mutex_.AssertHeld();
  if (!cached_recoverable_state_empty_) {
    // Only for write-prepared and write-unprepared.
    assert(seq_per_batch_);
    bool dont_care_bool;
    SequenceNumber next_seq;
    if (two_write_queues_) {
      // All N: concurrent WAL writers on any partition touch the cached
      // state, so draining it must exclude every partition.
      LockWalWriteMutexes();
    }
    SequenceNumber seq;
    if (two_write_queues_) {
      seq = versions_->FetchAddLastAllocatedSequence(0);
    } else {
      seq = versions_->LastSequence();
    }
    WriteBatchInternal::SetSequence(&cached_recoverable_state_, seq + 1);
    auto status = WriteBatchInternal::InsertInto(
        &cached_recoverable_state_, column_family_memtables_.get(),
        &flush_scheduler_, &trim_history_scheduler_, true,
        0 /*recovery_log_number*/, this, false /* concurrent_memtable_writes */,
        &next_seq, &dont_care_bool, seq_per_batch_);
    auto last_seq = next_seq - 1;
    if (status.ok()) {  // Don't publish a partial batch write
      if (two_write_queues_) {
        versions_->FetchAddLastAllocatedSequence(last_seq - seq);
        versions_->SetLastPublishedSequence(last_seq);
      } else if (NumWalPartitions() > 1 &&
                 versions_->LastAllocatedSequence() < last_seq) {
        // Every leader is excluded, so nothing allocates concurrently.
        versions_->SetLastAllocatedSequence(last_seq);
      }
      versions_->SetLastSequence(last_seq);
    } else {
      HandleMemTableInsertFailure(status, seq + 1);
    }
    if (two_write_queues_) {
      UnlockWalWriteMutexes();
    }
    if (status.ok() && recoverable_state_pre_release_callback_) {
      const bool DISABLE_MEMTABLE = true;
      for (uint64_t sub_batch_seq = seq + 1;
           sub_batch_seq < next_seq && status.ok(); sub_batch_seq++) {
        uint64_t const no_log_num = 0;
        // Unlock it since the callback might end up locking mutex. e.g.,
        // AddCommitted -> AdvanceMaxEvictedSeq -> GetSnapshotListFromDB
        mutex_.Unlock();
        status = recoverable_state_pre_release_callback_->Callback(
            sub_batch_seq, !DISABLE_MEMTABLE, no_log_num, 0, 1);
        mutex_.Lock();
      }
    }
    if (status.ok()) {
      cached_recoverable_state_.Clear();
      cached_recoverable_state_empty_ = true;
    } else {
      // FIXME: !ok status is untested
    }
    return status;
  }
  return Status::OK();
}

void DBImpl::SelectColumnFamiliesForAtomicFlush(
    autovector<ColumnFamilyData*>* selected_cfds,
    const autovector<ColumnFamilyData*>& provided_candidate_cfds,
    FlushReason flush_reason) {
  mutex_.AssertHeld();
  assert(selected_cfds);

  autovector<ColumnFamilyData*> candidate_cfds;

  // Generate candidate cfds if not provided
  if (provided_candidate_cfds.empty()) {
    for (ColumnFamilyData* cfd : *versions_->GetColumnFamilySet()) {
      if (!cfd->IsDropped() && cfd->initialized()) {
        cfd->Ref();
        candidate_cfds.push_back(cfd);
      }
    }
  } else {
    candidate_cfds = provided_candidate_cfds;
  }

  for (ColumnFamilyData* cfd : candidate_cfds) {
    if (cfd->IsDropped()) {
      continue;
    }
    if (cfd->imm()->NumNotFlushed() != 0 || !cfd->mem()->IsEmpty() ||
        !cached_recoverable_state_empty_.load() ||
        IsRecoveryFlush(flush_reason)) {
      selected_cfds->push_back(cfd);
    }
  }

  // Unref the newly generated candidate cfds (when not provided) in
  // `candidate_cfds`
  if (provided_candidate_cfds.empty()) {
    for (auto candidate_cfd : candidate_cfds) {
      candidate_cfd->UnrefAndTryDelete();
    }
  }
}

// Assign sequence number for atomic flush.
void DBImpl::AssignAtomicFlushSeq(const autovector<ColumnFamilyData*>& cfds) {
  auto seq = versions_->LastSequence();
  for (auto cfd : cfds) {
    // cfd can be nullptr, see ScheduleFlushes()
    if (cfd) {
      cfd->imm()->AssignAtomicFlushSeq(seq);
    }
  }
}

Status DBImpl::SwitchWAL(WriteContext* write_context) {
  mutex_.AssertHeld();
  assert(write_context != nullptr);
  Status status;

  // With more than one partition the DB's oldest live WAL can be on any of
  // them: an outstanding prepare can pin a later partition's WAL after
  // partition 0's older one retired. File numbers are unique across
  // partitions, so the minimum is unambiguous.
  size_t oldest_partition = 0;
  WalFileNumberSize oldest_alive_wal = OldestAliveWal(0);
  for (size_t p = 1; p < NumWalPartitions(); ++p) {
    const WalFileNumberSize candidate = OldestAliveWal(p);
    if (candidate.number < oldest_alive_wal.number) {
      oldest_partition = p;
      oldest_alive_wal = candidate;
    }
  }
  if (oldest_alive_wal.getting_flushed) {
    return status;
  }

  auto oldest_alive_log = oldest_alive_wal.number;
  bool flush_wont_release_oldest_log = false;
  if (allow_2pc()) {
    auto oldest_log_with_uncommitted_prep =
        logs_with_prep_tracker_.FindMinLogContainingOutstandingPrep();

    assert(oldest_log_with_uncommitted_prep == 0 ||
           oldest_log_with_uncommitted_prep >= oldest_alive_log);
    if (oldest_log_with_uncommitted_prep > 0 &&
        oldest_log_with_uncommitted_prep == oldest_alive_log) {
      if (unable_to_release_oldest_log_) {
        // we already attempted to flush all column families dependent on
        // the oldest alive log but the log still contained uncommitted
        // transactions so there is still nothing that we can do.
        return status;
      } else {
        ROCKS_LOG_WARN(
            immutable_db_options_.info_log,
            "Unable to release oldest log due to uncommitted transaction");
        unable_to_release_oldest_log_ = true;
        flush_wont_release_oldest_log = true;
      }
    }
  }
  if (!flush_wont_release_oldest_log) {
    // we only mark this log as getting flushed if we have successfully
    // flushed all data in this log. If this log contains outstanding prepared
    // transactions then we cannot flush this log until those transactions are
    // commited.
    unable_to_release_oldest_log_ = false;
    MarkOldestAliveWalGettingFlushed(oldest_partition);
  }

  ROCKS_LOG_INFO(
      immutable_db_options_.info_log,
      "Flushing all column families with data in WAL number %" PRIu64
      ". Total log size is %" PRIu64 " while max_total_wal_size is %" PRIu64,
      oldest_alive_log, wals_total_size_.LoadRelaxed(), GetMaxTotalWalSize());
  // no need to refcount because drop is happening in write thread, so can't
  // happen while we're in the write thread
  autovector<ColumnFamilyData*> cfds;
  if (immutable_db_options_.atomic_flush) {
    SelectColumnFamiliesForAtomicFlush(&cfds);
  } else {
    for (auto cfd : *versions_->GetColumnFamilySet()) {
      if (cfd->IsDropped()) {
        continue;
      }
      if (cfd->OldestLogToKeep() <= oldest_alive_log) {
        cfds.push_back(cfd);
      }
    }
    MaybeFlushStatsCF(&cfds);
  }
  AllPartitionsWriter nonmem_w(this, /*nonmem=*/true);
  if (two_write_queues_) {
    nonmem_w.EnterUnbatched();
  }

  const bool defer_wbm_accounting_refresh =
      immutable_db_options_.atomic_flush && cfds.size() > 1;
  bool switch_attempted = false;
  for (const auto cfd : cfds) {
    cfd->Ref();
    switch_attempted = true;
    status = defer_wbm_accounting_refresh
                 ? SwitchMemtableImpl(cfd, write_context, nullptr, 0,
                                      false /* refresh_wbm_accounting */)
                 : SwitchMemtable(cfd, write_context);
    cfd->UnrefAndTryDelete();
    if (!status.ok()) {
      break;
    }
  }
  if (defer_wbm_accounting_refresh && switch_attempted) {
    UpdateWBMAccountingAfterMemtableSwitch(true /* refresh_now */);
  }
  if (two_write_queues_) {
    nonmem_w.ExitUnbatched();
  }

  if (status.ok()) {
    if (immutable_db_options_.atomic_flush) {
      AssignAtomicFlushSeq(cfds);
    }
    for (auto cfd : cfds) {
      cfd->imm()->FlushRequested();
      if (!immutable_db_options_.atomic_flush) {
        FlushRequest flush_req;
        GenerateFlushRequest({cfd}, FlushReason::kWalFull, &flush_req);
        EnqueuePendingFlush(flush_req);
      }
    }
    if (immutable_db_options_.atomic_flush) {
      FlushRequest flush_req;
      flush_req.atomic_flush = true;
      GenerateFlushRequest(cfds, FlushReason::kWalFull, &flush_req);
      EnqueuePendingFlush(flush_req);
    }
    MaybeScheduleFlushOrCompaction();
  }
  return status;
}

Status DBImpl::HandleWriteBufferManagerFlush(WriteContext* write_context) {
  mutex_.AssertHeld();
  assert(write_context != nullptr);
  Status status;

  // Before a new memtable is added in SwitchMemtable(),
  // write_buffer_manager_->ShouldFlush() will keep returning true. If another
  // thread is writing to another DB with the same write buffer, they may also
  // be flushed. We may end up with flushing much more DBs than needed. It's
  // suboptimal but still correct.
  // no need to refcount because drop is happening in write thread, so can't
  // happen while we're in the write thread
  autovector<ColumnFamilyData*> cfds;
  if (immutable_db_options_.atomic_flush) {
    SelectColumnFamiliesForAtomicFlush(&cfds);
  } else {
    const WriteBufferFlushPolicy policy = write_buffer_manager_->flush_policy();
    // Cross-DB selection still flushes the largest CF within the chosen DB.
    const bool flush_largest =
        policy == WriteBufferFlushPolicy::kFlushLargest ||
        policy == WriteBufferFlushPolicy::kFlushLargestAcrossDBs;
    const bool include_waiting_immutable =
        policy == WriteBufferFlushPolicy::kFlushLargestAcrossDBs;
    const FlushableCFs flushable =
        CollectFlushableCFs(include_waiting_immutable);
    ColumnFamilyData* cfd_picked =
        flush_largest ? flushable.largest : flushable.oldest;
    if (cfd_picked != nullptr) {
      cfds.push_back(cfd_picked);
    }
    MaybeFlushStatsCF(&cfds);
  }
  if (!cfds.empty()) {
    ROCKS_LOG_INFO(
        immutable_db_options_.info_log,
        "Flushing triggered to alleviate write buffer memory usage. Write "
        "buffer is using %" ROCKSDB_PRIszt
        " bytes out of a total of %" ROCKSDB_PRIszt ".",
        write_buffer_manager_->memory_usage(),
        write_buffer_manager_->buffer_size());
  }

  AllPartitionsWriter nonmem_w(this, /*nonmem=*/true);
  if (two_write_queues_) {
    nonmem_w.EnterUnbatched();
  }
  const bool defer_wbm_accounting_refresh =
      immutable_db_options_.atomic_flush && cfds.size() > 1;
  for (const auto cfd : cfds) {
    if (cfd->mem()->IsEmpty()) {
      continue;
    }
    cfd->Ref();
    status = defer_wbm_accounting_refresh
                 ? SwitchMemtableImpl(cfd, write_context, nullptr, 0,
                                      false /* refresh_wbm_accounting */)
                 : SwitchMemtable(cfd, write_context);
    cfd->UnrefAndTryDelete();
    if (!status.ok()) {
      break;
    }
  }
  if (two_write_queues_) {
    nonmem_w.ExitUnbatched();
  }

  if (status.ok()) {
    if (immutable_db_options_.atomic_flush) {
      AssignAtomicFlushSeq(cfds);
    }
    for (const auto cfd : cfds) {
      cfd->imm()->FlushRequested();
      if (!immutable_db_options_.atomic_flush) {
        FlushRequest flush_req;
        GenerateFlushRequest({cfd}, FlushReason::kWriteBufferManager,
                             &flush_req);
        EnqueuePendingFlush(flush_req);
      }
    }
    if (immutable_db_options_.atomic_flush) {
      FlushRequest flush_req;
      flush_req.atomic_flush = true;
      GenerateFlushRequest(cfds, FlushReason::kWriteBufferManager, &flush_req);
      EnqueuePendingFlush(flush_req);
    }
  }
  if (write_buffer_manager_->ShouldTrackFlushInitiator()) {
    RefreshFlushableMemAccounting();
  }
  if (status.ok()) {
    MaybeScheduleFlushOrCompaction();
  }
  return status;
}

uint64_t DBImpl::GetMaxTotalWalSize() const {
  uint64_t max_total_wal_size =
      max_total_wal_size_.load(std::memory_order_acquire);
  if (max_total_wal_size > 0) {
    return max_total_wal_size;
  }
  return 4 * max_total_in_memory_state_.load(std::memory_order_acquire);
}

// REQUIRES: mutex_ is held
// REQUIRES: this thread is currently at the leader for write_thread
Status DBImpl::DelayWrite(uint64_t num_bytes, WriteThread& write_thread,
                          const WriteOptions& write_options) {
  mutex_.AssertHeld();
  uint64_t start_time = 0;
  bool delayed = false;
  {
    // Timed delays apply on every partition's main write queue, never on a
    // nonmem queue. Partition leaders may sleep in parallel, but each one
    // charges the single write controller, whose refill time accumulates
    // every charge, so together they still admit about delayed_write_rate.
    uint64_t delay;
    if (IsMainWriteThread(write_thread)) {
      delay =
          write_controller_.GetDelay(immutable_db_options_.clock, num_bytes);
    } else {
      assert(num_bytes == 0);
      delay = 0;
    }
    TEST_SYNC_POINT_CALLBACK("DBImpl::DelayWrite:Start", &num_bytes);
    start_time = immutable_db_options_.clock->NowMicros();

    if (delay > 0) {
      if (write_options.no_slowdown) {
        return Status::Incomplete("Write stall");
      }
      TEST_SYNC_POINT("DBImpl::DelayWrite:Sleep");

      // Notify write_thread about the stall so it can setup a barrier and
      // fail any pending writers with no_slowdown
      write_thread.BeginWriteStall();
      mutex_.Unlock();
      TEST_SYNC_POINT("DBImpl::DelayWrite:BeginWriteStallDone");
      // We will delay the write until we have slept for `delay` microseconds
      // or we don't need a delay anymore. We check for cancellation every 1ms
      // (slightly longer because WriteController minimum delay is 1ms, in
      // case of sleep imprecision, rounding, etc.)
      const uint64_t kDelayInterval = 1001;
      uint64_t stall_end = start_time + delay;
      while (write_controller_.NeedsDelay()) {
        if (immutable_db_options_.clock->NowMicros() >= stall_end) {
          // We already delayed this write `delay` microseconds
          break;
        }

        delayed = true;
        // Sleep for 0.001 seconds
        immutable_db_options_.clock->SleepForMicroseconds(kDelayInterval);
      }
      mutex_.Lock();
      write_thread.EndWriteStall();
    }

    // Don't wait if there's a background error that is not pending recovery
    // since recovery might never be attempted.
    while ((error_handler_.GetBGError().ok() ||
            error_handler_.IsRecoveryInProgress()) &&
           write_controller_.IsStopped() &&
           !shutting_down_.load(std::memory_order_relaxed)) {
      if (write_options.no_slowdown) {
        return Status::Incomplete("Write stall");
      }
      delayed = true;

      // Notify write_thread about the stall so it can setup a barrier and
      // fail any pending writers with no_slowdown
      write_thread.BeginWriteStall();
      if (IsMainWriteThread(write_thread)) {
        TEST_SYNC_POINT("DBImpl::DelayWrite:Wait");
      } else {
        TEST_SYNC_POINT("DBImpl::DelayWrite:NonmemWait");
      }
      bg_cv_.Wait();
      TEST_SYNC_POINT_CALLBACK("DBImpl::DelayWrite:AfterWait", &mutex_);
      write_thread.EndWriteStall();
    }
  }
  assert(!delayed || !write_options.no_slowdown);
  if (delayed) {
    auto time_delayed = immutable_db_options_.clock->NowMicros() - start_time;
    default_cf_internal_stats_->AddDBStats(
        InternalStats::kIntStatsWriteStallMicros, time_delayed);
    RecordTick(stats_, STALL_MICROS, time_delayed);
    RecordInHistogram(stats_, WRITE_STALL, time_delayed);
  }

  // If DB is not in read-only mode and write_controller is not stopping
  // writes, we can ignore any background errors and allow the write to
  // proceed
  Status s;
  if (write_controller_.IsStopped()) {
    if (!shutting_down_.load(std::memory_order_relaxed)) {
      // If writes are still stopped and db not shutdown, it means we bailed
      // due to a background error
      s = Status::Incomplete(error_handler_.GetBGError().ToString());
    } else {
      s = Status::ShutdownInProgress("stalled writes");
    }
  }
  if (error_handler_.IsDBStopped()) {
    s = error_handler_.GetBGError();
  }
  return s;
}

// REQUIRES: mutex_ is held
// REQUIRES: this thread is currently at the front of the writer queue
Status DBImpl::WriteBufferManagerStallWrites(size_t wal_partition,
                                             WriteContext* write_context) {
  mutex_.AssertHeld();
  WriteThread& write_thread = PartitionWriteThread(wal_partition);
  WBMStallInterface* const stall =
      static_cast<WBMStallInterface*>(wbm_stall_[wal_partition].get());
  for (;;) {
    // First block future writer threads who want to add themselves to the
    // queue of WriteThread.
    write_thread.BeginWriteStall();
    const bool hid_flush_initiator =
        wbm_flush_initiator_ != nullptr &&
        write_buffer_manager_->ShouldTrackFlushInitiator();
    if (hid_flush_initiator && wbm_leaders_hiding_flush_initiator_++ == 0) {
      wbm_flush_initiator_->SetFlushable(false);
    }
    mutex_.Unlock();

    stall->SetState(WBMStallInterface::State::BLOCKED);
    write_buffer_manager_->BeginWriteStall(stall);
    TEST_SYNC_POINT("DBImpl::WriteBufferManagerStallWrites:BeforeBlock");
    bool retry_local_flush = false;
    if (write_buffer_manager_->ShouldTrackFlushInitiator()) {
      for (;;) {
        if (stall->BlockFor(
                write_buffer_manager_->GetLocalFlushRetryMicros())) {
          break;
        }
        if (write_buffer_manager_->TryAcquireLocalFlush(stall)) {
          write_buffer_manager_->RemoveDBFromQueue(stall);
          retry_local_flush = true;
          break;
        }
      }
    } else {
      stall->Block();
    }
    TEST_SYNC_POINT_CALLBACK("DBImpl::WriteBufferManagerStallWrites:Unblocked",
                             &retry_local_flush);

    mutex_.Lock();
    // Allow this front writer to retry the manager-wide emergency lease. It
    // must not let its write through while the hard limit is still exceeded.
    write_thread.EndWriteStall();
    // Restore the state hidden by this stall even if the runtime policy
    // changed while the writer was blocked. Otherwise switching back to the
    // across-DB policy can leave this DB permanently ineligible.
    if (hid_flush_initiator && --wbm_leaders_hiding_flush_initiator_ == 0) {
      wbm_flush_initiator_->SetFlushable(true);
      if (write_buffer_manager_->ShouldTrackFlushInitiator()) {
        write_buffer_manager_->NotifyFlushInitiatorChanged();
      }
    }
    if (!retry_local_flush || !write_buffer_manager_->ShouldStall()) {
      return Status::OK();
    }
    CloseMemTableSwitchGate();
    WaitForPendingWrites();
    Status status = HandleWriteBufferManagerFlush(write_context);
    OpenMemTableSwitchGate();
    if (!status.ok()) {
      return status;
    }
  }
}

void DBImpl::CloseMemTableSwitchGate() {
  mutex_.AssertHeld();
  if (NumWalPartitions() == 1) {
    return;
  }
  // Another leader's switch may still be in flight with mutex_ released.
  while (switch_gate_closed_.load()) {
    bg_cv_.Wait();
  }
  switch_gate_closed_.store(true);
  while (leaders_past_switch_gate_.load() != 0) {
    TEST_SYNC_POINT("DBImpl::CloseMemTableSwitchGate:WaitForLeaders");
    bg_cv_.Wait();
  }
}

void DBImpl::OpenMemTableSwitchGate() {
  mutex_.AssertHeld();
  if (NumWalPartitions() == 1) {
    return;
  }
  switch_gate_closed_.store(false);
  bg_cv_.SignalAll();
}

void DBImpl::EnterMemTableSwitchGate() {
  if (NumWalPartitions() == 1) {
    return;
  }
  // Pairs with CloseMemTableSwitchGate(): each side publishes its own flag
  // before reading the other's, so at least one of them sees the other.
  while (true) {
    leaders_past_switch_gate_.fetch_add(1);
    if (!switch_gate_closed_.load()) {
      return;
    }
    ExitMemTableSwitchGate();
    InstrumentedMutexLock l(&mutex_);
    while (switch_gate_closed_.load()) {
      bg_cv_.Wait();
    }
  }
}

void DBImpl::ExitMemTableSwitchGate() {
  if (NumWalPartitions() == 1) {
    return;
  }
  if (leaders_past_switch_gate_.fetch_sub(1) == 1 &&
      switch_gate_closed_.load()) {
    // Taking mutex_ orders this signal after the closer's last check, which
    // it makes holding mutex_.
    InstrumentedMutexLock l(&mutex_);
    bg_cv_.SignalAll();
  }
}

Status DBImpl::ThrottleLowPriWritesIfNeeded(const WriteOptions& write_options,
                                            WriteBatch* my_batch) {
  assert(write_options.low_pri);
  // This is called outside the DB mutex. Although it is safe to make the call,
  // the consistency condition is not guaranteed to hold. It's OK to live with
  // it in this case.
  // If we need to speed compaction, it means the compaction is left behind
  // and we start to limit low pri writes to a limit.
  if (write_controller_.NeedSpeedupCompaction()) {
    if (allow_2pc() && (my_batch->HasCommit() || my_batch->HasRollback())) {
      // For 2PC, we only rate limit prepare, not commit.
      return Status::OK();
    }
    if (write_options.no_slowdown) {
      return Status::Incomplete("Low priority write stall");
    } else {
      assert(my_batch != nullptr);
      // Rate limit those writes. The reason that we don't completely wait
      // is that in case the write is heavy, low pri writes may never have
      // a chance to run. Now we guarantee we are still slowly making
      // progress.
      PERF_TIMER_FOR_WAIT_GUARD(write_delay_time);
      auto data_size = my_batch->GetDataSize();
      while (data_size > 0) {
        size_t allowed = write_controller_.low_pri_rate_limiter()->RequestToken(
            data_size, 0 /* alignment */, Env::IO_HIGH, nullptr /* stats */,
            RateLimiter::OpType::kWrite);
        data_size -= allowed;
      }
    }
  }
  return Status::OK();
}

void DBImpl::MaybeFlushStatsCF(autovector<ColumnFamilyData*>* cfds) {
  assert(cfds != nullptr);
  if (!cfds->empty() && immutable_db_options_.persist_stats_to_disk) {
    ColumnFamilyData* cfd_stats =
        versions_->GetColumnFamilySet()->GetColumnFamily(
            kPersistentStatsColumnFamilyName);
    if (cfd_stats != nullptr && !cfd_stats->mem()->IsEmpty()) {
      for (ColumnFamilyData* cfd : *cfds) {
        if (cfd == cfd_stats) {
          // stats CF already included in cfds
          return;
        }
      }
      // force flush stats CF when its log number is less than all other CF's
      // log numbers
      bool force_flush_stats_cf = true;
      for (auto* loop_cfd : *versions_->GetColumnFamilySet()) {
        if (loop_cfd == cfd_stats) {
          continue;
        }
        if (loop_cfd->GetLogNumber() <= cfd_stats->GetLogNumber()) {
          force_flush_stats_cf = false;
        }
      }
      if (force_flush_stats_cf) {
        cfds->push_back(cfd_stats);
        ROCKS_LOG_INFO(immutable_db_options_.info_log,
                       "Force flushing stats CF with automated flush "
                       "to avoid holding old logs");
      }
    }
  }
}

Status DBImpl::TrimMemtableHistory(WriteContext* context) {
  autovector<ColumnFamilyData*> cfds;
  ColumnFamilyData* tmp_cfd;
  while ((tmp_cfd = trim_history_scheduler_.TakeNextColumnFamily()) !=
         nullptr) {
    cfds.push_back(tmp_cfd);
  }
  for (auto& cfd : cfds) {
    autovector<MemTable*> to_delete;
    bool trimmed = cfd->imm()->TrimHistory(&context->memtables_to_free_,
                                           cfd->mem()->MemoryAllocatedBytes());
    if (trimmed) {
      context->superversion_context.NewSuperVersion();
      assert(context->superversion_context.new_superversion.get() != nullptr);
      cfd->InstallSuperVersion(&context->superversion_context, &mutex_);
    }

    if (cfd->UnrefAndTryDelete()) {
      cfd = nullptr;
    }
  }
  return Status::OK();
}

Status DBImpl::ScheduleFlushes(WriteContext* context) {
  autovector<ColumnFamilyData*> cfds;
  FlushReason atomic_flush_reason = FlushReason::kWriteBufferFull;
  if (immutable_db_options_.atomic_flush) {
    // Atomic flush has one request-level reason. Derive it only from CFs that
    // scheduled the flush, not from every CF selected into the atomic cut.
    ColumnFamilyData* trigger_cfd;
    while ((trigger_cfd = flush_scheduler_.TakeNextColumnFamily()) != nullptr) {
      if (!trigger_cfd->mem()->IsEmpty() &&
          trigger_cfd->mem()->GetFlushReason() ==
              FlushReason::kMemtableMaxRangeDeletions) {
        atomic_flush_reason = FlushReason::kMemtableMaxRangeDeletions;
      }
      trigger_cfd->UnrefAndTryDelete();
    }
    SelectColumnFamiliesForAtomicFlush(&cfds);
    for (auto cfd : cfds) {
      cfd->Ref();
    }
  } else {
    ColumnFamilyData* tmp_cfd;
    while ((tmp_cfd = flush_scheduler_.TakeNextColumnFamily()) != nullptr) {
      cfds.push_back(tmp_cfd);
    }
    MaybeFlushStatsCF(&cfds);
  }
  Status status;
  AllPartitionsWriter nonmem_w(this, /*nonmem=*/true);
  if (two_write_queues_) {
    nonmem_w.EnterUnbatched();
  }

  TEST_SYNC_POINT_CALLBACK("DBImpl::ScheduleFlushes:PreSwitchMemtable",
                           nullptr);
  autovector<FlushReason> flush_reasons;
  flush_reasons.reserve(cfds.size());
  const bool defer_wbm_accounting_refresh =
      immutable_db_options_.atomic_flush && cfds.size() > 1;
  bool switch_attempted = false;
  for (auto& cfd : cfds) {
    FlushReason flush_reason = FlushReason::kWriteBufferFull;
    if (status.ok() && !cfd->mem()->IsEmpty()) {
      flush_reason = cfd->mem()->GetFlushReason();
      switch_attempted = true;
      status = defer_wbm_accounting_refresh
                   ? SwitchMemtableImpl(cfd, context, nullptr, 0,
                                        false /* refresh_wbm_accounting */)
                   : SwitchMemtable(cfd, context);
    }
    flush_reasons.push_back(flush_reason);
    if (cfd->UnrefAndTryDelete()) {
      cfd = nullptr;
    }
  }
  if (defer_wbm_accounting_refresh && switch_attempted) {
    UpdateWBMAccountingAfterMemtableSwitch(true /* refresh_now */);
  }

  if (two_write_queues_) {
    nonmem_w.ExitUnbatched();
  }

  if (status.ok()) {
    if (immutable_db_options_.atomic_flush) {
      AssignAtomicFlushSeq(cfds);
      FlushRequest flush_req;
      flush_req.atomic_flush = true;
      GenerateFlushRequest(cfds, atomic_flush_reason, &flush_req);
      EnqueuePendingFlush(flush_req);
    } else {
      assert(flush_reasons.size() == cfds.size());
      for (size_t i = 0; i < cfds.size(); ++i) {
        FlushRequest flush_req;
        GenerateFlushRequest({cfds[i]}, flush_reasons[i], &flush_req);
        EnqueuePendingFlush(flush_req);
      }
    }
    MaybeScheduleFlushOrCompaction();
  }
  return status;
}

void DBImpl::NotifyOnMemTableSealed(ColumnFamilyData* /*cfd*/,
                                    const MemTableInfo& mem_table_info) {
  if (immutable_db_options_.listeners.size() == 0U) {
    return;
  }
  if (shutting_down_.load(std::memory_order_acquire)) {
    return;
  }

  mutex_.Unlock();
  for (const auto& listener : immutable_db_options_.listeners) {
    listener->OnMemTableSealed(mem_table_info);
  }
  mutex_.Lock();
}

Status DBImpl::SwitchMemtable(ColumnFamilyData* cfd, WriteContext* context,
                              ReadOnlyMemTable* new_imm,
                              SequenceNumber last_seqno) {
  return SwitchMemtableImpl(cfd, context, new_imm, last_seqno,
                            true /* refresh_wbm_accounting */);
}

void DBImpl::UpdateWBMAccountingAfterMemtableSwitch(bool refresh_now) {
  mutex_.AssertHeld();
  if (wbm_flush_initiator_ == nullptr ||
      !write_buffer_manager_->ShouldTrackFlushInitiator()) {
    return;
  }
  if (refresh_now) {
    TEST_SYNC_POINT("DBImpl::UpdateWBMAccountingAfterMemtableSwitch:Refresh");
    RefreshFlushableMemAccounting();
  } else {
    // Do not let the sorter consume an intermediate snapshot while a caller is
    // switching a batch of memtables. The caller refreshes once after the
    // complete batch, including a partial-error exit.
    wbm_flush_initiator_->InvalidateLargestFlushableCFMem();
  }
}

Status DBImpl::SwitchMemtableImpl(ColumnFamilyData* cfd, WriteContext* context,
                                  ReadOnlyMemTable* new_imm,
                                  SequenceNumber last_seqno,
                                  bool refresh_wbm_accounting) {
  mutex_.AssertHeld();
  assert(lock_wal_owner_thread_id_counts_.empty());

  // TODO: plumb Env::IOActivity, Env::IOPriority
  const WriteOptions write_options;

  log::Writer* new_log = nullptr;
  // WALs for partitions 1..N-1; partition 0's is new_log.
  std::vector<uint64_t> extra_wal_numbers;
  std::vector<log::Writer*> extra_wals;
  UnpublishedWAL prepared_wal;
  MemTable* new_mem = nullptr;
  IOStatus io_s;
  bool current_wal_write_buffer_failed = false;
  uint64_t failed_wal_number = 0;

  // Covers WriteRecoverableState's stamp and the new memtable's earliest
  // sequence.
  AssertLastSequenceCoversAllocated();
  // Recoverable state is persisted in WAL. After memtable switch, WAL might
  // be deleted, so we write the state to memtable to be persisted as well.
  Status s = WriteRecoverableState();
  if (!s.ok()) {
    return s;
  }

  // Attempt to switch to a new memtable and trigger flush of old.
  // Do this without holding the dbmutex lock.
  assert(versions_->prev_log_number() == 0);
  if (two_write_queues_) {
    WalWriteMutex(0).Lock();
  }
  bool creating_new_log =
      !WalEmpty(0) || (recovering_from_wal_write_error_ &&
                       cur_wal_number_ <= recover_wal_through_number_);
  if (two_write_queues_) {
    WalWriteMutex(0).Unlock();
  }
  // All partitions rotate together, so any partition holding appends rotates
  // every partition.
  for (size_t p = 1; p < NumWalPartitions() && !creating_new_log; ++p) {
    if (two_write_queues_) {
      WalWriteMutex(p).Lock();
    }
    creating_new_log = !WalEmpty(p);
    if (two_write_queues_) {
      WalWriteMutex(p).Unlock();
    }
  }
  uint64_t recycle_log_number = 0;
  // If file deletion is disabled, don't recycle logs since it'll result in
  // the file getting renamed
  if (creating_new_log && immutable_db_options_.recycle_log_file_num &&
      !wal_recycle_files_.empty() && IsFileDeletionsEnabled()) {
    recycle_log_number = wal_recycle_files_.front();
  }
  uint64_t new_log_number = cur_wal_number_;
  if (creating_new_log) {
    prepared_wal = WaitForAsyncWALPrecreate();
    if (prepared_wal.writer) {
      new_log_number = prepared_wal.log_number;
    } else {
      assert(prepared_wal.log_number == 0);
      new_log_number = versions_->NewFileNumber();
    }
    // Partition 0 takes the lowest number of the new generation, so
    // cur_wal_number_ still bounds from below every WAL written after this
    // switch.
    for (size_t p = 1; p < NumWalPartitions(); ++p) {
      extra_wal_numbers.push_back(versions_->NewFileNumber());
      assert(extra_wal_numbers.back() > new_log_number);
    }
  }
  // For use outside of holding DB mutex
  const MutableCFOptions mutable_cf_options_copy =
      cfd->GetLatestMutableCFOptions();

  // Set memtable_info for memtable sealed callback
  // TODO: memtable_info for `new_imm`
  MemTableInfo memtable_info;
  memtable_info.cf_name = cfd->GetName();
  memtable_info.first_seqno = cfd->mem()->GetFirstSequenceNumber();
  memtable_info.earliest_seqno = cfd->mem()->GetEarliestSequenceNumber();
  memtable_info.num_entries = cfd->mem()->NumEntries();
  memtable_info.num_deletes = cfd->mem()->NumDeletion();
  if (!cfd->ioptions().persist_user_defined_timestamps &&
      cfd->user_comparator()->timestamp_size() > 0) {
    Slice newest_udt = cfd->mem()->GetNewestUDT();
    memtable_info.newest_udt.assign(newest_udt.data(), newest_udt.size());
  }
  // Log this later after lock release. It may be outdated, e.g., if background
  // flush happens before logging, but that should be ok.
  int num_imm_unflushed = cfd->imm()->NumNotFlushed();
  const DBOptions db_options_snapshot =
      BuildDBOptions(immutable_db_options_, mutable_db_options_);
  const auto preallocate_block_size =
      GetWalPreallocateBlockSize(mutable_cf_options_copy.write_buffer_size);
  // Every writer is excluded (the switch gate or all queues entered), so
  // every index allocated so far is in a current WAL or an older one.
  uint64_t wal_index_high = 0;
  if (log::WALIndexEnabled(immutable_db_options_.partition_wal_usage)) {
    InstrumentedMutexLock l(&sequence_wal_index_mutex_);
    wal_index_high = next_wal_index_ - 1;
  }
  mutex_.Unlock();
  auto wal_index_predecessor = [this, wal_index_high](size_t partition) {
    log::WALIndexPredecessorSpan span;
    if (!log::WALIndexEnabled(immutable_db_options_.partition_wal_usage)) {
      return span;
    }
    WalWriteMutexLock l(&WalWriteMutex(partition));
    const LogWriterNumber* current_wal = CurrentWalOrNull(partition);
    if (current_wal != nullptr) {
      span = log::MakeWALIndexPredecessorSpan(
          current_wal->writer->get_log_number(),
          current_wal->writer->first_wal_index_recorded(), wal_index_high);
    }
    return span;
  };
  auto predecessor_wal_info = [this](size_t partition) {
    PredecessorWALInfo info;
    WalWriteMutexLock l(&WalWriteMutex(partition));
    const LogWriterNumber* current_wal = CurrentWalOrNull(partition);
    if (current_wal != nullptr) {
      log::Writer* cur_log_writer = current_wal->writer;
      info = PredecessorWALInfo(cur_log_writer->get_log_number(),
                                cur_log_writer->file()->GetFileSize(),
                                cur_log_writer->GetLastSeqnoRecorded());
    }
    return info;
  };
  if (creating_new_log) {
    PredecessorWALInfo info = predecessor_wal_info(0);
    // TODO: Write buffer size passed in should be max of all CF's instead
    // of mutable_cf_options.write_buffer_size.
    const log::WALIndexPredecessorSpan span = wal_index_predecessor(0);
    if (prepared_wal.writer) {
      io_s = StartWALFile(write_options, info, span, prepared_wal.writer.get());
      if (io_s.ok()) {
        new_log = prepared_wal.writer.release();
      }
    } else {
      io_s = CreateWAL(db_options_snapshot, write_options, new_log_number,
                       recycle_log_number, preallocate_block_size, info, span,
                       /*wal_partition=*/0, &new_log);
    }
    TEST_SYNC_POINT_CALLBACK("DBImpl::SwitchMemtable:AfterCreateWAL", &io_s);
    if (s.ok()) {
      s = io_s;
    }
    for (size_t i = 0; s.ok() && i < extra_wal_numbers.size(); ++i) {
      log::Writer* extra_wal = nullptr;
      io_s = CreateWAL(db_options_snapshot, write_options, extra_wal_numbers[i],
                       /*recycle_log_number=*/0, preallocate_block_size,
                       predecessor_wal_info(i + 1),
                       wal_index_predecessor(i + 1), i + 1, &extra_wal);
      s = io_s;
      if (s.ok()) {
        extra_wals.push_back(extra_wal);
      }
    }
  }
  if (!s.ok() && prepared_wal.writer) {
    const uint64_t prepared_wal_number = prepared_wal.log_number;
    const std::string prepared_wal_fname =
        LogFileName(immutable_db_options_.GetWalDir(), prepared_wal_number);
    prepared_wal.Reset();
    Status delete_s = env_->DeleteFile(prepared_wal_fname);
    if (!delete_s.ok() && !delete_s.IsNotFound()) {
      ROCKS_LOG_WARN(immutable_db_options_.info_log,
                     "Failed to delete unstarted async precreated WAL #%" PRIu64
                     ": %s",
                     prepared_wal_number, delete_s.ToString().c_str());
    }
    delete_s.PermitUncheckedError();
  }
  if (s.ok()) {
    // FIXME: from the comment for GetEarliestSequenceNumber(), any key with
    //  seqno >= earliest_seqno should be in this or later memtable. This means
    //  we should use LastSequence() + 1 or last_seqno + 1 here. And it needs to
    //  be incremented with file ingestion and other operations that consumes
    //  sequence number.
    SequenceNumber seq;
    if (new_imm) {
      assert(last_seqno > versions_->LastSequence());
      seq = last_seqno;
    } else {
      seq = versions_->LastSequence();
    }
    new_mem = cfd->ConstructNewMemtable(mutable_cf_options_copy,
                                        /*earliest_seq=*/seq);
    context->superversion_context.NewSuperVersion();

    ROCKS_LOG_INFO(immutable_db_options_.info_log,
                   "[%s] New memtable created with log file: #%" PRIu64
                   ". Immutable memtables: %d.\n",
                   cfd->GetName().c_str(), new_log_number, num_imm_unflushed);
  }

  // MarkImmutable() blocks concurrent AddLogicallyRedundantRangeTombstone()
  // from reader threads. Once blocked, safe to construct the fragmented range
  // tombstone list. Both calls happen outside the DB mutex to avoid increasing
  // write stall. MarkImmutable() is idempotent, so Add() calling it again
  // inside the mutex is harmless.
  cfd->mem()->MarkImmutable();
  cfd->mem()->ConstructFragmentedRangeTombstones();

  mutex_.Lock();
  if (recycle_log_number != 0) {
    // Since renaming the file is done outside DB mutex, we need to ensure
    // concurrent full purges don't delete the file while we're recycling it.
    // To achieve that we hold the old log number in the recyclable list until
    // after it has been renamed.
    assert(wal_recycle_files_.front() == recycle_log_number);
    wal_recycle_files_.pop_front();
  }
  if (s.ok() && creating_new_log) {
    // Rotation holds every partition's mutex (plus mutex_, already held):
    // readers of the globals touched below hold mutex_ or all N, so they
    // always share one of those.
    AllWalWriteMutexesLock l(this);
    assert(new_log != nullptr);
    for (size_t p = 0; s.ok() && p < NumWalPartitions(); ++p) {
      const LogWriterNumber* current_wal = CurrentWalOrNull(p);
      if (current_wal == nullptr) {
        continue;
      }
      // Always flush the buffer of the last log before switching to a new one,
      // except when recovery is replacing a file-scoped failed WAL. Its
      // accepted contents are persisted by the atomic recovery flush.
      log::Writer* cur_log_writer = current_wal->writer;
      const bool abandon_current_wal =
          recovering_from_wal_write_error_ &&
          cur_log_writer->get_log_number() <= recover_wal_through_number_;
      if (abandon_current_wal) {
        TEST_SYNC_POINT("DBImpl::SwitchMemtable:AbandonWAL");
        ROCKS_LOG_WARN(immutable_db_options_.info_log,
                       "[WAL recovery] [%s] Replacing failed WAL #%" PRIu64
                       " with WAL #%" PRIu64,
                       cfd->GetName().c_str(), cur_log_writer->get_log_number(),
                       new_log_number);
      } else if (error_handler_.IsRecoveryInProgress()) {
        // In recovery path, we force another try of writing WAL buffer.
        cur_log_writer->file()->reset_seen_error();
      }
      if (!abandon_current_wal) {
        // Progress policy: with more than one partition, every retiring WAL
        // asserts that nothing below the new generation's first index is
        // still coming from it, so a live merge waiting on an idle partition
        // advances at the next rotation.
        io_s = NumWalPartitions() > 1
                   ? cur_log_writer->AddWALIndexProgressRecord(
                         write_options, wal_index_high + 1)
                   : IOStatus::OK();
        if (io_s.ok()) {
          io_s = cur_log_writer->WriteBuffer(write_options);
        }
        current_wal_write_buffer_failed = !io_s.ok();
        if (current_wal_write_buffer_failed) {
          failed_wal_number = cur_log_writer->get_log_number();
        }
      }
      if (s.ok()) {
        s = io_s;
      }
      if (!s.ok()) {
        ROCKS_LOG_WARN(immutable_db_options_.info_log,
                       "[%s] Failed to switch from #%" PRIu64 " to #%" PRIu64
                       "  WAL file\n",
                       cfd->GetName().c_str(), cur_log_writer->get_log_number(),
                       new_log_number);
      }
    }
    if (s.ok()) {
      cur_wal_number_ = new_log_number;
      wal_dir_synced_ = false;
      for (size_t p = 0; p < NumWalPartitions(); ++p) {
        const uint64_t number =
            p == 0 ? new_log_number : extra_wal_numbers[p - 1];
        assert(p == 0 || number > cur_wal_number_);
        WalEmpty(p) = true;
        InstallCurrentWal(p, number, p == 0 ? new_log : extra_wals[p - 1]);
      }
    }
  }

  if (!s.ok()) {
    // how do we fail if we're not creating new log?
    assert(creating_new_log);
    delete new_mem;
    delete new_log;
    for (log::Writer* extra_wal : extra_wals) {
      delete extra_wal;
    }
    context->superversion_context.new_superversion.reset();
    // Only a WriteBuffer() failure belongs to the current WAL. CreateWAL() and
    // StartWALFile() operate on an unpublished replacement, so a failure there
    // must not quarantine an otherwise healthy current WAL.
    if (!io_s.ok()) {
      if (!current_wal_write_buffer_failed) {
        ROCKS_LOG_WARN(immutable_db_options_.info_log,
                       "Failed to create or start replacement WAL #%" PRIu64
                       "; current WAL #%" PRIu64 " remains active: %s",
                       new_log_number, cur_wal_number_,
                       io_s.ToString().c_str());
      }
      if (current_wal_write_buffer_failed && io_s.IsIOError() &&
          !io_s.IsIOFenced() &&
          io_s.GetScope() == IOStatus::IOErrorScope::kIOErrorScopeFile) {
        DBRecoverContext recovery_context;
        recovery_context.failed_wal_number = failed_wal_number;
        ExtendWALRecoveryCutoff(recovery_context.failed_wal_number);
        error_handler_.SetBGError(io_s, BackgroundErrorReason::kMemTable,
                                  /*wal_related=*/true, recovery_context);
      } else {
        error_handler_.SetBGError(io_s, BackgroundErrorReason::kMemTable,
                                  /*wal_related=*/true);
      }
    } else {
      error_handler_.SetBGError(s, BackgroundErrorReason::kMemTable);
    }
    // Read back bg_error in order to get the right severity. A WAL I/O
    // failure while rotating wakes parked completion gates through the
    // transition hook in the SetBGError above, the same path as every other
    // background error; no explicit wake (and no mutex dance) here.
    s = error_handler_.GetBGError();
    UpdateWBMAccountingAfterMemtableSwitch(refresh_wbm_accounting);
    return s;
  }

  bool empty_cf_updated = false;
  if (immutable_db_options_.track_and_verify_wals_in_manifest &&
      !immutable_db_options_.allow_2pc && creating_new_log) {
    // In non-2pc mode, WALs become obsolete if they do not contain unflushed
    // data. Updating the empty CF's log number might cause some WALs to become
    // obsolete. So we should track the WAL obsoletion event before actually
    // updating the empty CF's log number.
    uint64_t min_wal_number_to_keep =
        versions_->PreComputeMinLogNumberWithUnflushedData(cur_wal_number_);
    if (min_wal_number_to_keep >
        versions_->GetWalSet().GetMinWalNumberToKeep()) {
      // TODO: plumb Env::IOActivity, Env::IOPriority
      const ReadOptions read_options;
      // Get a snapshot of the empty column families.
      // LogAndApply may release and reacquire db
      // mutex, during that period, column family may become empty (e.g. its
      // flush succeeds), then it affects the computed min_log_number_to_keep,
      // so we take a snapshot for consistency of column family data
      // status. If a column family becomes non-empty afterwards, its active log
      // should still be the created new log, so the min_log_number_to_keep is
      // not affected.
      autovector<ColumnFamilyData*> empty_cfs;
      for (auto cf : *versions_->GetColumnFamilySet()) {
        if (cf->IsEmpty()) {
          empty_cfs.push_back(cf);
        }
      }

      VersionEdit wal_deletion;
      wal_deletion.DeleteWalsBefore(min_wal_number_to_keep);
      s = versions_->LogAndApplyToDefaultColumnFamily(
          read_options, write_options, &wal_deletion, &mutex_,
          directories_.GetDbDir());
      if (!s.ok() && versions_->io_status().IsIOError()) {
        error_handler_.SetBGError(versions_->io_status(),
                                  BackgroundErrorReason::kManifestWrite);
      }
      if (!s.ok()) {
        UpdateWBMAccountingAfterMemtableSwitch(refresh_wbm_accounting);
        return s;
      }

      for (auto cf : empty_cfs) {
        if (cf->IsEmpty()) {
          cf->SetLogNumber(cur_wal_number_);
          // MEMPURGE: No need to change this, because new adds
          // should still receive new sequence numbers.
          cf->mem()->SetCreationSeq(versions_->LastSequence());
        }  // cf may become non-empty.
      }
      empty_cf_updated = true;
    }
  }
  if (!empty_cf_updated) {
    for (auto cf : *versions_->GetColumnFamilySet()) {
      // all this is just optimization to delete logs that
      // are no longer needed -- if CF is empty, that means it
      // doesn't need that particular log to stay alive, so we just
      // advance the log number. no need to persist this in the manifest
      if (cf->IsEmpty()) {
        if (creating_new_log) {
          cf->SetLogNumber(cur_wal_number_);
        }
        cf->mem()->SetCreationSeq(versions_->LastSequence());
      }
    }
  }

  if (cfd->blob_partition_manager() != nullptr) {
    cfd->blob_partition_manager()->RotateCurrentGeneration();
  }

  cfd->mem()->SetNextLogNumber(cur_wal_number_);
  assert(new_mem != nullptr);
  cfd->imm()->Add(cfd->mem(), &context->memtables_to_free_);
  TEST_SYNC_POINT(
      "DBImpl::SwitchMemtable:AfterConstructFragmentedRangeTombstones");
  if (new_imm) {
    // Need to assign memtable id here before SetMemtable() below assigns id to
    // the new live memtable
    cfd->AssignMemtableID(new_imm);
    // NOTE: new_imm and cfd->mem() references the same WAL and has the same
    // NextLogNumber(). They should be flushed together. For non-atomic-flush,
    // we always try to flush all immutable memtable. For atomic flush, these
    // two memtables will be marked eligible for flush in the same call to
    // AssignAtomicFlushSeq().
    new_imm->SetNextLogNumber(cur_wal_number_);
    cfd->imm()->Add(new_imm, &context->memtables_to_free_);
  }
  new_mem->Ref();
  cfd->SetMemtable(new_mem);
  UpdateWBMAccountingAfterMemtableSwitch(refresh_wbm_accounting);
  InstallSuperVersionAndScheduleWork(cfd, &context->superversion_context);
  MaybeScheduleAsyncWALPrecreate(preallocate_block_size);

  // Notify client that memtable is sealed, now that we have successfully
  // installed a new memtable
  NotifyOnMemTableSealed(cfd, memtable_info);
  // It is possible that we got here without checking the value of i_os, but
  // that is okay.  If we did, it most likely means that s was already an error.
  // In any case, ignore any unchecked error for i_os here.
  io_s.PermitUncheckedError();
  // We guarantee that if a non-ok status is returned, `new_imm` was not added
  // to the db.
  assert(s.ok());
  return s;
}

size_t DBImpl::GetWalPreallocateBlockSize(uint64_t write_buffer_size) const {
  mutex_.AssertHeld();
  size_t bsize =
      static_cast<size_t>(write_buffer_size / 10 + write_buffer_size);
  // Some users might set very high write_buffer_size and rely on
  // max_total_wal_size or other parameters to control the WAL size.
  if (mutable_db_options_.max_total_wal_size > 0) {
    bsize = std::min<size_t>(
        bsize, static_cast<size_t>(mutable_db_options_.max_total_wal_size));
  }
  if (immutable_db_options_.db_write_buffer_size > 0) {
    bsize = std::min<size_t>(bsize, immutable_db_options_.db_write_buffer_size);
  }
  if (immutable_db_options_.write_buffer_manager &&
      immutable_db_options_.write_buffer_manager->enabled()) {
    bsize = std::min<size_t>(
        bsize, immutable_db_options_.write_buffer_manager->buffer_size());
  }

  return bsize;
}

// Default implementations of convenience methods that subclasses of DB
// can call if they wish
Status DB::Put(const WriteOptions& opt, ColumnFamilyHandle* column_family,
               const Slice& key, const Slice& value) {
  // Pre-allocate size of write batch conservatively.
  // 8 bytes are taken by header, 4 bytes for count, 1 byte for type,
  // and we allocate 11 extra bytes for key length, as well as value length.
  WriteBatch batch(key.size() + value.size() + 24, 0 /* max_bytes */,
                   opt.protection_bytes_per_key, 0 /* default_cf_ts_sz */);
  Status s = batch.Put(column_family, key, value);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::Put(const WriteOptions& opt, ColumnFamilyHandle* column_family,
               const Slice& key, const Slice& ts, const Slice& value) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  Status s = batch.Put(column_family, key, ts, value);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::PutEntity(const WriteOptions& options,
                     ColumnFamilyHandle* column_family, const Slice& key,
                     const WideColumns& columns) {
  const ColumnFamilyHandle* const default_cf = DefaultColumnFamily();
  assert(default_cf);

  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);

  WriteBatch batch(/* reserved_bytes */ 0, /* max_bytes */ 0,
                   options.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());

  const Status s = batch.PutEntity(column_family, key, columns);
  if (!s.ok()) {
    return s;
  }

  return Write(options, &batch);
}

Status DB::PutEntity(const WriteOptions& options, const Slice& key,
                     const AttributeGroups& attribute_groups) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   options.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  const Status s = batch.PutEntity(key, attribute_groups);
  if (!s.ok()) {
    return s;
  }
  return Write(options, &batch);
}

Status DB::Delete(const WriteOptions& opt, ColumnFamilyHandle* column_family,
                  const Slice& key) {
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key, 0 /* default_cf_ts_sz */);
  Status s = batch.Delete(column_family, key);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::Delete(const WriteOptions& opt, ColumnFamilyHandle* column_family,
                  const Slice& key, const Slice& ts) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  Status s = batch.Delete(column_family, key, ts);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::SingleDelete(const WriteOptions& opt,
                        ColumnFamilyHandle* column_family, const Slice& key) {
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key, 0 /* default_cf_ts_sz */);
  Status s = batch.SingleDelete(column_family, key);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::SingleDelete(const WriteOptions& opt,
                        ColumnFamilyHandle* column_family, const Slice& key,
                        const Slice& ts) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  Status s = batch.SingleDelete(column_family, key, ts);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::DeleteRange(const WriteOptions& opt,
                       ColumnFamilyHandle* column_family,
                       const Slice& begin_key, const Slice& end_key) {
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key, 0 /* default_cf_ts_sz */);
  Status s = batch.DeleteRange(column_family, begin_key, end_key);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::DeleteRange(const WriteOptions& opt,
                       ColumnFamilyHandle* column_family,
                       const Slice& begin_key, const Slice& end_key,
                       const Slice& ts) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  Status s = batch.DeleteRange(column_family, begin_key, end_key, ts);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::Merge(const WriteOptions& opt, ColumnFamilyHandle* column_family,
                 const Slice& key, const Slice& value) {
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key, 0 /* default_cf_ts_sz */);
  Status s = batch.Merge(column_family, key, value);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

Status DB::Merge(const WriteOptions& opt, ColumnFamilyHandle* column_family,
                 const Slice& key, const Slice& ts, const Slice& value) {
  ColumnFamilyHandle* default_cf = DefaultColumnFamily();
  assert(default_cf);
  const Comparator* const default_cf_ucmp = default_cf->GetComparator();
  assert(default_cf_ucmp);
  WriteBatch batch(0 /* reserved_bytes */, 0 /* max_bytes */,
                   opt.protection_bytes_per_key,
                   default_cf_ucmp->timestamp_size());
  Status s = batch.Merge(column_family, key, ts, value);
  if (!s.ok()) {
    return s;
  }
  return Write(opt, &batch);
}

}  // namespace ROCKSDB_NAMESPACE
