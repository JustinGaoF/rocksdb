//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "db/column_family.h"
#include "db/compaction/compaction_iterator.h"
#include "db/compaction/compaction_job.h"
#include "db/db_impl/wal_durability_tracker.h"
#include "db/error_handler.h"
#include "db/event_helpers.h"
#include "db/external_sst_file_ingestion_job.h"
#include "db/flush_job.h"
#include "db/flush_scheduler.h"
#include "db/import_column_family_job.h"
#include "db/internal_stats.h"
#include "db/log_writer.h"
#include "db/logs_with_prep_tracker.h"
#include "db/memtable_list.h"
#include "db/periodic_task_scheduler.h"
#include "db/post_memtable_callback.h"
#include "db/pre_release_callback.h"
#include "db/range_del_aggregator.h"
#include "db/read_callback.h"
#include "db/seqno_to_time_mapping.h"
#include "db/snapshot_checker.h"
#include "db/snapshot_impl.h"
#include "db/trim_history_scheduler.h"
#include "db/version_edit.h"
#include "db/wal_manager.h"
#include "db/write_controller.h"
#include "db/write_thread.h"
#include "logging/event_logger.h"
#include "memtable/flush_initiator.h"
#include "memtable/wbwi_memtable.h"
#include "monitoring/instrumented_mutex.h"
#include "options/db_options.h"
#include "options/options_helper.h"
#include "port/port.h"
#include "rocksdb/attribute_groups.h"
#include "rocksdb/db.h"
#include "rocksdb/env.h"
#include "rocksdb/memtablerep.h"
#include "rocksdb/status.h"
#include "rocksdb/trace_reader_writer.h"
#include "rocksdb/user_write_callback.h"
#include "rocksdb/utilities/replayer.h"
#include "rocksdb/utilities/write_batch_with_index.h"
#include "rocksdb/wal_iterator.h"
#include "rocksdb/write_buffer_manager.h"
#include "table/merging_iterator.h"
#include "util/autovector.h"
#include "util/coro_utils.h"
#include "util/hash.h"
#include "util/mutexlock.h"
#include "util/repeatable_thread.h"
#include "util/stop_watch.h"
#include "util/thread_local.h"

#if USE_COROUTINES
#include "rocksdb/coro_db.h"
#endif

namespace ROCKSDB_NAMESPACE {

class Arena;
class ArenaWrappedDBIter;
class Cleanable;
class FileIngestionHandleImpl;
class SameFileBlobReader;
class InMemoryStatsHistoryIterator;
class MemTable;
class PersistentStatsHistoryIterator;
class TableCache;
class TaskLimiterToken;
class Version;
class VersionEdit;
class VersionSet;
class WriteCallback;
struct JobContext;
struct ExternalSstFileInfo;
struct MemTableInfo;

// Captured under the same DB mutex hold as a subset checkpoint's MANIFEST
// boundary and live CF set. This prevents the generated OPTIONS membership
// from describing a different point in time than the copied MANIFEST.
struct SubsetCheckpointOptionsSnapshot {
  uint64_t options_file_number = 0;
  bool commit_options_file_number = false;
  DBOptions db_options;
  std::vector<std::string> cf_names;
  std::vector<ColumnFamilyOptions> cf_options;
};

// Class to maintain directories for all database paths other than main one.
class Directories {
 public:
  IOStatus SetDirectories(FileSystem* fs, const std::string& dbname,
                          const std::string& wal_dir,
                          const std::vector<DbPath>& data_paths);

  FSDirectory* GetDataDir(size_t path_id) const {
    assert(path_id < data_dirs_.size());
    FSDirectory* ret_dir = data_dirs_[path_id].get();
    if (ret_dir == nullptr) {
      // Should use db_dir_
      return db_dir_.get();
    }
    return ret_dir;
  }

  FSDirectory* GetWalDir() {
    if (wal_dir_) {
      return wal_dir_.get();
    }
    return db_dir_.get();
  }

  FSDirectory* GetDbDir() { return db_dir_.get(); }

  IOStatus Close(const IOOptions& options, IODebugContext* dbg) {
    // close all directories for all database paths
    IOStatus s = IOStatus::OK();

    // The default implementation for Close() in Directory/FSDirectory class
    // "NotSupported" status, the upper level interface should be able to
    // handle this error so that Close() does not fail after upgrading when
    // run on FileSystems that have not implemented `Directory::Close()` or
    // `FSDirectory::Close()` yet

    if (db_dir_) {
      IOStatus temp_s = db_dir_->Close(options, dbg);
      if (!temp_s.ok() && !temp_s.IsNotSupported() && s.ok()) {
        s = std::move(temp_s);
      }
    }

    // Attempt to close everything even if one fails
    s.PermitUncheckedError();

    if (wal_dir_) {
      IOStatus temp_s = wal_dir_->Close(options, dbg);
      if (!temp_s.ok() && !temp_s.IsNotSupported() && s.ok()) {
        s = std::move(temp_s);
      }
    }

    s.PermitUncheckedError();

    for (auto& data_dir_ptr : data_dirs_) {
      if (data_dir_ptr) {
        IOStatus temp_s = data_dir_ptr->Close(options, dbg);
        if (!temp_s.ok() && !temp_s.IsNotSupported() && s.ok()) {
          s = std::move(temp_s);
        }
      }
    }

    // Ready for caller
    s.MustCheck();
    return s;
  }

 private:
  std::unique_ptr<FSDirectory> db_dir_;
  std::vector<std::unique_ptr<FSDirectory>> data_dirs_;
  std::unique_ptr<FSDirectory> wal_dir_;
};

struct DBOpenLogRecordReadReporter : public log::Reader::Reporter {
  Env* env;
  Logger* info_log;
  const char* fname;
  Status* status;  // nullptr if immutable_db_options_.paranoid_checks==false
  bool* old_log_record;
  void Corruption(size_t bytes, const Status& s,
                  uint64_t log_number = kMaxSequenceNumber) override;

  void OldLogRecord(size_t bytes) override;

  uint64_t GetCorruptedLogNumber() const { return corrupted_wal_number_; }

 private:
  uint64_t corrupted_wal_number_ = kMaxSequenceNumber;
};

// While DB is the public interface of RocksDB, and DBImpl is the actual
// class implementing it. It's the entrance of the core RocksdB engine.
// All other DB implementations, e.g. TransactionDB, BlobDB, etc, wrap a
// DBImpl internally.
// Other than functions implementing the DB interface, some public
// functions are there for other internal components to call. For
// example, TransactionDB directly calls DBImpl::WriteImpl() and
// BlobDB directly calls DBImpl::GetImpl(). Some other functions
// are for sub-components to call. For example, ColumnFamilyHandleImpl
// calls DBImpl::FindObsoleteFiles().
//
// Since it's a very large class, the definition of the functions is
// divided in several db_impl_*.cc files, besides db_impl.cc.
#if USE_COROUTINES
class DBImpl : public DB,
               public CoroDB
#else
class DBImpl : public DB
#endif
{
 public:
  DBImpl(const DBOptions& options, const std::string& dbname,
         const bool seq_per_batch = false, const bool batch_per_txn = true,
         bool read_only = false);
  // No copying or moving allowed
  DBImpl(const DBImpl&) = delete;
  void operator=(const DBImpl&) = delete;
  DBImpl(DBImpl&&) = delete;
  void operator=(DBImpl&&) = delete;

  ~DBImpl() override;

  // Largest DBOptions::wal_partitions Open() accepts. Each partition owns a
  // write thread, a WAL file and a merge buffer, sized at construction, and
  // appends gain nothing once partitions outnumber the device's queue depth;
  // 64 is four times the most any test or the crash test runs.
  static constexpr uint32_t kMaxWalPartitions = 64;

  // ---- Implementations of the DB interface ----

  using DB::Resume;
  Status Resume() override;

  using DB::Put;
  Status Put(const WriteOptions& options, ColumnFamilyHandle* column_family,
             const Slice& key, const Slice& value) override;
  Status Put(const WriteOptions& options, ColumnFamilyHandle* column_family,
             const Slice& key, const Slice& ts, const Slice& value) override;

  using DB::PutEntity;
  Status PutEntity(const WriteOptions& options,
                   ColumnFamilyHandle* column_family, const Slice& key,
                   const WideColumns& columns) override;
  Status PutEntity(const WriteOptions& options, const Slice& key,
                   const AttributeGroups& attribute_groups) override;

  using DB::Merge;
  Status Merge(const WriteOptions& options, ColumnFamilyHandle* column_family,
               const Slice& key, const Slice& value) override;
  Status Merge(const WriteOptions& options, ColumnFamilyHandle* column_family,
               const Slice& key, const Slice& ts, const Slice& value) override;

  using DB::Delete;
  Status Delete(const WriteOptions& options, ColumnFamilyHandle* column_family,
                const Slice& key) override;
  Status Delete(const WriteOptions& options, ColumnFamilyHandle* column_family,
                const Slice& key, const Slice& ts) override;

  using DB::SingleDelete;
  Status SingleDelete(const WriteOptions& options,
                      ColumnFamilyHandle* column_family,
                      const Slice& key) override;
  Status SingleDelete(const WriteOptions& options,
                      ColumnFamilyHandle* column_family, const Slice& key,
                      const Slice& ts) override;

  using DB::DeleteRange;
  Status DeleteRange(const WriteOptions& options,
                     ColumnFamilyHandle* column_family, const Slice& begin_key,
                     const Slice& end_key) override;
  Status DeleteRange(const WriteOptions& options,
                     ColumnFamilyHandle* column_family, const Slice& begin_key,
                     const Slice& end_key, const Slice& ts) override;

  using DB::Write;
  Status Write(const WriteOptions& options, WriteBatch* updates) override;

  using DB::WriteWithCallback;
  Status WriteWithCallback(const WriteOptions& options, WriteBatch* updates,
                           UserWriteCallback* user_write_cb) override;

  Status IngestWriteBatchWithIndex(
      const WriteOptions& options,
      std::shared_ptr<WriteBatchWithIndex> wbwi) override;

  // Returns true if any live column family currently has blob direct write
  // enabled.
  bool HasAnyBlobDirectWriteColumnFamily();

  using DB::Get;
  DECLARE_SYNC_AND_ASYNC_OVERRIDE(Status, Get, const ReadOptions& _read_options,
                                  ColumnFamilyHandle* column_family,
                                  const Slice& key, PinnableSlice* value,
                                  std::string* timestamp);

  using DB::GetWithMetadata;
  Status GetWithMetadata(const ReadOptions& _read_options,
                         ColumnFamilyHandle* column_family, const Slice& key,
                         PinnableSlice* value,
                         OutputMetadata* output_metadata) override;

  using DB::GetEntity;
  DECLARE_SYNC_AND_ASYNC_OVERRIDE(Status, GetEntity,
                                  const ReadOptions& _read_options,
                                  ColumnFamilyHandle* column_family,
                                  const Slice& key,
                                  PinnableWideColumns* columns);
  DECLARE_SYNC_AND_ASYNC_OVERRIDE(Status, GetEntity,
                                  const ReadOptions& _read_options,
                                  const Slice& key,
                                  PinnableAttributeGroups* result);

  Status GetEntityLazy(const ReadOptions& options,
                       ColumnFamilyHandle* column_family, const Slice& key,
                       LazyWideColumns* result) override;

  using DB::GetMergeOperands;
  Status GetMergeOperands(const ReadOptions& options,
                          ColumnFamilyHandle* column_family, const Slice& key,
                          PinnableSlice* merge_operands,
                          GetMergeOperandsOptions* get_merge_operands_options,
                          int* number_of_operands) override {
    GetImplOptions get_impl_options;
    get_impl_options.column_family = column_family;
    get_impl_options.merge_operands = merge_operands;
    get_impl_options.get_merge_operands_options = get_merge_operands_options;
    get_impl_options.number_of_operands = number_of_operands;
    get_impl_options.get_value = false;
    return GetImpl(options, key, get_impl_options);
  }

  using DB::MultiGet;
  // This MultiGet is a batched version, which may be faster than calling Get
  // multiple times, especially if the keys have some spatial locality that
  // enables them to be queried in the same SST files/set of files. The larger
  // the batch size, the more scope for batching and performance improvement
  // The values and statuses parameters are arrays with number of elements
  // equal to keys.size(). This allows the storage for those to be alloacted
  // by the caller on the stack for small batches
  DECLARE_SYNC_AND_ASYNC_OVERRIDE(void, MultiGet,
                                  const ReadOptions& _read_options,
                                  const size_t num_keys,
                                  ColumnFamilyHandle** column_families,
                                  const Slice* keys, PinnableSlice* values,
                                  std::string* timestamps, Status* statuses,
                                  const bool sorted_input = false);

  using DB::MultiGetWithMetadata;
  void MultiGetWithMetadata(const ReadOptions& _read_options,
                            const size_t num_keys,
                            ColumnFamilyHandle* const* column_families,
                            const Slice* keys, PinnableSlice* values,
                            Status* statuses,
                            MultiGetOutputMetadata* output_metadata,
                            const bool sorted_input = false) override;

  void MultiGetWithCallback(
      const ReadOptions& _read_options, ColumnFamilyHandle* column_family,
      ReadCallback* callback,
      autovector<KeyContext*, MultiGetContext::MAX_BATCH_SIZE>* sorted_keys);

  using DB::MultiGetEntity;

  void MultiGetEntity(const ReadOptions& options,
                      ColumnFamilyHandle* column_family, size_t num_keys,
                      const Slice* keys, PinnableWideColumns* results,
                      Status* statuses, bool sorted_input) override;

  void MultiGetEntity(const ReadOptions& options, size_t num_keys,
                      ColumnFamilyHandle** column_families, const Slice* keys,
                      PinnableWideColumns* results, Status* statuses,
                      bool sorted_input) override;
  void MultiGetEntity(const ReadOptions& options, size_t num_keys,
                      const Slice* keys,
                      PinnableAttributeGroups* results) override;

  void MultiGetEntityLazy(const ReadOptions& options,
                          ColumnFamilyHandle* column_family, size_t num_keys,
                          const Slice* keys, LazyWideColumnsBatch* result,
                          Status* statuses, bool sorted_input) override;

  void MultiGetEntityWithCallback(
      const ReadOptions& read_options, ColumnFamilyHandle* column_family,
      ReadCallback* callback,
      autovector<KeyContext*, MultiGetContext::MAX_BATCH_SIZE>* sorted_keys);

  Status CreateColumnFamily(const ColumnFamilyOptions& cf_options,
                            const std::string& column_family,
                            ColumnFamilyHandle** handle) override {
    // TODO: plumb Env::IOActivity, Env::IOPriority
    return CreateColumnFamily(ReadOptions(), WriteOptions(), cf_options,
                              column_family, handle);
  }
  virtual Status CreateColumnFamily(const ReadOptions& read_options,
                                    const WriteOptions& write_options,
                                    const ColumnFamilyOptions& cf_options,
                                    const std::string& column_family,
                                    ColumnFamilyHandle** handle);
  Status CreateColumnFamilies(
      const ColumnFamilyOptions& cf_options,
      const std::vector<std::string>& column_family_names,
      std::vector<ColumnFamilyHandle*>* handles) override {
    // TODO: plumb Env::IOActivity, Env::IOPriority
    return CreateColumnFamilies(ReadOptions(), WriteOptions(), cf_options,
                                column_family_names, handles);
  }
  virtual Status CreateColumnFamilies(
      const ReadOptions& read_options, const WriteOptions& write_options,
      const ColumnFamilyOptions& cf_options,
      const std::vector<std::string>& column_family_names,
      std::vector<ColumnFamilyHandle*>* handles);

  Status CreateColumnFamilies(
      const std::vector<ColumnFamilyDescriptor>& column_families,
      std::vector<ColumnFamilyHandle*>* handles) override {
    // TODO: plumb Env::IOActivity, Env::IOPriority
    return CreateColumnFamilies(ReadOptions(), WriteOptions(), column_families,
                                handles);
  }
  virtual Status CreateColumnFamilies(
      const ReadOptions& read_options, const WriteOptions& write_options,
      const std::vector<ColumnFamilyDescriptor>& column_families,
      std::vector<ColumnFamilyHandle*>* handles);
  Status DropColumnFamily(ColumnFamilyHandle* column_family) override;
  Status DropColumnFamilies(
      const std::vector<ColumnFamilyHandle*>& column_families) override;

  // Returns false if key doesn't exist in the database and true if it may.
  // If value_found is not passed in as null, then return the value if found in
  // memory. On return, if value was found, then value_found will be set to true
  // , otherwise false.
  using DB::KeyMayExist;
  bool KeyMayExist(const ReadOptions& options,
                   ColumnFamilyHandle* column_family, const Slice& key,
                   std::string* value, std::string* timestamp,
                   bool* value_found = nullptr) override;

  using DB::NewIterator;
  Iterator* NewIterator(const ReadOptions& _read_options,
                        ColumnFamilyHandle* column_family) override;
  Status NewIterators(const ReadOptions& _read_options,
                      const std::vector<ColumnFamilyHandle*>& column_families,
                      std::vector<Iterator*>* iterators) override;
  Status NewIterators(const std::vector<ReadOptions>& read_options,
                      const std::vector<ColumnFamilyHandle*>& column_families,
                      std::vector<Iterator*>* iterators) override;

  using DB::NewMultiScan;
  std::unique_ptr<MultiScan> NewMultiScan(
      const ReadOptions& _read_options, ColumnFamilyHandle* column_family,
      const MultiScanArgs& scan_opts) override;

  const Snapshot* GetSnapshot() override;
  void ReleaseSnapshot(const Snapshot* snapshot) override;

  std::unique_ptr<Iterator> NewCoalescingIterator(
      const ReadOptions& options,
      const std::vector<ColumnFamilyHandle*>& column_families) override;

  std::unique_ptr<AttributeGroupIterator> NewAttributeGroupIterator(
      const ReadOptions& options,
      const std::vector<ColumnFamilyHandle*>& column_families) override;

  // Create a timestamped snapshot. This snapshot can be shared by multiple
  // readers. If any of them uses it for write conflict checking, then
  // is_write_conflict_boundary is true. For simplicity, set it to true by
  // default.
  std::pair<Status, std::shared_ptr<const Snapshot>> CreateTimestampedSnapshot(
      SequenceNumber snapshot_seq, uint64_t ts);
  std::shared_ptr<const SnapshotImpl> GetTimestampedSnapshot(uint64_t ts) const;
  void ReleaseTimestampedSnapshotsOlderThan(
      uint64_t ts, size_t* remaining_total_ss = nullptr);
  Status GetTimestampedSnapshots(uint64_t ts_lb, uint64_t ts_ub,
                                 std::vector<std::shared_ptr<const Snapshot>>&
                                     timestamped_snapshots) const;

  using DB::GetProperty;
  bool GetProperty(ColumnFamilyHandle* column_family, const Slice& property,
                   std::string* value) override;
  using DB::GetMapProperty;
  bool GetMapProperty(ColumnFamilyHandle* column_family, const Slice& property,
                      std::map<std::string, std::string>* value) override;
  using DB::GetIntProperty;
  bool GetIntProperty(ColumnFamilyHandle* column_family, const Slice& property,
                      uint64_t* value) override;
  using DB::GetAggregatedIntProperty;
  bool GetAggregatedIntProperty(const Slice& property,
                                uint64_t* aggregated_value) override;
  using DB::GetApproximateSizes;
  Status GetApproximateSizes(const SizeApproximationOptions& options,
                             ColumnFamilyHandle* column_family,
                             const Range* range, int n,
                             uint64_t* sizes) override;
  using DB::GetApproximateMemTableStats;
  void GetApproximateMemTableStats(ColumnFamilyHandle* column_family,
                                   const Range& range, uint64_t* const count,
                                   uint64_t* const size) override;
  using DB::CompactRange;
  Status CompactRange(const CompactRangeOptions& options,
                      ColumnFamilyHandle* column_family, const Slice* begin,
                      const Slice* end) override;

  using DB::CompactFiles;
  Status CompactFiles(
      const CompactionOptions& compact_options,
      ColumnFamilyHandle* column_family,
      const std::vector<std::string>& input_file_names, const int output_level,
      const int output_path_id = -1,
      std::vector<std::string>* const output_file_names = nullptr,
      CompactionJobInfo* compaction_job_info = nullptr) override;

  Status PauseBackgroundWork() override;
  Status ContinueBackgroundWork() override;

  Status EnableAutoCompaction(
      const std::vector<ColumnFamilyHandle*>& column_family_handles) override;

  void EnableManualCompaction() override;
  void DisableManualCompaction() override;
  void AbortAllCompactions() override;
  void ResumeAllCompactions() override;
  void AbortCompactions(ColumnFamilyHandle* column_family) override;
  void ResumeCompactions(ColumnFamilyHandle* column_family) override;

  using DB::SetOptions;
  Status SetOptions(
      const std::unordered_map<ColumnFamilyHandle*,
                               std::unordered_map<std::string, std::string>>&
          column_families_opts_map) override;

  Status SetDBOptions(
      const std::unordered_map<std::string, std::string>& options_map) override;

  using DB::NumberLevels;
  int NumberLevels(ColumnFamilyHandle* column_family) override;
  using DB::Level0StopWriteTrigger;
  int Level0StopWriteTrigger(ColumnFamilyHandle* column_family) override;
  const std::string& GetName() const override;
  Env* GetEnv() const override;
  FileSystem* GetFileSystem() const override;
  using DB::GetOptions;
  Options GetOptions(ColumnFamilyHandle* column_family) const override;
  using DB::GetDBOptions;
  DBOptions GetDBOptions() const override;
  using DB::Flush;
  Status Flush(const FlushOptions& options,
               ColumnFamilyHandle* column_family) override;
  Status Flush(
      const FlushOptions& options,
      const std::vector<ColumnFamilyHandle*>& column_families) override;
  Status FlushWAL(bool sync) override {
    FlushWALOptions options;
    options.sync = sync;
    return FlushWAL(options);
  }

  Status FlushWAL(const FlushWALOptions& options) override;

  virtual Status FlushWAL(const WriteOptions& write_options, bool sync);
  bool WALBufferIsEmpty();
  Status SyncWAL() override;
  Status LockWAL() override;
  Status UnlockWAL() override;

  SequenceNumber GetLatestSequenceNumber() const override;

  // IncreaseFullHistoryTsLow(ColumnFamilyHandle*, std::string) will acquire
  // and release db_mutex
  Status IncreaseFullHistoryTsLow(ColumnFamilyHandle* column_family,
                                  std::string ts_low) override;

  // GetFullHistoryTsLow(ColumnFamilyHandle*, std::string*) will acquire and
  // release db_mutex
  Status GetFullHistoryTsLow(ColumnFamilyHandle* column_family,
                             std::string* ts_low) override;

  Status GetNewestUserDefinedTimestamp(ColumnFamilyHandle* column_family,
                                       std::string* newest_timestamp) override;

  Status GetDbIdentity(std::string& identity) const override;

  virtual Status GetDbIdentityFromIdentityFile(const IOOptions& opts,
                                               std::string* identity) const;

  Status GetDbSessionId(std::string& session_id) const override;

  ColumnFamilyHandle* DefaultColumnFamily() const override;

  ColumnFamilyHandle* PersistentStatsColumnFamily() const;

  Status Close() override;

  Status DisableFileDeletions() override;

  Status EnableFileDeletions() override;

  virtual bool IsFileDeletionsEnabled() const;

  Status GetStatsHistory(
      uint64_t start_time, uint64_t end_time,
      std::unique_ptr<StatsHistoryIterator>* stats_iterator) override;

  using DB::ResetStats;
  Status ResetStats() override;
  // All the returned filenames start with "/"
  Status GetLiveFiles(std::vector<std::string>&, uint64_t* manifest_file_size,
                      bool flush_memtable = true) override;
  Status GetSortedWalFiles(VectorWalPtr& files) override;
  Status GetSortedWalFilesImpl(VectorWalPtr& files, bool need_seqnos);

  // Get the known flushed sizes of WALs that might still be written to
  // or have pending sync.
  // NOTE: unlike the live-file deques, this function includes WALs that might
  // be obsolete (but not obsolete to a pending Checkpoint) and not yet fully
  // synced.
  Status GetOpenWalSizes(std::map<uint64_t, uint64_t>& number_to_size);
  Status GetCurrentWalFile(std::unique_ptr<WalFile>* current_wal_file) override;
  Status GetCreationTimeOfOldestFile(uint64_t* creation_time) override;

  Status GetUpdatesSince(SequenceNumber seq_number,
                         std::unique_ptr<WalIterator>* iter,
                         const WalIterator::ReadOptions& read_options =
                             WalIterator::ReadOptions()) override;
  Status DeleteFilesInRanges(ColumnFamilyHandle* column_family,
                             const RangeOpt* ranges, size_t n,
                             bool include_end = true);

  void GetLiveFilesMetaData(std::vector<LiveFileMetaData>* metadata) override;

  Status GetLiveFilesChecksumInfo(FileChecksumList* checksum_list) override;

  Status GetLiveFilesStorageInfo(
      const LiveFilesStorageInfoOptions& opts,
      std::vector<LiveFileStorageInfo>* files) override;

  // Variant of GetLiveFilesStorageInfo used by subset-CF Checkpoint. Only
  // table/blob files of the given column families are captured; the rest are
  // reported in excluded_cf_ids so the caller can reconcile the copied
  // MANIFEST (typically via AppendColumnFamilyDropsToManifest below). Caller
  // must supply a non-empty include_cf_ids containing the default CF id and a
  // non-null excluded_cf_ids; both are internal-contract preconditions.
  Status GetLiveFilesStorageInfoForSubsetCheckpoint(
      const LiveFilesStorageInfoOptions& opts,
      const std::vector<uint32_t>& include_cf_ids,
      std::vector<LiveFileStorageInfo>* files,
      std::vector<uint32_t>* excluded_cf_ids,
      SubsetCheckpointOptionsSnapshot* options_snapshot);

  // Shared body of GetLiveFilesStorageInfo and its subset-checkpoint variant.
  // include_cf_ids empty -> no filter; otherwise records skipped CF ids in
  // *excluded_cf_ids (which must be non-null in that case).
  Status GetLiveFilesStorageInfoImpl(
      const LiveFilesStorageInfoOptions& opts,
      const std::vector<uint32_t>& include_cf_ids,
      std::vector<LiveFileStorageInfo>* files,
      std::vector<uint32_t>* excluded_cf_ids,
      SubsetCheckpointOptionsSnapshot* options_snapshot);

  // Appends kColumnFamilyDrop records to the MANIFEST file at manifest_path for
  // the given column family ids. Used by Checkpoint to make a subset-CF
  // checkpoint's copied MANIFEST consistent with the reduced file set. Operates
  // only on the given file; does not mutate this DB's live state.
  Status AppendColumnFamilyDropsToManifest(
      const std::string& manifest_path, uint64_t manifest_size,
      const std::vector<uint32_t>& cf_ids,
      uint64_t committed_options_file_number);

  // Writes a previously captured subset-specific OPTIONS snapshot into the
  // checkpoint. A zero file number is a no-op for a legacy source DB.
  Status CreateOptionsFileForSubsetCheckpoint(
      const std::string& checkpoint_dir,
      const SubsetCheckpointOptionsSnapshot& options_snapshot);

  Status GetPreparedFileInfoForExternalSstIngestion(
      const std::string& file_path,
      std::shared_ptr<const PreparedFileInfo>* file_info) override;

  // Obtains the meta data of the specified column family of the DB.
  // TODO(yhchiang): output parameter is placed in the end in this codebase.
  void GetColumnFamilyMetaData(ColumnFamilyHandle* column_family,
                               ColumnFamilyMetaData* metadata) override;

  // Get column family metadata with filtering based on key range and level
  void GetColumnFamilyMetaData(ColumnFamilyHandle* column_family,
                               const GetColumnFamilyMetaDataOptions& options,
                               ColumnFamilyMetaData* metadata) override;

  void GetAllColumnFamilyMetaData(
      std::vector<ColumnFamilyMetaData>* metadata) override;

  Status SuggestCompactRange(ColumnFamilyHandle* column_family,
                             const Slice* begin, const Slice* end) override;

  Status PromoteL0(ColumnFamilyHandle* column_family,
                   int target_level) override;

  using DB::IngestExternalFile;
  Status IngestExternalFile(
      ColumnFamilyHandle* column_family,
      const std::vector<std::string>& external_files,
      const IngestExternalFileOptions& ingestion_options) override;

  using DB::IngestExternalFiles;
  Status IngestExternalFiles(
      const std::vector<IngestExternalFileArg>& args) override;

  using DB::PrepareFileIngestion;
  Status PrepareFileIngestion(
      const std::vector<IngestExternalFileArg>& args,
      std::unique_ptr<FileIngestionHandle>* handle) override;

  Status CommitFileIngestionHandles(
      std::vector<std::unique_ptr<FileIngestionHandle>> handles) override;

  using DB::CreateColumnFamilyWithImport;
  Status CreateColumnFamilyWithImport(
      const ColumnFamilyOptions& options, const std::string& column_family_name,
      const ImportColumnFamilyOptions& import_options,
      const std::vector<const ExportImportFilesMetaData*>& metadatas,
      ColumnFamilyHandle** handle) override;

  using DB::ClipColumnFamily;
  Status ClipColumnFamily(ColumnFamilyHandle* column_family,
                          const Slice& begin_key,
                          const Slice& end_key) override;

  using DB::VerifyFileChecksums;
  Status VerifyFileChecksums(const ReadOptions& read_options) override;

  using DB::VerifyChecksum;
  Status VerifyChecksum(const ReadOptions& /*read_options*/) override;
  // Verify the checksums of files in db. Currently only tables are checked.
  //
  // read_options: controls file I/O behavior, e.g. read ahead size while
  //               reading all the live table files.
  //
  // use_file_checksum: if false, verify the block checksums of all live table
  //                    in db. Otherwise, obtain the file checksums and compare
  //                    with the MANIFEST. Currently, file checksums are
  //                    recomputed by reading all table files.
  //
  // Returns: OK if there is no file whose file or block checksum mismatches.
  Status VerifyChecksumInternal(const ReadOptions& read_options,
                                bool use_file_checksum);

  Status VerifyFullFileChecksum(const std::string& file_checksum_expected,
                                const std::string& func_name_expected,
                                const std::string& fpath,
                                const ReadOptions& read_options);

  using DB::StartTrace;
  Status StartTrace(const TraceOptions& options,
                    std::unique_ptr<TraceWriter>&& trace_writer) override;

  using DB::EndTrace;
  Status EndTrace() override;

  using DB::NewDefaultReplayer;
  Status NewDefaultReplayer(const std::vector<ColumnFamilyHandle*>& handles,
                            std::unique_ptr<TraceReader>&& reader,
                            std::unique_ptr<Replayer>* replayer) override;

  using DB::StartBlockCacheTrace;
  Status StartBlockCacheTrace(
      const TraceOptions& trace_options,
      std::unique_ptr<TraceWriter>&& trace_writer) override;

  Status StartBlockCacheTrace(
      const BlockCacheTraceOptions& options,
      std::unique_ptr<BlockCacheTraceWriter>&& trace_writer) override;

  using DB::EndBlockCacheTrace;
  Status EndBlockCacheTrace() override;

  using DB::StartIOTrace;
  Status StartIOTrace(const TraceOptions& options,
                      std::unique_ptr<TraceWriter>&& trace_writer) override;

  using DB::EndIOTrace;
  Status EndIOTrace() override;

  using DB::GetPropertiesOfAllTables;
  Status GetPropertiesOfAllTables(ColumnFamilyHandle* column_family,
                                  TablePropertiesCollection* props) override;
  Status GetPropertiesOfTablesInRange(
      ColumnFamilyHandle* column_family, const Range* range, std::size_t n,
      TablePropertiesCollection* props) override;

  Status GetPropertiesOfTablesByLevel(
      ColumnFamilyHandle* column_family,
      std::vector<std::unique_ptr<TablePropertiesCollection>>* props_by_level)
      override;

  // ---- End of implementations of the DB interface ----
  SystemClock* GetSystemClock() const;

  struct GetImplOptions {
    ColumnFamilyHandle* column_family = nullptr;
    PinnableSlice* value = nullptr;
    PinnableWideColumns* columns = nullptr;
    std::string* timestamp = nullptr;
    bool* newer_version_present = nullptr;
    bool* value_found = nullptr;
    ReadCallback* callback = nullptr;
    bool* is_blob_index = nullptr;
    // If true return value associated with key via value pointer else return
    // all merge operands for key via merge_operands pointer
    bool get_value = true;
    // Pointer to an array of size
    // get_merge_operands_options.expected_max_number_of_operands allocated by
    // user
    PinnableSlice* merge_operands = nullptr;
    GetMergeOperandsOptions* get_merge_operands_options = nullptr;
    int* number_of_operands = nullptr;

    // Lazy wide-column mode (used by GetEntityLazy / MultiGetEntityLazy). When
    // `lazy_columns_pin` is non-null, the SST point-lookup path leaves a
    // wide-column entity's blob references *unresolved* (see
    // GetContext::SaveWideColumnEntityToColumns), so the caller can resolve
    // them on demand later. On a successful lookup GetImpl then:
    //   - sets `*lazy_columns_version` to the Version the entity was read from
    //     (the resolution context for blob references), and
    //   - transfers an extra SuperVersion reference by registering a cleanup on
    //     `*lazy_columns_pin`, so the result may outlive this call (as an
    //     iterator's result does).
    // The SameFileBlobReader for the SST that held the entity (if any) is
    // stored in `*lazy_columns_same_file_reader` by the read path, so same-file
    // ("embedded") references stay resolvable later. All three must be set
    // together. `columns` must also be set (the entity output).
    Cleanable* lazy_columns_pin = nullptr;
    const Version** lazy_columns_version = nullptr;
    const SameFileBlobReader** lazy_columns_same_file_reader = nullptr;

    // Batched lazy read (MultiGetEntityLazy). When `lazy_columns_shared_sv` is
    // non-null, GetImpl reads against this externally-owned SuperVersion at
    // `lazy_columns_snapshot_seq` (the consistent per-batch sequence number)
    // instead of acquiring its own SuperVersion and deriving its own sequence
    // number, and it does NOT transfer a per-key pin -- the batch holds one
    // shared SuperVersion pin per column family. `lazy_columns_pin` must then
    // be null; `lazy_columns_version` / `lazy_columns_same_file_reader` are
    // still filled. (Single-key GetEntityLazy leaves this null and uses
    // `lazy_columns_pin`.)
    SuperVersion* lazy_columns_shared_sv = nullptr;
    SequenceNumber lazy_columns_snapshot_seq = kMaxSequenceNumber;
  };

  DECLARE_SYNC_AND_ASYNC(Status, GetImpl, const ReadOptions& read_options,
                         ColumnFamilyHandle* column_family, const Slice& key,
                         PinnableSlice* value);

  DECLARE_SYNC_AND_ASYNC(Status, GetImpl, const ReadOptions& read_options,
                         ColumnFamilyHandle* column_family, const Slice& key,
                         PinnableSlice* value, std::string* timestamp);

  Status GetImpl(const ReadOptions& read_options,
                 ColumnFamilyHandle* column_family, const Slice& key,
                 PinnableSlice* value, std::string* timestamp,
                 bool* newer_version_present);

  // Function that Get and KeyMayExist call with no_io true or false
  // Note: 'value_found' from KeyMayExist propagates here
  // This function is also called by GetMergeOperands
  // If get_impl_options.get_value = true get value associated with
  // get_impl_options.key via get_impl_options.value
  // If get_impl_options.get_value = false get merge operands associated with
  // get_impl_options.key via get_impl_options.merge_operands
  DECLARE_SYNC_AND_ASYNC_VIRTUAL(Status, GetImpl, const ReadOptions& options,
                                 const Slice& key,
                                 GetImplOptions& get_impl_options);

  // Shared implementation of GetEntityLazy / MultiGetEntityLazy for a single
  // key: performs a lazy point lookup for `key` (leaving blob references
  // unresolved) into `*result`, which takes ownership of a SuperVersion pin so
  // it may outlive this call. `read_options` must already be finalized (e.g.
  // io_activity set). Returns OK on success, NotFound (with an empty `*result`)
  // if there is no entry, or another non-OK status on error. REQUIRES:
  // max_open_files == -1 has already been validated by the caller.
  Status GetEntityLazyImpl(const ReadOptions& read_options,
                           ColumnFamilyHandle* column_family, const Slice& key,
                           LazyWideColumns* result);

  // Batched-read counterpart of GetEntityLazyImpl for one key of
  // MultiGetEntityLazy: performs the lazy point lookup against the batch's
  // shared `super_version` at the batch's consistent `snapshot_seq` (rather
  // than acquiring its own), and binds the entity's resolver to that Version
  // without taking a per-key pin (the batch owns one shared pin per column
  // family).
  Status GetEntityLazyForBatch(const ReadOptions& read_options,
                               ColumnFamilyHandle* column_family,
                               SuperVersion* super_version,
                               SequenceNumber snapshot_seq, const Slice& key,
                               LazyWideColumns* result);

  // If `snapshot` == kMaxSequenceNumber, set a recent one inside the file.
  ArenaWrappedDBIter* NewIteratorImpl(const ReadOptions& options,
                                      ColumnFamilyHandleImpl* cfh,
                                      SuperVersion* sv, SequenceNumber snapshot,
                                      ReadCallback* read_callback,
                                      bool expose_blob_index = false,
                                      bool allow_refresh = true);

  virtual SequenceNumber GetLastPublishedSequence() const {
    if (last_seq_same_as_publish_seq_) {
      return versions_->LastSequence();
    } else {
      return versions_->LastPublishedSequence();
    }
  }

  // Durability frontier: every sequence at or below it completed its WAL
  // append's write(). Computed on the I/O-completion path. With more than one
  // WAL partition the completion gate waits on it, and LastSequence() never
  // passes it for a sequence with a WAL record (PublishAppliedPrefix).
  // Separate from last_published_sequence_, which ingestion and recovery jump
  // forward and which must never overstate this lower bound.
  SequenceNumber WalDurabilitySafePoint() const {
    return wal_durability_tracker_->SafePoint();
  }

  // One WAL append attempt, published via the
  // DBImpl::WriteToWAL:AppendOutcome sync point after AddRecord returns
  // (and after the AfterAddRecord hook may have replaced its status, so
  // `ok` is the outcome the write path acts on). Tests record these to
  // prove a failed write was attempted exactly once, never moved between
  // partitions, and that no acknowledged sequence lies above a failed hole,
  // and to prove which WAL file an append went to.
  struct WalAppendOutcome {
    size_t wal_partition = 0;
    SequenceNumber first_sequence = 0;
    size_t sequence_count = 0;
    bool ok = true;
    uint64_t wal_number = 0;
  };

  // REQUIRES: joined the main write queue if two_write_queues is disabled, and
  // the second write queue otherwise. With more than one WAL partition
  // (two_write_queues is rejected there) nothing calls it: write groups
  // publish only through PublishLastSequence, bounded by the applied prefix.
  virtual void SetLastPublishedSequence(SequenceNumber seq);
  // Returns LastSequence in last_seq_same_as_publish_seq_
  // mode and LastAllocatedSequence otherwise. This is useful when visiblility
  // depends also on data written to the WAL but not to the memtable.
  SequenceNumber TEST_GetLastVisibleSequence() const;

  // Similar to Write() but will call the callback once on the single write
  // thread to determine whether it is safe to perform the write.
  virtual Status WriteWithCallback(const WriteOptions& write_options,
                                   WriteBatch* my_batch,
                                   WriteCallback* callback,
                                   UserWriteCallback* user_write_cb = nullptr);

  // Returns the sequence number that is guaranteed to be smaller than or equal
  // to the sequence number of any key that could be inserted into the current
  // memtables. It can then be assumed that any write with a larger(or equal)
  // sequence number will be present in this memtable or a later memtable.
  //
  // If the earliest sequence number could not be determined,
  // kMaxSequenceNumber will be returned.
  //
  // If include_history=true, will also search Memtables in MemTableList
  // History.
  SequenceNumber GetEarliestMemTableSequenceNumber(SuperVersion* sv,
                                                   bool include_history);

  // For a given key, check to see if there are any records for this key
  // in the memtables, including memtable history.  If cache_only is false,
  // SST files will also be checked.
  //
  // `key` should NOT have user-defined timestamp appended to user key even if
  // timestamp is enabled.
  //
  // If a key is found, *found_record_for_key will be set to true and
  // *seq will be set to the stored sequence number for the latest
  // operation on this key or kMaxSequenceNumber if unknown. If user-defined
  // timestamp is enabled for this column family and timestamp is not nullptr,
  // then *timestamp will be set to the stored timestamp for the latest
  // operation on this key.
  // If no key is found, *found_record_for_key will be set to false.
  //
  // Note: If cache_only=false, it is possible for *seq to be set to 0 if
  // the sequence number has been cleared from the record.  If the caller is
  // holding an active db snapshot, we know the missing sequence must be less
  // than the snapshot's sequence number (sequence numbers are only cleared
  // when there are no earlier active snapshots).
  //
  // If NotFound is returned and found_record_for_key is set to false, then no
  // record for this key was found.  If the caller is holding an active db
  // snapshot, we know that no key could have existing after this snapshot
  // (since we do not compact keys that have an earlier snapshot).
  //
  // Only records newer than or at `lower_bound_seq` are guaranteed to be
  // returned. Memtables and files may not be checked if it only contains data
  // older than `lower_bound_seq`.
  //
  // Returns OK or NotFound on success,
  // other status on unexpected error.
  // TODO(andrewkr): this API need to be aware of range deletion operations
  Status GetLatestSequenceForKey(SuperVersion* sv, const Slice& key,
                                 bool cache_only,
                                 SequenceNumber lower_bound_seq,
                                 SequenceNumber* seq, std::string* timestamp,
                                 bool* found_record_for_key,
                                 bool* is_blob_index);

  Status TraceIteratorSeek(const uint32_t& cf_id, const Slice& key,
                           const Slice& lower_bound, const Slice upper_bound);
  Status TraceIteratorSeekForPrev(const uint32_t& cf_id, const Slice& key,
                                  const Slice& lower_bound,
                                  const Slice upper_bound);

  // Similar to GetSnapshot(), but also lets the db know that this snapshot
  // will be used for transaction write-conflict checking.  The DB can then
  // make sure not to compact any keys that would prevent a write-conflict from
  // being detected.
  const Snapshot* GetSnapshotForWriteConflictBoundary();

  // max_file_num_to_ignore allows bottom level compaction to filter out newly
  // compacted SST files. Setting max_file_num_to_ignore to kMaxUint64 will
  // disable the filtering
  // If `final_output_level` is not nullptr, it is set to manual compaction's
  // output level if returned status is OK, and it may or may not be set to
  // manual compaction's output level if returned status is not OK.
  Status RunManualCompaction(ColumnFamilyData* cfd, int input_level,
                             int output_level,
                             const CompactRangeOptions& compact_range_options,
                             const Slice* begin, const Slice* end,
                             bool exclusive, bool disallow_trivial_move,
                             uint64_t max_file_num_to_ignore,
                             const std::string& trim_ts,
                             int* final_output_level = nullptr);

  // Return an internal iterator over the current state of the database.
  // The keys of this iterator are internal keys (see format.h).
  // The returned iterator should be deleted when no longer needed.
  // If allow_unprepared_value is true, the returned iterator may defer reading
  // the value and so will require PrepareValue() to be called before value();
  // allow_unprepared_value = false is convenient when this optimization is not
  // useful, e.g. when reading the whole column family.
  //
  // read_options.ignore_range_deletions determines whether range tombstones are
  // processed in the returned interator internally, i.e., whether range
  // tombstone covered keys are in this iterator's output.
  // @param read_options Must outlive the returned iterator.
  InternalIterator* NewInternalIterator(
      const ReadOptions& read_options, Arena* arena, SequenceNumber sequence,
      ColumnFamilyHandle* column_family = nullptr,
      bool allow_unprepared_value = false);

  // Note: to support DB iterator refresh, memtable range tombstones in the
  // underlying merging iterator needs to be refreshed. If db_iter is not
  // nullptr, db_iter->SetMemtableRangetombstoneIter() is called with the
  // memtable range tombstone iterator used by the underlying merging iterator.
  // This range tombstone iterator can be refreshed later by db_iter.
  // @param read_options Must outlive the returned iterator.
  // @param sequence The snapshot sequence captured when the DB iterator was
  // created. Child iterators must use this instead of dereferencing
  // read_options.snapshot, which may be released before lazy initialization.
  // @param scan_opts Optional bounded scan ranges used only to prune the
  // iterator tree during lazy Prepare() initialization.
  InternalIterator* NewInternalIterator(
      const ReadOptions& read_options, ColumnFamilyData* cfd,
      SuperVersion* super_version, Arena* arena, SequenceNumber sequence,
      bool allow_unprepared_value, ArenaWrappedDBIter* db_iter = nullptr,
      const MultiScanArgs* scan_opts = nullptr);

  // Release a SuperVersion held by an iterator. This preserves the cleanup
  // behavior used by materialized internal iterators even when the DB iterator
  // never needed to lazily build its child iterator tree.
  void CleanupIteratorSuperVersion(SuperVersion* super_version,
                                   bool background_purge);

  // Transfer an extra reference to `super_version` into `pin` (via a cleanup
  // callback), so `pin`'s owner keeps the SuperVersion -- and thus the data it
  // references -- alive after the current call returns, as an iterator does.
  // Shared by the lazy wide-column read path (GetEntityLazy) across the
  // primary, read-only, and secondary GetImpl overrides.
  void TransferSuperVersionPin(SuperVersion* super_version, Cleanable* pin);

  LogsWithPrepTracker* logs_with_prep_tracker() {
    return &logs_with_prep_tracker_;
  }

  struct BGJobLimits {
    int max_flushes;
    int max_compactions;
  };
  // Returns maximum background flushes and compactions allowed to be scheduled
  BGJobLimits GetBGJobLimits() const;
  // Need a static version that can be called during SanitizeOptions().
  static BGJobLimits GetBGJobLimits(int max_background_flushes,
                                    int max_background_compactions,
                                    int max_background_jobs,
                                    bool parallelize_compactions);

  // move logs pending closing from job_context to the DB queue and
  // schedule a purge
  void ScheduleBgLogWriterClose(JobContext* job_context);

  uint64_t MinLogNumberToKeep();

  uint64_t MinLogNumberToRecycle();

  // Returns the lower bound file number for SSTs that won't be deleted, even if
  // they're obsolete. This lower bound is used internally to prevent newly
  // created flush/compaction output files from being deleted before they're
  // installed. This technique avoids the need for tracking the exact numbers of
  // files pending creation, although it prevents more files than necessary from
  // being deleted.
  uint64_t MinObsoleteSstNumberToKeep();

  uint64_t GetObsoleteSstFilesSize();

  uint64_t MinOptionsFileNumberToKeep();

  // Returns the list of live files in 'live' and the list
  // of all files in the filesystem in 'candidate_files'.
  // If force == false and the last call was less than
  // db_options_.delete_obsolete_files_period_micros microseconds ago,
  // it will not fill up the job_context
  void FindObsoleteFiles(JobContext* job_context, bool force,
                         bool no_full_scan = false);

  // Diffs the files listed in filenames and those that do not
  // belong to live files are possibly removed. Also, removes all the
  // files in sst_delete_files and log_delete_files.
  // It is not necessary to hold the mutex when invoking this method.
  // If FindObsoleteFiles() was run, we need to also run
  // PurgeObsoleteFiles(), even if disable_delete_obsolete_files_ is true
  void PurgeObsoleteFiles(JobContext& background_contet,
                          bool schedule_only = false);

  // Schedule a background job to actually delete obsolete files.
  void SchedulePurge();

  const SnapshotList& snapshots() const { return snapshots_; }

  // load list of snapshots to `snap_vector` that is no newer than `max_seq`
  // in ascending order.
  // `oldest_write_conflict_snapshot` is filled with the oldest snapshot
  // which satisfies SnapshotImpl.is_write_conflict_boundary_ = true.
  void LoadSnapshots(std::vector<SequenceNumber>* snap_vector,
                     SequenceNumber* oldest_write_conflict_snapshot,
                     const SequenceNumber& max_seq) const {
    InstrumentedMutexLock l(mutex());
    snapshots().GetAll(snap_vector, oldest_write_conflict_snapshot, max_seq);
  }

  const ImmutableDBOptions& immutable_db_options() const {
    return immutable_db_options_;
  }

  // Cancel all background jobs, including flush, compaction, background
  // purging, stats dumping threads, etc. If `wait` = true, wait for the
  // running jobs to abort or finish before returning. Otherwise, only
  // sends the signals.
  void CancelAllBackgroundWork(bool wait);

  // Find Super version and reference it. Based on options, it might return
  // the thread local cached one.
  // Call ReturnAndCleanupSuperVersion() when it is no longer needed.
  SuperVersion* GetAndRefSuperVersion(ColumnFamilyData* cfd);

  // Similar to the previous function but looks up based on a column family id.
  // nullptr will be returned if this column family no longer exists.
  // REQUIRED: this function should only be called on the write thread or if the
  // mutex is held.
  SuperVersion* GetAndRefSuperVersion(uint32_t column_family_id);

  // Un-reference the super version and clean it up if it is the last reference.
  void CleanupSuperVersion(SuperVersion* sv);

  // Un-reference the super version and return it to thread local cache if
  // needed. If it is the last reference of the super version. Clean it up
  // after un-referencing it.
  void ReturnAndCleanupSuperVersion(ColumnFamilyData* cfd, SuperVersion* sv);

  // Similar to the previous function but looks up based on a column family id.
  // nullptr will be returned if this column family no longer exists.
  // REQUIRED: this function should only be called on the write thread.
  void ReturnAndCleanupSuperVersion(uint32_t colun_family_id, SuperVersion* sv);

  // REQUIRED: this function should only be called on the write thread or if the
  // mutex is held.  Return value only valid until next call to this function or
  // mutex is released.
  ColumnFamilyHandle* GetColumnFamilyHandle(uint32_t column_family_id);

  // Same as above, should called without mutex held and not on write thread.
  std::unique_ptr<ColumnFamilyHandle> GetColumnFamilyHandleUnlocked(
      uint32_t column_family_id);

  // Returns the number of currently running flushes.
  // REQUIREMENT: mutex_ must be held when calling this function.
  int num_running_flushes() {
    mutex_.AssertHeld();
    return num_running_flushes_;
  }

  const WriteController& write_controller() { return write_controller_; }

  // hollow transactions shell used for recovery.
  // these will then be passed to TransactionDB so that
  // locks can be reacquired before writing can resume.
  struct RecoveredTransaction {
    std::string name_;
    bool unprepared_;

    struct BatchInfo {
      uint64_t log_number_;
      // TODO(lth): For unprepared, the memory usage here can be big for
      // unprepared transactions. This is only useful for rollbacks, and we
      // can in theory just keep keyset for that.
      WriteBatch* batch_;
      // Number of sub-batches. A new sub-batch is created if txn attempts to
      // insert a duplicate key,seq to memtable. This is currently used in
      // WritePreparedTxn/WriteUnpreparedTxn.
      size_t batch_cnt_;
    };

    // This maps the seq of the first key in the batch to BatchInfo, which
    // contains WriteBatch and other information relevant to the batch.
    //
    // For WriteUnprepared, batches_ can have size greater than 1, but for
    // other write policies, it must be of size 1.
    std::map<SequenceNumber, BatchInfo> batches_;

    explicit RecoveredTransaction(const uint64_t log, const std::string& name,
                                  WriteBatch* batch, SequenceNumber seq,
                                  size_t batch_cnt, bool unprepared)
        : name_(name), unprepared_(unprepared) {
      batches_[seq] = {log, batch, batch_cnt};
    }

    // No copying allowed
    RecoveredTransaction(const RecoveredTransaction&) = delete;
    RecoveredTransaction& operator=(const RecoveredTransaction&) = delete;
    RecoveredTransaction(RecoveredTransaction&&) = delete;
    RecoveredTransaction& operator=(RecoveredTransaction&&) = delete;

    ~RecoveredTransaction() {
      for (auto& it : batches_) {
        delete it.second.batch_;
      }
    }

    void AddBatch(SequenceNumber seq, uint64_t log_number, WriteBatch* batch,
                  size_t batch_cnt, bool unprepared) {
      assert(batches_.count(seq) == 0);
      batches_[seq] = {log_number, batch, batch_cnt};
      // Prior state must be unprepared, since the prepare batch must be the
      // last batch.
      assert(unprepared_);
      unprepared_ = unprepared;
    }
  };

  using RecoveredTransactionMap =
      std::unordered_map<std::string, RecoveredTransaction*>;

  bool allow_2pc() const { return immutable_db_options_.allow_2pc; }

  RecoveredTransactionMap recovered_transactions() {
    return recovered_transactions_;
  }

  RecoveredTransaction* GetRecoveredTransaction(const std::string& name) {
    auto it = recovered_transactions_.find(name);
    if (it == recovered_transactions_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  void InsertRecoveredTransaction(const uint64_t log, const std::string& name,
                                  WriteBatch* batch, SequenceNumber seq,
                                  size_t batch_cnt, bool unprepared_batch) {
    // For WriteUnpreparedTxn, InsertRecoveredTransaction is called multiple
    // times for every unprepared batch encountered during recovery.
    //
    // If the transaction is prepared, then the last call to
    // InsertRecoveredTransaction will have unprepared_batch = false.
    auto rtxn = recovered_transactions_.find(name);
    if (rtxn == recovered_transactions_.end()) {
      recovered_transactions_[name] = new RecoveredTransaction(
          log, name, batch, seq, batch_cnt, unprepared_batch);
    } else {
      rtxn->second->AddBatch(seq, log, batch, batch_cnt, unprepared_batch);
    }
    logs_with_prep_tracker_.MarkLogAsContainingPrepSection(log);
  }

  // Deletes the recovered transaction `it` points to and returns the iterator
  // following it, like std::unordered_map::erase().
  RecoveredTransactionMap::iterator DeleteRecoveredTransaction(
      RecoveredTransactionMap::iterator it) {
    assert(it != recovered_transactions_.end());
    auto* trx = it->second;
    RecoveredTransactionMap::iterator next = recovered_transactions_.erase(it);
    for (const auto& info : trx->batches_) {
      logs_with_prep_tracker_.MarkLogAsHavingPrepSectionFlushed(
          info.second.log_number_);
    }
    delete trx;
    return next;
  }

  void DeleteRecoveredTransaction(const std::string& name) {
    RecoveredTransactionMap::iterator it = recovered_transactions_.find(name);
    DeleteRecoveredTransaction(it);
  }

  void DeleteAllRecoveredTransactions() {
    for (auto it = recovered_transactions_.begin();
         it != recovered_transactions_.end(); ++it) {
      delete it->second;
    }
    recovered_transactions_.clear();
  }

  void AddToLogsToFreeQueue(log::Writer* log_writer) {
    mutex_.AssertHeld();
    wals_to_free_queue_.push_back(log_writer);
  }

  void AddSuperVersionsToFreeQueue(SuperVersion* sv) {
    superversions_to_free_queue_.push_back(sv);
  }

  void EnableTrackPublishedSeqInSnapshotContext();
  void SetSnapshotChecker(SnapshotChecker* snapshot_checker);

  // Fill JobContext with snapshot information needed by flush and compaction.
  void InitSnapshotContext(JobContext* job_context);

  // Not thread-safe.
  void SetRecoverableStatePreReleaseCallback(PreReleaseCallback* callback);

  InstrumentedMutex* mutex() const { return &mutex_; }
  const std::atomic<uint64_t>* GetMutexHoldLogThreshold() const {
    return &mutex_hold_log_threshold_us_;
  }
  const std::shared_ptr<SlowMutexLoggingContext>& GetSlowMutexLogging() const {
    return slow_mutex_logging_;
  }

  // Initialize a brand new DB. The DB directory is expected to be empty before
  // calling it. Push new manifest file name into `new_filenames`.
  Status NewDB(std::vector<std::string>* new_filenames);

  // This is to be used only by internal rocksdb classes.
  static Status Open(const DBOptions& db_options, const std::string& name,
                     const std::vector<ColumnFamilyDescriptor>& column_families,
                     std::vector<ColumnFamilyHandle*>* handles,
                     std::unique_ptr<DB>* dbptr, const bool seq_per_batch,
                     const bool batch_per_txn, const bool is_retry,
                     bool* can_retry);

  static IOStatus CreateAndNewDirectory(
      FileSystem* fs, const std::string& dirname,
      std::unique_ptr<FSDirectory>* directory);

  // find stats map from stats_history_ with smallest timestamp in
  // the range of [start_time, end_time)
  bool FindStatsByTime(uint64_t start_time, uint64_t end_time,
                       uint64_t* new_time,
                       std::map<std::string, uint64_t>* stats_map);

  // Print information of all tombstones of all iterators to the std::string
  // This is only used by ldb. The output might be capped. Tombstones
  // printed out are not guaranteed to be in any order.
  Status TablesRangeTombstoneSummary(ColumnFamilyHandle* column_family,
                                     int max_entries_to_print,
                                     std::string* out_str);

  VersionSet* GetVersionSet() const { return versions_.get(); }

  Status WaitForCompact(
      const WaitForCompactOptions& wait_for_compact_options) override;

#ifndef NDEBUG
  // Compact any files in the named level that overlap [*begin, *end]
  Status TEST_CompactRange(int level, const Slice* begin, const Slice* end,
                           ColumnFamilyHandle* column_family = nullptr,
                           bool disallow_trivial_move = false);

  Status TEST_SwitchWAL();

  bool TEST_UnableToReleaseOldestLog() { return unable_to_release_oldest_log_; }

  bool TEST_IsLogGettingFlushed() {
    InstrumentedMutexLock l(&mutex_);
    return OldestAliveWal(0).getting_flushed;
  }

  Status TEST_SwitchMemtable(ColumnFamilyData* cfd = nullptr);

  size_t TEST_GetWBMFlushableMemUsage();
  bool TEST_IsWBMFlushInitiatorFlushable() const;

  // Force current memtable contents to be flushed.
  Status TEST_FlushMemTable(bool wait = true, bool allow_write_stall = false,
                            ColumnFamilyHandle* cfh = nullptr);

  Status TEST_FlushMemTable(ColumnFamilyData* cfd,
                            const FlushOptions& flush_opts);

  // Like TEST_FlushMemTable(bool wait, ...), but also sets
  // FlushOptions::listener_wait so the flush does not return until the
  // registered EventListener::OnFlushCompleted callbacks for the flushed
  // memtables have finished running.
  Status TEST_FlushMemTableWithListenerWait(bool allow_write_stall = false,
                                            ColumnFamilyHandle* cfh = nullptr);

  // Flush (multiple) ColumnFamilyData without using ColumnFamilyHandle. This
  // is because in certain cases, we can flush column families, wait for the
  // flush to complete, but delete the column family handle before the wait
  // finishes. For example in CompactRange.
  Status TEST_AtomicFlushMemTables(
      const autovector<ColumnFamilyData*>& provided_candidate_cfds,
      const FlushOptions& flush_opts, bool non_blocking_write_thread = false);

  void TEST_BeginWriteStall();
  void TEST_EndWriteStall();

  // Wait for background threads to complete scheduled work.
  Status TEST_WaitForBackgroundWork();

  // Wait for memtable compaction
  Status TEST_WaitForFlushMemTable(ColumnFamilyHandle* column_family = nullptr);

  Status TEST_WaitForCompact();
  Status TEST_WaitForCompact(
      const WaitForCompactOptions& wait_for_compact_options);

  // Wait for any background purge
  Status TEST_WaitForPurge();

  // Get the background error status
  Status TEST_GetBGError();

  void TEST_SetBGError(const IOStatus& error, BackgroundErrorReason reason);

  bool TEST_IsRecoveryInProgress();

  Status TEST_ResumeImpl(DBRecoverContext context);

  // Return the maximum overlapping data (in bytes) at next level for any
  // file at a level >= 1.
  uint64_t TEST_MaxNextLevelOverlappingBytes(
      ColumnFamilyHandle* column_family = nullptr);

  // Return the current manifest file no.
  uint64_t TEST_Current_Manifest_FileNo();

  // Returns the number that'll be assigned to the next file that's created.
  uint64_t TEST_Current_Next_FileNo();

  // get total level0 file size. Only for testing.
  uint64_t TEST_GetLevel0TotalSize();

  void TEST_GetFilesMetaData(
      ColumnFamilyHandle* column_family,
      std::vector<std::vector<FileMetaData>>* metadata,
      std::vector<std::shared_ptr<BlobFileMetaData>>* blob_metadata = nullptr);

  void TEST_LockMutex();

  void TEST_UnlockMutex();

  InstrumentedMutex* TEST_Mutex() { return &mutex_; }

  void TEST_SignalAllBgCv();

  // REQUIRES: mutex locked
  void* TEST_BeginWrite();

  // REQUIRES: mutex locked
  // pass the pointer that you got from TEST_BeginWrite()
  void TEST_EndWrite(void* w);

  uint64_t TEST_MaxTotalInMemoryState() const {
    return max_total_in_memory_state_;
  }

  size_t TEST_LogsToFreeSize();

  uint64_t TEST_LogfileNumber();

  uint64_t TEST_wals_total_size() const {
    return wals_total_size_.LoadRelaxed();
  }

  void TEST_GetAllBlockCaches(std::unordered_set<const Cache*>* cache_set);

  // Return the lastest MutableCFOptions of a column family
  Status TEST_GetLatestMutableCFOptions(ColumnFamilyHandle* column_family,
                                        MutableCFOptions* mutable_cf_options);

  Cache* TEST_table_cache() { return table_cache_.get(); }

  WriteController& TEST_write_controler() { return write_controller_; }

  uint64_t TEST_FindMinLogContainingOutstandingPrep();
  uint64_t TEST_FindMinPrepLogReferencedByMemTable();
  size_t TEST_PreparedSectionCompletedSize();
  size_t TEST_LogsWithPrepSize();

  int TEST_BGCompactionsAllowed() const;
  int TEST_BGFlushesAllowed() const;
  int TEST_NumRunningBottomCompactions() const;
  size_t TEST_GetWalPreallocateBlockSize(uint64_t write_buffer_size) const;
  void TEST_WaitForPeriodicTaskRun(std::function<void()> callback) const;
  SeqnoToTimeMapping TEST_GetSeqnoToTimeMapping() const;
  const autovector<uint64_t>& TEST_GetFilesToQuarantine() const;
  size_t TEST_EstimateInMemoryStatsHistorySize() const;

  // Same snapshot as TEST_CurrentWalNumber(0), kept for the N=1 tests that
  // predate partitions.
  uint64_t TEST_GetCurrentLogNumber() { return TEST_CurrentWalNumber(0); }

  size_t TEST_NumWalPartitions() const { return NumWalPartitions(); }

  // As WalPartitionForWrite, with deferred PutEntity ops given by column
  // family id (the only field routing reads). Null means no deferred ops.
  static Status TEST_WalPartitionForWrite(
      const WriteBatch& batch, size_t num_partitions, size_t* wal_partition,
      const std::vector<uint32_t>* deferred_cf_ids = nullptr);

  // A WAL partition's alive WALs in deque order, as {number, size} pairs, and
  // the number of the WAL its writer appends to.
  std::vector<std::pair<uint64_t, uint64_t>> TEST_AliveWalFiles(
      size_t partition);
  uint64_t TEST_CurrentWalNumber(size_t partition);
  // A WAL partition's open-WAL entries in deque order, as {number,
  // getting_synced} pairs.
  std::vector<std::pair<uint64_t, bool>> TEST_WalSyncMarks(size_t partition);

  // Partition p's oldest live-file entry as {number, getting_flushed}: what
  // OldestAliveWal returns, read mutex_-only like SwitchWAL does, for the
  // race test.
  std::pair<uint64_t, bool> TEST_OldestAliveWalStableFields(size_t partition);

  // Returns the reusable merge buffer for a WAL partition.
  const WriteBatch* TEST_GetWalMergeBuffer(size_t partition) {
    return &WalMergeBuffer(partition);
  }

  // Waiters currently parked in WaitForWalDurability. Tests use it to prove
  // a write is parked in its completion gate before injecting a failure.
  size_t TEST_WalDurabilityWaiters() const {
    return wal_durability_waiters_.load(std::memory_order_relaxed);
  }

  // Background-error leg of the durability park predicate. Tests use it to
  // prove the flag clears when the background error clears.
  bool TEST_WalDurabilityBGError() const {
    return wal_durability_bg_error_.load(std::memory_order_acquire);
  }

  // Every gate fence as {low, high} (high is kMaxSequenceNumber while the
  // fence is open). Tests use it to prove a write admitted by recovery
  // allocated above every closed fence.
  std::vector<std::pair<SequenceNumber, SequenceNumber>> TEST_WalGateFences() {
    InstrumentedMutexLock l(&sequence_wal_index_mutex_);
    std::vector<std::pair<SequenceNumber, SequenceNumber>> fences;
    for (const WalGateFence& fence : wal_gate_fences_) {
      fences.emplace_back(fence.low, fence.high);
    }
    return fences;
  }

  // Partition p's lowest registered, not yet completed sequence, or
  // WalDurabilityTracker::kIdle. Tests use it to observe a registration
  // directly, whether or not it is later completed.
  uint64_t TEST_WalDurabilityPendingLow(size_t partition) const {
    return wal_durability_tracker_->PendingLow(partition);
  }

  // Highest sequence partition p has completed. Tests use it to observe
  // that nothing was ever completed.
  uint64_t TEST_WalDurabilityDurableUpto(size_t partition) const {
    return wal_durability_tracker_->DurableUpto(partition);
  }

  // The inclusive WAL-number cutoff a file-scoped WAL failure quarantines.
  uint64_t TEST_WalRecoveryCutoff() const {
    return wal_recovery_cutoff_.load(std::memory_order_acquire);
  }

  void TEST_DeleteObsoleteFiles();

  const std::unordered_set<uint64_t>& TEST_GetFilesGrabbedForPurge() const {
    return files_grabbed_for_purge_;
  }

  const PeriodicTaskScheduler& TEST_GetPeriodicTaskScheduler() const;

  static Status TEST_ValidateOptions(const DBOptions& db_options) {
    return ValidateOptions(db_options);
  }

  // Exposes sequence and optional WAL-index allocation for concurrency tests.
  IOStatus TEST_AllocateSequenceAndWALIndex(
      size_t sequence_count, bool write_wal,
      SequenceNumber* last_sequence_before, uint64_t* wal_index) {
    return AllocateSequenceAndWALIndex(sequence_count, write_wal,
                                       last_sequence_before, wal_index,
                                       /*wal_partition=*/0,
                                       /*register_durability=*/false);
  }

  void TEST_SetNextWALIndex(uint64_t next_wal_index) {
    InstrumentedMutexLock lock(&sequence_wal_index_mutex_);
    next_wal_index_ = next_wal_index;
  }
#endif  // NDEBUG

  // In certain configurations, verify that the table/blob file cache only
  // contains entries for live files, to check for effective leaks of open
  // files. This can only be called when purging of obsolete files has
  // "settled," such as during parts of DB Close().
  void TEST_VerifyNoObsoleteFilesCached(bool db_mutex_already_held) const;

  // persist stats to column family "_persistent_stats"
  void PersistStats();

  // dump rocksdb.stats to LOG
  void DumpStats();

  // flush LOG out of application buffer
  void FlushInfoLog();

  // For the background timer job
  void RecordSeqnoToTimeMapping();

  // Compactions rely on an event triggers like flush/compaction/SetOptions.
  // We need to trigger periodic compactions even when there is no such trigger.
  // This function checks and schedules available compactions and will run
  // periodically.
  void TriggerPeriodicCompaction();

  // REQUIRES: DB mutex held
  std::pair<SequenceNumber, uint64_t> GetSeqnoToTimeSample() const;

  // REQUIRES: DB mutex held or during open
  void EnsureSeqnoToTimeMapping(const MinAndMaxPreserveSeconds& preserve_secs);

  // Computes the seqno->time preserve-window lower bound from
  // seqno_to_time_mapping_ and stores it on cfd's current version, so
  // bottommost file marking does not mark files whose largest seqno cannot be
  // zeroed yet (which would loop). No-op for column families without
  // preserve/preclude enabled. Returns true if the stored value changed, so
  // callers can recompute bottommost marking when the boundary moves.
  // Note: right after opening an existing DB, seqno_to_time_mapping_ may not be
  // fully reconstructed, so this bound can be imprecise until the first
  // periodic RecordSeqnoToTimeMapping. That is safe: CompactionJob folds this
  // same bound into kBottommostFiles compactions, so any marked file still
  // makes progress (never loops), and the periodic task self-corrects the
  // bound. REQUIRES: DB mutex held
  bool MaybeUpdatePreserveTimeMinSeqno(ColumnFamilyData* cfd);

  // Only called during open
  void PrepopulateSeqnoToTimeMapping(
      const MinAndMaxPreserveSeconds& preserve_secs);

  // Interface to block and signal the DB in case of stalling writes by
  // WriteBufferManager. Each DBImpl object contains ptr to WBMStallInterface.
  // When DB needs to be blocked or signalled by WriteBufferManager,
  // state_ is changed accordingly.
  class WBMStallInterface : public StallInterface {
   public:
    enum State {
      BLOCKED = 0,
      RUNNING,
    };

    WBMStallInterface() : state_cv_(&state_mutex_) {
      MutexLock lock(&state_mutex_);
      state_ = State::RUNNING;
    }

    void SetState(State state) {
      MutexLock lock(&state_mutex_);
      state_ = state;
    }

    // Change the state_ to State::BLOCKED and wait until its state is
    // changed by WriteBufferManager. When stall is cleared, Signal() is
    // called to change the state and unblock the DB.
    void Block() override {
      MutexLock lock(&state_mutex_);
      while (state_ == State::BLOCKED) {
        TEST_SYNC_POINT("WBMStallInterface::BlockDB");
        state_cv_.Wait();
      }
    }

    // Returns false when the wait times out while the DB is still blocked.
    bool BlockFor(uint64_t timeout_micros) {
      MutexLock lock(&state_mutex_);
      const std::chrono::microseconds deadline(
          SystemClock::Default()->NowMicros() + timeout_micros);
      while (state_ == State::BLOCKED) {
        TEST_SYNC_POINT("WBMStallInterface::BlockDB");
        if (SystemClock::Default()->TimedWait(&state_cv_, deadline) &&
            state_ == State::BLOCKED) {
          return false;
        }
      }
      return true;
    }

    // Called from WriteBufferManager. This function changes the state_
    // to State::RUNNING indicating the stall is cleared and DB can proceed.
    void Signal() override {
      {
        MutexLock lock(&state_mutex_);
        state_ = State::RUNNING;
      }
      state_cv_.Signal();
    }

   private:
    // Conditional variable and mutex to block and
    // signal the DB during stalling process.
    port::Mutex state_mutex_;
    port::CondVar state_cv_;
    // state represting whether DB is running or blocked because of stall by
    // WriteBufferManager.
    State state_;
  };

  // Adapts DBImpl to the shared WriteBufferManager flush registry.
  class WBMFlushInitiator : public FlushInitiator {
   public:
    WBMFlushInitiator(DBImpl* db, bool atomic_flush)
        : FlushInitiator(atomic_flush), db_(db) {}

    bool ScheduleFlush() override {
      return db_->ScheduleWriteBufferManagerFlush();
    }

    bool TryRefreshMemoryAccounting() override {
      return db_->TryRefreshFlushableMemAccounting();
    }

   private:
    DBImpl* const db_;
  };

  static void TEST_ResetDbSessionIdGen();
  static std::string GenerateDbSessionId(Env* env);

  bool seq_per_batch() const { return seq_per_batch_; }

#if USE_COROUTINES
  CoroDB* GetCoroDB() override { return this; }
#endif

 protected:
  const std::string dbname_;
  const bool read_only_;
  // TODO(peterd): unify with VersionSet::db_id_
  std::string db_id_;
  // db_session_id_ is an identifier that gets reset
  // every time the DB is opened
  std::string db_session_id_;
  // Declared before versions_ so memtables cannot outlive their allocation
  // accounting target during destruction.
  std::unique_ptr<FlushInitiator> wbm_flush_initiator_;
  std::unique_ptr<VersionSet> versions_;
  // Flag to check whether we allocated and own the info log file
  bool own_info_log_;
  Status init_logger_creation_s_;
  const DBOptions initial_db_options_;
  Env* const env_;
  std::shared_ptr<IOTracer> io_tracer_;
  const ImmutableDBOptions immutable_db_options_;
  FileSystemPtr fs_;
  MutableDBOptions mutable_db_options_;
  Statistics* stats_;
  std::atomic<uint64_t> mutex_hold_log_threshold_us_;
  std::shared_ptr<SlowMutexLoggingContext> slow_mutex_logging_;
  RecoveredTransactionMap recovered_transactions_;
  std::unique_ptr<Tracer> tracer_;
  InstrumentedMutex trace_mutex_;
  BlockCacheTracer block_cache_tracer_;

  // constant false canceled flag, used when the compaction is not manual
  const std::atomic<bool> kManualCompactionCanceledFalse_{false};

  // State below is protected by mutex_
  // With two_write_queues enabled, some of the variables that accessed during
  // WriteToWAL need different synchronization: wal_empties_, the WAL deques
  // (see their accessors), cur_wal_number_. Refer to the definition of each
  // variable below for more description.
  //
  // Protects access to most ColumnFamilyData methods, see more in comment for
  // each method.
  //
  // `mutex_` can be a hot lock in some workloads, so it deserves dedicated
  // cachelines.
  mutable CacheAlignedInstrumentedMutex mutex_;

  ColumnFamilyHandleImpl* default_cf_handle_ = nullptr;
  InternalStats* default_cf_internal_stats_ = nullptr;

  // table_cache_ provides its own synchronization
  std::shared_ptr<Cache> table_cache_;

  ErrorHandler error_handler_;

  // Unified interface for logging events
  EventLogger event_logger_;

  // only used for dynamically adjusting max_total_wal_size. it is a sum of
  // [write_buffer_size * max_write_buffer_number] over all column families
  std::atomic<uint64_t> max_total_in_memory_state_ = 0;

  // The options to access storage files
  const FileOptions file_options_;

  // Additonal options for compaction and flush
  FileOptions file_options_for_compaction_;

  std::unique_ptr<ColumnFamilyMemTablesImpl> column_family_memtables_;

  // Increase the sequence number after writing each batch, whether memtable is
  // disabled for that or not. Otherwise the sequence number is increased after
  // writing each key into memtable. This implies that when disable_memtable is
  // set, the seq is not increased at all.
  //
  // Default: false
  const bool seq_per_batch_;
  // This determines during recovery whether we expect one writebatch per
  // recovered transaction, or potentially multiple writebatches per
  // transaction. For WriteUnprepared, this is set to false, since multiple
  // batches can exist per transaction.
  //
  // Default: true
  const bool batch_per_txn_;

  // Each flush or compaction gets its own job id. this counter makes sure
  // they're unique
  std::atomic<int> next_job_id_ = 1;

  std::atomic<bool> shutting_down_ = false;
  // Protected by mutex_. This is separate from shutting_down_ because
  // OnDBShutdownBegin must fire before shutting_down_ is published, and the
  // notification releases mutex_ while invoking listeners. Marking that the
  // notification has started prevents a concurrent or reentrant
  // CancelAllBackgroundWork() call from firing it again during that unlocked
  // callback window.
  bool shutdown_notification_sent_ = false;

  // No new background jobs can be queued if true. This is used to prevent new
  // background jobs from being queued after WaitForCompact() completes waiting
  // all background jobs then attempts to close when close_db_ option is true.
  bool reject_new_background_jobs_ = false;

  // Identifies an open-ended obsolete suffix in one WAL file.
  struct WALIndexSupersession {
    uint64_t wal_number;
    uint64_t first_obsolete_wal_index;
  };

  // Stores recovery version edits and the WAL supersessions that must be
  // persisted before the recovered database becomes writable.
  class RecoveryContext {
   public:
    RecoveryContext() = default;
    // No copying allowed
    RecoveryContext(const RecoveryContext&) = delete;
    RecoveryContext& operator=(const RecoveryContext&) = delete;
    RecoveryContext(RecoveryContext&&) = delete;
    RecoveryContext& operator=(RecoveryContext&&) = delete;

    ~RecoveryContext() {
      for (auto& edit_list : edit_lists_) {
        for (auto* edit : edit_list) {
          delete edit;
        }
      }
    }

    void UpdateVersionEdits(ColumnFamilyData* cfd, const VersionEdit& edit) {
      assert(cfd != nullptr);
      if (map_.find(cfd->GetID()) == map_.end()) {
        uint32_t size = static_cast<uint32_t>(map_.size());
        map_.emplace(cfd->GetID(), size);
        cfds_.emplace_back(cfd);
        edit_lists_.emplace_back(autovector<VersionEdit*>());
      }
      uint32_t i = map_[cfd->GetID()];
      edit_lists_[i].emplace_back(new VersionEdit(edit));
    }

    bool HasVersionEdits() const {
      for (const auto& edit_list : edit_lists_) {
        if (!edit_list.empty()) {
          return true;
        }
      }
      return false;
    }

    std::unordered_map<uint32_t, uint32_t> map_;  // cf_id to index;
    autovector<ColumnFamilyData*> cfds_;
    autovector<autovector<VersionEdit*>> edit_lists_;
    // Files found in the primary DB directory by MaybeUpdateNextFileNumber().
    // Reused for OPTIONS selection so writable open performs one directory
    // scan instead of two.
    std::vector<std::string> existing_db_files_;
    // All existing data files (SST files and Blob files) found during DB::Open.
    std::vector<std::string> existing_data_files_;
    bool is_new_db_ = false;
    std::vector<WALIndexSupersession> wal_index_supersessions_;
  };

  // Persist options to options file. Must be holding options_mutex_.
  // Will lock DB mutex if !db_mutex_already_held.
  Status WriteOptionsFile(const WriteOptions& write_options,
                          bool db_mutex_already_held);

  // Writes, renames, and directory-syncs an OPTIONS snapshot, but does not
  // publish it in VersionSet. The caller must either commit its file number in
  // MANIFEST or leave it as an unreferenced orphan.
  Status PersistOptionsFile(const WriteOptions& write_options,
                            const DBOptions& db_options,
                            const std::vector<std::string>& cf_names,
                            const std::vector<ColumnFamilyOptions>& cf_opts,
                            uint64_t* options_file_number,
                            uint64_t* options_file_size);

  // For tracked DBs, writes the OPTIONS snapshot that is committed with a
  // column-family add/drop. Returns file number zero in legacy mode.
  Status PersistOptionsFileForColumnFamilyManipulation(
      const WriteOptions& write_options,
      const std::vector<ColumnFamilyDescriptor>& added_column_families,
      const std::vector<uint32_t>& excluded_cf_ids,
      uint64_t* options_file_number, uint64_t* options_file_size);

  Status CompactRangeInternal(const CompactRangeOptions& options,
                              ColumnFamilyHandle* column_family,
                              const Slice* begin, const Slice* end,
                              const std::string& trim_ts);

  // The following two functions can only be called when:
  // 1. WriteThread::Writer::EnterUnbatched() is used.
  // 2. db_mutex is NOT held
  Status RenameTempFileToOptionsFile(const std::string& file_name,
                                     uint64_t options_file_number,
                                     uint64_t* options_file_size);
  void MaybeDeleteObsoleteOptionsFiles();
  Status DeleteObsoleteOptionsFiles(bool schedule_only);

  void NotifyOnManualFlushScheduled(autovector<ColumnFamilyData*> cfds,
                                    FlushReason flush_reason);

  void NotifyOnFlushBegin(ColumnFamilyData* cfd, FileMetaData* file_meta,
                          const MutableCFOptions& mutable_cf_options,
                          int job_id, FlushReason flush_reason);

  void NotifyOnFlushCompleted(
      ColumnFamilyData* cfd, const MutableCFOptions& mutable_cf_options,
      std::list<std::unique_ptr<FlushJobInfo>>* flush_jobs_info);

  void NotifyOnCompactionBegin(ColumnFamilyData* cfd, Compaction* c,
                               const Status& st,
                               const CompactionJobStats& job_stats, int job_id);

  void NotifyOnCompactionCompleted(ColumnFamilyData* cfd, Compaction* c,
                                   const Status& st,
                                   const CompactionJobStats& job_stats,
                                   int job_id);
  // Fires the OnCompactionPreCommit listener callback. Mirrors
  // NotifyOnCompactionCompleted but is invoked before ReleaseCompactionFiles()
  // (i.e. while input files still have being_compacted == true). Idempotent:
  // safe to call from multiple potential release sites; only fires once per
  // compaction, and only if NotifyOnCompactionBegin previously fired.
  void NotifyOnCompactionPreCommit(ColumnFamilyData* cfd, Compaction* c,
                                   const Status& st,
                                   const CompactionJobStats& job_stats,
                                   int job_id);
  void NotifyOnDBShutdownBegin();
  void NotifyOnMemTableSealed(ColumnFamilyData* cfd,
                              const MemTableInfo& mem_table_info);

  void NotifyOnExternalFileIngested(
      ColumnFamilyData* cfd, const ExternalSstFileIngestionJob& ingestion_job);

  Status FlushAllColumnFamilies(const FlushOptions& flush_options,
                                FlushReason flush_reason);

  virtual Status FlushForGetLiveFiles(bool force_atomic_flush = false);

  void NewThreadStatusCfInfo(ColumnFamilyData* cfd) const;

  void EraseThreadStatusCfInfo(ColumnFamilyData* cfd) const;

  void EraseThreadStatusDbInfo() const;

  // For CFs that has updates in `wbwi`, their memtable will be switched,
  // and `wbwi` will be added as the latest immutable memtable.
  //
  // REQUIRES: this thread is currently at the front of the main writer queue.
  // @param prep_log refers to the WAL that contains prepare record
  // for the transaction based on wbwi.
  // @param assigned_seqno Sequence numbers for the ingested memtable.
  // @param last_seqno the value of versions_->LastSequence() after the write
  // ingests `wbwi` is done.
  // @param memtable_updated Whether the same write that ingests wbwi has
  // updated memtable. This is useful for determining whether to set bg
  // error when IngestWBWIAsMemtable fails.
  // @param ingest_wbwi_for_commit Whether wbwi ingestion is publishing the
  // committed data of a prepared transaction. This means a failure can leave
  // committed data durable in WAL but not published in memtables.
  // @param ignore_missing_cf If true, skip column families not found in the DB
  // instead of returning an error.
  Status IngestWBWIAsMemtable(std::shared_ptr<WriteBatchWithIndex> wbwi,
                              const WBWIMemTable::SeqnoRange& assigned_seqno,
                              uint64_t min_prep_log, SequenceNumber last_seqno,
                              bool memtable_updated,
                              bool ingest_wbwi_for_commit,
                              bool ignore_missing_cf);

  // If disable_memtable is set the application logic must guarantee that the
  // batch will still be skipped from memtable during the recovery. An excption
  // to this is seq_per_batch_ mode, in which since each batch already takes one
  // seq, it is ok for the batch to write to memtable during recovery as long as
  // it only takes one sequence number: i.e., no duplicate keys.
  // In WriteCommitted it is guarnateed since disable_memtable is used for
  // prepare batch which will be written to memtable later during the commit,
  // and in WritePrepared it is guaranteed since it will be used only for WAL
  // markers which will never be written to memtable. If the commit marker is
  // accompanied with CommitTimeWriteBatch that is not written to memtable as
  // long as it has no duplicate keys, it does not violate the one-seq-per-batch
  // policy.
  // batch_cnt is expected to be non-zero in seq_per_batch mode and
  // indicates the number of sub-patches. A sub-patch is a subset of the write
  // batch that does not have duplicate keys.
  // `callback` is called before WAL write.
  // See more in comment above WriteCallback::Callback().
  // pre_release_callback is called after WAL write and before memtable write.
  // See more in comment above PreReleaseCallback::Callback().
  // post_memtable_callback is called after memtable write but before publishing
  // the sequence number to readers.
  // `trace_batch_override`, when non-null, supplies the logical batch to trace
  // instead of `updates`. Fast paths may rewrite or rebuild `updates` before
  // apply, but tracing should still record the original user-visible batch.
  // `skip_blob_direct_write_transform` indicates the caller has already
  // performed any needed batch-level blob direct-write preprocessing.
  // `deferred_put_entities`, when non-null, carries structured `PutEntity()`
  // ops that should be materialized into `updates` after `PreprocessWrite()`
  // has selected the target memtable/blob generation.
  //
  // The main write queue. This is the only write queue that updates
  // LastSequence. When using one write queue, the same sequence also indicates
  // the last published sequence.
  struct DeferredPutEntityBatch;
  Status WriteImpl(
      const WriteOptions& options, WriteBatch* updates,
      WriteCallback* callback = nullptr,
      UserWriteCallback* user_write_cb = nullptr, uint64_t* wal_used = nullptr,
      uint64_t log_ref = 0, bool disable_memtable = false,
      uint64_t* seq_used = nullptr, size_t batch_cnt = 0,
      PreReleaseCallback* pre_release_callback = nullptr,
      PostMemTableCallback* post_memtable_callback = nullptr,
      std::shared_ptr<WriteBatchWithIndex> wbwi = nullptr,
      WriteBatch* trace_batch_override = nullptr,
      bool skip_blob_direct_write_transform = false,
      const DeferredPutEntityBatch* deferred_put_entities = nullptr);

  // Per-WriteImpl state that keeps BDW column families pinned through
  // referenced SuperVersions until the transformed write either commits or
  // rolls back.
  struct BlobDirectWriteContext;
  // High-level `PutEntity()` fast-path flow:
  //
  //   non-BDW target CF:
  //     PutEntityFastPath()
  //       -> AppendPreprocessedPutEntityToBatch()
  //       -> WritePreprocessedPutEntityBatch()
  //       -> WriteImpl(skip_blob_direct_write_transform = true)
  //
  //   BDW-enabled target CF:
  //     PutEntityFastPath()
  //       -> sort columns and stash DeferredPutEntityBatch::Op
  //       -> WritePreprocessedPutEntityBatch()
  //       -> WriteImpl(..., deferred_put_entities)
  //       -> after PreprocessWrite() picks the memtable/blob generation:
  //            AppendSortedPutEntityToBatch()
  //
  // The BDW branch defers final serialization until `WriteImpl()` knows the
  // target memtable/blob generation, avoiding the old
  // serialize -> deserialize -> serialize cycle on the write hot path.
  // Staging buffer for the `PutEntity()` fast path when blob direct write may
  // need to rewrite wide-column values. `PutEntityFastPath()` records the
  // logical operations here instead of serializing them into a `WriteBatch`
  // immediately, then passes this struct to `WriteImpl()`. Once
  // `PreprocessWrite()` has selected the memtable/blob-file generation,
  // `WriteImpl()` materializes each op directly into the final batch with blob
  // direct-write preprocessing applied. This avoids the old
  // serialize-deserialize-serialize cycle on the write hot path.
  struct DeferredPutEntityBatch {
    // The fast path sorts columns up front so `WriteImpl()` can append the
    // final serialized entity without redoing that work.
    struct Op {
      // Target column family for the logical `PutEntity()` op.
      uint32_t column_family_id = 0;
      // Owned key bytes kept alive until `WriteImpl()` rebuilds the batch.
      std::string key;
      // Pre-sorted columns ready for final serialization and blob
      // direct-write preprocessing.
      WideColumns sorted_columns;
    };

    // Logical `PutEntity()` ops collected by the fast path and replayed into
    // the final `WriteBatch` inside `WriteImpl()`.
    std::vector<Op> ops;
  };
  // Rewrites a write batch for blob direct write when the current DB and batch
  // shape are compatible, recording touched managers and rollback metadata in
  // `blob_direct_write_ctx`.
  Status MaybeTransformBatchForBlobDirectWrite(
      const WriteOptions& write_options, WriteBatch** batch,
      bool should_write_to_memtable, bool lookup_from_write_thread,
      WriteBatch* transformed_storage,
      BlobDirectWriteContext* blob_direct_write_ctx);
  // Sorts and preprocesses one `PutEntity()` op, then appends the final
  // serialized form to `batch`.
  Status AppendPreprocessedPutEntityToBatch(
      const WriteOptions& write_options, WriteBatch* batch,
      ColumnFamilyHandle* column_family, const Slice& key,
      const WideColumns& columns,
      BlobDirectWriteContext* blob_direct_write_ctx);
  // Appends a `PutEntity()` op whose columns are already sorted, allowing the
  // fast path to avoid re-sorting before final serialization.
  Status AppendSortedPutEntityToBatch(
      const WriteOptions& write_options, WriteBatch* batch,
      uint32_t column_family_id, const Slice& key,
      const WideColumns& sorted_columns,
      BlobDirectWriteContext* blob_direct_write_ctx);
  // Specialized `PutEntity()` write path for a single column family. This
  // avoids building an intermediate serialized entity when blob direct write
  // needs to rewrite the final batch inside `WriteImpl()`.
  Status PutEntityFastPath(const WriteOptions& write_options,
                           ColumnFamilyHandle* column_family, const Slice& key,
                           const WideColumns& columns);
  // Specialized `PutEntity(AttributeGroups)` path that defers final batch
  // materialization until `WriteImpl()` when any target CF may use blob direct
  // write.
  Status PutEntityFastPath(const WriteOptions& write_options, const Slice& key,
                           const AttributeGroups& attribute_groups);
  // Writes a batch prepared by the `PutEntity()` fast path. When
  // `deferred_put_entities` is present, `WriteImpl()` rebuilds `batch`
  // internally after `PreprocessWrite()` and disables normal write batching to
  // keep the staging state local to this write. `trace_batch` stays separate
  // so tracing can still emit the logical `PutEntity()` batch even when the
  // applied batch is materialized later inside `WriteImpl()`.
  Status WritePreprocessedPutEntityBatch(
      const WriteOptions& write_options, WriteBatch* batch,
      WriteBatch* trace_batch,
      const DeferredPutEntityBatch* deferred_put_entities = nullptr);
  // Flushes or syncs all blob direct-write managers touched by the current
  // transformed write before the write can proceed.
  Status SyncBlobDirectWriteManagers(
      const WriteOptions& write_options,
      const BlobDirectWriteContext& blob_direct_write_ctx);

  Status PipelinedWriteImpl(const WriteOptions& options, WriteBatch* updates,
                            WriteBatch* trace_batch, size_t wal_partition,
                            WriteCallback* callback = nullptr,
                            UserWriteCallback* user_write_cb = nullptr,
                            uint64_t* wal_used = nullptr, uint64_t log_ref = 0,
                            bool disable_memtable = false,
                            uint64_t* seq_used = nullptr);

  // Write only to memtables without joining any write queue
  Status UnorderedWriteMemtable(const WriteOptions& write_options,
                                WriteBatch* my_batch, WriteCallback* callback,
                                uint64_t log_ref, SequenceNumber seq,
                                const size_t sub_batch_cnt);

  // Whether the batch requires to be assigned with an order
  enum AssignOrder : bool { kDontAssignOrder, kDoAssignOrder };
  // Whether it requires publishing last sequence or not
  enum PublishLastSeq : bool { kDontPublishLastSeq, kDoPublishLastSeq };

  // Join the write_thread to write the batch only to the WAL. It is the
  // responsibility of the caller to also write the write batch to the memtable
  // if it required.
  //
  // sub_batch_cnt is expected to be non-zero when assign_order = kDoAssignOrder
  // indicating the number of sub-batches in my_batch. A sub-patch is a subset
  // of the write batch that does not have duplicate keys. When seq_per_batch is
  // not set, each key is a separate sub_batch. Otherwise each duplicate key
  // marks start of a new sub-batch.
  Status WriteImplWALOnly(WriteThread* write_thread,
                          const WriteOptions& options, WriteBatch* updates,
                          WriteBatch* trace_batch, WriteCallback* callback,
                          UserWriteCallback* user_write_cb, uint64_t* wal_used,
                          const uint64_t log_ref, uint64_t* seq_used,
                          const size_t sub_batch_cnt,
                          PreReleaseCallback* pre_release_callback,
                          const AssignOrder assign_order,
                          const PublishLastSeq publish_last_seq,
                          const bool disable_memtable, size_t wal_partition);

  // write cached_recoverable_state_ to memtable if it is not empty
  // The writer must be the leader in write_thread_ and holding mutex_
  Status WriteRecoverableState();

  // Actual implementation of Close()
  virtual Status CloseImpl();

  // Recover the descriptor from persistent storage.  May do a significant
  // amount of work to recover recently logged updates.  Any changes to
  // be made to the descriptor are added to *edit.
  // recovered_seq is set to less than kMaxSequenceNumber if the log's tail is
  // skipped.
  // recovery_ctx stores the context about version edits and all those
  // edits are persisted to new Manifest after successfully syncing the new WAL.
  virtual Status Recover(
      const std::vector<ColumnFamilyDescriptor>& column_families,
      bool read_only = false, bool error_if_wal_file_exists = false,
      bool error_if_data_exists_in_wals = false, bool is_retry = false,
      uint64_t* recovered_seq = nullptr,
      RecoveryContext* recovery_ctx = nullptr, bool* can_retry = nullptr);

  virtual bool OwnTablesAndLogs() const { return true; }

  // Read/create DB identity file (as appropriate), and write DB ID to
  // version_edit if provided.
  Status SetupDBId(const WriteOptions& write_options, bool read_only,
                   bool is_new_db, bool is_retry, VersionEdit* version_edit);
  // Assign db_id_ and write DB ID to version_edit if provided.
  void SetDBId(std::string&& id, bool read_only, VersionEdit* version_edit);

  // Collect a deduplicated collection of paths used by this DB, including
  // dbname_, DBOptions.db_paths, ColumnFamilyOptions.cf_paths.
  std::set<std::string> CollectAllDBPaths();

  // REQUIRES: db mutex held when calling this function, but the db mutex can
  // be released and re-acquired. Db mutex will be held when the function
  // returns.
  // It stores all existing data files (SST and Blob) in RecoveryContext. In
  // the meantime, we find out the largest file number present in the paths, and
  // bump up the version set's next_file_number_ to be 1 + largest_file_number.
  // recovery_ctx stores the context about version edits. All those edits are
  // persisted to new Manifest after successfully syncing the new WAL.
  Status MaybeUpdateNextFileNumber(RecoveryContext* recovery_ctx);

  // Track existing data files, including both referenced and unreferenced SST
  // and Blob files in SstFileManager. This is only called during DB::Open and
  // it's called before any file deletion start so that their deletion can be
  // properly rate limited.
  // Files may not be referenced in the MANIFEST because (e.g.
  // 1. It's best effort recovery;
  // 2. The VersionEdits referencing the SST files are appended to
  // RecoveryContext, DB crashes when syncing the MANIFEST, the VersionEdits are
  // still not synced to MANIFEST during recovery.)
  //
  // If the file is referenced in Manifest (typically that's the
  // vast majority of all files), since it already has the file size
  // on record, we don't need to query the file system. Otherwise, we query the
  // file system for the size of an unreferenced file.
  // REQUIRES: mutex unlocked
  void TrackExistingDataFiles(
      const std::vector<std::string>& existing_data_files);

  // Untrack data files in sst manager. This is only called during DB::Close on
  // an unowned SstFileManager, to return it to a consistent state.
  // REQUIRES: mutex unlocked
  void UntrackDataFiles();

  // SetDbSessionId() should be called in the constuctor DBImpl()
  // to ensure that db_session_id_ gets updated every time the DB is opened
  void SetDbSessionId();

  Status FailIfCfHasTs(const ColumnFamilyHandle* column_family) const;
  Status FailIfTsMismatchCf(ColumnFamilyHandle* column_family,
                            const Slice& ts) const;
  Status FailIfTableFilterWithRangeConversion(
      const ReadOptions& read_options,
      const MutableCFOptions& mutable_cf_options) const;

  // Check that the read timestamp `ts` is at or above the `full_history_ts_low`
  // timestamp in a `SuperVersion`. It's necessary to do this check after
  // grabbing the SuperVersion. If the check passed, the referenced SuperVersion
  // this read holds on to can ensure the read won't be affected if
  // `full_history_ts_low` is increased concurrently, and this achieves that
  // without explicitly locking by piggybacking the SuperVersion.
  Status FailIfReadCollapsedHistory(const ColumnFamilyData* cfd,
                                    const SuperVersion* sv,
                                    const Slice& ts) const;

  // recovery_ctx stores the context about version edits and
  // LogAndApplyForRecovery persist all those edits to new Manifest after
  // successfully syncing new WAL.
  // LogAndApplyForRecovery should be called only once during recovery and it
  // should be called when RocksDB writes to a first new MANIFEST since this
  // recovery.
  Status LogAndApplyForRecovery(RecoveryContext& recovery_ctx);

  // Sets, lowers or clears the MANIFEST's WAL index guard to match the WALs
  // that may carry WAL index records once recovery edits are applied. Runs
  // before Open returns, so before any write can be acknowledged.
  Status UpdateWALIndexGuard();

  // Schedule background work to open and validate SST files asynchronously.
  // Called when open_files_async is enabled.
  void ScheduleAsyncFileOpening();

  // Mark async file opening as not needed. Used by subclasses that load
  // table files through a different mechanism (e.g., ReactiveVersionSet).
  void MarkAsyncFileOpenNotNeeded();

  // Background work function for async file opening.
  static void BGWorkAsyncFileOpen(void* arg);

  // Block the until any in-flight async file open work has
  // completed. No-op when open_files_async is false. Returns early if
  // shutdown begins.
  void WaitForAsyncFileOpen();

  void InvokeWalFilterIfNeededOnColumnFamilyToWalNumberMap();

  // Return true to proceed with current WAL record whose content is stored in
  // `batch`. Return false to skip current WAL record.
  bool InvokeWalFilterIfNeededOnWalRecord(uint64_t wal_number,
                                          const std::string& wal_fname,
                                          log::Reader::Reporter& reporter,
                                          Status& status, bool& stop_replay,
                                          WriteBatch& batch);

  // Indicate DB was opened successfully
  bool opened_successfully_ = false;

 private:
  friend class DB;
  friend class DBImplReadOnly;
  friend class DBImplSecondary;
  friend class ErrorHandler;
  friend class InternalStats;
  friend class PessimisticTransaction;
  friend class TransactionBaseImpl;
  friend class WriteCommittedTxn;
  friend class WritePreparedTxn;
  friend class WritePreparedTxnDB;
  friend class WriteBatchWithIndex;
  friend class WriteUnpreparedTxnDB;
  friend class WriteUnpreparedTxn;
  friend class FileIngestionHandleImpl;

  friend class ForwardIterator;
  friend struct SuperVersion;
  friend class CompactedDBImpl;
  friend class DBImplFollower;
#ifndef NDEBUG
  friend class DBTest_ConcurrentFlushWAL_Test;
  friend class DBWritePartitionTest;
  friend class DBTest_MixedSlowdownOptionsStop_Test;
  friend class DBCompactionTest_CompactBottomLevelFilesWithDeletions_Test;
  friend class DBCompactionTest_CompactionDuringShutdown_Test;
  friend class DBCompactionTest_DelayCompactBottomLevelFilesWithDeletions_Test;
  friend class DBCompactionTest_DisableCompactBottomLevelFiles_Test;
  friend class StatsHistoryTest_PersistentStatsCreateColumnFamilies_Test;
  friend class DBTest2_ReadCallbackTest_Test;
  friend class WriteCallbackPTest_WriteWithCallbackTest_Test;
  friend class XFTransactionWriteHandler;
  friend class DBBlobIndexTest;
  friend class WriteUnpreparedTransactionTest_RecoveryTest_Test;
  friend class CompactionServiceTest_PreservedOptionsLocalCompaction_Test;
  friend class CompactionServiceTest_PreservedOptionsRemoteCompaction_Test;
#endif

  // Same as HasAnyBlobDirectWriteColumnFamily(), but requires `mutex_` held.
  bool HasAnyBlobDirectWriteColumnFamilyWithLockHeld();
  // Returns true if any BDW column family still owns blob files that have not
  // yet been made visible through MANIFEST. Requires `mutex_` held.
  bool HasInFlightBlobDirectWriteFilesWithLockHeld();
  // Creates and attaches the per-CF blob direct-write partition manager when
  // the column family is opened with the feature enabled.
  void MaybeInitBlobDirectWriteColumnFamily(
      ColumnFamilyData* cfd, const ColumnFamilyOptions& cf_options,
      const std::string& column_family_name);
  // Increments the DB-level count of live blob direct-write column families.
  void RegisterBlobDirectWriteColumnFamily();
  // Decrements the DB-level count of live blob direct-write column families.
  void UnregisterBlobDirectWriteColumnFamily();

  struct CompactionState;
  struct PrepickedCompaction;
  struct PurgeFileInfo;

  struct WriteContext {
    SuperVersionContext superversion_context;
    autovector<ReadOnlyMemTable*> memtables_to_free_;

    explicit WriteContext(bool create_superversion = false)
        : superversion_context(create_superversion) {}

    // No copying allowed
    WriteContext(const WriteContext&) = delete;
    WriteContext& operator=(const WriteContext&) = delete;
    WriteContext(WriteContext&&) = delete;
    WriteContext& operator=(WriteContext&&) = delete;

    ~WriteContext() {
      superversion_context.Clean();
      for (auto& m : memtables_to_free_) {
        delete m;
      }
    }
  };

  struct WalFileNumberSize {
    explicit WalFileNumberSize(uint64_t _number) : number(_number) {}
    WalFileNumberSize() {}
    void AddSize(uint64_t new_size) { size += new_size; }
    uint64_t number;
    uint64_t size = 0;
    bool getting_flushed = false;
  };

  struct LogWriterNumber {
    // pass ownership of _writer
    LogWriterNumber(uint64_t _number, log::Writer* _writer)
        : number(_number), writer(_writer) {}

    log::Writer* ReleaseWriter() {
      auto* w = writer;
      writer = nullptr;
      return w;
    }
    Status ClearWriter() {
      Status s;
      if (writer->file()) {
        // TODO: plumb Env::IOActivity, Env::IOPriority
        s = writer->WriteBuffer(WriteOptions());
        if (attempt_truncate_size < SIZE_MAX &&
            attempt_truncate_size < writer->file()->GetFileSize()) {
          Status s2 = writer->file()->writable_file()->Truncate(
              attempt_truncate_size, IOOptions{}, nullptr);
          // This is just a best effort attempt
          s2.PermitUncheckedError();
        }
      }
      delete writer;
      writer = nullptr;
      return s;
    }

    bool IsSyncing() { return getting_synced; }

    uint64_t GetPreSyncSize() {
      assert(getting_synced);
      return pre_sync_size;
    }

    void PrepareForSync() {
      assert(!getting_synced);
      // Ensure the deque head is marked as getting_synced if any entry is.
      getting_synced = true;
      // If last sync failed on a later WAL, this could be a fully synced
      // and closed WAL that just needs to be recorded as synced in the
      // manifest.
      if (writer->file()) {
        // Size is expected to be monotonically increasing.
        assert(writer->file()->GetFlushedSize() >= pre_sync_size);
        pre_sync_size = writer->file()->GetFlushedSize();
      }
    }

    void FinishSync() {
      assert(getting_synced);
      getting_synced = false;
    }

    // Whether the open file holds bytes past synced_size: an upper bound on
    // unsynced data, since synced_size is a lower bound on what was synced.
    bool HasUnsyncedBytes() const {
      return writer->file() != nullptr &&
             writer->file()->GetFileSize() > synced_size;
    }

    void SetAttemptTruncateSize(uint64_t size) {
      assert(attempt_truncate_size == SIZE_MAX);
      attempt_truncate_size = size;
    }

    uint64_t number;
    // Visual Studio doesn't support deque's member to be noncopyable because
    // of a std::unique_ptr as a member.
    log::Writer* writer;  // own
    // File size a completed sync covered: advanced by the leadership sync's
    // own fsync (SyncWalPartitionFiles) and, for every successful sync, to
    // the size at its mark when the sync finishes (FinishWalPartitionSync).
    uint64_t synced_size = 0;

   private:
    // true for some prefix of the partition's deque
    bool getting_synced = false;
    // The size of the file before the sync happens. This amount is guaranteed
    // to be persisted even if appends happen during sync so it can be used for
    // tracking the synced size in MANIFEST.
    uint64_t pre_sync_size = 0;
    // When < SIZE_MAX, attempt to truncate the WAL to this size on close,
    // because a bad entry was written to it beyond that point and it likely
    // won't be recoverable with the bad entry.
    uint64_t attempt_truncate_size = SIZE_MAX;
  };

  struct WalContext {
    explicit WalContext(bool need_sync = false)
        : need_wal_sync(need_sync), need_wal_dir_sync(need_sync) {}
    bool need_wal_sync = false;
    bool need_wal_dir_sync = false;
    size_t wal_partition = 0;
    log::Writer* writer = nullptr;
    // Bound by SelectWalForWrite under WalWriteMutex(wal_partition). After
    // that mutex is released, only ->number (immutable) may be touched,
    // except by the N==1 writer that adds the size without it; see
    // CurrentAliveWal's escape rule.
    WalFileNumberSize* wal_file_number_size = nullptr;
    uint64_t prev_size = SIZE_MAX;
    // Sync writes only. The leadership claim's horizon, read under all N WAL
    // mutexes before the claim loop: a WAL installed after it is numbered
    // above it. The completion judges "at or below the horizon" by this.
    uint64_t wal_sync_up_to = 0;
    // Sync writes only. Per partition, whether the leadership claim marked
    // anything there; the completion finishes or cancels only those.
    autovector<uint8_t, 1> wal_sync_claimed;
    // Sync writes only: how many partitions held unsynced bytes at the
    // claim. Above one partition the leadership gate sync counts its own
    // (SyncWalsFromCompletionGate).
    uint32_t wal_sync_unsynced_partitions = 0;
    // Sync writes only. The WAL whose leadership-time fsync failed; 0 if
    // none failed.
    uint64_t sync_failed_wal_number = 0;
  };

  // PurgeFileInfo is a structure to hold information of files to be deleted in
  // purge_files_
  struct PurgeFileInfo {
    std::string fname;
    std::string dir_to_sync;
    FileType type;
    uint64_t number;
    int job_id;
    PurgeFileInfo(std::string fn, std::string d, FileType t, uint64_t num,
                  int jid)
        : fname(fn), dir_to_sync(d), type(t), number(num), job_id(jid) {}
  };

  // Argument required by background flush thread.
  struct BGFlushArg {
    BGFlushArg()
        : cfd_(nullptr),
          max_memtable_id_(0),
          superversion_context_(nullptr),
          flush_reason_(FlushReason::kOthers) {}
    BGFlushArg(ColumnFamilyData* cfd, uint64_t max_memtable_id,
               SuperVersionContext* superversion_context,
               FlushReason flush_reason, bool atomic_flush)
        : cfd_(cfd),
          max_memtable_id_(max_memtable_id),
          superversion_context_(superversion_context),
          flush_reason_(flush_reason),
          atomic_flush_(atomic_flush) {}

    // Column family to flush.
    ColumnFamilyData* cfd_;
    // Maximum ID of memtable to flush. In this column family, memtables with
    // IDs smaller than this value must be flushed before this flush completes.
    uint64_t max_memtable_id_;
    // Pointer to a SuperVersionContext object. After flush completes, RocksDB
    // installs a new superversion for the column family. This operation
    // requires a SuperVersionContext object (currently embedded in JobContext).
    SuperVersionContext* superversion_context_;
    FlushReason flush_reason_;
    // Whether this flush should use atomic flush code path.
    bool atomic_flush_ = false;
  };

  // Argument passed to flush thread.
  struct FlushThreadArg {
    DBImpl* db_;

    Env::Priority thread_pri_;
  };

  // Information for a manual compaction
  struct ManualCompactionState {
    ManualCompactionState(ColumnFamilyData* _cfd, int _input_level,
                          int _output_level, uint32_t _output_path_id,
                          bool _exclusive, bool _disallow_trivial_move,
                          std::atomic<bool>* _canceled)
        : cfd(_cfd),
          input_level(_input_level),
          output_level(_output_level),
          output_path_id(_output_path_id),
          exclusive(_exclusive),
          disallow_trivial_move(_disallow_trivial_move),
          canceled(_canceled ? *_canceled : canceled_internal_storage) {}
    // When _canceled is not provided by ther user, we assign the reference of
    // canceled_internal_storage to it to consolidate canceled and
    // manual_compaction_paused since DisableManualCompaction() might be
    // called

    ColumnFamilyData* cfd;
    int input_level;
    int output_level;
    uint32_t output_path_id;
    Status status;
    bool done = false;
    bool in_progress = false;    // compaction request being processed?
    bool incomplete = false;     // only part of requested range compacted
    bool exclusive;              // current behavior of only one manual
    bool disallow_trivial_move;  // Force actual compaction to run
    const InternalKey* begin = nullptr;  // nullptr means beginning of key range
    const InternalKey* end = nullptr;    // nullptr means end of key range
    InternalKey* manual_end = nullptr;   // how far we are compacting
    InternalKey tmp_storage;   // Used to keep track of compaction progress
    InternalKey tmp_storage1;  // Used to keep track of compaction progress

    // When the user provides a canceled pointer in CompactRangeOptions, the
    // above varaibe is the reference of the user-provided
    // `canceled`, otherwise, it is the reference of canceled_internal_storage
    std::atomic<bool> canceled_internal_storage = false;
    std::atomic<bool>& canceled;  // Compaction canceled pointer reference
  };
  struct PrepickedCompaction {
    // background compaction takes ownership of `compaction`.
    // TODO(hx235): consider using std::shared_ptr for easier ownership
    // management
    Compaction* compaction;
    // caller retains ownership of `manual_compaction_state` as it is reused
    // across background compactions.
    ManualCompactionState* manual_compaction_state;  // nullptr if non-manual
    // task limiter token is requested during compaction picking.
    std::unique_ptr<TaskLimiterToken> task_token;
    // If true, `compaction` is picked temporarily to express compaction intent
    // and will be released before re-picking a real compaction based on the
    // updated LSM shape when thread associated with `compaction` is ready to
    // run
    bool need_repick;
  };

  struct CompactionArg {
    // caller retains ownership of `db`.
    DBImpl* db;
    // background compaction takes ownership of `prepicked_compaction`.
    PrepickedCompaction* prepicked_compaction;
    Env::Priority compaction_pri_;
  };

  static bool IsRecoveryFlush(FlushReason flush_reason) {
    return flush_reason == FlushReason::kErrorRecoveryRetryFlush ||
           flush_reason == FlushReason::kErrorRecovery;
  }
  // Initialize the built-in column family for persistent stats. Depending on
  // whether on-disk persistent stats have been enabled before, it may either
  // create a new column family and column family handle or just a column family
  // handle.
  // Required: DB mutex held
  Status InitPersistStatsColumnFamily();

  // Persistent Stats column family has two format version key which are used
  // for compatibility check. Write format version if it's created for the
  // first time, read format version and check compatibility if recovering
  // from disk. This function requires DB mutex held at entrance but may
  // release and re-acquire DB mutex in the process.
  // Required: DB mutex held
  Status PersistentStatsProcessFormatVersion();

  // `io_activity` is used to construct the ReadOptions/WriteOptions for the
  // internal recovery work (e.g. the MANIFEST write in LogAndApply). Recovery
  // is driven by flush operations, so callers pass Env::IOActivity::kFlush.
  Status ResumeImpl(DBRecoverContext context, Env::IOActivity io_activity);

  void MaybeIgnoreError(Status* s) const;

  const Status CreateArchivalDirectory();

  // Create a column family, without some of the follow-up work yet
  Status CreateColumnFamilyImpl(const ReadOptions& read_options,
                                const WriteOptions& write_options,
                                const ColumnFamilyOptions& cf_options,
                                const std::string& cf_name,
                                ColumnFamilyHandle** handle);
  Status CreateColumnFamiliesImpl(
      const ReadOptions& read_options, const WriteOptions& write_options,
      const std::vector<ColumnFamilyDescriptor>& column_families,
      std::vector<ColumnFamilyHandle*>* handles);
  Status CreateColumnFamilyWithPreparedOptions(
      const ReadOptions& read_options, const WriteOptions& write_options,
      const ColumnFamilyOptions& sanitized_cf_options,
      const std::string& cf_name, uint64_t options_file_number,
      uint64_t options_file_size, ColumnFamilyHandle** handle);

  // Follow-up work to user creating a column family or (families)
  Status WrapUpCreateColumnFamilies(
      const WriteOptions& write_options,
      const std::vector<const ColumnFamilyOptions*>& cf_options);

  Status DropColumnFamilyImpl(ColumnFamilyHandle* column_family);
  Status DropColumnFamilyWithOptionsFile(ColumnFamilyHandle* column_family,
                                         uint64_t options_file_number,
                                         uint64_t options_file_size);

  // Delete any unneeded files and stale in-memory entries.
  void DeleteObsoleteFiles();
  // Delete obsolete files and log status and information of file deletion
  void DeleteObsoleteFileImpl(int job_id, const std::string& fname,
                              const std::string& path_to_sync, FileType type,
                              uint64_t number);
  // Returns true when a blob file must be preserved because it is still
  // tracked by blob direct write or is still footer-less on disk.
  bool ShouldKeepBlobFileDuringPurge(uint64_t number,
                                     const std::string& blob_file_path);

  // Background process needs to call
  //     auto x = CaptureCurrentFileNumberInPendingOutputs()
  //     auto file_num = versions_->NewFileNumber();
  //     <do something>
  //     ReleaseFileNumberFromPendingOutputs(x)
  // This will protect any file with number `file_num` or greater from being
  // deleted while <do something> is running.
  // -----------
  // This function will capture current file number and append it to
  // pending_outputs_. This will prevent any background process to delete any
  // file created after this point.
  std::list<uint64_t>::iterator CaptureCurrentFileNumberInPendingOutputs();
  // This function should be called with the result of
  // CaptureCurrentFileNumberInPendingOutputs(). It then marks that any file
  // created between the calls CaptureCurrentFileNumberInPendingOutputs() and
  // ReleaseFileNumberFromPendingOutputs() can now be deleted (if it's not live
  // and blocked by any other pending_outputs_ calls)
  void ReleaseFileNumberFromPendingOutputs(
      std::unique_ptr<std::list<uint64_t>::iterator>& v);

  // Rolls back one prepared file ingestion (delete its staged files, release
  // the reserved file numbers)
  void RollbackPreparedFileIngestion(FileIngestionHandleImpl* const h);

  // Similar to pending_outputs, preserve OPTIONS file. Used for remote
  // compaction.
  std::list<uint64_t>::iterator CaptureOptionsFileNumber();
  void ReleaseOptionsFileNumber(
      std::unique_ptr<std::list<uint64_t>::iterator>& v);

  // Sets bg error if there is an error writing to WAL.
  IOStatus SyncClosedWals(const WriteOptions& write_options,
                          JobContext* job_context, VersionEdit* synced_wals,
                          bool error_recovery_in_prog);

  // Flush the in-memory write buffer to storage.  Switches to a new
  // log-file/memtable and writes a new descriptor iff successful. Then
  // installs a new super version for the column family.
  Status FlushMemTableToOutputFile(ColumnFamilyData* cfd,
                                   const MutableCFOptions& mutable_cf_options,
                                   bool* madeProgress, JobContext* job_context,
                                   FlushReason flush_reason,
                                   SuperVersionContext* superversion_context,
                                   LogBuffer* log_buffer,
                                   Env::Priority thread_pri);

  // Flush the memtables of (multiple) column families to multiple files on
  // persistent storage.
  Status FlushMemTablesToOutputFiles(
      const autovector<BGFlushArg>& bg_flush_args, bool* made_progress,
      JobContext* job_context, LogBuffer* log_buffer, Env::Priority thread_pri);

  Status AtomicFlushMemTablesToOutputFiles(
      const autovector<BGFlushArg>& bg_flush_args, bool* made_progress,
      JobContext* job_context, LogBuffer* log_buffer, Env::Priority thread_pri);

  // REQUIRES: log_numbers are sorted in ascending order
  // corrupted_wal_found is set to true if we recover from a corrupted log file.
  Status RecoverLogFiles(const std::vector<uint64_t>& log_numbers,
                         SequenceNumber* next_sequence, bool read_only,
                         bool is_retry, bool* corrupted_wal_found,
                         RecoveryContext* recovery_ctx);

  Status SetNextWALIndexAfterRecovery(uint64_t wal_index);
  IOStatus WriteWALIndexSupersessions(
      const WriteOptions& write_options, log::Writer* new_log,
      const std::vector<WALIndexSupersession>& supersessions);
  // With more than one partition, writes the index floor into the head of
  // every WAL Open created whose predecessor span carries none.
  IOStatus WriteWALIndexHeadFloors(const WriteOptions& write_options);

  void SetupLogFilesRecovery(
      const std::vector<uint64_t>& wal_numbers,
      std::unordered_map<int, VersionEdit>* version_edits, int* job_id,
      uint64_t* min_wal_number);

  Status ProcessLogFiles(const std::vector<uint64_t>& wal_numbers,
                         bool read_only, bool is_retry, uint64_t min_wal_number,
                         int job_id, SequenceNumber* next_sequence,
                         std::unordered_map<int, VersionEdit>* version_edits,
                         bool* corrupted_wal_found,
                         RecoveryContext* recovery_ctx,
                         std::vector<WALIndexSupersession>* supersessions);

  // Indexed-WAL recovery through the N-file merge: one ordered stream feeds
  // the same per-record path as ProcessLogFile. Predecessor chaining is
  // absent (track_and_verify_wals falls back to ProcessLogFiles), so this
  // takes no predecessor info.
  // WALs numbered below `merge_from` carry no index; they replay
  // sequentially, as ProcessLogFiles would, before the merge reads the rest.
  Status ProcessLogFilesMerged(
      const std::vector<uint64_t>& wal_numbers, uint64_t merge_from,
      bool read_only, bool is_retry, uint64_t min_wal_number, int job_id,
      SequenceNumber* next_sequence,
      std::unordered_map<int, VersionEdit>* version_edits,
      bool* corrupted_wal_found, RecoveryContext* recovery_ctx,
      std::vector<WALIndexSupersession>* supersessions);

  // Pre-replay peek at each WAL head: sets *unindexed_high to the highest live
  // WAL that carries no index (zero when none), and *indexed_low to the lowest
  // live WAL carrying a marker or indexed data (UINT64_MAX when none). Fails
  // fast on indexed data without a marker over an intact head; a damaged
  // head stays merge-side for the attestation path. Sets *superseding when a
  // head declares a supersession for a WAL in the set.
  Status CheckWALFilesCarryWALIndex(const std::vector<uint64_t>& wal_numbers,
                                    uint64_t min_wal_number, bool is_retry,
                                    uint64_t* unindexed_high,
                                    uint64_t* indexed_low, bool* superseding);

  Status ProcessLogFile(uint64_t wal_number, uint64_t min_wal_number,
                        bool is_retry, bool read_only, int job_id,
                        SequenceNumber* next_sequence,
                        bool* stop_replay_for_corruption,
                        bool* stop_replay_by_wal_filter,
                        uint64_t* corrupted_wal_number,
                        bool* corrupted_wal_found,
                        std::unordered_map<int, VersionEdit>* version_edits,
                        bool* flushed, PredecessorWALInfo& predecessor_wal_info,
                        std::vector<WALIndexSupersession>* supersessions);

  void SetupLogFileProcessing(uint64_t wal_number);

  // Opens the physical stream shared by every recovery-time WAL reader.
  Status OpenWALForRecovery(const std::string& fname, bool is_retry,
                            std::unique_ptr<SequentialFileReader>* file_reader);

  Status InitializeLogReader(uint64_t wal_number, bool is_retry,
                             std::string& fname,

                             bool stop_replay_for_corruption,
                             uint64_t min_wal_number,
                             const PredecessorWALInfo& predecessor_wal_info,
                             bool* const old_log_record,
                             Status* const reporter_status,
                             DBOpenLogRecordReadReporter* reporter,
                             std::unique_ptr<log::Reader>& reader);
  Status ProcessLogRecord(
      Slice record, const std::unique_ptr<log::Reader>& reader,
      const UnorderedMap<uint32_t, size_t>& running_ts_sz, uint64_t wal_number,
      const std::string& fname, bool read_only, int job_id,
      const std::function<void()>& logFileDropped,
      DBOpenLogRecordReadReporter* reporter, uint64_t* record_checksum,
      SequenceNumber* last_seqno_observed, SequenceNumber* next_sequence,
      bool* stop_replay_for_corruption, Status* status,
      bool* stop_replay_by_wal_filter,
      std::unordered_map<int, VersionEdit>* version_edits, bool* flushed);

  Status InitializeWriteBatchForLogRecord(
      Slice record, const std::unique_ptr<log::Reader>& reader,
      const UnorderedMap<uint32_t, size_t>& running_ts_sz, WriteBatch* batch,
      std::unique_ptr<WriteBatch>& new_batch, WriteBatch*& batch_to_use,
      uint64_t* record_checksum);

  void MaybeReviseStopReplayForCorruption(
      SequenceNumber sequence, SequenceNumber const* const next_sequence,
      bool* stop_replay_for_corruption);

  Status InsertLogRecordToMemtable(WriteBatch* batch_to_use,
                                   uint64_t wal_number,
                                   SequenceNumber* next_sequence,
                                   bool* has_valid_writes, bool read_only);

  Status MaybeWriteLevel0TableForRecovery(
      bool has_valid_writes, bool read_only, uint64_t wal_number, int job_id,
      SequenceNumber const* const next_sequence,
      std::unordered_map<int, VersionEdit>* version_edits, bool* flushed);

  Status HandleNonOkStatusOrOldLogRecord(
      uint64_t wal_number, SequenceNumber const* const next_sequence,
      Status status, const DBOpenLogRecordReadReporter& reporter,
      bool* old_log_record, bool* stop_replay_for_corruption,
      uint64_t* corrupted_wal_number, bool* corrupted_wal_found);

  Status UpdatePredecessorWALInfo(uint64_t wal_number,
                                  const SequenceNumber last_seqno_observed,
                                  const std::string& fname,
                                  PredecessorWALInfo& predecessor_wal_info);

  void FinishLogFileProcessing(const Status& status,
                               const SequenceNumber* next_sequence);

  // Return `Status::Corruption()` when `stop_replay_for_corruption == true` and
  // exits inconsistency between SST and WAL data
  Status MaybeHandleStopReplayForCorruptionForInconsistency(
      bool stop_replay_for_corruption, uint64_t corrupted_wal_number);

  Status MaybeFlushFinalMemtableOrRestoreActiveLogFiles(
      const std::vector<uint64_t>& wal_numbers, bool read_only, int job_id,
      bool flushed, std::unordered_map<int, VersionEdit>* version_edits,
      RecoveryContext* recovery_ctx);

  // Check that DB sequence number is not set back during recovery between
  // replaying of WAL files and between replaying of WriteBatches.
  Status CheckSeqnoNotSetBackDuringRecovery(SequenceNumber prev_next_seqno,
                                            SequenceNumber current_next_seqno);

  void FinishLogFilesRecovery(int job_id, const Status& status);
  // The following two methods are used to flush a memtable to
  // storage. The first one is used at database RecoveryTime (when the
  // database is opened) and is heavyweight because it holds the mutex
  // for the entire period. The second method WriteLevel0Table supports
  // concurrent flush memtables to storage.
  Status WriteLevel0TableForRecovery(int job_id, ColumnFamilyData* cfd,
                                     MemTable* mem, VersionEdit* edit);

  // Get the size of a log file and, if truncate is true, truncate the
  // log file to its actual size, thereby freeing preallocated space.
  // Return success even if truncate fails
  Status GetLogSizeAndMaybeTruncate(uint64_t wal_number, bool truncate,
                                    WalFileNumberSize* log);

  // Restore the live-file deques and wals_total_size_ after recovery.
  // It needs to run only when there's no flush during recovery
  // (e.g. avoid_flush_during_recovery=true). May also trigger flush
  // in case wals_total_size > max_total_wal_size.
  Status RestoreAliveLogFiles(const std::vector<uint64_t>& log_numbers);

  // num_bytes: for slowdown case, delay time is calculated based on
  //            `num_bytes` going through.
  Status DelayWrite(uint64_t num_bytes, WriteThread& write_thread,
                    const WriteOptions& write_options);

  // Begin stalling of writes to `wal_partition` when memory usage increases
  // beyond a certain threshold.
  Status WriteBufferManagerStallWrites(size_t wal_partition,
                                       WriteContext* write_context);

  // Whether `write_thread` is one of the per-partition main write queues
  // rather than a nonmem queue.
  bool IsMainWriteThread(const WriteThread& write_thread) {
    for (size_t p = 0; p < NumWalPartitions(); ++p) {
      if (&write_thread == &PartitionWriteThread(p)) {
        return true;
      }
    }
    return false;
  }

  // Memtable-switch gate, used only with more than one WAL partition. Leaders
  // of different partitions append and insert into the shared memtables
  // concurrently, so a leader that must switch memtables or WALs in
  // PreprocessWrite() first closes the gate and waits for the leaders already
  // past it to exit their groups; leaders arriving while it is closed wait at
  // it. Callers outside the write path exclude every leader by entering all
  // partitions' queues unbatched instead.
  //
  // A closing leader has not entered the gate itself, and a leader past the
  // gate never closes it, so no leader waits on itself.
  // REQUIRES: mutex_ held; it is released while waiting.
  void CloseMemTableSwitchGate();
  // REQUIRES: mutex_ held.
  void OpenMemTableSwitchGate();
  // Called by PreprocessWrite() after any switch, before it binds a WAL.
  // REQUIRES: mutex_ not held.
  void EnterMemTableSwitchGate();
  // Called once per group that entered, by whichever thread exits the group.
  void ExitMemTableSwitchGate();

  // Publishes a write group that allocated [first_sequence, last_sequence]
  // on `wal_partition` (first_sequence > last_sequence when it consumed none),
  // once its memtable insert is done. At one partition that publishes
  // last_sequence. With more than one, groups finish out of sequence order,
  // so the group only marks its range applied and publishes the contiguous
  // applied prefix (PublishAppliedPrefix).
  void PublishLastSequence(size_t wal_partition, SequenceNumber first_sequence,
                           SequenceNumber last_sequence, bool wal_record) {
    if (NumWalPartitions() == 1) {
      versions_->SetLastSequence(last_sequence);
    } else {
      PublishAppliedPrefix(wal_partition, first_sequence, last_sequence,
                           wal_record);
    }
  }

  // N>1 only. Marks the group's range applied in applied_tracker_ and raises
  // LastSequence() to the applied frontier: the highest sequence at or below
  // which every allocated range, on any partition, has been inserted. A range
  // with a WAL record is marked applied only after its append completed, so
  // the frontier is also at or below the WAL durability safe point for every
  // sequence that has a record; ranges without one (disableWAL) are bounded
  // only by the ranges below them. Never lowers LastSequence(), which
  // ingestion and recovery may have raised past the frontier.
  // REQUIRES: the group was registered by AllocateSequenceAndWALIndex.
  void PublishAppliedPrefix(size_t wal_partition, SequenceNumber first_sequence,
                            SequenceNumber last_sequence, bool wal_record);

  // N>1 only. Raises LastSequence() to the applied frontier, never into an
  // open gate fence.
  // REQUIRES: sequence_wal_index_mutex_ held.
  void PublishAppliedFrontierLocked();

  // N>1, debug only. For code holding every write queue (or the closed
  // memtable switch gate) before it assigns or stamps LastSequence() + 1:
  // every group publishes before it leaves its queue, so no live write holds
  // a sequence above LastSequence(). Only a range a failure pinned stays
  // unpublished, and it comes with a background error.
  // REQUIRES: mutex_ held.
  void AssertLastSequenceCoversAllocated();

  Status ThrottleLowPriWritesIfNeeded(const WriteOptions& write_options,
                                      WriteBatch* my_batch);

  // REQUIRES: mutex locked and in write thread.
  Status ScheduleFlushes(WriteContext* context);

  void MaybeFlushStatsCF(autovector<ColumnFamilyData*>* cfds);

  Status TrimMemtableHistory(WriteContext* context);

  // Switches the current live memtable to immutable/read-only memtable.
  // A new WAL is created if the current WAL is not empty.
  // If `new_imm` is not nullptr, it will be added as the newest immutable
  // memtable, if and only if OK status is returned.
  // `last_seqno` needs to be provided if `new_imm` is not nullptr. It is
  // the value of versions_->LastSequence() after the write that ingests new_imm
  // is done.
  //
  // REQUIRES: mutex_ is held
  // REQUIRES: this thread is currently at the front of the writer queue
  // REQUIRES: this thread is currently at the front of the 2nd writer queue if
  // two_write_queues_ is true (This is to simplify the reasoning.)
  Status SwitchMemtable(ColumnFamilyData* cfd, WriteContext* context,
                        ReadOnlyMemTable* new_imm = nullptr,
                        SequenceNumber last_seqno = 0);

  // `refresh_wbm_accounting` can be false when a caller switches a batch of
  // memtables and refreshes the DB-wide accounting once after the batch.
  Status SwitchMemtableImpl(ColumnFamilyData* cfd, WriteContext* context,
                            ReadOnlyMemTable* new_imm,
                            SequenceNumber last_seqno,
                            bool refresh_wbm_accounting);

  // REQUIRES: mutex_ is held
  void UpdateWBMAccountingAfterMemtableSwitch(bool refresh_now);

  // Select and output column families qualified for atomic flush in
  // `selected_cfds`. If `provided_candidate_cfds` is non-empty, it will be used
  // as candidate CFs to select qualified ones from. Otherwise, all column
  // families are used as candidate to select from.
  //
  // REQUIRES: mutex held
  void SelectColumnFamiliesForAtomicFlush(
      autovector<ColumnFamilyData*>* selected_cfds,
      const autovector<ColumnFamilyData*>& provided_candidate_cfds = {},
      FlushReason flush_reason = FlushReason::kOthers);

  // Force current memtable contents to be flushed.
  Status FlushMemTable(ColumnFamilyData* cfd, const FlushOptions& options,
                       FlushReason flush_reason,
                       bool entered_write_thread = false);

  // Atomic-flush memtables from quanlified CFs among `provided_candidate_cfds`
  // (if non-empty) or amomg all column families and atomically record the
  // result to the MANIFEST.
  Status AtomicFlushMemTables(
      const FlushOptions& options, FlushReason flush_reason,
      const autovector<ColumnFamilyData*>& provided_candidate_cfds = {},
      bool entered_write_thread = false);

  enum class WriteThreadJoinMode { kBlocking, kNonBlocking };

  Status FlushMemTableImpl(ColumnFamilyData* cfd, const FlushOptions& options,
                           FlushReason flush_reason, bool entered_write_thread,
                           WriteThreadJoinMode join_mode, bool* made_progress);

  Status AtomicFlushMemTablesImpl(
      const FlushOptions& options, FlushReason flush_reason,
      const autovector<ColumnFamilyData*>& provided_candidate_cfds,
      bool entered_write_thread, WriteThreadJoinMode join_mode,
      bool* made_progress);

  // REQUIRES: mutex locked and write queues drained up to the recovery flush
  // fence that is about to switch memtables.
  void MaybeSyncLastSequenceWithAllocatedForRecovery(FlushReason flush_reason);

  Status RetryFlushesForErrorRecovery(FlushReason flush_reason, bool wait);

  // Wait until flushing this column family won't stall writes
  Status WaitUntilFlushWouldNotStallWrites(ColumnFamilyData* cfd,
                                           bool* flush_needed);

  // Wait for memtable flushed.
  // If flush_memtable_id is non-null, wait until the memtable with the ID
  // gets flush. Otherwise, wait until the column family don't have any
  // memtable pending flush.
  // resuming_from_bg_err indicates whether the caller is attempting to resume
  // from background error.
  // If wait_for_listener_notifications is true, also wait until the
  // EventListener::OnFlushCompleted callbacks for the flushed memtables have
  // finished running (see FlushOptions::listener_wait).
  Status WaitForFlushMemTable(
      ColumnFamilyData* cfd, const uint64_t* flush_memtable_id = nullptr,
      bool resuming_from_bg_err = false,
      std::optional<FlushReason> flush_reason = std::nullopt,
      bool wait_for_listener_notifications = false) {
    return WaitForFlushMemTables({cfd}, {flush_memtable_id},
                                 resuming_from_bg_err, flush_reason,
                                 wait_for_listener_notifications);
  }
  // Wait for memtables to be flushed for multiple column families.
  Status WaitForFlushMemTables(
      const autovector<ColumnFamilyData*>& cfds,
      const autovector<const uint64_t*>& flush_memtable_ids,
      bool resuming_from_bg_err, std::optional<FlushReason> flush_reason,
      bool wait_for_listener_notifications = false);

  inline void WaitForPendingWrites() {
    mutex_.AssertHeld();
    TEST_SYNC_POINT("DBImpl::WaitForPendingWrites:BeforeBlock");
    // In case of pipelined write is enabled, wait for all pending memtable
    // writers.
    if (immutable_db_options_.enable_pipelined_write) {
      // Memtable writers may call DB::Get in case max_successive_merges > 0,
      // which may lock mutex. Unlocking mutex here to avoid deadlock.
      mutex_.Unlock();
      for (size_t p = 0; p < NumWalPartitions(); ++p) {
        PartitionWriteThread(p).WaitForMemTableWriters();
      }
      mutex_.Lock();
    }

    if (immutable_db_options_.unordered_write) {
      // Wait for the ones who already wrote to the WAL to finish their
      // memtable write.
      if (pending_memtable_writes_.load() != 0) {
        // XXX: suspicious wait while holding DB mutex?
        std::unique_lock<std::mutex> guard(switch_mutex_);
        switch_cv_.wait(guard,
                        [&] { return pending_memtable_writes_.load() == 0; });
      }
    } else {
      // (Writes are finished before the next write group starts.)
    }

    // Wait for any LockWAL to clear
    while (!lock_wal_owner_thread_id_counts_.empty()) {
      bg_cv_.Wait();
    }
  }

  // TaskType is used to identify tasks in thread-pool, currently only
  // differentiate manual compaction, which could be unscheduled from the
  // thread-pool.
  enum class TaskType : uint8_t {
    kDefault = 0,
    kManualCompaction = 1,
    kCount = 2,
  };

  // Task tag is used to identity tasks in thread-pool, which is
  // dbImpl obj address + type
  inline void* GetTaskTag(TaskType type) {
    return GetTaskTag(static_cast<uint8_t>(type));
  }

  inline void* GetTaskTag(uint8_t type) {
    return static_cast<uint8_t*>(static_cast<void*>(this)) + type;
  }

  // Checks DB-local write queues; manager-wide stall state would reject healthy
  // DBs. REQUIRES: mutex_ held.
  bool WouldBlockJoiningWriteThread() {
    mutex_.AssertHeld();
    for (size_t p = 0; p < NumWalPartitions(); ++p) {
      if (PartitionWriteThread(p).GetBegunCountOfOutstandingStall() != 0 ||
          (two_write_queues_ &&
           NonmemWriteThread(p).GetBegunCountOfOutstandingStall() != 0)) {
        return true;
      }
    }
    return false;
  }

  class AllPartitionsWriter;
  Status EnterWriteThreadForNonBlockingFlush(AllPartitionsWriter* w,
                                             AllPartitionsWriter* nonmem_w);

  // REQUIRES: mutex locked and in write thread.
  void AssignAtomicFlushSeq(const autovector<ColumnFamilyData*>& cfds);

  // REQUIRES: mutex locked and in write thread.
  Status SwitchWAL(WriteContext* write_context);

  // REQUIRES: mutex locked and in write thread.
  Status HandleWriteBufferManagerFlush(WriteContext* write_context);

  // Column families and memory eligible for a WriteBufferManager flush.
  struct FlushableCFs {
    ColumnFamilyData* largest = nullptr;  // most reclaimable memtable memory
    ColumnFamilyData* oldest = nullptr;   // smallest memtable creation seq
    size_t largest_mem = 0;
    size_t total_mem = 0;
  };

  // Shared selection for bidding and flushing. REQUIRES: mutex_ held.
  FlushableCFs CollectFlushableCFs(bool include_waiting_immutable);

  // Registers an opened read-write DB without holding mutex_.
  void MaybeRegisterFlushInitiator();

  // Refreshes the published mutable-memory counters after memtables change.
  // REQUIRES: mutex_ held.
  void RefreshFlushableMemAccounting();

  bool TryRefreshFlushableMemAccounting();

  // Queues one WBM flush; false lets the coordinator try another DB.
  bool ScheduleWriteBufferManagerFlush();

  void BackgroundCallWBMFlush();

  // REQUIRES: mutex locked
  Status PreprocessWrite(const WriteOptions& write_options,
                         size_t wal_partition, WalContext* log_context,
                         WriteContext* write_context);

  // Merge write batches in the write group into merged_batch.
  // Returns OK if merge is successful.
  // Returns Corruption if corruption in write batch is detected.
  // When `ordered_writers` is non-null the batches merge in that order
  // instead of the group's link order; it must hold exactly the group's
  // writers. Only an N>1 sync-led group with async members passes one, to
  // lay the async members' keys first so they take the lower sequences.
  Status MergeBatch(
      const WriteThread::WriteGroup& write_group, WriteBatch* tmp_batch,
      WriteBatch** merged_batch, size_t* write_with_wal,
      WriteBatch** to_be_cached_state,
      const std::vector<WriteThread::Writer*>* ordered_writers = nullptr);

  // Reserves a sequence block and, when a WAL record will be written, its
  // wal_index in one DB-wide critical section. The mutex is released before
  // batch merging, WAL append, or any other I/O. The durability watermark's
  // Register runs in this same section when register_durability is set, so
  // no later scan can miss the outstanding range.
  //
  // Rule: a failed append leaves its entry in pending_low[p], and the cover
  // for the burned index must be written and synced before that entry may be
  // popped. Popping first would advance the watermark past an uncovered hole
  // and acknowledge writes above it. With one WAL at a time (the only layout
  // today) no extra sync is needed -- appends to one file are ordered, so
  // the cover inherits the durability of whatever is synced after it.
  //
  // `split_applied_ranges`, set only by an N>1 sync-led group with async
  // members, registers the applied range as two halves: the first
  // `applied_first_count` sequences and the rest (either half may be empty).
  // The group publishes the async members' lower half before its fsync and
  // the sync members' higher half after. Both halves must be completed (or
  // pinned for recovery). The durability range stays whole: the group is
  // still one WAL record.
  IOStatus AllocateSequenceAndWALIndex(size_t sequence_count, bool write_wal,
                                       SequenceNumber* last_sequence_before,
                                       uint64_t* wal_index,
                                       size_t wal_partition,
                                       bool register_durability,
                                       bool split_applied_ranges = false,
                                       size_t applied_first_count = 0);

  // Writes a void record covering `wal_index`, which was allocated but whose
  // data record never reached the WAL. Every allocated index must end up
  // either on a data record or under a cover, so that a later gap check can
  // tell a deliberate hole from a lost record and does not truncate on it.
  // No-op when `wal_index` is 0, meaning nothing was consumed.
  //
  // A failed cover leaves the index uncovered, and no write above it may then
  // be acknowledged. See the implementation for why that holds today.
  //
  // Taking a single writer means the cover can only go to the one that just
  // failed. That is the whole story with one WAL at a time, but under
  // partitioning a poisoned writer for partition p may want to fall back to a
  // healthy partition, so this signature needs revisiting then -- unless the
  // durability watermark rule above makes the fallback unnecessary.
  // Takes WalWriteMutex(wal_partition) unless `wal_mutex_held`, whenever
  // WriteToWAL would take it.
  IOStatus CoverBurnedWALIndex(const WriteOptions& write_options,
                               log::Writer* log_writer, uint64_t wal_index,
                               size_t wal_partition, bool wal_mutex_held);

  // Binds `wal_context` to the WAL that writes should be appended to, and to
  // that WAL's file-number/size accounting.
  //
  // This remains the single seam that binds a resolved route to a WAL. The
  // route is resolved once, before group formation, and carried here by the
  // group leader.
  //
  // REQUIRES: WalWriteMutex(wal_partition) held. See the contracts on
  // CurrentWal() and CurrentAliveWal().
  void SelectWalForWrite(size_t wal_partition, WalContext* wal_context);

  // `wal_context` names the WAL these writes go to and carries the sync
  // requirements for it. It is populated by PreprocessWrite().
  IOStatus WriteToWAL(const WriteBatch& merged_batch,
                      const WriteOptions& write_options,
                      WalContext& wal_context, uint64_t* wal_used,
                      uint64_t* log_size, SequenceNumber sequence,
                      uint64_t wal_index, size_t sequence_count,
                      bool wal_mutex_held);

  IOStatus WriteGroupToWAL(
      const WriteThread::WriteGroup& write_group, WalContext& wal_context,
      uint64_t* wal_used, SequenceNumber sequence, uint64_t wal_index,
      size_t sequence_count, bool complete_durability,
      const std::vector<WriteThread::Writer*>* ordered_writers = nullptr);

  IOStatus ConcurrentWriteGroupToWAL(const WriteThread::WriteGroup& write_group,
                                     uint64_t* wal_used,
                                     SequenceNumber* last_sequence,
                                     size_t seq_inc);

  // Blocks until LastSequence() covers `target` and, for a group with a WAL
  // record (`wal_record`), the durability frontier covers it too, i.e. every
  // WAL record at or below it completed. Its only other exits: the status of
  // a gate fence holding `fence_target` (a range at or below it failed, so
  // it can never be acknowledged, even once recovery covers it), and
  // ShutdownInProgress once shutdown begins while still uncovered. Any other
  // background error leaves it waiting, as one WAL leaves a group whose
  // append succeeded. `target` and `fence_target` differ for a sync-led
  // group: coverage waits on the published async half's high (waiting on the
  // group's own low would park a pure-sync group on its unpublished self),
  // while a fence holds the group's own low (a write at or above a failure
  // fails even once covered). Called without holding any mutex; takes
  // sequence_wal_index_mutex_ only to check fences and park, never together
  // with mutex_.
  Status WaitForWalDurability(SequenceNumber target, bool wal_record,
                              SequenceNumber fence_target);

  // Wakes durability waiters after the frontier advanced. Call with
  // sequence_wal_index_mutex_ held, which every Complete site already holds;
  // the waiter-count check makes the no-waiter case a single relaxed load.
  void SignalWalDurabilityWaiters() {
    if (wal_durability_waiters_.load(std::memory_order_relaxed) > 0) {
      wal_durability_cv_.SignalAll();
    }
  }

  // Wakes durability waiters while a background error is recorded: on each
  // publication that stores an error or repeats the recorded severity.
  // Called only from ErrorHandler (CheckAndSetRecoveryAndBGError, and
  // HandleKnownErrors for a repeat) with mutex_ held, never while bg_error_
  // is OK. Sets the flag, then signals under sequence_wal_index_mutex_.
  // Woken gates only recheck their exits; an error that fences no gate
  // (InstallWalGateFence) leaves them parked. Safe with no waiters; takes
  // no WAL or sync-mark mutex.
  void SignalWalDurabilityWaitersOnBGError();

  // Clears the durability park predicate's background-error leg, and
  // releases every range whose append failed (failed_wal_ranges_) from both
  // trackers: recovery succeeded, and those ranges hold no data. Before that,
  // closes every standing gate fence at the last allocated sequence, so a
  // write allocated above a released range still fails whenever it reaches
  // its gate. A fence on a range that consumed no sequences closes to an
  // empty range holding nothing, and records no indeterminate gap. Called
  // only from ErrorHandler::ClearBGError with mutex_ held, before it clears
  // the stopped flag, so no write admitted after recovery falls inside a
  // fence.
  void ClearWalDurabilityBGError();

  // N>1 only. Writes a void cover for every index in burned_wal_indices_ into
  // partition 0's current WAL, which recovery's rotation installed, and syncs
  // it and the WAL directory, so no later recovery reads the burned index as
  // an unexplained hole below writes admitted after it. Called by ResumeImpl,
  // with mutex_ held, before ClearBGError; releases mutex_ around the I/O. On
  // failure the indices stay pending and the DB stays write-stopped.
  Status ExplainBurnedWALIndices(const WriteOptions& write_options);

  // N>1 only. Fences every gate whose target is at or above `low`: a range
  // starting there failed (its append, its gate sync, or its memtable
  // insert), so nothing at or above it can ever be acknowledged (D-1). The
  // fence stays open-ended until recovery releases the range
  // (`released_by_recovery`; a failed insert's range is never released), and
  // holds `status` (the background error) for those gates to return. Wakes
  // parked gates.
  // REQUIRES: mutex_ held.
  void InstallWalGateFence(SequenceNumber low, const Status& status,
                           bool released_by_recovery);

  // The status of a fence holding `target`, or OK.
  // REQUIRES: sequence_wal_index_mutex_ held.
  Status WalGateFenceStatusLocked(SequenceNumber target) const;

  // As above, without the mutex when no fence can hold `target`.
  Status WalGateFenceStatus(SequenceNumber target) {
    if (target < wal_gate_fence_low_.load(std::memory_order_acquire) ||
        target > wal_gate_fence_high_.load(std::memory_order_acquire)) {
      return Status::OK();
    }
    InstrumentedMutexLock l(&sequence_wal_index_mutex_);
    return WalGateFenceStatusLocked(target);
  }

  // The lowest sequence an unreleased fence holds, or kMaxSequenceNumber.
  // Every memtable entry at or above it belongs to a write that can never be
  // acknowledged, so a flush drops it.
  // REQUIRES: mutex_ held.
  SequenceNumber UnreleasedWalGateFenceLow();

  // As above, with sequence_wal_index_mutex_ held instead.
  SequenceNumber OpenWalGateFenceLowLocked() const;

  // Runs a write group's completion gate: waits until the durability
  // frontier and publication cover `target` (see WaitForWalDurability), and
  // returns the outcome to merge into the status handed to followers.
  Status RunCompletionGate(SequenceNumber target);

  // N>1 only. A sync write's own fsync (D-9), run from its leadership after
  // its WAL append and before its memtable insert and publication, so the
  // write becomes visible only once durable: claims every partition, fsyncs
  // every live WAL, marks them synced and records them in the MANIFEST. A
  // failure stops the DB. Sets `*wal_horizon` to the highest allocated WAL
  // number, which the failure path reports so recovery retires every live
  // WAL. Does not record the background error; the caller pins its range.
  IOStatus SyncWalsFromCompletionGate(Env::IOPriority rate_limiter_priority,
                                      uint64_t* wal_horizon);

  // N>1 only. Runs a leadership gate's `sync`, which must cover every record
  // written before it starts, unless a gate sync that started after this
  // call, or (for a group that consumes sequences) one whose safe point read
  // covered `target`, already succeeded; while another gate's sync runs,
  // waits for it and checks again (group commit). Takes only
  // wal_gate_sync_mutex_, never across `sync`. A gate that queued on a sync
  // that fails returns that sync's status; the caller pins and fences its
  // own range, as for a failed append.
  IOStatus RunCoalescedGateSync(SequenceNumber low, SequenceNumber target,
                                const std::function<IOStatus()>& sync);

  // Fsyncs every partition's live WALs, each partition under its own WAL
  // mutex above one partition (all N with manual flush); at one partition
  // without manual flush it takes none. Stops at the first failure, naming
  // its WAL in `*failed_wal_number`; counts the partitions fsynced.
  // `previously_failed` as for SyncWalPartitionFiles.
  IOStatus SyncEveryWalPartition(const WriteOptions& write_options,
                                 uint64_t* failed_wal_number,
                                 uint64_t* fsynced_partitions,
                                 bool* previously_failed = nullptr);

  // Used by WriteImpl to update bg_error_ if paranoid check is enabled.
  // Caller must hold mutex_.
  void WriteStatusCheckOnLocked(const Status& status);

  // Used by WriteImpl to update bg_error_ if paranoid check is enabled.
  void WriteStatusCheck(const Status& status);

  // Used by WriteImpl to update bg_error_ when IO error happens, e.g., write
  // WAL, sync WAL fails, if paranoid check is enabled.
  // `pinned_wal_partition`, when not kNoPinnedWalRange, is the partition on
  // which the failed group registered [failed_wal_sequence_start,
  // failed_wal_sequence] for N>1 durability and publication; the range is
  // kept in failed_wal_ranges_ until recovery clears the error.
  // `pinned_split_at`, when not kMaxSequenceNumber, additionally pins each
  // applied half of a split group whose append failed before either half
  // published ([start, pinned_split_at] and [pinned_split_at + 1, seq]).
  static constexpr size_t kNoPinnedWalRange = SIZE_MAX;
  void WALIOStatusCheck(const IOStatus& status, uint64_t failed_wal_number = 0,
                        SequenceNumber failed_wal_sequence = 0,
                        SequenceNumber failed_wal_sequence_start = 0,
                        size_t pinned_wal_partition = kNoPinnedWalRange,
                        SequenceNumber pinned_split_at = kMaxSequenceNumber);

  // Extends the inclusive WAL-number cutoff that must not receive more I/O.
  // This is lock-free because a WAL error can race with public SyncWAL().
  void ExtendWALRecoveryCutoff(uint64_t failed_wal_number);

  // Used by WriteImpl to update bg_error_ in case of memtable insert error.
  // At N>1 the group's range, starting at `first_sequence`, stays
  // unpublished for good, so it also fences every gate at or above it.
  void HandleMemTableInsertFailure(const Status& nonok_memtable_insert_status,
                                   SequenceNumber first_sequence);

  Status CompactFilesImpl(const CompactionOptions& compact_options,
                          ColumnFamilyData* cfd, Version* version,
                          const std::vector<std::string>& input_file_names,
                          std::vector<std::string>* const output_file_names,
                          const int output_level, int output_path_id,
                          JobContext* job_context, LogBuffer* log_buffer,
                          CompactionJobInfo* compaction_job_info);

  // Helper function to perform trivial move by updating manifest metadata
  // without rewriting data files. This is called when IsTrivialMove() is true.
  // REQUIRES: mutex held
  // Populates the Compaction's edit with DeleteFile/AddFile entries for
  // a trivial move (no data rewriting). Does not commit the edit.
  void PrepareTrivialMoveEdit(Compaction& c, LogBuffer* log_buffer,
                              size_t& moved_files, size_t& moved_bytes);

  // REQUIRES: mutex held, PrepareTrivialMoveEdit already called
  // Commits the trivial move by calling LogAndApply on the prepared edit.
  // Returns: Status of the manifest write
  Status CommitTrivialMove(Compaction& c, bool& compaction_released);

  // REQUIRES: mutex unlocked
  void TrackOrUntrackFiles(const std::vector<std::string>& existing_data_files,
                           bool track);

  void MaybeScheduleFlushOrCompaction();

  BackgroundJobPressure CaptureBackgroundJobPressure() const;
  void NotifyOnBackgroundJobPressureChanged();

  struct FlushRequest {
    FlushReason flush_reason;
    // Whether this flush request should use the atomic flush code path.
    bool atomic_flush = false;
    // A map from column family to flush to largest memtable id to persist for
    // each column family. Once all the memtables whose IDs are smaller than or
    // equal to this per-column-family specified value, this flush request is
    // considered to have completed its work of flushing this column family.
    // After completing the work for all column families in this request, this
    // flush is considered complete. EnqueuePendingFlush() acquires one
    // reference for each CFD when it successfully queues this request.
    // PopFirstFromFlushQueue() transfers responsibility for those references
    // to its caller, which must release each one with UnrefAndTryDelete() after
    // processing or discarding the request.
    std::unordered_map<ColumnFamilyData*, uint64_t>
        cfd_to_max_mem_id_to_persist;

#ifndef NDEBUG
    int reschedule_count = 1;
#endif /* !NDEBUG */
  };

  // In case of atomic flush, generates a `FlushRequest` for the latest atomic
  // cuts for these `cfds`. Atomic cuts are recorded in
  // `AssignAtomicFlushSeq()`. For each entry in `cfds`, all CFDs sharing the
  // same latest atomic cut must also be present.
  //
  // REQUIRES: mutex held
  void GenerateFlushRequest(const autovector<ColumnFamilyData*>& cfds,
                            FlushReason flush_reason, FlushRequest* req);

  // Below functions are for executing flush, compaction in the background. A
  // dequeue is the communication channel between threads that asks for the work
  // to be done and the available threads in the thread pool that pick it up to
  // execute it. We use these terminologies to describe the state of the work
  // and its transitions:
  // 1) It becomes pending once it's successfully enqueued into the
  //    corresponding dequeue, a work in this state is also called unscheduled.
  //    Counter `unscheduled_*_` counts work in this state.
  // 2) When `MaybeScheduleFlushOrCompaction` schedule a thread to run `BGWork*`
  //    for the work, it becomes scheduled
  //    Counter `bg_*_scheduled_` counts work in this state.
  // 3) Once the thread start to execute `BGWork*`, the work is popped from the
  //    dequeue, it is now in running state
  //    Counter `num_running_*_` counts work in this state.
  // 4) Eventually, the work is finished. We don't need to specifically track
  //    finished work.

  // Returns true if `req` is successfully enqueued.
  bool EnqueuePendingFlush(const FlushRequest& req);

  void EnqueuePendingCompaction(ColumnFamilyData* cfd);
  void SchedulePendingPurge(std::string fname, std::string dir_to_sync,
                            FileType type, uint64_t number, int job_id);
  static void BGWorkCompaction(void* arg);
  // Runs a pre-chosen universal compaction involving bottom level in a
  // separate, bottom-pri thread pool.
  static void BGWorkBottomCompaction(void* arg);
  static void BGWorkFlush(void* arg);
  static void BGWorkPurge(void* arg);
  static void UnscheduleCompactionCallback(void* arg);
  static void UnscheduleFlushCallback(void* arg);
  void BackgroundCallCompaction(PrepickedCompaction* prepicked_compaction,
                                Env::Priority thread_pri);
  void BackgroundCallFlush(Env::Priority thread_pri);
  void BackgroundCallPurge();
  // Recursively removes all children of the DB session temporary directory.
  // Best effort: errors are logged, not returned. REQUIRES: mutex_ not held.
  void CleanupSessionTmpDir();
  Status BackgroundCompaction(bool* madeProgress, JobContext* job_context,
                              LogBuffer* log_buffer,
                              PrepickedCompaction* prepicked_compaction,
                              Env::Priority thread_pri);
  Status BackgroundFlush(bool* madeProgress, JobContext* job_context,
                         LogBuffer* log_buffer, FlushReason* reason,
                         bool* flush_rescheduled_to_retain_udt,
                         Env::Priority thread_pri);

  Compaction* CreateIntendedCompactionForwardedToBottomPriorityPool(
      Compaction* c);
  bool EnoughRoomForCompaction(ColumnFamilyData* cfd,
                               const std::vector<CompactionInputFiles>& inputs,
                               bool* sfm_bookkeeping, LogBuffer* log_buffer);

  // Request compaction tasks token from compaction thread limiter.
  // It always succeeds if force = true or limiter is disable.
  bool RequestCompactionToken(ColumnFamilyData* cfd, bool force,
                              std::unique_ptr<TaskLimiterToken>* token,
                              LogBuffer* log_buffer);

  // Return true if the `FlushRequest` can be rescheduled to retain the UDT.
  // Only true if there are user-defined timestamps in the involved MemTables
  // with newer than cutoff timestamp `full_history_ts_low` and not flushing
  // immediately will not cause entering write stall mode.
  bool ShouldRescheduleFlushRequestToRetainUDT(const FlushRequest& flush_req);

  // Schedule background tasks
  Status StartPeriodicTaskScheduler();

  // Compute the repeat period for the kTriggerCompaction task, which ensures
  // compactions not dependent on writes (flushes) are eventually triggered when
  // there are no writes (flushes). NOT thread safe; only called during DB open
  // (StartPeriodicTaskScheduler). KNOWN LIMITATION: doesn't get updated with
  // dynamic option updates. (Probably not worth the extra complexity.)
  uint64_t ComputeTriggerCompactionPeriod();

  // Cancel scheduled periodic tasks
  Status CancelPeriodicTaskScheduler();

  Status RegisterRecordSeqnoTimeWorker();

  void PrintStatistics();

  size_t EstimateInMemoryStatsHistorySize() const;

  // Return the minimum empty level that could hold the total data in the
  // input level. Return the input level, if such level could not be found.
  int FindMinimumEmptyLevelFitting(ColumnFamilyData* cfd,
                                   const MutableCFOptions& mutable_cf_options,
                                   int level);

  // Move the files in the input level to the target level.
  // If target_level < 0, automatically calculate the minimum level that could
  // hold the data set.
  Status ReFitLevel(ColumnFamilyData* cfd, int level, int target_level = -1);

  // helper functions for adding and removing from flush & compaction queues
  void AddToCompactionQueue(ColumnFamilyData* cfd);
  ColumnFamilyData* PopFirstFromCompactionQueue();
  FlushRequest PopFirstFromFlushQueue();

  struct CompactionQueueThrottled {};
  struct CompactionQueueParked {};
  struct CompactionQueuePicked {
    ColumnFamilyData* cfd;
  };
  using CompactionQueuePickResult =
      std::variant<CompactionQueuePicked, CompactionQueueThrottled,
                   CompactionQueueParked>;

  // Pick the first unthrottled compaction with task token from queue.
  CompactionQueuePickResult PickCompactionFromQueue(
      std::unique_ptr<TaskLimiterToken>* token, LogBuffer* log_buffer);
  bool IsCompactionAborted(ColumnFamilyData* cfd) const;
  bool MustWaitForCompaction(ColumnFamilyData* cfd);
  bool HasPendingManualCompaction(ColumnFamilyData* cfd);
  bool RestoreParkedCompaction(ColumnFamilyData* cfd);
  void BeginInFlightCompaction(ColumnFamilyData* cfd);
  bool EndInFlightCompaction(ColumnFamilyData* cfd);

  IOStatus SyncWalImpl(bool include_current_wal,
                       const WriteOptions& write_options,
                       JobContext* job_context, VersionEdit* synced_wals,
                       bool error_recovery_in_prog);

  // helper function to call after some of the WALs were synced
  // Finishes only the partitions flagged in claimed (the caller's record of
  // the partitions its claim marked: SyncWalImpl's, or a sync write's
  // WalContext::wal_sync_claimed).
  // REQUIRES: all N WAL write mutexes held (see LockWalWriteMutexes()).
  void MarkLogsSynced(uint64_t up_to, bool synced_dir, VersionEdit* edit,
                      const autovector<uint8_t, 1>& claimed);
  Status ApplyWALToManifest(const ReadOptions& read_options,
                            const WriteOptions& write_options,
                            VersionEdit* edit);
  // WALs with log number up to up_to are not synced successfully.
  // claimed is as for MarkLogsSynced.
  // REQUIRES: all N WAL write mutexes held (see LockWalWriteMutexes()).
  void MarkLogsNotSynced(uint64_t up_to, const autovector<uint8_t, 1>& claimed);

  SnapshotImpl* GetSnapshotImpl(bool is_write_conflict_boundary,
                                bool lock = true);

  // If snapshot_seq != kMaxSequenceNumber, then this function can only be
  // called from the write thread that publishes sequence numbers to readers.
  // For 1) write-committed, or 2) write-prepared + one-write-queue, this will
  // be the write thread performing memtable writes. For write-prepared with
  // two write queues, this will be the write thread writing commit marker to
  // the WAL.
  // If snapshot_seq == kMaxSequenceNumber, this function is called by a caller
  // ensuring no writes to the database.
  std::pair<Status, std::shared_ptr<const SnapshotImpl>>
  CreateTimestampedSnapshotImpl(SequenceNumber snapshot_seq, uint64_t ts,
                                bool lock = true);

  uint64_t GetMaxTotalWalSize() const;

  FSDirectory* GetDataDir(ColumnFamilyData* cfd, size_t path_id) const;

  Status MaybeReleaseTimestampedSnapshotsAndCheck();

  Status MaybeWriteWalMarkersToManifestOnClose();

  Status CloseHelper();

  void WaitForBackgroundWork();

  // Background threads call this function, which is just a wrapper around
  // the InstallSuperVersion() function. Background threads carry
  // sv_context to allow allocation of SuperVersion object outside of holding
  // the DB mutex.
  // All ColumnFamily state changes go through this function. Here we analyze
  // the new state and we schedule background work if we detect that the new
  // state needs flush or compaction.
  // See also InstallSuperVersionForConfigChange().
  void InstallSuperVersionAndScheduleWork(
      ColumnFamilyData* cfd, SuperVersionContext* sv_context,
      std::optional<std::shared_ptr<SeqnoToTimeMapping>>
          new_seqno_to_time_mapping = {});

  // A variant of InstallSuperVersionAndScheduleWork() that must be used for
  // new CFs or for changes to mutable_cf_options. This is so that it can
  // update seqno_to_time_mapping cached for the new SuperVersion as relevant.
  void InstallSuperVersionForConfigChange(ColumnFamilyData* cfd,
                                          SuperVersionContext* sv_context);

  bool GetIntPropertyInternal(ColumnFamilyData* cfd,
                              const DBPropertyInfo& property_info,
                              bool is_locked, uint64_t* value);
  bool GetPropertyHandleOptionsStatistics(std::string* value);

  bool HasPendingManualCompaction();
  bool HasExclusiveManualCompaction();
  void AddManualCompaction(ManualCompactionState* m);
  void RemoveManualCompaction(ManualCompactionState* m);
  bool ShouldntRunManualCompaction(ManualCompactionState* m);
  bool HaveManualCompaction(ColumnFamilyData* cfd);
  bool MCOverlap(ManualCompactionState* m, ManualCompactionState* m1);
  void UpdateFIFOCompactionStatus(const std::unique_ptr<Compaction>& c);

  // May open and read table files for table property.
  // Should not be called while holding mutex_.
  void BuildCompactionJobInfo(const ColumnFamilyData* cfd, Compaction* c,
                              const Status& st,
                              const CompactionJobStats& compaction_job_stats,
                              const int job_id,
                              CompactionJobInfo* compaction_job_info) const;
  // Reserve the next 'num' file numbers for to-be-ingested external SST files,
  // and return the current file_number in 'next_file_number'.
  // Write a version edit to the MANIFEST.
  Status ReserveFileNumbersBeforeIngestion(
      ColumnFamilyData* cfd, uint64_t num,
      std::unique_ptr<std::list<uint64_t>::iterator>& pending_output_elem,
      uint64_t* next_file_number);

  bool ShouldPurge(uint64_t file_number) const;
  void MarkAsGrabbedForPurge(uint64_t file_number);

  size_t GetWalPreallocateBlockSize(uint64_t write_buffer_size) const;
  Env::WriteLifeTimeHint CalculateWALWriteHint() { return Env::WLTH_SHORT; }

  // Returns true when async WAL precreation is enabled and compatible with the
  // active WAL strategy. WAL recycling already avoids file creation latency, so
  // precreation is disabled when recycle_log_file_num is non-zero.
  bool AsyncWALPrecreateEnabled() const;

  // A WAL file that has a reserved file number and may have an opened writer,
  // but has not been added to DBImpl's in-memory logical WAL tracking lists
  // (the WAL writer and live-file deques).
  struct UnpublishedWAL {
    uint64_t log_number = 0;
    std::unique_ptr<log::Writer> writer;

    UnpublishedWAL() = default;
    UnpublishedWAL(const UnpublishedWAL&) = delete;
    UnpublishedWAL& operator=(const UnpublishedWAL&) = delete;

    UnpublishedWAL(UnpublishedWAL&& other) noexcept {
      *this = std::move(other);
    }
    UnpublishedWAL& operator=(UnpublishedWAL&& other) noexcept {
      if (this != &other) {
        log_number = other.log_number;
        writer = std::move(other.writer);
        other.Reset();
      }
      return *this;
    }

    ~UnpublishedWAL() = default;

    void Reset() {
      log_number = 0;
      writer.reset();
    }
  };

  // Reserves the next WAL file number and schedules a HIGH-priority background
  // task to precreate that WAL file. A precreated WAL is not a logical WAL
  // until a foreground WAL rotation consumes it.
  void MaybeScheduleAsyncWALPrecreate(size_t preallocate_block_size);

  // Background task for opening the reserved future WAL and publishing the
  // result under mutex_.
  static void BGWorkAsyncWALPrecreate(void* arg);

  // Waits for an in-flight async WAL precreation and returns a prepared WAL if
  // one is available. If precreation failed, returns an empty WAL and lets the
  // foreground rotation create the WAL synchronously. Caller must hold mutex_.
  UnpublishedWAL WaitForAsyncWALPrecreate();

  // Opens and preallocates a WAL writer without writing logical WAL records.
  // Used by async WAL precreation and by synchronous WAL creation.
  IOStatus CreateWALWriter(const DBOptions& db_options, uint64_t log_file_num,
                           uint64_t recycle_log_number,
                           size_t preallocate_block_size,
                           UnpublishedWAL* new_wal);

  // Starts an opened WAL file by writing the initial records required before it
  // can be installed as the current WAL for foreground writes.
  // `wal_index_predecessor` is the span the file attests for its partition's
  // previous WAL (log::MakeWALIndexPredecessorSpan); none when its wal_number
  // is zero or WAL indexing is off.
  IOStatus StartWALFile(
      const WriteOptions& write_options,
      const PredecessorWALInfo& predecessor_wal_info,
      const log::WALIndexPredecessorSpan& wal_index_predecessor,
      log::Writer* new_log);
  // `wal_partition` is recorded in the file's WAL index marker so recovery
  // can return the file to that partition.
  IOStatus CreateWAL(const DBOptions& db_options,
                     const WriteOptions& write_options, uint64_t log_file_num,
                     uint64_t recycle_log_number, size_t preallocate_block_size,
                     const PredecessorWALInfo& predecessor_wal_info,
                     const log::WALIndexPredecessorSpan& wal_index_predecessor,
                     size_t wal_partition, log::Writer** new_log);

  // The span the first WAL Open creates for `wal_partition` attests: that
  // partition's newest WAL recovered through the merge, placed as
  // RestoreAliveLogFiles() places it. None when no such WAL was recovered.
  log::WALIndexPredecessorSpan RecoveredWALIndexPredecessor(
      size_t wal_partition) const;

  // Validate self-consistency of DB options
  static Status ValidateOptions(const DBOptions& db_options);
  // Validate self-consistency of DB options and its consistency with cf options
  static Status ValidateOptions(
      const DBOptions& db_options,
      const std::vector<ColumnFamilyDescriptor>& column_families);

  // Utility function to do some debug validation and sort the given vector
  // of MultiGet keys
  void PrepareMultiGetKeys(
      const size_t num_keys, bool sorted,
      autovector<KeyContext*, MultiGetContext::MAX_BATCH_SIZE>* key_ptrs);

  void MultiGetCommon(const ReadOptions& options,
                      ColumnFamilyHandle* column_family, const size_t num_keys,
                      const Slice* keys, PinnableSlice* values,
                      PinnableWideColumns* columns, std::string* timestamps,
                      Status* statuses, bool sorted_input);

  DECLARE_SYNC_AND_ASYNC(void, MultiGetCommon, const ReadOptions& options,
                         const size_t num_keys,
                         ColumnFamilyHandle** column_families,
                         const Slice* keys, PinnableSlice* values,
                         PinnableWideColumns* columns, std::string* timestamps,
                         Status* statuses,
                         std::vector<uint8_t>* newer_version_present,
                         bool sorted_input);

  // A structure to hold the information required to process MultiGet of keys
  // belonging to one column family. For a multi column family MultiGet, there
  // will be a container of these objects.
  struct MultiGetKeyRangePerCf {
    // For the batched MultiGet which relies on sorted keys, start specifies
    // the index of first key belonging to this column family in the sorted
    // list.
    size_t start;

    // For the batched MultiGet case, num_keys specifies the number of keys
    // belonging to this column family in the sorted list
    size_t num_keys;

    MultiGetKeyRangePerCf() : start(0), num_keys(0) {}

    MultiGetKeyRangePerCf(size_t first, size_t count)
        : start(first), num_keys(count) {}
  };

  // A structure to contain ColumnFamilyData and the SuperVersion obtained for
  // the consistent view of DB
  struct ColumnFamilySuperVersionPair {
    ColumnFamilyHandleImpl* cfh;
    ColumnFamilyData* cfd;

    // SuperVersion for the column family obtained in a manner that ensures a
    // consistent view across all column families in the DB
    SuperVersion* super_version;
    ColumnFamilySuperVersionPair(ColumnFamilyHandle* column_family,
                                 SuperVersion* sv)
        : cfh(static_cast<ColumnFamilyHandleImpl*>(column_family)),
          cfd(cfh->cfd()),
          super_version(sv) {}

    ColumnFamilySuperVersionPair() = default;
  };

  // A common function to obtain a consistent snapshot, which can be implicit
  // if the user doesn't specify a snapshot in read_options, across
  // multiple column families. It will attempt to get an implicit
  // snapshot without acquiring the db_mutes, but will give up after a few
  // tries and acquire the mutex if a memtable flush happens. The template
  // allows both the batched and non-batched MultiGet to call this with
  // either an std::unordered_map or autovector of column families.
  //
  // If callback is non-null, the callback is refreshed with the snapshot
  // sequence number
  //
  // `extra_sv_ref` is used to indicate whether thread-local SuperVersion
  // should be obtained with an extra ref (by GetReferencedSuperVersion()) or
  // not (by GetAndRefSuperVersion()). For instance, point lookup like MultiGet
  // does not require SuperVersion to be re-acquired throughout the entire
  // invocation (no need extra ref), while MultiCfIterators may need the
  // SuperVersion to be updated during Refresh() (requires extra ref).
  //
  // `sv_from_thread_local` being set to false indicates that the SuperVersion
  // obtained from the ColumnFamilyData, whereas true indicates they are thread
  // local.
  //
  // A non-OK status will be returned if for a column family that enables
  // user-defined timestamp feature, the specified `ReadOptions.timestamp`
  // attemps to read collapsed history.
  template <class T, typename IterDerefFuncType>
  Status MultiCFSnapshot(const ReadOptions& read_options,
                         ReadCallback* callback,
                         IterDerefFuncType iter_deref_func, T* cf_list,
                         bool extra_sv_ref, SequenceNumber* snapshot,
                         bool* sv_from_thread_local);

  // The actual implementation of the batching MultiGet. The caller is expected
  // to have acquired the SuperVersion and pass in a snapshot sequence number
  // in order to construct the LookupKeys. The start_key and num_keys specify
  // the range of keys in the sorted_keys vector for a single column family.
  DECLARE_SYNC_AND_ASYNC(
      Status, MultiGetImpl, const ReadOptions& read_options, size_t start_key,
      size_t num_keys,
      autovector<KeyContext*, MultiGetContext::MAX_BATCH_SIZE>* sorted_keys,
      SuperVersion* sv, SequenceNumber snap_seqnum, ReadCallback* callback);

  void MultiGetWithCallbackImpl(
      const ReadOptions& read_options, ColumnFamilyHandle* column_family,
      ReadCallback* callback,
      autovector<KeyContext*, MultiGetContext::MAX_BATCH_SIZE>* sorted_keys);

  Status DisableFileDeletionsWithLock();

  Status IncreaseFullHistoryTsLowImpl(ColumnFamilyData* cfd,
                                      std::string ts_low);

  bool ShouldReferenceSuperVersion(const MergeContext& merge_context);
  // Resolves a plain value whose current payload is an encoded direct-write
  // blob index, either through `value` directly or through the default-column
  // view layered on `columns`.
  static Status ResolveDirectWritePlainValue(const Slice& key,
                                             const BlobFetcher& blob_fetcher,
                                             PinnableSlice* value,
                                             PinnableWideColumns* columns);
  // Resolves each unresolved direct-write blob-valued column in `columns` and
  // rebuilds the serialized wide-column entity in place.
  static Status ResolveDirectWriteWideColumns(const Slice& key,
                                              const BlobFetcher& blob_fetcher,
                                              PinnableWideColumns* columns);
  // Dispatches between plain-value and wide-column direct-write resolution for
  // read results observed before flush makes the corresponding blob file
  // visible through normal Version metadata.
  static bool MaybeResolveDirectWriteValue(
      const ReadOptions& read_options, const Slice& key,
      bool resolve_direct_write_value, const BlobFetcher& blob_fetcher,
      PinnableSlice* value, PinnableWideColumns* columns, Status* s,
      bool* is_blob_index, bool* value_found = nullptr);
  // Completes primary memtable hits that can still carry direct-write blob
  // references before the blob file is visible in Version metadata.
  static void PostprocessDirectWriteValueRead(
      const ReadOptions& read_options, const Slice& key,
      const std::string* timestamp, bool resolve_direct_write_value,
      const BlobFetcher* blob_fetcher, PinnableSlice* value,
      PinnableWideColumns* columns, Status* s, bool* is_blob_index,
      bool* value_found = nullptr);
  // Resolves memtable read results that still carry blob references through
  // either a raw blob-index payload in `value` or unresolved blob columns in
  // `columns`. This generic path is also used by read-only/secondary DBs. Sets
  // *did_resolve to true iff there was a reference to resolve -- in which case
  // this function has finalized `value`/`columns` (populated on success,
  // cleared on error) -- and false iff there was nothing to resolve (the caller
  // finalizes the plain value). Only call with an OK incoming status.
  static Status MaybeResolveMemtableBlobValue(
      const Slice& key, const BlobFetcher* blob_fetcher, PinnableSlice* value,
      PinnableWideColumns* columns, bool* did_resolve, bool* is_blob_index,
      bool* value_found = nullptr);
  // Completes read-only/secondary memtable Get()/GetEntity() hits. Given the
  // status from the memtable read, resolves blob-backed payloads when
  // `resolve_blob_backed_memtable_value` is true, self-pins a plain value on
  // success, and clears outputs on any error (never leaving a half-populated
  // result). When the caller explicitly requested raw blob indices via
  // `GetImplOptions::is_blob_index`, this helper leaves that payload untouched.
  // `memtable_blob_fetcher` may be null when blob support is disabled for the
  // column family. Returns the finalized status.
  static Status PostprocessMemtableValueRead(
      const Slice& key, const std::string* timestamp,
      bool resolve_blob_backed_memtable_value,
      const BlobFetcher* memtable_blob_fetcher, PinnableSlice* value,
      PinnableWideColumns* columns, Status memtable_read_status,
      bool* is_blob_index, bool* value_found = nullptr);

  template <typename IterType, typename ImplType,
            typename ErrorIteratorFuncType>
  std::unique_ptr<IterType> NewMultiCfIterator(
      const ReadOptions& _read_options,
      const std::vector<ColumnFamilyHandle*>& column_families,
      ErrorIteratorFuncType error_iterator_func);

  bool ShouldPickCompaction(bool is_prepicked,
                            const PrepickedCompaction* prepicked_compaction);

  void ResetBottomPriCompactionIntent(ColumnFamilyData* cfd,
                                      std::unique_ptr<Compaction>& c);
  // Lock over the persistent DB state.  Non-nullptr iff successfully acquired.
  FileLock* db_lock_ = nullptr;

  // Guards changes to DB and CF options to ensure consistency between
  // * In-memory options objects
  // * Settings in effect
  // * Options file contents
  // while allowing the DB mutex to be released during slow operations like
  // persisting options file or modifying global periodic task timer.
  // Always acquired *before* DB mutex when this one is applicable.
  InstrumentedMutex options_mutex_;

  // Guards reads and writes to in-memory stats_history_.
  InstrumentedMutex stats_history_mutex_;

  // WAL-write exclusion, one mutex per WAL partition: WalWriteMutex(p)
  // serializes appends to, rotation of, and bookkeeping for partition p's
  // WALs, and WalSyncCV(p) is the condition variable bound to it. In
  // addition to mutex_, these protect writes to the WAL writer deques and
  // cur_wal_number_. With two_write_queues they also protect the live-file
  // deques and wal_empties_. Refer to the definition of each variable below
  // for more details.
  //
  // LOCK ORDER (acyclic; only rule 2 is mechanically checked, in debug
  // builds -- see WalWriteMutexType. Every other rule below is documented
  // and reviewed, not enforced: call-site AssertHeld checks cover a single
  // mutex's presence, never an order).
  //  1. mutex_ (the DB mutex) comes before every WAL write mutex. Scoped
  //     mutex_ takes after WAL mutexes were released (the MarkLogsSynced
  //     sections, SyncWalImpl's mark recording) nest nothing.
  //  2. WAL write mutexes are taken in ascending partition-index order, and
  //     a thread holding more than one always took them in that order.
  //     There is exactly one implementation of that order,
  //     LockWalWriteMutexes()/UnlockWalWriteMutexes() (and the
  //     AllWalWriteMutexesLock RAII wrapper); every multi-partition
  //     acquisition goes through it, and single-partition takes use
  //     WalWriteMutexLock on WalWriteMutex(p) directly.
  //  3. sequence_wal_index_mutex_ is a leaf below mutex_ and below the WAL
  //     write mutexes. Open takes it under mutex_ alone (with WAL
  //     indexing, to allocate the post-recovery marker's index). The
  //     standard and pipelined write paths take it with no other mutex held,
  //     for both allocation and completion, as do the WAL-only path's
  //     disableWAL allocation and a parked durability waiter. Only
  //     ConcurrentWriteGroupToWAL and CoverBurnedWALIndex (to note a burned
  //     index) take it under a WAL write mutex (the partition's). The
  //     background-error signal takes it under mutex_;
  //     the frontier signal runs with it already held by the completion
  //     site. Nothing acquired under it takes mutex_ or a WAL write mutex. It
  //     stays global because it guards the one LastAllocatedSequence counter
  //     and next_wal_index_ every partition allocates from, plus the
  //     cross-partition durability frontier: contiguous sequence
  //     assignment and one frontier need one mutex. It is never held
  //     across I/O or batch merging.
  //  4. wal_sync_mark_mutex_ serializes sync-marking and sits above the WAL
  //     write mutexes: a marker takes it first, then one WAL mutex at a
  //     time. It is never held with mutex_.
  //  5. Write-thread queues are entered in ascending partition-index order
  //     (AllPartitionsWriter enters 0..N-1 and exits in reverse); mutex_,
  //     which queue entry requires, is released while waiting for entry.
  //  6. The memtable-switch gate (N > 1 only): leaders Enter before any WAL
  //     take and Exit once their group has exited. Both the leader's
  //     closed-gate wait and rotation's drain wait run on bg_cv_ under
  //     mutex_ with no WAL mutex held, and rotation closes the gate before
  //     taking all N, so gate waits never meet WAL-mutex holders.
  //  7. cached_recoverable_state_mutex_ is a leaf: WriteGroupToWAL takes it
  //     with no other mutex held, only to cache the recoverable state.
  //  8. wal_gate_sync_mutex_ is a leaf: a leadership gate sync takes it
  //     with no other mutex held, and releases it before its sync takes
  //     any.
  // A write's resolve step (SelectWalForWrite) and append (WriteToWAL) hold
  // exactly its partition's mutex, but the write path does take all N in
  // several places, always in index order through rule 2's implementation:
  // PreprocessWrite's wal_dir_synced_ read (written under the rotated
  // partition's mutex or all N), sync-write completion's MarkLogsSynced /
  // MarkLogsNotSynced (WriteImpl, PipelinedWriteImpl), the manual-flush
  // single-queue leadership sync in WriteGroupToWAL, at N > 1 the same
  // three steps in a sync write's leadership gate sync
  // (SyncWalsFromCompletionGate; without manual flush its fsync takes each
  // partition's mutex sequentially instead), and the two-queue
  // WriteRecoverableState drain (concurrent WAL writers on any partition
  // touch the cached state). Off the write path, all N are held by
  // rotation in SwitchMemtable (plus mutex_), obsolete-file cleanup in
  // FindObsoleteFiles (queue check and free-list takeover, plus mutex_),
  // SyncWalImpl's horizon snapshot and closed-WAL publish (its claims are
  // sequential per-partition takes), FlushWAL's buffer flush, CloseHelper's
  // WAL free, the close-time WAL marker's lookup, the whole-DB readers
  // WALBufferIsEmpty and GetOpenWalSizes, and the test-only
  // TEST_LogsToFreeSize.
  // Waits never hold two locks that can deadlock them. A Wait() on
  // WalSyncCV(p) holds WalWriteMutex(p) and no other partition mutex:
  // pthread_cond_wait releases only the mutex the condvar is bound to, so
  // waiting while holding others would keep them held while asleep, and
  // the thread that signals (a sync completion holding all N) would block
  // on them forever. Multi-partition waits are therefore sequential: take
  // one partition's mutex, wait, mark, release, then move to the next. A
  // Wait() on wal_durability_cv_ holds sequence_wal_index_mutex_ only,
  // never a WAL mutex or mutex_: frontier and error signals take that one
  // mutex briefly and block on nothing. Per-writer state mutexes inside
  // WriteThread are leaf-private to queue waiting and order against
  // nothing above. This diff adds no lock; its one new order edge is the
  // background-error signal taking sequence_wal_index_mutex_ under mutex_
  // (rule 3's "under either").
  //
  // Private inheritance keeps InstrumentedMutexLock/InstrumentedMutexUnlock,
  // which would bypass the check, from compiling against these mutexes.
  // Cache-line aligned: every partition's leader takes its own on each
  // append, so adjacent partitions' mutexes must not share a line.
  class ALIGN_AS(CACHE_LINE_SIZE) WalWriteMutexType
      : private InstrumentedMutex {
   public:
    // Debug builds assert the calling thread holds no WAL write mutex of the
    // same DB at an equal or higher partition index (LOCK ORDER rule 2).
    void Lock() {
      AssertLockOrder();
      InstrumentedMutex::Lock();
      NoteHeld();
    }
    void Unlock() {
      NoteReleased();
      InstrumentedMutex::Unlock();
    }
    using InstrumentedMutex::AssertHeld;
    using InstrumentedMutex::ConfigureSlowHoldLogging;

    // Only for binding this partition's WalSyncCV.
    InstrumentedMutex* CondVarMutex() { return this; }

#ifndef NDEBUG
    void SetPartition(const DBImpl* db, size_t partition) {
      db_ = db;
      partition_ = partition;
    }

   private:
    void AssertLockOrder() const;
    void NoteHeld();
    void NoteReleased();
    // WAL write mutexes the calling thread holds, across all DBs.
    static std::vector<const WalWriteMutexType*>& HeldByThisThread();
    const DBImpl* db_ = nullptr;
    size_t partition_ = 0;
#else
    void SetPartition(const DBImpl* /*db*/, size_t /*partition*/) {}

   private:
    void AssertLockOrder() const {}
    void NoteHeld() {}
    void NoteReleased() {}
#endif
  };
  static_assert(alignof(WalWriteMutexType) >= CACHE_LINE_SIZE);
  static_assert(sizeof(WalWriteMutexType) % CACHE_LINE_SIZE == 0);
  std::vector<WalWriteMutexType> wal_write_mutexes_ =
      std::vector<WalWriteMutexType>(1);

  // One sync condition variable per WAL partition, each bound to its own
  // partition's mutex (see the LOCK ORDER note above for why a Wait() must
  // never hold a second partition mutex). Signaled when getting_synced
  // becomes false for some of that partition's WAL writers. Filled in the
  // DBImpl constructor, once wal_write_mutexes_ exists to bind to; never
  // resized after that, like every other partition vector.
  std::vector<InstrumentedCondVar> wal_sync_cvs_;

  // Serializes sync-marking across every multi-partition marker (sync
  // writes' gate syncs and SyncWalImpl claims), taken only above one
  // partition: a single partition's markers cannot interleave. Leadership
  // marks run others-then-own while SyncWalImpl marks ascending; without
  // exclusion two markers interleave marks across partitions and
  // ABBA-deadlock on each other's WalSyncCV waits, neither reaching the
  // sync that would finish them. Held only across marking (mutex takes,
  // flag sets, and waits for already-released markers' finishes), never
  // across I/O and never with mutex_; WAL mutexes take below it (see the
  // LOCK ORDER note).
  InstrumentedMutex wal_sync_mark_mutex_;

  // Leadership-gate sync coalescing (RunCoalescedGateSync), N>1 only: at
  // most one gate sync runs at a time, and wal_gate_synced_upto_ is the
  // highest sequence a completed one covered (every record at or below it is
  // synced). A leaf: held only to check and update these, never across I/O.
  InstrumentedMutex wal_gate_sync_mutex_;
  InstrumentedCondVar wal_gate_sync_cv_{&wal_gate_sync_mutex_};
  bool wal_gate_sync_running_ = false;
  SequenceNumber wal_gate_synced_upto_ = 0;
  // Gate syncs are numbered as they start; wal_gate_synced_epoch_ is the
  // number of the latest one that succeeded, wal_gate_sync_failed_epoch_ of
  // the latest one that failed, with its status. A gate is covered by any
  // sync numbered above the last one started before it asked, whatever its
  // group's count, and fails with any sync it queued on. Each group pins and
  // fences its own range on that failure, as for a failed append.
  uint64_t wal_gate_sync_started_epoch_ = 0;
  uint64_t wal_gate_synced_epoch_ = 0;
  uint64_t wal_gate_sync_failed_epoch_ = 0;
  IOStatus wal_gate_sync_failed_status_;

  // Serializes sequence-block and WAL-index allocation across all WAL
  // partitions. It is never held across batch merging or I/O.
  InstrumentedMutex sequence_wal_index_mutex_;
  // Parks completion-gate waiters until the durability frontier advances.
  // Bound to sequence_wal_index_mutex_ because every frontier advance runs
  // under it, so signaling needs no extra locking; a waiter holds it only
  // to check fences and while parked, never with mutex_ (see
  // WaitForWalDurability), so the leaf rule in the LOCK ORDER note holds.
  InstrumentedCondVar wal_durability_cv_{&sequence_wal_index_mutex_};
  // Waiters parked in WaitForWalDurability, counted under
  // sequence_wal_index_mutex_. Tests use it to prove a write is parked
  // before injecting a failure; the frontier signal consults it under that
  // mutex, so uncontended writes pay one relaxed load; CloseHelper waits for
  // it to drain before teardown frees the mutex and cv.
  std::atomic<size_t> wal_durability_waiters_{0};
  // Set on every background-error wake, which ErrorHandler issues only
  // while an error is recorded, and cleared only in
  // ErrorHandler::ClearBGError. Such a wake only makes parked gates
  // recheck: a gate fails on a fence, never on this flag. Tests read it.
  std::atomic<bool> wal_durability_bg_error_{false};
  // Gate fences (InstallWalGateFence): a gate whose target lies in
  // [low, high] returns `status`. high is kMaxSequenceNumber until recovery
  // releases the failed range. Kept for the DB's lifetime, one per failure,
  // so a write held past recovery still fails. Guarded by
  // sequence_wal_index_mutex_; the bounds below are stored under it and read
  // lock-free, so a gate outside every fence pays two loads.
  struct WalGateFence {
    SequenceNumber low;
    SequenceNumber high;
    Status status;
    bool released_by_recovery;
  };
  std::vector<WalGateFence> wal_gate_fences_;
  std::atomic<SequenceNumber> wal_gate_fence_low_{kMaxSequenceNumber};
  std::atomic<SequenceNumber> wal_gate_fence_high_{0};
  uint64_t next_wal_index_ = log::kWALIndexStartNumber;

  // If zero, manual compactions are allowed to proceed. If non-zero, manual
  // compactions may still be running, but will quickly fail with
  // `Status::Incomplete`. The value indicates how many threads have paused
  // manual compactions. It is accessed in read mode outside the DB mutex in
  // compaction code paths.
  std::atomic<int> manual_compaction_paused_ = false;

  // If non-zero, all compaction jobs (background automatic compactions,
  // manual compactions via CompactRange, and foreground CompactFiles calls)
  // are being aborted. Compactions will be signaled to stop. Any new
  // compaction job would fail immediately. The value indicates how many threads
  // have called AbortAllCompactions(). It is accessed in read mode outside the
  // DB mutex in compaction code paths.
  std::atomic<int> compaction_aborted_ = 0;

  // Number of AbortCompactions() callers waiting for a targeted compaction to
  // finish. Protected by the DB mutex.
  int per_cf_compaction_abort_waiters_ = 0;

  // This condition variable is signaled on these conditions:
  // * whenever bg_compaction_scheduled_ goes down to 0
  // * if AnyManualCompaction, whenever a compaction finishes, even if it hasn't
  // made any progress
  // * whenever a compaction made any progress
  // * whenever bg_flush_scheduled_ or bg_purge_scheduled_ value decreases
  // (i.e. whenever a flush is done, even if it didn't make any progress)
  // * whenever there is an error in background purge, flush or compaction
  // * whenever num_running_ingest_file_ goes to 0.
  // * whenever pending_purge_obsolete_files_ goes to 0.
  // * whenever disable_delete_obsolete_files_ goes to 0.
  // * whenever SetOptions successfully updates options.
  // * whenever a column family is dropped.
  InstrumentedCondVar bg_cv_;

  // Set only while ResumeImpl performs a file-scoped WAL recovery. Protected
  // by mutex_; SwitchMemtable uses it to avoid touching the failed writer.
  bool recovering_from_wal_write_error_ = false;
  uint64_t recover_wal_through_number_ = 0;

  // Once a file-scoped WAL error is reported, all writers through this
  // inclusive cutoff are quarantined until recovery detaches them. A
  // concurrent SyncWAL() reads this without taking the DB mutex.
  std::atomic<uint64_t> wal_recovery_cutoff_{0};

  ColumnFamilyHandleImpl* persist_stats_cf_handle_ = nullptr;

  bool persistent_stats_cfd_exists_ = true;

  // Writes are protected by locking both mutex_ and the rotated partition's
  // WalWriteMutex, and reads must be under either mutex_ or all N WAL write
  // mutexes (a reader holding all N shares the rotated partition's mutex
  // with any writer). Since after ::Open, cur_wal_number_ is currently
  // updated only in write_thread_, it can be read from the same write_thread_
  // without any locks.
  uint64_t cur_wal_number_ = 0;

  // Log files that we can recycle. Must be protected by db mutex_.
  std::deque<uint64_t> wal_recycle_files_;

  // The minimum log file number taht can be recycled, if log recycling is
  // enabled. This is used to ensure that log files created by previous
  // instances of the database are not recycled, as we cannot be sure they
  // were created in the recyclable format.
  uint64_t min_wal_number_to_recycle_ = 0;

  // Written under mutex_ plus the rotated partition's WalWriteMutex (by
  // rotation) or under all N WAL write mutexes (by sync completion); read
  // under all N, so every reader shares a mutex with every writer.
  bool wal_dir_synced_ = false;

  // Whether partition p's current WAL has received no append since it was
  // created. One entry per partition: SwitchMemtable()'s create-vs-reuse
  // decision and WriteToWAL()'s mark-non-empty each concern one partition's
  // WAL, and per-partition entries keep writers to different partitions on
  // different cache lines: every leader writes its own entry on each append.
  // Without two_write_queues, reads and writes to WalEmpty(p) are protected
  // by mutex_. Since it is currently updated/read only in write_thread_, it
  // can be accessed from the same write_thread_ without any locks. With
  // two_write_queues writes, where it can be updated in different threads,
  // reads and writes are protected by WalWriteMutex(p) instead. This is to
  // avoid expensive mutex_ lock during WAL write, which updates WalEmpty(p).
  // (uint8_t rather than bool: std::vector<bool> is bit-packed, so its
  // elements are not independently addressable mutex-protected bytes.)
  std::vector<CacheAlignedWrapper<uint8_t>> wal_empties_ =
      std::vector<CacheAlignedWrapper<uint8_t>>(
          1, CacheAlignedWrapper<uint8_t>{1});
  static_assert(sizeof(CacheAlignedWrapper<uint8_t>) >= CACHE_LINE_SIZE);

  // The current WAL file and those that have not been found obsolete from
  // memtable flushes. A WAL not on this list might still be pending writer
  // flush and/or sync and close and might still be in partition p's writer
  // deque below. One deque per WAL partition, in partition-index order.
  //
  // Locking is per field. The vector itself is sized at construction and
  // never resized, so it needs no lock. Each partition's deque is touched
  // only through the accessors below, which state and debug-assert their own
  // contracts: deque structure (push/pop) and getting_flushed change only
  // under mutex_, with retirement also holding the partition's WAL write
  // mutex; number never changes after the push; size accumulates under
  // the partition's WAL write mutex, except at N==1 with one write queue
  // and no manual_wal_flush, where the writer adds it with no WAL mutex
  // held, as before partitioning. That writer is either the write-thread
  // leader, which leadership serializes with other appends and with
  // rotation, or Open's post-recovery marker, which runs single-threaded
  // under mutex_ before Open returns. Either way only the current entry,
  // which is never popped, is appended to.
  std::vector<std::deque<WalFileNumberSize>> alive_wal_files_ =
      std::vector<std::deque<WalFileNumberSize>>(1);
  // Nonzero WAL partitions read from recovered WALs' markers, by WAL number.
  // Filled by CheckWALFilesCarryWALIndex() and read by RestoreAliveLogFiles(),
  // both during Open with mutex_ held.
  UnorderedMap<uint64_t, uint32_t> recovered_wal_partitions_;
  // First wal_index of each WAL recovered through the merge (zero when it
  // bears none), by WAL number. Filled by ProcessLogFilesMerged() and read by
  // RecoveredWALIndexPredecessor(), both during Open with mutex_ held.
  UnorderedMap<uint64_t, uint64_t> recovered_wal_first_indices_;

  // Total size of all "alive" WALs (for easy access without synchronization)
  RelaxedAtomic<uint64_t> wals_total_size_{0};

  // Log files that aren't fully synced, and the current log file. One deque
  // per WAL partition, in partition-index order. The vector itself is sized
  // at construction and never resized, so it needs no lock. Each
  // partition's deque is touched only through the accessors below, which
  // state and debug-assert their own contracts: every access holds the
  // partition's WAL write mutex, and cross-partition walks hold all N in
  // ascending partition-index order via LockWalWriteMutexes().
  //
  // Element addresses are stable: deque elements never move, and the vector
  // is never resized, so a reference or pointer to one entry stays valid
  // while the caller holds the partition's mutex (which excludes every
  // pop). The current WAL and sync-claimed entries are never popped, and a
  // sync claim is always finished by the sync that made it; both are
  // enforced by the accessors. Removal stays per-entry, so a future
  // reference-driven lifetime scheme can replace the retirement predicate
  // without reshaping the containers.
  std::vector<std::deque<LogWriterNumber>> logs_ =
      std::vector<std::deque<LogWriterNumber>>(1);

  // Each WAL partition reuses its own merge buffer.
  // WriteBatchInternal::Contents borrows this storage through WriteToWAL(), so
  // a partition must not clear or reuse its buffer until that append returns.
  // The two-write-queues path keeps its invocation-local buffer because its
  // write groups are not partition leaders.
  std::vector<WriteBatch> wal_merge_buffers_ = std::vector<WriteBatch>(1);

  // WAL deque access. Operation-shaped: each caller gets only the operation
  // it needs on partition p's writer and live-file deques, never a deque.
  // Every accessor states its locking contract and debug-asserts the
  // mutexes guarding the state it touches. Nothing here returns a
  // container, range, or iterator: a single-entry reference stays valid
  // while the caller holds the partition's WAL write mutex (deque addresses
  // are stable and every pop holds that mutex), and snapshots are returned
  // by value. Do not touch the deques directly; use these.
  //
  // Writer entries (partition p's logs_ deque).
  //
  // Partition p's current WAL: the newest writer entry. Asserts non-empty:
  // every caller runs where a current WAL is guaranteed (writes bind only
  // to live partitions; recovery checks for the replacement first).
  // REQUIRES: WalWriteMutex(partition) held.
  LogWriterNumber& CurrentWal(size_t partition) {
    WalWriteMutex(partition).AssertHeld();
    assert(!logs_[partition].empty());
    return logs_[partition].back();
  }
  // Partition p's current WAL, or nullptr when p tracks no WAL (empty at
  // open/recovery/close boundaries and on secondary instances).
  // REQUIRES: WalWriteMutex(partition) held.
  LogWriterNumber* CurrentWalOrNull(size_t partition) {
    WalWriteMutex(partition).AssertHeld();
    if (logs_[partition].empty()) {
      return nullptr;
    }
    return &logs_[partition].back();
  }
  // Partition p's oldest tracked WAL writer, or nullptr when p tracks none.
  // Peeks only: the caller examines the entry (sync state, close dance)
  // before retiring it with PopOldestWal.
  // REQUIRES: WalWriteMutex(partition) held.
  LogWriterNumber* OldestWalOrNull(size_t partition) {
    WalWriteMutex(partition).AssertHeld();
    if (logs_[partition].empty()) {
      return nullptr;
    }
    return &logs_[partition].front();
  }
  // Retires partition p's oldest tracked WAL writer. The caller hands the
  // writer to WalsToFree(p) first (ReleaseWriter) and, for obsolete-prefix
  // retirement, closes the file outside the mutex between the peek and this
  // pop. Never removes the current WAL: every caller stops at it.
  // REQUIRES: WalWriteMutex(partition) held.
  void PopOldestWal(size_t partition) {
    WalWriteMutex(partition).AssertHeld();
    assert(logs_[partition].size() > 1);
    assert(logs_[partition].front().writer == nullptr);
    logs_[partition].pop_front();
  }
  // Installs a new current WAL on partition p: the writer entry and its
  // live-file metadata, together. Coupled rotation installs one per
  // partition; open installs one per partition.
  // REQUIRES: mutex_ and WalWriteMutex(partition) held.
  void InstallCurrentWal(size_t partition, uint64_t number,
                         log::Writer* writer) {
    mutex_.AssertHeld();
    WalWriteMutex(partition).AssertHeld();
    logs_[partition].emplace_back(number, writer);
    alive_wal_files_[partition].emplace_back(number);
  }
  // Marks every WAL on partition p getting_synced for a leadership sync
  // (PreprocessWrite), waiting out a sync already in progress on the old
  // end first. Skips empty partitions. The marks are finished by the
  // write's completion (MarkLogsSynced/MarkLogsNotSynced under all N).
  // Returns whether it marked anything (false for an empty partition), and
  // counts the partition in *unsynced_partitions if a WAL it marked
  // HasUnsyncedBytes().
  // REQUIRES: WalWriteMutex(partition) held, plus wal_sync_mark_mutex_
  // above one partition.
  bool ClaimWalPartitionForLeadershipSync(size_t partition,
                                          uint32_t* unsynced_partitions);
  // Marks partition p's WALs getting_synced for SyncWalImpl and lists the
  // open files to sync. Without the current WAL, a partition holding only
  // it contributes nothing and waits for nothing. Quarantined WALs (at or
  // below the recovery cutoff) are marked but never listed: their handles
  // are unusable. The marks are finished by MarkLogsSynced/MarkLogsNotSynced.
  // Returns whether it marked anything (and so holds the front's mark).
  // REQUIRES: WalWriteMutex(partition) held, plus wal_sync_mark_mutex_
  // above one partition.
  bool ClaimWalPartitionForSyncWalImpl(
      size_t partition, bool include_current_wal,
      autovector<std::pair<log::Writer*, bool>, 1>* wals_to_sync);
  // Syncs every live WAL file on partition p for a leadership sync
  // (WriteToWAL): closed WALs need only manifest recording, so only open
  // files are synced. Stops at the first error, and sets *failed_wal_number
  // to the WAL that failed.
  // REQUIRES: WalWriteMutex(partition) held, except the N==1 single-queue
  // leadership sync, which holds no WAL mutex (single writer; the
  // getting_synced marks exclude SyncWAL).
  // `previously_failed`, when not null (the gate sync, N>1), also skips
  // every WAL at or below wal_recovery_cutoff_, and is set when a WAL was
  // skipped or its writer had already failed.
  IOStatus SyncWalPartitionFiles(size_t partition,
                                 const WriteOptions& write_options,
                                 uint64_t* failed_wal_number,
                                 bool* previously_failed = nullptr);
  // Closes partition p's WAL writers and drops the deque (CloseHelper).
  // Warns per WAL and returns the first error.
  // REQUIRES: WalWriteMutex(partition) held (the driver holds all N).
  Status CloseAndClearWalPartition(size_t partition);
  // Adds partition p's open WALs and their flushed sizes to number_to_size.
  // REQUIRES: WalWriteMutex(partition) held.
  void CollectOpenWalSizes(size_t partition,
                           std::map<uint64_t, uint64_t>* number_to_size);
  // Finishes a sync on partition p (MarkLogsSynced): WALs past the horizon
  // are other partitions' live WALs, finished but never recorded or
  // reclaimed; quarantined WALs are finished but skipped; closed or fully
  // synced inactive WALs are recorded in the manifest and their writers
  // handed to WalsToFree(p).
  // REQUIRES: WalWriteMutex(partition) held (the driver holds all N, so no
  // other marker, sync, or retirement runs concurrently).
  void FinishWalPartitionSync(size_t partition, uint64_t up_to,
                              VersionEdit* synced_wals);
  // Unmarks a failed sync on partition p (MarkLogsNotSynced).
  // REQUIRES: WalWriteMutex(partition) held (the driver holds all N).
  void CancelWalPartitionSync(size_t partition, uint64_t up_to);
  //
  // Live-file entries (partition p's alive_wal_files_ deque).
  //
  // Live-file metadata of partition p's current WAL, where appends
  // accumulate their sizes. The reference stays valid under
  // WalWriteMutex(partition): the current entry is never popped, and every
  // pop holds that mutex. SelectWalForWrite binds it into the WalContext.
  // Escape rule: after WalWriteMutex(partition) is released, only the
  // entry's number may be read through the pointer. The number never
  // changes after the push, so those reads need no lock; the size
  // accumulates under the WAL mutex and deque membership changes under
  // mutex_, so neither may be touched after the unlock. The one exception
  // is the N==1 single-queue path without manual_wal_flush, whose writer
  // (the write-thread leader, or Open's post-recovery marker) adds the size
  // after the unlock (see alive_wal_files_).
  // REQUIRES: WalWriteMutex(partition) held.
  WalFileNumberSize& CurrentAliveWal(size_t partition) {
    WalWriteMutex(partition).AssertHeld();
    assert(!alive_wal_files_[partition].empty());
    return alive_wal_files_[partition].back();
  }
  // Pops partition p's oldest live-file entry into *retired when it is
  // obsolete (number below min_log_number); otherwise leaves the deque
  // alone and returns false. The current WAL is never obsolete, so a pop
  // always leaves the deque non-empty.
  // REQUIRES: mutex_ and WalWriteMutex(partition) held.
  bool TryPopObsoleteAliveWal(size_t partition, uint64_t min_log_number,
                              WalFileNumberSize* retired) {
    mutex_.AssertHeld();
    WalWriteMutex(partition).AssertHeld();
    if (alive_wal_files_[partition].empty() ||
        alive_wal_files_[partition].front().number >= min_log_number) {
      return false;
    }
    *retired = alive_wal_files_[partition].front();
    alive_wal_files_[partition].pop_front();
    assert(!alive_wal_files_[partition].empty());
    return true;
  }
  // Records a recovered closed WAL in partition p's live-file deque
  // (RestoreAliveLogFiles). The partition comes from the WAL's marker, or
  // partition 0 when the marker named none or an absent partition; that
  // policy lives with the caller.
  // REQUIRES: mutex_ held (open-time; every writer holds mutex_).
  void AddRecoveredAliveWal(size_t partition, const WalFileNumberSize& wal) {
    mutex_.AssertHeld();
    assert(partition < alive_wal_files_.size());
    alive_wal_files_[partition].push_back(wal);
  }
  // Partition p's oldest live-file entry: its number and flush flag
  // only. The size is deliberately not copied: appends accumulate it under
  // the WAL mutex, which this caller does not hold. The number never
  // changes after the push and the flag changes only under mutex_, so no
  // WAL write mutex is needed for the two copied fields. SwitchWAL reads
  // the flag and number.
  // REQUIRES: mutex_ held.
  WalFileNumberSize OldestAliveWal(size_t partition) {
    mutex_.AssertHeld();
    assert(partition < alive_wal_files_.size());
    assert(!alive_wal_files_[partition].empty());
    const WalFileNumberSize& front = alive_wal_files_[partition].front();
    WalFileNumberSize oldest(front.number);
    oldest.getting_flushed = front.getting_flushed;
    return oldest;
  }
  // Marks partition p's oldest live-file entry getting_flushed (SwitchWAL).
  // REQUIRES: mutex_ held.
  void MarkOldestAliveWalGettingFlushed(size_t partition) {
    mutex_.AssertHeld();
    assert(partition < alive_wal_files_.size());
    assert(!alive_wal_files_[partition].empty());
    alive_wal_files_[partition].front().getting_flushed = true;
  }
  // Number of partition p's newest live-file entry, which is its current WAL
  // once open has installed one, or zero when p tracks none. The number never
  // changes after the push, and pushes hold mutex_.
  // REQUIRES: mutex_ held.
  uint64_t NewestAliveWalNumber(size_t partition) {
    mutex_.AssertHeld();
    assert(partition < alive_wal_files_.size());
    if (alive_wal_files_[partition].empty()) {
      return 0;
    }
    return alive_wal_files_[partition].back().number;
  }
  // Number of live-file entries on partition p.
  // REQUIRES: mutex_ held.
  size_t NumAliveWalFiles(size_t partition) {
    mutex_.AssertHeld();
    assert(partition < alive_wal_files_.size());
    return alive_wal_files_[partition].size();
  }
  // Whether partition p tracks WALs in both deques. False only before any
  // WAL is installed (recovery, secondary): retirement skips such
  // partitions.
  // REQUIRES: mutex_ and WalWriteMutex(partition) held.
  bool WalPartitionHasTrackedWals(size_t partition) {
    mutex_.AssertHeld();
    WalWriteMutex(partition).AssertHeld();
    return !alive_wal_files_[partition].empty() && !logs_[partition].empty();
  }
  // Partition p's live WALs in deque order, as {number, size} pairs.
  // The sizes accumulate under the WAL mutex, so the snapshot needs it too.
  // On the N==1 path that adds sizes without it (see alive_wal_files_),
  // the current WAL's size is exact only with no write in flight.
  // REQUIRES: mutex_ and WalWriteMutex(partition) held.
  std::vector<std::pair<uint64_t, uint64_t>> SnapshotAliveWalFiles(
      size_t partition);
  // Partition access for the remaining per-partition vectors: every use
  // names its partition explicitly so a future change cannot silently mix
  // partitions, and the asserts pin the layout (all nine vectors always have
  // one entry per partition). Do not touch the vectors directly; use these.
  // Element addresses are stable: deque elements never move, and none of the
  // vectors is ever resized, so references and pointers into a partition's
  // state stay valid.
  WriteBatch& WalMergeBuffer(size_t partition) {
    assert(partition < wal_merge_buffers_.size());
    return wal_merge_buffers_[partition];
  }
  WalWriteMutexType& WalWriteMutex(size_t partition) {
    assert(partition < wal_write_mutexes_.size());
    return wal_write_mutexes_[partition];
  }
  InstrumentedCondVar& WalSyncCV(size_t partition) {
    assert(partition < wal_sync_cvs_.size());
    return wal_sync_cvs_[partition];
  }
  uint8_t& WalEmpty(size_t partition) {
    assert(partition < wal_empties_.size());
    return wal_empties_[partition].obj_;
  }
  autovector<log::Writer*>& WalsToFree(size_t partition) {
    assert(partition < wals_to_free_.size());
    return wals_to_free_[partition];
  }
  WriteThread& PartitionWriteThread(size_t partition) {
    assert(partition < write_threads_.size());
    return *write_threads_[partition];
  }
  WriteThread& NonmemWriteThread(size_t partition) {
    assert(partition < nonmem_write_threads_.size());
    return *nonmem_write_threads_[partition];
  }
  // Number of WAL partitions. All nine vectors plus the durability tracker
  // always have this many entries.
  size_t NumWalPartitions() const {
    assert(logs_.size() == alive_wal_files_.size());
    assert(logs_.size() == wal_merge_buffers_.size());
    assert(logs_.size() == wal_write_mutexes_.size());
    assert(logs_.size() == wal_sync_cvs_.size());
    assert(logs_.size() == wal_empties_.size());
    assert(logs_.size() == wals_to_free_.size());
    assert(logs_.size() == write_threads_.size());
    assert(logs_.size() == nonmem_write_threads_.size());
    assert(logs_.size() == wal_durability_tracker_->num_partitions());
    return logs_.size();
  }

  // The partitioned-WAL histograms record only above one partition, so at
  // N=1 these read no clock and record nothing. WalPartitionStats() is for
  // StopWatch; WalPartitionTimerStart() returns 0, without reading the clock,
  // unless timers record, and WalPartitionTimerElapsed(0) is 0.
  Statistics* WalPartitionStats() const {
    return NumWalPartitions() > 1 ? stats_ : nullptr;
  }
  bool WalPartitionTimersEnabled() const {
    return NumWalPartitions() > 1 && stats_ != nullptr &&
           stats_->get_stats_level() > StatsLevel::kExceptTimers;
  }
  uint64_t WalPartitionTimerStart() const {
    return WalPartitionTimersEnabled()
               ? immutable_db_options_.clock->NowMicros()
               : 0;
  }
  uint64_t WalPartitionTimerElapsed(uint64_t start) const {
    return start == 0 ? 0 : immutable_db_options_.clock->NowMicros() - start;
  }

  // One Writer per WAL partition, for entering every partition's main (or
  // nonmem) write queue unbatched. Queues are entered in ascending partition
  // order and exited in reverse. A leader only ever occupies its own
  // partition's queue, so callers that follow this order exclude all writers
  // without deadlocking against leaders or against each other.
  class AllPartitionsWriter {
   public:
    AllPartitionsWriter(DBImpl* db, bool nonmem) : db_(db), nonmem_(nonmem) {}
    // REQUIRES: db mutex held; it is released while waiting.
    void EnterUnbatched() {
      AllocateWriters();
      for (size_t p = 0; p < db_->NumWalPartitions(); ++p) {
        Queue(p).EnterUnbatched(&Writer(p), &db_->mutex_);
      }
    }
    // Refuses a stall on any partition's queue. Call ExitUnbatched() only when
    // this returns true; false leaves no queue entered.
    // REQUIRES: db mutex held; it is released while waiting.
    bool EnterUnbatchedNonBlocking() {
      AllocateWriters();
      for (size_t p = 0; p < db_->NumWalPartitions(); ++p) {
        if (!Queue(p).EnterUnbatchedNonBlocking(&Writer(p), &db_->mutex_)) {
          for (; p > 0; --p) {
            Queue(p - 1).ExitUnbatched(&Writer(p - 1));
          }
          return false;
        }
      }
      return true;
    }
    void ExitUnbatched() {
      for (size_t p = db_->NumWalPartitions(); p > 0; --p) {
        Queue(p - 1).ExitUnbatched(&Writer(p - 1));
      }
    }

   private:
    void AllocateWriters() {
      const size_t num_partitions = db_->NumWalPartitions();
      if (num_partitions > 1) {
        other_writers_.reset(new WriteThread::Writer[num_partitions - 1]);
      }
    }
    WriteThread& Queue(size_t p) {
      return nonmem_ ? db_->NonmemWriteThread(p) : db_->PartitionWriteThread(p);
    }
    WriteThread::Writer& Writer(size_t p) {
      return p == 0 ? first_writer_ : other_writers_[p - 1];
    }
    DBImpl* const db_;
    const bool nonmem_;
    WriteThread::Writer first_writer_;
    std::unique_ptr<WriteThread::Writer[]> other_writers_;
  };

  // Resolves a write's partition before group formation. Data-bearing batches
  // use a stable hash of their smallest column-family id. This explicit
  // smallest-id rule makes multi-CF routing independent of record order.
  // CF-less transaction markers hash their smallest XID so commits do not all
  // become partition-0 stragglers; genuinely XID-less batches use partition 0.
  // Sequence numbers, keys, values, and caller threads are not inputs.
  // Never fails: a malformed batch still routes deterministically, and the
  // write path reports it. REQUIRES: num_partitions > 0.
  static size_t WalPartitionForWrite(
      const WriteBatch& batch,
      const DeferredPutEntityBatch* deferred_put_entities,
      size_t num_partitions);

  // Takes every WAL write mutex in ascending partition-index order. This and
  // UnlockWalWriteMutexes() are the only implementation of the multi-take
  // order; see the LOCK ORDER note above. Never wait on a WalSyncCV while
  // holding the result (see the note).
  void LockWalWriteMutexes() {
    StopWatch sw(immutable_db_options_.clock, WalPartitionStats(),
                 WAL_WRITE_MUTEX_ALL_WAIT_MICROS);
    for (size_t p = 0; p < NumWalPartitions(); ++p) {
      WalWriteMutex(p).Lock();
    }
  }
  void UnlockWalWriteMutexes() {
    for (size_t p = NumWalPartitions(); p > 0; --p) {
      WalWriteMutex(p - 1).Unlock();
    }
  }

  // RAII hold of a single partition's WAL write mutex.
  class WalWriteMutexLock {
   public:
    explicit WalWriteMutexLock(WalWriteMutexType* mutex) : mutex_(mutex) {
      mutex_->Lock();
    }
    ~WalWriteMutexLock() { mutex_->Unlock(); }
    WalWriteMutexLock(const WalWriteMutexLock&) = delete;
    WalWriteMutexLock& operator=(const WalWriteMutexLock&) = delete;

   private:
    WalWriteMutexType* const mutex_;
  };

  // RAII wrapper around LockWalWriteMutexes()/UnlockWalWriteMutexes().
  class AllWalWriteMutexesLock {
   public:
    explicit AllWalWriteMutexesLock(DBImpl* db) : db_(db) {
      db_->LockWalWriteMutexes();
    }
    ~AllWalWriteMutexesLock() { db_->UnlockWalWriteMutexes(); }
    AllWalWriteMutexesLock(const AllWalWriteMutexesLock&) = delete;
    AllWalWriteMutexesLock& operator=(const AllWalWriteMutexesLock&) = delete;

   private:
    DBImpl* db_;
  };

  // The WAL currently being appended to on partition 0. Only
  // WALIOStatusCheck's file error reset still names it, and it resets
  // partition 0's WAL alone; every other caller uses CurrentWal(p).
  // REQUIRES: WalWriteMutex(0) held.
  LogWriterNumber& CurrentWal() { return CurrentWal(0); }

  // This is the app-level state that is written to the WAL but will be used
  // only during recovery. Using this feature enables not writing the state to
  // memtable on normal writes and hence improving the throughput. Each new
  // write of the state will replace the previous state entirely even if the
  // keys in the two consecutive states do not overlap.
  // It is protected by the selected partition's WalWriteMutex when
  // two_write_queues_ is enabled. Otherwise only the heaad of write_thread_
  // can access it. Both write queues select partition 0 today, so concurrent
  // WAL writers share that mutex; N-partition selection with two_write_queues
  // (a future change) must keep a common mutex for this state.
  WriteBatch cached_recoverable_state_;
  std::atomic<bool> cached_recoverable_state_empty_ = {true};
  // Without two_write_queues, partition leaders cache the state concurrently:
  // they assign it under this leaf mutex (LOCK ORDER rule 7), and only
  // replace a state cached at a lower sequence, recorded here. The drain in
  // WriteRecoverableState() runs with every leader excluded.
  InstrumentedMutex cached_recoverable_state_mutex_;
  SequenceNumber cached_recoverable_state_seq_ = 0;

  // If partition p's list is non-empty, we need to delete these log files in
  // background threads. Appends hold WalWriteMutex(p); the takeover into a
  // JobContext (FindObsoleteFiles) and the drain (CloseHelper) hold all N.
  std::vector<autovector<log::Writer*>> wals_to_free_ =
      std::vector<autovector<log::Writer*>>(1);

  bool is_snapshot_supported_ = true;

  std::map<uint64_t, std::map<std::string, uint64_t>> stats_history_;

  std::map<std::string, uint64_t> stats_slice_;

  bool stats_slice_initialized_ = false;

  Directories directories_;

  WriteBufferManager* write_buffer_manager_;

  // One write queue per WAL partition, indexed like the partition vectors: a
  // writer joins the queue of the partition it was routed to, so leaders of
  // different partitions run concurrently and share no queue state. Sized in
  // the constructor and never resized.
  std::vector<std::unique_ptr<WriteThread>> write_threads_;
  // The per-partition write queues for writers that have no memtable write.
  // These are used in 2PC to batch the prepares separately from the serial
  // commit.
  std::vector<std::unique_ptr<WriteThread>> nonmem_write_threads_;

  // Per-partition WAL durability frontier, sized with the partition vectors
  // in the constructor and never resized. Every Register/Complete runs under
  // sequence_wal_index_mutex_, which orders each registration before any
  // later scan.
  std::unique_ptr<WalDurabilityTracker> wal_durability_tracker_;
  // Per-partition memtable-applied frontier (N>1 only), with the same range
  // bookkeeping: AllocateSequenceAndWALIndex registers every range the
  // durability tracker registers plus every disableWAL range that consumes
  // sequences, and PublishAppliedPrefix completes it after the memtable
  // insert. Its safe point is the highest sequence N>1 may publish. Every
  // Register/Complete runs under sequence_wal_index_mutex_.
  std::unique_ptr<WalDurabilityTracker> applied_tracker_;
  // N>1 ranges registered by groups whose WAL append or leadership sync
  // failed. They stay registered in both trackers while the background error
  // stands, holding every later gated write, and are released when recovery
  // clears it. Guarded by mutex_.
  struct FailedWalRange {
    size_t partition;
    SequenceNumber low;
    SequenceNumber high;
  };
  std::vector<FailedWalRange> failed_wal_ranges_;
  // N>1 WAL indices whose append failed before the record landed, with the
  // WAL that burned each, until ExplainBurnedWALIndices covers them durably.
  // Guarded by sequence_wal_index_mutex_.
  struct BurnedWalIndex {
    uint64_t wal_number;
    uint64_t wal_index;
  };
  std::vector<BurnedWalIndex> burned_wal_indices_;

  // See CloseMemTableSwitchGate().
  std::atomic<size_t> leaders_past_switch_gate_{0};
  std::atomic<bool> switch_gate_closed_{false};

  WriteController write_controller_;

  // Size of the last batch group, per WAL partition. In slowdown mode, next
  // write needs to sleep if it uses up the quota. Each partition's leader
  // charges its own partition's last group, as the only leader does at N=1.
  // Note: This is to protect memtable and compaction. If the batch only writes
  // to the WAL its size need not to be included in this.
  std::vector<RelaxedAtomic<uint64_t>> last_batch_group_size_ =
      std::vector<RelaxedAtomic<uint64_t>>(1);

  FlushScheduler flush_scheduler_;

  TrimHistoryScheduler trim_history_scheduler_;

  SnapshotList snapshots_;

  TimestampedSnapshotList timestamped_snapshots_;

  // For each background job, pending_outputs_ keeps the current file number at
  // the time that background job started.
  // FindObsoleteFiles()/PurgeObsoleteFiles() never deletes any file that has
  // number bigger than any of the file number in pending_outputs_. Since file
  // numbers grow monotonically, this also means that pending_outputs_ is always
  // sorted. After a background job is done executing, its file number is
  // deleted from pending_outputs_, which allows PurgeObsoleteFiles() to clean
  // it up.
  // State is protected with db mutex.
  std::list<uint64_t> pending_outputs_;

  // Similar to pending_outputs_, FindObsoleteFiles()/PurgeObsoleteFiles() never
  // deletes any OPTIONS file that has number bigger than any of the file number
  // in min_options_file_numbers_.
  std::list<uint64_t> min_options_file_numbers_;

  // flush_queue_ and compaction_queue_ hold column families that we need to
  // flush and compact, respectively.
  // A column family is inserted into flush_queue_ when it satisfies condition
  // cfd->imm()->IsFlushPending()
  // A column family is inserted into compaction_queue_ when it satisfied
  // condition cfd->NeedsCompaction()
  // Column families in these collections are all Ref()-erenced.
  // TODO(icanadi) Provide some kind of ReferencedColumnFamily class that will
  // do RAII on ColumnFamilyData
  // Column families are in this queue when they need to be flushed or
  // compacted. Consumers of these queues are flush and compaction threads. When
  // column family is put on this queue, we increase unscheduled_flushes_ and
  // unscheduled_compactions_. When these variables are bigger than zero, that
  // means we need to schedule background threads for flush and compaction.
  // Once the background threads are scheduled, we decrease unscheduled_flushes_
  // and unscheduled_compactions_. That way we keep track of number of
  // compaction and flush threads we need to schedule. This scheduling is done
  // in MaybeScheduleFlushOrCompaction()
  // invariant(column family present in flush_queue_ <==>
  // ColumnFamilyData::pending_flush_ == true)
  std::deque<FlushRequest> flush_queue_;
  // invariant((column family in compaction_queue_ or
  //            column family in parked_compaction_cfds_)
  //           <=> queued_for_compaction() == true)
  //
  // "Parked" means a column family whose compaction was aborted via
  // AbortCompactions() while it was queued. When an aborted CF reaches the
  // front of compaction_queue_, PickCompactionFromQueue() moves it to
  // parked_compaction_cfds_ instead of scheduling it. The CF does not
  // immediately lose its place -- only when it reaches the front of the
  // queue is it parked. This preserves fairness for CFs behind it and
  // avoids restoring the scheduler credit (which would cause a hot re-pick
  // loop). On the final ResumeCompactions(), RestoreParkedCompaction()
  // re-enqueues the CF (or releases its ref if it no longer needs
  // compaction). Each parked CF carries exactly one Ref() from the
  // original AddToCompactionQueue() call.
  std::deque<ColumnFamilyData*> compaction_queue_;
  std::unordered_set<ColumnFamilyData*> parked_compaction_cfds_;

  // A map to store file numbers and filenames of the files to be purged
  std::unordered_map<uint64_t, PurgeFileInfo> purge_files_;

  // A vector to store the file numbers that have been assigned to certain
  // JobContext. Current implementation tracks table and blob files only.
  std::unordered_set<uint64_t> files_grabbed_for_purge_;

  // A queue to store log writers to close. Protected by db mutex_.
  std::deque<log::Writer*> wals_to_free_queue_;

  std::deque<SuperVersion*> superversions_to_free_queue_;

  int unscheduled_flushes_ = 0;

  int unscheduled_compactions_ = 0;

  // count how many background compactions are running or have been scheduled in
  // the BOTTOM pool
  int bg_bottom_compaction_scheduled_ = 0;

  // count how many background compactions are running or have been scheduled
  int bg_compaction_scheduled_ = 0;

  // stores the number of compactions are currently running
  int num_running_compactions_ = 0;

  // stores the number of BOTTOM-priority compactions currently running
  int num_running_bottom_compactions_ = 0;

  // Number of compaction service jobs currently waiting in
  // CompactionService::Wait(), counted per executed subcompaction.
  std::atomic<int> num_running_remote_compactions_ = 0;

  // number of background memtable flush jobs, submitted to the HIGH pool
  int bg_flush_scheduled_ = 0;

  // LOW avoids consuming a flush worker while joining the target write thread.
  static constexpr Env::Priority kWBMFlushPriority = Env::Priority::LOW;

  // Outstanding cross-DB WBM flush jobs, drained during shutdown.
  int bg_wbm_flush_scheduled_ = 0;

  // stores the number of flushes are currently running
  int num_running_flushes_ = 0;

  // number of background obsolete file purge jobs, submitted to the HIGH pool
  int bg_purge_scheduled_ = 0;

  // number of pressure callbacks currently in progress (for destructor safety)
  int bg_pressure_callback_in_progress_ = 0;

  enum class AsyncFileOpenState : uint8_t {
    kNotScheduled = 0,  // Async file opening has not been scheduled.
    kScheduled,         // Async file opening is in-flight in the HIGH pool.
    kComplete,          // Async file opening has finished (or was not needed).
  };

  // Tracks whether background async file opening has been scheduled/completed.
  AsyncFileOpenState bg_async_file_open_state_ =
      AsyncFileOpenState::kNotScheduled;

  // State machine for the single async WAL precreation slot protected by
  // mutex_. Background precreation failure returns to kNotScheduled; foreground
  // rotation handles it the same as no prepared WAL and creates one
  // synchronously. kScheduled owns a reserved file number; kReady owns an
  // opened writer that has not been started or added to logical WAL tracking.
  enum class AsyncWALPrecreateState : uint8_t {
    kNotScheduled = 0,  // No WAL precreate work is in-flight or ready.
    kScheduled,         // Background task owns creation of the reserved WAL.
    kReady,             // Reserved WAL writer is open but not logically live.
  };

  // Protected by mutex_. Tracks at most one background precreated WAL. A
  // precreated WAL is only reserved empty storage until SwitchMemtable()
  // consumes it and installs it in DBImpl's in-memory logical WAL tracking
  // lists (the WAL writer and live-file deques).
  AsyncWALPrecreateState async_wal_precreate_state_ =
      AsyncWALPrecreateState::kNotScheduled;

  // Reserved in-flight/ready precreated WAL. The writer is populated only while
  // state is kReady.
  UnpublishedWAL async_wal_precreate_wal_;

  std::deque<ManualCompactionState*> manual_compaction_dequeue_;

  // shall we disable deletion of obsolete files
  // if 0 the deletion is enabled.
  // if non-zero, files will not be getting deleted
  // This enables two different threads to call
  // EnableFileDeletions() and DisableFileDeletions()
  // without any synchronization
  int disable_delete_obsolete_files_ = 0;

  // Number of times FindObsoleteFiles has found deletable files and the
  // corresponding call to PurgeObsoleteFiles has not yet finished.
  int pending_purge_obsolete_files_ = 0;

  // last time when DeleteObsoleteFiles with full scan was executed. Originally
  // initialized with startup time.
  uint64_t delete_obsolete_files_last_run_;

  // The thread that wants to switch memtable, can wait on this cv until the
  // pending writes to memtable finishes.
  std::condition_variable switch_cv_;
  // The mutex used by switch_cv_. mutex_ should be acquired beforehand.
  std::mutex switch_mutex_;
  // Number of threads intending to write to memtable
  std::atomic<size_t> pending_memtable_writes_{0};

  // A flag indicating whether the current rocksdb database has any
  // data that is not yet persisted into either WAL or SST file.
  // Used when disableWAL is true.
  std::atomic<bool> has_unpersisted_data_{false};

  // Number of live column families with an active blob direct write partition
  // manager. This keeps the read/write fast paths at O(1) when the feature is
  // completely disabled for the DB.
  std::atomic<uint32_t> blob_direct_write_cf_count_{0};

  // if an attempt was made to flush all column families that
  // the oldest log depends on but uncommitted data in the oldest
  // log prevents the log from being released.
  // We must attempt to free the dependent memtables again
  // at a later time after the transaction in the oldest
  // log is fully commited.
  bool unable_to_release_oldest_log_{false};

  // Number of running IngestExternalFile() or CreateColumnFamilyWithImport()
  // calls.
  // REQUIRES: mutex held
  int num_running_ingest_file_ = 0;

  // Number of FileIngestionHandle objects produced by PrepareFileIngestion()
  // that have not been committed or destroyed yet.
  std::atomic<uint32_t> num_outstanding_prepared_ingestions_{0};

  WalManager wal_manager_;

  // A value of > 0 temporarily disables scheduling of background work
  int bg_work_paused_ = 0;

  // A value of > 0 temporarily disables scheduling of background compaction
  int bg_compaction_paused_ = 0;

  // Guard against multiple concurrent refitting
  bool refitting_level_ = false;

  // The min threshold to triggere bottommost compaction for removing
  // garbages, among all column families.
  SequenceNumber bottommost_files_mark_threshold_ = kMaxSequenceNumber;

  // The min threshold to trigger compactions for standalone range deletion
  // files that are marked for compaction.
  SequenceNumber standalone_range_deletion_files_mark_threshold_ =
      kMaxSequenceNumber;

  LogsWithPrepTracker logs_with_prep_tracker_;

  // Callback for compaction to check if a key is visible to a snapshot.
  // REQUIRES: mutex held
  std::unique_ptr<SnapshotChecker> snapshot_checker_;
  // When set, InitSnapshotContext() appends GetLastPublishedSequence() to the
  // job's snapshot_seqs (if not already present) so that flush/compaction
  // preserves the published-sequence boundary even when no explicit user
  // snapshot exists there. Unlike taking a real ManagedSnapshot, this just
  // pins the seqno into the snapshot list: no allocation, no snapshot list
  // mutation, and no SnapshotChecker.
  bool track_published_seq_in_snapshot_context_ = false;

  // Callback for when the cached_recoverable_state_ is written to memtable
  // Only to be set during initialization
  std::unique_ptr<PreReleaseCallback> recoverable_state_pre_release_callback_;

  // Scheduler to run DumpStats(), PersistStats(), and FlushInfoLog().
  // Currently, internally it has a global timer instance for running the tasks.
  PeriodicTaskScheduler periodic_task_scheduler_;

  // It contains the implementations for each periodic task.
  std::map<PeriodicTaskType, const PeriodicTaskFunc> periodic_task_functions_;

  // When set, we use a separate queue for writes that don't write to memtable.
  // In 2PC these are the writes at Prepare phase.
  const bool two_write_queues_;
  const bool manual_wal_flush_;

  // LastSequence also indicates last published sequence visibile to the
  // readers. Otherwise LastPublishedSequence should be used.
  const bool last_seq_same_as_publish_seq_;
  // It indicates that a customized gc algorithm must be used for
  // flush/compaction and if it is not provided vis SnapshotChecker, we should
  // disable gc to be safe.
  const bool use_custom_gc_;
  // Flag to indicate that the DB instance shutdown has been initiated. This
  // different from shutting_down_ atomic in that it is set at the beginning
  // of shutdown sequence, specifically in order to prevent any background
  // error recovery from going on in parallel. The latter, shutting_down_,
  // is set a little later during the shutdown after scheduling memtable
  // flushes
  std::atomic<bool> shutdown_initiated_{false};
  // Flag to indicate whether sst_file_manager object was allocated in
  // DB::Open() or passed to us
  bool own_sfm_;

  // Flag to check whether Close() has been called on this DB
  bool closed_ = false;
  // save the closing status, for re-calling the close()
  Status closing_status_;
  // mutex for DB::Close()
  InstrumentedMutex closing_mutex_;

  // Conditional variable to coordinate installation of atomic flush results.
  // With atomic flush, each bg thread installs the result of flushing multiple
  // column families, and different threads can flush different column
  // families. It's difficult to rely on one thread to perform batch
  // installation for all threads. This is different from the non-atomic flush
  // case.
  // atomic_flush_install_cv_ makes sure that threads install atomic flush
  // results sequentially. Flush results of memtables with lower IDs get
  // installed to MANIFEST first.
  InstrumentedCondVar atomic_flush_install_cv_;

  bool wal_in_db_path_ = false;
  std::atomic<uint64_t> max_total_wal_size_;

  BlobFileCompletionCallback blob_callback_;

  // WriteBufferManager stalling interfaces, one per WAL partition: each
  // partition's leader stalls on its own, so the manager queues, signals and
  // removes every stalled leader separately.
  std::vector<std::unique_ptr<StallInterface>> wbm_stall_;
  // Stalled leaders currently hiding wbm_flush_initiator_; the DB is marked
  // flushable again only when the last of them leaves. Guarded by mutex_.
  size_t wbm_leaders_hiding_flush_initiator_ = 0;

  // seqno_to_time_mapping_ stores the sequence number to time mapping, it's not
  // thread safe, both read and write need db mutex hold.
  SeqnoToTimeMapping seqno_to_time_mapping_;

  // Stop write token that is acquired when first LockWAL() is called.
  // Destroyed when last UnlockWAL() is called. Controlled by DB mutex.
  // See lock_wal_owner_thread_id_counts_
  std::unique_ptr<WriteControllerToken> lock_wal_write_token_;

  // Thread IDs of valid LockWAL() owners and their recursive lock counts. Each
  // owning thread must call UnlockWAL() the same number of times before writes
  // can resume.
  std::unordered_map<uint64_t, uint32_t> lock_wal_owner_thread_id_counts_;
};

class GetWithTimestampReadCallback : public ReadCallback {
 public:
  explicit GetWithTimestampReadCallback(SequenceNumber seq)
      : ReadCallback(seq) {}
  // A null result is used by MultiGet, whose per-key results live in
  // KeyContext/GetContext.
  void EnableNewerVersionTracking(SequenceNumber read_snapshot_seq,
                                  SequenceNumber upper_bound_seq,
                                  bool* single_key_result) {
    metadata_read_bounds_.emplace(
        MetadataReadBounds{read_snapshot_seq, upper_bound_seq});
    newer_version_present_ = single_key_result;
  }
  const MetadataReadBounds* GetMetadataReadBounds() const override {
    return metadata_read_bounds_.has_value() ? &*metadata_read_bounds_
                                             : nullptr;
  }
  bool NeedToTrackNewerVersions(
      const bool* per_key_result = nullptr) const override {
    const bool* result =
        per_key_result != nullptr ? per_key_result : newer_version_present_;
    return metadata_read_bounds_.has_value() && result != nullptr && !*result;
  }
  bool IsVisibleFullCheck(SequenceNumber seq) override {
    return seq <= max_visible_seq_;
  }
  bool IsNewerVisibleForMetadataRead(SequenceNumber seq) override {
    const MetadataReadBounds* bounds = GetMetadataReadBounds();
    return bounds != nullptr && bounds->read_snapshot_seq < seq &&
           seq <= bounds->newer_version_upper_bound_seq;
  }
  void MaybeRecordNewerVersion(SequenceNumber seq, ValueType type,
                               bool* per_key_result = nullptr) override {
    bool* result =
        per_key_result != nullptr ? per_key_result : newer_version_present_;
    if (result != nullptr && !*result &&
        (IsValueType(type) || type == kTypeRangeDeletion) &&
        IsNewerVisibleForMetadataRead(seq)) {
      *result = true;
    }
  }

 private:
  std::optional<MetadataReadBounds> metadata_read_bounds_;
  bool* newer_version_present_ = nullptr;
};

Options SanitizeOptions(const std::string& db, const Options& src,
                        bool read_only = false,
                        Status* logger_creation_s = nullptr);

DBOptions SanitizeOptions(const std::string& db, const DBOptions& src,
                          bool read_only = false,
                          Status* logger_creation_s = nullptr);

CompressionType GetCompressionFlush(const ImmutableCFOptions& ioptions,
                                    const MutableCFOptions& mutable_cf_options);

// Return a VersionEdit for the DB's recovery when the `memtables` of the
// specified column family are obsolete. Specifically, the min log number to
// keep, and the WAL files that can be deleted.
VersionEdit GetDBRecoveryEditForObsoletingMemTables(
    VersionSet* vset, const ColumnFamilyData& cfd,
    const autovector<VersionEdit*>& edit_list,
    const autovector<ReadOnlyMemTable*>& memtables,
    LogsWithPrepTracker* prep_tracker);

// Return the earliest log file to keep after the memtable flush is
// finalized.
// `cfd_to_flush` is the column family whose memtable (specified in
// `memtables_to_flush`) will be flushed and thus will not depend on any WAL
// file.
// The function is only applicable to 2pc mode.
uint64_t PrecomputeMinLogNumberToKeep2PC(
    VersionSet* vset, const ColumnFamilyData& cfd_to_flush,
    const autovector<VersionEdit*>& edit_list,
    const autovector<ReadOnlyMemTable*>& memtables_to_flush,
    LogsWithPrepTracker* prep_tracker);
// For atomic flush.
uint64_t PrecomputeMinLogNumberToKeep2PC(
    VersionSet* vset, const autovector<ColumnFamilyData*>& cfds_to_flush,
    const autovector<autovector<VersionEdit*>>& edit_lists,
    const autovector<const autovector<ReadOnlyMemTable*>*>& memtables_to_flush,
    LogsWithPrepTracker* prep_tracker);

// In non-2PC mode, WALs with log number < the returned number can be
// deleted after the cfd_to_flush column family is flushed successfully.
uint64_t PrecomputeMinLogNumberToKeepNon2PC(
    VersionSet* vset, const ColumnFamilyData& cfd_to_flush,
    const autovector<VersionEdit*>& edit_list);
// For atomic flush.
uint64_t PrecomputeMinLogNumberToKeepNon2PC(
    VersionSet* vset, const autovector<ColumnFamilyData*>& cfds_to_flush,
    const autovector<autovector<VersionEdit*>>& edit_lists);

// `cfd_to_flush` is the column family whose memtable will be flushed and thus
// will not depend on any WAL file. nullptr means no memtable is being flushed.
// The function is only applicable to 2pc mode.
uint64_t FindMinPrepLogReferencedByMemTable(
    VersionSet* vset, const autovector<ReadOnlyMemTable*>& memtables_to_flush);
// For atomic flush.
uint64_t FindMinPrepLogReferencedByMemTable(
    VersionSet* vset,
    const autovector<const autovector<ReadOnlyMemTable*>*>& memtables_to_flush);

// Fix user-supplied options to be reasonable
template <class T, class V>
static void ClipToRange(T* ptr, V minvalue, V maxvalue) {
  if (static_cast<V>(*ptr) > maxvalue) {
    *ptr = maxvalue;
  }
  if (static_cast<V>(*ptr) < minvalue) {
    *ptr = minvalue;
  }
}

inline Status DBImpl::FailIfCfHasTs(
    const ColumnFamilyHandle* column_family) const {
  if (!column_family) {
    return Status::InvalidArgument("column family handle cannot be null");
  }
  assert(column_family);
  const Comparator* const ucmp = column_family->GetComparator();
  assert(ucmp);
  if (ucmp->timestamp_size() > 0) {
    std::ostringstream oss;
    oss << "cannot call this method on column family "
        << column_family->GetName() << " that enables timestamp";
    return Status::InvalidArgument(oss.str());
  }
  return Status::OK();
}

inline Status DBImpl::FailIfTsMismatchCf(ColumnFamilyHandle* column_family,
                                         const Slice& ts) const {
  if (!column_family) {
    return Status::InvalidArgument("column family handle cannot be null");
  }
  assert(column_family);
  const Comparator* const ucmp = column_family->GetComparator();
  assert(ucmp);
  if (0 == ucmp->timestamp_size()) {
    std::stringstream oss;
    oss << "cannot call this method on column family "
        << column_family->GetName() << " that does not enable timestamp";
    return Status::InvalidArgument(oss.str());
  }
  const size_t ts_sz = ts.size();
  if (ts_sz != ucmp->timestamp_size()) {
    std::stringstream oss;
    oss << "Timestamp sizes mismatch: expect " << ucmp->timestamp_size() << ", "
        << ts_sz << " given";
    return Status::InvalidArgument(oss.str());
  }
  return Status::OK();
}

inline Status DBImpl::FailIfTableFilterWithRangeConversion(
    const ReadOptions& read_options,
    const MutableCFOptions& mutable_cf_options) const {
  if (HasTableFilter(read_options) &&
      mutable_cf_options.min_tombstones_for_range_conversion > 0) {
    return Status::InvalidArgument(
        "ReadOptions::table_filter is not supported when "
        "min_tombstones_for_range_conversion > 0");
  }
  return Status::OK();
}

inline Status DBImpl::FailIfReadCollapsedHistory(const ColumnFamilyData* cfd,
                                                 const SuperVersion* sv,
                                                 const Slice& ts) const {
  // Reaching to this point means the timestamp size matching sanity check in
  // `DBImpl::FailIfTsMismatchCf` already passed. So we skip that and assume
  // column family has the same user-defined timestamp format as `ts`.
  const Comparator* const ucmp = cfd->user_comparator();
  assert(ucmp);
  const std::string& full_history_ts_low = sv->full_history_ts_low;
  assert(full_history_ts_low.empty() ||
         full_history_ts_low.size() == ts.size());
  if (!full_history_ts_low.empty() &&
      ucmp->CompareTimestamp(ts, full_history_ts_low) < 0) {
    std::stringstream oss;
    oss << "Read timestamp: " << ucmp->TimestampToString(ts)
        << " is smaller than full_history_ts_low: "
        << ucmp->TimestampToString(full_history_ts_low) << std::endl;
    return Status::InvalidArgument(oss.str());
  }
  return Status::OK();
}
}  // namespace ROCKSDB_NAMESPACE
