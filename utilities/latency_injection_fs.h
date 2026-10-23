//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "port/port.h"
#include "rocksdb/file_system.h"
#include "rocksdb/system_clock.h"

namespace ROCKSDB_NAMESPACE {

// A FileSystem wrapper that delays writes to WAL files, so a high-latency WAL
// device (networked storage) can be reproduced on local disk, and one slow
// WAL can be forced in tests.
//
// Append latency delays both Append and both PositionedAppend overloads; sync
// latency delays Sync, Fsync and RangeSync. Files that ParseFileName does not
// classify as kWalFile are returned unwrapped, so SST, MANIFEST, OPTIONS and
// blob writes are untouched and pay nothing. Each WAL takes the latency set
// for its file number, or the default.
//
// The delay is a sleep in the calling thread before the wrapped operation:
// the caller blocks holding whatever it holds, as it would on slow storage,
// and the operation takes effect only after the delay, so a crash during the
// delay loses it. There is no concurrency cap.
//
// Jitter adds to each operation a value in [0, jitter_micros] that is a pure
// function of (seed, WAL number, the operation's index on that WAL), so a
// configuration, a seed and a per-WAL operation sequence always give the same
// schedule, whatever the interleaving across WALs.
class LatencyInjectionFileSystem : public FileSystemWrapper {
 public:
  struct Latency {
    uint64_t append_micros = 0;
    uint64_t sync_micros = 0;
    uint64_t jitter_micros = 0;
  };

  struct Counters {
    // Append and PositionedAppend calls, either overload.
    uint64_t appends = 0;
    // Bytes of the appends that succeeded.
    uint64_t append_bytes = 0;
    // Sync, Fsync and RangeSync calls.
    uint64_t syncs = 0;
    // Delay requested by this wrapper.
    uint64_t injected_sleep_micros = 0;
    // Time spent in the operations: the injected delay plus the wrapped
    // FileSystem's own time.
    uint64_t wall_micros = 0;

    std::string ToString() const;
  };

  LatencyInjectionFileSystem(const std::shared_ptr<FileSystem>& base,
                             const Latency& default_latency, uint64_t seed);
  ~LatencyInjectionFileSystem() override;

  static const char* kClassName() { return "LatencyInjectionFileSystem"; }
  const char* Name() const override { return kClassName(); }

  IOStatus NewWritableFile(const std::string& fname,
                           const FileOptions& file_opts,
                           std::unique_ptr<FSWritableFile>* result,
                           IODebugContext* dbg) override;
  IOStatus ReopenWritableFile(const std::string& fname,
                              const FileOptions& file_opts,
                              std::unique_ptr<FSWritableFile>* result,
                              IODebugContext* dbg) override;
  IOStatus ReuseWritableFile(const std::string& fname,
                             const std::string& old_fname,
                             const FileOptions& file_opts,
                             std::unique_ptr<FSWritableFile>* result,
                             IODebugContext* dbg) override;

  // Applies to every WAL without an override, including WALs already open.
  void SetDefaultLatency(const Latency& latency);
  // Overrides the default for one WAL number, whether or not that WAL is
  // open yet.
  void SetWalLatency(uint64_t wal_number, const Latency& latency);
  void ClearWalLatency(uint64_t wal_number);

  // Counters of one WAL (zero for a WAL never opened), and of all WALs.
  Counters GetWalCounters(uint64_t wal_number) const;
  Counters GetAggregateCounters() const;
  // Every WAL number opened or configured so far, ascending.
  std::vector<uint64_t> GetWalNumbers() const;
  // The aggregate counters, then one line per WAL.
  std::string PrintCounters() const;
  // Zeroes the counters. Jitter schedules continue where they were.
  void ResetCounters();

 private:
  struct AtomicCounters {
    std::atomic<uint64_t> appends{0};
    std::atomic<uint64_t> append_bytes{0};
    std::atomic<uint64_t> syncs{0};
    std::atomic<uint64_t> injected_sleep_micros{0};
    std::atomic<uint64_t> wall_micros{0};

    void Record(bool is_sync, uint64_t bytes, uint64_t injected, uint64_t wall);
    Counters Load() const;
    void Reset();
  };

  struct WalState {
    explicit WalState(uint64_t number) : wal_number(number) {}
    const uint64_t wal_number;
    std::atomic<bool> has_override{false};
    std::atomic<uint64_t> append_micros{0};
    std::atomic<uint64_t> sync_micros{0};
    std::atomic<uint64_t> jitter_micros{0};
    // Index of the next operation on this WAL, the jitter schedule's key.
    std::atomic<uint64_t> next_op{0};
    AtomicCounters counters;
  };

  class WalFile;

  // Returns the state for a WAL file name, or nullptr for any other file.
  std::shared_ptr<WalState> StateForFile(const std::string& fname);
  std::shared_ptr<WalState> GetOrCreateState(uint64_t wal_number);
  IOStatus WrapIfWal(const std::string& fname, IOStatus s,
                     std::unique_ptr<FSWritableFile>* result);
  // Delays, then runs `op`, then records the operation.
  template <typename Op>
  IOStatus Delayed(WalState* state, bool is_sync, uint64_t bytes, Op op);

  const uint64_t seed_;
  const std::shared_ptr<SystemClock> clock_;
  std::atomic<uint64_t> default_append_micros_;
  std::atomic<uint64_t> default_sync_micros_;
  std::atomic<uint64_t> default_jitter_micros_;
  mutable port::Mutex mutex_;
  std::map<uint64_t, std::shared_ptr<WalState>> wals_;
};

}  // namespace ROCKSDB_NAMESPACE
