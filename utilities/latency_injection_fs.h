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
// Each WAL file stands for its own device with a base latency and a
// bandwidth: an append also sleeps for its own bytes at the append bandwidth,
// and a sync sleeps for the bytes it makes durable (appended to that WAL
// since its last successful sync) at the sync bandwidth. Either size term is
// 0 when its bandwidth is 0.
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
    // Size terms: an append of b bytes also sleeps SizeDelayMicros(b,
    // append_bytes_per_sec), and a sync sleeps for the bytes it makes durable
    // at sync_bytes_per_sec. 0 disables the term it belongs to.
    uint64_t append_bytes_per_sec = 0;
    uint64_t sync_bytes_per_sec = 0;
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
    // Of that delay, the part that came from the size term (also counted
    // above), so a run can show how much of its delay came from size.
    uint64_t size_sleep_micros = 0;
    // Time spent in the operations: the injected delay plus the wrapped
    // FileSystem's own time.
    uint64_t wall_micros = 0;

    std::string ToString() const;
  };

  // ceil(bytes * 1e6 / bytes_per_sec) microseconds; 0 when either is 0.
  // Exact for any byte count up to 4 GiB and any rate from 1 byte/s up, and
  // saturating (never wrapping) beyond that.
  static uint64_t SizeDelayMicros(uint64_t bytes, uint64_t bytes_per_sec);

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
    std::atomic<uint64_t> size_sleep_micros{0};
    std::atomic<uint64_t> wall_micros{0};

    void Record(bool is_sync, uint64_t bytes, uint64_t injected, uint64_t size,
                uint64_t wall);
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
    std::atomic<uint64_t> append_bytes_per_sec{0};
    std::atomic<uint64_t> sync_bytes_per_sec{0};
    // Index of the next operation on this WAL, the jitter schedule's key.
    std::atomic<uint64_t> next_op{0};
    // Bytes appended successfully since the last successful sync; each sync
    // claims up to its range from this and is charged for the claim.
    std::atomic<uint64_t> unsynced_bytes{0};
    AtomicCounters counters;
  };

  class WalFile;

  // Returns the state for a WAL file name, or nullptr for any other file.
  std::shared_ptr<WalState> StateForFile(const std::string& fname);
  std::shared_ptr<WalState> GetOrCreateState(uint64_t wal_number);
  IOStatus WrapIfWal(const std::string& fname, IOStatus s,
                     std::unique_ptr<FSWritableFile>* result);
  // Syncs of the whole file pass this as `bytes` to Delayed.
  static constexpr uint64_t kAllUnsyncedBytes = UINT64_MAX;

  // Delays, then runs `op`, then records the operation. For appends `bytes`
  // is the appended size; for syncs it caps the unsynced bytes the sync is
  // charged for (RangeSync's nbytes, or kAllUnsyncedBytes for Sync/Fsync).
  template <typename Op>
  IOStatus Delayed(WalState* state, bool is_sync, uint64_t bytes, Op op);
  // Atomically claims up to `limit` unsynced bytes and returns the claim.
  static uint64_t ClaimUnsyncedBytes(WalState* state, uint64_t limit);

  const uint64_t seed_;
  const std::shared_ptr<SystemClock> clock_;
  std::atomic<uint64_t> default_append_micros_;
  std::atomic<uint64_t> default_sync_micros_;
  std::atomic<uint64_t> default_jitter_micros_;
  std::atomic<uint64_t> default_append_bytes_per_sec_;
  std::atomic<uint64_t> default_sync_bytes_per_sec_;
  mutable port::Mutex mutex_;
  std::map<uint64_t, std::shared_ptr<WalState>> wals_;
};

}  // namespace ROCKSDB_NAMESPACE
