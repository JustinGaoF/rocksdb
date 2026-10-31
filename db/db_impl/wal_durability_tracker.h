//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "port/port.h"
#include "rocksdb/rocksdb_namespace.h"
#include "test_util/sync_point.h"
#include "util/mutexlock.h"

namespace ROCKSDB_NAMESPACE {

// Tracks per-partition WAL durability so that N independently-failing WAL
// files cannot leave holes: sequences that are missing but not at the tail,
// which a replica reporting its high-water mark would never detect.
//
// Each partition registers the [low, high] sequence range it allocates before
// the shared counter advance is visible, and completes it after its WAL
// append's write() returns (sync=false; fsync is a later threshold). The
// global frontier is W = min(pending_low) with safe point W - 1: every
// sequence at or below the safe point is durable on every partition that was
// assigned anything at or below it. An idle partition contributes infinity so
// it never drags the minimum down; when all partitions are idle the safe
// point is the maximum durable sequence instead.
//
// The safe point lives in this tracker's own atomic and is exposed through a
// single DBImpl accessor. It deliberately shares no field with
// last_published_sequence_, which ingestion and recovery legitimately jump
// forward; the watermark is a lower bound whose worth is that it cannot be
// overstated. DBImpl's completion gate waits on it. DBImpl also keeps a second
// instance whose ranges complete after the memtable insert instead; that
// instance's safe point bounds what N>1 publishes (see PublishAppliedPrefix).
//
// Threading: each partition's in-flight ranges are guarded by that
// partition's own mutex, so its Register/Complete/Release calls may come
// from any thread. Ranges must be registered in allocation order (in DBImpl,
// under sequence_wal_index_mutex_, which issues them). Cross-partition
// concurrency is lock-free: a scan first reads the highest registered
// sequence, then every partition's pending low. A range registered before
// that read is visible to the scan, and one registered after it lies above
// it, so bounding the safe point by it means no scan passes an outstanding
// range. Pending lows are stored and loaded sequentially consistent, so of
// two partitions completing at once, the later scan sees both; observing an
// advanced frontier observes the WAL append that advanced it.
class WalDurabilityTracker {
 public:
  // Idle partitions contribute infinity to the minimum. A separate in-flight
  // flag would cost a second array and a branch per entry on the rescan path;
  // the sentinel keeps the scan a single min over one contiguous array with
  // one all-idle check at the end.
  static constexpr uint64_t kIdle = std::numeric_limits<uint64_t>::max();

  // `safe_point_test_hook` exposes the safe point to the prefix oracle's
  // off-by-one self-test, which targets the WAL durability watermark only.
  explicit WalDurabilityTracker(size_t num_partitions = 1,
                                bool safe_point_test_hook = true)
      : partitions_(num_partitions),
        safe_point_test_hook_(safe_point_test_hook) {
    assert(num_partitions > 0);
    for (size_t p = 0; p < num_partitions; ++p) {
      partitions_[p].pending_low.store(kIdle, std::memory_order_relaxed);
      partitions_[p].durable_upto.store(0, std::memory_order_relaxed);
      // In-flight depth is typically one to four groups; reserving up front
      // keeps the hot path allocation-free.
      partitions_[p].inflight.reserve(4);
    }
    safe_point_.store(0, std::memory_order_relaxed);
    registered_high_.store(0, std::memory_order_relaxed);
  }

  // No copying or moving: DBImpl owns one instance sized at construction.
  WalDurabilityTracker(const WalDurabilityTracker&) = delete;
  void operator=(const WalDurabilityTracker&) = delete;

  size_t num_partitions() const { return partitions_.size(); }

  // Records [low, high] as outstanding on partition p. Ranges register in
  // allocation order; in DBImpl this runs inside AllocateSequenceAndWALIndex
  // under sequence_wal_index_mutex_, before the allocating writer does
  // anything a scan could observe. Never lowers the safe point: a newly
  // allocated range is the highest yet, so the global minimum is unchanged
  // (or moves from infinity to low - 1 == old maximum when leaving
  // all-idle, which is also unchanged).
  // high == low - 1 is an empty range (a record that consumes no sequence):
  // it pins low while outstanding, and completes durable only through
  // low - 1, the last sequence allocated when it registered.
  void Register(size_t partition, uint64_t low, uint64_t high) {
    assert(partition < partitions_.size());
    assert(low <= high + 1);
    assert(high != kIdle);
    Partition& part = partitions_[partition];
    {
      MutexLock lock(&part.mu);
      part.inflight.push_back({low, high});
      uint64_t new_pending = kIdle;
      for (const Range& r : part.inflight) {
        if (r.low < new_pending) {
          new_pending = r.low;
        }
      }
      part.pending_low.store(new_pending, std::memory_order_seq_cst);
    }
    // After the pending low: a scan that reads this bound sees the range.
    uint64_t cur = registered_high_.load(std::memory_order_relaxed);
    while (high > cur && !registered_high_.compare_exchange_weak(
                             cur, high, std::memory_order_seq_cst,
                             std::memory_order_relaxed)) {
    }
  }

  // Marks one occurrence of [low, high] on partition p durable after its
  // write() completed. Duplicate lows (a WAL-only marker's empty range
  // [last_allocated + 1, last_allocated] shares its low with the next real
  // write) are stored as separate entries: the minimum is indifferent
  // to duplicates and does not advance past the shared value until every
  // entry carrying it completes, which is correct since an undurable commit
  // marker strands its transaction. Recomputes the safe point unless the
  // partition's minimum is unchanged.
  void Complete(size_t partition, uint64_t low, uint64_t high) {
    assert(partition < partitions_.size());
    bool found = false;
    if (Remove(partition, low, high, &found)) {
      ForceRecalculate();
    }
    // Unpaired completion: a bug in the caller. Assert in debug; tolerate in
    // release rather than corrupting the frontier.
    assert(found);
  }

  // Drops [low, high] on partition p as Complete does, if it is still
  // registered. For a range whose WAL append failed, once recovery from that
  // failure succeeded: it holds no data, so it must stop holding the frontier
  // below every later write. Unlike Complete, an absent range is not a bug (a
  // group whose append landed and whose sync failed completed it already).
  void Release(size_t partition, uint64_t low, uint64_t high) {
    assert(partition < partitions_.size());
    bool found = false;
    if (Remove(partition, low, high, &found)) {
      ForceRecalculate();
    }
  }

  // Current durability frontier: every sequence at or below it is durable.
  // Monotonic; 0 before anything completes.
  uint64_t SafePoint() const {
    return safe_point_.load(std::memory_order_acquire);
  }

  uint64_t PendingLow(size_t partition) const {
    assert(partition < partitions_.size());
    return partitions_[partition].pending_low.load(std::memory_order_acquire);
  }

  uint64_t DurableUpto(size_t partition) const {
    assert(partition < partitions_.size());
    return partitions_[partition].durable_upto.load(std::memory_order_acquire);
  }

  // Full rescan, which every completion that moved its partition's minimum
  // runs. Tests also call it after joining concurrent threads.
  void ForceRecalculate() {
    // First: every range at or below it is visible below (see Threading).
    const uint64_t registered =
        registered_high_.load(std::memory_order_seq_cst);
    uint64_t min_pending = kIdle;
    for (size_t p = 0; p < partitions_.size(); ++p) {
      const uint64_t v =
          partitions_[p].pending_low.load(std::memory_order_seq_cst);
      if (v < min_pending) {
        min_pending = v;
      }
    }
    if (min_pending == kIdle) {
      uint64_t max_durable = 0;
      for (size_t p = 0; p < partitions_.size(); ++p) {
        const uint64_t d =
            partitions_[p].durable_upto.load(std::memory_order_acquire);
        if (d > max_durable) {
          max_durable = d;
        }
      }
      AdvanceSafePoint(std::min(max_durable, registered));
    } else {
      uint64_t candidate = std::min(min_pending - 1, registered);
      if (safe_point_test_hook_) {
        // Lets the prefix oracle's self-test shift the safe point by one.
        TEST_SYNC_POINT_CALLBACK(
            "WalDurabilityTracker::ForceRecalculate:Pending", &candidate);
      }
      AdvanceSafePoint(candidate);
    }
  }

 private:
  struct Range {
    uint64_t low;
    uint64_t high;
  };

  void AdvanceSafePoint(uint64_t candidate) {
    uint64_t cur = safe_point_.load(std::memory_order_acquire);
    while (candidate > cur && !safe_point_.compare_exchange_weak(
                                  cur, candidate, std::memory_order_seq_cst,
                                  std::memory_order_acquire)) {
    }
  }

  // Removes one occurrence of [low, high] from partition p, setting *found.
  // Returns whether p's minimum moved, so the global frontier may have.
  bool Remove(size_t partition, uint64_t low, uint64_t high, bool* found) {
    Partition& part = partitions_[partition];
    MutexLock lock(&part.mu);
    std::vector<Range>& fifo = part.inflight;
    size_t index = fifo.size();
    for (size_t i = 0; i < fifo.size(); ++i) {
      if (fifo[i].low == low && fifo[i].high == high) {
        index = i;
        break;
      }
    }
    *found = index != fifo.size();
    if (!*found) {
      return false;
    }
    const uint64_t old_pending =
        part.pending_low.load(std::memory_order_relaxed);
    fifo[index] = fifo.back();
    fifo.pop_back();
    const uint64_t cur_durable =
        part.durable_upto.load(std::memory_order_relaxed);
    if (high > cur_durable) {
      part.durable_upto.store(high, std::memory_order_release);
    }
    uint64_t new_pending = kIdle;
    for (const Range& r : fifo) {
      if (r.low < new_pending) {
        new_pending = r.low;
      }
    }
    part.pending_low.store(new_pending, std::memory_order_seq_cst);
    // Minimum unchanged (a duplicate remains, or several entries shared the
    // old minimum): the global frontier cannot have moved.
    return new_pending != old_pending;
  }

  // One partition's state, written on every register and complete of its
  // ranges. Each starts its own cache line, so leaders of different
  // partitions never write a shared line (no false sharing); the rescan
  // reads one line per partition.
  struct ALIGN_AS(CACHE_LINE_SIZE) Partition {
    // Guards `inflight` and orders the stores below for one partition.
    port::Mutex mu;
    // Lowest outstanding low, or kIdle when empty.
    std::atomic<uint64_t> pending_low{kIdle};
    // Highest completed high. Only read when all partitions are idle, where
    // the safe point is their maximum: with nothing outstanding, every
    // allocated sequence completed, so the maximum completed is the maximum
    // allocated and everything at or below it is durable.
    std::atomic<uint64_t> durable_upto{0};
    // In-flight ranges, in no particular order; the minimum is recomputed by
    // linear scan on removal, which is O(depth) at depth <= 4. Per-partition
    // rather than shared: a shared structure would serialize all
    // partitions' completions and mix unrelated head-of-line blocking.
    std::vector<Range> inflight;
  };
  static_assert(alignof(Partition) >= CACHE_LINE_SIZE);
  static_assert(sizeof(Partition) % CACHE_LINE_SIZE == 0);

  std::vector<Partition> partitions_;
  std::atomic<uint64_t> safe_point_{0};
  // The highest sequence registered so far, which bounds every scan.
  std::atomic<uint64_t> registered_high_{0};
  const bool safe_point_test_hook_;
};

}  // namespace ROCKSDB_NAMESPACE
