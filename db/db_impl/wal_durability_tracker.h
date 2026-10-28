//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "port/port.h"
#include "rocksdb/rocksdb_namespace.h"
#include "test_util/sync_point.h"

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
// A write that consumes no sequence (a WAL-only marker stamped with
// last_allocated + 1 without consuming it) registers the empty range
// [low, low - 1]: it pins `low` like any entry, but is durable or applied
// only through `low - 1`, so no safe point or published sequence ever
// exceeds an in-flight range's low minus one. Registering it as [low, low]
// instead would let an all-idle completion advance the monotonic safe point
// past a sequence nobody owns yet, publishing over the range that later
// takes it.
//
// The safe point lives in this tracker's own atomic and is exposed through a
// single DBImpl accessor. It deliberately shares no field with
// last_published_sequence_, which ingestion and recovery legitimately jump
// forward; the watermark is a lower bound whose worth is that it cannot be
// overstated. DBImpl's completion gate waits on it. DBImpl also keeps a second
// instance whose ranges complete after the memtable insert instead; that
// instance's safe point bounds what N>1 publishes (see PublishAppliedPrefix).
//
// Threading: same-partition Register/Complete calls must be externally
// synchronized (in DBImpl, sequence_wal_index_mutex_ serializes every tracker
// op, which also orders each registration before any later scan so no scan
// can miss an outstanding range). Cross-partition concurrency is lock-free:
// pending_low/durable_upto use release stores on the completing writer and
// acquire loads on the scanner, so observing an advanced frontier observes
// the WAL append that advanced it. min_owner_ is a relaxed hint; a stale read
// only costs an extra rescan or a conservatively delayed advance, both safe.
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
    min_owner_.store(0, std::memory_order_relaxed);
  }

  // No copying or moving: DBImpl owns one instance sized at construction.
  WalDurabilityTracker(const WalDurabilityTracker&) = delete;
  void operator=(const WalDurabilityTracker&) = delete;

  size_t num_partitions() const { return partitions_.size(); }

  // Records [low, high] as outstanding on partition p. Must be called before
  // the allocation's counter advance is visible to other partitions' scans;
  // in DBImpl this runs inside AllocateSequenceAndWALIndex under
  // sequence_wal_index_mutex_. Never lowers the safe point: a newly allocated
  // range is the highest yet, so the global minimum is unchanged (or moves
  // from infinity to low - 1 == old maximum when leaving all-idle, which is
  // also unchanged). Updates min_owner_ on that leaving-all-idle transition
  // so a later Complete by another partition does not skip its rescan.
  // `high` may be `low - 1` for a write that consumes no sequence (an empty
  // range, pinning `low` while durable only through `low - 1`); `low` is then
  // an allocated-plus-one sequence, never 0.
  void Register(size_t partition, uint64_t low, uint64_t high) {
    assert(partition < partitions_.size());
    assert(low <= high + 1);
    assert(low > 0 || low <= high);
    assert(high != kIdle);
    std::vector<Range>& fifo = partitions_[partition].inflight;
    fifo.push_back({low, high});
    uint64_t new_pending = kIdle;
    for (const Range& r : fifo) {
      if (r.low < new_pending) {
        new_pending = r.low;
      }
    }
    partitions_[partition].pending_low.store(new_pending,
                                             std::memory_order_release);
    // Leaving all-idle: the minimum was infinity, so it is now this range.
    size_t owner = min_owner_.load(std::memory_order_relaxed);
    if (partitions_[owner].pending_low.load(std::memory_order_acquire) ==
            kIdle &&
        new_pending != kIdle) {
      min_owner_.store(partition, std::memory_order_relaxed);
    }
  }

  // Marks one occurrence of [low, high] on partition p durable after its
  // write() completed. Duplicate entries (concurrent WAL-only markers stamped
  // with the same last_allocated + 1) are stored separately: the minimum is
  // indifferent to duplicates and does not advance past the shared value
  // until every entry carrying it completes, which is correct since an
  // undurable commit marker strands its transaction. Completing an empty
  // [low, low - 1] range advances durable_upto only to low - 1. Recomputes
  // the safe point unless the partition's minimum is provably unchanged and
  // unowned.
  void Complete(size_t partition, uint64_t low, uint64_t high) {
    assert(partition < partitions_.size());
    std::vector<Range>& fifo = partitions_[partition].inflight;
    size_t found = fifo.size();
    for (size_t i = 0; i < fifo.size(); ++i) {
      if (fifo[i].low == low && fifo[i].high == high) {
        found = i;
        break;
      }
    }
    if (found == fifo.size()) {
      // Unpaired completion: a bug in the caller. Assert in debug; tolerate
      // in release rather than corrupting the frontier.
      assert(false);
      return;
    }
    const uint64_t old_pending =
        partitions_[partition].pending_low.load(std::memory_order_relaxed);
    fifo[found] = fifo.back();
    fifo.pop_back();
    const uint64_t cur_durable =
        partitions_[partition].durable_upto.load(std::memory_order_relaxed);
    if (high > cur_durable) {
      partitions_[partition].durable_upto.store(high,
                                                std::memory_order_release);
    }
    uint64_t new_pending = kIdle;
    for (const Range& r : fifo) {
      if (r.low < new_pending) {
        new_pending = r.low;
      }
    }
    partitions_[partition].pending_low.store(new_pending,
                                             std::memory_order_release);
    if (new_pending == old_pending) {
      // Minimum unchanged (a duplicate remains, or several entries shared
      // the old minimum): the global frontier cannot have moved.
      return;
    }
    Recompute(partition, new_pending);
  }

  // Drops [low, high] on partition p as Complete does, if it is still
  // registered. For a range whose WAL append failed, once recovery from that
  // failure succeeded: it holds no data, so it must stop holding the frontier
  // below every later write. Unlike Complete, an absent range is not a bug (a
  // group whose append landed and whose sync failed completed it already).
  void Release(size_t partition, uint64_t low, uint64_t high) {
    assert(partition < partitions_.size());
    for (const Range& r : partitions_[partition].inflight) {
      if (r.low == low && r.high == high) {
        Complete(partition, low, high);
        return;
      }
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

  // Full rescan ignoring the min-owner shortcut. Tests use it after joining
  // concurrent threads to shake out stale-hint delays deterministically; the
  // write path never needs it because the owning partition's own completion
  // always rescans.
  void ForceRecalculate() {
    uint64_t min_pending = kIdle;
    size_t owner = 0;
    for (size_t p = 0; p < partitions_.size(); ++p) {
      const uint64_t v =
          partitions_[p].pending_low.load(std::memory_order_acquire);
      if (v < min_pending) {
        min_pending = v;
        owner = p;
      }
    }
    min_owner_.store(owner, std::memory_order_relaxed);
    if (min_pending == kIdle) {
      uint64_t max_durable = 0;
      for (size_t p = 0; p < partitions_.size(); ++p) {
        const uint64_t d =
            partitions_[p].durable_upto.load(std::memory_order_acquire);
        if (d > max_durable) {
          max_durable = d;
        }
      }
      AdvanceSafePoint(max_durable);
    } else {
      uint64_t candidate = min_pending - 1;
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
                                  cur, candidate, std::memory_order_release,
                                  std::memory_order_acquire)) {
    }
  }

  // Recomputes W = min(pending_low) after partition p advanced to
  // new_pending. A linear scan over the contiguous atomic array beats a heap
  // at N = 4-32 on cache locality; when p does not own the minimum and is
  // still at or above it, the minimum is unchanged and the scan is skipped.
  void Recompute(size_t completed_partition, uint64_t new_pending) {
    const size_t owner = min_owner_.load(std::memory_order_relaxed);
    if (completed_partition != owner) {
      const uint64_t owner_pending =
          partitions_[owner].pending_low.load(std::memory_order_acquire);
      if (new_pending >= owner_pending) {
        return;
      }
    }
    ForceRecalculate();
  }

  // One partition's state, written by that partition's leader on every
  // register and complete. Each starts its own cache line, so leaders of
  // different partitions never write a shared line (no false sharing); the
  // rescan reads one line per partition.
  struct ALIGN_AS(CACHE_LINE_SIZE) Partition {
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
  std::atomic<size_t> min_owner_{0};
  const bool safe_point_test_hook_;
};

}  // namespace ROCKSDB_NAMESPACE
