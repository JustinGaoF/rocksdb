//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "utilities/latency_injection_fs.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <thread>

#include "file/filename.h"
#include "util/coding.h"
#include "util/hash.h"
#include "util/mutexlock.h"

namespace ROCKSDB_NAMESPACE {

namespace {
uint64_t JitterMicros(uint64_t seed, uint64_t wal_number, uint64_t op_index,
                      uint64_t jitter_micros) {
  char key[16];
  EncodeFixed64(key, wal_number);
  EncodeFixed64(key + 8, op_index);
  return Hash64(key, sizeof(key), seed) % (jitter_micros + 1);
}
}  // namespace

template <typename Op>
IOStatus LatencyInjectionFileSystem::Delayed(WalState* state, bool is_sync,
                                             uint64_t bytes, Op op) {
  const uint64_t start = clock_->NowMicros();
  const bool has_override = state->has_override.load(std::memory_order_acquire);
  uint64_t delay;
  uint64_t jitter;
  if (has_override) {
    delay = is_sync ? state->sync_micros.load(std::memory_order_relaxed)
                    : state->append_micros.load(std::memory_order_relaxed);
    jitter = state->jitter_micros.load(std::memory_order_relaxed);
  } else {
    delay = is_sync ? default_sync_micros_.load(std::memory_order_relaxed)
                    : default_append_micros_.load(std::memory_order_relaxed);
    jitter = default_jitter_micros_.load(std::memory_order_relaxed);
  }
  const uint64_t op_index =
      state->next_op.fetch_add(1, std::memory_order_relaxed);
  if (jitter > 0) {
    delay += JitterMicros(seed_, state->wal_number, op_index, jitter);
  }
  if (delay > 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(delay));
  }
  IOStatus s = op();
  const uint64_t wall = clock_->NowMicros() - start;
  const uint64_t ok_bytes = s.ok() ? bytes : 0;
  state->counters.Record(is_sync, ok_bytes, delay, wall);
  return s;
}

class LatencyInjectionFileSystem::WalFile : public FSWritableFileOwnerWrapper {
 public:
  WalFile(std::unique_ptr<FSWritableFile>&& file,
          LatencyInjectionFileSystem* fs, std::shared_ptr<WalState> state)
      : FSWritableFileOwnerWrapper(std::move(file)),
        fs_(fs),
        state_(std::move(state)) {}

  IOStatus Append(const Slice& data, const IOOptions& options,
                  IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/false, data.size(),
                        [&] { return target()->Append(data, options, dbg); });
  }

  IOStatus Append(const Slice& data, const IOOptions& options,
                  const DataVerificationInfo& info,
                  IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/false, data.size(), [&] {
      return target()->Append(data, options, info, dbg);
    });
  }

  IOStatus PositionedAppend(const Slice& data, uint64_t offset,
                            const IOOptions& options,
                            IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/false, data.size(), [&] {
      return target()->PositionedAppend(data, offset, options, dbg);
    });
  }

  IOStatus PositionedAppend(const Slice& data, uint64_t offset,
                            const IOOptions& options,
                            const DataVerificationInfo& info,
                            IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/false, data.size(), [&] {
      return target()->PositionedAppend(data, offset, options, info, dbg);
    });
  }

  IOStatus Sync(const IOOptions& options, IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/true, /*bytes=*/0,
                        [&] { return target()->Sync(options, dbg); });
  }

  IOStatus Fsync(const IOOptions& options, IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/true, /*bytes=*/0,
                        [&] { return target()->Fsync(options, dbg); });
  }

  IOStatus RangeSync(uint64_t offset, uint64_t nbytes, const IOOptions& options,
                     IODebugContext* dbg) override {
    return fs_->Delayed(state_.get(), /*is_sync=*/true, /*bytes=*/0, [&] {
      return target()->RangeSync(offset, nbytes, options, dbg);
    });
  }

 private:
  LatencyInjectionFileSystem* const fs_;
  const std::shared_ptr<WalState> state_;
};

void LatencyInjectionFileSystem::AtomicCounters::Record(bool is_sync,
                                                        uint64_t bytes,
                                                        uint64_t injected,
                                                        uint64_t wall) {
  if (is_sync) {
    syncs.fetch_add(1, std::memory_order_relaxed);
  } else {
    appends.fetch_add(1, std::memory_order_relaxed);
    append_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  if (injected > 0) {
    injected_sleep_micros.fetch_add(injected, std::memory_order_relaxed);
  }
  wall_micros.fetch_add(wall, std::memory_order_relaxed);
}

LatencyInjectionFileSystem::Counters
LatencyInjectionFileSystem::AtomicCounters::Load() const {
  Counters c;
  c.appends = appends.load(std::memory_order_relaxed);
  c.append_bytes = append_bytes.load(std::memory_order_relaxed);
  c.syncs = syncs.load(std::memory_order_relaxed);
  c.injected_sleep_micros =
      injected_sleep_micros.load(std::memory_order_relaxed);
  c.wall_micros = wall_micros.load(std::memory_order_relaxed);
  return c;
}

void LatencyInjectionFileSystem::AtomicCounters::Reset() {
  appends.store(0, std::memory_order_relaxed);
  append_bytes.store(0, std::memory_order_relaxed);
  syncs.store(0, std::memory_order_relaxed);
  injected_sleep_micros.store(0, std::memory_order_relaxed);
  wall_micros.store(0, std::memory_order_relaxed);
}

std::string LatencyInjectionFileSystem::Counters::ToString() const {
  char buf[256];
  snprintf(buf, sizeof(buf),
           "appends=%" PRIu64 " append_bytes=%" PRIu64 " syncs=%" PRIu64
           " injected_sleep_micros=%" PRIu64 " wall_micros=%" PRIu64,
           appends, append_bytes, syncs, injected_sleep_micros, wall_micros);
  return buf;
}

LatencyInjectionFileSystem::LatencyInjectionFileSystem(
    const std::shared_ptr<FileSystem>& base, const Latency& default_latency,
    uint64_t seed)
    : FileSystemWrapper(base),
      seed_(seed),
      clock_(SystemClock::Default()),
      default_append_micros_(default_latency.append_micros),
      default_sync_micros_(default_latency.sync_micros),
      default_jitter_micros_(default_latency.jitter_micros) {}

LatencyInjectionFileSystem::~LatencyInjectionFileSystem() = default;

std::shared_ptr<LatencyInjectionFileSystem::WalState>
LatencyInjectionFileSystem::GetOrCreateState(uint64_t wal_number) {
  MutexLock l(&mutex_);
  std::shared_ptr<WalState>& state = wals_[wal_number];
  if (state == nullptr) {
    state = std::make_shared<WalState>(wal_number);
  }
  return state;
}

std::shared_ptr<LatencyInjectionFileSystem::WalState>
LatencyInjectionFileSystem::StateForFile(const std::string& fname) {
  const size_t slash = fname.find_last_of('/');
  const std::string base_name =
      slash == std::string::npos ? fname : fname.substr(slash + 1);
  uint64_t number = 0;
  FileType type;
  if (!ParseFileName(base_name, &number, &type) || type != kWalFile) {
    return nullptr;
  }
  return GetOrCreateState(number);
}

IOStatus LatencyInjectionFileSystem::WrapIfWal(
    const std::string& fname, IOStatus s,
    std::unique_ptr<FSWritableFile>* result) {
  if (!s.ok()) {
    return s;
  }
  std::shared_ptr<WalState> state = StateForFile(fname);
  if (state != nullptr) {
    result->reset(new WalFile(std::move(*result), this, std::move(state)));
  }
  return s;
}

IOStatus LatencyInjectionFileSystem::NewWritableFile(
    const std::string& fname, const FileOptions& file_opts,
    std::unique_ptr<FSWritableFile>* result, IODebugContext* dbg) {
  return WrapIfWal(
      fname, target()->NewWritableFile(fname, file_opts, result, dbg), result);
}

IOStatus LatencyInjectionFileSystem::ReopenWritableFile(
    const std::string& fname, const FileOptions& file_opts,
    std::unique_ptr<FSWritableFile>* result, IODebugContext* dbg) {
  return WrapIfWal(fname,
                   target()->ReopenWritableFile(fname, file_opts, result, dbg),
                   result);
}

IOStatus LatencyInjectionFileSystem::ReuseWritableFile(
    const std::string& fname, const std::string& old_fname,
    const FileOptions& file_opts, std::unique_ptr<FSWritableFile>* result,
    IODebugContext* dbg) {
  return WrapIfWal(
      fname,
      target()->ReuseWritableFile(fname, old_fname, file_opts, result, dbg),
      result);
}

void LatencyInjectionFileSystem::SetDefaultLatency(const Latency& latency) {
  default_append_micros_.store(latency.append_micros,
                               std::memory_order_relaxed);
  default_sync_micros_.store(latency.sync_micros, std::memory_order_relaxed);
  default_jitter_micros_.store(latency.jitter_micros,
                               std::memory_order_relaxed);
}

void LatencyInjectionFileSystem::SetWalLatency(uint64_t wal_number,
                                               const Latency& latency) {
  std::shared_ptr<WalState> state = GetOrCreateState(wal_number);
  state->append_micros.store(latency.append_micros, std::memory_order_relaxed);
  state->sync_micros.store(latency.sync_micros, std::memory_order_relaxed);
  state->jitter_micros.store(latency.jitter_micros, std::memory_order_relaxed);
  state->has_override.store(true, std::memory_order_release);
}

void LatencyInjectionFileSystem::ClearWalLatency(uint64_t wal_number) {
  GetOrCreateState(wal_number)
      ->has_override.store(false, std::memory_order_release);
}

LatencyInjectionFileSystem::Counters LatencyInjectionFileSystem::GetWalCounters(
    uint64_t wal_number) const {
  MutexLock l(&mutex_);
  const std::map<uint64_t, std::shared_ptr<WalState>>::const_iterator it =
      wals_.find(wal_number);
  return it == wals_.end() ? Counters() : it->second->counters.Load();
}

LatencyInjectionFileSystem::Counters
LatencyInjectionFileSystem::GetAggregateCounters() const {
  MutexLock l(&mutex_);
  Counters total;
  for (const auto& entry : wals_) {
    const Counters c = entry.second->counters.Load();
    total.appends += c.appends;
    total.append_bytes += c.append_bytes;
    total.syncs += c.syncs;
    total.injected_sleep_micros += c.injected_sleep_micros;
    total.wall_micros += c.wall_micros;
  }
  return total;
}

std::vector<uint64_t> LatencyInjectionFileSystem::GetWalNumbers() const {
  MutexLock l(&mutex_);
  std::vector<uint64_t> numbers;
  numbers.reserve(wals_.size());
  for (const auto& entry : wals_) {
    numbers.push_back(entry.first);
  }
  return numbers;
}

std::string LatencyInjectionFileSystem::PrintCounters() const {
  std::string out = "all WALs: " + GetAggregateCounters().ToString() + "\n";
  for (uint64_t number : GetWalNumbers()) {
    out += "WAL " + std::to_string(number) + ": " +
           GetWalCounters(number).ToString() + "\n";
  }
  return out;
}

void LatencyInjectionFileSystem::ResetCounters() {
  MutexLock l(&mutex_);
  for (auto& entry : wals_) {
    entry.second->counters.Reset();
  }
}

}  // namespace ROCKSDB_NAMESPACE
