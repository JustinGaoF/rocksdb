//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "db/read_callback.h"

namespace ROCKSDB_NAMESPACE {

// N>1 only. Hides from one read every sequence range whose sync write failed
// its gate fsync after it was inserted and published (D-8): the write has
// returned its error, so no read that starts afterwards may see it. Wraps
// the visibility callback the read already has, if any (a transaction's or
// the timestamp one), and defers to it for every sequence outside the
// ranges. The ranges are immutable and shared, so an iterator keeps hiding
// them across Refresh().
class WalGateReadMask : public ReadCallback {
 public:
  using Ranges = std::vector<std::pair<SequenceNumber, SequenceNumber>>;

  // `snapshot` is the read's sequence, used when there is no `inner`.
  WalGateReadMask(std::shared_ptr<const Ranges> ranges, ReadCallback* inner,
                  SequenceNumber snapshot)
      : ReadCallback(inner != nullptr ? inner->max_visible_seq() : snapshot,
                     MinVisibleWithoutCheck(*ranges, inner, snapshot)),
        ranges_(std::move(ranges)),
        inner_(inner) {}

  bool IsVisibleFullCheck(SequenceNumber seq) override {
    return !Masked(seq) && (inner_ == nullptr || inner_->IsVisible(seq));
  }

  void Refresh(SequenceNumber seq) override {
    if (inner_ != nullptr) {
      inner_->Refresh(seq);
      max_visible_seq_ = inner_->max_visible_seq();
    } else {
      max_visible_seq_ = seq;
    }
  }

  const MetadataReadBounds* GetMetadataReadBounds() const override {
    return inner_ != nullptr ? inner_->GetMetadataReadBounds() : nullptr;
  }

  bool NeedToTrackNewerVersions(
      const bool* per_key_result = nullptr) const override {
    return inner_ != nullptr &&
           inner_->NeedToTrackNewerVersions(per_key_result);
  }

  bool IsNewerVisibleForMetadataRead(SequenceNumber seq) override {
    return !Masked(seq) && inner_ != nullptr &&
           inner_->IsNewerVisibleForMetadataRead(seq);
  }

  void MaybeRecordNewerVersion(SequenceNumber seq, ValueType type,
                               bool* per_key_result = nullptr) override {
    if (!Masked(seq) && inner_ != nullptr) {
      inner_->MaybeRecordNewerVersion(seq, type, per_key_result);
    }
  }

 private:
  // Below every range, and below anything `inner` must check (or above the
  // snapshot without one), a sequence is visible without a check.
  static SequenceNumber MinVisibleWithoutCheck(const Ranges& ranges,
                                               ReadCallback* inner,
                                               SequenceNumber snapshot) {
    SequenceNumber min =
        inner != nullptr
            ? inner->min_uncommitted()
            : std::max<SequenceNumber>(snapshot + 1, kMinUnCommittedSeq);
    for (const auto& range : ranges) {
      min = std::min(min, std::max(range.first, kMinUnCommittedSeq));
    }
    return min;
  }

  bool Masked(SequenceNumber seq) const {
    for (const auto& range : *ranges_) {
      if (range.first <= seq && seq <= range.second) {
        return true;
      }
    }
    return false;
  }

  const std::shared_ptr<const Ranges> ranges_;
  ReadCallback* const inner_;
};

// As WalGateReadMask, for a tailing iterator, which reads at the latest
// sequence and so must hide ranges masked after it was created. Every
// sequence takes the full check, which reloads the ranges from `source`
// only when `version` has moved since the last load.
class LiveWalGateReadMask : public ReadCallback {
 public:
  using Ranges = WalGateReadMask::Ranges;

  LiveWalGateReadMask(const std::atomic<uint64_t>* version,
                      std::function<std::shared_ptr<const Ranges>()> source)
      : ReadCallback(kMaxSequenceNumber, kMinUnCommittedSeq),
        version_(version),
        source_(std::move(source)) {}

  bool IsVisibleFullCheck(SequenceNumber seq) override {
    const uint64_t version = version_->load(std::memory_order_acquire);
    if (version != loaded_version_) {
      ranges_ = source_();
      loaded_version_ = version;
    }
    if (ranges_ == nullptr) {
      return true;
    }
    for (const auto& range : *ranges_) {
      if (range.first <= seq && seq <= range.second) {
        return false;
      }
    }
    return true;
  }

  void Refresh(SequenceNumber /*seq*/) override {}

 private:
  const std::atomic<uint64_t>* const version_;
  const std::function<std::shared_ptr<const Ranges>()> source_;
  uint64_t loaded_version_ = std::numeric_limits<uint64_t>::max();
  std::shared_ptr<const Ranges> ranges_;
};

}  // namespace ROCKSDB_NAMESPACE
