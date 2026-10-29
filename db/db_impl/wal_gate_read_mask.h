//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <algorithm>
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

  const Ranges* GetHiddenSeqRanges() const override { return ranges_.get(); }

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

}  // namespace ROCKSDB_NAMESPACE
