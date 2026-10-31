// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <functional>

#include "db/dbformat.h"
#include "rocksdb/types.h"

namespace ROCKSDB_NAMESPACE {

struct MetadataReadBounds {
  // The snapshot-visible sequence and the upper bound sampled for this read.
  // Lookups scan through the upper bound but return data only through the
  // snapshot sequence.
  SequenceNumber read_snapshot_seq;
  SequenceNumber newer_version_upper_bound_seq;
};

class ReadCallback {
 public:
  explicit ReadCallback(SequenceNumber last_visible_seq)
      : max_visible_seq_(last_visible_seq) {}
  ReadCallback(SequenceNumber last_visible_seq, SequenceNumber min_uncommitted)
      : max_visible_seq_(last_visible_seq), min_uncommitted_(min_uncommitted) {}

  virtual ~ReadCallback() = default;

  // Will be called to see if the seq number visible; if not it moves on to
  // the next seq number.
  virtual bool IsVisibleFullCheck(SequenceNumber seq) = 0;

  inline bool IsVisible(SequenceNumber seq) {
    assert(min_uncommitted_ > 0);
    assert(min_uncommitted_ >= kMinUnCommittedSeq);
    if (seq < min_uncommitted_) {  // handles seq == 0 as well
      assert(seq <= max_visible_seq_);
      return true;
    } else if (max_visible_seq_ < seq) {
      assert(seq != 0);
      return false;
    } else {
      assert(seq != 0);  // already handled in the first if-then clause
      return IsVisibleFullCheck(seq);
    }
  }

  virtual const MetadataReadBounds* GetMetadataReadBounds() const {
    return nullptr;
  }

  virtual bool NeedToTrackNewerVersions(
      const bool* /*per_key_result*/ = nullptr) const {
    return false;
  }

  virtual bool IsNewerVisibleForMetadataRead(SequenceNumber /*seq*/) {
    // Metadata reads with custom visibility callbacks are unsupported. The
    // timestamp callback overrides this method for regular DB reads.
    return false;
  }

  virtual void MaybeRecordNewerVersion(SequenceNumber /*seq*/,
                                       ValueType /*type*/,
                                       bool* /*per_key_result*/ = nullptr) {}

  inline SequenceNumber max_visible_seq() { return max_visible_seq_; }

  inline SequenceNumber min_uncommitted() const { return min_uncommitted_; }

  // Refresh to a more recent visible seq
  virtual void Refresh(SequenceNumber seq) { max_visible_seq_ = seq; }

 protected:
  // The max visible seq, it is usually the snapshot but could be larger if
  // transaction has its own writes written to db.
  SequenceNumber max_visible_seq_ = kMaxSequenceNumber;
  // Any seq less than min_uncommitted_ is committed.
  const SequenceNumber min_uncommitted_ = kMinUnCommittedSeq;
};

// A tailing-iterator bound at N>1: a write there inserts before it is
// published, so visibility follows the published sequence, sampled live on
// every check so newly published writes appear without a refresh. (As a side
// effect DBIter skips inserting logically redundant range tombstones for
// these iterators, which is a read optimization, not a behavior change.)
class PublishedSequenceReadCallback : public ReadCallback {
 public:
  explicit PublishedSequenceReadCallback(
      std::function<SequenceNumber()> published_sequence)
      : ReadCallback(kMaxSequenceNumber),
        published_sequence_(std::move(published_sequence)) {}

  bool IsVisibleFullCheck(SequenceNumber seq) override {
    return seq <= published_sequence_();
  }

 private:
  std::function<SequenceNumber()> published_sequence_;
};

}  // namespace ROCKSDB_NAMESPACE
