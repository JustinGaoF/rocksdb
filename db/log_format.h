//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Log format information shared by reader and writer.
// See ../doc/log_format.txt for more detail.

#pragma once

#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {
namespace log {

enum RecordType : uint8_t {
  // Zero is reserved for preallocated files
  kZeroType = 0,
  kFullType = 1,

  // For fragments
  kFirstType = 2,
  kMiddleType = 3,
  kLastType = 4,

  // For recycled log files
  kRecyclableFullType = 5,
  kRecyclableFirstType = 6,
  kRecyclableMiddleType = 7,
  kRecyclableLastType = 8,

  // Compression Type
  kSetCompressionType = 9,

  // Recyclability is not encoded in the type value (in particular, not in the
  // low bit); recyclable and non-recyclable types are not laid out uniformly.
  // Use IsRecyclableRecordType to test for it.
  // User-defined timestamp sizes
  kUserDefinedTimestampSizeType = 10,
  kRecyclableUserDefinedTimestampSizeType = 11,

  // Marks a WAL file as carrying per-record ordering numbers (wal_index / LSN).
  // Kept < 128 and without the safe-ignore bit so that older readers report it
  // as an unknown record type instead of silently skipping it. A report is not
  // a failure: under point-in-time or skip recovery, an older binary stops
  // replaying the file there and opens without its records.
  // The MANIFEST's WAL index guard (VersionEdit::SetWALIndexGuard) is what
  // makes such a binary fail to open the DB.
  //
  // Its value is positional: it is the first record in the file, so an older
  // binary reports on record #1 rather than partway through replay, and the
  // report names the file rather than an arbitrary offset within it.
  //
  // The payload is empty, or a fixed32 WAL partition index. Empty means
  // partition 0, and only partitions above 0 write one, so a single-partition
  // WAL is unchanged. Readers that predate the payload never look at it, so
  // they still read the file and place it in partition 0. Placement of a
  // closed WAL affects only retirement order, never which records replay.
  kWALIndexMarkerType = 12,
  kRecyclableWALIndexMarkerType = 13,

  // Data records carrying a leading wal_index. Each physical fragment is
  // self-describing so the marker is not required for correct decoding.
  kWALIndexFullType = 14,
  kWALIndexFirstType = 15,
  kWALIndexMiddleType = 16,
  kWALIndexLastType = 17,
  kRecyclableWALIndexFullType = 18,
  kRecyclableWALIndexFirstType = 19,
  kRecyclableWALIndexMiddleType = 20,
  kRecyclableWALIndexLastType = 21,

  // Declares that a closed range of wal_index values is not part of the live
  // WAL history and must not be treated as a missing record. A producer may
  // use it for indices that never carried data or for records it deliberately
  // discarded. Consumes no wal_index.
  //
  // A cover is written after the append it stands in for failed, so it lives
  // mid-file rather than in the head, but it is order-constrained: the file's
  // data indices and cover lower bounds must read in nondecreasing order. A
  // cover trailing behind higher-index data would be invisible to a streaming
  // merge at the moment it decides the hole the cover explains, so the reader
  // rejects it. Covers get their own types rather than a marker payload
  // because the reader enforces the marker's "first record" rule, which a
  // cover would violate by construction.
  kWALIndexVoidType = 22,
  kRecyclableWALIndexVoidType = 23,

  // Declares that records at or above a wal_index in one identified older WAL
  // file are obsolete. Records carrying the same indices in other files remain
  // live. Consumes no wal_index.
  kWALIndexSupersessionType = 24,
  kRecyclableWALIndexSupersessionType = 25,

  // Asserts that no record with a wal_index below the bound is still coming
  // from this file. A merge cannot emit index k until every file has shown
  // something above k or finished; an open, idle file does neither and would
  // pin the stream, so it asserts progress instead. Consumes no wal_index.
  //
  // A progress assertion closes the file's head: supersessions and
  // predecessor spans must precede the first progress record, so reading one
  // proves the head metadata is complete. Bounds are nondecreasing; a file
  // cannot un-assert progress.
  kWALIndexProgressType = 26,
  kRecyclableWALIndexProgressType = 27,

  // Attests the wal_index span borne by this file's numeric predecessor: the
  // predecessor's file number, first index, and last index. A merge derives
  // its lower bound from surviving data, so a damaged head silently absorbs
  // real loss; the successor-side span lets the merge detect the gap instead.
  // A span of [0, 0] means the predecessor bears no indices. Consumes no
  // wal_index. Precedes the first data record and the first progress record.
  kWALIndexPredecessorSpanType = 28,
  kRecyclableWALIndexPredecessorSpanType = 29,

  // For WAL verification
  kPredecessorWALInfoType = 130,
  kRecyclePredecessorWALInfoType = 131,
};
// Unknown type of value with the 8-th bit set will be ignored
constexpr uint8_t kRecordTypeSafeIgnoreMask = 1 << 7;
constexpr uint8_t kMaxRecordType = kRecyclePredecessorWALInfoType;

// The WAL index types stay below the safe-ignore bit so that an older binary
// reports an indexed record rather than discarding it without a trace. The
// report fails only the strict recovery modes; renumbering one of them into
// the ignorable range would silence it in those modes too.
// A skipped void record is worse than a skipped marker: the reader would take
// the covered range for data that is simply missing.
static_assert(kWALIndexMarkerType < kRecordTypeSafeIgnoreMask &&
                  kRecyclableWALIndexMarkerType < kRecordTypeSafeIgnoreMask &&
                  kWALIndexFullType < kRecordTypeSafeIgnoreMask &&
                  kWALIndexFirstType < kRecordTypeSafeIgnoreMask &&
                  kWALIndexMiddleType < kRecordTypeSafeIgnoreMask &&
                  kWALIndexLastType < kRecordTypeSafeIgnoreMask &&
                  kRecyclableWALIndexFullType < kRecordTypeSafeIgnoreMask &&
                  kRecyclableWALIndexFirstType < kRecordTypeSafeIgnoreMask &&
                  kRecyclableWALIndexMiddleType < kRecordTypeSafeIgnoreMask &&
                  kRecyclableWALIndexLastType < kRecordTypeSafeIgnoreMask &&
                  kWALIndexVoidType < kRecordTypeSafeIgnoreMask &&
                  kRecyclableWALIndexVoidType < kRecordTypeSafeIgnoreMask,
              "WAL index record types must fail closed on older readers");
static_assert(kRecyclableWALIndexSupersessionType < kRecordTypeSafeIgnoreMask,
              "WAL index supersession types must fail closed on older readers");
static_assert(kRecyclableWALIndexProgressType < kRecordTypeSafeIgnoreMask,
              "WAL index progress types must fail closed on older readers");
static_assert(kRecyclableWALIndexPredecessorSpanType <
                  kRecordTypeSafeIgnoreMask,
              "WAL index predecessor span types must fail closed on older "
              "readers");

inline constexpr bool IsWALIndexRecordType(uint8_t type) {
  return (type >= kWALIndexFullType && type <= kWALIndexLastType) ||
         (type >= kRecyclableWALIndexFullType &&
          type <= kRecyclableWALIndexLastType);
}

// Metadata records that carry wal_index bookkeeping rather than user data.
// They are never returned as logical records and never consume an index.
inline constexpr bool IsWALIndexMetadataRecordType(uint8_t type) {
  return type == kWALIndexMarkerType || type == kRecyclableWALIndexMarkerType ||
         type == kWALIndexVoidType || type == kRecyclableWALIndexVoidType ||
         type == kWALIndexSupersessionType ||
         type == kRecyclableWALIndexSupersessionType ||
         type == kWALIndexProgressType ||
         type == kRecyclableWALIndexProgressType ||
         type == kWALIndexPredecessorSpanType ||
         type == kRecyclableWALIndexPredecessorSpanType;
}

inline constexpr bool IsRecyclableRecordType(uint8_t type) {
  return (type >= kRecyclableFullType && type <= kRecyclableLastType) ||
         (type >= kRecyclableWALIndexFullType &&
          type <= kRecyclableWALIndexLastType) ||
         type == kRecyclableUserDefinedTimestampSizeType ||
         type == kRecyclableWALIndexMarkerType ||
         type == kRecyclableWALIndexVoidType ||
         type == kRecyclableWALIndexSupersessionType ||
         type == kRecyclableWALIndexProgressType ||
         type == kRecyclableWALIndexPredecessorSpanType ||
         type == kRecyclePredecessorWALInfoType;
}

constexpr unsigned int kBlockSize = 32768;

// Number of bytes of the fixed64 wal_index prefixed to each logical record when
// WAL index is enabled.
constexpr uint32_t kWALIndexSize = 8;

// A void record's payload is the covered range as fixed64 lo followed by
// fixed64 hi, both inclusive.
constexpr uint32_t kWALIndexVoidPayloadSize = 2 * kWALIndexSize;

// A supersession record's payload is the superseded WAL file number followed
// by its first obsolete wal_index. The index range is open-ended.
constexpr uint32_t kWALIndexSupersessionPayloadSize = 2 * kWALIndexSize;

// A progress record's payload is one fixed64: the bound below which nothing
// is still coming from this file.
constexpr uint32_t kWALIndexProgressPayloadSize = kWALIndexSize;

// Size of the marker's WAL partition field, when present.
constexpr uint32_t kWALIndexMarkerPartitionSize = 4;

// A predecessor span record's payload is the predecessor WAL file number,
// its first wal_index, and its last wal_index.
constexpr uint32_t kWALIndexPredecessorSpanPayloadSize = 3 * kWALIndexSize;

// Header is checksum (4 bytes), length (2 bytes), type (1 byte)
constexpr int kHeaderSize = 4 + 2 + 1;

// Recyclable header is checksum (4 bytes), length (2 bytes), type (1 byte),
// log number (4 bytes).
constexpr int kRecyclableHeaderSize = 4 + 2 + 1 + 4;

}  // namespace log
}  // namespace ROCKSDB_NAMESPACE
