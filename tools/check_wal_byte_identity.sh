#!/usr/bin/env bash
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under both the GPLv2 (found in the
#  COPYING file in the root directory) and Apache 2.0 License
#  (found in the LICENSE.Apache file in the root directory).
#
# Checks that two db_bench builds write byte-identical WAL files for a fixed
# single-threaded workload, in each of these configurations:
#   none       partition_wal_usage=none (wal_partitions=1)
#   indexed    partition_wal_usage=partition_by_column_family, one partition
#   partitions partition_wal_usage=partition_by_column_family, four
#              partitions over eight column families, without pipelined
#              writes, which wal_partitions > 1 rejects (one writer, so the
#              routing and every WAL's bytes are deterministic)
#
# Usage: tools/check_wal_byte_identity.sh DB_BENCH_A DB_BENCH_B
#
# Build both with DEBUG_LEVEL=0 make db_bench, for example the parent commit
# as A and the commit under test as B. Passing the same binary twice checks
# that the workload is deterministic. For each configuration the script
# prints the SHA-256 of every WAL file both builds wrote, then MATCH or
# MISMATCH; it exits non-zero if any configuration mismatches. The databases
# go under $TEST_TMPDIR (or /tmp).

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 DB_BENCH_A DB_BENCH_B" >&2
  exit 2
fi
DB_BENCH_A=$1
DB_BENCH_B=$2
WORK_DIR=$(mktemp -d "${TEST_TMPDIR:-/tmp}/wal_byte_identity.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

COMMON_ARGS="--benchmarks=fillseq --threads=1 --num=10000 --seed=20260928 \
--disable_auto_compactions=1"

declare -A CONFIG_ARGS=(
  [none]="--partition_wal_usage=none"
  [indexed]="--partition_wal_usage=partition_by_column_family"
  [partitions]="--partition_wal_usage=partition_by_column_family \
--wal_partitions=4 --num_column_families=8 --enable_pipelined_write=0"
)

# Runs build $1 in configuration $2 and prints "<sha256>  <WAL file name>"
# for each WAL file, in name order.
wal_hashes() {
  local db="$WORK_DIR/$2_$3"
  # shellcheck disable=SC2086
  "$1" $COMMON_ARGS ${CONFIG_ARGS[$2]} --db="$db" > "$db.out" 2>&1 || {
    echo "db_bench failed; see output below" >&2
    cat "$db.out" >&2
    exit 1
  }
  (cd "$db" && sha256sum -- *.log)
}

status=0
for config in none indexed partitions; do
  hashes_a=$(wal_hashes "$DB_BENCH_A" "$config" a)
  hashes_b=$(wal_hashes "$DB_BENCH_B" "$config" b)
  echo "== $config: ${CONFIG_ARGS[$config]}"
  echo "A:"
  echo "$hashes_a"
  echo "B:"
  echo "$hashes_b"
  if [ "$hashes_a" = "$hashes_b" ]; then
    echo "MATCH $config"
  else
    echo "MISMATCH $config"
    status=1
  fi
done
exit $status
