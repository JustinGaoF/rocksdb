#!/bin/bash
# Q5 (N-7-4) byte-identity vehicle: WAL file bytes for a fixed single-threaded
# workload must match the pinned hashes and be deterministic.
#
# Usage (from the repository root, with a release db_bench built):
#   DEBUG_LEVEL=0 make -j... db_bench
#   TEST_TMPDIR=... tools/pwal_byte_identity.sh
#
# Checks:
#   n1-none  N=1 kNone fillseq bytes equal the base hash (byte identity).
#   n1-pbcf  N=1 partition_by_column_family bytes equal the base hash.
#   n8-det   Two N=8 runs produce identical bytes (deterministic routing).
#   n8-pin   N=8 bytes equal the base hash.
# Prints MATCH or MISMATCH per check with the measured hashes; exits 0 only
# if every check matches. Neuter: change one WAL routing seed and rebuild;
# the N=8 checks must report a mismatch while the N=1 checks still match.
set -u

PIN_N1_NONE="6d1351117a289a2784449a68d5421b1aa385ab0b383273c2c5918befa9181b09"
PIN_N1_PBCF="9a69a3fc66777fadce05e8a999479a5f57364aca27cd6d22aa00e4631adfc796"
PIN_N8_PBCF="7842da3347263c17a7e8029a80f166e321eb7ebc3821d012d35f9ac6f48b2e50"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/db_bench"
if [ ! -x "$BENCH" ]; then
  echo "MISSING: $BENCH not built; build release db_bench first:" 1>&2
  echo "  DEBUG_LEVEL=0 make -j... db_bench" 1>&2
  exit 2
fi

TMPBASE="$(mktemp -d "${TEST_TMPDIR:-/tmp}/pwal-byte-identity.XXXXXX")"
cleanup() { rm -rf "$TMPBASE"; }
trap cleanup EXIT

wal_hash() {
  # SHA-256 over the concatenation of the DB's *.log files, sorted by name.
  cat $(ls "$1"/*.log | sort) | sha256sum | cut -d' ' -f1
}

run_fillseq() {
  # run_fillseq <dir> [extra db_bench flags...]
  local dir="$1"
  shift
  "$BENCH" --benchmarks=fillseq --threads=1 --num=10000 --seed=20260928 \
    --disable_auto_compactions=1 "$@" --db="$dir" >/dev/null 2>&1
}

fail=0
check() {
  # check <name> <actual> <expected>
  if [ "$2" = "$3" ]; then
    echo "MATCH    $1 $2"
  else
    echo "MISMATCH $1 actual=$2 expected=$3"
    fail=1
  fi
}

run_fillseq "$TMPBASE/n1-none" --partition_wal_usage=none
check "n1-none" "$(wal_hash "$TMPBASE/n1-none")" "$PIN_N1_NONE"

run_fillseq "$TMPBASE/n1-pbcf" --partition_wal_usage=partition_by_column_family
check "n1-pbcf" "$(wal_hash "$TMPBASE/n1-pbcf")" "$PIN_N1_PBCF"

# N>1 rejects db_bench's default pipelined write; single-threaded fillseq is
# otherwise identical, so routing is deterministic.
run_fillseq "$TMPBASE/n8a" --partition_wal_usage=partition_by_column_family \
  --wal_partitions=8 --enable_pipelined_write=false
run_fillseq "$TMPBASE/n8b" --partition_wal_usage=partition_by_column_family \
  --wal_partitions=8 --enable_pipelined_write=false
n8a="$(wal_hash "$TMPBASE/n8a")"
n8b="$(wal_hash "$TMPBASE/n8b")"
check "n8-det" "$n8a" "$n8b"
check "n8-pin" "$n8a" "$PIN_N8_PBCF"

exit $fail
