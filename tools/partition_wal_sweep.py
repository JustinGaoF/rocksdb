#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under both the GPLv2 (found in the
# COPYING file in the root directory) and Apache 2.0 License
# (found in the LICENSE.Apache file in the root directory).

"""Reproducible N sweep for partitioned WALs (GC16, G-53).

Runs a db_bench matrix over WAL partitions x injected append latency x sync
mode x value size, parses throughput, write-latency percentiles, the GC13
histograms and the GC15 latency-FileSystem counters out of each run, and
evaluates the Tasks:975 abandon trigger (D-G53 (A): p99 gate wait over p99
write latency at N=8 on the injected profile).

The script builds nothing itself: it takes a db_bench binary path and an
output directory. Every cell runs at least 3 times; the output reports the
median and the spread (min-max) across repetitions.

Injected and uninjected cells are never mixed: tables are grouped by the
injected flag, and the trigger computation refuses uninjected cells.

The script also compares every N>1 cell against both N=1 baselines (kNone and
indexed) with a helps/neutral/hurts verdict, compares the one-slow-partition
cell against the even one, checks each run for steady state, and with
--compare_to prints a reproducibility verdict against an earlier results.json
by the criterion fixed in REPRO_METRICS / REPRO_MAX_DELTA.

Example:
  tools/partition_wal_sweep.py --db-bench buck-out/.../db_bench \\
      --output-dir /tmp/gc16-sweep
  tools/partition_wal_sweep.py --db-bench ... --output-dir /tmp/gc16-quick \\
      --quick
  tools/partition_wal_sweep.py --db-bench ... --output-dir /tmp/gc16-quick2 \\
      --quick --compare_to /tmp/gc16-quick/results.json
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time

# The latency FileSystem waits with a plain sleep and has no concurrency cap.
# Verified in utilities/latency_injection_fs.cc (std::this_thread::sleep_for
# in LatencyInjectionFileSystem::Delayed) and .h ("There is no concurrency
# cap"). Recorded here because both change what the numbers mean.
LATENCY_FS_WAIT = "sleep(std::this_thread::sleep_for)"
LATENCY_FS_CONCURRENCY_CAP = "none"

# Histograms the sweep reads out of the STATISTICS block, short name first.
HISTOGRAMS = [
    ("writers_per_group", "rocksdb.writers.per.write.group"),
    ("cap_excluded", "rocksdb.write.group.cap.excluded"),
    ("join_wait", "rocksdb.join.batch.group.wait.micros"),
    ("gate_wait", "rocksdb.wal.durability.wait.micros"),
    ("completion", "rocksdb.wal.partition.completion.micros"),
    ("fsynced_partitions", "rocksdb.wal.sync.fsynced.partitions"),
    ("unsynced_partitions", "rocksdb.wal.sync.unsynced.partitions"),
    ("sync_barrier", "rocksdb.wal.sync.barrier.micros"),
    ("ingestion_stall", "rocksdb.wal.partition.ingestion.stall.micros"),
    ("wal_mutex_wait", "rocksdb.wal.write.mutex.wait.micros"),
    ("wal_mutex_all_wait", "rocksdb.wal.write.mutex.all.wait.micros"),
]

SYNC_MODES = ("none", "mix", "all")
# 1-in-20 sync for the realistic mix.
MIX_ONE_IN = 20

FULL_NS = (1, 2, 4, 8, 16)
FULL_LATENCIES = (50, 200, 1000, 5000)
FULL_SYNC_MODES = ("none", "mix", "all")
FULL_VALUE_SIZES = (100, 1024)
FULL_REPS = 3

QUICK_NS = (1, 8)
QUICK_LATENCIES = (500,)
QUICK_SYNC_MODES = ("none", "mix")
QUICK_VALUE_SIZES = (100,)
QUICK_REPS = 3
QUICK_NUM = 2000

# The trigger's designated "injected-latency profile" (D-G53 (A)): mid-range
# injected latency for a networked WAL device, the realistic sync mix, and
# KB-scale values. The trigger table shows every (latency, sync) cell at N=8
# so the choice is auditable.
TRIGGER_LATENCY_US = 1000
TRIGGER_SYNC_MODE = "mix"
TRIGGER_VALUE_SIZE = 1024
TRIGGER_THRESHOLD = 0.10
# Thread counts the trigger table covers at N=8 for every sync mode: the
# --threads cells plus extra N=8 cells at each count below that differs.
TRIGGER_THREAD_COUNTS = (8, 16, 32)

# Comparison verdicts against an N=1 baseline of the same latency profile,
# from the throughput ratio N>1 / N=1.
HELPS_ABOVE = 1.10
HURTS_BELOW = 0.90

# Per-run steady state: the mean interval qps of the second half of the run
# is within this fraction of the first half's (the first interval is warmup
# and skipped). Runs with fewer than STEADY_MIN_INTERVALS intervals after it
# report n/a.
STEADY_MAX_DRIFT = 0.15
STEADY_MIN_INTERVALS = 2

# --compare_to criterion, fixed here before any data: the run reproduces when
# (1) every helps/neutral/hurts verdict both runs have is unchanged, (2) the
# trigger verdict is unchanged, and (3) every REPRO_METRICS median of every
# cell both runs have moved by at most REPRO_MAX_DELTA of the earlier median.
REPRO_METRICS = ("ops_per_sec", "write_p99_us")
REPRO_MAX_DELTA = 0.10


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--db-bench", required=True, help="db_bench binary path")
    p.add_argument("--output-dir", required=True, help="output directory")
    p.add_argument("--quick", action="store_true",
                   help="small subset for CI/reviewers (a few minutes)")
    p.add_argument("--reps", type=int, default=None,
                   help="repetitions per cell (default: 3)")
    p.add_argument("--ns", default=None,
                   help="comma N list (default: 1,2,4,8,16)")
    p.add_argument("--append-latencies-us", default=None,
                   help="comma append-latency list in us (default: full grid)")
    p.add_argument("--sync-modes", default=None,
                   help="comma sync-mode list out of none,mix,all")
    p.add_argument("--value-sizes", default=None, help="comma value-size list")
    p.add_argument("--num", type=int, default=2000,
                   help="db_bench --num per thread; 2000 x 16 threads = 32k "
                   "writes per cell, enough for a stable p99 (default: 2000)")
    p.add_argument("--threads", type=int, default=16,
                   help="db_bench --threads (default: 16)")
    p.add_argument("--num-column-families", type=int, default=64,
                   help="CFs, fixed across N to isolate N (default: 64)")
    p.add_argument("--seed-base", type=int, default=1000,
                   help="first per-rep seed (default: 1000)")
    p.add_argument("--build-mode", default="opt",
                   help="build mode of the binary, recorded (default: opt)")
    p.add_argument("--no-slow-cell", action="store_true",
                   help="skip the one-slow-partition cell at N=8")
    p.add_argument("--keep-dbs", action="store_true",
                   help="keep per-cell DB dirs (default: removed)")
    p.add_argument("--compare_to", default=None,
                   help="results.json of an earlier run: print a "
                   "reproducibility verdict (criterion: REPRO_* constants)")
    return p.parse_args(argv)


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    return r.returncode, r.stdout


def collect_environment(args):
    env = {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "uname": platform.uname()._asdict(),
        "cpu_count": os.cpu_count(),
        "db_bench": os.path.abspath(args.db_bench),
        "build_mode_declared": args.build_mode,
        "latency_fs_wait": LATENCY_FS_WAIT,
        "latency_fs_concurrency_cap": LATENCY_FS_CONCURRENCY_CAP,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    env["mem_total"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    rc, out = run(["df", "-h", args.output_dir])
    if rc == 0:
        env["df_output_dir"] = out.strip()
    rc, out = run([args.db_bench, "--build_info"])
    env["db_bench_build_info"] = out.strip()
    rc, out = run(["sl", "log", "-r", ".", "-T", "{node} {desc|firstline}\n"])
    env["commit"] = out.strip() if rc == 0 else "unknown"
    return env


def build_matrix(args):
    """The full matrix, or the --quick subset, or an explicit selection."""
    if args.quick:
        ns, lats, syncs, sizes, reps = (
            QUICK_NS, QUICK_LATENCIES, QUICK_SYNC_MODES, QUICK_VALUE_SIZES,
            QUICK_REPS)
        num = QUICK_NUM
    else:
        ns = tuple(int(x) for x in (args.ns or "1,2,4,8,16").split(","))
        lats = tuple(int(x) for x in
                     (args.append_latencies_us or "50,200,1000,5000")
                     .split(","))
        syncs = tuple((args.sync_modes or "none,mix,all").split(","))
        for s in syncs:
            if s not in SYNC_MODES:
                raise ValueError("bad sync mode: %s" % s)
        sizes = tuple(int(x) for x in
                      (args.value_sizes or "100,1024").split(","))
        reps = FULL_REPS
        num = args.num
    if args.reps is not None:
        reps = args.reps
    cells = []
    seen = set()

    def add(cell):
        # The trigger and slow cells can repeat a matrix cell; run it once.
        key = config_key(cell) + (cell["rep"],)
        if key not in seen:
            seen.add(key)
            cells.append(cell)

    for n in ns:
        usages = ("none", "partition_by_column_family") if n == 1 else (
            "partition_by_column_family",)
        for usage in usages:
            for lat in lats:
                for sync in syncs:
                    for size in sizes:
                        for rep in range(reps):
                            add({
                                "n": n, "usage": usage, "append_us": lat,
                                "sync": sync, "value_size": size, "rep": rep,
                                "seed": args.seed_base + rep,
                                "num": num, "slow_partition": -1,
                                "injected": True, "threads": args.threads,
                            })
    # The trigger profile, or the --quick one, at N=8 and every trigger
    # thread count, for every sync mode, with both N=1 baselines at each
    # thread count so every N=8 trigger cell has its comparison.
    t_lat, t_size = ((lats[0], sizes[0]) if args.quick else
                     (TRIGGER_LATENCY_US, TRIGGER_VALUE_SIZE))
    if 8 in ns and t_lat in lats and t_size in sizes:
        for threads in TRIGGER_THREAD_COUNTS:
            for sync in syncs:
                for n, usage in ((8, "partition_by_column_family"),
                                 (1, "none"),
                                 (1, "partition_by_column_family")):
                    for rep in range(reps):
                        add({
                            "n": n, "usage": usage, "append_us": t_lat,
                            "sync": sync, "value_size": t_size, "rep": rep,
                            "seed": args.seed_base + rep, "num": num,
                            "slow_partition": -1, "injected": True,
                            "threads": threads,
                        })
    if not args.no_slow_cell:
        # One slow partition at N=8, mirroring the trigger profile (or the
        # --quick one) so the straggler cost reads directly against the
        # even N=8 cell.
        t_sync = syncs[-1] if args.quick else TRIGGER_SYNC_MODE
        for rep in range(reps):
            add({
                "n": 8, "usage": "partition_by_column_family",
                "append_us": t_lat, "sync": t_sync,
                "value_size": t_size, "rep": rep,
                "seed": args.seed_base + rep, "num": num,
                "slow_partition": 7, "injected": True,
                "threads": args.threads,
            })
    return cells


def cell_command(args, cell, cell_dir):
    db_dir = os.path.join(cell_dir, "db")
    cmd = [
        args.db_bench, "--benchmarks=fillrandom",
        "--num=%d" % cell["num"], "--threads=%d" % cell["threads"],
        "--num_column_families=%d" % args.num_column_families,
        "--wal_partitions=%d" % cell["n"],
        "--partition_wal_usage=%s" % cell["usage"],
        "--enable_pipelined_write=false",
        "--value_size=%d" % cell["value_size"],
        "--seed=%d" % cell["seed"],
        "--db=%s" % db_dir,
        "--statistics", "--histogram",
        # kExceptTimeForMutex: keeps the join-wait columns filled now that
        # the join timer records only above kExceptDetailedTimers (GC28 Q6).
        "--stats_level=4",
        "--disable_auto_compactions=1",
        "--report_interval_seconds=1",
        "--report_file=%s" % os.path.join(cell_dir, "report.csv"),
    ]
    if cell["injected"]:
        cmd += [
            "--inject_wal_latency",
            "--wal_append_latency_us=%d" % cell["append_us"],
            "--wal_sync_latency_us=%d" % cell["append_us"],
            "--wal_latency_jitter_us=%d" % (cell["append_us"] // 10),
            "--wal_latency_seed=%d" % cell["seed"],
        ]
    if cell["sync"] == "all":
        cmd.append("--sync")
    elif cell["sync"] == "mix":
        cmd.append("--sync_write_one_in=%d" % MIX_ONE_IN)
    if cell["slow_partition"] >= 0:
        cmd.append("--wal_slow_partition=%d" % cell["slow_partition"])
    return cmd, db_dir


THROUGHPUT_RE = re.compile(
    r"^(\S+)\s+:\s+([\d.]+) micros/op\s+(\d+) ops/sec\s+([\d.]+) seconds\s+"
    r"(\d+) operations", re.MULTILINE)
LATENCY_RE = re.compile(
    r"Microseconds per write:\n"
    r"Count: (\d+) Average: ([\d.]+)\s+StdDev: ([\d.]+)\n"
    r"Min: (\d+)\s+Median: ([\d.]+)\s+Max: (\d+)\n"
    r"Percentiles: P50: ([\d.]+) P75: ([\d.]+) P99: ([\d.]+) "
    r"P99\.9: ([\d.]+) P99\.99: ([\d.]+)")
STAT_RE = re.compile(
    r"^(rocksdb\.\S+) P50 : (\S+) P95 : (\S+) P99 : (\S+) P100 : (\S+) "
    r"COUNT : (\d+) SUM : (\d+)$", re.MULTILINE)
FS_ALL_RE = re.compile(
    r"all WALs: appends=(\d+) append_bytes=(\d+) syncs=(\d+) "
    r"injected_sleep_micros=(\d+) wall_micros=(\d+)")
FS_WAL_RE = re.compile(
    r"WAL (\d+): appends=(\d+) append_bytes=(\d+) syncs=(\d+) "
    r"injected_sleep_micros=(\d+) wall_micros=(\d+)")


def parse_run(output, cell):
    """Parse one db_bench run; raises ValueError naming what is missing."""
    text = output.replace("\r", "\n")
    m = THROUGHPUT_RE.search(text)
    if not m:
        raise ValueError("no throughput line")
    metrics = {
        "benchmark": m.group(1), "micros_per_op": float(m.group(2)),
        "ops_per_sec": int(m.group(3)), "seconds": float(m.group(4)),
        "operations": int(m.group(5)),
    }
    m = LATENCY_RE.search(text)
    if not m:
        raise ValueError("no write-latency percentiles (need --histogram)")
    metrics["write_p50_us"] = float(m.group(7))
    metrics["write_p99_us"] = float(m.group(9))
    metrics["write_p999_us"] = float(m.group(10))
    histograms = {}
    for m in STAT_RE.finditer(text):
        histograms[m.group(1)] = {
            "p50": float(m.group(2)), "p95": float(m.group(3)),
            "p99": float(m.group(4)), "p100": float(m.group(5)),
            "count": int(m.group(6)), "sum": int(m.group(7)),
        }
    for short, full in HISTOGRAMS:
        if full not in histograms:
            raise ValueError("no STATISTICS line for %s" % full)
        for k, v in histograms[full].items():
            metrics["%s_%s" % (short, k)] = v
    if cell["injected"]:
        m = FS_ALL_RE.search(text)
        if not m:
            raise ValueError("no latency-FS counters (need "
                             "--inject_wal_latency)")
        metrics["fs_appends"] = int(m.group(1))
        metrics["fs_append_bytes"] = int(m.group(2))
        metrics["fs_syncs"] = int(m.group(3))
        metrics["fs_injected_sleep_us"] = int(m.group(4))
        metrics["fs_wall_us"] = int(m.group(5))
        wals = {}
        for m in FS_WAL_RE.finditer(text):
            wals[int(m.group(1))] = {
                "appends": int(m.group(2)),
                "append_bytes": int(m.group(3)), "syncs": int(m.group(4)),
                "injected_sleep_us": int(m.group(5)),
                "wall_us": int(m.group(6)),
            }
        if not wals:
            raise ValueError("no per-WAL counters")
        metrics["wals"] = wals
        metrics["fsyncs_per_write"] = (metrics["fs_syncs"] /
                                      metrics["operations"])
    # One cap_excluded sample per group, and a group that hit the cap
    # excludes at least one writer, so SUM / COUNT bounds the fraction of
    # groups that hit it from above.
    groups = metrics["cap_excluded_count"]
    metrics["cap_hit_rate_upper"] = (
        min(1.0, metrics["cap_excluded_sum"] / groups) if groups else 0.0)
    metrics["trigger_ratio"] = (metrics["gate_wait_p99"] /
                                metrics["write_p99_us"]
                                if metrics["write_p99_us"] else 0.0)
    return metrics


def steady_state(report_path):
    """(drift, steady) of one run, from db_bench's per-second report.

    drift is (second-half mean qps - first-half) / first-half after
    skipping the first interval; steady is None when the run is too short.
    """
    try:
        with open(report_path) as f:
            qps = [float(r["interval_qps"]) for r in csv.DictReader(f)][1:]
    except (OSError, KeyError, ValueError):
        return None, None
    if len(qps) < STEADY_MIN_INTERVALS:
        return None, None
    half = len(qps) // 2
    first = statistics.mean(qps[:half])
    if first <= 0:
        return None, None
    drift = (statistics.mean(qps[half:]) - first) / first
    return drift, abs(drift) <= STEADY_MAX_DRIFT


def check_coverage(cell, metrics):
    """Every partition's WALs must carry writes, or the cell is invalid.

    Coupled open/rotation creates each generation of N WALs in partition
    order with ascending numbers, so chunking the ascending WAL numbers into
    consecutive runs of N recovers (generation, partition). A partition is
    covered when one of its WALs holds more than a header's worth of bytes.
    """
    if cell["n"] <= 1 or not cell["injected"]:
        return []
    numbers = sorted(metrics["wals"])
    n = cell["n"]
    problems = []
    if len(numbers) < n or len(numbers) % n != 0:
        return ["%d WALs is not a multiple of N=%d" % (len(numbers), n)]
    covered = [False] * n
    for g in range(len(numbers) // n):
        gen = numbers[g * n:(g + 1) * n]
        if any(b - a != 1 for a, b in zip(gen, gen[1:])):
            return ["WAL numbers %s are not %d consecutive generations"
                    % (gen, n)]
        for p in range(n):
            if metrics["wals"][gen[p]]["append_bytes"] > 100:
                covered[p] = True
    missing = [p for p in range(n) if not covered[p]]
    if missing:
        problems.append("partitions %s got no writes" % missing)
    return problems


AGG_KEYS = ["ops_per_sec", "micros_per_op", "write_p50_us", "write_p99_us",
            "write_p999_us", "fsyncs_per_write", "cap_hit_rate_upper",
            "trigger_ratio"] + [
    "%s_%s" % (short, stat)
    for short, _ in HISTOGRAMS
    for stat in ("p50", "p99", "count", "sum")]


def aggregate(reps):
    """Median and spread (min-max) of each metric across repetitions."""
    out = {}
    for key in AGG_KEYS:
        vals = [r["metrics"][key] for r in reps if key in r["metrics"]]
        if vals:
            out[key] = {"median": statistics.median(vals),
                        "min": min(vals), "max": max(vals)}
    out["steady"] = {"steady": sum(1 for r in reps if r.get("steady")),
                     "unsteady": sum(1 for r in reps
                                     if r.get("steady") is False),
                     "na": sum(1 for r in reps if r.get("steady") is None)}
    return out


CONFIG_FIELDS = ("n", "usage", "append_us", "sync", "value_size",
                 "slow_partition", "injected", "threads")


def config_key(cell):
    return tuple(cell[k] for k in CONFIG_FIELDS)


def cell_label(key):
    n, usage, lat, sync, size, slow, injected, threads = key
    label = "N=%d %s %dus %s %dB t%d" % (
        n, "knone" if usage == "none" else "idx", lat, sync, size, threads)
    if slow >= 0:
        label += " slow%d" % slow
    if not injected:
        label += " uninjected"
    return label


def tput_verdict(ratio):
    if ratio > HELPS_ABOVE:
        return "helps"
    if ratio < HURTS_BELOW:
        return "hurts"
    return "neutral"


def compare_baselines(aggregates):
    """Every N>1 cell against both N=1 cells (kNone and indexed) of the same
    latency profile and thread count."""
    rows = []
    for key, agg in sorted(aggregates.items()):
        n, _usage, lat, sync, size, slow, injected, threads = key
        if n == 1 or slow >= 0:
            continue
        for usage in ("none", "partition_by_column_family"):
            base_key = (1, usage, lat, sync, size, -1, injected, threads)
            base = aggregates.get(base_key)
            if base is None:
                continue
            tput = (agg["ops_per_sec"]["median"] /
                    base["ops_per_sec"]["median"])
            rows.append({"cell": cell_label(key),
                         "baseline": cell_label(base_key),
                         "throughput_ratio": tput,
                         "write_p99_ratio": (agg["write_p99_us"]["median"] /
                                             base["write_p99_us"]["median"]),
                         "verdict": tput_verdict(tput)})
    return rows


def slow_vs_even(aggregates):
    """Each slow-partition cell against the even cell of the same config."""
    rows = []
    for key, agg in sorted(aggregates.items()):
        if key[5] < 0:
            continue
        even_key = key[:5] + (-1,) + key[6:]
        even = aggregates.get(even_key)
        if even is None:
            continue
        rows.append({"cell": cell_label(key), "even": cell_label(even_key),
                     "throughput_ratio": (agg["ops_per_sec"]["median"] /
                                          even["ops_per_sec"]["median"]),
                     "write_p99_ratio": (agg["write_p99_us"]["median"] /
                                         even["write_p99_us"]["median"]),
                     "gate_p99_even": even["gate_wait_p99"]["median"],
                     "gate_p99_slow": agg["gate_wait_p99"]["median"]})
    return rows


def compute_trigger(aggregates, threads):
    """D-G53 (A): p99 gate wait over p99 write latency at N=8, injected.

    Refuses uninjected cells: comparing injected against uninjected measures
    the injector, not the watermark.
    """
    rows = []
    for key, agg in sorted(aggregates.items()):
        n, _usage, lat, sync, size, slow, injected, cell_threads = key
        if n != 8 or slow >= 0:
            continue
        if not injected:
            raise ValueError("trigger refuses uninjected cells")
        gate = agg["gate_wait_p99"]["median"]
        write = agg["write_p99_us"]["median"]
        rows.append({"append_us": lat, "sync": sync, "value_size": size,
                     "threads": cell_threads, "cell": cell_label(key),
                     "gate_p99_us": gate, "write_p99_us": write,
                     "ratio": gate / write if write else None})
    verdict = None
    for r in rows:
        if (r["append_us"] == TRIGGER_LATENCY_US and
                r["sync"] == TRIGGER_SYNC_MODE and
                r["value_size"] == TRIGGER_VALUE_SIZE and
                r["threads"] == threads):
            verdict = {"cell": r, "threshold": TRIGGER_THRESHOLD,
                       "fires": r["ratio"] is not None and
                       r["ratio"] > TRIGGER_THRESHOLD}
    return rows, verdict


def fmt(v, width=0):
    if v is None:
        s = "-"
    elif isinstance(v, float):
        if v >= 1000:
            s = "%d" % v
        elif v >= 10:
            s = "%.1f" % v
        elif v >= 1:
            s = "%.2f" % v
        else:
            s = "%.3f" % v
    else:
        s = str(v)
    return s.rjust(width) if width else s


def render_table(aggregates, keys, title):
    """Human-readable table: one row per config, median (spread in JSON)."""
    header = ("N", "usage", "append_us", "sync", "size", "slow", "threads",
              "steady", "ops/s", "us/op", "wr_p50", "wr_p99", "wr_p99.9",
              "gate_p50", "gate_p99", "join_p50", "join_p99",
              "fsync/wr", "cap_med", "cap_hit<=", "fanout_med", "barr_p99",
              "mtx_p99", "mtxall_p99")
    lines = [title, " ".join(h.rjust(8) for h in header)]
    for key in sorted(keys):
        n, usage, lat, sync, size, slow, _inj, threads = key
        agg = aggregates[key]
        use = "knone" if usage == "none" else "idx"
        st = agg["steady"]
        row = [fmt(n, 8), use.rjust(8), fmt(lat, 8), sync.rjust(8),
               fmt(size, 8), fmt(slow, 8), fmt(threads, 8),
               ("%d/%d/%d" % (st["steady"], st["unsteady"], st["na"]))
               .rjust(8)]
        for k in ("ops_per_sec", "micros_per_op", "write_p50_us",
                  "write_p99_us", "write_p999_us", "gate_wait_p50",
                  "gate_wait_p99", "join_wait_p50", "join_wait_p99",
                  "fsyncs_per_write", "cap_excluded_p50",
                  "cap_hit_rate_upper", "fsynced_partitions_p50",
                  "sync_barrier_p99", "wal_mutex_wait_p99",
                  "wal_mutex_all_wait_p99"):
            v = agg.get(k)
            row.append(fmt(v["median"], 8) if v else "-".rjust(8))
        lines.append(" ".join(row))
    return "\n".join(lines)


def reproducibility(results, previous_path):
    """--compare_to: this run against an earlier results.json, by the
    criterion fixed at REPRO_METRICS / REPRO_MAX_DELTA above."""
    with open(previous_path) as f:
        previous = json.load(f)
    lines = ["## Reproducibility against %s" % previous_path,
             "criterion: helps/neutral/hurts and trigger verdicts unchanged, "
             "and every %s median within %d%% of the earlier median"
             % ("/".join(REPRO_METRICS), REPRO_MAX_DELTA * 100),
             "cell metric this earlier delta ok"]
    earlier = {tuple(a["config"][k] for k in CONFIG_FIELDS): a["agg"]
               for a in previous.get("aggregates", [])}
    failed = []
    for a in results["aggregates"]:
        key = tuple(a["config"][k] for k in CONFIG_FIELDS)
        if key not in earlier:
            continue
        for metric in REPRO_METRICS:
            now = a["agg"][metric]["median"]
            old = earlier[key][metric]["median"]
            delta = (now - old) / old if old else float("inf")
            ok = abs(delta) <= REPRO_MAX_DELTA
            if not ok:
                failed.append("%s %s" % (cell_label(key), metric))
            lines.append("%s %s %s %s %+.1f%% %s" % (
                cell_label(key), metric, fmt(now), fmt(old), delta * 100,
                "yes" if ok else "NO"))
    old_verdicts = {(r["cell"], r["baseline"]): r["verdict"]
                    for r in previous.get("comparisons", [])}
    changed = ["%s vs %s: %s -> %s" % (r["cell"], r["baseline"],
                                       old_verdicts[(r["cell"],
                                                     r["baseline"])],
                                       r["verdict"])
               for r in results["comparisons"]
               if old_verdicts.get((r["cell"], r["baseline"]),
                                   r["verdict"]) != r["verdict"]]
    now_v, old_v = results["trigger_verdict"], previous.get("trigger_verdict")
    trigger_same = ((now_v and now_v["fires"]) ==
                    (old_v and old_v["fires"]))
    ok = not failed and not changed and trigger_same
    lines += ["medians outside %d%%: %s" % (REPRO_MAX_DELTA * 100,
                                            ", ".join(failed) or "none"),
              "changed helps/neutral/hurts verdicts: %s"
              % ("; ".join(changed) or "none"),
              "trigger verdict unchanged: %s" % ("yes" if trigger_same
                                                 else "NO"),
              "REPRODUCIBLE: %s" % ("yes" if ok else "no")]
    return lines, ok


def write_outputs(args, results):
    """results.json + results.md from the runs so far (incremental)."""
    by_config = {}
    for entry in results["runs"]:
        if "error" not in entry:
            by_config.setdefault(config_key(entry["config"]), []).append(entry)
    aggregates = {k: aggregate(v) for k, v in by_config.items()}
    results["aggregates"] = [
        {"config": dict(zip(CONFIG_FIELDS, k)), "agg": a}
        for k, a in sorted(aggregates.items())]

    injected_keys = [k for k in aggregates if k[6]]
    uninjected_keys = [k for k in aggregates if not k[6]]
    md = ["# Partition-WAL N sweep", "",
          "commit: %s" % results["environment"]["commit"],
          "db_bench: %s (%s)" % (results["environment"]["db_bench"],
                                 results["environment"]["build_mode_declared"]),
          "date: %s" % results["environment"]["date_utc"],
          "host: %s (%s, %s CPUs)" % (
              results["environment"]["host"],
              results["environment"].get("cpu_model", "?"),
              results["environment"]["cpu_count"]),
          "latency FS wait: %s; concurrency cap: %s" % (
              results["environment"]["latency_fs_wait"],
              results["environment"]["latency_fs_concurrency_cap"]),
          ""]
    md.append(render_table(aggregates, injected_keys,
                           "## Injected cells (median across reps)"))
    if uninjected_keys:
        md += ["",
               render_table(aggregates, uninjected_keys,
                            "## Uninjected cells (separate table: never "
                            "compared against injected)")]
    results["comparisons"] = compare_baselines(aggregates)
    md += ["", "## N>1 against both N=1 baselines (throughput ratio: helps "
           "> %.2f, hurts < %.2f)" % (HELPS_ABOVE, HURTS_BELOW),
           "cell | baseline | ops/s ratio | write p99 ratio | verdict"]
    for r in results["comparisons"]:
        md.append("%s | %s | %.3f | %.3f | %s"
                  % (r["cell"], r["baseline"], r["throughput_ratio"],
                     r["write_p99_ratio"], r["verdict"]))
    results["slow_vs_even"] = slow_vs_even(aggregates)
    md += ["", "## One slow partition (10x latency) against the even cell",
           "cell | ops/s x even | write p99 x even | gate p99 even | "
           "gate p99 slow"]
    for r in results["slow_vs_even"]:
        md.append("%s | %.3f | %.3f | %s | %s"
                  % (r["cell"], r["throughput_ratio"], r["write_p99_ratio"],
                     fmt(r["gate_p99_even"]), fmt(r["gate_p99_slow"])))
    trigger_rows, verdict = compute_trigger(aggregates, args.threads)
    results["trigger_rows"] = trigger_rows
    results["trigger_verdict"] = verdict
    md += ["", "## Abandon trigger (D-G53 (A)) at N=8",
           "append_us sync size threads gate_p99_us write_p99_us ratio"]
    for r in trigger_rows:
        md.append("%d %s %d %d %s %s %.4f"
                  % (r["append_us"], r["sync"], r["value_size"], r["threads"],
                     fmt(r["gate_p99_us"]), fmt(r["write_p99_us"]),
                     r["ratio"] if r["ratio"] is not None else -1))
    worst = max((r for r in trigger_rows if r["ratio"] is not None),
                key=lambda r: r["ratio"], default=None)
    worst_text = ("; worst injected N=8 cell %s ratio %.4f"
                  % (worst["cell"], worst["ratio"]) if worst else "")
    if verdict is None:
        verdict_line = ("TRIGGER: no (append=%d, sync=%s, size=%d) cell at "
                        "N=8; no verdict%s"
                        % (TRIGGER_LATENCY_US, TRIGGER_SYNC_MODE,
                           TRIGGER_VALUE_SIZE, worst_text))
    else:
        verdict_line = ("TRIGGER: ratio %.4f at %s %s (threshold %.2f)%s"
                        % (verdict["cell"]["ratio"], verdict["cell"]["cell"],
                           "FIRES" if verdict["fires"] else "does not fire",
                           TRIGGER_THRESHOLD, worst_text))
    md += ["", verdict_line, ""]
    if args.compare_to:
        repro_lines, results["reproducible"] = reproducibility(
            results, args.compare_to)
        md += repro_lines + [""]
        verdict_line += "\n" + repro_lines[-1]
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, sort_keys=True, default=str)
    with open(os.path.join(args.output_dir, "results.md"), "w") as f:
        f.write("\n".join(md))
    return verdict_line


def main(argv):
    args = parse_args(argv)
    if not os.path.isfile(args.db_bench) or not os.access(args.db_bench, os.X_OK):
        print("not an executable: %s" % args.db_bench, file=sys.stderr)
        return 2
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        cells = build_matrix(args)
    except ValueError as e:
        print("bad matrix: %s" % e, file=sys.stderr)
        return 2
    print("sweep: %d runs (%d cells) -> %s"
          % (len(cells), len({config_key(c) for c in cells}), args.output_dir))

    results = {"environment": collect_environment(args), "runs": []}
    failures = 0
    for i, cell in enumerate(cells):
        cell_dir = os.path.join(args.output_dir, "cell-%04d" % i)
        os.makedirs(cell_dir, exist_ok=True)
        cmd, db_dir = cell_command(args, cell, cell_dir)
        t0 = time.time()
        rc, output = run(cmd)
        elapsed = time.time() - t0
        with open(os.path.join(cell_dir, "command.txt"), "w") as f:
            f.write(" ".join(cmd) + "\n")
        with open(os.path.join(cell_dir, "output.log"), "w") as f:
            f.write(output)
        if not args.keep_dbs:
            shutil.rmtree(db_dir, ignore_errors=True)
        entry = {"config": {k: cell[k] for k in
                            CONFIG_FIELDS + ("rep", "seed", "num")},
                 "command": cmd, "cell_dir": cell_dir,
                 "exit_code": rc, "elapsed_s": round(elapsed, 1)}
        entry["steady_drift"], entry["steady"] = steady_state(
            os.path.join(cell_dir, "report.csv"))
        if rc != 0:
            entry["error"] = "db_bench exit %d" % rc
            failures += 1
        else:
            try:
                entry["metrics"] = parse_run(output, cell)
                problems = check_coverage(cell, entry["metrics"])
                if problems:
                    entry["error"] = "; ".join(problems)
                    failures += 1
            except ValueError as e:
                entry["error"] = "parse: %s" % e
                failures += 1
        results["runs"].append(entry)
        status = "FAIL %s" % entry["error"] if "error" in entry else "ok"
        print("[%d/%d] %s rep=%d: %s steady=%s (%.1fs)"
              % (i + 1, len(cells), cell_label(config_key(cell)), cell["rep"],
                 status, entry["steady"], elapsed),
              flush=True)
        write_outputs(args, results)
    print(write_outputs(args, results))
    if failures:
        print("%d/%d runs failed" % (failures, len(cells)), file=sys.stderr)
        return 1
    return 0 if results.get("reproducible", True) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
