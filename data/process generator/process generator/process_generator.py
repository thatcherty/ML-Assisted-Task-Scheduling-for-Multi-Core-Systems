"""
generate_workloads.py
─────────────────────────────────────────────────────────────────────────────
Generates three process workloads and exports each to a text file.

Arrival times
─────────────
  All workloads use cumulative arrivals starting from 0.
  Process 0 always arrives at time 0; each subsequent process arrives
  some random number of ticks after the previous one (randint(0, 5)),
  matching the style: last_arrival += random.randint(0, 5).
  Arrivals are then shuffled so process names don't align with arrival order.
"""

import random
import os
from dataclasses import dataclass
from typing import List


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_PROCESSES   = 100

BURST_LONG_MIN  = 50
BURST_LONG_MAX  = 100

BURST_SHORT_MIN = 1
BURST_SHORT_MAX = 25

MIXED_SHORT_PCT = 0.70      # 70 % short, 30 % long

MAX_ARRIVAL_STEP = 5        # max ticks between consecutive arrivals

OUTPUT_DIR = "."


# ── Cumulative arrival generator ───────────────────────────────────────────────

def make_arrivals(n: int) -> List[int]:
    """
    Cumulative arrival times: process 0 arrives at 0, each subsequent
    process arrives last_arrival + randint(0, MAX_ARRIVAL_STEP) ticks later.
    Arrivals are then shuffled so process index doesn't correlate with
    arrival order.
    """
    if n == 1:
        return [0]
    arrivals = [0]
    last_arrival = 0
    for _ in range(1, n):
        last_arrival += random.randint(0, MAX_ARRIVAL_STEP)
        arrivals.append(last_arrival)
    return arrivals


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class Process:
    name:    str
    arrival: int
    burst:   int


# ── Workload generators ────────────────────────────────────────────────────────

def make_long(n: int) -> List[Process]:
    arrivals = make_arrivals(n)
    return [
        Process(
            name    = str(i + 1),
            arrival = arrivals[i],
            burst   = random.randint(BURST_LONG_MIN, BURST_LONG_MAX),
        )
        for i in range(n)
    ]


def make_short(n: int) -> List[Process]:
    arrivals = make_arrivals(n)
    return [
        Process(
            name    = str(i + 1),
            arrival = arrivals[i],
            burst   = random.randint(BURST_SHORT_MIN, BURST_SHORT_MAX),
        )
        for i in range(n)
    ]


def make_mixed(n: int, short_pct: float) -> List[Process]:
    n_short  = round(n * short_pct)
    n_long   = n - n_short
    arrivals = make_arrivals(n)

    # Build burst list (short class + long class) then shuffle
    # → guarantees exact 70/30 split with no ordering bias
    bursts = (
        [random.randint(BURST_SHORT_MIN, BURST_SHORT_MAX) for _ in range(n_short)] +
        [random.randint(BURST_LONG_MIN,  BURST_LONG_MAX)  for _ in range(n_long)]
    )
    random.shuffle(bursts)

    return [
        Process(
            name    = str(i + 1),
            arrival = arrivals[i],
            burst   = bursts[i],
        )
        for i in range(n)
    ]


# ── Export ─────────────────────────────────────────────────────────────────────

def export_txt(procs: List[Process], filename: str, label: str) -> str:
    arrivals   = [p.arrival for p in procs]
    max_arrival = max(arrivals)
    filepath   = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write(f"# Workload    : {label}\n")
        f.write(f"# Processes   : {len(procs)}\n")
        f.write(f"# Max arrival : {max_arrival}\n")
        f.write(f"# Arrivals    : cumulative integers, step randint(0, {MAX_ARRIVAL_STEP})\n")
        f.write("#\n")
        f.write("name,arrival,burst\n")
        for p in procs:
            f.write(f"{p.name},{p.arrival},{p.burst}\n")
    print(f"  Exported {len(procs):>3} processes  max_arrival={max_arrival:<6} → {filepath}")
    return filepath


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(procs: List[Process], label: str):
    bursts   = [p.burst   for p in procs]
    arrivals = [p.arrival for p in procs]
    print(f"\n  {label}")
    print(f"  {'─'*56}")
    print(f"  Count       : {len(procs)}")
    print(f"  Arrival     : range [{min(arrivals)}, {max(arrivals)}]  "
          f"avg={sum(arrivals)/len(arrivals):.1f}")
    print(f"  Burst       : min={min(bursts)}  max={max(bursts)}"
          f"  avg={sum(bursts)/len(bursts):.1f}")
    print(f"  {'─'*56}")
    print(f"  {'Name':<8}{'Arrival':<12}{'Burst'}")
    for p in procs:
        print(f"  {'P'+p.name:<8}{p.arrival:<12}{p.burst}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Process Workload Generator")
    print("=" * 60)

    n       = NUM_PROCESSES
    n_short = round(n * MIXED_SHORT_PCT)   # 70
    n_long  = n - n_short                  # 30

    print(f"\n  Arrival style  : cumulative, step = randint(0, {MAX_ARRIVAL_STEP})")
    print(f"  Processes      : {n}")
    print(f"  Mixed split    : {n_short} short / {n_long} long")

    long_procs  = make_long(n)
    short_procs = make_short(n)
    mixed_procs = make_mixed(n, MIXED_SHORT_PCT)

    print_summary(long_procs,  "Long  workload  (burst 50–100)")
    print_summary(short_procs, "Short workload  (burst  1–25)")

    n_short_actual = sum(1 for p in mixed_procs if p.burst <= BURST_SHORT_MAX)
    print_summary(
        mixed_procs,
        f"Mixed workload  ({n_short_actual} short / "
        f"{n - n_short_actual} long, randomly interleaved)",
    )

    print("\n" + "=" * 60)
    print("  Exporting …")
    print("=" * 60)

    f1 = export_txt(long_procs,  "long_processes.txt",  "Long (burst 50–100)")
    f2 = export_txt(short_procs, "short_processes.txt", "Short (burst 1–25)")
    f3 = export_txt(mixed_procs, "mixed_processes.txt", "Mixed (70% short / 30% long)")

    print("\n  Done.  Three files written:")
    for f in (f1, f2, f3):
        print(f"    • {f}")
    print()


if __name__ == "__main__":
    main()