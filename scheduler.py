import random
from collections import deque
from dataclasses import dataclass
from typing import List
from enum import Enum

NUM_CORES = 4


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Process:
    name: str
    arrival: int
    burst: int
    remaining: int = 0
    done: bool = False
    finish_time: int = -1
    first_run: int = -1
    assigned_core: int = -1

    def reset(self):
        self.remaining     = self.burst
        self.done          = False
        self.finish_time   = -1
        self.first_run     = -1
        self.assigned_core = -1


@dataclass
class CoreState:
    pid: int = -1
    time_slice_left: int = 0
    feedback_level: int = 0


class Policy(Enum):
    FCFS = "FCFS"
    SPN  = "SPN"
    SRT  = "SRT"
    HRRN = "HRRN"


@dataclass
class CoreConfig:
    policy: Policy
    quantum: int
    preemptive: bool
    feedback: bool
    label: str


# ── Algorithm definitions ──────────────────────────────────────────────────────

ALGORITHMS = {
    "1": CoreConfig(Policy.FCFS, 4, True,  False, "Round Robin"),
    "2": CoreConfig(Policy.SPN,  0, False, False, "SPN"),
    "3": CoreConfig(Policy.SRT,  1, True,  False, "SRT"),
    "4": CoreConfig(Policy.HRRN, 0, False, False, "HRRN"),
    "5": CoreConfig(Policy.FCFS, 4, True,  True,  "Feedback"),
}


# ── Process generation ─────────────────────────────────────────────────────────

def generate_processes() -> List[Process]:
    n = random.randint(990, 1000)
    procs = []
    for i in range(n):
        p = Process(
            name    = str(i + 1),
            arrival = random.randint(0, 19),
            burst   = random.randint(2, 9),
        )
        p.reset()
        procs.append(p)
    return procs


def reset_all(procs: List[Process]):
    for p in procs:
        p.reset()


# ── Scheduling helpers ─────────────────────────────────────────────────────────

def pick_from_ready(ready: List[int], procs: List[Process],
                    pol: Policy, time: int) -> int:
    if not ready:
        return -1
    best = ready[0]
    for idx in ready:
        cand   = procs[idx]
        best_p = procs[best]
        if pol == Policy.FCFS:
            if cand.arrival < best_p.arrival:
                best = idx
        elif pol == Policy.SPN:
            if cand.burst < best_p.burst:
                best = idx
        elif pol == Policy.SRT:
            if cand.remaining < best_p.remaining:
                best = idx
        elif pol == Policy.HRRN:
            rr_cand = ((time - cand.arrival) + cand.remaining) / cand.remaining
            rr_best = ((time - best_p.arrival) + best_p.remaining) / best_p.remaining
            if rr_cand > rr_best:
                best = idx
    return best


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_timeline(timeline: List[List[str]], core_configs: List[CoreConfig]):
    length     = len(timeline[0])
    chunk_size = 40
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        print(f"\nTime  : {''.join(f'{t:>5}' for t in range(start, end))}")
        for c in range(NUM_CORES):
            row = "".join(f"{timeline[c][t]:>5}" for t in range(start, end))
            print(f"Core{c} [{core_configs[c].label:<12}]: {row}")


def print_metrics(procs: List[Process], core_configs: List[CoreConfig]):
    n = len(procs)
    header = (f"{'Proc':<6}{'Core':<6}{'Arrival':<10}{'Burst':<8}"
              f"{'Finish':<10}{'Turnaround':<14}{'Waiting':<12}{'Response':<14}")
    print(f"\n{header}")
    print("-" * 80)

    sum_ta = sum_wt = sum_rt = 0.0

    for p in procs:
        ta = p.finish_time - p.arrival
        wt = ta - p.burst
        rt = p.first_run - p.arrival

        sum_ta += ta
        sum_wt += wt
        sum_rt += rt

        print(f"{'P'+p.name:<6}{'C'+str(p.assigned_core):<6}{p.arrival:<10}{p.burst:<8}"
              f"{p.finish_time:<10}{ta:<14}{wt:<12}{rt:<14}")

    print("-" * 80)
    print(f"{'Average':<22}{sum_ta/n:<14.2f}{sum_wt/n:<12.2f}{sum_rt/n:<14.2f}")

# ── Excel Export ───────────────────────────────────────────────────────────────

EXCEL_FILE = "scheduler_results.xlsx"

def export_to_excel(procs, core_configs, total_ticks):
    if not EXCEL_AVAILABLE:
        print("  [Excel export skipped: run 'pip install openpyxl' to enable]")
        return

    # Colours
    HDR_FILL   = PatternFill("solid", start_color="1E3A5F")
    AVG_FILL   = PatternFill("solid", start_color="2D6A4F")
    ALT_FILL   = PatternFill("solid", start_color="0F1E2E")
    WHITE      = Font(color="FFFFFF", bold=True, name="Arial", size=10)
    NORMAL     = Font(color="DDEEFF", name="Arial", size=10)
    GREEN_F    = Font(color="FFFFFF", bold=True, name="Arial", size=10)
    THIN       = Side(style="thin", color="2A4A6A")
    BORDER     = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    CENTER     = Alignment(horizontal="center", vertical="center")
    LEFT       = Alignment(horizontal="left",   vertical="center")

    # Load or create workbook
    if os.path.exists(EXCEL_FILE):
        wb = load_workbook(EXCEL_FILE)
    else:
        wb = Workbook()
        # Remove default sheet
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    # Build a unique sheet name: "Run 1", "Run 2", ...
    existing = [s for s in wb.sheetnames if s.startswith("Run ")]
    run_num  = len(existing) + 1
    sheet_name = f"Run {run_num}"
    ws = wb.create_sheet(sheet_name)

    # ── Summary block ─────────────────────────────────────────────────────────
    combo = "  |  ".join(f"C{i}:{core_configs[i].label}" for i in range(NUM_CORES))
    ws["A1"] = f"Run {run_num}  –  {combo}"
    ws["A1"].font = Font(color="FFFFFF", bold=True, name="Arial", size=12)
    ws["A1"].fill = PatternFill("solid", start_color="0A1628")
    ws["A1"].alignment = LEFT
    ws.merge_cells("A1:H1")

    ws["A2"] = f"Total ticks: {total_ticks}    |    Processes: {len(procs)}"
    ws["A2"].font = Font(color="88AACC", name="Arial", size=9)
    ws["A2"].fill = PatternFill("solid", start_color="0A1628")
    ws["A2"].alignment = LEFT
    ws.merge_cells("A2:H2")

    # ── Header row ────────────────────────────────────────────────────────────
    headers = ["Process", "Core", "Algorithm", "Arrival", "Burst",
               "Finish", "Turnaround", "Waiting", "Response"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=4, column=col, value=h)
        cell.font      = WHITE
        cell.fill      = HDR_FILL
        cell.alignment = CENTER
        cell.border    = BORDER

    # ── Per-process rows ──────────────────────────────────────────────────────
    for row_idx, p in enumerate(procs, 5):
        ta = p.finish_time - p.arrival
        wt = ta - p.burst
        rt = p.first_run - p.arrival
        fill = PatternFill("solid", start_color="0F1E2E") if row_idx % 2 == 0                else PatternFill("solid", start_color="122030")
        row_data = [
            f"P{p.name}",
            f"Core {p.assigned_core}",
            core_configs[p.assigned_core].label,
            p.arrival,
            p.burst,
            p.finish_time,
            ta,
            wt,
            rt,
        ]
        for col, val in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col, value=val)
            cell.font      = NORMAL
            cell.fill      = fill
            cell.alignment = CENTER
            cell.border    = BORDER

    # ── Averages row ──────────────────────────────────────────────────────────
    avg_row = 5 + len(procs)
    n       = len(procs)
    ta_vals = [p.finish_time - p.arrival           for p in procs]
    wt_vals = [p.finish_time - p.arrival - p.burst for p in procs]
    rt_vals = [p.first_run   - p.arrival            for p in procs]

    avg_data = ["AVERAGE", "", "", "", "",
                "",
                f"=AVERAGE(G5:G{avg_row-1})",
                f"=AVERAGE(H5:H{avg_row-1})",
                f"=AVERAGE(I5:I{avg_row-1})"]
    for col, val in enumerate(avg_data, 1):
        cell = ws.cell(row=avg_row, column=col, value=val)
        cell.font      = GREEN_F
        cell.fill      = AVG_FILL
        cell.alignment = CENTER
        cell.border    = BORDER

    # ── Per-core summary block ────────────────────────────────────────────────
    summary_row = avg_row + 2
    ws.cell(row=summary_row, column=1, value="Per-Core Summary").font =         Font(color="FFFFFF", bold=True, name="Arial", size=10)
    ws.cell(row=summary_row, column=1).fill = HDR_FILL
    ws.merge_cells(f"A{summary_row}:I{summary_row}")

    sub_hdr = ["Core", "Algorithm", "Processes", "Avg Turnaround", "Avg Waiting", "Avg Response"]
    for col, h in enumerate(sub_hdr, 1):
        cell = ws.cell(row=summary_row+1, column=col, value=h)
        cell.font = WHITE; cell.fill = HDR_FILL
        cell.alignment = CENTER; cell.border = BORDER

    for c in range(NUM_CORES):
        core_procs = [p for p in procs if p.assigned_core == c]
        if not core_procs:
            continue
        ta_c = [p.finish_time - p.arrival           for p in core_procs]
        wt_c = [p.finish_time - p.arrival - p.burst for p in core_procs]
        rt_c = [p.first_run   - p.arrival            for p in core_procs]
        row_data = [
            f"Core {c}",
            core_configs[c].label,
            len(core_procs),
            round(sum(ta_c)/len(ta_c), 2),
            round(sum(wt_c)/len(wt_c), 2),
            round(sum(rt_c)/len(rt_c), 2),
        ]
        r = summary_row + 2 + c
        fill = PatternFill("solid", start_color="0F1E2E") if c % 2 == 0                else PatternFill("solid", start_color="122030")
        for col, val in enumerate(row_data, 1):
            cell = ws.cell(row=r, column=col, value=val)
            cell.font = NORMAL; cell.fill = fill
            cell.alignment = CENTER; cell.border = BORDER

    # ── Column widths ─────────────────────────────────────────────────────────
    widths = [10, 10, 16, 10, 8, 10, 14, 12, 12]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    ws.row_dimensions[1].height = 22
    ws.freeze_panes = "A5"

    # ── Summary sheet – update or create ──────────────────────────────────────
    if "Summary" not in wb.sheetnames:
        ws_sum = wb.create_sheet("Summary", 0)
        s_hdr = ["Run", "Combo", "Processes", "Total Ticks",
                 "Avg Turnaround", "Avg Waiting", "Avg Response"]
        for col, h in enumerate(s_hdr, 1):
            cell = ws_sum.cell(row=1, column=col, value=h)
            cell.font = WHITE; cell.fill = HDR_FILL
            cell.alignment = CENTER; cell.border = BORDER
        sum_widths = [6, 55, 12, 12, 16, 14, 14]
        for i, w in enumerate(sum_widths, 1):
            ws_sum.column_dimensions[get_column_letter(i)].width = w
    else:
        ws_sum = wb["Summary"]

    sum_row = ws_sum.max_row + 1
    ta_all = [p.finish_time - p.arrival           for p in procs]
    wt_all = [p.finish_time - p.arrival - p.burst for p in procs]
    rt_all = [p.first_run   - p.arrival            for p in procs]
    fill_s = PatternFill("solid", start_color="0F1E2E") if run_num % 2 == 0              else PatternFill("solid", start_color="122030")
    sum_data = [
        run_num,
        combo,
        len(procs),
        total_ticks,
        round(sum(ta_all)/len(ta_all), 2),
        round(sum(wt_all)/len(wt_all), 2),
        round(sum(rt_all)/len(rt_all), 2),
    ]
    for col, val in enumerate(sum_data, 1):
        cell = ws_sum.cell(row=sum_row, column=col, value=val)
        cell.font = NORMAL; cell.fill = fill_s
        cell.alignment = CENTER; cell.border = BORDER

    wb.save(EXCEL_FILE)
    print(f"\n  [Excel] Results saved to {EXCEL_FILE}  (sheet: {sheet_name})")


# ── Main simulation ────────────────────────────────────────────────────────────

def simulate(original_procs: List[Process], core_configs: List[CoreConfig]):
    procs = [Process(p.name, p.arrival, p.burst) for p in original_procs]
    reset_all(procs)

    n    = len(procs)
    done = 0

    timeline: List[List[str]] = [[] for _ in range(NUM_CORES)]

    # Each core has its own ready queue and feedback queues.
    # When a process finishes or its slice expires it goes back to its
    # assigned core's queue — so it is never stranded.
    ready_queues = [[] for _ in range(NUM_CORES)]
    fb_queues    = [[deque(), deque(), deque()] for _ in range(NUM_CORES)]
    cores        = [CoreState() for _ in range(NUM_CORES)]
    arrived      = [False] * n
    admit_ctr    = 0

    # Upper bound: all burst work divided across cores, plus headroom
    total_burst = sum(p.burst for p in procs)
    max_time = (total_burst // NUM_CORES) + max(p.arrival for p in procs) + 50
    time     = 0

    while done < n and time < max_time:

        # ── Admit newly arrived processes (round-robin across cores) ──────────
        for i in range(n):
            if not arrived[i] and procs[i].arrival <= time and not procs[i].done:
                arrived[i] = True
                c_target   = admit_ctr % NUM_CORES
                admit_ctr += 1
                procs[i].assigned_core = c_target
                if core_configs[c_target].feedback:
                    fb_queues[c_target][0].append(i)
                else:
                    ready_queues[c_target].append(i)

        # ── Assign idle cores from their own queues (steal if empty) ──────────
        for c in range(NUM_CORES):
            if cores[c].pid != -1:
                continue
            cfg    = core_configs[c]
            chosen = -1

            if cfg.feedback:
                for lv in range(3):
                    if fb_queues[c][lv]:
                        chosen = fb_queues[c][lv].popleft()
                        cores[c].feedback_level = lv
                        break
                # steal from another core's fb queue if ours is empty
                if chosen == -1:
                    for other in range(NUM_CORES):
                        if other == c:
                            continue
                        for lv in range(3):
                            if fb_queues[other][lv]:
                                chosen = fb_queues[other][lv].popleft()
                                procs[chosen].assigned_core = c
                                cores[c].feedback_level = lv
                                break
                        if chosen != -1:
                            break
            else:
                if ready_queues[c]:
                    chosen = pick_from_ready(ready_queues[c], procs, cfg.policy, time)
                    ready_queues[c].remove(chosen)
                else:
                    # steal from another core's ready queue
                    for other in range(NUM_CORES):
                        if other == c or not ready_queues[other]:
                            continue
                        chosen = pick_from_ready(ready_queues[other], procs, cfg.policy, time)
                        ready_queues[other].remove(chosen)
                        procs[chosen].assigned_core = c
                        break

            if chosen != -1:
                cores[c].pid             = chosen
                cores[c].time_slice_left = cfg.quantum if cfg.quantum > 0 else procs[chosen].remaining
                if procs[chosen].first_run == -1:
                    procs[chosen].first_run = time

        # ── SRT preemption (per core) ──────────────────────────────────────────
        for c in range(NUM_CORES):
            cfg = core_configs[c]
            if not cfg.preemptive or cfg.policy != Policy.SRT:
                continue
            pid = cores[c].pid
            if pid == -1 or not ready_queues[c]:
                continue
            best_ready = pick_from_ready(ready_queues[c], procs, Policy.SRT, time)
            if procs[best_ready].remaining < procs[pid].remaining:
                ready_queues[c].remove(best_ready)
                ready_queues[c].append(pid)
                cores[c].pid             = best_ready
                cores[c].time_slice_left = procs[best_ready].remaining
                if procs[best_ready].first_run == -1:
                    procs[best_ready].first_run = time

        # ── Tick each core ─────────────────────────────────────────────────────
        for c in range(NUM_CORES):
            cfg = core_configs[c]
            pid = cores[c].pid
            if pid == -1 or procs[pid].done:
                timeline[c].append("-")
                cores[c].pid = -1
                continue

            timeline[c].append(procs[pid].name)
            procs[pid].remaining     -= 1
            cores[c].time_slice_left -= 1

            if procs[pid].remaining == 0:
                procs[pid].done        = True
                procs[pid].finish_time = time + 1
                cores[c].pid           = -1
                done                  += 1
            elif cores[c].time_slice_left == 0:
                cores[c].pid = -1
                ac = procs[pid].assigned_core  # always return to assigned core
                if cfg.feedback:
                    next_level = min(cores[c].feedback_level + 1, 2)
                    fb_queues[ac][next_level].append(pid)
                else:
                    ready_queues[ac].append(pid)

        time += 1

    max_len = max(len(row) for row in timeline)
    for row in timeline:
        while len(row) < max_len:
            row.append("-")

    label_parts = "  |  ".join(f"Core{c}: {core_configs[c].label}" for c in range(NUM_CORES))
    print(f"\n{'='*70}")
    print(f"  {label_parts}")
    print(f"{'='*70}", end="")
    print_timeline(timeline, core_configs)
    print(f"\n  Total ticks: {max_len}")
    print_metrics(procs, core_configs)


# ── Per-core configuration prompt ─────────────────────────────────────────────

def prompt_core_configs() -> List[CoreConfig]:
    print("\nConfigure each core's scheduling algorithm:")
    print("  1. Round Robin   2. SPN   3. SRT   4. HRRN   5. Feedback")
    configs = []
    for c in range(NUM_CORES):
        while True:
            try:
                choice = input(f"  Core {c} algorithm [1-5]: ").strip()
                if choice not in ALGORITHMS:
                    raise ValueError
                base = ALGORITHMS[choice]
                q    = base.quantum
                if base.label in ("Round Robin", "Feedback"):
                    while True:
                        try:
                            q = int(input(f"  Core {c} time quantum (e.g. 4): ").strip())
                            if q < 1:
                                raise ValueError
                            break
                        except ValueError:
                            print("  Please enter a positive integer.")
                configs.append(CoreConfig(base.policy, q, base.preemptive, base.feedback, base.label))
                break
            except ValueError:
                print("  Please enter a number between 1 and 5.")
    return configs


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Generating random processes...")
    procs = generate_processes()

    print(f"Generated {len(procs)} processes:\n")
    print(f"{'Name':<6}{'Arrival':<10}Burst")
    print("-" * 25)
    for p in procs:
        print(f"{p.name:<6}{p.arrival:<10}{p.burst}")

    while True:
        print("\n" + "="*40)
        print("  1. Run simulation")
        print("  2. Regenerate processes")
        print("  0. Exit")
        try:
            choice = input("Choice: ").strip()
        except (ValueError, EOFError):
            break

        if choice == "0":
            break

        elif choice == "1":
            print("\nMode:")
            print("  1. Same algorithm for all cores")
            print("  2. Different algorithm per core")
            mode = input("Mode [1/2]: ").strip()

            if mode == "1":
                print("\nAlgorithm for all cores:")
                print("  1. Round Robin   2. SPN   3. SRT   4. HRRN   5. Feedback")
                while True:
                    try:
                        algo_choice = input("Algorithm [1-5]: ").strip()
                        if algo_choice not in ALGORITHMS:
                            raise ValueError
                        base = ALGORITHMS[algo_choice]
                        q    = base.quantum
                        if base.label in ("Round Robin", "Feedback"):
                            q = int(input("Time Quantum: "))
                        core_configs = [
                            CoreConfig(base.policy, q, base.preemptive, base.feedback, base.label)
                            for _ in range(NUM_CORES)
                        ]
                        break
                    except ValueError:
                        print("  Please enter a number between 1 and 5.")
            else:
                core_configs = prompt_core_configs()

            simulate(procs, core_configs)

        elif choice == "2":
            procs = generate_processes()
            print(f"Regenerated {len(procs)} processes.")

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
