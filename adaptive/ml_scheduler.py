import random
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Deque
from enum import Enum
from collections import deque
from pathlib import Path
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score, classification_report
import joblib


class Algorithm(Enum):
    SJF = "SJF"
    FCFS = "FCFS"
    HRRN = "HRRN"

SIM_EPOCHS = 3
ALL_ALGORITHMS = [Algorithm.SJF, Algorithm.FCFS, Algorithm.HRRN]
ALGORITHM_NAMES = [alg.value for alg in ALL_ALGORITHMS]


def build_combo_maps(num_cores: int):
    """
    Deterministic class mapping using lexicographic itertools.product order.
    This same mapping must be used for:
      - dataset creation
      - training
      - prediction decoding
      - class reporting
    """
    combos = list(itertools.product(ALGORITHM_NAMES, repeat=num_cores))
    combo_to_class = {"|".join(combo): i for i, combo in enumerate(combos)}
    class_to_combo = {i: "|".join(combo) for i, combo in enumerate(combos)}
    return combo_to_class, class_to_combo


@dataclass
class Process:
    name: int
    arrival: int
    burst: int
    start_time: int = -1
    finish_time: int = -1
    running_time: int = 0
    core: int = -1

    def clone_for_sim(self):
        return Process(
            name=self.name,
            arrival=self.arrival,
            burst=self.burst,
            start_time=self.start_time,
            finish_time=self.finish_time,
            running_time=self.running_time,
            core=self.core,
        )

    @property
    def turnaround(self):
        return self.finish_time - self.arrival if self.finish_time != -1 else -1

    @property
    def waiting(self):
        return self.turnaround - self.burst if self.finish_time != -1 else -1

    @property
    def response(self):
        return self.start_time - self.arrival if self.start_time != -1 else -1

    @property
    def remaining_time(self):
        return self.burst - self.running_time

    def waiting_at_time(self, time: int) -> int:
        if time < self.arrival:
            return 0

        if self.finish_time != -1 and self.finish_time <= time:
            return self.waiting

        return max((time - self.arrival) - self.running_time, 0)


@dataclass
class Core:
    name: str
    running: int = -1
    algorithm: Algorithm = Algorithm.SJF

    def clone_for_sim(self):
        return Core(
            name=self.name,
            running=self.running,
            algorithm=self.algorithm
        )

    def schedule(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        if self.algorithm == Algorithm.SJF:
            self.sjf(processes, ready, time, sim)
        elif self.algorithm == Algorithm.FCFS:
            self.fcfs(processes, ready, time, sim)
        elif self.algorithm == Algorithm.HRRN:
            self.hrrn(processes, ready, time, sim)

    def _start_process(self, pid: int, processes: List[Process], time: int):
        self.running = pid
        processes[pid].core = int(self.name[1:]) if self.name.startswith("C") else -1
        if processes[pid].start_time == -1:
            processes[pid].start_time = time

    def _finish_if_done(self, processes: List[Process], time: int, sim: bool = False) -> bool:
        if self.running == -1:
            return False

        p = processes[self.running]
        if p.running_time >= p.burst:
            p.finish_time = time
            if sim:
                print(f"{self.name}: process {self.running} completed at time {time}")
            self.running = -1
            return True
        return False

    def sjf(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        # Non-preemptive shortest process next/shortest job first
        if self.running == -1 and ready:
            pid = min(ready, key=lambda p: processes[p].burst)
            ready.remove(pid)
            self._start_process(pid, processes, time)

        if self.running != -1:
            processes[self.running].running_time += 1
            self._finish_if_done(processes, time + 1, sim)

    def fcfs(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        # Non-preemptive first come first served
        if self.running == -1 and ready:
            pid = ready.popleft()
            self._start_process(pid, processes, time)

        if self.running != -1:
            processes[self.running].running_time += 1
            self._finish_if_done(processes, time + 1, sim)

    def hrrn(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        # Non-preemptive highest response ratio next
        if self.running == -1 and ready:
            pid = max(
                ready,
                key=lambda p: ((time - processes[p].arrival) + processes[p].burst) / processes[p].burst
            )
            ready.remove(pid)
            self._start_process(pid, processes, time)

        if self.running != -1:
            processes[self.running].running_time += 1
            self._finish_if_done(processes, time + 1, sim)


@dataclass
class CPU:
    cores: List[Core] = field(default_factory=list)
    processes: List[Process] = field(default_factory=list)
    queue: Deque[int] = field(default_factory=deque)
    ready: Deque[int] = field(default_factory=deque)
    done: List[int] = field(default_factory=list)
    system_time: int = 0
    epoch: int = 10
    num_cores: int = 4
    training_rows: List[dict] = field(default_factory=list)
    default_algorithm: Algorithm = Algorithm.SJF
    verbose: bool = False
    context_switches: int = 0
    tie_count: int = 0
    tie_rows: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.combo_to_class, self.class_to_combo = build_combo_maps(self.num_cores)

    def init_cores(self):
        if len(self.cores) == self.num_cores:
            return

        self.cores.clear()
        for i in range(self.num_cores):
            self.cores.append(Core(f"C{i}", algorithm=self.default_algorithm))

    def init_queue(self):
        self.queue.clear()
        for i in range(len(self.processes)):
            self.queue.append(i)
        self.queue = deque(sorted(self.queue, key=lambda p: self.processes[p].arrival))

    def update_done(self):
        self.done = [i for i, p in enumerate(self.processes) if p.finish_time != -1]

    def all_finished(self):
        return len(self.done) == len(self.processes)

    def assign_algorithms(self, combo):
        for i, alg in enumerate(combo):
            self.cores[i].algorithm = alg

    def generate_algorithm_combinations(self):
        return list(itertools.product(ALL_ALGORITHMS, repeat=self.num_cores))

    def move_arrivals_to_ready(self):
        while self.queue and self.processes[self.queue[0]].arrival == self.system_time:
            self.ready.append(self.queue.popleft())

    def step(self, sim: bool = False):
        self.move_arrivals_to_ready()

        prev_running = [core.running for core in self.cores]

        for core in self.cores:
            core.schedule(self.processes, self.ready, self.system_time, sim=sim)

        for prev_pid, core in zip(prev_running, self.cores):
            curr_pid = core.running

            # Count any dispatch/change in running process.
            # This includes:
            #   idle -> process   (initial dispatch)
            #   process -> idle   (process completed)
            #   process -> process (preemption / replacement)
            if curr_pid != prev_pid:
                if prev_pid != -1 or curr_pid != -1:
                    self.context_switches += 1

        self.update_done()
        self.system_time += 1

    def clone_for_epoch_simulation(self):
        return CPU(
            cores=[core.clone_for_sim() for core in self.cores],
            processes=[p.clone_for_sim() for p in self.processes],
            queue=deque(self.queue),
            ready=deque(self.ready),
            done=list(self.done),
            system_time=self.system_time,
            epoch=self.epoch,
            num_cores=self.num_cores,
            training_rows=[],
            default_algorithm=self.default_algorithm,
            verbose=False,
            context_switches=self.context_switches,
            tie_count=self.tie_count,
            tie_rows=[]
        )

    def simulate_until(self, end_time: int, sim: bool = False):
        while self.system_time < end_time and not self.all_finished():
            self.step(sim=sim)

    def extract_boundary_features(self):
        active_ids = list(self.ready) + [core.running for core in self.cores if core.running != -1]

        if active_ids:
            burst_vals = np.array([self.processes[pid].burst for pid in active_ids], dtype=float)
            remaining_vals = np.array([self.processes[pid].remaining_time for pid in active_ids], dtype=float)
            wait_vals = np.array([self.processes[pid].waiting_at_time(self.system_time) for pid in active_ids], dtype=float)
        else:
            burst_vals = np.array([], dtype=float)
            remaining_vals = np.array([], dtype=float)
            wait_vals = np.array([], dtype=float)

        unfinished_ids = [i for i, p in enumerate(self.processes) if p.finish_time == -1]
        future_arrivals = np.array(
            [self.processes[i].arrival for i in unfinished_ids if self.processes[i].arrival >= self.system_time],
            dtype=float
        )

        if len(future_arrivals) > 1:
            arrival_window = future_arrivals.max() - future_arrivals.min()
            arrival_rate = len(future_arrivals) / arrival_window if arrival_window > 0 else float(len(future_arrivals))
        elif len(future_arrivals) == 1:
            arrival_rate = 1.0
        else:
            arrival_rate = 0.0

        burst_mean = float(np.mean(burst_vals)) if len(burst_vals) > 0 else 0.0
        remaining_mean = float(np.mean(remaining_vals)) if len(remaining_vals) > 0 else 0.0
        wait_mean = float(np.mean(wait_vals)) if len(wait_vals) > 0 else 0.0

        return {
            "time": self.system_time,
            "ready_count": len(self.ready),
            "running_count": sum(1 for core in self.cores if core.running != -1),
            "unfinished_count": len(unfinished_ids),
            "burst_mean": burst_mean,
            "burst_variance": float(np.var(burst_vals, ddof=1)) if len(burst_vals) > 1 else 0.0,
            "remaining_mean": remaining_mean,
            "remaining_variance": float(np.var(remaining_vals, ddof=1)) if len(remaining_vals) > 1 else 0.0,
            "wait_mean": wait_mean,
            "wait_variance": float(np.var(wait_vals, ddof=1)) if len(wait_vals) > 1 else 0.0,
            "max_wait_current": float(np.max(wait_vals)) if len(wait_vals) > 0 else 0.0,
            "short_job_ratio": float(np.mean(remaining_vals <= remaining_mean)) if len(remaining_vals) > 0 else 0.0,
            "arrival_rate": arrival_rate,
        }

    def collect_epoch_metrics(self, boundary_time: int, epoch_end: int, context_switches_at_boundary: int = 0):
        completed_in_window = [
            p for p in self.processes
            if p.finish_time != -1 and boundary_time < p.finish_time <= epoch_end
        ]

        arrived_unfinished = [
            p for p in self.processes
            if p.arrival <= epoch_end and (p.finish_time == -1 or p.finish_time > epoch_end)
        ]

        turnaround_vals = [p.turnaround for p in completed_in_window if p.turnaround != -1]
        waiting_vals = [p.waiting for p in completed_in_window if p.waiting != -1]
        waiting_now = [p.waiting_at_time(epoch_end) for p in arrived_unfinished]

        avg_turnaround = float(np.mean(turnaround_vals)) if turnaround_vals else float("inf")
        avg_waiting = float(np.mean(waiting_vals)) if waiting_vals else float("inf")
        max_waiting_now = float(np.max(waiting_now)) if waiting_now else 0.0
        waiting_variance_now = float(np.var(waiting_now, ddof=1)) if len(waiting_now) > 1 else 0.0
        context_switch_delta = int(self.context_switches - context_switches_at_boundary)

        return {
            "score_avg_turnaround": avg_turnaround,
            "score_avg_waiting": avg_waiting,
            "score_max_waiting_now": max_waiting_now,
            "score_waiting_variance_now": waiting_variance_now,
            "score_context_switches": context_switch_delta,
            "score_completed_count": len(completed_in_window)
        }

    def score_metrics(self, metrics: dict, norm_stats: dict) -> float:
        def bounded_sigmoid_score(val, mean_v, std_v):
            if np.isinf(val):
                return 1.0
            if std_v == 0:
                return 0.5
            z = (val - mean_v) / std_v
            score = 1.0 / (1.0 + np.exp(-z))
            return float(np.clip(score, 0.0, 1.0))

        turnaround = bounded_sigmoid_score(
            metrics["score_avg_turnaround"],
            norm_stats["turnaround_mean"],
            norm_stats["turnaround_std"]
        )

        waiting = bounded_sigmoid_score(
            metrics["score_avg_waiting"],
            norm_stats["waiting_mean"],
            norm_stats["waiting_std"]
        )

        max_wait = bounded_sigmoid_score(
            metrics["score_max_waiting_now"],
            norm_stats["max_wait_mean"],
            norm_stats["max_wait_std"]
        )

        variance = bounded_sigmoid_score(
            metrics["score_waiting_variance_now"],
            norm_stats["variance_mean"],
            norm_stats["variance_std"]
        )

        context = bounded_sigmoid_score(
            metrics["score_context_switches"],
            norm_stats["context_mean"],
            norm_stats["context_std"]
        )

        # Priority:
        # 1) Turnaround
        # 2) Waiting
        # 3) Fairness (max wait + variance)
        weighted_sum = (
            2.5 * turnaround +
            1.5 * waiting +
            1.0 * max_wait +
            0.5 * variance
        )

        total_weight = 2.5 + 1.5 + 1.0 + 0.5
        final_score = weighted_sum / total_weight
        return float(np.clip(final_score, 0.0, 1.0))

    def epoch_boundary(self):
        boundary_time = self.system_time
        epoch_end = boundary_time + SIM_EPOCHS * self.epoch

        if self.verbose:
            print(f"\nReached boundary at time {boundary_time}")

        features = self.extract_boundary_features()

        best_score = float("inf")
        best_combos = []
        best_metrics_by_combo = {}

        all_metrics = []

        for combo in self.generate_algorithm_combinations():
            sim_cpu = self.clone_for_epoch_simulation()
            sim_cpu.assign_algorithms(combo)
            context_switches_at_boundary = sim_cpu.context_switches
            sim_cpu.simulate_until(epoch_end, sim=False)

            metrics = sim_cpu.collect_epoch_metrics(
                boundary_time,
                epoch_end,
                context_switches_at_boundary=context_switches_at_boundary
            )
            all_metrics.append((combo, metrics))

        if all_metrics:
            def finite_or_zero(values):
                cleaned = [0.0 if np.isinf(v) else float(v) for v in values]
                return np.array(cleaned, dtype=float)

            turnaround_vals = finite_or_zero([m["score_avg_turnaround"] for _, m in all_metrics])
            waiting_vals = finite_or_zero([m["score_avg_waiting"] for _, m in all_metrics])
            max_wait_vals = finite_or_zero([m["score_max_waiting_now"] for _, m in all_metrics])
            variance_vals = finite_or_zero([m["score_waiting_variance_now"] for _, m in all_metrics])
            context_vals = finite_or_zero([m["score_context_switches"] for _, m in all_metrics])

            norm_stats = {
                "turnaround_mean": float(np.mean(turnaround_vals)),
                "turnaround_std": float(np.std(turnaround_vals)),
                "waiting_mean": float(np.mean(waiting_vals)),
                "waiting_std": float(np.std(waiting_vals)),
                "max_wait_mean": float(np.mean(max_wait_vals)),
                "max_wait_std": float(np.std(max_wait_vals)),
                "variance_mean": float(np.mean(variance_vals)),
                "variance_std": float(np.std(variance_vals)),
                "context_mean": float(np.mean(context_vals)),
                "context_std": float(np.std(context_vals)),
            }

            for combo, metrics in all_metrics:
                score = self.score_metrics(metrics, norm_stats)
                combo_str = "|".join(alg.value for alg in combo)

                if score < best_score:
                    best_score = score
                    best_combos = [combo]
                    best_metrics_by_combo = {combo_str: metrics}
                elif np.isclose(score, best_score, rtol=0.0, atol=1e-12):
                    best_combos.append(combo)
                    best_metrics_by_combo[combo_str] = metrics

        if best_combos:
            if len(best_combos) > 1:
                self.tie_count += 1
                tied_combo_strings = ["|".join(alg.value for alg in combo) for combo in best_combos]
                self.tie_rows.append({
                    "boundary_time": boundary_time,
                    "epoch_end": epoch_end,
                    "tie_count_at_boundary": len(best_combos),
                    "tied_combos": " ; ".join(tied_combo_strings),
                    "selected_combo": ""
                })

            selected_index = random.randrange(len(best_combos))
            best_combo = best_combos[selected_index]
            best_combo_str = "|".join(alg.value for alg in best_combo)
            best_metrics = best_metrics_by_combo[best_combo_str]

            if len(best_combos) > 1:
                self.tie_rows[-1]["selected_combo"] = best_combo_str
        else:
            best_combo = tuple(core.algorithm for core in self.cores)
            best_metrics = {
                "score_avg_turnaround": float("inf"),
                "score_avg_waiting": float("inf"),
                "score_max_waiting_now": float("inf"),
                "score_waiting_variance_now": float("inf"),
                "score_context_switches": float("inf"),
                "score_completed_count": 0,
            }
            best_score = 1.0
            best_combo_str = "|".join(alg.value for alg in best_combo)

        combo_class = self.combo_to_class[best_combo_str]

        row = {
            **features,
            "best_combo": best_combo_str,
            "combo_class": combo_class,
        }

        for i, alg in enumerate(best_combo):
            row[f"core{i}_alg"] = alg.value

        row["score"] = float(np.clip(best_score, 0.0, 1.0))
        row.update(best_metrics)

        self.training_rows.append(row)
        self.assign_algorithms(best_combo)

        if self.verbose:
            print(
                "Best combo:",
                [alg.value for alg in best_combo],
                "| Class:",
                combo_class,
                "| Score:",
                round(row["score"], 4)
            )

    def simulate(self):
        self.init_cores()
        self.init_queue()
        self.update_done()

        while not self.all_finished():
            if self.system_time % self.epoch == 0:
                self.epoch_boundary()

            self.step(sim=self.verbose)

    def tie_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.tie_rows)

    def summary_metrics(self) -> dict:
        completed = [p for p in self.processes if p.finish_time != -1]
        completed_count = len(completed)
        total_count = len(self.processes)
        completion_ratio = (completed_count / total_count) if total_count > 0 else 0.0

        if not completed:
            return {
                "completed_processes": completed_count,
                "total_processes": total_count,
                "completion_ratio": completion_ratio,
                "avg_turnaround": 0.0,
                "avg_normalized_turnaround": 0.0,
                "avg_response": 0.0,
                "avg_waiting": 0.0,
                "avg_normalized_waiting": 0.0,
                "total_context_switches": int(self.context_switches),
            }

        turnaround_vals = [p.turnaround for p in completed if p.turnaround != -1]
        response_vals = [p.response for p in completed if p.response != -1]
        waiting_vals = [p.waiting for p in completed if p.waiting != -1]
        normalized_turnaround_vals = [
            (p.turnaround / p.burst) for p in completed
            if p.turnaround != -1 and p.burst > 0
        ]
        normalized_waiting_vals = [
            (p.waiting / p.burst) for p in completed
            if p.waiting != -1 and p.burst > 0
        ]

        return {
            "completed_processes": completed_count,
            "total_processes": total_count,
            "completion_ratio": completion_ratio,
            "avg_turnaround": float(np.mean(turnaround_vals)) if turnaround_vals else 0.0,
            "avg_normalized_turnaround": float(np.mean(normalized_turnaround_vals)) if normalized_turnaround_vals else 0.0,
            "avg_response": float(np.mean(response_vals)) if response_vals else 0.0,
            "avg_waiting": float(np.mean(waiting_vals)) if waiting_vals else 0.0,
            "avg_normalized_waiting": float(np.mean(normalized_waiting_vals)) if normalized_waiting_vals else 0.0,
            "total_context_switches": int(self.context_switches),
        }

    def predict_epoch_boundary(self, model_artifact: dict):
        features = self.extract_boundary_features()
        features_df = pd.DataFrame([features])

        predicted_algs = []
        for i in range(self.num_cores):
            target_col = f"core{i}_alg"
            model = model_artifact["models"].get(target_col)

            if model is None:
                predicted_algs.append(self.default_algorithm)
                continue

            pred = model.predict(features_df)[0]
            if isinstance(pred, Algorithm):
                predicted_algs.append(pred)
            else:
                predicted_algs.append(Algorithm(str(pred)))

        self.assign_algorithms(tuple(predicted_algs))

        if self.verbose:
            print(
                f"Predicted algorithms at time {self.system_time}: "
                f"{[alg.value for alg in predicted_algs]}"
            )

    def simulate_with_model(self, model_artifact: dict):
        self.init_cores()
        self.init_queue()
        self.update_done()

        while not self.all_finished():
            if self.system_time % self.epoch == 0:
                self.predict_epoch_boundary(model_artifact)

            self.step(sim=self.verbose)

    def training_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.training_rows)

    def feature_columns(self) -> List[str]:
        return [
            "time",
            "ready_count",
            "running_count",
            "unfinished_count",
            "burst_mean",
            "burst_variance",
            "remaining_mean",
            "remaining_variance",
            "wait_mean",
            "wait_variance",
            "max_wait_current",
            "short_job_ratio",
            "arrival_rate",
        ]

    def multiclass_target_column(self) -> str:
        return "combo_class"

    def multioutput_target_columns(self) -> List[str]:
        return [f"core{i}_alg" for i in range(self.num_cores)]

    def class_mapping_dataframe(self) -> pd.DataFrame:
        rows = [
            {"combo_class": class_id, "best_combo": combo}
            for class_id, combo in sorted(self.class_to_combo.items())
        ]
        return pd.DataFrame(rows)

    def class_distribution_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = (
            df["combo_class"]
            .value_counts()
            .sort_index()
            .rename("count")
            .reset_index()
            .rename(columns={"index": "combo_class"})
        )

        mapping = self.class_mapping_dataframe()
        merged = mapping.merge(counts, on="combo_class", how="left")
        merged["count"] = merged["count"].fillna(0).astype(int)
        return merged.sort_values("combo_class").reset_index(drop=True)


def load_processes(filename: str, eval: bool = False) -> List[Process]:
    """
    Load processes from ../workloads relative to this script.

    Expected format:
        # comments
        name,arrival,burst
        1,0,79
        2,0,61
        ...

    Internal indexing remains zero-based by list position.
    The 'name' field is preserved from file.
    """
    file_path = Path(__file__).resolve().parent.parent / "workloads" / filename
    eval_path = Path(__file__).resolve().parent.parent / filename
    processes = []

    if eval:
        with eval_path.open("r", encoding="cp1252") as f:
            header_found = False

            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                if not header_found:
                    if line.lower() == "name,arrival,burst":
                        header_found = True
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    raise ValueError(f"Invalid row in {file_path}: {line}")

                name, arrival, burst = map(int, parts)
                processes.append(Process(name=name, arrival=arrival, burst=burst))

        return processes

    with file_path.open("r", encoding="cp1252") as f:
        header_found = False

        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if not header_found:
                if line.lower() == "name,arrival,burst":
                    header_found = True
                continue

            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"Invalid row in {file_path}: {line}")

            name, arrival, burst = map(int, parts)
            processes.append(Process(name=name, arrival=arrival, burst=burst))

    return processes



def save_processes_to_txt(processes: List[Process], filename: str):
    """
    Save processes to ../workloads relative to this script using the format expected by load_processes().
    """
    file_path = Path(__file__).resolve().parent.parent / "workloads" / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        f.write("# Generated workload\n")
        f.write("name,arrival,burst\n")
        for p in processes:
            f.write(f"{p.name},{p.arrival},{p.burst}\n")


def generate_synthetic_workload(
    num_processes: int,
    workload_type: str,
    short_range=(1, 20),
    long_range=(50, 100),
    long_ratio: float = 0.30
) -> List[Process]:
    """
    Generate synthetic workloads with cumulative arrivals.
    Rules:
      - first process always arrives at 0
      - every subsequent process arrival = previous arrival + randint(0, 5)
      - short: bursts in short_range
      - long: bursts in long_range
      - mixed: 70% short and 30% long, randomly interleaved
    """
    if workload_type not in {"short", "long", "mixed"}:
        raise ValueError("workload_type must be one of: short, long, mixed")

    processes: List[Process] = []
    arrival = 0

    for i in range(num_processes):
        if i == 0:
            arrival = 0
        else:
            arrival += random.randint(0, 5)

        if workload_type == "short":
            burst = random.randint(*short_range)
        elif workload_type == "long":
            burst = random.randint(*long_range)
        else:
            is_long = random.random() < long_ratio
            burst = random.randint(*(long_range if is_long else short_range))

        processes.append(Process(name=i + 1, arrival=arrival, burst=burst))

    return processes


def generate_and_save_workloads(num_processes: int = 1000, force_regenerate: bool = False):
    """
    Ensure three workload files exist in ../workloads:
      - short_processes.txt
      - long_processes.txt
      - mixed_processes.txt

    If a file already exists, it is reused.
    If it does not exist, it is generated once and saved.
    Set force_regenerate=True to overwrite existing files.
    """
    data_dir = Path(__file__).resolve().parent.parent / "workloads"
    data_dir.mkdir(parents=True, exist_ok=True)

    workload_specs = [
        (f"short_{num_processes}_processes.txt", "short"),
        (f"long_{num_processes}_processes.txt", "long"),
        (f"mixed_{num_processes}_processes.txt", "mixed"),
    ]

    workload_files = []
    for filename, workload_type in workload_specs:
        file_path = data_dir / filename

        if force_regenerate or not file_path.exists():
            processes = generate_synthetic_workload(
                num_processes=num_processes,
                workload_type=workload_type
            )
            save_processes_to_txt(processes, filename)
        else:
            print(f"Using existing file {file_path}.")

        workload_files.append(filename)

    return workload_files


def run_workload_to_dataframe(
    filename: str,
    epoch: int = 25,
    num_cores: int = 4,
    default_algorithm: Algorithm = Algorithm.HRRN,
    verbose: bool = False
):
    """
    Load a workload file, run the existing simulator, and return the resulting CPU and training dataframe.
    """
    processes = load_processes(filename)

    cpu = CPU(
        processes=processes,
        epoch=epoch,
        num_cores=num_cores,
        default_algorithm=default_algorithm,
        verbose=verbose
    )

    cpu.cores = [Core(f"C{i}", algorithm=default_algorithm) for i in range(cpu.num_cores)]
    cpu.simulate()

    df = cpu.training_dataframe().copy()
    df["workload_file"] = filename

    return cpu, df


def load_dataset_and_train_model(
    dataset_path: str,
    model_output_path: str = "scheduler_per_core_models.joblib",
    n_splits: int = 5,
    min_class_count: int | None = None
):
    """
    Load the saved dataframe and train one model per core algorithm target.

    Uses an imblearn Pipeline so SMOTE is applied only inside training folds
    during cross-validation and only on the training partition during final fit.
    This avoids data leakage.
    """
    df = pd.read_csv(dataset_path)

    if df.empty:
        raise ValueError("Dataset is empty. Cannot train model.")

    def build_preprocessor(X: pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

    feature_cols = [
        "time",
        "ready_count",
        "running_count",
        "unfinished_count",
        "burst_mean",
        "burst_variance",
        "remaining_mean",
        "remaining_variance",
        "wait_mean",
        "wait_variance",
        "max_wait_current",
        "short_job_ratio",
        "arrival_rate",
    ]

    target_cols = [col for col in df.columns if col.startswith("core") and col.endswith("_alg")]
    if not target_cols:
        raise ValueError("No per-core target columns found in dataset.")

    if min_class_count is None:
        min_class_count = max(2, n_splits)

    trained_models = {}
    per_core_results = {}

    for target_col in target_cols:
        target_df = df[feature_cols + [target_col]].dropna().copy()

        original_label_counts = target_df[target_col].value_counts().sort_index()
        excluded_label_counts = original_label_counts[original_label_counts < min_class_count]
        kept_labels = original_label_counts[original_label_counts >= min_class_count].index

        filtered_df = target_df[target_df[target_col].isin(kept_labels)].copy()

        if filtered_df.empty:
            per_core_results[target_col] = {
                "trained": False,
                "reason": f"No rows remain after filtering labels with fewer than {min_class_count} samples.",
                "rows_original": len(target_df),
                "rows_after_filter": 0,
                "labels_original": int(original_label_counts.shape[0]),
                "labels_after_filter": 0,
                "excluded_labels": excluded_label_counts.to_dict(),
            }
            continue

        filtered_label_counts = filtered_df[target_col].value_counts().sort_index()
        smallest_class = int(filtered_label_counts.min())

        if smallest_class < 2:
            per_core_results[target_col] = {
                "trained": False,
                "reason": "Not enough class support after filtering for stratified train/test split.",
                "rows_original": len(target_df),
                "rows_after_filter": len(filtered_df),
                "labels_original": int(original_label_counts.shape[0]),
                "labels_after_filter": int(filtered_label_counts.shape[0]),
                "excluded_labels": excluded_label_counts.to_dict(),
            }
            continue

        X = filtered_df[feature_cols]
        y = filtered_df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        preprocessor = build_preprocessor(X_train)

        safe_splits = min(n_splits, smallest_class)
        if safe_splits < 2:
            per_core_results[target_col] = {
                "trained": False,
                "reason": "Not enough class support for stratified cross-validation.",
                "rows_original": len(target_df),
                "rows_after_filter": len(filtered_df),
                "labels_original": int(original_label_counts.shape[0]),
                "labels_after_filter": int(filtered_label_counts.shape[0]),
                "excluded_labels": excluded_label_counts.to_dict(),
            }
            continue

        smote_k_neighbors = max(1, min(5, smallest_class - 1))

        base_estimator = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
        try:
            model = AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=150,
                learning_rate=0.5,
                random_state=42
            )
        except TypeError:
            model = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=150,
                learning_rate=0.5,
                random_state=42
            )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTETomek(random_state=42)),
            ("adaboost", model)
        ])

        cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)

        cv_results = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=["f1_macro", "accuracy"],
            return_train_score=False,
            error_score="raise"
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        trained_models[target_col] = pipe
        per_core_results[target_col] = {
            "trained": True,
            "model_type": "AdaBoost + SMOTE",
            "smote_k_neighbors": int(smote_k_neighbors),
            "accuracy": float(np.mean(cv_results["test_accuracy"])),
            "f1_score": float(np.mean(cv_results["test_f1_macro"])),
            "classification_report": classification_report(y_test, preds, zero_division=0),
            "rows_original": len(target_df),
            "rows_after_filter": len(filtered_df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "labels_original": int(original_label_counts.shape[0]),
            "labels_after_filter": int(filtered_label_counts.shape[0]),
            "excluded_label_count": int(excluded_label_counts.shape[0]),
            "excluded_labels": excluded_label_counts.to_dict(),
            "min_class_count_used": int(min_class_count),
            "cv_splits_used": int(safe_splits),
        }

    if not trained_models:
        raise ValueError("No per-core models could be trained after filtering rare labels.")

    artifact = {
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "models": trained_models,
        "per_core_results": per_core_results,
    }

    joblib.dump(artifact, model_output_path)

    trained_core_metrics = [
        result for result in per_core_results.values()
        if result.get("trained", False)
    ]

    mean_accuracy = float(np.mean([r["accuracy"] for r in trained_core_metrics])) if trained_core_metrics else 0.0
    mean_f1 = float(np.mean([r["f1_score"] for r in trained_core_metrics])) if trained_core_metrics else 0.0

    return {
        "model": artifact,
        "per_core_results": per_core_results,
        "accuracy": mean_accuracy,
        "f1_score": mean_f1,
        "rows": len(df),
        "model_path": model_output_path,
    }


def evaluate_saved_model_on_workloads(
    model_path: str,
    workload_files: List[str],
    epoch: int = 25,
    num_cores: int = 4,
    default_algorithm: Algorithm = Algorithm.HRRN,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load a saved per-core model bundle and evaluate it on workload files.

    Returns a dataframe with:
      - workload_file
      - completed_processes
      - total_processes
      - completion_ratio
      - avg_turnaround
      - avg_response
      - avg_waiting
      - total_context_switches
    """
    artifact = joblib.load(model_path)

    rows = []
    for workload_file in workload_files:
        processes = load_processes(workload_file, True)

        cpu = CPU(
            processes=processes,
            epoch=epoch,
            num_cores=num_cores,
            default_algorithm=default_algorithm,
            verbose=verbose
        )
        cpu.cores = [Core(f"C{i}", algorithm=default_algorithm) for i in range(cpu.num_cores)]
        cpu.simulate_with_model(artifact)

        metrics = cpu.summary_metrics()
        rows.append({
            "workload_file": workload_file,
            **metrics
        })

    return pd.DataFrame(rows)


def evaluate_static_assignments_on_workloads(
    workload_files: List[str],
    output_path: str = "scheduler_training_static_baselines.csv",
    epoch: int = 25,
    num_cores: int = 4,
    verbose: bool = False,
    force_regenerate: bool = False
) -> pd.DataFrame:
    """
    Evaluate static core assignments on the training workloads and save results.

    Static baselines:
      - all SJF
      - all FCFS
      - all HRRN

    For each workload/baseline pair, report:
      - avg_turnaround
      - avg_normalized_turnaround
      - avg_waiting
      - avg_normalized_waiting
      - total_context_switches

    This file is generated once unless force_regenerate=True.
    """
    output_file = Path(output_path)
    if not force_regenerate and output_file.exists():
        return pd.read_csv(output_file)

    static_baselines = [
        ("all_sjf", Algorithm.SJF),
        ("all_fcfs", Algorithm.FCFS),
        ("all_hrrn", Algorithm.HRRN),
    ]

    rows = []

    for workload_file in workload_files:
        original_processes = load_processes(workload_file)

        for baseline_name, baseline_algorithm in static_baselines:
            cpu = CPU(
                processes=[p.clone_for_sim() for p in original_processes],
                epoch=epoch,
                num_cores=num_cores,
                default_algorithm=baseline_algorithm,
                verbose=verbose
            )
            cpu.cores = [Core(f"C{i}", algorithm=baseline_algorithm) for i in range(cpu.num_cores)]
            cpu.init_queue()
            cpu.update_done()

            # Static assignment: run fixed algorithms for the whole workload.
            while not cpu.all_finished():
                cpu.step(sim=cpu.verbose)

            summary = cpu.summary_metrics().copy()
            summary["workload_file"] = workload_file
            summary["baseline_name"] = baseline_name
            summary["baseline_algorithm"] = baseline_algorithm.value
            summary["final_system_time"] = cpu.system_time

            rows.append(summary)

            print(f"Completed static baseline: {baseline_name} on {workload_file}")
            print(f"  Avg turnaround: {summary['avg_turnaround']:.4f}")
            print(f"  Avg normalized turnaround: {summary['avg_normalized_turnaround']:.4f}")
            print(f"  Avg waiting: {summary['avg_waiting']:.4f}")
            print(f"  Avg normalized waiting: {summary['avg_normalized_waiting']:.4f}")
            print(f"  Context switches: {summary['total_context_switches']}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df

def ensure_simulation_outputs(
    workload_files: List[str],
    dataset_path: str = "scheduler_training_dataset.csv",
    mapping_path: str = "scheduler_class_mapping.csv",
    class_dist_path: str = "scheduler_class_distribution.csv",
    summary_path: str = "scheduler_training_workload_summary.csv",
    tie_log_path: str = "scheduler_training_tie_log.csv",
    epoch: int = 25,
    num_cores: int = 4,
    default_algorithm: Algorithm = Algorithm.SJF,
    verbose: bool = False,
    force_regenerate: bool = False
):
    """
    Ensure the combined simulation dataset and supporting CSVs exist.

    If the dataset files already exist, they are reused.
    If they do not exist, workloads are loaded, simulated, and saved once.
    Set force_regenerate=True to rerun simulations and overwrite outputs.
    """
    dataset_file = Path(dataset_path)
    mapping_file = Path(mapping_path)
    class_dist_file = Path(class_dist_path)
    summary_file = Path(summary_path)
    tie_log_file = Path(tie_log_path)

    if (
        not force_regenerate
        and dataset_file.exists()
        and mapping_file.exists()
        and class_dist_file.exists()
        and summary_file.exists()
        and tie_log_file.exists()
    ):
        return {
            "dataset_path": dataset_path,
            "mapping_path": mapping_path,
            "class_dist_path": class_dist_path,
            "summary_path": summary_path,
            "tie_log_path": tie_log_path,
            "generated": False
        }

    all_results = []
    class_mappings = []
    class_distributions = []
    workload_summaries = []
    tie_logs = []

    for workload_file in workload_files:
        cpu, df = run_workload_to_dataframe(
            filename=workload_file,
            epoch=epoch,
            num_cores=num_cores,
            default_algorithm=default_algorithm,
            verbose=verbose
        )

        all_results.append(df)

        mapping_df = cpu.class_mapping_dataframe().copy()
        mapping_df["workload_file"] = workload_file
        class_mappings.append(mapping_df)

        class_dist = cpu.class_distribution_dataframe(df).copy()
        class_dist["workload_file"] = workload_file
        class_distributions.append(class_dist)

        summary = cpu.summary_metrics().copy()
        summary["workload_file"] = workload_file
        summary["final_system_time"] = cpu.system_time
        summary["training_rows_collected"] = len(df)
        summary["tie_boundaries"] = cpu.tie_count
        workload_summaries.append(summary)

        tie_df = cpu.tie_dataframe().copy()
        if not tie_df.empty:
            tie_df["workload_file"] = workload_file
            tie_logs.append(tie_df)

        print(f"Completed workload: {workload_file}")
        print(f"  Training rows collected: {len(df)}")
        print(f"  Final system time: {cpu.system_time}")
        print(f"  Avg turnaround: {summary['avg_turnaround']:.4f}")
        print(f"  Avg normalized turnaround: {summary['avg_normalized_turnaround']:.4f}")
        print(f"  Avg waiting: {summary['avg_waiting']:.4f}")
        print(f"  Avg normalized waiting: {summary['avg_normalized_waiting']:.4f}")
        print(f"  Context switches: {summary['total_context_switches']}")

    combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined_mapping_df = pd.concat(class_mappings, ignore_index=True) if class_mappings else pd.DataFrame()
    combined_class_dist_df = pd.concat(class_distributions, ignore_index=True) if class_distributions else pd.DataFrame()
    workload_summary_df = pd.DataFrame(workload_summaries)
    tie_log_df = pd.concat(tie_logs, ignore_index=True) if tie_logs else pd.DataFrame(
        columns=["boundary_time", "epoch_end", "tie_count_at_boundary", "tied_combos", "selected_combo", "workload_file"]
    )

    combined_df.to_csv(dataset_path, index=False)
    combined_mapping_df.to_csv(mapping_path, index=False)
    combined_class_dist_df.to_csv(class_dist_path, index=False)
    workload_summary_df.to_csv(summary_path, index=False)
    tie_log_df.to_csv(tie_log_path, index=False)

    return {
        "dataset_path": dataset_path,
        "mapping_path": mapping_path,
        "class_dist_path": class_dist_path,
        "summary_path": summary_path,
        "tie_log_path": tie_log_path,
        "generated": True
    }



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    test = 0

    if test:
        num_processes = 1000
        last_arrival = 0
        processes = []

        for i in range(num_processes):
            if i == 0:
                processes.append(Process(i, last_arrival, random.randint(50, 100)))
            else:
                last_arrival += random.randint(0, 5)
                processes.append(Process(i, last_arrival, random.randint(50, 100)))

        cpu = CPU(
            processes=processes,
            epoch=15,
            num_cores=4,
            default_algorithm=Algorithm.HRRN,
            verbose=False
        )

        cpu.cores = [Core(f"C{i}", algorithm=Algorithm.HRRN) for i in range(cpu.num_cores)]
        cpu.simulate()

        print("\nP | C | A | B | S | F | T | W | R")
        for p in cpu.processes:
            print(
                f"{p.name} | {p.core} | {p.arrival} | {p.burst} | {p.start_time} | "
                f"{p.finish_time} | {p.turnaround} | {p.waiting} | {p.response}"
            )

        df = cpu.training_dataframe()
        print("\nTraining dataset preview:")
        print(df.head().to_string(index=False) if not df.empty else "No training rows collected.")

    else:
        # Generate workloads only if they do not already exist
        workload_files = generate_and_save_workloads(num_processes=1000, force_regenerate=False)

        dataset_path = "scheduler_training_dataset.csv"
        mapping_path = "scheduler_class_mapping.csv"
        class_dist_path = "scheduler_class_distribution.csv"
        summary_path = "scheduler_training_workload_summary.csv"
        tie_log_path = "scheduler_training_tie_log.csv"

        simulation_outputs = ensure_simulation_outputs(
            workload_files=workload_files,
            dataset_path=dataset_path,
            mapping_path=mapping_path,
            class_dist_path=class_dist_path,
            summary_path=summary_path,
            tie_log_path=tie_log_path,
            epoch=25,
            num_cores=4,
            default_algorithm=Algorithm.SJF,
            verbose=False,
            force_regenerate=False
        )

        if simulation_outputs["generated"]:
            print("\nSaved combined dataset to scheduler_training_dataset.csv")
            print("Saved combined class mapping to scheduler_class_mapping.csv")
            print("Saved combined class distribution to scheduler_class_distribution.csv")
            print("Saved workload summary to scheduler_training_workload_summary.csv")
            print("Saved tie log to scheduler_training_tie_log.csv")
        else:
            print("\nUsing existing scheduler_training_dataset.csv")
            print("Using existing scheduler_class_mapping.csv")
            print("Using existing scheduler_class_distribution.csv")
            print("Using existing scheduler_training_workload_summary.csv")
            print("Using existing scheduler_training_tie_log.csv")

        static_baseline_df = evaluate_static_assignments_on_workloads(
            workload_files=workload_files,
            output_path="scheduler_training_static_baselines.csv",
            epoch=25,
            num_cores=4,
            verbose=False,
            force_regenerate=False
        )
        print("\nSaved or loaded static baseline comparison file: scheduler_training_static_baselines.csv")

        training_summary = load_dataset_and_train_model(
            dataset_path=dataset_path,
            model_output_path="scheduler_decision_tree.joblib"
        )

        print("\nML training complete")
        print(f"Dataset rows: {training_summary['rows']}")
        print(f"Mean per-core CV Accuracy: {training_summary['accuracy']:.4f}")
        print(f"Mean per-core CV F1 Macro: {training_summary['f1_score']:.4f}")
        print(f"Saved model bundle to: {training_summary['model_path']}")

        print("\nPer-core results:")
        for target_col, result in training_summary["per_core_results"].items():
            print(f"\n{target_col}")
            if not result.get("trained", False):
                print("  Trained: False")
                print(f"  Reason: {result.get('reason', 'Unknown')}")
                print(f"  Rows original: {result.get('rows_original', 0)}")
                print(f"  Rows after filter: {result.get('rows_after_filter', 0)}")
                print(f"  Labels original: {result.get('labels_original', 0)}")
                print(f"  Labels after filter: {result.get('labels_after_filter', 0)}")
                if result.get("excluded_labels"):
                    print("  Excluded labels:")
                    for label, count in result["excluded_labels"].items():
                        print(f"    {label}: {count} sample(s)")
                else:
                    print("  Excluded labels: none")
                continue

            print("  Trained: True")
            print(f"  Rows original: {result['rows_original']}")
            print(f"  Rows after filter: {result['rows_after_filter']}")
            print(f"  Train rows: {result['train_rows']}")
            print(f"  Test rows: {result['test_rows']}")
            print(f"  Labels original: {result['labels_original']}")
            print(f"  Labels after filter: {result['labels_after_filter']}")
            print(f"  Excluded labels: {result['excluded_label_count']}")
            if result["excluded_labels"]:
                for label, count in result["excluded_labels"].items():
                    print(f"    {label}: {count} sample(s)")
            else:
                print("    none")
            print(f"  CV splits used: {result['cv_splits_used']}")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  F1 Macro: {result['f1_score']:.4f}")
            print("  Classification report:")
            print(result["classification_report"])

        original_eval_files = [
            "data/short_processes.txt",
            "data/long_processes.txt",
            "data/mixed_processes.txt",
        ]

        eval_df = evaluate_saved_model_on_workloads(
            model_path=training_summary["model_path"],
            workload_files=original_eval_files,
            epoch=25,
            num_cores=4,
            default_algorithm=Algorithm.HRRN,
            verbose=False
        )

        eval_output_path = "scheduler_original_100_process_evaluation.csv"
        eval_df.to_csv(eval_output_path, index=False)

        print("\nEvaluation on original 100-process workloads:")
        print(eval_df.to_string(index=False))
        print(f"\nSaved workload evaluation to: {eval_output_path}")
