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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, accuracy_score, classification_report
import joblib


class Algorithm(Enum):
    SPN = "SPN"
    SRT = "SRT"
    HRRN = "HRRN"

SIM_EPOCHS = 3
ALL_ALGORITHMS = [Algorithm.SPN, Algorithm.SRT, Algorithm.HRRN]
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
    algorithm: Algorithm = Algorithm.SPN

    def clone_for_sim(self):
        return Core(
            name=self.name,
            running=self.running,
            algorithm=self.algorithm
        )

    def schedule(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        if self.algorithm == Algorithm.SPN:
            self.spn(processes, ready, time, sim)
        elif self.algorithm == Algorithm.SRT:
            self.srt(processes, ready, time, sim)
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

    def spn(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        # Non-preemptive shortest process next
        if self.running == -1 and ready:
            pid = min(ready, key=lambda p: processes[p].burst)
            ready.remove(pid)
            self._start_process(pid, processes, time)

        if self.running != -1:
            processes[self.running].running_time += 1
            self._finish_if_done(processes, time + 1, sim)

    def srt(self, processes: List[Process], ready: Deque[int], time: int, sim: bool = False):
        # Preemptive shortest remaining time
        candidates = list(ready)
        if self.running != -1:
            candidates.append(self.running)

        if not candidates:
            return

        best_pid = min(candidates, key=lambda p: processes[p].remaining_time)

        if self.running != -1 and best_pid != self.running:
            ready.append(self.running)
            self.running = -1

        if self.running == -1:
            if best_pid in ready:
                ready.remove(best_pid)
            self._start_process(best_pid, processes, time)

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
    default_algorithm: Algorithm = Algorithm.HRRN
    verbose: bool = False
    context_switches: int = 0

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
            if prev_pid != -1 and core.running != prev_pid:
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
            context_switches=self.context_switches
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

    def collect_epoch_metrics(self, boundary_time: int, epoch_end: int):
        arrived_unfinished = [
            p for p in self.processes
            if p.arrival <= epoch_end and (p.finish_time == -1 or p.finish_time > epoch_end)
        ]

        completed_in_window = [
            p for p in self.processes
            if p.finish_time != -1 and boundary_time < p.finish_time <= epoch_end
        ]

        started_in_window = [
            p for p in self.processes
            if p.start_time != -1 and boundary_time <= p.start_time < epoch_end
        ]

        waiting_now = [p.waiting_at_time(epoch_end) for p in arrived_unfinished]
        remaining_now = [p.remaining_time for p in arrived_unfinished]
        response_started = [p.response for p in started_in_window if p.response != -1]

        avg_waiting_now = float(np.mean(waiting_now)) if waiting_now else 0.0
        max_waiting_now = float(np.max(waiting_now)) if waiting_now else 0.0
        waiting_variance_now = float(np.var(waiting_now, ddof=1)) if len(waiting_now) > 1 else 0.0
        avg_remaining_now = float(np.mean(remaining_now)) if remaining_now else 0.0
        avg_response_started = float(np.mean(response_started)) if response_started else 0.0

        return {
            "score_avg_waiting_now": avg_waiting_now,
            "score_max_waiting_now": max_waiting_now,
            "score_waiting_variance_now": waiting_variance_now,
            "score_avg_remaining_now": avg_remaining_now,
            "score_avg_response_started": avg_response_started,
            "score_completed_count": len(completed_in_window),
            "score_unfinished_count_end": len(arrived_unfinished)
        }

    def score_metrics(self, metrics: dict) -> float:
        return (
            0.30 * metrics["score_avg_waiting_now"] +
            0.25 * metrics["score_max_waiting_now"] +
            0.15 * metrics["score_waiting_variance_now"] +
            0.15 * metrics["score_avg_remaining_now"] +
            0.10 * metrics["score_avg_response_started"] -
            0.15 * metrics["score_completed_count"]
        )

    def epoch_boundary(self):
        boundary_time = self.system_time
        epoch_end = boundary_time + SIM_EPOCHS * self.epoch

        if self.verbose:
            print(f"\nReached boundary at time {boundary_time}")

        features = self.extract_boundary_features()

        best_score = float("inf")
        best_combo = None
        best_metrics = None

        for combo in self.generate_algorithm_combinations():
            sim_cpu = self.clone_for_epoch_simulation()
            sim_cpu.assign_algorithms(combo)
            sim_cpu.simulate_until(epoch_end, sim=False)

            metrics = sim_cpu.collect_epoch_metrics(boundary_time, epoch_end)
            score = sim_cpu.score_metrics(metrics)

            if score < best_score:
                best_score = score
                best_combo = combo
                best_metrics = metrics

        if best_combo is None:
            best_combo = tuple(core.algorithm for core in self.cores)
            best_metrics = {
                "score_avg_waiting_now": float("inf"),
                "score_max_waiting_now": float("inf"),
                "score_waiting_variance_now": float("inf"),
                "score_avg_remaining_now": float("inf"),
                "score_avg_response_started": float("inf"),
                "score_completed_count": 0,
                "score_unfinished_count_end": len([p for p in self.processes if p.finish_time == -1]),
            }
            best_score = float("inf")

        best_combo_str = "|".join(alg.value for alg in best_combo)
        combo_class = self.combo_to_class[best_combo_str]

        row = {
            **features,
            "best_combo": best_combo_str,
            "combo_class": combo_class,
        }

        for i, alg in enumerate(best_combo):
            row[f"core{i}_alg"] = alg.value

        row["score"] = float(best_score)
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
                round(best_score, 4) if np.isfinite(best_score) else "inf"
            )

    def simulate(self):
        self.init_cores()
        self.init_queue()
        self.update_done()

        while not self.all_finished():
            if self.system_time % self.epoch == 0:
                self.epoch_boundary()

            self.step(sim=self.verbose)

    def summary_metrics(self) -> dict:
        completed = [p for p in self.processes if p.finish_time != -1]

        if not completed:
            return {
                "avg_turnaround": 0.0,
                "avg_response": 0.0,
                "avg_waiting": 0.0,
                "total_context_switches": self.context_switches,
            }

        turnaround_vals = [p.turnaround for p in completed if p.turnaround != -1]
        response_vals = [p.response for p in completed if p.response != -1]
        waiting_vals = [p.waiting for p in completed if p.waiting != -1]

        return {
            "avg_turnaround": float(np.mean(turnaround_vals)) if turnaround_vals else 0.0,
            "avg_response": float(np.mean(response_vals)) if response_vals else 0.0,
            "avg_waiting": float(np.mean(waiting_vals)) if waiting_vals else 0.0,
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


def load_processes(filename: str) -> List[Process]:
    """
    Load processes from ../data relative to this script.

    Expected format:
        # comments
        name,arrival,burst
        1,0,79
        2,0,61
        ...

    Internal indexing remains zero-based by list position.
    The 'name' field is preserved from file.
    """
    file_path = Path(__file__).resolve().parent.parent / "data" / filename
    processes = []

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
    Save processes to ../data relative to this script using the format expected by load_processes().
    """
    file_path = Path(__file__).resolve().parent.parent / "data" / filename
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
    Ensure three workload files exist in ../data:
      - short_1000_processes.txt
      - long_1000_processes.txt
      - mixed_1000_processes.txt

    If a file already exists, it is reused.
    If it does not exist, it is generated once and saved.
    Set force_regenerate=True to overwrite existing files.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    workload_specs = [
        ("short_1000_processes.txt", "short"),
        ("long_1000_processes.txt", "long"),
        ("mixed_1000_processes.txt", "mixed"),
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

    The existing dataset already supports this because it contains:
      - core0_alg
      - core1_alg
      - core2_alg
      - core3_alg

    Rare labels are filtered per-core and reported.
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

        model = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample"
        )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("random_forest", model)
        ])

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

        cv = StratifiedKFold(n_splits=safe_splits, shuffle=True, random_state=42)

        cv_results = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=["f1_macro", "accuracy"],
            return_train_score=False
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        trained_models[target_col] = pipe
        per_core_results[target_col] = {
            "trained": True,
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
      - avg_turnaround
      - avg_response
      - avg_waiting
      - total_context_switches
    """
    artifact = joblib.load(model_path)

    rows = []
    for workload_file in workload_files:
        processes = load_processes(workload_file)

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

def ensure_simulation_outputs(
    workload_files: List[str],
    dataset_path: str = "scheduler_training_dataset.csv",
    mapping_path: str = "scheduler_class_mapping.csv",
    class_dist_path: str = "scheduler_class_distribution.csv",
    epoch: int = 25,
    num_cores: int = 4,
    default_algorithm: Algorithm = Algorithm.HRRN,
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

    if (
        not force_regenerate
        and dataset_file.exists()
        and mapping_file.exists()
        and class_dist_file.exists()
    ):
        return {
            "dataset_path": dataset_path,
            "mapping_path": mapping_path,
            "class_dist_path": class_dist_path,
            "generated": False
        }

    all_results = []
    class_mappings = []
    class_distributions = []

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

        print(f"Completed workload: {workload_file}")
        print(f"  Training rows collected: {len(df)}")
        print(f"  Final system time: {cpu.system_time}")

    combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined_mapping_df = pd.concat(class_mappings, ignore_index=True) if class_mappings else pd.DataFrame()
    combined_class_dist_df = pd.concat(class_distributions, ignore_index=True) if class_distributions else pd.DataFrame()

    combined_df.to_csv(dataset_path, index=False)
    combined_mapping_df.to_csv(mapping_path, index=False)
    combined_class_dist_df.to_csv(class_dist_path, index=False)

    return {
        "dataset_path": dataset_path,
        "mapping_path": mapping_path,
        "class_dist_path": class_dist_path,
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
        workload_files = generate_and_save_workloads(num_processes=10000, force_regenerate=False)

        dataset_path = "scheduler_training_dataset.csv"
        mapping_path = "scheduler_class_mapping.csv"
        class_dist_path = "scheduler_class_distribution.csv"

        simulation_outputs = ensure_simulation_outputs(
            workload_files=workload_files,
            dataset_path=dataset_path,
            mapping_path=mapping_path,
            class_dist_path=class_dist_path,
            epoch=25,
            num_cores=4,
            default_algorithm=Algorithm.HRRN,
            verbose=False,
            force_regenerate=False
        )

        if simulation_outputs["generated"]:
            print("\nSaved combined dataset to scheduler_training_dataset.csv")
            print("Saved combined class mapping to scheduler_class_mapping.csv")
            print("Saved combined class distribution to scheduler_class_distribution.csv")
        else:
            print("\nUsing existing scheduler_training_dataset.csv")
            print("Using existing scheduler_class_mapping.csv")
            print("Using existing scheduler_class_distribution.csv")

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
            "short_processes.txt",
            "long_processes.txt",
            "mixed_processes.txt",
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
