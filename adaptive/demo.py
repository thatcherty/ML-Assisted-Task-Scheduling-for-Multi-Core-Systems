import random
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "ml_scheduler.py"

spec = importlib.util.spec_from_file_location("ml_scheduler_demo_source", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {MODULE_PATH}")

sched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sched)


class DemoCPU(sched.CPU):
    def epoch_boundary(self):
        boundary_time = self.system_time
        super().epoch_boundary()

        if self.training_rows:
            row = self.training_rows[-1]
            best_combo = row.get("best_combo", "unknown")
            combo_class = row.get("combo_class", "unknown")
            score = row.get("score", 0.0)
            print(
                f"Epoch at time {boundary_time:>2}: "
                f"selected {best_combo} | class {combo_class} | score {score:.4f}"
            )


def clone_processes(processes):
    return [p.clone_for_sim() for p in processes]


def run_static_baseline(processes, algorithm, label):
    cpu = sched.CPU(
        processes=clone_processes(processes),
        epoch=5,
        num_cores=4,
        default_algorithm=algorithm,
        verbose=False
    )
    cpu.cores = [sched.Core(f"C{i}", algorithm=algorithm) for i in range(cpu.num_cores)]
    cpu.init_cores()
    cpu.init_queue()
    cpu.update_done()

    while not cpu.all_finished():
        cpu.step(sim=False)

    summary = cpu.summary_metrics()

    print(f"\n{label} baseline:")
    print(f"  Avg turnaround: {summary['avg_turnaround']:.2f}")
    print(f"  Avg waiting: {summary['avg_waiting']:.2f}")
    return summary


def main():
    random.seed(42)
    sched.random.seed(42)
    sched.np.random.seed(42)

    processes = sched.generate_synthetic_workload(
        num_processes=15,
        workload_type="short",
        short_range=(1, 20)
    )

    print("Demo workload: 15 short processes")
    print("Epoch length: 5")
    print("Selected algorithm combination at each epoch:\n")

    adaptive_cpu = DemoCPU(
        processes=clone_processes(processes),
        epoch=5,
        num_cores=4,
        default_algorithm=sched.Algorithm.SJF,
        verbose=False
    )
    adaptive_cpu.cores = [sched.Core(f"C{i}", algorithm=sched.Algorithm.SJF) for i in range(adaptive_cpu.num_cores)]
    adaptive_cpu.simulate()

    adaptive_summary = adaptive_cpu.summary_metrics()

    print("\nAdaptive summary:")
    print(f"  Avg turnaround: {adaptive_summary['avg_turnaround']:.2f}")
    print(f"  Avg waiting: {adaptive_summary['avg_waiting']:.2f}")

    sjf_summary = run_static_baseline(processes, sched.Algorithm.SJF, "SJF")
    hrrn_summary = run_static_baseline(processes, sched.Algorithm.HRRN, "HRRN")
    fcfs_summary = run_static_baseline(processes, sched.Algorithm.FCFS, "FCFS")

    print("\nComparison summary:")
    print("  Method     Avg Turnaround   Avg Waiting")
    print(f"  Adaptive   {adaptive_summary['avg_turnaround']:>13.2f}   {adaptive_summary['avg_waiting']:>11.2f}")
    print(f"  SJF        {sjf_summary['avg_turnaround']:>13.2f}   {sjf_summary['avg_waiting']:>11.2f}")
    print(f"  HRRN       {hrrn_summary['avg_turnaround']:>13.2f}   {hrrn_summary['avg_waiting']:>11.2f}")
    print(f"  FCFS       {fcfs_summary['avg_turnaround']:>13.2f}   {fcfs_summary['avg_waiting']:>11.2f}")

    print("\nGenerated processes:")
    for p in processes:
        print(f"Process {p.name:>2}: arrival={p.arrival:>2}, burst={p.burst:>2}")


if __name__ == "__main__":
    main()
