# ML Assisted Task Scheduling for Multi-Core Systems

## Overview

This project investigates whether scheduling performance in a multicore environment can be improved by dynamically selecting different scheduling algorithms based on workload characteristics. Traditional operating systems typically apply a single scheduling algorithm across all cores. However, different scheduling algorithms perform better under different workload conditions.

The objective of this research is to evaluate whether a lightweight adaptive scheduler can improve performance by selecting the most appropriate scheduling algorithm for each core based on the current workload.

This project builds on the CPU scheduling simulator developed in Assignment 3. The existing implementation will be extended to support multicore scheduling, workload characterization, and adaptive algorithm selection.

The project also explores whether machine learning can assist in selecting scheduling algorithms based on system state.

---

## Research Objective

The goal of this project is to evaluate whether workload-aware scheduling algorithm selection improves performance compared to traditional single-algorithm scheduling.

Specifically, the project examines:

- Whether different scheduling algorithms perform better under different workload conditions
- Whether workload features can be used to select the best scheduling algorithm
- Whether a lightweight machine learning model can predict the optimal scheduler for a given workload

---

## Approach

The project uses a discrete-event scheduling simulator rather than modifying a real operating system. The simulator models process arrival, CPU burst execution, and scheduling decisions across multiple cores.

Each core maintains its own ready queue and scheduling algorithm. Scheduling algorithms may change periodically using an epoch-based reassessment strategy.

The system collects workload features during execution and uses those features to determine which scheduling algorithm performs best for the current workload.

Adaptive algorithm selection will be evaluated using one or both of the following approaches:

1. Rule-based scheduler selection
2. Machine learning-based scheduler selection

Results will be compared against baseline scheduling strategies.

---

## Scheduling Algorithms

The following scheduling algorithms will be implemented:

- Shortest Job First
- Shortest Response Time
- HRRN

Each core can run any of these algorithms independently.

---

## Workload Generation

Synthetic workloads will be generated to simulate different types of system behavior.

Three workload types will be evaluated.

### CPU-Bound Workloads

Processes with long CPU bursts.

Burst times are generated using:

`Uniform(50, 200)`

### I/O-Bound Workloads

Processes with short CPU bursts.

Burst times are generated using:

`Uniform(1, 25)`

### Mixed Workloads

A mixture of short and long bursts.

Example configuration:

- 70% short bursts using `Uniform(1, 25)`
- 30% long bursts using `Uniform(50, 100)`

These workloads allow evaluation of scheduling performance under different conditions.

---

## Epoch-Based Algorithm Selection

The adaptive scheduler periodically reassesses which scheduling algorithm should be used.

At each epoch:

1. Workload features are calculated for each core
2. Candidate scheduling algorithms are evaluated
3. The best-performing algorithm is selected

Epochs are triggered after a fixed number of scheduling events or simulated time units.

---

## Workload Features

At each epoch, the simulator calculates workload characteristics for each core.

Features include:

- Queue length
- Average burst time
- Burst time variance
- Short-job ratio
- Arrival rate
- Maximum waiting time

These features describe the state of the scheduling queue.

---

## Dataset Generation

To train the machine learning model, the simulator generates a dataset of workload states.

For each epoch:

1. Workload features are recorded
2. The simulator evaluates each candidate scheduling algorithm
3. The algorithm with the best performance score is recorded as the label

Example dataset entry:

queue_length,avg_burst,burst_variance,short_ratio,arrival_rate,max_wait,best_algorithm

Example:

10,42,800,0.7,0.2,120,SJF

The dataset will be used to train a classifier that predicts the best scheduling algorithm based on workload features.

---

## Performance Metrics

Scheduling performance will be evaluated using the following metrics:

- Average turnaround time
- Normalized turnaround time
- Average waiting time
- Average response time
- Context switches

These metrics will be compared across scheduling strategies.

---

## Baseline Experiments

Baseline experiments apply the same scheduling algorithm to all cores.

| Experiment | Core 1 | Core 2 | Core 3 | Core 4 |
|-----------|--------|--------|--------|--------|
| Baseline A | SRT | SRT | SRT | SRT |
| Baseline B | SJF | SJF | SJF | SJF |
| Baseline C | HRRN | HRRN | HRRN | HRRN |

These results establish a reference for evaluating adaptive schedulers.

---

## Adaptive Scheduling Experiments

Two adaptive strategies will be evaluated.

### Rule-Based Scheduling

A set of workload rules selects the scheduling algorithm.

Example rule:

```
if avg_burst < threshold:
    use SJF
elif burst_variance > threshold:
    use RR
else:
    use Priority
```

### Machine Learning Scheduling

A trained classifier predicts the best scheduling algorithm based on workload features.

---

## Repository Structure
To be determined

Ideal:
```
scheduler/
│
├── simulator/
│   ├── core_scheduler.py
│   ├── process.py
│   └── metrics.py
│
├── algorithms/
│   ├── round_robin.py
│   ├── sjf.py
│   └── priority.py
│
├── workloads/
│   └── generator.py
│
├── adaptive/
│   ├── feature_extraction.py
│   ├── rule_scheduler.py
│   └── ml_scheduler.py
│
├── experiments/
│   ├── baseline_tests.py
│   └── adaptive_tests.py
│
├── data/
│   └── scheduling_dataset.csv
│
└── paper/
    └── research_paper.tex
```

---

## Project Timeline

### Week 1
Extend the Assignment 3 simulator for multicore support.

Tasks:
- Implement per-core ready queues
- Add scheduling metrics
- Validate simulator behavior

### Week 2
Generate workloads and run baseline experiments.

Tasks:
- Implement workload generator
- Run baseline scheduling experiments
- Record performance metrics

### Week 3
Implement epoch-based scheduling and dataset generation.

Tasks:
- Compute workload features
- Generate dataset samples
- Determine best scheduling algorithm labels

### Week 4
Train ML model and perform final experiments.

Tasks:
- Train a decision tree classifier
- Integrate ML scheduler
- Run comparative experiments
- Analyze results

---

## Expected Contribution

This project evaluates whether selecting scheduling algorithms dynamically in a multicore environment improves scheduling performance compared to traditional fixed-algorithm scheduling approaches.

The results will help determine whether workload-aware scheduling policies provide measurable improvements in turnaround time, waiting time, and response time.
