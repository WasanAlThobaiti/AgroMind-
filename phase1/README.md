
# AgroMind Phase 1 — Base Model Deployment and Benchmark

## Overview

This folder contains the Phase 1 deployment and benchmarking work for AgroMind.

The deployed base model is:

`Qwen/Qwen2.5-VL-7B-Instruct`

The model was served using vLLM on Kubernetes and evaluated using the provided 500-image AgroMind dataset.

## Folder structure

```text
phase1/
├── README.md
├── k8s/
│   ├── agromind-deployment.yaml
│   └── agromind-service.yaml
├── evaluation/
│   ├── run_benchmark.py
│   ├── scoring.py
│   ├── benchmark_concurrency.py
│   └── results/
│       ├── benchmark_raw.csv
│       ├── benchmark_scored.csv
│       └── concurrency_results.csv
├── prompts/
│   └── agromind_prompt.txt
└── reports/
    └── AgroMind_Benchmark_Report_Phase1.md
```

## Deployment

Deploy the model and service:

```bash
kubectl apply -f phase1/k8s/agromind-deployment.yaml
kubectl apply -f phase1/k8s/agromind-service.yaml
```

Check the deployment:

```bash
kubectl get pods -n team
kubectl get svc -n team
```

## Dataset

The benchmark uses the two provided dataset archives:

```text
AgroMind_dataset_500_part1.zip
AgroMind_dataset_500_part2.zip
```

Together they contain 500 images.
The dataset is stored on the project server and is not included in this GitHub repository.

Server dataset structure:

```text
~/aidc/agromind/data/
├── part1/
│   ├── images/
│   └── labels.csv
└── part2/
    ├── images/
    └── labels.csv
```

## Run the benchmark

```bash
python3 phase1/evaluation/run_benchmark.py
```

## Score the results

```bash
python3 phase1/evaluation/scoring.py
```

## Run the concurrency benchmark

```bash
python3 phase1/evaluation/benchmark_concurrency.py
```

## Results

Raw and processed benchmark results are stored in:

```text
phase1/evaluation/results/
```

For the complete benchmark results, methodology, metrics, challenges, and conclusion, see:

```text
phase1/reports/AgroMind_Benchmark_Report_Phase1.md
```

## Scope

Phase 1 covers deployment and benchmarking of the base model only.
