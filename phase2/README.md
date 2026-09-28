# AgroMind — Phase 2: QLoRA Fine-Tuning

## Overview

This repository contains the Phase 2 fine-tuning work for the AgroMind project.

The base model, `Qwen/Qwen2.5-VL-7B-Instruct`, was fine-tuned using 4-bit QLoRA for crop and disease/pest recognition.

Two fine-tuning experiments were conducted:

- QLoRA baseline: 3 training epochs
- Student variation: 5 training epochs

The number of training epochs was the single controlled change between the two experiments.

## Dataset

The Phase 2 dataset consists of:

- Training: 300 images
- Validation: 86 images
- Evaluation: 116 images

The same held-out evaluation set was used to evaluate the base model and both fine-tuned models.

## Results

| Model | Crop Accuracy | Disease/Pest Accuracy | Combined Accuracy |
|---|---:|---:|---:|
| Base model | 81.03% | 2.59% | 1.72% |
| QLoRA baseline — 3 epochs | 93.10% | 53.45% | 51.72% |
| Student variation — 5 epochs | 94.83% | 56.90% | 56.03% |

The 5-epoch variation improved combined diagnosis accuracy by 4.31 percentage points compared with the 3-epoch baseline.

## Repository Structure

phase2/
├── README.md
├── report/
│   └── AI_DATA_CENTER_Phase2_report.md
├── configs/
│   ├── baseline_3epoch_config.json
│   └── variation_5epoch_config.json
├── scripts/
│   ├── train_qwen25vl_qlora_starter.py
│   └── evaluate_qwen25vl_starter.py
├── output_qlora_baseline/
├── output_qlora_exp3/
├── results/
└── k8s/
    ├── deployment.yaml
    └── service.yaml

## Folder Description

- `report/` — Phase 2 technical report.
- `configs/` — Configuration files for the 3-epoch baseline and 5-epoch variation.
- `scripts/` — Training and evaluation scripts.
- `output_qlora_baseline/` — Saved artifacts from the 3-epoch QLoRA baseline.
- `output_qlora_exp3/` — Saved artifacts from the 5-epoch variation.
- `results/` — Evaluation predictions and summary results for the base model and both fine-tuned models.
- `k8s/` — Kubernetes Deployment and Service manifests used for the Phase 2 workload.

## Model Artifacts

The LoRA adapter weight files (`adapter_model.safetensors`, approximately 181 MB each) are stored on the server/shared storage and are not committed to GitHub due to the GitHub file-size limit.

The corresponding adapter metadata, processor/tokenizer files, training configuration artifacts, and `WEIGHTS_NOTE.txt` files are included in the output directories.

## Training Environment

- GPU: NVIDIA RTX A6000
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Fine-tuning method: 4-bit QLoRA
- Baseline: 3 epochs
- Student variation: 5 epochs
- Baseline training time: approximately 35 minutes
- 5-epoch variation training time: approximately 62 minutes

## Report

For the complete methodology, training configuration, evaluation procedure, results, error analysis, and reproducibility information, see the Phase 2 report in the `report/` directory.