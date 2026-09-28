# AgroMind

AgroMind is an agricultural vision AI project developed as part of the AI Data Center Bootcamp.

The project explores how a vision-language model can be deployed, evaluated, and fine-tuned for agricultural image understanding, with a focus on identifying crops and recognizing associated diseases and pests.

## Project Objectives

AgroMind aims to:

- Develop an AI system for crop and disease/pest recognition from agricultural images.
- Deploy and evaluate a vision-language model on GPU infrastructure.
- Improve model performance through fine-tuning and controlled experimentation.

## Project Development

The project is organized into two main phases:

### Phase 1 — Model Serving and Benchmarking

Phase 1 establishes the model-serving and benchmarking pipeline for AgroMind.

Detailed implementation, benchmarking, results, and artifacts are documented in [`phase1/README.md`](phase1/README.md).

### Phase 2 — QLoRA Fine-Tuning

Phase 2 extends AgroMind through parameter-efficient fine-tuning and controlled experimentation to improve crop and disease/pest recognition.

Detailed fine-tuning methodology, experiments, evaluation results, and artifacts are documented in [`phase2/README.md`](phase2/README.md).

## Repository Structure

```text
AgroMind/
├── README.md
│
├── docs/
│   ├── phase1_architecture.md
│   └── phase2_architecture.md
│
├── phase1/
│   ├── README.md
│   ├── evaluation/
│   │   └── results/
│   ├── k8s/
│   ├── prompts/
│   └── reports/
│
└── phase2/
    ├── README.md
    ├── requirements_phase2.txt
    ├── report/
    ├── configs/
    ├── scripts/
    ├── output_qlora_baseline/
    ├── output_qlora_exp3/
    ├── results/
    └── k8s/
```

## Architecture Documentation

System architecture is documented separately for each phase:

- [`docs/phase1_architecture.md`](docs/phase1_architecture.md) — Model serving and benchmarking architecture.
- [`docs/phase2_architecture.md`](docs/phase2_architecture.md) — Fine-tuning and evaluation architecture.

## Technology Stack

- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Language:** Python
- **Deep learning framework:** PyTorch
- **Model library:** Hugging Face Transformers
- **Model serving:** vLLM
- **Fine-tuning:** QLoRA with PEFT and bitsandbytes
- **Container orchestration:** Kubernetes
- **GPU:** NVIDIA RTX A6000