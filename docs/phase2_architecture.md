# AgroMind Phase 2 Architecture

## System architecture

```text
Phase 2 Dataset
   ├── Training: 300 images
   ├── Validation: 86 images
   └── Evaluation: 116 images
          ↓
Kubernetes Deployment / Pod
          ↓
QLoRA Fine-Tuning
          ↓
Qwen/Qwen2.5-VL-7B-Instruct
          ↓
NVIDIA RTX A6000
          ↓
LoRA Adapter Artifacts
   ├── Baseline: 3 epochs
   └── Variation: 5 epochs
          ↓
Evaluation
          ↓
Base Model / 3-Epoch / 5-Epoch
          ↓
Evaluation Results
```
**Main components:**
- Dataset: 300 training, 86 validation, and 116 evaluation images.
- Kubernetes: Deployment and Service used for the Phase 2 workload.
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Fine-tuning method: 4-bit QLoRA.
- Baseline experiment: 3 training epochs.
- Student variation: 5 training epochs.
- Controlled change: number of training epochs.
- GPU: NVIDIA RTX A6000.
- Adapter outputs: `output_qlora_baseline/` and `output_qlora_exp3/`.
- Evaluation outputs: `results/`.

**Training and evaluation flow:**
1. The Phase 2 workload runs in the Kubernetes environment using the configured Deployment and Service.
2. The training script reads the Phase 2 training dataset.
3. `Qwen/Qwen2.5-VL-7B-Instruct` is loaded using 4-bit quantization and fine-tuned using QLoRA on the NVIDIA RTX A6000.
4. The QLoRA baseline is trained for 3 epochs.
5. The student variation is trained for 5 epochs, with the number of epochs as the single controlled change.
6. The resulting LoRA adapter artifacts are stored in `output_qlora_baseline/` and `output_qlora_exp3/`.
7. The base model and both fine-tuned models are evaluated using the same 116-image held-out evaluation set.
8. Evaluation predictions and summary metrics are stored in `results/`.