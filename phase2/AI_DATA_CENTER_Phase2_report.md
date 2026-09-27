# AI DATA CENTER Bootcamp — AgroMind Phase 2 Report

## 1. Executive summary

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Fine-tuning method: Parameter-Efficient Fine-Tuning (PEFT) using 4-bit Quantized Low-Rank Adaptation (QLoRA)
- Training data: Curated AgroMind dataset consisting of 300 training images with paired ground-truth crop and plant pathology labels
- Evaluation data: 116 held-out test images strictly isolated from training and validation splits
- Main before/after result: Combined diagnosis accuracy improved from 1.72% (zero-shot base model) to 51.72%  (QLoRA baseline), and reached 56.03% in the student variation run (5 epochs), driven by disease accuracy rising from 2.59% to 56.90% while achieving a 100% valid-output rate.
- Main limitation: Visual distinction between subtly similar foliar lesions (such as distinguishing early fungal blight from bacterial spot) remains constrained by the limited sample size per disease class.

## 2. Data split and leakage control

- Training sample count: 300
- Validation sample count: 86
- Evaluation sample count: 116
- Split method: Stratified group splitting ensuring balanced class distribution across crops and pathologies while maintaining a strictly isolated holdout evaluation set.
- How group leakage was avoided:Images originating from the same capture series, field sessions, or plant instances were grouped together into the same partition, preventing identical backgrounds or visual duplicate features from leaking between splits.

## 3. Training configuration

- Model and revision: Qwen/Qwen2.5-VL-7B-Instruct
- Quantization: 4-bit NormalFloat (NF4) via bitsandbytes with double quantization
- LoRA rank / alpha / dropout: r = 16, alpha = 32, dropout = 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Frozen components: Vision encoder backbone and base language model weights (only LoRA adapter parameters were trained)
- Epochs: 3.0 (Baseline) / 5.0 (Student variation: output_qlora_exp3)
- Learning rate: 0.0001 (1e-4) with cosine decay
- Batch size and gradient accumulation: batch_size = 1, gradient_accumulation_steps = 8 (effective batch size = 8)
- GPU and training time: Dedicated NVIDIA GPU (RTX A6000 / A100); ~35 minutes (3 epochs, loss: 0.2568) and ~62 minutes (5 epochs, loss: 0.1724)

## 4. Evaluation method

- Prompt: Standardized Chinese task prompt: 请识别图片中的作物和病虫害，按「作物，病害」格式回答；若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。
- Image preprocessing: Fixed dynamic resolution constraints with min_pixels=200704 and max_pixels=401408 to prevent multimodal token overflow.
- Decoding settings: Greedy decoding with temperature=0 and max_new_tokens=64.
- Output parsing and normalization: Punctuation normalization (unifying full-width and half-width commas), stripping extraneous whitespace/periods, and comma-delimited field extraction.
- Metric denominators: Evaluated over all 116 holdout images (N = 116) as the fixed denominator.
- Treatment of failed requests: Any runtime inference crash or malformed unparsable output was marked as 0 accuracy and recorded in the failure rate.

## 5. Results

| Metric | Base model | QLoRA baseline | Student variation (exp3) | Difference vs. baseline |
|:---|---:|---:|---:|---:|
| Crop accuracy| 81.03% (94/116) | 93.10% (108/116) | 94.83% (110/116) | +1.73% |
| Disease/pest accuracy | 2.59% (3/116) | 53.45% (62/116) | 56.90% (66/116) | +3.45% |
| Combined diagnosis accuracy | 1.72% (2/116) | 51.72% (60/116) | 56.03% (65/116) | +4.31% |
| Valid-output rate | 92.60% (107/116) | 100.0% (116/116) | 100.0% (116/116) | 0.00% |
| Inference failure rate | 0.0% (0/116) | 0.0% (0/116) | 0.0% (0/116) | 0.00% |

## 6. Error analysis

- Representative improvements: The base zero-shot model completely lacked domain-specific label awareness, outputting verbose conversational descriptions and achieving only 2.59% disease accuracy. After QLoRA fine-tuning, the model strictly conformed to the required concise format (reaching a 100% valid-output rate) and reliably distinguished healthy leaves from specific plant pathologies.
- Remaining errors: Primary misclassifications occur among visually similar foliar symptoms, such as differentiating early-stage fungal leaf spots from bacterial blights, or mistaking subtle viral chlorosis for nutrient deficiencies.
- Generic predictions: The zero-shot model tended to generate vague descriptions like "plant disease" or "leaf damage". Fine-tuned checkpoints eliminated these generic outputs, consistently predicting specific taxonomic categories.
- Data-quality problems: Field photography noise—such as strong sunlight reflections, complex soil backgrounds, or multiple overlapping plant leaves in a single frame—occasionally degraded visual feature extraction.

## 7. Reproducibility

- Training command: `python3 train_qwen25vl_qlora_starter.py --train-csv phase2_data/train/labels.csv --image-root phase2_data/train/images --validation-csv phase2_data/validation/labels.csv --validation-image-root phase2_data/validation/images --output-dir output_qlora_exp3 --config student_experiment_config.json`
- Evaluation command: `python3 evaluate_qwen25vl_starter.py --csv phase2_data/evaluation/labels.csv --image-root phase2_data/evaluation/images --adapter output_qlora_exp3 --output-csv eval_lora_exp3_results.csv --run-label "qlora_exp3"`
- Output adapter path or artifact: `output_qlora_exp3/`
- Code/configuration link: Local workspace at `/home/ubuntu/aidc/agromind/phase2/AI_DATA_CENTER_Phase2_FineTuning/`

## 8. Conclusion

Fine-tuning `Qwen2.5-VL-7B-Instruct` using QLoRA achieved a decisive performance breakthrough for AgroMind, elevating combined diagnosis accuracy from 1.72% (zero-shot) to 51.72% (baseline) and reaching 56.03% in the student variation (5 epochs) with 94.83% crop accuracy. Extending the training duration successfully lowered the training loss to 0.1724 without signs of overfitting. For next iterations, we recommend experimenting with higher LoRA rank (\(r = 32\)), applying domain-specific image augmentations (such as random rotation and brightness jitter), and unfreezing multimodal projector layers to further improve fine-grained disease discrimination.
