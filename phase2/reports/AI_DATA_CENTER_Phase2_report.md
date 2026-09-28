# AI DATA CENTER Bootcamp — AgroMind Phase 2 Report

## 1. Executive summary

- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Fine-tuning method: Parameter-Efficient Fine-Tuning (PEFT) using 4-bit Quantized Low-Rank Adaptation (QLoRA)
- Training data: 300 AgroMind agricultural images with ground-truth crop and disease/pest labels.
- Evaluation data: 116 held-out images kept separate from the training and validation sets.
- Main before/after result: 
  - Combined diagnosis accuracy improved from 1.72% for the base model to 51.72% after the 3-epoch QLoRA baseline. 
  - Increasing the training duration from 3 to 5 epochs as the single controlled variation further improved combined diagnosis accuracy to 56.03%.
  - The 5-epoch model achieved 94.83% crop accuracy, 56.90% disease/pest accuracy, and a 100% valid-output rate.
- Main limitation: Although fine-tuning substantially improved performance, the best combined diagnosis accuracy was 56.03%, indicating that a considerable number of evaluation images were still not diagnosed correctly.

## 2. Data split and leakage control

- Training sample count: 300
- Validation sample count: 86
- Evaluation sample count: 116
- Split method: 
  - The provided Phase 2 dataset contains separate training, validation, and evaluation splits.
  - The 300 training images and 86 validation images were split from the previous clean training pool using a fixed group-split seed of 2275.
  - The 116 evaluation images form a separate clean holdout set.
- How group leakage was avoided: 
  - Related images from the same source post or image group were kept within the same split to reduce group leakage.
  - The evaluation set was kept separate from the training and validation data and was not used for gradient updates or experiment selection.

## 3. Training configuration

- Model and revision: Qwen/Qwen2.5-VL-7B-Instruct
- Quantization: 4-bit NormalFloat (NF4) via bitsandbytes with double quantization
- LoRA rank / alpha / dropout: r = 16, alpha = 32, dropout = 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Frozen components: Base-model weights, including the vision tower, were kept frozen; only the LoRA adapter parameters were trained.
- Epochs: 
  - 3.0 for the QLoRA baseline.
  - 5.0 for the student variation.
  - *Note: The number of epochs was the single controlled change between the two experiments.*
- Learning rate: 0.0001 (1e-4).
- Batch size and gradient accumulation: 
  - batch_size = 1
  - gradient_accumulation_steps = 8 (effective batch size = 8)
- GPU and training time: 
  - GPU: NVIDIA RTX A6000
  - Training Time:
    - approximately 35 minutes for the 3-epoch baseline (final training loss: 0.2568)
    - approximately 62 minutes for the 5-epoch variation (final training loss: 0.1724).
    
## 4. Evaluation method

- Prompt: 
  - The same standardized Chinese task prompt was used for all evaluation runs:
   - 请识别图片中的作物和病虫害，按「作物，病害」格式回答；若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。
- Image preprocessing: 
  - The same processor settings were used for all runs:
   - with min_pixels=200704 and max_pixels=401408.
- Decoding settings: Greedy decoding (do_sample=False) with max_new_tokens=32.
- Output parsing and normalization: 
  - Generated outputs were normalized by standardizing comma punctuation
  - Removing extraneous whitespace and periods
  - Extracting comma-separated fields for comparison with the ground-truth labels.
- Metric denominators: All metrics were calculated over the complete held-out evaluation set of 116 images (N = 116).
- Treatment of failed requests: 
  - Runtime inference exceptions were recorded as inference failures and received no accuracy credit. 
  - Output-format validity was evaluated separately based on whether the generated response contained the expected two or three comma-separated fields.

## 5. Results

| Metric | Base model | QLoRA baseline | Student variation (exp3) | Difference vs. baseline |
|:---|---:|---:|---:|---:|
| Crop accuracy| 81.03% (94/116) | 93.10% (108/116) | 94.83% (110/116) | +1.73% |
| Disease/pest accuracy | 2.59% (3/116) | 53.45% (62/116) | 56.90% (66/116) | +3.45% |
| Combined diagnosis accuracy | 1.72% (2/116) | 51.72% (60/116) | 56.03% (65/116) | +4.31% |
| Valid-output rate | 92.60% (107/116) | 100.0% (116/116) | 100.0% (116/116) | 0.00% |
| Inference failure rate | 0.0% (0/116) | 0.0% (0/116) | 0.0% (0/116) | 0.00% |

## 6. Error analysis
- **main points**
  - Fine-tuning substantially improved crop and disease/pest recognition compared with the base model. 
  - Combined diagnosis accuracy increased from 1.72% (2/116) for the base model to 51.72% (60/116) for the 3-epoch QLoRA baseline and 56.03% (65/116) for the 5-epoch variation. 
  - Inspection of individual predictions also showed that additional training corrected some baseline errors, although some previously correct predictions became incorrect.

- **Representative prediction changes:**

| Error-analysis example | Ground truth | 3-epoch baseline | 5-epoch variation | Outcome |
|---|---|---|---|---|
| Tomato disease | `番茄，疫病` | `番茄，病毒病` ❌ | `番茄，疫病` ✅ | Corrected |
| Pepper thrips | `辣椒，蓟马` | `辣椒，病毒病` ❌ | `辣椒，蓟马` ✅ | Corrected |
| Cucumber downy mildew | `黄瓜，霜霉病` | `黄瓜，螨，茶黄螨` ❌ | `黄瓜，霜霉病` ✅ | Corrected |
| Cucumber viral disease | `黄瓜，病毒病` | `黄瓜，病毒病` ✅ | `黄瓜，霜霉病` ❌ | Regression |

- **Effect of additional epochs:**
    - Increasing training from 3 to 5 epochs corrected **10 cases** that were incorrect in the 3-epoch baseline, while **5 previously correct cases became incorrect.**
    - This produced a net improvement of 5 correctly diagnosed images, increasing combined diagnosis accuracy from **51.72% (60/116)** to **56.03% (65/116).**

- **Remaining errors:**
  - Despite the overall improvement, the 5-epoch variation still produced 51 incorrect combined diagnoses out of 116 evaluation images.
  - This indicates that additional training improved overall performance but did not resolve all crop and disease/pest classification errors.

- **Generic predictions:**
  - The base model sometimes produced underspecified disease labels, such as 玉米，病害 (“corn, disease”) and 辣椒，病害 (“pepper, disease”), rather than predicting a specific disease/pest category. 
  - After QLoRA fine-tuning, predictions were more consistently aligned with the specific crop and disease/pest labels used in the AgroMind dataset.

- Data-quality problems: 
  - Inspection of the misclassified evaluation images showed that some difficult cases contained multiple or overlapping leaves, complex field backgrounds, or symptoms that were small or visually diffuse. These conditions may make fine-grained disease/pest recognition more difficult. However, misclassifications also occurred in relatively clear images, indicating that image quality alone does not explain the remaining errors. Some groups of related images were consistently confused between specific disease/pest classes, suggesting that further improvement in class discrimination is still needed.

## 7. Reproducibility
To ensure reproducibility, all configurations, commands, model artifacts, and evaluation results were documented for the base model, 3-epoch QLoRA baseline, and 5-epoch variation.

| Experiment | Training epochs | Adapter/output directory | Evaluation results |
|---|---:|---|---|
| Base model | — | No adapter | `results/eval_base_model_results.csv` |
| QLoRA baseline | 3 | `output_qlora_baseline/` | `results/eval_lora_baseline_results.csv` |
| Student variation | 5 | `output_qlora_exp3/` | `results/eval_lora_exp3_results.csv` |

- **QLoRA baseline Commands:**
  
  a) Training command:
  
      - `python3 scripts/train_qwen25vl_qlora_starter.py --train-csv phase2_data/train/labels.csv --image-root phase2_data/train/images --validation-csv phase2_data/validation/labels.csv --validation-image-root phase2_data/validation/images --output-dir output_qlora_baseline --config configs/baseline_3epoch_config.json`
  
  b) Evaluation command:
  
      - `python3 scripts/evaluate_qwen25vl_starter.py --csv phase2_data/evaluation/labels.csv --image-root phase2_data/evaluation/images --adapter output_qlora_baseline --output-csv results/eval_lora_baseline_results.csv --run-label "qlora_baseline"` 

- **5-epoch variation Commands:**
  
  a) Training command:
  
      - `python3 scripts/train_qwen25vl_qlora_starter.py --train-csv phase2_data/train/labels.csv --image-root phase2_data/train/images --validation-csv phase2_data/validation/labels.csv --validation-image-root phase2_data/validation/images --output-dir output_qlora_exp3 --config configs/variation_5epoch_config.json`
  
  b) Evaluation command:
  
      - `python3 scripts/evaluate_qwen25vl_starter.py --csv phase2_data/evaluation/labels.csv --image-root phase2_data/evaluation/images --adapter output_qlora_exp3 --output-csv results/eval_lora_exp3_results.csv --run-label "qlora_exp3"` 

- Output adapter path or artifact: 
  - LoRA adapter artifacts are stored in **output_qlora_baseline/** and **output_qlora_exp3/.**  
  - The corresponding adapter_model.safetensors weight files (~181 MB each) are stored on the server/shared storage and are not committed to GitHub due to the GitHub file-size limit.
  - Experiment configuration files are stored in **configs/.**  
  - Evaluation predictions and summaries are stored in **results/.**

- Code/configuration link: 
  - GitHub repository: https://github.com/WasanAlThobaiti/AgroMind-/tree/main/phase2

## 8. Conclusion

- Fine-tuning Qwen2.5-VL-7B-Instruct with QLoRA substantially improved AgroMind performance on the held-out evaluation set. 
- Combined diagnosis accuracy increased from **1.72% for the base model** to **51.72% for the 3-epoch QLoRA baseline**, and further increased to **56.03% for the 5-epoch variation**, which also achieved **94.83% crop accuracy.**
- Increasing training from 3 to 5 epochs reduced the final training loss from **0.2568 to 0.1724** and improved combined diagnosis accuracy by **4.31 percentage points**, although some individual predictions regressed while others were corrected. 
- Future work could investigate other controlled changes, such as LoRA rank or learning rate, and evaluate whether they further improve disease/pest recognition.

