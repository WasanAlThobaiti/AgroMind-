"""Educational QLoRA starter for Qwen2.5-VL-7B-Instruct.

Expected CSV columns:
    filename,crop,disease,虫子名

The script intentionally keeps the first experiment small and explicit. It
does not include the Phase 1 evaluation set; provide a separate training CSV
and image directory.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


PROMPT = (
    "请识别图片中的作物和病虫害，按「作物，病害」格式回答；"
    "若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。"
)


def target_text(row: pd.Series) -> str:
    pest = str(row.get("虫子名", row.get("pest", "")) or "").strip()
    disease = str(row["disease"]).strip()
    crop = str(row["crop"]).strip()
    answer = f"{crop}，{disease}"
    if pest and pest != disease:
        answer += f"，{pest}"
    return answer


def make_messages(row: pd.Series, image_root: str) -> list[dict[str, Any]]:
    image_path = os.path.join(image_root, str(row["filename"]))
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_text(row)}],
        },
    ]
    return messages


class AgroMindDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, image_root: str):
        self.rows = pd.read_csv(csv_path).fillna("")
        required = {"filename", "crop", "disease"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return make_messages(self.rows.iloc[index], self.image_root)


@dataclass
class VisionDataCollator:
    processor: Any

    def __call__(self, batch_messages: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
        self.processor.tokenizer.padding_side = "right"
        texts: list[str] = []
        images: list[Any] = []
        prompt_lengths: list[int] = []

        for messages in batch_messages:
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, _ = process_vision_info(messages)
            texts.append(full_text)
            images.append(image_inputs)

            prompt_only = messages[:-1]
            prompt_text = self.processor.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            )
            prompt_images, _ = process_vision_info(prompt_only)
            prompt_batch = self.processor(
                text=[prompt_text],
                images=[prompt_images],
                return_tensors="pt",
            )
            prompt_lengths.append(int(prompt_batch["input_ids"].shape[1]))

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, :prompt_length] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--validation-image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True, help="Completed student experiment JSON")
    return parser.parse_args()


def load_student_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "experiment_name",
        "hypothesis",
        "model",
        "epochs",
        "learning_rate",
        "batch_size",
        "gradient_accumulation_steps",
        "seed",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    for field in ("experiment_name", "hypothesis"):
        if not str(config[field]).strip() or "REPLACE_WITH" in str(config[field]):
            raise ValueError(f"Complete the student field before running: {field}")
    if config["model"] != "Qwen/Qwen2.5-VL-7B-Instruct":
        raise ValueError("The required Phase 2 baseline model is Qwen/Qwen2.5-VL-7B-Instruct")
    if not config["target_modules"]:
        raise ValueError("target_modules must contain at least one module name")
    return config


def main() -> None:
    args = parse_args()
    config = load_student_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    validation_dataset = AgroMindDataset(args.validation_csv, args.validation_image_root)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Hypothesis: {config['hypothesis']}")
    print(f"Training samples: {len(AgroMindDataset(args.train_csv, args.image_root))}")
    print(f"Validation samples: {len(validation_dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        config["model"],
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = AgroMindDataset(args.train_csv, args.image_root)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        report_to="none",
        seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VisionDataCollator(processor),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "student_experiment_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"Saved LoRA adapter and processor to: {args.output_dir}")


if __name__ == "__main__":
    main()
