"""Evaluate a base Qwen2.5-VL model or a QLoRA adapter on a CSV split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


PROMPT = (
    "请识别图片中的作物和病虫害，按「作物，病害」格式回答；"
    "若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。"
)


def normalize(value: str) -> str:
    value = str(value or "").strip()
    value = value.replace("，", ",").replace("：", ":")
    value = re.sub(r"\s+", "", value)
    return value


def parse_prediction(text: str) -> list[str]:
    text = text.strip().replace("，", ",")
    text = text.replace("作物:", "").replace("作物：", "")
    text = text.replace("病害:", ",").replace("病害：", ",")
    text = text.replace("虫种:", ",").replace("虫种：", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def target(row: pd.Series) -> list[str]:
    parts = [str(row["crop"]), str(row["disease"])]
    pest = str(row.get("虫子名", row.get("pest", "")) or "").strip()
    if pest and pest != parts[1]:
        parts.append(pest)
    return parts


def load_model(model_id: str, adapter: str | None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    processor = AutoProcessor.from_pretrained(
        adapter or model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()
    return model, processor


def predict(model, processor, image_path: str) -> str:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    generated = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--run-label", required=True, help="For example: base, qlora-baseline, or student-variation")
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    rows = pd.read_csv(args.csv).fillna("")
    if args.max_images:
        rows = rows.head(args.max_images)
    model, processor = load_model(args.model, args.adapter)

    output_rows = []
    crop_hits = disease_hits = combined_hits = valid_hits = 0
    failures = 0
    for _, row in rows.iterrows():
        image_path = os.path.join(args.image_root, str(row["filename"]))
        try:
            raw = predict(model, processor, image_path)
            parts = parse_prediction(raw)
            truth = target(row)
            crop_ok = bool(parts) and normalize(parts[0]) == normalize(truth[0])
            disease_ok = len(parts) > 1 and normalize(parts[1]) == normalize(truth[1])
            combined_ok = crop_ok and disease_ok
            valid = 2 <= len(parts) <= 3
        except Exception as exc:  # keep failed requests visible in the report
            raw = f"ERROR: {type(exc).__name__}: {exc}"
            parts = []
            truth = target(row)
            crop_ok = disease_ok = combined_ok = valid = False
            failures += 1

        crop_hits += int(crop_ok)
        disease_hits += int(disease_ok)
        combined_hits += int(combined_ok)
        valid_hits += int(valid)
        output_rows.append(
            {
                "filename": row["filename"],
                "ground_truth": "，".join(truth),
                "prediction": raw,
                "crop_correct": crop_ok,
                "disease_correct": disease_ok,
                "combined_correct": combined_ok,
                "valid_format": valid,
            }
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    n = len(rows)
    summary = {
        "run_label": args.run_label,
        "model": args.model,
        "adapter": args.adapter,
        "images_evaluated": n,
        "crop_accuracy": crop_hits / n,
        "disease_accuracy": disease_hits / n,
        "combined_accuracy": combined_hits / n,
        "valid_output_rate": valid_hits / n,
        "failure_rate": failures / n,
        "output_csv": args.output_csv,
    }
    summary_path = args.output_summary or os.path.splitext(args.output_csv)[0] + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Run label: {args.run_label}")
    print(f"Images evaluated: {n}")
    print(f"Crop accuracy: {summary['crop_accuracy']:.4f}")
    print(f"Disease accuracy: {summary['disease_accuracy']:.4f}")
    print(f"Combined accuracy: {summary['combined_accuracy']:.4f}")
    print(f"Valid-output rate: {summary['valid_output_rate']:.4f}")
    print(f"Failure rate: {summary['failure_rate']:.4f}")
    print(f"Predictions written to: {args.output_csv}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
