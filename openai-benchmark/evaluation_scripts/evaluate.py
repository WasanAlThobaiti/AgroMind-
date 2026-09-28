import re
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_PATH = Path("results/openai_500_predictions.csv")
OUTPUT_PATH = Path("results/openai_500_evaluated.csv")
SUMMARY_PATH = Path("results/openai_500_summary.txt")

# GPT-4o mini API pricing
# USD per 1 million tokens
INPUT_PRICE_PER_M = 0.15
OUTPUT_PRICE_PER_M = 0.60


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Normalize text for exact comparison.
    We only normalize whitespace and Chinese/English commas.
    We do NOT apply synonym mapping.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Chinese comma -> English comma
    text = text.replace("，", ",")

    # Remove whitespace
    text = re.sub(r"\s+", "", text)

    return text


# ============================================================
# Parse model prediction
# ============================================================

def parse_prediction(prediction):
    prediction = normalize_text(prediction)

    if not prediction:
        return "", "", "", 0

    parts = [
        part.strip()
        for part in prediction.split(",")
        if part.strip()
    ]

    # Required format:
    # crop,disease
    # OR
    # crop,disease,pest
    valid_format = 1 if len(parts) in (2, 3) else 0

    pred_crop = parts[0] if len(parts) >= 1 else ""
    pred_disease = parts[1] if len(parts) >= 2 else ""
    pred_pest = parts[2] if len(parts) >= 3 else ""

    return (
        pred_crop,
        pred_disease,
        pred_pest,
        valid_format,
    )


# ============================================================
# Load benchmark results
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {INPUT_PATH}. "
        "Run benchmark_openai.py first."
    )

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df)} rows from {INPUT_PATH}")


# ============================================================
# Basic request status
# ============================================================

df["status"] = df["status"].fillna("")

success_mask = df["status"] == "success"
image_missing_mask = df["status"] == "image_not_found"
api_error_mask = df["status"] == "api_error"

total_rows = len(df)

successful_requests = int(success_mask.sum())
missing_images = int(image_missing_mask.sum())
api_errors = int(api_error_mask.sum())

available_images = total_rows - missing_images


# ============================================================
# Parse predictions
# ============================================================

parsed = df["prediction"].apply(parse_prediction)

df["pred_crop"] = parsed.apply(lambda x: x[0])
df["pred_disease"] = parsed.apply(lambda x: x[1])
df["pred_pest"] = parsed.apply(lambda x: x[2])
df["valid_format"] = parsed.apply(lambda x: x[3])


# ============================================================
# Strict accuracy
# ============================================================

df["crop_correct"] = (
    df["pred_crop"].apply(normalize_text)
    ==
    df["true_crop"].apply(normalize_text)
).astype(int)

df["disease_correct"] = (
    df["pred_disease"].apply(normalize_text)
    ==
    df["true_disease"].apply(normalize_text)
).astype(int)

df["combined_correct"] = (
    (df["crop_correct"] == 1)
    &
    (df["disease_correct"] == 1)
).astype(int)


# ============================================================
# Quality metrics
# Only successful model responses
# ============================================================

successful_df = df[success_mask].copy()

if len(successful_df) > 0:

    crop_accuracy = (
        successful_df["crop_correct"].mean() * 100
    )

    disease_accuracy = (
        successful_df["disease_correct"].mean() * 100
    )

    combined_accuracy = (
        successful_df["combined_correct"].mean() * 100
    )

    valid_output_rate = (
        successful_df["valid_format"].mean() * 100
    )

else:

    crop_accuracy = 0
    disease_accuracy = 0
    combined_accuracy = 0
    valid_output_rate = 0


# ============================================================
# End-to-end accuracy
#
# This treats API errors as unsuccessful diagnoses.
# Missing dataset images are excluded.
# ============================================================

if available_images > 0:

    end_to_end_crop = (
        df.loc[~image_missing_mask, "crop_correct"].sum()
        / available_images
        * 100
    )

    end_to_end_disease = (
        df.loc[~image_missing_mask, "disease_correct"].sum()
        / available_images
        * 100
    )

    end_to_end_combined = (
        df.loc[
            ~image_missing_mask,
            "combined_correct"
        ].sum()
        / available_images
        * 100
    )

else:

    end_to_end_crop = 0
    end_to_end_disease = 0
    end_to_end_combined = 0


# ============================================================
# Latency metrics
# ============================================================

latencies = pd.to_numeric(
    successful_df["latency_seconds"],
    errors="coerce"
).dropna()

if len(latencies) > 0:

    mean_latency = latencies.mean()
    p50_latency = latencies.quantile(0.50)
    p95_latency = latencies.quantile(0.95)
    min_latency = latencies.min()
    max_latency = latencies.max()

else:

    mean_latency = 0
    p50_latency = 0
    p95_latency = 0
    min_latency = 0
    max_latency = 0


# ============================================================
# Token usage
# ============================================================

input_tokens = pd.to_numeric(
    successful_df["input_tokens"],
    errors="coerce"
).fillna(0)

output_tokens = pd.to_numeric(
    successful_df["output_tokens"],
    errors="coerce"
).fillna(0)

total_input_tokens = int(input_tokens.sum())
total_output_tokens = int(output_tokens.sum())


# ============================================================
# Estimated API cost
# ============================================================

input_cost = (
    total_input_tokens
    / 1_000_000
    * INPUT_PRICE_PER_M
)

output_cost = (
    total_output_tokens
    / 1_000_000
    * OUTPUT_PRICE_PER_M
)

total_cost = input_cost + output_cost

if successful_requests > 0:
    cost_per_image = total_cost / successful_requests
else:
    cost_per_image = 0

estimated_cost_1000 = cost_per_image * 1000


# ============================================================
# Request reliability
# ============================================================

if available_images > 0:
    request_success_rate = (
        successful_requests
        / available_images
        * 100
    )
else:
    request_success_rate = 0


# ============================================================
# Save evaluated CSV
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Create summary
# ============================================================

summary = f"""
GPT-4o mini — AgroMind Benchmark Summary
=========================================

Model:
gpt-4o-mini-2024-07-18

DATASET
-------
Total dataset rows: {total_rows}
Available images: {available_images}
Successful requests: {successful_requests}
API errors: {api_errors}
Missing images: {missing_images}
Request success rate: {request_success_rate:.2f}%

MODEL QUALITY
-------------
Crop accuracy: {crop_accuracy:.2f}%
Disease accuracy: {disease_accuracy:.2f}%
Combined diagnosis accuracy: {combined_accuracy:.2f}%
Valid-output rate: {valid_output_rate:.2f}%

END-TO-END QUALITY
------------------
Crop accuracy: {end_to_end_crop:.2f}%
Disease accuracy: {end_to_end_disease:.2f}%
Combined diagnosis accuracy: {end_to_end_combined:.2f}%

LATENCY
-------
Mean latency: {mean_latency:.3f} s
p50 latency: {p50_latency:.3f} s
p95 latency: {p95_latency:.3f} s
Minimum latency: {min_latency:.3f} s
Maximum latency: {max_latency:.3f} s

TOKEN USAGE
-----------
Input tokens: {total_input_tokens:,}
Output tokens: {total_output_tokens:,}

ESTIMATED API COST
------------------
Input cost: ${input_cost:.4f}
Output cost: ${output_cost:.4f}
Total cost: ${total_cost:.4f}
Average cost/image: ${cost_per_image:.6f}
Estimated cost/1,000 images: ${estimated_cost_1000:.4f}

EVALUATION METHOD
-----------------
Crop and disease accuracy use strict normalized string matching.
Whitespace and Chinese/English comma differences are normalized.
No synonym mapping or semantic matching is applied.
"""


SUMMARY_PATH.write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)


# ============================================================
# Print summary
# ============================================================

print()
print(summary)

print("Files saved:")
print(f"  Evaluated results: {OUTPUT_PATH}")
print(f"  Summary:           {SUMMARY_PATH}")