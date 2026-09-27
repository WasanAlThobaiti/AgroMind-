import csv
import re
from pathlib import Path

INPUT = Path.home() / "aidc/agromind/evaluation/results/benchmark_raw.csv"
OUTPUT = Path.home() / "aidc/agromind/evaluation/results/benchmark_scored.csv"


def parse_prediction(text):
    # Accept Chinese or English commas and remove extra spaces
    parts = [
        p.strip()
        for p in re.split(r"[，,]", text.strip())
        if p.strip()
    ]

    crop = parts[0] if len(parts) >= 1 else ""
    disease = parts[1] if len(parts) >= 2 else ""

    return crop, disease, len(parts)


rows = []

with open(INPUT, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["status"] != "ok":
            row["predicted_crop"] = ""
            row["predicted_disease"] = ""
            row["crop_correct"] = "0"
            row["disease_correct"] = "0"
            row["combined_correct"] = "0"
            row["format_valid"] = "0"
            rows.append(row)
            continue

        pred_crop, pred_disease, num_parts = parse_prediction(
            row["prediction"]
        )

        crop_correct = pred_crop == row["expected_crop"]
        disease_correct = pred_disease == row["expected_disease"]

        # Official format should contain 2 fields,
        # or 3 fields when a pest species is returned.
        format_valid = num_parts in (2, 3)

        row["predicted_crop"] = pred_crop
        row["predicted_disease"] = pred_disease
        row["crop_correct"] = str(int(crop_correct))
        row["disease_correct"] = str(int(disease_correct))
        row["combined_correct"] = str(
            int(crop_correct and disease_correct)
        )
        row["format_valid"] = str(int(format_valid))

        rows.append(row)


fields = list(rows[0].keys())

with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


total = len(rows)

crop_accuracy = sum(int(r["crop_correct"]) for r in rows) / total
disease_accuracy = sum(int(r["disease_correct"]) for r in rows) / total
combined_accuracy = sum(int(r["combined_correct"]) for r in rows) / total
format_rate = sum(int(r["format_valid"]) for r in rows) / total

print(f"Images: {total}")
print(f"Crop accuracy:       {crop_accuracy:.2%}")
print(f"Disease accuracy:    {disease_accuracy:.2%}")
print(f"Combined accuracy:   {combined_accuracy:.2%}")
print(f"Valid-output rate:   {format_rate:.2%}")
print(f"\nSaved: {OUTPUT}")