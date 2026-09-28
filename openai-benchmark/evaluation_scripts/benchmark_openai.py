import base64
import mimetypes
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------
# Configuration
# ----------------------------

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini-2024-07-18"

PROMPT = (
    "请识别图片中的作物和病虫害，按「作物，病害」格式回答；"
    "若为虫害，再补充具体虫种，按「作物，病害，虫种」。"
    "只回答名称，不要解释。"
)

DATASET_PARTS = [
    Path("dataset/part1"),
    Path("dataset/part2"),
]

OUTPUT_PATH = Path("results/openai_500_predictions.csv")

MAX_RETRIES = 3


# ----------------------------
# Image encoding
# ----------------------------

def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ----------------------------
# Load full dataset
# ----------------------------

all_rows = []

for part_path in DATASET_PARTS:

    labels_path = part_path / "labels.csv"

    labels = pd.read_csv(labels_path)

    labels["source_part"] = part_path.name

    all_rows.append(labels)


dataset = pd.concat(all_rows, ignore_index=True)

print(f"Total dataset rows: {len(dataset)}")


# ----------------------------
# Resume support
# ----------------------------

OUTPUT_PATH.parent.mkdir(exist_ok=True)

if OUTPUT_PATH.exists():

    existing = pd.read_csv(OUTPUT_PATH)

    completed = set(
        zip(
            existing["source_part"],
            existing["filename"]
        )
    )

    results = existing.to_dict("records")

    print(f"Existing completed rows: {len(completed)}")

else:

    completed = set()
    results = []


# ----------------------------
# Benchmark loop
# ----------------------------

total = len(dataset)

for i, row in dataset.iterrows():

    source_part = row["source_part"]
    filename = row["filename"]

    key = (source_part, filename)

    # Skip completed images
    if key in completed:
        continue

    image_path = (
        Path("dataset")
        / source_part
        / "images"
        / filename
    )

    print(
        f"\n[{i + 1}/{total}] "
        f"{source_part}/{filename}"
    )

    # Image missing
    if not image_path.exists():

        print("ERROR: image not found")

        results.append({
            "source_part": source_part,
            "filename": filename,
            "true_crop": row["crop"],
            "true_disease": row["disease"],
            "true_disease_type": row["disease_type"],
            "crop_en": row["crop_en"],
            "disease_en": row["disease_en"],
            "prediction": "",
            "latency_seconds": "",
            "input_tokens": "",
            "output_tokens": "",
            "status": "image_not_found",
        })

        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig"
        )

        continue


    # ----------------------------
    # API request with retry
    # ----------------------------

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            image_url = image_to_data_url(image_path)

            start = time.perf_counter()

            response = client.responses.create(
                model=MODEL,
                temperature=0,
                max_output_tokens=50,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": PROMPT,
                            },
                            {
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": "high",
                            },
                        ],
                    }
                ],
            )

            latency = time.perf_counter() - start

            prediction = response.output_text.strip()

            # Usage information
            usage = getattr(response, "usage", None)

            input_tokens = (
                getattr(usage, "input_tokens", "")
                if usage else ""
            )

            output_tokens = (
                getattr(usage, "output_tokens", "")
                if usage else ""
            )

            print(
                "Ground truth:",
                row["crop"],
                row["disease"]
            )

            print(
                "Prediction:  ",
                prediction
            )

            print(
                "Latency:     ",
                round(latency, 3),
                "s"
            )

            results.append({
                "source_part": source_part,
                "filename": filename,
                "true_crop": row["crop"],
                "true_disease": row["disease"],
                "true_disease_type": row["disease_type"],
                "crop_en": row["crop_en"],
                "disease_en": row["disease_en"],
                "prediction": prediction,
                "latency_seconds": round(latency, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "status": "success",
            })

            success = True

            break


        except Exception as e:

            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed:"
            )

            print(e)

            if attempt < MAX_RETRIES:
                wait_seconds = attempt * 2

                print(
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)


    # All retries failed
    if not success:

        results.append({
            "source_part": source_part,
            "filename": filename,
            "true_crop": row["crop"],
            "true_disease": row["disease"],
            "true_disease_type": row["disease_type"],
            "crop_en": row["crop_en"],
            "disease_en": row["disease_en"],
            "prediction": "",
            "latency_seconds": "",
            "input_tokens": "",
            "output_tokens": "",
            "status": "api_error",
        })


    # ----------------------------
    # Save checkpoint after EVERY image
    # ----------------------------

    pd.DataFrame(results).to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )


# ----------------------------
# Final summary
# ----------------------------

final_df = pd.DataFrame(results)

successful = (
    final_df["status"] == "success"
).sum()

failed = len(final_df) - successful

print("\n==============================")
print("Benchmark finished")
print("==============================")

print(f"Total rows:       {len(final_df)}")
print(f"Successful:       {successful}")
print(f"Failed:           {failed}")

print(f"\nSaved to:")
print(OUTPUT_PATH)