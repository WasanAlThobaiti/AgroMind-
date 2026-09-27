import base64
import csv
import json
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:32509/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = """请识别图片中的作物和病虫害，按「作物，病害」格式回答；若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。"""

DATA_DIR = Path.home() / "aidc" / "agromind" / "data"
OUTPUT = Path.home() / "aidc" / "agromind" / "evaluation" / "results" / "benchmark_raw.csv"


LIMIT = None


def load_dataset():
    rows = []

    for part in ["part1", "part2"]:
        labels_file = DATA_DIR / part / "labels.csv"
        images_dir = DATA_DIR / part / "images"

        with open(labels_file, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                row["part"] = part
                row["image_path"] = str(images_dir / row["filename"])
                rows.append(row)

    return rows


def image_to_data_url(image_path):
    path = Path(image_path)

    if path.suffix.lower() == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


def run_request(row):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(row["image_path"])
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }

    request = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)

    latency = time.perf_counter() - start
    prediction = result["choices"][0]["message"]["content"].strip()

    return prediction, latency


def main():
    dataset = load_dataset()

    if LIMIT is not None:
        dataset = dataset[:LIMIT]

    print(f"Testing {len(dataset)} images")

    fields = [
        "part",
        "filename",
        "expected_crop",
        "expected_disease",
        "expected_disease_type",
        "prediction",
        "latency_seconds",
        "status",
        "error",
    ]

    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(dataset, start=1):
            try:
                prediction, latency = run_request(row)

                result = {
                    "part": row["part"],
                    "filename": row["filename"],
                    "expected_crop": row["crop"],
                    "expected_disease": row["disease"],
                    "expected_disease_type": row["disease_type"],
                    "prediction": prediction,
                    "latency_seconds": round(latency, 4),
                    "status": "ok",
                    "error": "",
                }

                print(
                    f"[{i}/{len(dataset)}] "
                    f"{row['filename']} -> {prediction} "
                    f"({latency:.2f}s)"
                )

            except Exception as e:
                result = {
                    "part": row["part"],
                    "filename": row["filename"],
                    "expected_crop": row["crop"],
                    "expected_disease": row["disease"],
                    "expected_disease_type": row["disease_type"],
                    "prediction": "",
                    "latency_seconds": "",
                    "status": "error",
                    "error": str(e),
                }

                print(f"[{i}/{len(dataset)}] ERROR: {e}")

            writer.writerow(result)
            out.flush()

    print(f"\nSaved results to: {OUTPUT}")


if __name__ == "__main__":
    main()