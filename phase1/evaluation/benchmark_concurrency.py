import base64
import csv
import json
import mimetypes
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "http://127.0.0.1:32509/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = """请识别图片中的作物和病虫害，按「作物，病害」格式回答；若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。"""

CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 40
TIMEOUT = 180

DATA_ROOT = Path("data")


def collect_images():
    images = []

    for part in ["part1", "part2"]:
        label_file = DATA_ROOT / part / "labels.csv"

        with open(label_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_path = DATA_ROOT / part / "images" / row["filename"]

                if image_path.exists():
                    images.append(image_path)

    return images


def prepare_payload(image_path):
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}"
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


def send_request(payload):
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        BASE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            response.read()

        latency = time.perf_counter() - start
        return True, latency, ""

    except Exception as e:
        latency = time.perf_counter() - start
        return False, latency, str(e)


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


images = collect_images()[:REQUESTS_PER_LEVEL]

print(f"Preparing {len(images)} image requests...")
payloads = [prepare_payload(img) for img in images]

print("\nStarting throughput benchmark...\n")

results = []

for concurrency in CONCURRENCY_LEVELS:

    latencies = []
    errors = []

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, payload)
            for payload in payloads
        ]

        for future in as_completed(futures):
            ok, latency, error = future.result()

            if ok:
                latencies.append(latency)
            else:
                errors.append(error)

    wall_time = time.perf_counter() - wall_start

    successful = len(latencies)
    throughput = successful / wall_time if wall_time > 0 else 0

    avg_latency = (
        sum(latencies) / len(latencies)
        if latencies else 0
    )

    p95 = percentile(latencies, 0.95)

    row = {
        "concurrency": concurrency,
        "requests": REQUESTS_PER_LEVEL,
        "successful": successful,
        "errors": len(errors),
        "throughput_req_s": throughput,
        "avg_latency_s": avg_latency,
        "p95_latency_s": p95,
        "wall_time_s": wall_time,
    }

    results.append(row)

    print(
        f"Concurrency {concurrency}: "
        f"throughput={throughput:.2f} req/s | "
        f"avg={avg_latency:.3f}s | "
        f"p95={p95:.3f}s | "
        f"errors={len(errors)}"
    )

    if errors:
        print("First error:", errors[0])

output = Path("evaluation/results/concurrency_results.csv")

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to: {output}")
