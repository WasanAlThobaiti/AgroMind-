# AgroMind OpenAI Benchmark

This repository contains the GPT-4o mini benchmark and evaluation results for the AgroMind agricultural image dataset.

## Model

`gpt-4o-mini-2024-07-18`

## Dataset

The benchmark uses 500 agricultural images:

- Part 1: 250 images
- Part 2: 250 images

The dataset contains ground-truth labels for crop, disease, and disease type.

## Repository Structure

```text
agromind-openai-benchmark/
├── README.md
├── requirements.txt
├── evaluation_scripts/
│   ├── benchmark_openai.py
│   └── evaluate.py
└── results/
    ├── openai_500_predictions.csv
    ├── openai_500_evaluated.csv
    └── openai_500_summary.txt
```

## Files

### `evaluation_scripts/benchmark_openai.py`

Runs the GPT-4o mini benchmark on the 500 AgroMind images.

It records:

- Model prediction
- Ground-truth crop
- Ground-truth disease
- Latency
- Input tokens
- Output tokens
- Request status

The script also supports retries and resume from previously saved results.

### `evaluation_scripts/evaluate.py`

Evaluates the saved predictions and calculates:

- Crop accuracy
- Disease accuracy
- Combined diagnosis accuracy
- Valid-output rate
- Mean latency
- p50 latency
- p95 latency
- Request success rate
- Token usage
- Estimated API cost

### `results/openai_500_predictions.csv`

Contains the raw predictions and measurements produced during the benchmark.

### `results/openai_500_evaluated.csv`

Contains the predictions after parsing and evaluation, including correctness columns for crop, disease, and combined diagnosis.

### `results/openai_500_summary.txt`

Contains the final benchmark metrics and summary.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
openai
pandas
python-dotenv
```

## API Key

The OpenAI API key should be stored as an environment variable or inside a local `.env` file:

```text
OPENAI_API_KEY=your_api_key
```

## Run the Benchmark

From the project root directory:

```bash
python evaluation_scripts/benchmark_openai.py
```

The benchmark processes both dataset parts for a total of 500 images.

The output is saved to:

```text
results/openai_500_predictions.csv
```

If the benchmark is interrupted, the script can be run again to continue from previously saved results.

## Run the Evaluation

After the benchmark finishes:

```bash
python evaluation_scripts/evaluate.py
```

The evaluation generates:

```text
results/openai_500_evaluated.csv
results/openai_500_summary.txt
```

## Benchmark Configuration

The same configuration is kept fixed across all 500 images:

```text
Model: gpt-4o-mini-2024-07-18
Temperature: 0
Image detail: high
Maximum output tokens: 50
```

The same prompt is also used for every image:

```text
请识别图片中的作物和病虫害，按「作物，病害」格式回答；若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。
```

## Evaluation Method

The evaluation uses strict string matching.

Only the following normalization is applied:

- Whitespace removal
- Chinese comma `，` and English comma `,` normalization

No synonym or semantic matching is applied.

This keeps the evaluation method consistent and reproducible.
