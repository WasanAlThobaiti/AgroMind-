# AI DATA CENTER Bootcamp — AgroMind Benchmark Report

## 1. Executive summary

- Model: Qwen/Qwen2.5-VL-7B-Instruct
- Deployment method: vLLM deployed on Kubernetes with a NodePort service
- GPU: NVIDIA RTX A6000, 49,140 MiB VRAM
- Main result: 
    - The final Phase 1 benchmark completed successfully on all 500 images
    - 500/500 successful requests
    - zero request errors
    - The deployment achieved: 0.359 s p50 latency, 0.585 s p95 latency, and up to 7.88 requests/s at concurrency 8
- Main limitation:
    - Model-quality performance was limited, especially for disease identification.
    - Crop accuracy was 49.60%
    - disease/pest accuracy was 7.80%
    - combined diagnosis accuracy was 3.40%

## 2. Model and deployment

- Model name and version: Qwen/Qwen2.5-VL-7B-Instruct
- Serving framework and version: vLLM 0.7.3
- Container/runtime details:
    - Container image: vllm/vllm-openai:v0.7.3
    - Kubernetes namespace: team
    - Deployment: agromind-vllm
    - Service: agromind-serving
    - Service type: NodePort
    - Internal serving port: 8000
    - Benchmark endpoint: http://127.0.0.1:32509/v1/chat/completions
    - GPU resource: 1 x NVIDIA RTX A6000
    - Pod memory limit: 16 GiB
    - dtype=bfloat16
    - max_model_len=8192
    - gpu_memory_utilization=0.85
    - limit_mm_per_prompt=image=1
    - Multimodal preprocessing:
        - min_pixels=200704
        - max_pixels=401408

- Input and output API:
  - Input: OpenAI-compatible /v1/chat/completions API with one agricultural image and one text prompt.
  - Output: The model returns a short Chinese crop/disease prediction.  

- Prompt and decoding settings:
    - Prompt: 请识别图片中的作物和病虫害，按「作物，病害」格式回答；若为虫害，再补充具体虫种，按「作物，病害，虫种」。只回答名称，不要解释。
    - decoding settings: temperature = 0, max_tokens = 64

**NOTE: The same prompt, decoding settings, model version, and image preprocessing settings were kept fixed for the final benchmark.**

## 3. Dataset and evaluation method

- Number of images evaluated: 500 
- Label fields used:
    - filename
    - crop
    - disease
    - disease_type
    - crop_en
    - disease_en
    - disease_type_en
- Matching/parsing method:
    - Model outputs were parsed as comma-separated fields.
    - Both the Chinese comma ， and English comma , were accepted as separators.
    - Surrounding whitespace was stripped from parsed fields before comparison.
    - A two-field response was interpreted as crop, disease.
    - A three-field response was interpreted as crop, disease, pest species.
    - Crop and disease/pest labels were compared against the Chinese ground-truth labels using exact string matching after parsing.
    - Crop accuracy was counted as correct only when the predicted crop exactly matched the ground-truth crop.
    - Disease/pest accuracy was counted as correct only when the predicted disease/pest label exactly matched the ground-truth disease/pest label.
    - Combined diagnosis was counted as correct only when both crop and disease/pest were correct.
    - A response was counted as format-valid when it parsed into the expected two- or three-field structure.
    - Pest-species accuracy was not reported as a separate Phase 1 metric.

- Known label or image-quality issues:
    - Issue: Some high-resolution images initially generated too many multimodal tokens and exceeded the configured context length.
    - Solution: Fixed image preprocessing limits were applied consistently to all benchmark images. After this change, the final 500-image benchmark completed with no HTTP 400 errors and no serving failures.

## 4. Model-quality results

| Metric                            | correct / total | Result |
|-----------------------------------|---------------- |--------|
| Crop accuracy                     |   248 / 500     | 49.60% |
| Disease/pest accuracy             |   39  / 500     | 7.80%  |
| Combined diagnosis accuracy       |   17  / 500     | 3.40%  |
| Valid-output rate                 |   463 / 500     | 92.60% |

### Request success vs. valid output
- **Serving success:** 500/500 requests completed successfully and returned a model response.
- **Valid-output format:** 463/500 responses matched the expected two- or three-field output structure.
- **Invalid-format outputs:** 37/500 requests returned a response but could not be parsed into the required structure.

*NOTE: A successful request therefore does not necessarily mean that the model followed the required output format.*

**Include a few correct predictions and failure cases:**
|Case No.| Correctness | Ground Truth |    Model Prediction    |              NOTES                 |
|--------|-------------|--------------|------------------------|------------------------------------|
|    1   |  Correct    |  葡萄，白粉病  |       葡萄，白粉病       | Both are Correct (Crop & Disease)  |
|    2   |  Failure1   |  柑橘，炭疽病  |       柑橘，叶斑病       |  Crop Correct, Disease Incorrect   |
|    3   |  Failure2   |  黄瓜，白粉病  |       南瓜，白粉病       |  Crop Incorrect, Disease Correct   |
|    4   |  Failure3   |  苹果，轮纹病  |    番茄，番茄斑萎病毒     |  Crop Incorrect, Disease Incorrect |


## 5. Infrastructure benchmark

| Metric              |                               Result                                    |
|---------------------|-------------------------------------------------------------------------|
| GPU and VRAM        | NVIDIA RTX A6000; 49,140 MiB total VRAM; 41,751 MiB used; 6,919 MiB free|
| Model loading time  |       Approx. 153 s initial startup including weight download.          |
| p50 latency         |       0.359 s                                                           |
| p95 latency         |       0.585 s                                                           |
| Throughput          |       Up to 7.88 requests/s at concurrency 8                            |
| Concurrency tested  |       1, 2, 4, 8                                                        |
| Estimated cost      |       ~$0.030 for the 500-image                                         |

*IMPORTANT NOTE: The estimated cost was computed using a reference RTX A6000 rate of $0.57/hour*
- **Estimated inference cost:**
The 500-image sequential benchmark required approximately 0.0528 GPU-hours.
Using a reference RTX A6000 rental rate of $0.57/hour:
    - Estimated cost for 500 requests: ~$0.030
    - Estimated cost per request: ~$0.000060
    - Estimated cost per 1,000 requests: ~$0.060

- **Additional latency results**
    | Metric                            | Result  |
    |-----------------------------------|---------|
    | Successful requests               | 500/500 |
    | Mean latency                      | 0.380 s |
    | Minimum latency                   | 0.219 s |
    | Maximum latency                   | 1.852 s |

- **Throughput by concurrency**
    |Concurrency| Throughput  | Average latency |   p95 latency   | Errors |
    |-----------|-------------|-----------------|-----------------|--------|
    |     1     |  3.25 req/s |     0.307 s     |     0.491 s     |    0   |
    |     2     |  4.79 req/s |     0.410 s     |     0.749 s     |    0   |
    |     4     |  6.20 req/s |     0.630 s     |     1.078 s     |    0   |
    |     8     |  7.88 req/s |     0.967 s     |     1.578 s     |    0   |

    - Benchmark methodology:
      
    *NOTE: The full 500-image sequential benchmark and the dedicated concurrency benchmark were separate test runs and should not be interpreted as identical measurements.*

    a) The full sequential benchmark processed all 500 images and was used for the main quality and latency results.

    b) The concurrency benchmark was a separate controlled test using 40 requests per concurrency level to measure throughput and latency under concurrent load.

    Clarification: The difference in request-rate values is expected because the two benchmarks used different test runs, sample sizes, and workloads.


- **Model startup details**
    | Startup component                          | Result  |
    |--------------------------------------------|---------|
    | Model-weight download                      | 79.55 s |
    | Safetensor checkpoint loading              |  ~21 s  |
    | Engine initialization / KV cache / warm-up | 36.05 s |
    | CUDA graph capture                         |   18 s  | 
    | Approx. initial startup including download |  ~153 s |  
    | Model weights loaded                       |15.627 GB| 

- **Model startup details**
    - Peak observed pod RAM after the preprocessing fix: 12,361 MiB
    - Final benchmark: 500/500 successful requests
    - Request errors in final run: 0
    - Pod restarts during final run: 0
    - Total sequential inference time for 500 images: 190.23 s
    - Equivalent compute time: 0.0528 GPU-hours

- **Deployment Challenges**
    - issue1: Unstable endpoint during long benchmark runs
      - Solution: Replaced `kubectl port-forward` with a Kubernetes `NodePort` Service (`agromind-serving`) to provide a more stable endpoint for benchmarking.
    - issue2: Large images exceeded the model context window
      - Solution: Applied fixed image preprocessing limits (`min_pixels=200704`, `max_pixels=401408`) to reduce excessive image tokens and stabilize memory usage.

## 6. Reproducibility artifacts
The required artifacts to reproduce the Phase 1 deployment and benchmark:
|                Artifact                     |               Purpose                 |
|---------------------------------------------|---------------------------------------|
| `agromind-deployment.yaml`                  | Kubernetes deployment configuration   |
| `agromind-service.yaml`                     | Kubernetes service configuration      |
| `evaluation/run_benchmark.py`               | Runs the full 500-image benchmark     |
| `evaluation/scoring.py`                     | Calculates the quality metrics        |
| `evaluation/benchmark_concurrency.py`       | Runs the concurrency/throughput tests |
| `evaluation/results/benchmark_raw.csv`      | Raw per-image benchmark results, including expected labels, model prediction, latency, request status, and errors |
| `evaluation/results/benchmark_scored.csv`   | Scored results used to calculate accuracy and format-validity metrics |
| `evaluation/results/concurrency_results.csv`| Results for concurrency levels 1, 2, 4, and 8                         |

*NOTE: The raw results are preserved before scoring so that the reported metrics can be traced back to the original model outputs and request-level measurements.*

- **Main commands**:
  1. Deploy the model and service:
     - kubectl apply -f agromind-deployment.yaml
     - kubectl apply -f agromind-service.yaml
  2. Check Kubernetes resources:
     - kubectl get pods -n team
     - kubectl get svc -n team
  3. Run the full benchmark:
     - python3 evaluation/run_benchmark.py
  4. Score the predictions:
     - python3 evaluation/scoring.py
  5. Run the concurrency benchmark:
     - python3 evaluation/benchmark_concurrency.py

- **Repository**:
    - Repository URL: https://github.com/WasanAlThobaiti/AgroMind-.git
    - Commit hash: 7b883c092d65bec9c1490e035526fafbfad88988

## 7. Conclusion
 - The AgroMind Phase 1 deployment is reproducible because the model, serving configuration, Kubernetes resources, preprocessing settings, and evaluation workflow are all documented.

- The deployment was stable for the Phase 1 benchmark and successfully completed the required evaluation workload. It provides a reproducible baseline for the next project phase and a reference point for future model and infrastructure evaluations.

- The current results should be interpreted as benchmark evidence rather than proof of general production readiness.
