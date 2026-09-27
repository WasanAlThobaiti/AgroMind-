# AgroMind Phase 1 Architecture

## System architecture

```text
Dataset on Project Server
        ↓
Benchmark Script
        ↓
Kubernetes NodePort Service
   `agromind-serving`
        ↓
Kubernetes Deployment / Pod
     `agromind-vllm`
        ↓
vLLM OpenAI-Compatible API
        ↓
Qwen/Qwen2.5-VL-7B-Instruct
        ↓
NVIDIA RTX A6000
```
**Main components:**
- Dataset: stored on the project server at ~/aidc/agromind/data/
- Service: agromind-serving
- Deployment: agromind-vllm
- Serving framework: vLLM
- Model: Qwen/Qwen2.5-VL-7B-Instruct
- GPU: NVIDIA RTX A6000

**Request flow:**
1. The benchmark script reads an image from the server dataset.
2. The image and AgroMind prompt are sent to the Kubernetes NodePort Service.
3. The Service forwards the request to the vLLM pod.
4. vLLM runs inference on the Qwen2.5-VL model using the RTX A6000.
5. The model response is returned to the benchmark script.

