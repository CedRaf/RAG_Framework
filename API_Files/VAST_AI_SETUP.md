# Qwen API Deployment on Vast.AI (4070 Super)

## Setup Instructions

### 1. Create Vast.AI Instance
- Select **RTX 4070 Super** (12GB VRAM)
- Choose PyTorch template or Ubuntu 22.04 with CUDA 12.1+
- Recommended: At least 32GB RAM, 100GB disk space

### 2. SSH into Instance
```bash
ssh -p [PORT] root@[VAST_IP] -L 8002:localhost:8002
```

### 3. Install Dependencies
```bash
# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements_qwen.txt

# If flash-attn fails to install, use pre-built wheel:
pip install flash-attn --no-build-isolation
```

### 4. Run the API Server
```bash
python Qwen_API.py
```

The server will start on `0.0.0.0:8002` and can be accessed from your local machine via the SSH tunnel.

### 5. Test the API

**Health Check:**
```bash
curl http://localhost:8002/health
```

**Generate Request:**
```bash
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

## Optimizations Applied

1. **4-bit Quantization (NF4)**: Reduces memory from ~28GB to ~7-8GB
2. **Double Quantization**: Further compresses the model
3. **Flash Attention 2**: 2-4x faster inference with lower memory
4. **BFloat16**: Better numerical stability than FP16
5. **KV Cache**: Enabled for faster sequential token generation
6. **Automatic Memory Cleanup**: Clears CUDA cache after each request

## Expected Performance

- **Model Size**: ~7-8GB VRAM (fits comfortably on 4070 Super)
- **Inference Speed**: ~15-30 tokens/second (depending on context length)
- **Quality**: Minimal degradation with 4-bit NF4 quantization

## Local Client Example

```python
import requests

def query_qwen(prompt, max_tokens=512):
    response = requests.post(
        "http://localhost:8002/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    )
    return response.json()["response"]

# Usage
answer = query_qwen("Explain quantum computing in simple terms.")
print(answer)
```

## Troubleshooting

**Out of Memory Error:**
- Reduce `max_new_tokens` in requests
- Consider using 3-bit quantization (change `load_in_4bit` to `load_in_3bit`)

**Slow First Request:**
- First generation loads model weights into GPU (30-60 seconds)
- Subsequent requests will be much faster

**Flash Attention Installation Issues:**
```bash
# Alternative: Skip flash attention (will use standard attention)
# Comment out the attn_implementation line in Qwen_API.py
```
