from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn
import gc

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
app = FastAPI(title="Qwen2.5-14B-Instruct API")

tokenizer = None
model = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@app.on_event("startup")
def load_model():
    global tokenizer, model
    print(f"Loading {MODEL_NAME} with 4-bit quantization...")
    
    # 4-bit quantization config for 4070 Super (12GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Qwen models typically have padding token, but verify
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Try loading with Flash Attention 2, fallback to eager if unavailable
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",  # Qwen 2.5 fully supports FA2
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print("Loaded with Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention 2 unavailable, using eager attention: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
    
    # Set model to eval mode
    model.eval()
    
    print("Model loaded successfully with 4-bit quantization!")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        # Prepare input with chat template
        messages = [{"role": "user", "content": request.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache for faster generation
            )
        
        # Decode only the generated tokens (exclude input)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"response": response_text}
        
    except Exception as e:
        # Clear cache on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None:
        return {"status": "not ready"}
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    
    return {
        "status": "ready",
        "model": MODEL_NAME,
        "memory_allocated_gb": round(memory_allocated, 2),
        "memory_reserved_gb": round(memory_reserved, 2)
    }

if __name__ == "__main__":
    uvicorn.run("Qwen_API:app", host="0.0.0.0", port=8002, reload=False)