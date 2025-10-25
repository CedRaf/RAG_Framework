from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import uvicorn

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
app = FastAPI(title="Qwen2.5-7B-Instruct API")

tokenizer = None
model = None
pipe = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512  # Reduced to 512 tokens
    temperature: float = 0.7

@app.on_event("startup")
def load_model():
    global tokenizer, model, pipe
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    # Remove the device parameter when using device_map="auto"
    pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    print("Model loaded successfully!")

@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        output = pipe(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            return_full_text=False  # This ensures only new text is returned
        )
        return {"response": output[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("qwen_api:app", host="0.0.0.0", port=8002)