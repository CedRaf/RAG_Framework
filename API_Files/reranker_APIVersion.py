import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    top_k: int = 5

class RerankResponse(BaseModel):
    reranked: list[dict]

reranker_tokenizer = None
reranker_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_reranker():
    global reranker_tokenizer, reranker_model, device
    if reranker_tokenizer is None or reranker_model is None:
        reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3",
            torch_dtype=torch.float16 
        )
        reranker_model = reranker_model.eval().to(device)

@app.on_event("startup")
def startup_event():
    initialize_reranker()

@app.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest):
    global reranker_tokenizer, reranker_model, device
    query = request.query
    passages = request.passages
    top_k = request.top_k
    
    if not passages:
        return {"reranked": []}

    try:
        all_scores = []
        batch_size = 16
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            batch_queries = [query] * len(batch_passages)
            inputs = reranker_tokenizer(
                batch_queries,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                batch_scores = reranker_model(**inputs).logits.squeeze(-1)
                all_scores.extend(batch_scores.cpu().tolist())
            del inputs, batch_scores

        ranked = sorted(zip(passages, all_scores), key=lambda x: x[1], reverse=True)
        reranked = [
            {"text": text, "score": float(score)}
            for text, score in ranked[:top_k]
        ]
        return {"reranked": reranked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("reranker_APIVersion:app", host="0.0.0.0", port=8500, reload=True)