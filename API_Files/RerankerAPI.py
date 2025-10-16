"""
Reranker API Service
A separate microservice for text reranking using BGE or Qwen3 reranker models.
Exposes a FastAPI endpoint that accepts a query and list of passages, returns ranked results.
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import uvicorn

app = FastAPI(title="Reranker API", description="Text reranking microservice")

# Global model variables
reranker_model = None
reranker_tokenizer = None
MODEL_NAME = "Qwen/Qwen3-Reranker-4B"  # Change to "Qwen/Qwen3-Reranker-4B" if needed

class RerankRequest(BaseModel):
    query: str
    passages: List[str]
    top_k: int = 5

class RerankResponse(BaseModel):
    results: List[dict]  # [{"text": str, "score": float, "rank": int}]
    model_name: str

def initialize_reranker():
    """Load the reranker model into memory"""
    global reranker_model, reranker_tokenizer
    
    if reranker_model is None:
        print(f"Loading reranker model: {MODEL_NAME}")
        reranker_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        reranker_model.eval()
        
        # Start on CPU, move to GPU during inference
        reranker_model = reranker_model.to("cpu")
        print("Reranker model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Initialize model when API starts"""
    initialize_reranker()

@app.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest):
    """
    Rerank passages based on relevance to query
    
    Args:
        query: The search query
        passages: List of text passages to rerank
        top_k: Number of top results to return
    
    Returns:
        Ranked list of passages with scores
    """
    global reranker_model, reranker_tokenizer
    
    if not request.passages:
        return RerankResponse(results=[], model_name=MODEL_NAME)
    
    if reranker_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker_model = reranker_model.to(device)
        
        # BGE reranker format: separate query and passage lists
        if "bge" in MODEL_NAME.lower():
            inputs = reranker_tokenizer(
                [request.query] * len(request.passages),
                request.passages,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                scores = reranker_model(**inputs).logits.squeeze(-1)
        
        # Qwen3 reranker format: combined "query: X document: Y" strings
        elif "qwen" in MODEL_NAME.lower():
            pairs = [f"query: {request.query} document: {passage}" for passage in request.passages]
            
            inputs = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = reranker_model(**inputs)
                scores = outputs.logits[:, 0]  # Take first column for relevance
        
        else:
            raise ValueError(f"Unknown reranker model format: {MODEL_NAME}")
        
        # Rank results by score
        ranked = sorted(
            zip(request.passages, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )[:request.top_k]
        
        # Format output
        results = [
            {
                "rank": i + 1,
                "text": text,
                "score": float(score)
            }
            for i, (text, score) in enumerate(ranked)
        ]
        
        return RerankResponse(results=results, model_name=MODEL_NAME)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reranking failed: {str(e)}")
    
    finally:
        # Move model back to CPU to free GPU memory
        reranker_model = reranker_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.get("/health")
def health_check():
    """Check if the API is running and model is loaded"""
    return {
        "status": "healthy",
        "model_loaded": reranker_model is not None,
        "model_name": MODEL_NAME,
        "cuda_available": torch.cuda.is_available()
    }

@app.get("/")
def root():
    """API information"""
    return {
        "service": "Reranker API",
        "model": MODEL_NAME,
        "endpoints": {
            "/rerank": "POST - Rerank passages",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    # Run on port 8001 to avoid conflict with main retrieval API (port 8000)
    uvicorn.run(
        "RerankerAPI:app",
        host="0.0.0.0",
        port=8001,
        reload=False  # Set to False for production, model reloading is slow
    )
