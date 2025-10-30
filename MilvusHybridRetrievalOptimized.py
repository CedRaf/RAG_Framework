import os
import json
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import traceback
from datetime import datetime
from pymilvus import MilvusClient, DataType, Function, FunctionType, Collection, AnnSearchRequest, RRFRanker, WeightedRanker, connections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from queryExpansion import sparse_formatting, dense_formatting
from TickerFilter import get_tickers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import requests
import time

# Use orjson for faster JSON (pip install orjson)
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    print("orjson not installed, using standard json (slower)")

from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
HF_KEY = os.getenv("HF_KEY")

# Global variables for models
embedding_model = None
reranker_model = None
reranker_tokenizer = None
device = None

# HTTP Session for connection pooling (IMPORTANT!)
http_session = requests.Session()
http_session.headers.update({'Connection': 'keep-alive'})

def initialize_models():
    global embedding_model, reranker_model, reranker_tokenizer, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on {device}")
    
    # Load in FP16/half precision to save memory
    if embedding_model is None:
        login(HF_KEY)  # Comment this line if Qwen embedding
        embedding_model = SentenceTransformer("google/embeddinggemma-300m")
        # embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embedding_model = embedding_model.half().to(device)  # FP16
        print("Embedding model loaded to GPU (FP16)")

    # Reranker initialization commented out since we're using API
    # if reranker_model is None:
    #     reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    #     reranker_model = AutoModelForSequenceClassification.from_pretrained(
    #         "BAAI/bge-reranker-v2-m3",
    #         torch_dtype=torch.float16
    #     )
    #     reranker_model = reranker_model.eval().to(device)
    #     print("Reranker model loaded to GPU (FP16)")

def cleanup_models():
    global embedding_model, reranker_model, http_session
    if embedding_model is not None:
        embedding_model = embedding_model.to("cpu")
    if reranker_model is not None:
        reranker_model = reranker_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Close HTTP session
    http_session.close()
    print("Models moved to CPU and cache cleared")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_models()
    yield
    # Shutdown
    cleanup_models()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    query: str
    reranked: list
    fusion_method: str
    error: str | None = None

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", port="19530")

def normalize(arr):
    if len(arr) == 0 or arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def rrf(rank, k):
    return 1/(rank + k)

def milvus_hybrid_search(user_query: str, limit=50, use_rrf=True, alpha=0.3):
    COLLECTION_NAME = "sec_chunks_hybrid"
    global embedding_model, device
    
    sparse_query = sparse_formatting(user_query)
    dense_query = dense_formatting(user_query)
    
    try:
        # GEMMA embedding
        dense_query_vector = embedding_model.encode(
            dense_query,
            prompt_name="Retrieval-query",
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        ).tolist()
        
        # Ticker filter
        detected_tickers = get_tickers(user_query)
        filter_expr = None
        if detected_tickers:
            ticker_filter = ",".join([f'"{t}"' for t in detected_tickers])
            filter_expr = f'ticker in [{ticker_filter}]'
        
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        dense_search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 256}
        }
        
        sparse_search_params = {
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.6,
                "bm25_b": 0.4
            }
        }

        dense_results = collection.search(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["text", "chunk_id", "ticker", "form_type", "source_file"]
        )

        sparse_results = collection.search(
            data=[sparse_query],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["text", "chunk_id", "ticker", "form_type", "source_file"],
        )

        if not dense_results or not sparse_results:
            print("No results found in hybrid search!")
            return []
        
        dense_scores = np.array([hit.distance for hit in dense_results[0]])
        sparse_scores = np.array([hit.distance for hit in sparse_results[0]])

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        final_scores = {}

        if use_rrf:
            for rank, hit in enumerate(dense_results[0]):
                final_scores[hit.id] = final_scores.get(hit.id, 0) + rrf(rank, 30)
            
            for rank, hit in enumerate(sparse_results[0]):
                final_scores[hit.id] = final_scores.get(hit.id, 0) + rrf(rank, 30)
        else:
            for hit, score in zip(dense_results[0], dense_norm):
                final_scores[hit.id] = alpha * score

            for hit, score in zip(sparse_results[0], sparse_norm):
                final_scores[hit.id] = final_scores.get(hit.id, 0) + (1 - alpha) * score

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [r[0] for r in ranked[:limit]] 

        all_hits = {hit.id: hit for hit in dense_results[0]}  
        all_hits.update({hit.id: hit for hit in sparse_results[0]})

        formatted_results = [[
            {
                "id": hit_id,
                "distance": final_scores[hit_id],
                "entity": {
                    "text": all_hits[hit_id].entity.get("text", ""),
                    "chunk_id": all_hits[hit_id].entity.get("chunk_id", ""),
                    "ticker": all_hits[hit_id].entity.get("ticker", ""),
                    "form_type": all_hits[hit_id].entity.get("form_type", ""),
                    "source_file": all_hits[hit_id].entity.get("source_file", "")
                }
            }
            for hit_id in top_ids
        ]]
        
        return formatted_results
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def call_reranker_api(query: str, passages: list[str], top_k: int = 5):
    """Optimized API call with connection pooling and fast JSON"""
    global http_session
    
    reranker_api_url = "http://localhost:8500/rerank"
    payload = {
        "query": query,
        "passages": passages,
        "top_k": top_k
    }
    
    try:
        start_time = time.time()
        
        if USE_ORJSON:
            # Use orjson for faster JSON encoding
            payload_json = orjson.dumps(payload)
            response = http_session.post(
                reranker_api_url,
                data=payload_json,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            result = orjson.loads(response.content)
        else:
            # Standard requests with session (still uses connection pooling)
            response = http_session.post(reranker_api_url, json=payload)
            response.raise_for_status()
            result = response.json()
        
        elapsed = time.time() - start_time
        print(f"Reranker API call took: {elapsed:.3f}s")
        
        reranked_items = result["reranked"]
        # Convert to expected format: list of (text, score) tuples
        return [(item["text"], item["score"]) for item in reranked_items]
        
    except requests.exceptions.RequestException as e:
        print(f"Reranker API call failed: {e}")
        return []

def milvus_hybrid_retrieval_test(user_query: str):
    hybrid_results = milvus_hybrid_search(user_query, limit=50)
    
    passages = []
    passage_metadata = []

    for result in hybrid_results[0]:
        text = result.get("entity", {}).get("text", "")
        if text:
            passages.append(text)
            passage_metadata.append({
                "chunk_id": result.get("entity", {}).get("chunk_id", ""),
                "ticker": result.get("entity", {}).get("ticker", ""),
                "form_type": result.get("entity", {}).get("form_type", ""),
                "source_file": result.get("entity", {}).get("source_file", ""),
                "hybrid_score": result.get("distance", 0)
            })
    
    if not passages:
        print("No results found from hybrid search!")
        return {
            "query": user_query,
            "reranked": [],
            "fusion_method": "RRFRanker(k=30)",
            "error": "No results found"
        }
    
    print(f"Found {len(passages)} passages for reranking")
    
    # Call optimized reranker API
    reranked = call_reranker_api(user_query, passages, top_k=5)
    
    if not reranked:
        return {
            "query": user_query,
            "reranked": [],
            "fusion_method": "RRFRanker(k=30)",
            "error": "Reranking failed"
        }

    # Map reranked results back to their metadata with deduplication
    reranked_with_metadata = []
    seen_chunks = set()
    
    for i, (text, score) in enumerate(reranked):
        for j, passage in enumerate(passages):
            if passage == text:
                metadata = passage_metadata[j]
                chunk_id = metadata["chunk_id"]
                
                if chunk_id not in seen_chunks:
                    reranked_with_metadata.append({
                        "rank": i + 1,
                        "rerank_score": float(score),
                        "text": text,
                        "chunk_id": chunk_id,
                        "ticker": metadata["ticker"],
                        "form_type": metadata["form_type"],
                        "source_file": metadata["source_file"],
                        "hybrid_score": metadata["hybrid_score"]
                    })
                    seen_chunks.add(chunk_id)
                    break
    
    output_data = {
        "query": user_query,
        "fusion_method": "RRFRanker(k=30)",
        "reranked": reranked_with_metadata,
    }
    
    return output_data

def save_results_to_file(output_data):
    os.makedirs("hybrid_retrieval_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("hybrid_retrieval_outputs", f"milvus_hybrid_results_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Milvus hybrid results saved to {output_path}")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """API endpoint for Milvus hybrid retrieval"""
    try:
        print(f"Received query: {request.user_query}")
        results = milvus_hybrid_retrieval_test(request.user_query)
        return QueryResponse(**results)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("MilvusHybridRetrievalOptimized:app", host="0.0.0.0", port=8000, reload=True)