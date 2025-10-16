import os
import json
import torch
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import traceback
from datetime import datetime
from pymilvus import MilvusClient, DataType, Function, FunctionType, Collection, AnnSearchRequest, RRFRanker, WeightedRanker, connections
from sentence_transformers import SentenceTransformer
import torch
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from queryExpansion import sparse_formatting, dense_formatting
from TickerFilter import get_tickers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    query: str
    reranked: list
    hybrid_raw_results: list
    fusion_method: str
    error: str | None = None

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", port="19530")

embedding_model = None

# Reranker API configuration
RERANKER_API_URL = "http://localhost:8001/rerank"  # Change if hosted elsewhere

def initialize_models():
    global embedding_model
    
    if embedding_model is None:
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embedding_model = embedding_model.to("cpu")

def normalize(arr):
    if len(arr) == 0 or arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def rrf(rank, k):
    return 1/(rank + k)

def milvus_hybrid_search(user_query: str, limit=50, use_rrf=True, alpha=0.3):
    
    COLLECTION_NAME = "sec_chunks_hybrid"
    initialize_models()
    global embedding_model
    
    sparse_query = sparse_formatting(user_query)
    dense_query = dense_formatting(user_query)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = embedding_model.to(device)
    
    try:
        query_text = f"Task: Financial retrieval. Query: {dense_query}"
        dense_query_vector = embedding_model.encode(
            query_text,
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
        embedding_model = embedding_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def rerank_results_api(query, passages, top_k=5):
    if not passages:
        print("No passages to rerank")
        return []
    
    try:
        # Call reranker API with longer timeout for large batches
        response = requests.post(
            RERANKER_API_URL,
            json={
                "query": query,
                "passages": passages,
                "top_k": top_k
            },
            timeout=120  # Increased to 120 seconds for 50 passages
        )
        
        if response.status_code != 200:
            print(f"Reranker API error: {response.status_code} - {response.text}")
            # Fallback: return passages in original order with dummy scores
            return [(p, 1.0 - i*0.1) for i, p in enumerate(passages[:top_k])]
        
        data = response.json()
        results = data.get("results", [])
        
        # Convert API response to expected format: [(text, score), ...]
        ranked = [(r["text"], r["score"]) for r in results]
        
        print(f"Reranker API returned {len(ranked)} results (model: {data.get('model_name', 'unknown')})")
        return ranked
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to reranker API: {e}")
        print("Fallback: returning passages in original order")
        # Fallback: return passages in original order with dummy scores
        return [(p, 1.0 - i*0.1) for i, p in enumerate(passages[:top_k])]

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
            "reranked": [],  # Changed from "reranked_results" to match QueryResponse schema
            "hybrid_raw_results": [],
            "fusion_method": "RRFRanker(k=30)",
            "error": "No results found"
        }
    
    print(f"Found {len(passages)} passages for reranking")
    
    # Call external reranker API instead of local model
    reranked = rerank_results_api(user_query, passages, top_k=5)
    
    # Map reranked results back to their metadata with deduplication
    reranked_with_metadata = []
    seen_chunks = set()  # Track which chunk_ids have been mapped
    
    for i, (text, score) in enumerate(reranked):
        # Find the matching metadata for this reranked text
        for j, passage in enumerate(passages):
            if passage == text:
                metadata = passage_metadata[j]
                chunk_id = metadata["chunk_id"]
                
                # Only add if we haven't seen this chunk_id before
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
        "reranked": reranked_with_metadata,  # Changed from "reranked_results" to match QueryResponse schema
        "hybrid_raw_results": [
            {
                "rank": i+1,
                "hybrid_score": result.get("distance", 0),
                "chunk_id": result.get("entity", {}).get("chunk_id", ""),
                "ticker": result.get("entity", {}).get("ticker", ""),
                "form_type": result.get("entity", {}).get("form_type", ""),
                "source_file": result.get("entity", {}).get("source_file", ""),
                "text": result.get("entity", {}).get("text", "")[:500] + "..."  #truncated
            }
            for i, result in enumerate(hybrid_results[0][:50]) 
        ],
        "error": None
    }
    
    save_results_to_file(output_data)
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
    print("=" * 80)
    print("Milvus Hybrid Retrieval API (with External Reranker)")
    print("=" * 80)
    print(f"Reranker API URL: {RERANKER_API_URL}")
    print("Make sure the reranker API is running on port 8001!")
    print("Start reranker: python RerankerAPI.py")
    print("=" * 80)
    uvicorn.run("MilvusHybridRetrieval_APIVersion:app", host="0.0.0.0", port=8000, reload=True)
