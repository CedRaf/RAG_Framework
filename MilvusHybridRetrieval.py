import os
import json
import re
from datetime import datetime
from pymilvus import MilvusClient, DataType, Function, FunctionType, Collection, AnnSearchRequest, RRFRanker, WeightedRanker, connections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from queryExpansion import sparse_formatting, dense_formatting
from TickerFilter import get_tickers

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", port="19530")

embedding_model = None
reranker_model = None
reranker_tokenizer = None

def initialize_models():
    global embedding_model, reranker_model, reranker_tokenizer
    
    if embedding_model is None:
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embedding_model = embedding_model.to("cpu")
    
    if reranker_model is None:
        reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        reranker_model.eval().to("cpu")

def milvus_hybrid_search(user_query: str, limit=50):
    
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
        
        dense_req = AnnSearchRequest(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=limit,
            expr=filter_expr
        )
        
        sparse_search_params = {
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.6,
                "bm25_b": 0.7
            }
        }
        
        sparse_req = AnnSearchRequest(
            data=[sparse_query],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr
        )
        
        # Use WeightedRanker for fusion (Dense: 40%, Sparse: 60%)
        # Can adjust these weights to bring up sparse results cuz w this config dense dominates
        # Can change this to RRFRanker() to experiment
        ranker = WeightedRanker(0.4, 0.6) 
        
        search_results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=limit,
            output_fields=["text", "chunk_id", "ticker", "form_type", "source_file"]
        )
        
        if not search_results or (len(search_results) > 0 and len(search_results[0]) == 0):
            return print("No results found in hybrid search!")

        formatted_results = [[
            {
                "id": hit.id,
                "distance": hit.distance,
                "entity": {
                    "text": hit.entity.get("text", ""),
                    "chunk_id": hit.entity.get("chunk_id", ""),
                    "ticker": hit.entity.get("ticker", ""),
                    "form_type": hit.entity.get("form_type", ""),
                    "source_file": hit.entity.get("source_file", "")
                }
            }
            for hit in search_results[0]
        ]]
        
        return formatted_results
        
    finally:
        embedding_model = embedding_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def rerank_results(query, passages, top_k=5):
    """Rerank results using BGE reranker"""
    global reranker_model, reranker_tokenizer
    
    if not passages:
        print("No passages to rerank")
        return []
    
    initialize_models()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_model = reranker_model.to(device)
    
    try:
        inputs = reranker_tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            scores = reranker_model(**inputs).logits.squeeze(-1)

        ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
        
    finally:
        reranker_model = reranker_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            "reranked_results": [],
            "hybrid_raw_results": [],
            "fusion_method": "milvus_built_in_hybrid_weighted",
            "error": "No results found"
        }
    
    print(f"Found {len(passages)} passages for reranking")
    
    reranked = rerank_results(user_query, passages, top_k=5)
    
    # Map reranked results back to their metadata
    reranked_with_metadata = []
    for i, (text, score) in enumerate(reranked):
        # Find the matching metadata for this reranked text
        for j, passage in enumerate(passages):
            if passage == text:
                metadata = passage_metadata[j]
                reranked_with_metadata.append({
                    "rank": i + 1,
                    "rerank_score": float(score),
                    "text": text,
                    "chunk_id": metadata["chunk_id"],
                    "ticker": metadata["ticker"],
                    "form_type": metadata["form_type"],
                    "source_file": metadata["source_file"],
                    "hybrid_score": metadata["hybrid_score"]
                })
                break
    
    output_data = {
        "query": user_query,
        "reranked_results": reranked_with_metadata,
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
        "fusion_method": "milvus_built_in_hybrid_weighted"
    }
    
    save_results_to_file(output_data)
    return output_data

def save_results_to_file(output_data):
    os.makedirs("retrieval_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("retrieval_outputs", f"milvus_hybrid_results_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Milvus hybrid results saved to {output_path}")

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    results = milvus_hybrid_retrieval_test(user_query)
    
    print("\nMILVUS HYBRID SEARCH RESULTS")
    for result in results["reranked_results"]:
        print(f"Rank {result['rank']} - Rerank Score: {result['rerank_score']:.4f}")
        print(f"Ticker: {result['ticker']} | Form: {result['form_type']} | Source: {result['source_file']}")
        print(result['text'][:300] + "...")
        print("-" * 60)