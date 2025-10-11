import os
import json
import re
from datetime import datetime
from pymilvus import MilvusClient, DataType, Function, FunctionType
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from queryExpansion import expand_query, sparse_formatting, dense_formatting
from TickerFilter import get_tickers

# Initialize client and models
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# Global models
embedding_model = None
reranker_model = None
reranker_tokenizer = None

def initialize_models():
    """Initialize models once"""
    global embedding_model, reranker_model, reranker_tokenizer
    
    if embedding_model is None:
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embedding_model = embedding_model.to("cpu")
    
    if reranker_model is None:
        reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        reranker_model.eval().to("cpu")

def create_hybrid_collection():
    """Create a hybrid collection with both dense and sparse vectors"""
    
    COLLECTION_NAME = "sec_chunks_hybrid"
    EMBEDDING_DIM = 1024
    MAX_TEXT_LENGTH = 10000
    
    # Drop existing collection if it exists
    if COLLECTION_NAME in client.list_collections():
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")
    
    # Create schema for hybrid collection
    schema = client.create_schema()
    
    # Primary key
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    
    # Text field for sparse vector generation
    schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LENGTH, enable_analyzer=True)
    
    # Dense vector field
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    
    # Sparse vector field
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    
    # Metadata fields
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=1000)
    schema.add_field("ticker", DataType.VARCHAR, max_length=32)
    schema.add_field("form_type", DataType.VARCHAR, max_length=16)
    schema.add_field("source_file", DataType.VARCHAR, max_length=512)
    schema.add_field("start_index", DataType.INT64)
    schema.add_field("end_index", DataType.INT64)
    
    # BM25 function for automatic sparse vector generation
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    
    # Create indexes
    index_params = client.prepare_index_params()
    
    # Dense vector index (HNSW)
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 512}
    )
    
    # Sparse vector index (BM25)
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.6,
            "bm25_b": 0.7
        }
    )
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    
    print(f"Created hybrid collection: {COLLECTION_NAME}")
    return COLLECTION_NAME

def insert_hybrid_data():
    """Insert data into hybrid collection from existing chunked data"""
    
    COLLECTION_NAME = "sec_chunks_hybrid"
    initialize_models()
    
    # Load dense embeddings data
    dense_embeddings = []
    dense_dir = "dense_embeddings"
    
    for root, dirs, files in os.walk(dense_dir):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            dense_embeddings.extend(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding {file_path}: {str(e)}")
    
    print(f"Loaded {len(dense_embeddings)} dense embeddings")
    
    # Load sparse data (chunked text)
    sparse_chunks = {}  # chunk_id -> chunk_data
    sparse_dir = "chunked_sparse"
    
    for form_type in os.listdir(sparse_dir):
        form_path = os.path.join(sparse_dir, form_type)
        if not os.path.isdir(form_path):
            continue
            
        for ticker in os.listdir(form_path):
            ticker_path = os.path.join(form_path, ticker)
            if not os.path.isdir(ticker_path):
                continue
                
            for file in os.listdir(ticker_path):
                if file.endswith(".json"):
                    file_path = os.path.join(ticker_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            chunks = json.load(f)
                            for chunk in chunks:
                                chunk_id = chunk.get("chunk_id")
                                if chunk_id:
                                    sparse_chunks[chunk_id] = chunk
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    print(f"Loaded {len(sparse_chunks)} sparse chunks")
    
    # Combine data for hybrid insertion
    hybrid_data = []
    
    for dense_entry in dense_embeddings:
        if not isinstance(dense_entry, dict) or "embedding" not in dense_entry:
            continue
            
        metadata = dense_entry.get("metadata", {})
        chunk_id = metadata.get("chunk_id", "")
        
        # Find corresponding sparse chunk
        sparse_chunk = sparse_chunks.get(chunk_id)
        if not sparse_chunk:
            continue
            
        # Prepare hybrid data entry
        hybrid_entry = {
            "text": sparse_chunk.get("content", ""),
            "dense_vector": dense_entry["embedding"],
            "chunk_id": chunk_id,
            "ticker": metadata.get("ticker", ""),
            "form_type": metadata.get("form_type", ""),
            "source_file": metadata.get("source_file", ""),
            "start_index": metadata.get("start_index", -1),
            "end_index": metadata.get("end_index", -1)
        }
        
        hybrid_data.append(hybrid_entry)
    
    print(f"Prepared {len(hybrid_data)} hybrid entries")
    
    # Insert data in batches
    if hybrid_data:
        batch_size = 1000
        for i in range(0, len(hybrid_data), batch_size):
            batch = hybrid_data[i:i+batch_size]
            client.insert(COLLECTION_NAME, batch)
            print(f"Inserted batch {i//batch_size + 1}/{(len(hybrid_data)-1)//batch_size + 1}")
        
        print("Hybrid data insertion complete!")
    else:
        print("No hybrid data to insert")

def milvus_hybrid_search(user_query: str, limit=50):
    """Perform Milvus built-in hybrid search"""
    
    COLLECTION_NAME = "sec_chunks_hybrid"
    initialize_models()
    
    # Generate query vectors
    pseudo_doc = expand_query(user_query)
    sparse_query = sparse_formatting(user_query, pseudo_doc)
    dense_query = dense_formatting(user_query, pseudo_doc)
    
    # Generate dense query vector
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
        
        # Ticker filtering
        detected_tickers = get_tickers(user_query)
        expr = None
        if detected_tickers:
            ticker_filter = ",".join([f'"{t}"' for t in detected_tickers])
            expr = f'ticker in [{ticker_filter}]'
        
        # Perform hybrid search
        search_results = client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[
                # Dense vector search request
                {
                    "data": [dense_query_vector],
                    "anns_field": "dense_vector",
                    "param": {
                        "metric_type": "COSINE",
                        "params": {"ef": 256}
                    },
                    "limit": limit,
                    "expr": expr
                },
                # Sparse vector search request
                {
                    "data": [sparse_query],
                    "anns_field": "sparse_vector",
                    "param": {
                        "metric_type": "BM25",
                        "params": {
                            "inverted_index_algo": "DAAT_MAXSCORE",
                            "bm25_k1": 1.6,
                            "bm25_b": 0.7
                        }
                    },
                    "limit": limit,
                    "expr": expr
                }
            ],
            # Fusion parameters
            rerank={
                "strategy": "weighted",  # or "rrf" (Reciprocal Rank Fusion)
                "params": {
                    "weights": [0.4, 0.6]  # Dense: 40%, Sparse: 60%
                }
            },
            limit=limit,
            output_fields=["text", "chunk_id", "ticker", "form_type", "source_file"]
        )
        
        return search_results
        
    finally:
        embedding_model = embedding_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def rerank_results(query, passages, top_k=5):
    """Rerank results using BGE reranker"""
    global reranker_model, reranker_tokenizer
    
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
    """Main hybrid retrieval function using Milvus built-in hybrid search"""
    
    # Perform hybrid search
    hybrid_results = milvus_hybrid_search(user_query, limit=50)
    
    # Extract passages for reranking
    passages = []
    for result in hybrid_results[0]:  # hybrid_search returns list of lists
        text = result.get("entity", {}).get("text", "")
        if text:
            passages.append(text)
    
    # Rerank results
    reranked = rerank_results(user_query, passages, top_k=5)
    
    # Prepare output data
    output_data = {
        "query": user_query,
        "reranked_results": [
            {
                "rank": i+1,
                "score": float(score),
                "text": text
            }
            for i, (text, score) in enumerate(reranked)
        ],
        "hybrid_raw_results": [
            {
                "rank": i+1,
                "hybrid_score": result.get("distance", 0),
                "chunk_id": result.get("entity", {}).get("chunk_id", ""),
                "ticker": result.get("entity", {}).get("ticker", ""),
                "form_type": result.get("entity", {}).get("form_type", ""),
                "source_file": result.get("entity", {}).get("source_file", ""),
                "text": result.get("entity", {}).get("text", "")[:500] + "..."  # Truncate for readability
            }
            for i, result in enumerate(hybrid_results[0][:10])
        ],
        "fusion_method": "milvus_built_in_hybrid_weighted"
    }
    
    # Save results
    save_results_to_file(output_data)
    return output_data

def save_results_to_file(output_data):
    """Save results to JSON file"""
    os.makedirs("retrieval_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("retrieval_outputs", f"milvus_hybrid_results_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Milvus hybrid results saved to {output_path}")

def setup_milvus_hybrid():
    """Complete setup for Milvus hybrid search"""
    print("Setting up Milvus hybrid collection...")
    
    # Step 1: Create hybrid collection
    collection_name = create_hybrid_collection()
    
    # Step 2: Insert hybrid data
    print("Inserting hybrid data...")
    insert_hybrid_data()
    
    print("Milvus hybrid setup complete!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        # Setup mode: create collection and insert data
        setup_milvus_hybrid()
    else:
        # Query mode: perform hybrid search
        user_query = input("Enter your query: ")
        results = milvus_hybrid_retrieval_test(user_query)
        
        print("\n=== MILVUS HYBRID SEARCH RESULTS ===")
        for result in results["reranked_results"]:
            print(f"Rank {result['rank']} - Score: {result['score']:.4f}")
            print(result['text'][:300] + "...")
            print("-" * 60)