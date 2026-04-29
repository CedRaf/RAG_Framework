import os
import json
from pymilvus import MilvusClient, DataType, Function, FunctionType, connections
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
HF_KEY = os.getenv("HF_KEY")

def initialize_models():
    global embedding_model, reranker_model, reranker_tokenizer
    if 'embedding_model' not in globals():
        #globals()['embedding_model'] = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B").to("cpu")
        login(HF_KEY)
        globals()['embedding_model'] = SentenceTransformer("google/embeddinggemma-300m").to("cpu")


    if 'reranker_model' not in globals():
        globals()['reranker_tokenizer'] = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        globals()['reranker_model'] = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").eval().to("cpu")

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", port="19530")

COLLECTION_NAME = "sec_chunks_hybrid"
EMBEDDING_DIM = 768
MAX_TEXT_LENGTH = 10000

def create_hybrid_collection():
    if COLLECTION_NAME in client.list_collections():
        client.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")
    schema = client.create_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LENGTH, enable_analyzer=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=1000)
    schema.add_field("ticker", DataType.VARCHAR, max_length=32)
    schema.add_field("form_type", DataType.VARCHAR, max_length=16)
    schema.add_field("source_file", DataType.VARCHAR, max_length=512)
    schema.add_field("start_index", DataType.INT64)
    schema.add_field("end_index", DataType.INT64)
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 512}
    )
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
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
    print(f"Created hybrid collection: {COLLECTION_NAME}")
    return COLLECTION_NAME

def insert_hybrid_data():
    initialize_models()
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
    sparse_chunks = []
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
                            if isinstance(chunks, list):
                                sparse_chunks.extend(chunks)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    print(f"Loaded {len(sparse_chunks)} sparse chunks")
    hybrid_data = []
    print("Processing dense embeddings...")
    for dense_entry in dense_embeddings:
        if not isinstance(dense_entry, dict) or "embedding" not in dense_entry:
            continue
        metadata = dense_entry.get("metadata", {})
        
        # Get text content from the dense entry
        text_content = dense_entry.get("text", "")
        
        # Skip if no text content (shouldn't happen with updated embeddings)
        if not text_content or len(text_content.strip()) == 0:
            print(f"Warning: Dense entry missing text: {metadata.get('chunk_id', 'unknown')}")
            continue
        
        hybrid_entry = {
            "text": text_content,
            "dense_vector": dense_entry["embedding"],
            "chunk_id": metadata.get("chunk_id", ""),
            "ticker": metadata.get("ticker", ""),
            "form_type": metadata.get("form_type", ""),
            "source_file": metadata.get("source_file", ""),
            "start_index": metadata.get("start_index", -1),
            "end_index": metadata.get("end_index", -1)
        }
        hybrid_data.append(hybrid_entry)
    print(f"Added {len(hybrid_data)} dense entries")
    zero_vector = [0.0] * EMBEDDING_DIM
    sparse_added = 0
    print("Processing sparse chunks...")
    for sparse_chunk in sparse_chunks:
        if not isinstance(sparse_chunk, dict):
            continue
        hybrid_entry = {
            "text": sparse_chunk.get("content", ""),
            "dense_vector": zero_vector,
            "chunk_id": sparse_chunk.get("chunk_id", ""),
            "ticker": sparse_chunk.get("ticker", ""),
            "form_type": sparse_chunk.get("form_type", ""),
            "source_file": sparse_chunk.get("source_file", ""),
            "start_index": sparse_chunk.get("start_index", -1),
            "end_index": sparse_chunk.get("end_index", -1)
        }
        hybrid_data.append(hybrid_entry)
        sparse_added += 1
    print(f"Added {sparse_added} sparse entries")
    print(f"Total prepared: {len(hybrid_data)} hybrid entries")
    if hybrid_data:
        batch_size = 1000
        for i in range(0, len(hybrid_data), batch_size):
            batch = hybrid_data[i:i+batch_size]
            client.insert(COLLECTION_NAME, batch)
            print(f"Inserted batch {i//batch_size + 1}/{(len(hybrid_data)-1)//batch_size + 1}")
        print("Hybrid data insertion complete!")
    else:
        print("No hybrid data to insert")

def setup_milvus_hybrid():
    print("Setting up Milvus hybrid collection...")
    create_hybrid_collection()
    print("Inserting hybrid data...")
    insert_hybrid_data()
    print("Milvus hybrid setup complete!")

if __name__ == "__main__":
    setup_milvus_hybrid()
