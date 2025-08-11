import os
import json
import time
import torch
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import csv
#os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

# <---------- CONFIGURATIONS ---------->
COLLECTION_NAME = "dense_miniLM"
EMBEDDED_DIR = "embedded_dense_allMini/10-K/AAPL"
EMBEDDING_DIM = 1024  # output dim for all-mini, have to change if we'll use smthn else (set to 1024 for bge-large-en-v1.5)
TOP_K = 5 # Number of documents to retrieve
QUERY = "What are the risks mentioned in Apple's 10-K filing?" # Hard-coded for now, have to change it later on

connections.connect(host="localhost", ports="19530")

def insert_dense_embedding_milvus():
    # schema creation, can add fields later on
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="ticker", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="form_type", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="start_index", dtype=DataType.INT64),
            FieldSchema(name="end_index", dtype=DataType.INT64)
        ]

        schema = CollectionSchema(fields, description="Dense MiniLM Embeddings")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        print(f"Created collection: {COLLECTION_NAME}")
        
        # create index immediately after collection creation
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name="dense_vector", index_params=index_params)
        print("Index created.")
        
        time.sleep(5)  # allow time for index creation
    else:
        collection = Collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")

    # load collection
    collection.load()
    print("Collection loaded.")

    # read embedded json files
    all_embeddings = []

    for filename in os.listdir(EMBEDDED_DIR):
    # if filename.endswith("_embedded.json"):
        file_path = os.path.join(EMBEDDED_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                # Load JSON data as list of dictionaries
                data = json.load(f)
                if isinstance(data, list):
                    all_embeddings.extend(data)
                else:
                    print(f"File {filename} is not a JSON array")
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {str(e)}")
                continue

    if not all_embeddings:
        print("No valid embedded data found to insert.")
        exit()

    # for inserting the data
    vectors = []
    source_files = []
    tickers = []
    form_types = []
    chunk_ids = []
    start_indices = []
    end_indices = []

    for entry in all_embeddings:
        # Ensure entry is a dictionary and has required keys
        if not isinstance(entry, dict):
            print(f"Non-dictionary entry: {entry}")
            continue
            
        if "embedding" not in entry:
            print(f"Missing 'embedding' field: {entry}")
            continue
            
        vectors.append(entry["embedding"])
        
        # set defaults
        metadata = entry.get("metadata", {})
        source_files.append(metadata.get("source_file", ""))
        tickers.append(metadata.get("ticker", ""))
        form_types.append(metadata.get("form_type", ""))
        chunk_ids.append(metadata.get("chunk_id", ""))
        start_indices.append(metadata.get("start_index", -1))
        end_indices.append(metadata.get("end_index", -1))

    # actual insertion of data
    # Insert as list of lists in FIELD ORDER (not including auto_id)
    collection.insert([
        vectors,           
        source_files,      
        tickers,           
        form_types, 
        chunk_ids,       
        start_indices,     
        end_indices        
    ])

    print(f"Inserted {len(vectors)} dense vectors into collection '{COLLECTION_NAME}'.")


def fetch_text(metadata):
    try:
        form_type = metadata.get("form_type", "")
        ticker = metadata.get("ticker", "")
        source_file = metadata.get("source_file", "")
        
        base_dir = "chunked_dense"

        form_dir = f"{form_type}" if form_type else ""

        chunk_dir = os.path.join(base_dir, form_dir, ticker.upper())
        chunk_file  = os.path.join(chunk_dir, source_file)

        with open(chunk_file, 'r', encoding="utf-8") as f:
            chunks = json.load(f)
        
        chunk_id = metadata.get("chunk_id", "")
        for chunk in chunks:
            if str(chunk.get("chunk_id", "")) == str(chunk_id):
                return chunk.get("content", "")
        
        return "Chunk not found"
    except FileNotFoundError:
        return f"File not found: {chunk_file}"
    except Exception as e:
        return f"Error: {str(e)}"


def retrieval_test():

    collection = Collection(COLLECTION_NAME)
    collection.load()
    print("Collection Loaded")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    query_embedding = model.encode(QUERY, convert_to_numpy=True)
    print("Query Embedded")

    search_params = {
        "metric_type": "L2", 
        "params": {"ef": 32}
    }

    results = collection.search(
        data = [query_embedding.tolist()],
        anns_field="dense_vector",
        param=search_params,
        limit=TOP_K,
        output_fields=["source_file", "ticker", "form_type", "chunk_id", "start_index", "end_index"]
    )

    if not results:
        print("No results found, exiting...")
        exit()
    
    print(f"\n Top {TOP_K} results for: '{QUERY}' \n")
    print("-" * 80)
    csv_data = []

    for i, hit in enumerate(results[0]):
        entity = hit.entity
        text_chunk = fetch_text({
            "form_type": entity.get("form_type", ""),
            "ticker": entity.get("ticker", ""),
            "source_file": entity.get("source_file", ""),
            "chunk_id": entity.get("chunk_id", "")
        })
        snippet = text_chunk[:200]
        print(f"Result #{i+1}: {text_chunk[:200]}...")

        csv_data.append({
            "Rank": i + 1,
            "Score": hit.distance,
            "Ticker": entity.get("ticker", ""),
            "Form Type": entity.get("form_type", ""),
            "Source File": entity.get("source_file", ""),
            "Chunk ID": entity.get("chunk_id", ""),
            "Start Index": entity.get("start_index", ""),
            "End Index": entity.get("end_index", ""),
            "Text Snippet": snippet
        })

        with open("retrieval_results.csv", "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Rank", "Score", "Ticker", "Form Type", "Source File", "Chunk ID", "Start Index", "End Index", "Text Snippet"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print("Results written to retrieval_results.csv")

insert_dense_embedding_milvus()
#retrieval_test()