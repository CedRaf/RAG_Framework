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
EMBEDDED_DIR = "embedded_dense_allMini/10-K/AAPL" # This is still hard coded, have to change
EMBEDDING_DIM = 1024  # output dim for all-mini, have to change if we'll use smthn else (set to 1024 for bge-large-en-v1.5)

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

insert_dense_embedding_milvus()
