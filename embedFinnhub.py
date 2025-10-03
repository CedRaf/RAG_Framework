import os
import json
from pymilvus import MilvusClient, DataType, Function, FunctionType
from tqdm import tqdm
import numpy as np
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")



# Directory with 1 JSON per ticker
NEWS_DIR = "./finnhub_news/"
COLLECTION_NAME = "news_articles"
EMBEDDING_DIM = 1024  

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
# --- DEFINE SCHEMA WITH METADATA ---
if COLLECTION_NAME not in client.list_collections():
    schema = client.create_schema()

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("content", DataType.VARCHAR, max_length=4096, enable_analyzer=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # Metadata fields
    schema.add_field("news_id", DataType.INT64)       
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("datetime", DataType.INT64)
    schema.add_field("ticker", DataType.VARCHAR, max_length=32)

    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["content"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }
    )

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params
    )
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")



def load_news_from_folder(base_path):
    docs = []
    for file in os.listdir(base_path):
        if file.endswith(".json"):
            file_path = os.path.join(base_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    news_items = json.load(f)
                    for item in news_items:
                        headline = item.get("headline", "").strip()
                        summary = item.get("summary", "").strip()
                        content = (headline + " " + summary).strip()

                        if content:
                            docs.append({
                                "content": content,
                                "news_id": item.get("id"),
                                "category": item.get("category"),
                                "datetime": item.get("datetime"),
                                "ticker": item.get("related")
                            })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return docs


docs = load_news_from_folder(NEWS_DIR)
print(f"Loaded {len(docs)} news articles.")

if docs:
    print(f"Inserting into '{COLLECTION_NAME}'...")
    for i in tqdm(range(0, len(docs), 1000)):
        batch = docs[i:i+1000]
        client.insert(COLLECTION_NAME, batch)

    print("Insertion complete.")
else:
    print("No news articles found to insert.")




