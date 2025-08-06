import os
import json
from pymilvus import MilvusClient, DataType, Function, FunctionType
from tqdm import tqdm
import csv
#os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")
COLLECTION_NAME = "sec_chunks_sparse"
CHUNK_BASE_DIR = "chunked_sparse"
MAX_TEXT_LENGTH = 10000

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# --- DEFINE SCHEMA WITH METADATA ---
if COLLECTION_NAME not in client.list_collections():
    schema = client.create_schema()

    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LENGTH, enable_analyzer=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)



    # Metadata fields
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_field("start_index", DataType.INT64)
    schema.add_field("end_index", DataType.INT64)
    schema.add_field("ticker", DataType.VARCHAR, max_length=100)
    schema.add_field("form_type", DataType.VARCHAR, max_length=16)
    schema.add_field("source_file", DataType.VARCHAR, max_length=256)

    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
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

# --- LOAD CHUNKS FROM JSON FILES ---
def load_chunks_from_folder(base_path):
    docs = []
    for form_type in os.listdir(base_path):
        form_path = os.path.join(base_path, form_type)
        if not os.path.isdir(form_path): continue

        for ticker in sorted(os.listdir(form_path)):
            ticker_path = os.path.join(form_path, ticker)
            if not os.path.isdir(ticker_path): continue

            for file in os.listdir(ticker_path):
                if file.endswith(".json"):
                    file_path = os.path.join(ticker_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            chunks = json.load(f)
                            for chunk in chunks:
                                content = chunk.get("content", "").strip()
                                if content:
                                    docs.append({
                                        "text": content,
                                        "chunk_id": chunk.get("chunk_id"),
                                        "start_index": chunk.get("start_index"),
                                        "end_index": chunk.get("end_index"),
                                        "ticker": chunk.get("ticker", ticker),
                                        "form_type": chunk.get("form_type", form_type),
                                        "source_file": chunk.get("source_file", file)
                                    })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    return docs


docs = load_chunks_from_folder(CHUNK_BASE_DIR)
print(f"Loaded {len(docs)} chunks.")

if docs:
    print(f"Inserting into '{COLLECTION_NAME}'...")
    for i in tqdm(range(0, len(docs), 1000)):
        batch = docs[i:i+1000]
        client.insert(COLLECTION_NAME, batch)

    print("Insertion complete.")
else:
    print(" No documents found to insert.")



# client.load_collection(COLLECTION_NAME)

# results = client.query(
#     collection_name=COLLECTION_NAME,
#     output_fields=["id", "ticker", "chunk_id", "form_type", "source_file", "text"],
#     limit=1000  # make sure this is larger than possible # of entries
# )
# print("Number of records found:", len(results))
# # Save to CSV
# csv_file = "milvus_results.csv"
# with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.DictWriter(file, fieldnames=["id", "ticker", "chunk_id", "form_type", "source_file", "text" ])
#     writer.writeheader()
#     writer.writerows(results)

# print(f" Results saved to {csv_file}")




# schema = client.describe_collection(COLLECTION_NAME)
# print(schema)




