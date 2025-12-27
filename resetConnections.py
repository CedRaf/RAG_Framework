from pymilvus import connections, MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "sec_chunks_dense"
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
connections.disconnect("default")

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
client = MilvusClient(MILVUS_URI, token=MILVUS_TOKEN)


if client.has_collection(COLLECTION_NAME):
    print(f"Dropping collection '{COLLECTION_NAME}'...")
    client.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")
