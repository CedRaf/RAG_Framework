from pymilvus import connections, MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "sec_chunks_dense"
connections.disconnect("default")

connections.connect("default", host="localhost", port="19530")

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
client = MilvusClient(MILVUS_URI, token=MILVUS_TOKEN)


if client.has_collection(COLLECTION_NAME):
    print(f"Dropping collection '{COLLECTION_NAME}'...")
    client.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")
