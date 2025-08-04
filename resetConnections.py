from pymilvus import MilvusClient

# Configuration
COLLECTION_NAME = "dense_miniLM"

# Connect to Milvus
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# Delete if it exists
if COLLECTION_NAME in client.list_collections():
    print(f"Dropping collection '{COLLECTION_NAME}'...")
    client.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")