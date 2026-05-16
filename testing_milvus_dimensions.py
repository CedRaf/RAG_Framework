from pymilvus import connections, MilvusClient

COLLECTION_NAME = "sec_chunks_hybrid"

connections.disconnect("default")
connections.connect("default", host="localhost", port="19530")

client = MilvusClient("http://localhost:19530", token="root:Milvus")

# Check if collection exists
if client.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' exists.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")