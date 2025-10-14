from pymilvus import connections, MilvusClient

# Collection names sec_chunks_dense, sec_chunks_sparse, sec_chunks_hybrid
COLLECTION_NAME = "sec_chunks_hybrid"
connections.disconnect("default")

connections.connect("default", host="localhost", port="19530")

client = MilvusClient("http://localhost:19530", token="root:Milvus")


if client.has_collection(COLLECTION_NAME):
    print(f"Dropping collection '{COLLECTION_NAME}'...")
    client.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")
