from pymilvus import connections, MilvusClient

# Collection names sec_chunks_dense and sec_chunks_sparse
COLLECTION_NAME = "sec_chunks_dense"
connections.disconnect("default")

connections.connect("default", host="localhost", port="19530")

client = MilvusClient("http://localhost:19530", token="root:Milvus")


if client.has_collection(COLLECTION_NAME):
    print(f"Dropping collection '{COLLECTION_NAME}'...")
    client.drop_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' dropped.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")
