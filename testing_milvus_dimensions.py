from pymilvus import connections, Collection

# Connect to your Milvus instance (adjust host/port if needed)
connections.connect(host="localhost", port="19530")

# Name of your collection
collection_name = "sec_chunks_hybrid"

# Load the collection
collection = Collection(collection_name)

# Print schema for all fields
print("Collection fields and types:")
for field in collection.schema.fields:
    print(f"Field: {field.name}, Type: {field.dtype}, Params: {field.params}")

# Specifically print the dimension of the dense_vector field
for field in collection.schema.fields:
    if field.name == "dense_vector":
        print(f"\nDense vector dimension: {field.params.get('dim')}")