import os
import json

dense_base = "chunked_dense"
sparse_base = "chunked_sparse"
output_file = "chunk_equivalents.json"

def get_all_chunks(base_folder):
    chunks = []
    for form_type in os.listdir(base_folder):
        form_path = os.path.join(base_folder, form_type)
        if not os.path.isdir(form_path):
            continue
        for ticker in os.listdir(form_path):
            ticker_path = os.path.join(form_path, ticker)
            if not os.path.isdir(ticker_path):
                continue
            for file in os.listdir(ticker_path):
                if file.endswith(".json"):
                    file_path = os.path.join(ticker_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        chunk_list = json.load(f)
                        for chunk in chunk_list:
                            chunks.append({
                                "chunk_id": chunk["chunk_id"],
                                "content": chunk["content"],
                                "file": file,
                                "file_path": file_path,
                                "start_index": chunk.get("start_index", 0),
                                "end_index": chunk.get("end_index", 0),
                                "item": chunk.get("item", "")
                            })
    return chunks

def get_file_suffix(filename):
    parts = filename.split('_', 1)
    return parts[1] if len(parts) > 1 else filename

print("Loading dense and sparse chunks...")
dense_chunks = get_all_chunks(dense_base)
sparse_chunks = get_all_chunks(sparse_base)

print(f"Dense chunks: {len(dense_chunks)}, Sparse chunks: {len(sparse_chunks)}")


# Many-to-many mapping: dense_chunk_id -> [sparse_chunk_id, ...]
equivalents = {}

print("Mapping dense to sparse chunks using index-based overlap...")

def map_chunks_by_index(dense_chunks, sparse_chunks):
    equivalents = {}
    
    for dense_chunk in dense_chunks:
        dense_start = dense_chunk["start_index"]
        dense_end = dense_chunk["end_index"]
        dense_suffix = get_file_suffix(dense_chunk["file"])
        dense_item = dense_chunk["item"]
        
        # Find overlapping sparse chunks from same file AND same item
        overlapping_sparse = []
        for sparse_chunk in sparse_chunks:
            # Must be same file AND same item (since indexes reset per item)
            if (get_file_suffix(sparse_chunk["file"]) != dense_suffix or 
                sparse_chunk["item"] != dense_item):
                continue
                
            sparse_start = sparse_chunk["start_index"]
            sparse_end = sparse_chunk["end_index"]
            
            # Check for overlap
            if sparse_start < dense_end and sparse_end > dense_start:
                overlap_start = max(dense_start, sparse_start)
                overlap_end = min(dense_end, sparse_end)
                overlap_length = overlap_end - overlap_start
                
                # Only include if significant overlap (>50% of smaller chunk)
                dense_length = dense_end - dense_start
                sparse_length = sparse_end - sparse_start
                min_chunk_size = min(dense_length, sparse_length)
                
                if overlap_length > 0.5 * min_chunk_size:
                    overlapping_sparse.append(sparse_chunk["chunk_id"])
        
        equivalents[dense_chunk["chunk_id"]] = overlapping_sparse
    
    return equivalents

equivalents = map_chunks_by_index(dense_chunks, sparse_chunks)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(equivalents, f, indent=2, ensure_ascii=False)

print(f"Saved chunk equivalents to {output_file}")