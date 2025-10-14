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
    """Map dense chunks to sparse chunks with improved logic"""
    equivalents = {}
    
    for dense_chunk in dense_chunks:
        dense_start = dense_chunk.get("start_index", 0)
        dense_end = dense_chunk.get("end_index", 0)
        dense_item = dense_chunk.get("item", "")
        dense_suffix = get_file_suffix(dense_chunk["file"])
        
        overlapping_sparse = []
        
        for sparse_chunk in sparse_chunks:
            # Must be same file and same item
            if get_file_suffix(sparse_chunk["file"]) != dense_suffix:
                continue
            if sparse_chunk.get("item", "") != dense_item:
                continue
                
            sparse_start = sparse_chunk.get("start_index", 0)
            sparse_end = sparse_chunk.get("end_index", 0)
            
            # Check for ANY overlap (not just 50%)
            if sparse_start < dense_end and sparse_end > dense_start:
                overlap_start = max(dense_start, sparse_start)
                overlap_end = min(dense_end, sparse_end)
                overlap_length = overlap_end - overlap_start
                
                # Calculate overlap percentage relative to BOTH chunks
                dense_size = dense_end - dense_start
                sparse_size = sparse_end - sparse_start
                
                if dense_size > 0 and sparse_size > 0:
                    # Overlap as percentage of each chunk
                    overlap_pct_dense = overlap_length / dense_size
                    overlap_pct_sparse = overlap_length / sparse_size
                    
                    # More lenient threshold: 30% of either chunk
                    # OR at least 100 characters/tokens of overlap
                    if (overlap_pct_dense >= 0.3 or overlap_pct_sparse >= 0.3) or overlap_length >= 100:
                        overlapping_sparse.append({
                            'chunk_id': sparse_chunk["chunk_id"],
                            'overlap_length': overlap_length,
                            'overlap_pct_dense': overlap_pct_dense,
                            'overlap_pct_sparse': overlap_pct_sparse
                        })
        
        # Sort by overlap length (prefer higher overlap)
        overlapping_sparse.sort(key=lambda x: x['overlap_length'], reverse=True)
        
        # Store the sparse chunk IDs (just the IDs, not the scores)
        equivalents[dense_chunk["chunk_id"]] = [s['chunk_id'] for s in overlapping_sparse]
        
        # If no overlaps found, try to find nearest sparse chunk
        if not overlapping_sparse:
            # Find sparse chunks in same item, ordered by distance
            same_item_sparse = []
            for sparse_chunk in sparse_chunks:
                if (get_file_suffix(sparse_chunk["file"]) == dense_suffix and 
                    sparse_chunk.get("item", "") == dense_item):
                    sparse_start = sparse_chunk.get("start_index", 0)
                    sparse_end = sparse_chunk.get("end_index", 0)
                    
                    # Calculate distance between chunks
                    if sparse_end <= dense_start:
                        distance = dense_start - sparse_end  # Sparse is before dense
                    elif sparse_start >= dense_end:
                        distance = sparse_start - dense_end  # Sparse is after dense
                    else:
                        distance = 0  # Should have been caught in overlap check
                    
                    same_item_sparse.append({
                        'chunk_id': sparse_chunk["chunk_id"],
                        'distance': distance
                    })
            
            # Sort by distance and take closest sparse chunk(s)
            if same_item_sparse:
                same_item_sparse.sort(key=lambda x: x['distance'])
                # Map to the closest sparse chunk as fallback
                equivalents[dense_chunk["chunk_id"]] = [same_item_sparse[0]['chunk_id']]
            else:
                # No sparse chunks in same item at all (rare edge case)
                equivalents[dense_chunk["chunk_id"]] = []
    
    return equivalents

# After mapping, print statistics
equivalents = map_chunks_by_index(dense_chunks, sparse_chunks)

# Statistics
total_dense = len(equivalents)
unmapped_dense = sum(1 for v in equivalents.values() if len(v) == 0)
single_mapped = sum(1 for v in equivalents.values() if len(v) == 1)
multi_mapped = sum(1 for v in equivalents.values() if len(v) > 1)

print(f"\nMapping Statistics:")
print(f"Total dense chunks: {total_dense}")
print(f"Unmapped dense chunks: {unmapped_dense} ({unmapped_dense/total_dense*100:.1f}%)")
print(f"Single-mapped: {single_mapped} ({single_mapped/total_dense*100:.1f}%)")
print(f"Multi-mapped: {multi_mapped} ({multi_mapped/total_dense*100:.1f}%)")

if unmapped_dense > 0:
    print(f"\nWarning: {unmapped_dense} dense chunks have no sparse equivalents")
    # Print first few unmapped for debugging
    unmapped_examples = [k for k, v in list(equivalents.items())[:10] if len(v) == 0]
    if unmapped_examples:
        print(f"Example unmapped dense chunks: {unmapped_examples[:3]}")

# Save mappings to JSON file
output_path = "chunk_equivalents_index.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(equivalents, f, indent=2, ensure_ascii=False)

print(f"\nMappings saved to: {output_path}")
print(f"Total mappings: {len(equivalents)}")

        