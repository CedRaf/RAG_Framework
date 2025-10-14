# filepath: c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG\ChunkEquivalentSameSize.py
import os
import json
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

dense_base = "chunked_dense"
sparse_base = "chunked_sparse"
output_file = "chunk_equivalents_index.json"

def get_all_chunks(base_folder):
    """Load all chunks from the specified folder"""
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
                                "file": file
                            })
    return chunks

def get_chunk_number(chunk_id):
    """Extract chunk number from chunk_id (e.g., 'dense-1-file.json' -> 1)"""
    parts = chunk_id.split('-')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def get_file_suffix(chunk_id):
    """Extract file suffix from chunk_id (everything after 'dense-N-' or 'sparse-N-')"""
    parts = chunk_id.split('-', 2)
    return parts[2] if len(parts) >= 3 else ""

print("Loading dense and sparse chunks...")
dense_chunks = get_all_chunks(dense_base)
sparse_chunks = get_all_chunks(sparse_base)

print(f"Dense chunks: {len(dense_chunks)}, Sparse chunks: {len(sparse_chunks)}")

# Create 1-to-1 mapping: dense-N-file.json -> sparse-N-file.json
equivalents = {}

print("Creating 1-to-1 mapping based on chunk numbers...")

for dense_chunk in dense_chunks:
    dense_id = dense_chunk["chunk_id"]
    dense_num = get_chunk_number(dense_id)
    dense_file_suffix = get_file_suffix(dense_id)
    
    if dense_num is None:
        print(f"Warning: Could not extract chunk number from {dense_id}")
        equivalents[dense_id] = []
        continue
    
    # Find corresponding sparse chunk with same number and file
    # Expected format: sparse-N-same_file_suffix.json
    expected_sparse_id = f"sparse-{dense_num}-{dense_file_suffix}"
    
    # Check if this sparse chunk exists
    sparse_match = None
    for sparse_chunk in sparse_chunks:
        if sparse_chunk["chunk_id"] == expected_sparse_id:
            sparse_match = sparse_chunk["chunk_id"]
            break
    
    if sparse_match:
        equivalents[dense_id] = [sparse_match]
    else:
        # No matching sparse chunk found
        equivalents[dense_id] = []

# Statistics
total_dense = len(equivalents)
unmapped_dense = sum(1 for v in equivalents.values() if len(v) == 0)
mapped_dense = total_dense - unmapped_dense

print(f"\nMapping Statistics:")
print(f"Total dense chunks: {total_dense}")
print(f"Mapped dense chunks: {mapped_dense} ({mapped_dense/total_dense*100:.1f}%)")
print(f"Unmapped dense chunks: {unmapped_dense} ({unmapped_dense/total_dense*100:.1f}%)")

if unmapped_dense > 0:
    print(f"\nWarning: {unmapped_dense} dense chunks have no sparse equivalents")
    # Print first few unmapped for debugging
    unmapped_examples = [k for k, v in list(equivalents.items())[:10] if len(v) == 0]
    if unmapped_examples:
        print(f"Example unmapped dense chunks: {unmapped_examples[:3]}")

# Save mappings to JSON file
output_path = output_file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(equivalents, f, indent=2, ensure_ascii=False)

print(f"\nMappings saved to: {output_path}")
print(f"Total mappings: {len(equivalents)}")
