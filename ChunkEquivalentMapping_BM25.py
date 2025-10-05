from rank_bm25 import BM25Okapi
import os
import json

dense_base = "chunked_dense"
sparse_base = "chunked_sparse"
output_file = "chunk_equivalents_bm25.json"

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
                                "file_path": file_path
                            })
    return chunks

def get_file_suffix(filename):
    parts = filename.split('_', 1)
    return parts[1] if len(parts) > 1 else filename

print("Loading dense and sparse chunks...")
dense_chunks = get_all_chunks(dense_base)
sparse_chunks = get_all_chunks(sparse_base)

print(f"Dense chunks: {len(dense_chunks)}, Sparse chunks: {len(sparse_chunks)}")

# One-to-many mapping: dense_chunk_id -> [sparse_chunk_id, ...]
equivalents = {}

print("Mapping dense to sparse chunks using BM25 (one-to-many)...")

for dense_chunk in dense_chunks:
    dense_suffix = get_file_suffix(dense_chunk["file"])
    candidate_sparse = [sc for sc in sparse_chunks if get_file_suffix(sc["file"]) == dense_suffix]
    if not candidate_sparse:
        equivalents[dense_chunk["chunk_id"]] = []
        continue

    # Prepare BM25 corpus
    sparse_texts = [sc["content"] for sc in candidate_sparse]
    tokenized_corpus = [text.split() for text in sparse_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Query is the dense chunk's content
    query_tokens = dense_chunk["content"].split()
    scores = bm25.get_scores(query_tokens)
    # Get indices of top 2 sparse chunks (one-to-many)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
    mapped_sparse_ids = [candidate_sparse[idx]["chunk_id"] for idx in top_indices]
    equivalents[dense_chunk["chunk_id"]] = mapped_sparse_ids

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(equivalents, f, indent=2, ensure_ascii=False)

print(f"Saved chunk equivalents to {output_file}")