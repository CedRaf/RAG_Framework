# filepath: c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG\ChunkAppend.py
import os
import json
import re
from collections import defaultdict
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

# Paths
chunk_equivalents_path = "chunk_equivalents_index.json"
sparse_base = "chunked_sparse"
output_path = "chunk_equivalents_expanded.json"

print("=" * 80)
print("CHUNK EQUIVALENTS EXPANDER - Cross-Year Text Matching")
print("=" * 80)

# Load chunk equivalents (1-to-1 mapping)
print("\n[1/5] Loading initial chunk equivalents...")
with open(chunk_equivalents_path, "r", encoding="utf-8") as f:
    chunk_equivalents = json.load(f)

print(f"     Loaded {len(chunk_equivalents)} dense chunk mappings")

# Load all sparse chunks organized by ticker
print("\n[2/5] Loading all sparse chunks and organizing by ticker...")
sparse_chunks_by_ticker = defaultdict(list)
chunk_text_map = {}  # Map chunk_id to its text content

for form_type in os.listdir(sparse_base):
    form_path = os.path.join(sparse_base, form_type)
    if not os.path.isdir(form_path):
        continue
    
    for ticker in os.listdir(form_path):
        ticker_path = os.path.join(form_path, ticker)
        if not os.path.isdir(ticker_path):
            continue
        
        for file in os.listdir(ticker_path):
            if file.endswith(".json"):
                file_path = os.path.join(ticker_path, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                        for chunk in chunks:
                            chunk_id = chunk.get("chunk_id")
                            content = chunk.get("content", "")
                            ticker_from_chunk = chunk.get("ticker")
                            
                            if chunk_id and content and ticker_from_chunk:
                                # Store chunk with ticker
                                sparse_chunks_by_ticker[ticker_from_chunk].append({
                                    "chunk_id": chunk_id,
                                    "content": content,
                                    "source_file": chunk.get("source_file", "")
                                })
                                # Map chunk_id to content for quick lookup
                                chunk_text_map[chunk_id] = content
                except Exception as e:
                    print(f"    ⚠ Error reading {file_path}: {e}")

print(f"     Loaded chunks from {len(sparse_chunks_by_ticker)} tickers")
for ticker, chunks in list(sparse_chunks_by_ticker.items())[:5]:
    print(f"      - {ticker}: {len(chunks)} chunks")

# Extract ticker and year from chunk_id or source file
def extract_ticker_year(chunk_id, source_file=""):
    """Extract ticker symbol and year from chunk_id or source file"""
    # Pattern: sparse-N-TICKER_FORMTYPE_YEAR_CIK.json
    # Example: sparse-1-320193_10K_2023_0000320193-23-000106.json
    
    # Try to extract from source_file first (more reliable)
    if source_file:
        match = re.search(r'(\d{6})_(10-[KQ]|8-K)_(\d{4})_', source_file)
        if match:
            cik = match.group(1)
            year = match.group(3)
            return cik, year
    
    # Fallback: try chunk_id
    match = re.search(r'sparse-\d+-(\d{6})_(?:10-[KQ]|8-K)_(\d{4})_', chunk_id)
    if match:
        cik = match.group(1)
        year = match.group(2)
        return cik, year
    
    return None, None

print("\n[3/5] Expanding chunk mappings by finding cross-year text matches...")
expanded_count = 0
no_match_count = 0
total_processed = 0

# Expand each mapping
for dense_id, sparse_ids in chunk_equivalents.items():
    total_processed += 1
    
    if not sparse_ids or len(sparse_ids) == 0:
        no_match_count += 1
        continue
    
    # Get the original sparse chunk (1-to-1 mapping)
    original_sparse_id = sparse_ids[0]
    
    # Get the text content of the original sparse chunk
    if original_sparse_id not in chunk_text_map:
        no_match_count += 1
        continue
    
    target_text = chunk_text_map[original_sparse_id]
    
    # Extract ticker and year from original
    # Try to get source_file from the chunk data
    original_ticker = None
    original_year = None
    
    for ticker, chunks in sparse_chunks_by_ticker.items():
        for chunk in chunks:
            if chunk["chunk_id"] == original_sparse_id:
                original_ticker = ticker
                _, original_year = extract_ticker_year(original_sparse_id, chunk["source_file"])
                break
        if original_ticker:
            break
    
    if not original_ticker:
        no_match_count += 1
        continue
    
    # Find all chunks from the same ticker with exact text match
    matching_chunks = set(sparse_ids)  # Start with original mapping
    
    if original_ticker in sparse_chunks_by_ticker:
        for chunk in sparse_chunks_by_ticker[original_ticker]:
            # Skip if it's the same chunk
            if chunk["chunk_id"] == original_sparse_id:
                continue
            
            # Check for exact text match
            if chunk["content"].strip() == target_text.strip():
                matching_chunks.add(chunk["chunk_id"])
                expanded_count += 1
    
    # Update mapping with expanded list
    chunk_equivalents[dense_id] = sorted(list(matching_chunks))
    
    # Progress indicator
    if total_processed % 100 == 0:
        print(f"    Processing: {total_processed}/{len(chunk_equivalents)} dense chunks...")

print(f"\n     Expanded {expanded_count} additional cross-year matches")
print(f"     {no_match_count} chunks had no additional matches")

# Statistics
print("\n[4/5] Calculating expansion statistics...")
expansion_stats = defaultdict(int)
for dense_id, sparse_ids in chunk_equivalents.items():
    count = len(sparse_ids)
    expansion_stats[count] += 1

print("\n    Distribution of matches per dense chunk:")
for count in sorted(expansion_stats.keys()):
    percentage = (expansion_stats[count] / len(chunk_equivalents)) * 100
    print(f"      {count} match(es): {expansion_stats[count]} chunks ({percentage:.1f}%)")

# Save expanded mappings
print(f"\n[5/5] Saving expanded mappings to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunk_equivalents, f, indent=2, ensure_ascii=False)

print(f"     Saved expanded mappings")

# Show examples
print("\n" + "=" * 80)
print("SAMPLE EXPANSIONS")
print("=" * 80)

example_count = 0
for dense_id, sparse_ids in chunk_equivalents.items():
    if len(sparse_ids) > 1:  # Only show expanded mappings
        print(f"\n{dense_id} →")
        for sparse_id in sparse_ids:
            # Try to show year info
            for ticker, chunks in sparse_chunks_by_ticker.items():
                for chunk in chunks:
                    if chunk["chunk_id"] == sparse_id:
                        _, year = extract_ticker_year(sparse_id, chunk["source_file"])
                        if year:
                            print(f"  - {sparse_id} (Year: {year})")
                        else:
                            print(f"  - {sparse_id}")
                        break
        
        example_count += 1
        if example_count >= 3:
            break

print("\n" + "=" * 80)
print("EXPANSION COMPLETE!")
print("=" * 80)
print(f"\nOriginal mappings: {len(chunk_equivalents)}")
print(f"Additional cross-year matches found: {expanded_count}")
print(f"Output saved to: {output_path}")
print("\nYou can now use this expanded mapping for hybrid retrieval!")
