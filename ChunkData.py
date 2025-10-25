import json
import os
import re
import pandas as pd
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import TokenTextSplitter
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

with open("company_tickers_exchange.json", "r") as f:
    CIK_dict = json.load(f)

CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])
CIK_to_ticker = {str(row["cik"]).zfill(10): row["ticker"] for _, row in CIK_df.iterrows()}

def segment_dense(base_path=".", input_folders=["10-K", "10-Q", "8-K"]):
    segment_config(
        base_path=base_path,
        input_folders=input_folders,
        output_folder="chunked_dense",
        chunk_size=512,      # ~300 tokens (set value 1300 when using langchain recursive character splitter)
        chunk_overlap=102,    # ~50(set value 200 when using langchain recursive character splitter)
        vector_type="dense"
    )

def segment_sparse(base_path=".", input_folders=["10-K", "10-Q", "8-K"]):
    segment_config(
        base_path=base_path,
        input_folders=input_folders,
        output_folder="chunked_sparse", 
        chunk_size=256,     # ~256 tokens (set value to 1024 when using langchain recursive character splitter)
        chunk_overlap=64,   # ~64(set value to 256 when using langchain recursive character splitter)
        vector_type="sparse"
    )

def segment_config(base_path, input_folders, output_folder, chunk_size, chunk_overlap, vector_type):
    input_base = os.path.join(base_path, "edgar-crawler", "datasets", "EXTRACTED_FILINGS")
    output_base = os.path.join(base_path, output_folder)
    os.makedirs(output_base, exist_ok=True)

    splitter = TokenTextSplitter( #change to use RecursiveCharacterTextSplitter if needed
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # splitter = RecursiveCharacterTextSplitter( #change to use RecursiveCharacterTextSplitter if needed
    #     chunk_size=chunk_size,
    #     chunk_overlap=chunk_overlap,
    #     length_function=len,
    #     separators=["\n\n", "\n", " ", ""]
    # )
    for folder in input_folders:
        form_type = folder  # e.g., "10-K", "10-Q", "8-K"
        input_dir = os.path.join(input_base, form_type)

        if not os.path.exists(input_dir):
            print(f"Input folder not found: {input_dir}")
            continue

        output_form_dir = os.path.join(output_base, form_type)
        os.makedirs(output_form_dir, exist_ok=True)

        print(f"\nProcessing {form_type}...")

        for file in sorted(os.listdir(input_dir)):
            if not file.endswith(".json"):
                continue

            # Use company name or fallback for output subfolder
            filename_parts = file.replace(".json", "").split("_")
            if len(filename_parts) >= 4:
                cik_stripped = filename_parts[0]
                cik_padded = cik_stripped.zfill(10)
                ticker = CIK_to_ticker.get(cik_padded, cik_stripped).upper()
            else:
                ticker = "UNKNOWN"

            file_path = os.path.join(input_dir, file)

            ticker_output_dir = os.path.join(output_form_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)

            output_filename = os.path.splitext(file)[0] + ".json"
            output_path = os.path.join(ticker_output_dir, output_filename)

            # File existing check
            if os.path.exists(output_path):
                print(f"Skipping already chunked file: {output_path}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                chunk_objects = []

                if not isinstance(data, dict):
                    print(f"Skipping non-dict file: {file}")
                    continue

                item_keys = [key for key in data if key.startswith("item_") or key.startswith("parts_") or key.startswith("part_")]
                # keep a running counter so chunk IDs don't reset for each item
                chunk_counter = 0
                for item_key in item_keys:
                    content = data[item_key].strip()
                    if not content:
                        continue  # Skip empty items

                    # Clean and normalize whitespace
                    content = re.sub(r"\s+", " ", content)

                    # Chunk normally
                    item_chunks = splitter.split_text(content) if len(content) > chunk_size else [content]

                    # Track position in original content to find accurate indices
                    # Use a more robust approach that handles duplicate text
                    search_start_pos = 0
                    prev_chunk_end = 0
                    
                    for i, chunk in enumerate(item_chunks):
                        # Find where this chunk appears in the original content
                        # Start searching from where we expect it based on previous chunk
                        
                        # For first chunk, search from beginning
                        if i == 0:
                            chunk_start = content.find(chunk, 0)
                        else:
                            # For subsequent chunks, search from near the end of previous chunk
                            # Allow backtrack for overlap
                            overlap_chars = chunk_overlap * 4  # Estimate: ~4 chars per token
                            search_from = max(0, prev_chunk_end - overlap_chars)
                            chunk_start = content.find(chunk, search_from)
                            
                            # If not found near previous chunk, it might be a duplicate
                            # Try searching from just after previous chunk start instead
                            if chunk_start == -1 or chunk_start < search_start_pos:
                                chunk_start = content.find(chunk, search_start_pos)
                        
                        if chunk_start == -1:
                            # If exact match not found (rare case), fall back to estimated position
                            # This can happen if TokenTextSplitter modified the text
                            chunk_start = prev_chunk_end
                            chunk_end = chunk_start + len(chunk)
                        else:
                            chunk_end = chunk_start + len(chunk)
                        
                        # use running counter so numbering continues across items in the same file
                        chunk_id = f"{vector_type}-{chunk_counter+1}-{file}"
                        chunk_objects.append({
                            "chunk_id": chunk_id,
                            "item": item_key,
                            "content": chunk,
                            "ticker": ticker,
                            "form_type": data.get("filing_type", form_type),
                            "filing_date": data.get("filing_date", ""),
                            "source_file": file,
                            "start_index": chunk_start,
                            "end_index": chunk_end
                        })
                        chunk_counter += 1
                        
                        # Update tracking variables for next iteration
                        # Move search position to just after current chunk start
                        # This prevents matching the same chunk twice while allowing overlap
                        search_start_pos = chunk_start + 1
                        prev_chunk_end = chunk_end

                if not chunk_objects:
                    print(f"Skipped (no valid chunks): {file}")
                    continue

                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(chunk_objects, json_file, indent=2, ensure_ascii=False)

                print(f"Chunked and saved: {file} -> {len(chunk_objects)} chunks")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
segment_dense()
#segment_sparse()


