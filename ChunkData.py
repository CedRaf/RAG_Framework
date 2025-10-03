import json
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def segment_dense(base_path=".", input_folders=["10-K", "10-Q", "8-K"]):
    segment_config(
        base_path=base_path,
        input_folders=input_folders,
        output_folder="chunked_dense",
        chunk_size=1300,      # ~300 tokens
        chunk_overlap=200,
        vector_type="dense"
    )

def segment_sparse(base_path=".", input_folders=["10-K", "10-Q", "8-K"]):
    segment_config(
        base_path=base_path,
        input_folders=input_folders,
        output_folder="chunked_sparse", 
        chunk_size=1024,     # ~256 tokens
        chunk_overlap=256,
        vector_type="sparse"
    )

def segment_config(base_path, input_folders, output_folder, chunk_size, chunk_overlap, vector_type):
    input_base = os.path.join(base_path, "edgar-crawler", "datasets", "EXTRACTED_FILINGS")
    output_base = os.path.join(base_path, output_folder)
    os.makedirs(output_base, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

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

                item_keys = [key for key in data if key.startswith("item_") or key.startswith("parts_")]
                for item_key in item_keys:
                    content = data[item_key].strip()
                    if not content:
                        continue  # Skip empty items

                    # Clean and normalize whitespace
                    content = re.sub(r"\s+", " ", content)

                    # Chunk normally
                    item_chunks = splitter.split_text(content) if len(content) > chunk_size else [content]

                    start = 0
                    for i, chunk in enumerate(item_chunks):
                        end = start + len(chunk)
                        chunk_id = f"{vector_type}-{i+1}-{file}"
                        chunk_objects.append({
                            "chunk_id": chunk_id,
                            "item": item_key,
                            "content": chunk,
                            "ticker": ticker,
                            "form_type": data.get("filing_type", form_type),
                            "filing_date": data.get("filing_date", ""),
                            "source_file": file,
                            "start_index": start,
                            "end_index": end
                        })
                        start = end - chunk_overlap

                if not chunk_objects:
                    print(f"Skipped (no valid chunks): {file}")
                    continue

                with open(output_path, "w", encoding="utf-8") as json_file:
                    json.dump(chunk_objects, json_file, indent=2, ensure_ascii=False)

                print(f"Chunked and saved: {file} -> {len(chunk_objects)} chunks")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
#segment_dense()
segment_sparse()


