import os
import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

def embed_chunks(base_path=".", input_folder="chunked_dense", output_folder="dense_embeddings", batch_size=256):
    input_base = os.path.join(base_path, input_folder)
    output_base = os.path.join(base_path, output_folder)
    os.makedirs(output_base, exist_ok=True)

    #model = SentenceTransformer('all-MiniLM-L6-v2') #Uncomment this to use all-miniLM
    #model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    model = SentenceTransformer(r"D:\models\bge_dense_finetuned_triplet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for form_type in sorted(os.listdir(input_base)):
        form_input_dir = os.path.join(input_base, form_type)
        form_output_dir = os.path.join(output_base, form_type)
        os.makedirs(form_output_dir, exist_ok=True)

        for ticker in sorted(os.listdir(form_input_dir)):
            ticker_input_dir = os.path.join(form_input_dir, ticker)
            ticker_output_dir = os.path.join(form_output_dir, ticker)
            os.makedirs(ticker_output_dir, exist_ok=True)

            for file in sorted(os.listdir(ticker_input_dir)):
                file_path = os.path.join(ticker_input_dir, file)
                

                output_file = file.replace("_chunks.json", "_embedded.json")
                output_path = os.path.join(ticker_output_dir, output_file)

                # Skip if already embedded
                if os.path.exists(output_path):
                    print(f"Skipping already embedded file: {output_path}")
                    continue
                with open(file_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                
                #texts = [chunk["content"] for chunk in chunks] #Uncomment this to use all-miniLM
                texts = [f"Represent this passage for retrieval: {chunk['content']}" for chunk in chunks]
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    device=device
                )

                embeddings = embeddings/np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                entries = []
                for chunk, emb in zip(chunks, embeddings):
                    entry = {
                        "embedding": emb.tolist(),
                        "metadata": {
                            "chunk_id": chunk.get("chunk_id", ""),
                            "source_file": chunk.get("source_file", file),
                            "ticker": chunk.get("ticker", ""),
                            "form_type": chunk.get("form_type", ""),
                            "start_index": chunk.get("start_index", -1),
                            "end_index": chunk.get("end_index", -1)
                        }
                    }
                    entries.append(entry)
                
                # output_file = file.replace("_chunks.json", "_embedded.json")
                # output_path = os.path.join(ticker_output_dir, output_file)
                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(entries, out, indent=2, ensure_ascii=False)
                
                print(f"Embedded: {file} -> {len(embeddings)} vectors")

    print("Embedding Complete")


embed_chunks()