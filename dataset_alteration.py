import os
import pandas as pd
import json
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Config
CORPUS_DIRECTORY = r"chunked_dense\10-K"
DATASET = "Financial-QA-10k.csv"
OUTPUT_FILE = "hard_negatives.jsonl"
TOP_K = 10       
SIMILARITY_THRESHOLD = 0.8
MAX_NEGATIVES = 4  

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def load_corpus(corpus_dir):
    corpus = []
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "content" in data:
                        corpus.append(data)
                    elif isinstance(data, list):
                        corpus.extend([chunk for chunk in data if "content" in chunk])
    return corpus

corpus = load_corpus(CORPUS_DIRECTORY)

# Tokenize BM25
tokenize_corpus = [c["content"].split() for c in corpus]
bm25 = BM25Okapi(tokenize_corpus)

corpus_texts = [c["content"] for c in corpus]
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

corpus_lookup = {c["content"]: idx for idx, c in enumerate(corpus)}

# Load dataset
df = pd.read_csv(DATASET).dropna(subset=["question", "context"])

triples = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    query = row["question"]
    positive = row["context"]

    # Find positive chunk
    positive_candidates = bm25.get_top_n(positive.split(), corpus, n=TOP_K)
    positive_chunk = next((cand for cand in positive_candidates if positive.strip() in cand["content"]), None)
    if not positive_chunk:
        positive_chunk = bm25.get_top_n(query.split(), corpus, n=1)[0]

    pos_idx = corpus_lookup[positive_chunk["content"]]
    pos_emb = corpus_embeddings[pos_idx].unsqueeze(0)

    # BM25 retrieve
    ranked_chunks = bm25.get_top_n(query.split(), corpus, n=TOP_K)
    ranked_indices = [corpus_lookup[c["content"]] for c in ranked_chunks if c["content"] != positive_chunk["content"]]

    # Batch similarity
    neg_embs = corpus_embeddings[ranked_indices]
    sims = util.cos_sim(pos_emb, neg_embs).squeeze(0)

    # Pick negatives below threshold
    neg_count = 0
    for idx, sim in zip(ranked_indices, sims):
        if sim.item() < SIMILARITY_THRESHOLD:
            triples.append({
                "query": query,
                "positive": positive_chunk["content"],
                "negative": corpus_texts[idx]
            })
            neg_count += 1
            if neg_count >= MAX_NEGATIVES:
                break

print("SAVING")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for triple in triples:
        f.write(json.dumps(triple) + "\n")

print("COMPLETED")
