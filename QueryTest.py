from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer, models
import json
from datetime import datetime
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from queryExpansion import expand_query, sparse_formatting, dense_formatting

reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")

reranker_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", ports="19530")

# SPARSE SEARCH
def sparse_retrieval_test(sparse_query:str):
    sparse_results = client.search(
        collection_name="sec_chunks_sparse",
        data=[sparse_query],  # BM25 function will convert text to sparse vector automatically
        anns_field="sparse",  # Search on the sparse vector field
        limit=50,  # Number of results to return
        output_fields=["text", "ticker", "form_type", "source_file", "chunk_id"],
        search_params={
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE"
            }
        }
    )

    print("Sparse Results Generated")

    return sparse_results

#DENSE SEARCH
def dense_retrieval_test(dense_query:str):

    collection = Collection("sec_chunks_dense")
    collection.load()
    print("Collection Loaded")

    model = SentenceTransformer("FinLang/finance-embeddings-investopedia")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    query_embedding= model.encode(
        dense_query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
    )

    print("Query Embedded")

    search_params = {
        "metric_type": "COSINE", 
        "params": {"ef": 128}
    }

    results = collection.search(
        data = [query_embedding.tolist()],
        anns_field="dense_vector",
        param=search_params,
        limit=50,
        output_fields=["source_file", "ticker", "form_type", "chunk_id", "start_index", "end_index"]
    )

    if not results:
        print("No results found, exiting...")
        exit()
    
    
    print("\nDense Results Generated")
    
    return results

def fetch_text(metadata):
    try:
        form_type = metadata.get("form_type", "")
        ticker = metadata.get("ticker", "")
        source_file = metadata.get("source_file", "")

        base_dir = "chunked_dense"

        form_dir = f"{form_type}" if form_type else ""

        chunk_dir = os.path.join(base_dir, form_dir, ticker.upper())
        chunk_file  = os.path.join(chunk_dir, source_file)

        with open(chunk_file, 'r', encoding="utf-8") as f:
            chunks = json.load(f)
        
        chunk_id = metadata.get("chunk_id", "")
        for chunk in chunks:
            if str(chunk.get("chunk_id", "")) == str(chunk_id):
                return chunk.get("content", "")
        
        return "Chunk not found"
    except FileNotFoundError:
        return f"File not found: {chunk_file}"
    except Exception as e:
        return f"Error: {str(e)}"

def rerank(query, passages, top_k=5):
    inputs = reranker_tokenizer(
        [query] * len(passages),
        passages,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(reranker_model.device)

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)

    ranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def vector_retrieval_test(user_query:str):
    psuedo_doc= expand_query(user_query)
    sparse_query= sparse_formatting(user_query, psuedo_doc)
    dense_query= dense_formatting(user_query, psuedo_doc)

    sparse_results= sparse_retrieval_test(sparse_query)
    dense_results= dense_retrieval_test(dense_query)

    compiled_results = []
    
    for result in sparse_results[0]:
        compiled_results.append(result['entity']['text'])

    for hit in dense_results[0]:
        text = fetch_text({...})  # Already in your code
        compiled_results.append(text)

    reranked = rerank(user_query, compiled_results, top_k=5)

    print("\nFinal Reranked Results:")
    print(f"Appended psuedo doc: {psuedo_doc}") 
    for i, (text, score) in enumerate(reranked):
        print(f"Rank {i+1} - Score: {score:.4f}")
        print(text[:300], "...\n", "-"*60)

    #Needed for the joint retrieval
    texts = [text for text, _ in reranked]
    scores = [score for _, score in reranked]

    # Saving to json file
    output_data = {
        "query": user_query,
        "results": [
            {
                "rank": i+1,
                "score": scores[i],
                "text": texts[i]
            }
            for i in range(len(texts))
        ],
        "sparse_top_50": [
            {
                "rank": i+1,
                "score": result['distance'],
                "ticker": result['entity']['ticker'],
                "form_type": result['entity']['form_type'],
                "source_file": result['entity']['source_file'],
                "text": result['entity']['text']
            }
            for i, result in enumerate(sparse_results[0])
        ],
        "dense_top_50": [
            {
                "rank": i+1,
                "score": hit['distance'],
                "ticker": hit['entity']['ticker'],
                "form_type": hit['entity']['form_type'],
                "source_file": hit['entity']['source_file'],
                "text": fetch_text({
                    "form_type": hit.entity.get("form_type", ""),
                    "ticker": hit.entity.get("ticker", ""),
                    "source_file": hit.entity.get("source_file", ""),
                    "chunk_id": hit.entity.get("chunk_id", "")
                })
            }
            for i, hit in enumerate(dense_results[0])
        ]
    }

    os.makedirs("retrieval_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("retrieval_outputs", f"retrieved_results_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Result saved to {output_path}")

# Same logic as 'vector_retrieval_test' just used as a callable function for other files
def execute_vector_retrieval(user_query:str):
    psuedo_doc= expand_query(user_query)
    sparse_query= sparse_formatting(user_query, psuedo_doc)
    dense_query= dense_formatting(user_query, psuedo_doc)

    sparse_results= sparse_retrieval_test(sparse_query)
    dense_results= dense_retrieval_test(dense_query)

    compiled_results = []
    
    for result in sparse_results[0]:
        compiled_results.append(result['entity']['text'])

    for hit in dense_results[0]:
        text = fetch_text({...})  # Already in your code
        compiled_results.append(text)

    reranked = rerank(user_query, compiled_results, top_k=5)
    
    texts = [text for text, _ in reranked]
    scores = [score for _, score in reranked]

    return {
        "results": texts,
        "scores": scores
    }

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    vector_retrieval_test(user_query)



