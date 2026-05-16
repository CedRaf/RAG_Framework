from pymilvus import connections, MilvusClient
import json
import re
from datetime import datetime
import os
from queryExpansion import sparse_formatting, dense_formatting
from QueryTest import sparse_retrieval_test, dense_retrieval_test, fetch_text, rerank

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", ports="19530")

def hybrid_retrieval(user_query: str, sparse_ratio=0.6):
    #pseudo_doc = expand_query(user_query)
    sparse_query = sparse_formatting(user_query)
    dense_query = dense_formatting(user_query)

    sparse_results = sparse_retrieval_test(sparse_query)
    dense_results = dense_retrieval_test(dense_query)

    # Calculate how many from each method
    total_candidates = 50
    sparse_count = int(total_candidates * sparse_ratio)
    dense_count = total_candidates - sparse_count

    candidates = []
    seen_chunks = set()

    # Add best sparse results
    for i, result in enumerate(sparse_results[0][:sparse_count * 2]):  # Get extra in case of duplicates
        chunk_id = result['entity']['chunk_id']
        if chunk_id not in seen_chunks and len([c for c in candidates if c['source'] == 'sparse']) < sparse_count:
            candidates.append({
                'content': result['entity']['text'],
                'chunk_id': chunk_id,
                'source': 'sparse',
                'metadata': result['entity']
            })
            seen_chunks.add(chunk_id)

    # Add best dense results
    for i, hit in enumerate(dense_results[0][:dense_count * 2]):
        chunk_id = hit['entity']['chunk_id']
        if chunk_id not in seen_chunks and len([c for c in candidates if c['source'] == 'dense']) < dense_count:
            text = fetch_text(hit['entity'])
            candidates.append({
                'content': text,
                'chunk_id': chunk_id,
                'source': 'dense',
                'metadata': hit['entity']
            })
            seen_chunks.add(chunk_id)

    return candidates


def enhance_query_for_reranking(query):
    query_lower = query.lower()
    
    # Financial queries
    financial_terms = ["revenue", "earnings", "profit", "loss", "income", "cash"]
    if any(term in query_lower for term in financial_terms):
        return f"{query} financial performance metrics"
    
    # Strategic queries
    strategic_terms = ["strategy", "plan", "future", "outlook", "growth"]
    if any(term in query_lower for term in strategic_terms):
        return f"{query} business strategy planning"
    
    # Comparison queries
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return f"{query} comparative analysis"
    
    return query

def preprocess_passages_for_reranker(passages, query):
    processed = []
    
    for passage in passages:
        cleaned = re.sub(r'\s+', ' ', passage).strip()
        
        if len(cleaned.split()) > 350:
            cleaned = ' '.join(cleaned.split()[:350])
        
        if len(cleaned.split()) < 20:
            continue # Skip short passages
            
        processed.append(cleaned)
    
    return processed

def two_stage_reranking(query, passages, top_k=10):
    if len(passages) <= top_k:
        return rerank(query, passages, top_k=len(passages))

    stage1 = rerank(query, passages, top_k=min(12, len(passages)))
    stage1_texts = [text for text, _ in stage1]

    final_results = rerank(query, stage1_texts, top_k=top_k)
    
    return final_results

def hybrid_retrieval_test(user_query: str):
    hybrid_candidates = hybrid_retrieval(user_query, sparse_ratio=0.6)
    compiled_results = [candidate['content'] for candidate in hybrid_candidates]
    cleaned_passages = preprocess_passages_for_reranker(compiled_results, user_query)
    enhanced_query = enhance_query_for_reranking(user_query)
    reranked = two_stage_reranking(enhanced_query, cleaned_passages, top_k=10)

    reranked_with_metadata = []
    seen_source_files = set()
    
    for reranked_text, score in reranked:
        if len(reranked_with_metadata) >= 5:
            break
            
        matching_candidates = [candidate for candidate in hybrid_candidates if candidate['content'] == reranked_text]
        
        for candidate in matching_candidates:
            source_file = candidate['metadata'].get('source_file', 'Unknown')
            
            if source_file not in seen_source_files:
                seen_source_files.add(source_file)
                reranked_with_metadata.append({
                    "text": reranked_text,
                    "chunk_id": candidate['chunk_id'],
                    "source_file": source_file,
                    "ticker": candidate['metadata'].get('ticker', 'Unknown'),
                    "form_type": candidate['metadata'].get('form_type', 'Unknown'),
                    "source": candidate['source']
                })
                break

    for reranked_text, score in reranked:
        if len(reranked_with_metadata) >= 5:
            break
            
        already_included = any(result['text'] == reranked_text for result in reranked_with_metadata)
        
        if not already_included:
            # Find the first matching candidate
            for candidate in hybrid_candidates:
                if candidate['content'] == reranked_text:
                    reranked_with_metadata.append({
                        "text": reranked_text,
                        "chunk_id": candidate['chunk_id'],
                        "source_file": candidate['metadata'].get('source_file', 'Unknown'),
                        "ticker": candidate['metadata'].get('ticker', 'Unknown'),
                        "form_type": candidate['metadata'].get('form_type', 'Unknown'),
                        "source": candidate['source']
                    })
                    break

    output_data = {
        "reranked": reranked_with_metadata,
        "original_query": user_query,
        "hybrid_candidates": hybrid_candidates,
    }

    save_results_to_file(output_data)
    return output_data

def save_results_to_file(output_data):
    os.makedirs("retrieval_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("retrieval_outputs", f"retrieved_results_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    user_query = input("Enter your query: ")
    results = hybrid_retrieval_test(user_query)
    print(results)