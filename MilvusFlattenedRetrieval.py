import os
import json
import torch
import numpy as np
import traceback
from datetime import datetime
from pymilvus import MilvusClient, Collection, connections
from sentence_transformers import SentenceTransformer
from queryExpansion import sparse_formatting, dense_formatting
from TickerFilter import get_tickers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import requests

from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

HF_KEY = os.getenv("HF_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variables for models
embedding_model = None
device = None

def initialize_models():
    global embedding_model, device
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on {device}")
    
    if embedding_model is None:
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embedding_model = embedding_model.to(device)
        print("Embedding model loaded to GPU")

def cleanup_models():
    global embedding_model
    if embedding_model is not None:
        embedding_model = embedding_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Models moved to CPU and cache cleared")

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_models()
    yield
    cleanup_models()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    user_query: str

class QueryResponse(BaseModel):
    query: str
    contexts: list
    answer: str
    error: str | None = None

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
connections.connect(host="localhost", port="19530")

COLLECTION_NAME = "flattened_10k_hybrid"

def normalize(arr):
    if len(arr) == 0 or arr.max() == arr.min():
        return np.ones_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def rrf(rank, k):
    return 1/(rank + k)

def milvus_flattened_hybrid_search(user_query: str, limit=50, use_rrf=True, alpha=0.3):
    """Perform hybrid search on flattened financial data collection"""
    
    global embedding_model, device
    
    sparse_query = sparse_formatting(user_query)
    dense_query = dense_formatting(user_query)
    
    try:
        # Generate dense embedding
        query = f"Task: Financial retrieval. Query: {dense_query}"

        dense_query_vector = embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        ).tolist()
        
        # Ticker filter
        detected_tickers = get_tickers(user_query)
        filter_expr = None
        if detected_tickers:
            ticker_filter = ",".join([f'"{t}"' for t in detected_tickers])
            filter_expr = f'ticker in [{ticker_filter}]'
        
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        dense_search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 256}
        }
        
        sparse_search_params = {
            "metric_type": "BM25",
            "params": {
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.6,
                "bm25_b": 0.4
            }
        }

        # Dense search
        dense_results = collection.search(
            data=[dense_query_vector],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["text", "ticker", "company_name", "fiscal_year", 
                          "line_item", "concept", "value", "value_formatted", 
                          "table_type", "document_type"]
        )

        # Sparse search
        sparse_results = collection.search(
            data=[sparse_query],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["text", "ticker", "company_name", "fiscal_year", 
                          "line_item", "concept", "value", "value_formatted", 
                          "table_type", "document_type"]
        )

        if not dense_results or not sparse_results:
            print("No results found in hybrid search!")
            return []
        
        # Normalize scores
        dense_scores = np.array([hit.distance for hit in dense_results[0]])
        sparse_scores = np.array([hit.distance for hit in sparse_results[0]])

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        final_scores = {}

        # RRF fusion
        if use_rrf:
            for rank, hit in enumerate(dense_results[0]):
                final_scores[hit.id] = final_scores.get(hit.id, 0) + rrf(rank, 30)
            
            for rank, hit in enumerate(sparse_results[0]):
                final_scores[hit.id] = final_scores.get(hit.id, 0) + rrf(rank, 30)
        else:
            # Weighted fusion
            for hit, score in zip(dense_results[0], dense_norm):
                final_scores[hit.id] = alpha * score

            for hit, score in zip(sparse_results[0], sparse_norm):
                final_scores[hit.id] = final_scores.get(hit.id, 0) + (1 - alpha) * score

        # Rank by final scores
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [r[0] for r in ranked[:limit]]

        # Collect all hits
        all_hits = {hit.id: hit for hit in dense_results[0]}
        all_hits.update({hit.id: hit for hit in sparse_results[0]})

        # Format results
        formatted_results = [[
            {
                "id": hit_id,
                "distance": final_scores[hit_id],
                "entity": {
                    "text": all_hits[hit_id].entity.get("text", ""),
                    "ticker": all_hits[hit_id].entity.get("ticker", ""),
                    "company_name": all_hits[hit_id].entity.get("company_name", ""),
                    "fiscal_year": all_hits[hit_id].entity.get("fiscal_year", ""),
                    "line_item": all_hits[hit_id].entity.get("line_item", ""),
                    "concept": all_hits[hit_id].entity.get("concept", ""),
                    "value": all_hits[hit_id].entity.get("value", ""),
                    "value_formatted": all_hits[hit_id].entity.get("value_formatted", ""),
                    "table_type": all_hits[hit_id].entity.get("table_type", ""),
                    "document_type": all_hits[hit_id].entity.get("document_type", "")
                }
            }
            for hit_id in top_ids
        ]]
        
        return formatted_results
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_answer_with_llm(user_query: str, contexts: list) -> str:
    """Generate answer using OpenAI API with retrieved contexts"""
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not found in environment variables"
    
    if not contexts:
        return "No relevant data found to answer the question."
    
    # Format contexts for LLM
    context_text = "\n\n".join([
        f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts[:10])
    ])
    
    prompt = f"""You are a financial analyst expert. Using ONLY the retrieved financial data, answer the user's query with strict adherence to the formatting and calculation patterns shown below.

User Query:
{user_query}

Query Results:
{context_text}

**CRITICAL RULES:**
- Use ONLY values present in query_results. Do not infer, estimate, or hallucinate any numbers.
- If query_results is empty or lacks the required data, respond exactly: "Data unavailable"
- If partial data exists (e.g., only one year when two needed), work with what's available and note the limitation.

**FORMATTING STANDARDS:**

1. **Money Format - MANDATORY:**
   - Always use: `$[full_value] ($[short_scale])`
   - Full value: Include all digits with commas (e.g., `$352,583,000,000`)
   - Short scale: Round to 2 decimals with B or M (e.g., `($352.58B)`, `($53.89B)`)
   - Negatives: Use minus sign prefix: `-$2,722,000,000 (-$2.72B)` NOT "(loss)" or "negative"
   - Examples from dataset:
     * `$352,583,000,000 ($352.58B)`
     * `-$2,722,000,000 (-$2.72B)`
     * `$391,000,000 ($0.39B)`

2. **Percentages:**
   - Calculated percentages: One decimal place (e.g., `17.1%`, `4.2%`)
   - Large percentage changes: Whole numbers acceptable (e.g., `348%`)
   - Always show the calculation: `((2023 value - 2021 value) ÷ 2021 value) x 100`

3. **Ratios:**
   - Round to 2 decimal places: `approximately 4.67`, `2.79`
   - Always include context: "calculated as [numerator] divided by [denominator]"

4. **Calculations - MUST BE EXPLICIT:**
   - Temporal changes: `$X in [year2] - $Y in [year1] = $Z`
   - Percentage changes: `($X - $Y) / $Y ≈ Z%`
   - Ratios: `$X / $Y ≈ Z`
   - Examples:
     * `$84,946,000,000 ($84.95B) divided by $30,425,000,000 ($30.43B)`
     * '($3,485M - $2,970M) / $2,970M ≈ 17.3% increase'

**ANSWER STRUCTURE PATTERNS:**

1. **Simple Lookup (single metric, single year):**
   Template: `[Company] reported [metric] of $[full_value] ($[short_scale]) in [year].`
   Example: `Apple reported total assets of $352,583,000,000 ($352.58B) in 2023.`

2. **Temporal Change (comparison between years):**
   Template: `[Company]'s [metric] [increased/decreased] from $[val1] ($[short1]) in [year1] to $[val2] ($[short2]) in [year2][, representing a [change] of $[delta] ($[delta_short])].`
   Show calculation when percentage requested.
   Example: `Starbucks' total liabilities decreased by $466,800,000 ($0.47B) from 2020 to 2021.`

3. **Multi-Company Comparison:**
   Template: `[Company A] [reported/had] [metric] of $[valA] ($[shortA]) [compared to/versus] [Company B]'s $[valB] ($[shortB]) in [year].`
   Always state the winner explicitly.
   Example: `Broadcom's net income ($11,223,000,000) was higher than NVIDIA's ($9,752,000,000) in 2022.`

4. **Trend Analysis (multi-year):**
   Option A: Year-by-year listing if all values needed
   Option B: Narrative description: `[Company]'s [metric] [increased/decreased/remained stable] from [start_year] to [end_year].`
   Example: `Gilead's total assets decreased steadily from $68,400,000,000 ($68.40B) in 2020 to $59,000,000,000 ($59.00B) in 2024.`

5. **Yes/No or Directional Questions:**
   Start with clear answer: `Yes` or `No`, then provide supporting data.
   Example: `Yes, Gilead's R&D expenses increased from $5,040,000,000 ($5.04B) in 2020 to $5,910,000,000 ($5.91B) in 2024, representing an increase of $868,000,000 ($0.87B).`

**DATA SELECTION RULES:**

When query_results contains multiple rows:
1. Filter to requested ticker(s), year(s), and concept(s) - ignore irrelevant rows
2. If duplicate rows exist for same ticker+year+concept:
   - Prefer exact concept match over label match
   - Prefer correct table_type (IncomeStatement/BalanceSheet/CashFlowStatement)
   - Prefer labels containing "Total" for aggregate metrics
   - Use the most recent filing or highest magnitude as final tie-breaker
3. For temporal comparisons: Extract values for ALL requested years
4. For cross-statement calculations: Ensure metrics from same ticker and same year

**PARTIAL DATA HANDLING:**

- If query asks for derived metric (e.g., "FCF yield", "ROE") but denominators missing:
  * Answer with available components (e.g., compare FCF values instead of yields)
  * Add brief note: "Yield calculation requires market cap, which is not in the provided data."
- If comparing companies but one is missing: Compare available companies and note the gap
- DO NOT fabricate or estimate missing values

**CONSTRAINTS:**

- No preamble, no repetition of the question, no commentary
- No references to "database", "graph", "query", or "retrieved data"
- No qualitative opinions or narrative context
- Stay focused: answer only what was asked
- Length: 1-3 sentences for simple queries; up to 1 short paragraph for complex comparisons

**OUTPUT REQUIREMENTS:**

Provide ONLY the answer text. No JSON, no explanations, no meta-commentary.
"""
    
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}"

def milvus_flattened_retrieval_pipeline(user_query: str, top_k=10):
    """Complete retrieval and generation pipeline"""
    
    print(f"\nProcessing query: {user_query}")
    
    # Step 1: Hybrid search
    hybrid_results = milvus_flattened_hybrid_search(user_query, limit=50)
    
    if not hybrid_results or not hybrid_results[0]:
        return {
            "query": user_query,
            "contexts": [],
            "answer": "No relevant data found to answer the question.",
            "error": "No results from hybrid search"
        }
    
    # Step 2: Extract text field as contexts
    contexts = []
    for result in hybrid_results[0][:top_k]:
        entity = result.get("entity", {})
        
        # Use the pre-generated text field from Milvus
        text = entity.get("text", "")
        if text:
            contexts.append(text)
    
    print(f"Retrieved {len(contexts)} contexts")
    
    # Step 3: Generate answer
    answer = generate_answer_with_llm(user_query, contexts)
    
    return {
        "query": user_query,
        "contexts": contexts,
        "answer": answer,
        "error": None
    }

def save_results_to_file(output_data):
    """Save results to JSON file"""
    os.makedirs("hybrid_retrieval_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("hybrid_retrieval_outputs", f"flattened_hybrid_results_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_path}")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """API endpoint for Milvus flattened hybrid retrieval"""
    try:
        print(f"Received query: {request.user_query}")
        results = milvus_flattened_retrieval_pipeline(request.user_query)
        return QueryResponse(**results)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Load EvaluationDataset.json and run on all questions
    import json
    
    initialize_models()
    
    # Load evaluation dataset
    with open("RAGAS/EvaluationDataset.json", "r") as f:
        evaluation_dataset = json.load(f)
    
    # Store results
    results = []
    
    print(f"\nProcessing {len(evaluation_dataset)} questions from EvaluationDataset.json")
    print("="*80)
    
    for i, item in enumerate(evaluation_dataset, 1):
        question = item["question"]
        ground_truth = item["answer"]
        
        print(f"\n[{i}/{len(evaluation_dataset)}] Processing: {question[:60]}...")
        
        # Run retrieval pipeline with top_k=5
        result = milvus_flattened_retrieval_pipeline(question, top_k=5)
        
        # Add ground truth and index
        result["ground_truth"] = ground_truth
        result["question_id"] = item["id"]
        results.append(result)
        
        print(f"  Generated answer: {result['answer'][:100]}...")
    
    # Save results to file
    output_path = "hybrid_retrieval_outputs/evaluation_dataset_results.json"
    os.makedirs("hybrid_retrieval_outputs", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to {output_path}")
    print(f"Total questions processed: {len(results)}")
    print("="*80)
    
    # Uncomment to run API server
    # import uvicorn
    # uvicorn.run("MilvusFlattenedRetrieval:app", host="0.0.0.0", port=8001, reload=True)
