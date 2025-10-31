import os
import json
import sys
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Add Graph directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "Graph"))
try:
    from GraphQueryTest import execute_graph_retrieval_process
except ImportError:
    print("Warning: Could not import GraphQueryTest - make sure Graph directory is in path")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# OpenAI Configuration (RAGAS uses OpenAI for evaluation)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# ============================================================================
# DATASET STRUCTURE EXPLANATION
# ============================================================================

"""
RAGAS requires a dataset with the following columns:

REQUIRED FOR ALL METRICS:
- question: str - The user's question/query
- answer: str - Your RAG system's generated answer
- contexts: list[str] - Retrieved context chunks from Neo4j (as list of strings)

OPTIONAL (but recommended for comprehensive evaluation):
- ground_truth: str - The correct/expected answer (needed for: context_recall, 
  answer_correctness, answer_similarity)

EXAMPLE STRUCTURE:
{
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "contexts": [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe with Paris as its capital."
    ],
    "ground_truth": "Paris is the capital of France."  # Optional but recommended
}
"""

# ============================================================================
# METRIC SELECTION GUIDE
# ============================================================================

"""
METRICS THAT DON'T NEED GROUND TRUTH:
- faithfulness: Measures if answer is grounded in the retrieved context
- answer_relevancy: Measures how relevant the answer is to the question
- context_precision: Measures if relevant contexts are ranked higher

METRICS THAT NEED GROUND TRUTH:
- context_recall: Measures if all relevant info in ground_truth is in contexts
- answer_correctness: Semantic + factual correctness vs ground_truth
- answer_similarity: Semantic similarity between answer and ground_truth
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_neo4j_driver():
    """Create Neo4j driver connection"""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def retrieve_contexts_from_neo4j(question: str, driver) -> list[str]:
    """
    Retrieve relevant context from Neo4j for a given question using GraphQueryTest.
    """
    try:
        # Use the enhanced graph retrieval
        result = execute_graph_retrieval_process(question)
        
        # Convert Neo4j results to context strings
        contexts = []
        for record in result['results']:
            if 'message' in record and record['message'] == 'No Data Found':
                contexts.append("No relevant data found in the graph database.")
            else:
                context_parts = []
                if 'ticker' in record:
                    context_parts.append(f"Company: {record.get('name', record['ticker'])} ({record['ticker']})")
                if 'line_item' in record:
                    context_parts.append(f"Metric: {record['line_item']}")
                if 'value' in record:
                    context_parts.append(f"Value: {record['value']}")
                if 'year' in record:
                    context_parts.append(f"Year: {record['year']}")
                if 'yoy_change_percent' in record and record['yoy_change_percent']:
                    context_parts.append(f"YoY Change: {record['yoy_change_percent']:.2f}%")
                if 'percentile' in record:
                    context_parts.append(f"Industry Percentile: {record['percentile']:.1f}%")
                
                contexts.append(" | ".join(context_parts))
        
        return contexts if contexts else ["No relevant data found."]
    
    except Exception as e:
        print(f"Error retrieving from Neo4j: {e}")
        return [f"Error: {str(e)}"]


def generate_answer_with_rag(question: str, contexts: list[str]) -> str:
    """
    Generate answer using your RAG pipeline.
    
    CUSTOMIZE THIS with your actual RAG implementation.
    """
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    context_text = "\n\n".join(contexts)
    prompt = f"""Based on the following context, answer the question.
    
Context:
{context_text}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_evaluation_dataset_without_ground_truth():
    """
    Prepare dataset WITHOUT ground truth answers.
    Use this if you only have questions and want to evaluate retrieval/generation quality.
    """
    driver = get_neo4j_driver()
    
    # Load questions from cedAnswer.json
    json_path = Path(__file__).parent / "cedAnswer.json"
    with open(json_path, 'r') as f:
        test_cases = json.load(f)
    
    # Extract just questions
    questions = [tc["question"] for tc in test_cases]
    
    data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    for question in questions:
        print(f"Processing: {question[:60]}...")
        
        # Retrieve contexts from Neo4j
        contexts = retrieve_contexts_from_neo4j(question, driver)
        
        # Generate answer using RAG
        answer = generate_answer_with_rag(question, contexts)
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
    
    driver.close()
    return Dataset.from_dict(data)


def prepare_evaluation_dataset_with_ground_truth():
    """
    Prepare dataset WITH ground truth answers from cedAnswer.json.
    Use this for comprehensive evaluation including answer correctness.
    """
    driver = get_neo4j_driver()
    
    # Load test cases from cedAnswer.json
    json_path = Path(__file__).parent / "cedAnswer.json"
    with open(json_path, 'r') as f:
        test_cases = json.load(f)
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for test_case in test_cases:
        question = test_case["question"]
        print(f"Processing: {question[:60]}...")
        
        # Retrieve contexts from Neo4j
        contexts = retrieve_contexts_from_neo4j(question, driver)
        
        # Generate answer using RAG
        answer = generate_answer_with_rag(question, contexts)
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(test_case["answer"])
    
    driver.close()
    return Dataset.from_dict(data)


def load_existing_dataset_from_file():
    """
    Load pre-existing evaluation dataset from file (CSV, JSON, etc.)
    """
    import pandas as pd
    
    # Example: Load from CSV
    df = pd.read_csv("evaluation_dataset.csv")
    
    # Ensure contexts is a list (if stored as string in CSV)
    df['contexts'] = df['contexts'].apply(eval)  # Convert string to list
    
    return Dataset.from_pandas(df)


def analyze_results_by_query_pattern(results_df, test_cases):
    """
    OPTIONAL: Analyze results AFTER evaluation to identify patterns.
    This is for YOUR analysis, not for influencing the evaluation.
    """
    # Extract query patterns from comments
    query_patterns = []
    for tc in test_cases:
        comment = tc.get('_comment', 'unknown')
        # Extract the main pattern type
        pattern = comment.split('-')[0].strip() if '-' in comment else comment
        query_patterns.append(pattern)
    
    results_df['query_pattern'] = query_patterns
    
    print("\n" + "="*80)
    print("PERFORMANCE BY QUERY PATTERN (Post-hoc Analysis)")
    print("="*80)
    
    grouped = results_df.groupby('query_pattern').agg({
        'answer_correctness': ['mean', 'std', 'count'],
        'faithfulness': ['mean', 'std'],
        'answer_relevancy': ['mean', 'std']
    }).round(3)
    
    print(grouped)
    print("\n")
    
    return grouped


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_without_ground_truth():
    """Evaluate using metrics that don't require ground truth"""
    print("Preparing dataset without ground truth...")
    dataset = prepare_evaluation_dataset_without_ground_truth()
    
    print(f"Dataset size: {len(dataset)}")
    print("\nEvaluating with metrics (no ground truth needed)...")
    
    # Metrics that work without ground truth
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
    
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=ChatOpenAI(model="gpt-4"),
        embeddings=OpenAIEmbeddings()
    )
    
    return results


def evaluate_with_ground_truth():
    """Evaluate using all metrics including those requiring ground truth"""
    print("Preparing dataset with ground truth...")
    dataset = prepare_evaluation_dataset_with_ground_truth()
    
    print(f"Dataset size: {len(dataset)}")
    print("\nEvaluating with all metrics...")
    
    # All metrics including those requiring ground truth
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,          # Needs ground_truth
        answer_correctness,      # Needs ground_truth
        answer_similarity        # Needs ground_truth
    ]
    
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=ChatOpenAI(model="gpt-4"),
        embeddings=OpenAIEmbeddings()
    )
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RAGAS Evaluation for Neo4j Graph RAG - Graph-Specific Capabilities")
    print("=" * 80)
    
    # Choose your evaluation approach:
    
    # Option 1: Without ground truth (faster, less comprehensive)
    # results = evaluate_without_ground_truth()
    
    # Option 2: With ground truth (more comprehensive, recommended)
    results = evaluate_with_ground_truth()
    
    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results)
    
    # Save results
    results_df = results.to_pandas()
    results_df.to_csv("ragas_evaluation_results.csv", index=False)
    print("\nResults saved to: ragas_evaluation_results.csv")
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for metric in results.keys():
        if metric not in ['question', 'answer', 'contexts', 'ground_truth']:
            print(f"{metric}: {results[metric]:.4f}")
    
    # OPTIONAL: Analyze by query pattern
    print("\n" + "=" * 80)
    print("ANALYZING PERFORMANCE BY QUERY TYPE...")
    print("=" * 80)
    
    with open(Path(__file__).parent / "cedAnswer.json", 'r') as f:
        test_cases = json.load(f)
    
    pattern_analysis = analyze_results_by_query_pattern(results_df, test_cases)
    pattern_analysis.to_csv("query_pattern_analysis.csv")
    print("\nPattern analysis saved to: query_pattern_analysis.csv")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)