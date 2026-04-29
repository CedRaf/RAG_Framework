import os
import json
import sys
from pathlib import Path

# Check dependencies before proceeding
try:
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
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError as e:
    print("=" * 80)
    print("MISSING DEPENDENCIES")
    print("=" * 80)
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("pip install ragas datasets langchain-openai")
    print("=" * 80)
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Add Graph directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "Graph"))
try:
    from GraphQueryTest import execute_graph_retrieval_with_rag
except ImportError as e:
    print(f"Warning: Could not import GraphQueryTest - {e}")
    print("Make sure Graph directory is in path and GraphQueryTest.py exists")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

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

def retrieve_and_generate_answer(question: str) -> dict:
    """
    Use GraphQueryTest.py to retrieve contexts and generate answer for a question.
    
    Returns:
        dict with keys: 'answer', 'contexts', 'cypher_query', 'query_results'
    """
    try:
        # Call the actual RAG pipeline from GraphQueryTest
        result = execute_graph_retrieval_with_rag(question)
        
        # Extract the answer
        answer = result.get('final_answer', '')
        
        # Convert Neo4j query results to context strings
        contexts = []
        query_results = result.get('query_results', [])
        
        for record in query_results:
            if isinstance(record, dict):
                if 'message' in record:
                    contexts.append(record['message'])
                elif 'error' in record:
                    contexts.append(f"Error: {record['error']}")
                else:
                    # Format the record as a context string
                    context_parts = []
                    if 'ticker' in record:
                        context_parts.append(f"Company: {record.get('name', record['ticker'])} ({record['ticker']})")
                    if 'line_item' in record:
                        context_parts.append(f"Metric: {record['line_item']}")
                    if 'value' in record:
                        context_parts.append(f"Value: {record['value']}")
                    if 'year' in record:
                        context_parts.append(f"Year: {record['year']}")
                    if 'filing_type' in record:
                        context_parts.append(f"Filing: {record['filing_type']}")
                    if 'source_table' in record:
                        context_parts.append(f"Source: {record['source_table']}")
                    
                    if context_parts:
                        contexts.append(" | ".join(context_parts))
        
        # If no contexts extracted, provide a default
        if not contexts:
            contexts = ["No relevant data retrieved from graph database."]
        
        return {
            'answer': answer,
            'contexts': contexts,
            'cypher_query': result.get('cypher_query', ''),
            'query_results': query_results
        }
    
    except Exception as e:
        print(f"Error in retrieve_and_generate_answer: {e}")
        return {
            'answer': f"Error: {str(e)}",
            'contexts': [f"Error retrieving data: {str(e)}"],
            'cypher_query': '',
            'query_results': []
        }


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_evaluation_dataset_without_ground_truth():
    """
    Prepare dataset WITHOUT ground truth answers.
    Use this if you only have questions and want to evaluate retrieval/generation quality.
    
    Uses EvaluationDataset.json for questions only.
    Includes checkpointing: saves progress after every question so you can resume if interrupted.
    """
    # Load questions from EvaluationDataset.json
    json_path = Path(__file__).parent / "EvaluationDataset.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Extract just questions
    questions = [tc["question"] for tc in test_cases]
    
    # Checkpoint file path
    checkpoint_path = Path(__file__).parent / "evaluation_checkpoint_no_gt.json"
    
    # Try to load existing checkpoint
    if checkpoint_path.exists():
        print("\n" + "=" * 80)
        print("FOUND EXISTING CHECKPOINT")
        print("=" * 80)
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        print(f"Resuming from question {checkpoint_data['last_completed'] + 1}/{len(questions)}")
        print(f"Already completed: {checkpoint_data['last_completed']} questions")
        
        user_input = input("\nPress ENTER to resume, or type 'restart' to start over: ").strip().lower()
        if user_input == 'restart':
            print("Starting fresh evaluation...\n")
            data = {
                "question": [],
                "answer": [],
                "contexts": []
            }
            start_index = 0
        else:
            print("Resuming from checkpoint...\n")
            data = checkpoint_data['data']
            start_index = checkpoint_data['last_completed']
    else:
        print("No checkpoint found. Starting fresh evaluation...\n")
        data = {
            "question": [],
            "answer": [],
            "contexts": []
        }
        start_index = 0
    
    print(f"Processing {len(questions)} questions...")
    
    try:
        for i in range(start_index, len(questions)):
            question = questions[i]
            
            print(f"\n[{i+1}/{len(questions)}] Processing: {question[:60]}...")
            
            # Use GraphQueryTest to retrieve and generate
            result = retrieve_and_generate_answer(question)
            
            data["question"].append(question)
            data["answer"].append(result['answer'])
            data["contexts"].append(result['contexts'])
            
            # Save checkpoint after each question
            checkpoint = {
                "last_completed": i + 1,
                "total": len(questions),
                "data": data
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"✅ Checkpoint saved: {i+1}/{len(questions)} completed")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("INTERRUPTED BY USER")
        print("=" * 80)
        print(f"Progress saved! Completed {len(data['question'])}/{len(questions)} questions")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print("Run the script again to resume from where you left off.")
        print("=" * 80)
        sys.exit(0)
    
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        print(f"Progress saved! Completed {len(data['question'])}/{len(questions)} questions")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print("Run the script again to resume from where you left off.")
        print("=" * 80)
        raise
    
    # SUCCESS - all questions processed
    print(f"\n✅ All {len(questions)} questions processed successfully!")
    print(f"📁 Checkpoint saved at: {checkpoint_path}")
    print("(Checkpoint file kept for safety - you can delete it manually if desired)\n")
    
    return Dataset.from_dict(data), checkpoint_path


def prepare_evaluation_dataset_with_ground_truth():
    """
    Prepare dataset WITH ground truth answers from EvaluationDataset.json.
    Use this for comprehensive evaluation including answer correctness.
    
    Includes checkpointing: saves progress after every question so you can resume if interrupted.
    """
    # Load test cases from EvaluationDataset.json
    json_path = Path(__file__).parent / "EvaluationDataset.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # Checkpoint file path
    checkpoint_path = Path(__file__).parent / "evaluation_checkpoint.json"
    
    # Try to load existing checkpoint
    if checkpoint_path.exists():
        print("\n" + "=" * 80)
        print("FOUND EXISTING CHECKPOINT")
        print("=" * 80)
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        print(f"Resuming from question {checkpoint_data['last_completed'] + 1}/{len(test_cases)}")
        print(f"Already completed: {checkpoint_data['last_completed']} questions")
        
        user_input = input("\nPress ENTER to resume, or type 'restart' to start over: ").strip().lower()
        if user_input == 'restart':
            print("Starting fresh evaluation...\n")
            data = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": []
            }
            start_index = 0
        else:
            print("Resuming from checkpoint...\n")
            data = checkpoint_data['data']
            start_index = checkpoint_data['last_completed']
    else:
        print("No checkpoint found. Starting fresh evaluation...\n")
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        start_index = 0
    
    print(f"Processing {len(test_cases)} test cases...")
    
    try:
        for i in range(start_index, len(test_cases)):
            test_case = test_cases[i]
            question = test_case["question"]
            ground_truth = test_case["answer"]
            
            print(f"\n[{i+1}/{len(test_cases)}] Processing: {question[:60]}...")
            
            # Use GraphQueryTest to retrieve and generate
            result = retrieve_and_generate_answer(question)
            
            data["question"].append(question)
            data["answer"].append(result['answer'])
            data["contexts"].append(result['contexts'])
            data["ground_truth"].append(ground_truth)
            
            # Save checkpoint after each question
            checkpoint = {
                "last_completed": i + 1,
                "total": len(test_cases),
                "data": data
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"✅ Checkpoint saved: {i+1}/{len(test_cases)} completed")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("INTERRUPTED BY USER")
        print("=" * 80)
        print(f"Progress saved! Completed {len(data['question'])}/{len(test_cases)} questions")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print("Run the script again to resume from where you left off.")
        print("=" * 80)
        sys.exit(0)
    
    except Exception as e:
        print("\n\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        print(f"Progress saved! Completed {len(data['question'])}/{len(test_cases)} questions")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print("Run the script again to resume from where you left off.")
        print("=" * 80)
        raise
    
    # SUCCESS - all questions processed
    print(f"\n✅ All {len(test_cases)} questions processed successfully!")
    print(f"📁 Checkpoint saved at: {checkpoint_path}")
    print("(Checkpoint file kept for safety - you can delete it manually if desired)\n")
    
    return Dataset.from_dict(data), checkpoint_path


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


def analyze_results_by_query_type(results_df, test_cases):
    """
    OPTIONAL: Analyze results AFTER evaluation to identify patterns based on id ranges.
    
    EvaluationDataset.json question types by ID:
    - 1-15: Single metric lookups
    - 16-25: Temporal changes/trends
    - 26-35: Cross-company comparisons
    - 36-50: Income statement metrics
    - 51-60: Income statement trends
    - 61-70: Income statement comparisons
    - 71-85: Cash flow metrics
    - 86-95: Cash flow trends
    - 96-100: Cash flow comparisons
    """
    # Define query type mappings by ID
    def get_query_type(question_id):
        if 1 <= question_id <= 15:
            return "Balance Sheet - Single Metric"
        elif 16 <= question_id <= 25:
            return "Balance Sheet - Temporal"
        elif 26 <= question_id <= 35:
            return "Balance Sheet - Comparison"
        elif 36 <= question_id <= 50:
            return "Income Statement - Single Metric"
        elif 51 <= question_id <= 60:
            return "Income Statement - Temporal"
        elif 61 <= question_id <= 70:
            return "Income Statement - Comparison"
        elif 71 <= question_id <= 85:
            return "Cash Flow - Single Metric"
        elif 86 <= question_id <= 95:
            return "Cash Flow - Temporal"
        elif 96 <= question_id <= 100:
            return "Cash Flow - Comparison"
        else:
            return "Unknown"
    
    # Extract query types
    query_types = [get_query_type(tc['id']) for tc in test_cases]
    results_df['query_type'] = query_types
    
    print("\n" + "="*80)
    print("PERFORMANCE BY QUERY TYPE")
    print("="*80)
    
    # Calculate average metrics by query type
    grouped = results_df.groupby('query_type').agg({
        'faithfulness': ['mean', 'std', 'count'],
        'answer_relevancy': ['mean', 'std'],
        'answer_correctness': ['mean', 'std'],
        'context_precision': ['mean', 'std'],
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
        llm=ChatOpenAI(model="gpt-4o-mini"),
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
    
    # Configure LLM with longer timeout and retries
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        timeout=180,  # 3 minutes timeout
        max_retries=3,  # Retry failed requests
        request_timeout=180
    )
    
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=OpenAIEmbeddings(),
        raise_exceptions=False  # Don't stop evaluation on individual failures
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
    print("\nStep 1: Generating answers for all questions...")
    print("=" * 80)
    dataset, checkpoint_path = prepare_evaluation_dataset_with_ground_truth()
    
    # Save intermediate dataset as backup
    intermediate_path = Path(__file__).parent / "dataset_before_ragas.json"
    print(f"\n💾 Saving intermediate dataset to: {intermediate_path}")
    dataset_dict = {
        "question": list(dataset["question"]),
        "answer": list(dataset["answer"]),
        "contexts": list(dataset["contexts"]),
        "ground_truth": list(dataset["ground_truth"])
    }
    with open(intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
    print("✅ Dataset saved! You can rerun RAGAS evaluation from this file if needed.\n")
    
    # Keep checkpoint file for safety (don't delete)
    print(f"📁 Checkpoint also saved at: {checkpoint_path}")
    print("(Both files kept for safety - you can delete them manually after successful evaluation)\n")
    
    print("\nStep 2: Running RAGAS evaluation...")
    print("=" * 80)
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
    
    # Configure LLM with longer timeout and retries
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        timeout=180,  # 3 minutes timeout
        max_retries=3,  # Retry failed requests
        request_timeout=180
    )
    
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=OpenAIEmbeddings(),
        raise_exceptions=False  # Don't stop evaluation on individual failures
    )
    
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
    
    # OPTIONAL: Analyze by query type
    print("\n" + "=" * 80)
    print("ANALYZING PERFORMANCE BY QUERY TYPE...")
    print("=" * 80)
    
    with open(Path(__file__).parent / "EvaluationDataset.json", 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    type_analysis = analyze_results_by_query_type(results_df, test_cases)
    type_analysis.to_csv("query_type_analysis.csv")
    print("\nType analysis saved to: query_type_analysis.csv")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    # Close Neo4j driver
    try:
        from GraphQueryTest import driver
        driver.close()
        print("Neo4j driver closed successfully.")
    except Exception as e:
        print(f"Note: Could not close driver - {e}")