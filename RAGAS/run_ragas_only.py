import os
import json
import sys
from pathlib import Path
from datetime import datetime

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

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


def analyze_results_by_query_type(results_df, test_cases):
    """
    Analyze results by question type based on ID ranges.
    """
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
# LOAD EXISTING DATASET
# ============================================================================

def load_dataset():
    """
    Load existing dataset from checkpoint or intermediate file.
    Priority: evaluation_checkpoint.json > dataset_before_ragas.json
    """
    checkpoint_path = Path(__file__).parent / "evaluation_checkpoint.json"
    intermediate_path = Path(__file__).parent / "dataset_before_ragas.json"
    
    if checkpoint_path.exists():
        print(f"✅ Found checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        data = checkpoint_data['data']
        print(f"Loaded {len(data['question'])} questions from checkpoint")
        print(f"Checkpoint shows: {checkpoint_data['last_completed']}/{checkpoint_data['total']} completed")
        
        if len(data['question']) < checkpoint_data['total']:
            print("\n⚠️  WARNING: Checkpoint is incomplete!")
            print(f"Only {len(data['question'])} out of {checkpoint_data['total']} questions were processed.")
            user_input = input("Continue with partial dataset? (yes/no): ").strip().lower()
            if user_input != 'yes':
                print("Aborted. Please complete answer generation first.")
                sys.exit(0)
        
        return Dataset.from_dict(data)
    
    elif intermediate_path.exists():
        print(f"✅ Found intermediate dataset: {intermediate_path}")
        with open(intermediate_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['question'])} questions from intermediate file")
        return Dataset.from_dict(data)
    
    else:
        print("\n" + "=" * 80)
        print("ERROR: NO EXISTING DATASET FOUND")
        print("=" * 80)
        print("Could not find:")
        print(f"  - {checkpoint_path}")
        print(f"  - {intermediate_path}")
        print("\nPlease run RagasEvaluation.py first to generate answers.")
        print("=" * 80)
        sys.exit(1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RAGAS Evaluation ONLY (Using Existing Dataset)")
    print("=" * 80)
    
    # Load existing dataset
    print("\nLoading existing dataset...")
    dataset = load_dataset()
    
    print(f"\nDataset loaded successfully!")
    print(f"  - Questions: {len(dataset['question'])}")
    print(f"  - Has ground truth: {'ground_truth' in dataset.column_names}")
    print(f"  - Columns: {dataset.column_names}")
    
    # Verify dataset structure
    sample = dataset[0]
    print(f"\nSample data:")
    print(f"  Question: {sample['question'][:80]}...")
    print(f"  Answer: {sample['answer'][:80]}...")
    print(f"  Contexts: {len(sample['contexts'])} context(s)")
    if 'ground_truth' in sample:
        print(f"  Ground truth: {sample['ground_truth'][:80]}...")
    
    input("\nPress ENTER to start RAGAS evaluation...")
    
    # Run RAGAS evaluation
    print("\n" + "=" * 80)
    print("RUNNING RAGAS EVALUATION")
    print("=" * 80)
    print("This may take 30-60 minutes depending on API speed...")
    print("Warnings about 'LLM returned 1 generation' are normal.\n")
    
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
    output_path = Path(__file__).parent / "ragas_evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    summary_stats = {}
    for metric in results.keys():
        if metric not in ['question', 'answer', 'contexts', 'ground_truth']:
            value = results[metric]
            print(f"{metric}: {value:.4f}")
            summary_stats[metric] = value
    
    # Analyze by query type
    print("\n" + "=" * 80)
    print("ANALYZING PERFORMANCE BY QUERY TYPE...")
    print("=" * 80)
    
    try:
        json_path = Path(__file__).parent / "EvaluationDataset.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            
            # Add query IDs to results based on order (1-100)
            results_df['query_id'] = range(1, len(results_df) + 1)
            
            type_analysis = analyze_results_by_query_type(results_df, test_cases)
            type_analysis_path = Path(__file__).parent / "query_type_analysis.csv"
            type_analysis.to_csv(type_analysis_path)
            print(f"\n✅ Type analysis saved to: {type_analysis_path}")
        else:
            print(f"⚠️  Warning: {json_path} not found. Skipping type analysis.")
    except Exception as e:
        print(f"⚠️  Error during type analysis: {str(e)}")
        print("Continuing with main results...")
    
    # Save summary report
    print("\n" + "=" * 80)
    print("SAVING SUMMARY REPORT")
    print("=" * 80)
    
    summary_report_path = Path(__file__).parent / "ragas_evaluation_summary.txt"
    with open(summary_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAGAS EVALUATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Questions Evaluated: {len(results_df)}\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        for metric, value in summary_stats.items():
            f.write(f"{metric:.<40} {value:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS BY QUESTION\n")
        f.write("=" * 80 + "\n\n")
        
        # Write detailed results
        for idx, row in results_df.iterrows():
            f.write(f"Question {idx + 1}\n")
            f.write(f"  Question: {row['question'][:100]}...\n")
            f.write(f"  Answer: {row['answer'][:100]}...\n")
            if 'faithfulness' in row:
                f.write(f"  Faithfulness: {row['faithfulness']:.4f}\n")
            if 'answer_relevancy' in row:
                f.write(f"  Answer Relevancy: {row['answer_relevancy']:.4f}\n")
            if 'context_precision' in row:
                f.write(f"  Context Precision: {row['context_precision']:.4f}\n")
            if 'context_recall' in row:
                f.write(f"  Context Recall: {row['context_recall']:.4f}\n")
            if 'answer_correctness' in row:
                f.write(f"  Answer Correctness: {row['answer_correctness']:.4f}\n")
            if 'answer_similarity' in row:
                f.write(f"  Answer Similarity: {row['answer_similarity']:.4f}\n")
            f.write("\n")
    
    print(f"✅ Summary report saved to: {summary_report_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {summary_report_path}")
    if Path(type_analysis_path).exists():
        print(f"  - {type_analysis_path}")
