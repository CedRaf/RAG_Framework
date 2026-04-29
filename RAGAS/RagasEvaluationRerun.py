import os
import json
import sys
import csv
import time
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
    import pandas as pd
except ImportError as e:
    print("=" * 80)
    print("MISSING DEPENDENCIES")
    print("=" * 80)
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("pip install ragas datasets langchain-openai pandas")
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

# Checkpoint file to track progress
CHECKPOINT_FILE = Path(__file__).parent / "rerun_checkpoint.json"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_checkpoint():
    """Load checkpoint of completed evaluations"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"completed_indices": []}

def save_checkpoint(completed_indices):
    """Save checkpoint of completed evaluations"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"completed_indices": completed_indices}, f, indent=2)

def find_failed_rows(csv_path):
    """
    Find rows with NaN values in any RAGAS metric columns.
    
    Returns:
        list of dicts containing: index, question, answer, contexts, ground_truth, 
        and which metric(s) had NaN
    """
    df = pd.read_csv(csv_path)
    
    # All RAGAS metrics to check
    metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 
               'context_precision', 'answer_correctness', 'answer_similarity']
    
    # Find rows with NaN in any metric
    nan_mask = df[metrics].isna().any(axis=1)
    failed_rows = df[nan_mask]
    
    results = []
    for idx, row in failed_rows.iterrows():
        failed_metrics = []
        for metric in metrics:
            if pd.isna(row[metric]):
                failed_metrics.append(metric)
        
        results.append({
            'csv_index': idx,
            'question': row['user_input'],
            'answer': row['response'],
            'contexts': ast.literal_eval(row['retrieved_contexts']) if isinstance(row['retrieved_contexts'], str) else row['retrieved_contexts'],
            'ground_truth': row['reference'],
            'failed_metrics': failed_metrics
        })
    
    return results, df


def rerun_failed_evaluations(failed_rows, original_df):
    """
    Rerun RAGAS evaluation ONE AT A TIME for all metrics on the failed rows.
    Includes checkpoint support to resume on API errors.
    
    Returns:
        dict mapping csv_index to new metric results
    """
    print(f"\n{'='*80}")
    print(f"RERUNNING FAILED EVALUATIONS (ONE AT A TIME)")
    print(f"{'='*80}")
    print(f"Found {len(failed_rows)} questions with NaN values")
    print(f"Evaluating each question individually with checkpointing")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_indices = set(checkpoint["completed_indices"])
    
    if completed_indices:
        print(f"📍 Resuming from checkpoint: {len(completed_indices)} already completed\n")
    
    # Configure LLM with longer timeout and retries
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        timeout=180,
        max_retries=3,
        request_timeout=180,
        api_key=OPENAI_API_KEY
    )
    
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Evaluate all RAGAS metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
        answer_similarity
    ]
    
    # Store results for each row
    all_results = {}
    
    for i, row_info in enumerate(failed_rows):
        csv_idx = row_info['csv_index']
        
        # Skip if already completed
        if csv_idx in completed_indices:
            print(f"[{i+1}/{len(failed_rows)}] ⏭️  Skipping row {csv_idx} (already completed)")
            continue
        
        print(f"[{i+1}/{len(failed_rows)}] 🔄 Evaluating row {csv_idx}: '{row_info['question'][:50]}...'")
        print(f"   Failed metrics: {', '.join(row_info['failed_metrics'])}")
        
        # Create single-question dataset
        dataset_dict = {
            "question": [row_info['question']],
            "answer": [row_info['answer']],
            "contexts": [row_info['contexts']],
            "ground_truth": [row_info['ground_truth']]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        try:
            # Evaluate this one question
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False
            )
            
            # Store results
            result_df = result.to_pandas()
            all_results[csv_idx] = result_df.iloc[0].to_dict()
            
            # Mark as completed and save checkpoint
            completed_indices.add(csv_idx)
            save_checkpoint(list(completed_indices))
            
            print(f"   ✅ Success")
            
            # Show the new metric values
            for metric in row_info['failed_metrics']:
                value = all_results[csv_idx][metric]
                print(f"      {metric}: {value:.4f}")
            print()
            
            # Brief delay to avoid rate limiting (if not the last question)
            if i < len(failed_rows) - 1:
                time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   Checkpoint saved. Rerun script to continue from here.")
            print()
            
            # Return partial results
            if all_results:
                print(f"\n⚠️  Returning partial results ({len(all_results)}/{len(failed_rows)} completed)")
                return all_results
            else:
                return None
    
    print(f"\n✅ All {len(failed_rows)} questions evaluated successfully!")
    
    # Clean up checkpoint on full success
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"   Checkpoint file cleaned up")
    
    return all_results


def update_csv_with_results(csv_path, failed_rows, new_results, output_path=None):
    """
    Update the original CSV with the new metric values from rerun.
    
    Args:
        csv_path: Path to original CSV
        failed_rows: List of failed row dicts
        new_results: Dict mapping csv_index to metric results
        output_path: Path to save updated CSV (defaults to csv_path if None)
    """
    if output_path is None:
        output_path = csv_path
    
    # Read original CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*80}")
    print(f"UPDATING CSV WITH NEW RESULTS")
    print(f"{'='*80}\n")
    
    if new_results is None:
        print("No results to update (evaluation failed)")
        return None
    
    # All RAGAS metrics
    metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 
               'context_precision', 'answer_correctness', 'answer_similarity']
    
    # Update each completed row in the original dataframe
    updated_count = 0
    for row_info in failed_rows:
        csv_idx = row_info['csv_index']
        
        # Check if we have results for this row
        if csv_idx not in new_results:
            continue
        
        print(f"Row {csv_idx}: '{row_info['question'][:50]}...'")
        
        # Update all metrics
        for metric in metrics:
            new_value = new_results[csv_idx][metric]
            df.loc[csv_idx, metric] = new_value
            
            # Only print if this metric was originally NaN
            if metric in row_info['failed_metrics']:
                print(f"  → {metric}: {new_value:.4f}")
        
        print()
        updated_count += 1
    
    # Save updated CSV
    df.to_csv(output_path, index=False)
    print(f"{'='*80}")
    print(f"✅ Updated CSV saved to: {output_path}")
    print(f"   Updated {updated_count} rows")
    print(f"{'='*80}\n")
    
    return df


def display_comparison(original_df, updated_df, failed_rows, new_results):
    """
    Display before/after comparison for updated rows.
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: BEFORE vs AFTER")
    print(f"{'='*80}\n")
    
    metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 
               'context_precision', 'answer_correctness', 'answer_similarity']
    
    for row_info in failed_rows:
        csv_idx = row_info['csv_index']
        
        # Skip if no results for this row
        if csv_idx not in new_results:
            continue
        
        question = row_info['question'][:60]
        
        print(f"Question {csv_idx}: {question}...")
        print(f"  Failed metrics: {', '.join(row_info['failed_metrics'])}")
        
        # Show before/after for failed metrics only
        for metric in row_info['failed_metrics']:
            before_val = original_df.loc[csv_idx, metric]
            after_val = updated_df.loc[csv_idx, metric]
            
            before_str = f"{before_val:.4f}" if pd.notna(before_val) else "NaN"
            after_str = f"{after_val:.4f}" if pd.notna(after_val) else "NaN"
            
            print(f"    {metric}: {before_str} → {after_str}")
        
        print()
    
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import ast
    
    print("=" * 80)
    print("RAGAS Evaluation Rerun")
    print("=" * 80)
    # Change file path to rerun failed rows. 
    csv_path = Path(__file__).parent / "flattened_ragas_evaluation_results.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    print(f"\n📁 Reading results from: {csv_path}\n")
    
    # Step 1: Find failed rows
    print("Step 1: Identifying questions with NaN metrics...")
    print("-" * 80)
    
    failed_rows, original_df = find_failed_rows(csv_path)
    
    if not failed_rows:
        print("✅ No failed rows found! All questions have been evaluated successfully.")
        sys.exit(0)
    
    print(f"Found {len(failed_rows)} questions that need rerun:\n")
    
    for row_info in failed_rows:
        failed_metrics_str = ", ".join(row_info['failed_metrics'])
        print(f"  • {row_info['question'][:65]}... [{failed_metrics_str}]")
    
    # Step 2: Confirm with user
    print(f"\n{'-'*80}")
    user_confirm = input("\nProceed with rerun? (yes/no): ").strip().lower()
    
    if user_confirm not in ['yes', 'y']:
        print("Cancelled by user.")
        sys.exit(0)
    
    # Step 3: Rerun evaluation
    print("\nStep 2: Running RAGAS evaluation on failed metrics...")
    print("-" * 80)
    
    new_results = rerun_failed_evaluations(failed_rows, original_df)
    
    if new_results is None:
        print("❌ Evaluation failed. No updates made.")
        sys.exit(1)
    
    # Step 4: Update CSV
    print("\nStep 3: Updating CSV with new results...")
    print("-" * 80)
    
    updated_df = update_csv_with_results(csv_path, failed_rows, new_results)
    
    if updated_df is not None:
        # Step 5: Display comparison
        display_comparison(original_df, updated_df, failed_rows, new_results)
        
        # Calculate summary statistics
        print(f"{'='*80}")
        print(f"SUMMARY STATISTICS - UPDATED METRICS")
        print(f"{'='*80}\n")
        
        metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 
                   'context_precision', 'answer_correctness', 'answer_similarity']
        
        for metric in metrics:
            values = updated_df[metric].dropna()
            print(f"{metric}:")
            print(f"  - Mean: {values.mean():.4f}")
            print(f"  - Std Dev: {values.std():.4f}")
            print(f"  - Min: {values.min():.4f}")
            print(f"  - Max: {values.max():.4f}")
            print()
        
        # Check if any NaN values remain
        print(f"Remaining NaN values:")
        for metric in metrics:
            remaining_nan = updated_df[metric].isna().sum()
            print(f"  - {metric}: {remaining_nan}")
        
        total_nan = sum(updated_df[metric].isna().sum() for metric in metrics)
        
        if total_nan == 0:
            print("\n✅ All metrics have been successfully evaluated!")
        else:
            print(f"\n⚠️  Still have {total_nan} total NaN values across all metrics")
            print("   These may require manual investigation or additional API retries")
        
        print(f"\n{'='*80}")
        print("RERUN COMPLETE")
        print(f"{'='*80}\n")
    
    sys.exit(0)
