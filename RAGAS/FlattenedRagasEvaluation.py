"""
RAGAS Evaluation for Milvus Flattened 10-K Retrieval System
Evaluates the flattened document retrieval and answer generation pipeline
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

# Check dependencies
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

# RAGAS models
llm = ChatOpenAI(model="gpt-4o-mini", timeout=180, max_retries=3, request_timeout=180, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# File paths
OUTPUT_JSON = Path(__file__).parent / "flattened_dataset_before_ragas.json"
OUTPUT_CSV = Path(__file__).parent / "flattened_ragas_evaluation_results.csv"
CHECKPOINT_FILE = Path(__file__).parent / "ragas_evaluation_checkpoint.json"

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_or_generate_dataset():
    """
    Load existing flattened dataset or convert from MilvusFlattenedRetrieval output
    """
    
    print("\n" + "=" * 80)
    print("LOADING EVALUATION DATASET")
    print("=" * 80)
    print()
    
    # Check for pre-generated results from MilvusFlattenedRetrieval.py
    milvus_output = Path(__file__).resolve().parent.parent / "hybrid_retrieval_outputs" / "evaluation_dataset_results.json"
    
    if OUTPUT_JSON.exists():
        print(f"Found RAGAS-formatted dataset: {OUTPUT_JSON.name}")
        response = input("Use existing dataset? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Loading existing dataset...")
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = Dataset.from_dict(data)
            print(f"✓ Loaded {len(dataset)} questions from existing dataset")
            return dataset
    
    if milvus_output.exists():
        print(f"Found MilvusFlattenedRetrieval output: {milvus_output.name}")
        print("Converting to RAGAS format...")
        
        with open(milvus_output, 'r', encoding='utf-8') as f:
            milvus_results = json.load(f)
        
        # Convert from MilvusFlattenedRetrieval format to RAGAS format
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for result in milvus_results:
            data["question"].append(result.get("query", ""))
            data["answer"].append(result.get("answer", ""))
            data["contexts"].append(result.get("contexts", []))
            data["ground_truth"].append(result.get("ground_truth", ""))
        
        print(f"✓ Converted {len(data['question'])} results to RAGAS format")
        print()
        
        # Save converted dataset
        print(f"Saving converted dataset to {OUTPUT_JSON.name}...")
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved")
        print()
        
        dataset = Dataset.from_dict(data)
        return dataset
    
    print(f"❌ No dataset found!")
    print(f"\nExpected one of:")
    print(f"  1. {OUTPUT_JSON} (RAGAS-formatted dataset)")
    print(f"  2. {milvus_output} (MilvusFlattenedRetrieval output)")
    print(f"\nRun MilvusFlattenedRetrieval.py first to generate results")
    sys.exit(1)

# ============================================================================
# RAGAS EVALUATION
# ============================================================================

def save_checkpoint(results_df, dataset_size):
    """Save evaluation checkpoint"""
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": dataset_size,
        "results": results_df.to_dict('records')
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    print(f"✓ Checkpoint saved ({len(results_df)} questions evaluated)")

def load_checkpoint():
    """Load evaluation checkpoint if exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def run_ragas_evaluation(dataset):
    """Run RAGAS evaluation on the generated dataset with checkpoint support"""
    
    print("\n" + "=" * 80)
    print("RUNNING RAGAS EVALUATION")
    print("=" * 80)
    print()
    
    print(f"Dataset size: {len(dataset)} questions")
    print()
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint and checkpoint.get('dataset_size') == len(dataset):
        print(f"Found checkpoint from {checkpoint['timestamp']}")
        print(f"Completed evaluation with {len(checkpoint['results'])} questions")
        response = input("Use cached results? (y/n): ").strip().lower()
        
        if response == 'y':
            print("Loading cached results...")
            import pandas as pd
            results_df = pd.DataFrame(checkpoint['results'])
            
            # Display summary statistics
            print()
            print("SUMMARY STATISTICS:")
            print("-" * 80)
            print(results_df[['faithfulness', 'answer_relevancy', 'context_precision,' 'context_recall', 
                             'answer_correctness', 'answer_similarity']].describe())
            print()
            
            # Display metric averages
            print("METRIC AVERAGES:")
            print("-" * 80)
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision',  
                           'context_recall', 'answer_correctness', 'answer_similarity']:
                avg = results_df[metric].mean()
                print(f"  {metric:20s}: {avg:.4f}")
            print()
            
            return results_df
        else:
            print("Discarding checkpoint and starting fresh...")
            if CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()
    elif checkpoint:
        print(f"Found checkpoint but dataset size changed ({checkpoint.get('dataset_size')} → {len(dataset)})")
        print("Starting fresh evaluation...")
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
    
    # Show sample
    print("Sample data:")
    print(f"  Question: {dataset[0]['question'][:60]}...")
    print(f"  Answer: {dataset[0]['answer'][:60]}...")
    print(f"  Contexts: {len(dataset[0]['contexts'])} retrieved")
    print(f"  Ground truth: {dataset[0]['ground_truth'][:60]}...")
    print()
    
    # Run evaluation
    print("Running RAGAS metrics...")
    print("This may take several minutes depending on dataset size...")
    print()
    
    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
                answer_similarity
            ],
            llm=llm,
            embeddings=embeddings
        )
        
        print()
        print("=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print()
        
        # Convert to DataFrame
        results_df = result.to_pandas()
        
        # Rename columns to match RagasEvaluationRerun.py expectations
        results_df.rename(columns={
            'question': 'user_input',
            'answer': 'response',
            'ground_truth': 'reference',
            'contexts': 'retrieved_contexts'
        }, inplace=True)
        
        # Add question_id
        results_df.insert(0, 'question_id', range(1, len(results_df) + 1))
        
        # Save checkpoint immediately after successful evaluation
        save_checkpoint(results_df, len(dataset))
        print()
        
        # Display summary statistics
        print("SUMMARY STATISTICS:")
        print("-" * 80)
        print(results_df[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 
                          'answer_correctness', 'answer_similarity']].describe())
        print()
        
        # Display metric averages
        print("METRIC AVERAGES:")
        print("-" * 80)
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 
                       'answer_correctness', 'answer_similarity']:
            avg = results_df[metric].mean()
            print(f"  {metric:20s}: {avg:.4f}")
        print()
        
        # Save results
        print(f"Saving results to {OUTPUT_CSV.name}...")
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✓ Results saved")
        print()
        
        print("=" * 80)
        print("FILE LOCATIONS:")
        print("=" * 80)
        print(f"  Dataset (JSON): {OUTPUT_JSON}")
        print(f"  Results (CSV):  {OUTPUT_CSV}")
        print(f"  Checkpoint:     {CHECKPOINT_FILE}")
        print()
        
        return results_df
    
    except Exception as e:
        print(f"\n❌ ERROR during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("\nEvaluation failed. Run the script again to retry.")
        print("Note: RAGAS evaluates all questions in batch, so partial progress cannot be saved.")
        sys.exit(1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution flow"""
    
    print("\n" + "=" * 80)
    print("FLATTENED MILVUS RAGAS EVALUATION PIPELINE")
    print("=" * 80)
    print()
    
    # Step 1: Load or convert dataset
    dataset = load_or_generate_dataset()
    
    # Step 2: Run RAGAS evaluation
    results_df = run_ragas_evaluation(dataset)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved to: {OUTPUT_CSV}")
    print()

if __name__ == "__main__":
    main()
