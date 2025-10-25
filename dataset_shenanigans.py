"""
Script to extract CIK from relevant_node, map to ticker, and prepend to questions.

Example transformation:
    Original: "Why is the availability of cloud computing services critical..."
    Result:   "(INTU) Why is the availability of cloud computing services critical..."
"""

import json
import os
import re
from pathlib import Path
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG")

# Load CIK to ticker mapping
def load_cik_to_ticker_mapping(mapping_file="company_tickers_exchange.json"):
    """Load CIK to ticker mapping from JSON file."""
    if not os.path.exists(mapping_file):
        print(f"Error: {mapping_file} not found")
        return {}
    
    with open(mapping_file, 'r') as f:
        ticker_data = json.load(f)
    
    cik_to_ticker = {}
    if isinstance(ticker_data, dict) and 'data' in ticker_data:
        for entry in ticker_data['data']:
            if isinstance(entry, list) and len(entry) >= 3:
                cik = str(entry[0])  # CIK without padding
                ticker = entry[2]  # Ticker
                if ticker and cik:
                    cik_to_ticker[cik] = ticker
    
    print(f"Loaded {len(cik_to_ticker)} CIK-to-ticker mappings")
    return cik_to_ticker


def extract_cik_from_node(relevant_node):
    """
    Extract CIK from relevant_node.
    
    Expected format: "dense-43-896878_10K_2023_0000896878-23-000034.json"
    CIK is the first part after splitting by underscore: "896878"
    
    Args:
        relevant_node: List of node identifiers or single string
        
    Returns:
        str: Extracted CIK (without padding) or None
    """
    if not relevant_node:
        return None
    
    # Handle if it's a list, take the first element
    node_str = relevant_node[0] if isinstance(relevant_node, list) else relevant_node
    
    # Extract filename from the node string
    # Format: dense-43-896878_10K_2023_0000896878-23-000034.json
    # We want the CIK which appears after the chunk number
    
    # Split by '-' and find the part with underscore (the filename part)
    parts = node_str.split('-')
    
    for part in parts:
        if '_' in part:
            # This should be something like "896878_10K_2023_0000896878"
            filename_part = part.split('_')[0]
            # Remove any non-digit characters
            cik = re.sub(r'\D', '', filename_part)
            if cik:
                return cik
    
    return None


def process_json_file(file_path, cik_to_ticker, dry_run=False):
    """
    Process a single JSON or JSONL file and prepend ticker to questions.
    
    Args:
        file_path: Path to the JSON/JSONL file
        cik_to_ticker: Dictionary mapping CIK to ticker
        dry_run: If True, only print changes without saving
        
    Returns:
        int: Number of questions modified
    """
    try:
        file_path = Path(file_path)
        
        # Handle JSONL files (multiple JSON objects, one per line)
        if file_path.suffix == '.jsonl':
            data_list = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        data_list.append(json.loads(line))
        else:
            # Handle regular JSON files
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data_list = data if isinstance(data, list) else [data]
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0
    
    modified_count = 0
    
    for item in data_list:
        if not isinstance(item, dict):
            continue
        
        # Check if question and relevant_node exist
        if 'question' not in item or 'relevant_node' not in item:
            continue
        
        question = item['question']
        relevant_node = item['relevant_node']
        
        # Skip if question already has ticker in parentheses at the start
        if re.match(r'^\([A-Z]+\)\s', question):
            continue
        
        # Extract CIK from relevant_node
        cik = extract_cik_from_node(relevant_node)
        
        if not cik:
            print(f"Warning: Could not extract CIK from {relevant_node}")
            continue
        
        # Map CIK to ticker
        ticker = cik_to_ticker.get(cik)
        
        if not ticker:
            print(f"Warning: No ticker found for CIK {cik}")
            continue
        
        # Prepend ticker to question
        new_question = f"({ticker}) {question}"
        item['question'] = new_question
        modified_count += 1
        
        if dry_run:
            print(f"\nOriginal: {question}")
            print(f"Modified: {new_question}")
    
    # Save the modified data if not dry run
    if not dry_run and modified_count > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # Write JSONL format (one JSON object per line)
                    for item in data_list:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                else:
                    # Write regular JSON format
                    if isinstance(data, list):
                        json.dump(data_list, f, indent=2, ensure_ascii=False)
                    else:
                        json.dump(data_list[0], f, indent=2, ensure_ascii=False)
            print(f" Saved {file_path}: {modified_count} questions modified")
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
    
    return modified_count


def process_directory(directory_path, cik_to_ticker, dry_run=False):
    """
    Process all JSON and JSONL files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory containing JSON files
        cik_to_ticker: Dictionary mapping CIK to ticker
        dry_run: If True, only print changes without saving
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory_path}")
        return
    
    # Recursively find all JSON and JSONL files in subdirectories
    json_files = list(directory.rglob("*.json"))
    jsonl_files = list(directory.rglob("*.jsonl"))
    all_files = json_files + jsonl_files
    
    if not all_files:
        print(f"No JSON/JSONL files found in {directory_path} or its subdirectories")
        return
    
    print(f"\nFound {len(all_files)} JSON/JSONL files to process (including subdirectories)")
    print("=" * 80)
    
    total_modified = 0
    
    for json_file in all_files:
        print(f"Processing: {json_file.relative_to(directory)}")
        modified = process_json_file(json_file, cik_to_ticker, dry_run)
        total_modified += modified
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"Total questions modified: {total_modified}")
    if dry_run:
        print("\n(DRY RUN - No files were actually modified)")


def main():
    """Main function to process JSON files and append tickers to questions."""
    
    # Path to the JSON files directory
    json_directory = r"C:\Users\Rhenz\Documents\School\CodeFolders\Thesis\Omni\OmniEval\data_generator\gen_datas_test_final"
    #json_directory = r"C:\Users\Cedric\Desktop\RAG THESIS\RAG_Framework\test"

    # Load CIK to ticker mapping
    print("Loading CIK to ticker mapping...")
    cik_to_ticker = load_cik_to_ticker_mapping()
    
    if not cik_to_ticker:
        print("Error: Could not load CIK to ticker mapping")
        return
    
    # Process the directory
    # Set dry_run=True to preview changes without modifying files
    # Set dry_run=False to actually modify the files
    process_directory(json_directory, cik_to_ticker, dry_run=False)
    
    # Uncomment the line below to actually modify the files
    # process_directory(json_directory, cik_to_ticker, dry_run=False)


if __name__ == "__main__":
    main()
