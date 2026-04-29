"""
Batch Document Flattening for Milvus Ingestion

This script processes all 10-K financial tables in the edgar-tools directory,
converting them to sentence-based natural language with rich metadata for
Milvus vector database ingestion.

Usage:
    python batch_flatten_for_milvus.py
"""

import json
import os
import sys
from pathlib import Path

# Add Graph directory to path BEFORE importing
script_dir = Path(__file__).resolve().parent
graph_dir = script_dir / "Graph"
sys.path.insert(0, str(graph_dir))

# Now import after path is set
from document_flattener import FinancialTableFlattener, BatchFlattener, FlattenConfig
from company_data import company_data


def load_company_mapping():
    """
    Load company ticker to name mapping from company_data.py.
    Includes special handling for GOOGL -> Google instead of Alphabet.
    """
    mapping = company_data.copy()
    
    # Special case: GOOGL -> Use "Google" instead of "Alphabet Inc."
    # No one says Alphabet
    if "GOOGL" in mapping:
        mapping["GOOGL"] = "Google"
    
    print(f"✓ Loaded {len(mapping)} company names from company_data.py")
    print(f"  Special handling: GOOGL -> 'Google' (instead of 'Alphabet Inc.')")
    
    return mapping


def enhance_records_with_company_names(records, company_mapping):
    """
    Add proper company names to records and update sentence text.
    Handles both simple and detailed sentence formats.
    """
    for record in records:
        ticker = record['ticker'].upper()
        
        if ticker in company_mapping:
            company_name = company_mapping[ticker]
            record['company_name'] = company_name
            
            # Update text based on format
            original_text = record['text']
            ticker_lower = ticker.lower()
            
            # Detailed format: "ticker (ticker) reported..." -> "Company Name (TICKER) reported..."
            if "according to" in original_text:
                parts = original_text.split('(')
                if len(parts) > 1:
                    # Replace "ticker (ticker)" with "Company Name (TICKER)"
                    after_ticker = original_text.split(')', 1)[1]
                    record['text'] = f"{company_name} ({ticker}){after_ticker}"
            
            # Simple format: "In YYYY, ticker reported..." -> "In YYYY, Company Name reported..."
            elif original_text.startswith("In "):
                # Pattern: "In 2024, ticker reported..." or "In 2024, TICKER reported..."
                parts = original_text.split(', ', 1)
                if len(parts) == 2:
                    year_part = parts[0]  # "In 2024"
                    rest = parts[1]  # "ticker reported..." or "TICKER reported..."
                    
                    # Replace ticker (case-insensitive) with company name
                    if rest.startswith(f"{ticker} reported"):
                        rest = rest.replace(f"{ticker} reported", f"{company_name} reported", 1)
                        record['text'] = f"{year_part}, {rest}"
                    elif rest.startswith(f"{ticker_lower} reported"):
                        rest = rest.replace(f"{ticker_lower} reported", f"{company_name} reported", 1)
                        record['text'] = f"{year_part}, {rest}"
    
    return records


def main():
    """Main execution function"""
    
    print("="*80)
    print("BATCH DOCUMENT FLATTENING FOR MILVUS")
    print("="*80)
    print()
    
    # Load company name mapping
    company_mapping = load_company_mapping()
    print()
    
    # Configuration - using detailed sentences with billions format
    config = FlattenConfig(
        output_format="detailed_sentence",  # Use detailed format for better context
        format_style="billions",            # "29.94 billion" instead of "29,943,000,000"
        include_concepts=True,              # Include GAAP concept codes
        null_handling="skip"                # Skip null values
    )
    
    print("Configuration:")
    print(f"  Output format: {config.output_format}")
    print(f"  Number style: {config.format_style}")
    print(f"  Include concepts: {config.include_concepts}")
    print(f"  Null handling: {config.null_handling}")
    print()
    
    # Initialize batch flattener
    batch_flattener = BatchFlattener(config)
    
    # Define directories
    root_dir = r"c:\Users\Cedric\Desktop\RAG THESIS\RAG_Framework\edgar-tools\10-K"
    output_dir = r"c:\Users\Cedric\Desktop\RAG THESIS\RAG_Framework\edgar-tools\flattened_json"
    
    print(f"Input directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    if not os.path.exists(root_dir):
        print(f"❌ ERROR: Input directory not found: {root_dir}")
        return
    
    # Process all multi_*.json files
    print("Starting batch processing...")
    print("="*80)
    print()
    
    results = batch_flattener.flatten_directory(
        root_dir=root_dir,
        output_dir=output_dir,
        file_pattern="*multi*.json"
    )
    
    print()
    print("="*80)
    print("Enhancing records with company names...")
    print("="*80)
    print()
    
    # Post-process all output files to add proper company names
    enhanced_count = 0
    for output_path in results.values():
        if output_path:
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                
                # Enhance with company names
                records = enhance_records_with_company_names(records, company_mapping)
                
                # Save back
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                
                enhanced_count += 1
            except Exception as e:
                print(f"⚠ Could not enhance {output_path}: {e}")
    
    print(f"✓ Enhanced {enhanced_count} output files with company names")
    print()
    
    # Generate summary report
    print("="*80)
    print("Generating summary report...")
    print("="*80)
    print()
    
    summary = batch_flattener.generate_summary_report(
        results, 
        output_file=os.path.join(output_dir, "flattening_summary.txt")
    )
    
    print(summary)
    
    # Additional statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    total_records = 0
    total_files = 0
    
    for output_path in results.values():
        if output_path and os.path.exists(output_path):
            total_files += 1
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                    total_records += len(records)
            except:
                pass
    
    print(f"\nTotal files processed: {total_files}")
    print(f"Total records generated: {total_records:,}")
    
    if total_records > 0:
        print(f"\nAverage records per file: {total_records/total_files:.1f}")
        
        estimated_size_mb = (total_records * 1.5) / 1024
        print(f"Estimated size: ~{estimated_size_mb:.1f} MB")
    
    print("\n" + "="*80)
    print("✓ BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutput location: {output_dir}")
    print("\nNext steps:")
    print("1. Review the generated JSON files in the output directory")
    print("2. Create a Milvus collection with appropriate schema")
    print("3. Use the JSON files to insert data into Milvus")
    print("4. Configure vector embedding for the 'text' field")
    print()


if __name__ == "__main__":
    main()