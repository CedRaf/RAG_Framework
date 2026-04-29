"""
Efficient document flattening script for 10-K financial tables.
Converts nested JSON financial data into sentence-based format optimized for vector search.

Strategy:
1. Sentence-based narrative format (natural language)
2. JSON output with metadata for Milvus ingestion
3. Efficient memory usage (streaming, not loading all at once)
4. Reusable for all table types (BalanceSheet, IncomeStatement, CashFlow)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class FlattenConfig:
    """Configuration for flattening behavior"""
    output_format: str = "sentence"  # "sentence" or "detailed_sentence"
    include_concepts: bool = True     # Include GAAP concept codes
    null_handling: str = "skip"       # "skip", "zero", or "n/a"
    format_style: str = "billions"    # "billions", "full", or "abbreviated"


class FinancialTableFlattener:
    """Efficiently flattens financial tables from JSON to sentence-based format for Milvus"""
    
    def __init__(self, config: FlattenConfig = None):
        self.config = config or FlattenConfig()
    
    def format_number(self, value: Any, year: str = "") -> tuple:
        """
        Format numbers for readability.
        Returns: (full_number, readable_format, unit)
        """
        if value is None or value == "":
            if self.config.null_handling == "skip":
                return None, None, None
            return "0", "0", "USD"
        
        if isinstance(value, str):
            return value, value, "USD"
        
        if not isinstance(value, (int, float)):
            return str(value), str(value), "USD"
        
        # For style selection
        if self.config.format_style == "billions":
            # 29.94 billion USD
            if abs(value) >= 1_000_000_000:
                readable = f"{value/1_000_000_000:.2f} billion"
                unit = "USD"
            elif abs(value) >= 1_000_000:
                readable = f"{value/1_000_000:.2f} million"
                unit = "USD"
            else:
                readable = f"{value:,.0f}"
                unit = "USD"
        elif self.config.format_style == "full":
            # 29,943,000,000 USD
            readable = f"{value:,.0f}"
            unit = "USD"
        else:  # abbreviated
            # $29.94B
            if abs(value) >= 1_000_000_000:
                readable = f"${value/1_000_000_000:.2f}B"
                unit = ""
            elif abs(value) >= 1_000_000:
                readable = f"${value/1_000_000:.2f}M"
                unit = ""
            else:
                readable = f"${value:,.0f}"
                unit = ""
        
        full_number = f"{value:,.0f}"
        return full_number, readable, unit
    
    def extract_years(self, row: Dict) -> List[str]:
        """Extract years from a financial row (dates as keys)"""
        exclude_fields = {"label", "concept"}
        years = [k for k in row.keys() if k not in exclude_fields]
        return sorted(years, reverse=True)  # Most recent first
    
    def parse_fiscal_year(self, date_str: str) -> str:
        """Extract fiscal year from date string (e.g., '2024-09-28' -> '2024')"""
        return date_str.split('-')[0]
    
    def generate_sentence(self, ticker: str, company_name: str, label: str, 
                         value: Any, year: str, concept: str, 
                         table_type: str) -> Dict:
        """
        Generate a sentence-based record for Milvus insertion.
        
        Returns dict with: id, text, ticker, concept, year, table_type, etc.
        """
        full_number, readable, unit = self.format_number(value, year)
        
        if full_number is None:  # Skip if null handling says so
            return None
        
        fiscal_year = self.parse_fiscal_year(year)
        
        # Choose output format
        if self.config.output_format == "sentence":
            # Simple format: "In 2024, Apple reported 29,943,000,000 USD in Cash and Cash Equivalents."
            if unit:
                text = f"In {fiscal_year}, {ticker} reported {full_number} {unit} in {label}."
            else:
                text = f"In {fiscal_year}, {ticker} reported {readable} in {label}."
                
        else:  # detailed_sentence
            # Detailed format: "Apple Inc. (AAPL) reported 29.94 billion USD in Cash and Cash Equivalents for fiscal year 2024, according to the Balance Sheet."
            if unit:
                text = f"{company_name} ({ticker}) reported {readable} {unit} in {label} for fiscal year {fiscal_year}, according to the {table_type}."
            else:
                text = f"{company_name} ({ticker}) reported {readable} in {label} for fiscal year {fiscal_year}, according to the {table_type}."
        
        # Generate unique ID
        record_id = f"{ticker}_{table_type.replace(' ', '')}_{fiscal_year}_{concept}"
        
        # Build metadata-rich record
        record = {
            "id": record_id,
            "text": text,
            "ticker": ticker,
            "company_name": company_name,
            "fiscal_year": fiscal_year,
            "fiscal_date": year,
            "line_item": label,
            "concept": concept,
            "value": float(value) if isinstance(value, (int, float)) else None,
            "value_formatted": readable,
            "table_type": table_type,
            "document_type": "10-K",  # Can be parameterized
            "timestamp": datetime.now().isoformat()
        }
        
        return record
    
    def flatten_table(self, table_data: List[Dict], table_name: str, 
                     ticker: str, company_name: str = None) -> List[Dict]:
        """
        Flatten entire financial table to sentence-based JSON records.
        
        Returns list of records, each containing text + metadata.
        """
        records = []
        
        # Use ticker as company name if not provided
        if company_name is None:
            company_name = ticker
        
        # Extract table type from name
        table_type = table_name.replace('_', ' ').title()
        
        for row in table_data:
            label = row.get("label", "Unknown")
            concept = row.get("concept", "")
            
            # Skip if concept should be included but is missing
            if self.config.include_concepts and not concept:
                continue
            
            years = self.extract_years(row)
            
            # Generate one record per year
            for year in years:
                value = row.get(year)
                
                # Skip null values if configured
                if value is None or value == "":
                    if self.config.null_handling == "skip":
                        continue
                
                record = self.generate_sentence(
                    ticker=ticker,
                    company_name=company_name,
                    label=label,
                    value=value,
                    year=year,
                    concept=concept,
                    table_type=table_type
                )
                
                if record:  # None if null handling skipped it
                    records.append(record)
        
        return records
    
    def flatten_file(self, json_path: str) -> List[Dict]:
        """
        Load JSON file and flatten all tables to sentence-based records.
        
        Returns list of all records from all tables in the file.
        """
        all_records = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract ticker from filename and convert to uppercase
            filename = os.path.basename(json_path)
            ticker = filename.split('_')[0].upper() if '_' in filename else "UNKNOWN"
            company_name = ticker  # Can be enhanced with company lookup
            
            # Detect table types and process
            if "multi_balance_sheet" in data:
                records = self.flatten_table(
                    data["multi_balance_sheet"], 
                    "Balance Sheet", 
                    ticker, 
                    company_name
                )
                all_records.extend(records)
                
            if "multi_income_statement" in data:
                records = self.flatten_table(
                    data["multi_income_statement"], 
                    "Income Statement", 
                    ticker, 
                    company_name
                )
                all_records.extend(records)
                
            if "multi_cash_flow" in data:
                records = self.flatten_table(
                    data["multi_cash_flow"], 
                    "Cash Flow Statement", 
                    ticker, 
                    company_name
                )
                all_records.extend(records)
            
            # Fallback for other structures
            if not all_records:
                if isinstance(data, list):
                    records = self.flatten_table(
                        data, 
                        "Financial Data", 
                        ticker, 
                        company_name
                    )
                    all_records.extend(records)
                elif isinstance(data, dict):
                    for table_name, table_data in data.items():
                        if isinstance(table_data, list):
                            records = self.flatten_table(
                                table_data, 
                                table_name, 
                                ticker, 
                                company_name
                            )
                            all_records.extend(records)
            
            return all_records
            
        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            return []


class BatchFlattener:
    """Efficiently flatten all files in a directory structure to JSON format"""
    
    def __init__(self, config: FlattenConfig = None):
        self.flattener = FinancialTableFlattener(config)
        self.config = config or FlattenConfig()
    
    def flatten_directory(self, root_dir: str, output_dir: str = None, 
                         file_pattern: str = "*multi*.json") -> Dict[str, str]:
        """
        Flatten all matching JSON files in directory tree.
        Outputs JSON files with sentence-based records.
        
        Returns dict of {input_path: output_path}
        """
        if output_dir is None:
            output_dir = os.path.join(root_dir, "flattened_json_output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        root_path = Path(root_dir)
        
        for json_file in root_path.rglob(file_pattern):
            try:
                print(f"Processing: {json_file}")
                
                # Flatten the file to list of records
                records = self.flattener.flatten_file(str(json_file))
                
                # Generate output filename
                relative_path = json_file.relative_to(root_path)
                output_filename = str(relative_path).replace(".json", "_flattened.json")
                output_path = Path(output_dir) / output_filename
                
                # Create subdirectories
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write output as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
                
                results[str(json_file)] = str(output_path)
                print(f"  ✓ Saved {len(records)} records to: {output_path}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                results[str(json_file)] = None
        
        return results
    
    def generate_summary_report(self, results: Dict[str, str], 
                               output_file: str = None) -> str:
        """Generate summary of flattening operation"""
        if output_file is None:
            output_file = "flattening_summary.txt"
        
        successful = {k: v for k, v in results.items() if v is not None}
        failed = {k: v for k, v in results.items() if v is None}
        
        report_lines = [
            "=" * 80,
            "DOCUMENT FLATTENING SUMMARY REPORT",
            "=" * 80,
            f"\nTotal files processed: {len(results)}",
            f"Successful: {len(successful)}",
            f"Failed: {len(failed)}",
            "\n" + "-" * 80,
            "\nSuccessful Files:",
            "-" * 80
        ]
        
        total_records = 0
        for input_path, output_path in successful.items():
            if output_path:
                # Count records in output file
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        records = json.load(f)
                        num_records = len(records)
                        total_records += num_records
                        report_lines.append(f"  {input_path}")
                        report_lines.append(f"    → {output_path} ({num_records} records)")
                except:
                    report_lines.append(f"  {input_path}")
                    report_lines.append(f"    → {output_path}")
        
        report_lines.extend([
            f"\nTotal records generated: {total_records}",
            "\n" + "-" * 80
        ])
        
        if failed:
            report_lines.extend([
                "\nFailed Files:",
                "-" * 80
            ])
            for input_path in failed.keys():
                report_lines.append(f"  {input_path}")
        
        report_lines.append("\n" + "=" * 80 + "\n")
        
        report_text = "\n".join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Flatten single file with simple sentences
    print("Example 1: Single file flattening (simple format)")
    print("-" * 80)
    
    config = FlattenConfig(
        output_format="sentence",
        include_concepts=True,
        format_style="full"
    )
    
    flattener = FinancialTableFlattener(config)
    
    # Flatten the Apple balance sheet
    json_file = r"c:\Users\Cedric\Desktop\RAG THESIS\RAG_Framework\edgar-tools\10-K\multi_balance_sheet\aapl_multi_balance_sheet-10K.json"
    
    if os.path.exists(json_file):
        records = flattener.flatten_file(json_file)
        
        # Preview first few records
        print(f"\nGenerated {len(records)} records. Showing first 5:\n")
        for record in records[:5]:
            print(f"Text: {record['text']}")
            print(f"ID: {record['id']}")
            print(f"Ticker: {record['ticker']} | Year: {record['fiscal_year']} | Value: {record['value_formatted']}")
            print("-" * 80)
        
        # Save to JSON file
        output_path = json_file.replace(".json", "_flattened.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved {len(records)} records to: {output_path}")
    else:
        print(f"File not found: {json_file}")
    
    print("\n\n")
    
    # Example 2: Detailed sentence format
    print("Example 2: Detailed sentence format")
    print("-" * 80)
    
    config_detailed = FlattenConfig(
        output_format="detailed_sentence",
        include_concepts=True,
        format_style="billions"
    )
    
    flattener_detailed = FinancialTableFlattener(config_detailed)
    
    if os.path.exists(json_file):
        records = flattener_detailed.flatten_file(json_file)
        
        print(f"\nGenerated {len(records)} records. Showing first 3:\n")
        for record in records[:3]:
            print(record['text'])
            print()
    
    print("\n\n")
    
    # Example 3: Batch flatten entire directory
    print("Example 3: Batch directory flattening")
    print("-" * 80)
    print("""
    To flatten all files in edgar-tools:
    
    config = FlattenConfig(
        output_format="detailed_sentence",
        format_style="billions",
        include_concepts=True
    )
    
    batch_flattener = BatchFlattener(config)
    results = batch_flattener.flatten_directory(
        root_dir=r'c:\\Users\\Cedric\\Desktop\\RAG THESIS\\RAG_Framework\\edgar-tools\\10-K',
        output_dir=r'c:\\Users\\Cedric\\Desktop\\RAG THESIS\\RAG_Framework\\edgar-tools\\flattened_json'
    )
    
    # Generate summary report
    summary = batch_flattener.generate_summary_report(results, 'batch_summary.txt')
    print(summary)
    """)

