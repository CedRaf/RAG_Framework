import os
import re
import json
from neo4j import GraphDatabase
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # load environment variables

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASS")

#JSON_DIR = r"C:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG\edgar-tools"
JSON_DIR = Path(__file__).resolve().parent.parent / "edgar-tools"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def load_10K_from_json(driver, directory=JSON_DIR):
    year_pattern = re.compile(r"(\d{4})-\d{2}-\d{2}")  # YYYY
    # Fix regex to allow -10K.json OR .json-10K.json patterns, case-insensitive for ticker
    filename_pattern = re.compile(r"([a-zA-Z]+)_.*[-_]?(10K|10Q)\.json$", re.IGNORECASE)

    with driver.session() as session:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)

                if not filename.endswith(".json"):
                    continue

                # only process 10-Ks for now
                match = filename_pattern.search(filename)
                if not match:
                    print(f"Skipping (no match): {filename}")
                    continue
                ticker, filing_type = match.groups()
                ticker = ticker.upper()
                if filing_type.upper() == "10K":
                    filing_type = "10-K"
                elif filing_type.upper() == "10Q":
                    filing_type = "10-Q"
                if filing_type != "10-K":
                    print(f"Skipping 10-Q file: {filename}")
                    continue

                print(f"Processing {filename} for {ticker} ({filing_type})")

                with open(filepath, "r") as f:
                    data = json.load(f)

                # Look for table key - could be multi_* or just a regular key
                table_key = next((k for k in data.keys() if k.startswith("multi_")), None)
                if not table_key:
                    # If no multi_* key, try to find any key that looks like a table
                    for k in data.keys():
                        if any(x in k.lower() for x in ["income", "balance", "cash"]):
                            table_key = k
                            break
                
                if not table_key:
                    print(f"No table key found in {filename}")
                    continue

                # Determine table type
                table_type = (
                    "IncomeStatement" if "income" in table_key else
                    "BalanceSheet" if "balance" in table_key else
                    "CashFlowStatement" if "cash" in table_key else "FinancialTable"
                )

                rows = data[table_key]

                for row in rows:
                    label = row.get("label")
                    concept = row.get("concept")  # Extract XBRL concept

                    for key, value in row.items():
                        year_match = year_pattern.match(key)
                        if year_match:
                            filing_year = year_match.group(1)

                            session.run(f"""
                                MERGE (c:Company {{company_ticker:$ticker}})
                                MERGE (c)-[:HAS_FILING]->(f:FilingType {{
                                    form_type:$form_type,
                                    company_ticker:$ticker
                                }})
                                MERGE (f)-[:HAS_YEAR]->(y:Year {{
                                    filing_year:$filing_year,
                                    form_type:$form_type,
                                    company_ticker:$ticker
                                }})
                                MERGE (y)-[:HAS_TABLE]->(t:{table_type} {{
                                    table_type:$table_type,
                                    filing_year:$filing_year,
                                    company_ticker:$ticker,
                                    form_type:$form_type
                                }})
                                CREATE (item:LineItem {{
                                    label:$label,
                                    concept:$concept,
                                    value:$value,
                                    filing_year:$filing_year,
                                    company_ticker:$ticker,
                                    form_type:$form_type,
                                    table_type:$table_type
                                }})
                                MERGE (t)-[:HAS_ITEM]->(item)
                            """, {
                                "ticker": ticker,
                                "form_type": filing_type,
                                "filing_year": filing_year,
                                "table_type": table_type,
                                "label": label,
                                "concept": concept,
                                "value": value
                            })

    print("Finished loading 10-K financials into Neo4j.")

def load_10Q_from_json(driver, directory=JSON_DIR):
    year_pattern = re.compile(r"(\d{4})-(\d{2})-\d{2}")  # YYYY-MM
    filename_pattern = re.compile(r"([a-zA-Z]+)_([a-z_]+)_(\d{4}-\d{2}-\d{2})[-_]?(10K|10Q)\.json$", re.IGNORECASE)
    
    with driver.session() as session:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                filepath = os.path.join(root, filename)

                if not filename.endswith(".json"):
                    continue

                match = filename_pattern.search(filename)
                if not match:
                    print(f"Skipping (no match): {filename}")
                    continue
                
                ticker, table_name, date_str, filing_type = match.groups()
                ticker = ticker.upper()
                
                if filing_type.upper() == "10K":
                    filing_type = "10-K"
                elif filing_type.upper() == "10Q":
                    filing_type = "10-Q"
                
                # Only process 10-Qs for this function
                if filing_type != "10-Q":
                    print(f"Skipping 10-K file: {filename}")
                    continue

                # Determine table type from filename
                table_type = (
                    "IncomeStatement" if "income" in table_name.lower() else
                    "BalanceSheet" if "balance" in table_name.lower() else
                    "CashFlowStatement" if "cash" in table_name.lower() else "FinancialTable"
                )

                print(f"Processing {filename} for {ticker} ({filing_type}) - {table_type}")

                with open(filepath, "r") as f:
                    data = json.load(f)

                # For 10-Q files, data is indexed by numeric keys (as strings)
                # Each row has a "label" and date columns
                rows = [row_data for key, row_data in data.items() if isinstance(row_data, dict) and "label" in row_data]
                
                if not rows:
                    print(f"No valid rows found in {filename}")
                    continue

                # First pass: collect all year-month combinations to determine quarters
                year_month_data = {}
                for row in rows:
                    for key, value in row.items():
                        year_match = year_pattern.match(key)
                        if year_match:
                            year = year_match.group(1)
                            month = year_match.group(2)
                            if year not in year_month_data:
                                year_month_data[year] = []
                            if month not in year_month_data[year]:
                                year_month_data[year].append(month)
                
                # Sort months for each year to assign quarters chronologically
                for year in year_month_data:
                    year_month_data[year].sort()

                # Second pass: process the actual data
                for row in rows:
                    label = row.get("label")
                    concept = row.get("concept")  # Extract XBRL concept
                    if not label:
                        continue

                    for key, value in row.items():
                        # Skip non-numeric keys like "label", "concept", etc.
                        if key in ["label", "concept", "level", "abstract", "dimension"]:
                            continue
                        
                        year_match = year_pattern.match(key)
                        if year_match:
                            filing_year = year_match.group(1)
                            month = year_match.group(2)
                            
                            # Only insert if there's an actual value
                            if value == "" or value is None:
                                continue
                            
                            # Determine quarter based on month position in sorted list for that year
                            sorted_months = year_month_data.get(filing_year, [])
                            if month in sorted_months:
                                quarter_index = sorted_months.index(month) + 1
                                quarter = f"Q{quarter_index}"
                            else:
                                continue

                            session.run(f"""
                                MERGE (c:Company {{company_ticker:$ticker}})
                                MERGE (c)-[:HAS_FILING]->(f:FilingType {{
                                    form_type:$form_type,
                                    company_ticker:$ticker
                                }})
                                MERGE (f)-[:HAS_YEAR]->(y:Year {{
                                    filing_year:$filing_year,
                                    form_type:$form_type,
                                    company_ticker:$ticker
                                }})
                                MERGE (y)-[:HAS_QUARTER]->(q:Quarter {{
                                    quarter:$quarter,
                                    filing_year:$filing_year,
                                    form_type:$form_type,
                                    company_ticker:$ticker
                                }})
                                MERGE (q)-[:HAS_TABLE]->(t:{table_type} {{
                                    table_type:$table_type,
                                    quarter:$quarter,
                                    filing_year:$filing_year,
                                    company_ticker:$ticker,
                                    form_type:$form_type
                                }})
                                CREATE (item:LineItem {{
                                    label:$label,
                                    concept:$concept,
                                    value:$value,
                                    quarter:$quarter,
                                    filing_year:$filing_year,
                                    company_ticker:$ticker,
                                    form_type:$form_type,
                                    table_type:$table_type
                                }})
                                MERGE (t)-[:HAS_ITEM]->(item)
                            """, {
                                "ticker": ticker,
                                "form_type": filing_type,
                                "filing_year": filing_year,
                                "quarter": quarter,
                                "table_type": table_type,
                                "label": label,
                                "concept": concept,
                                "value": value
                            })

    print("Finished loading 10-Q financials into Neo4j.")

if __name__ == "__main__":
    load_10K_from_json(driver, directory=JSON_DIR)
    #load_10Q_from_json(driver, directory=JSON_DIR)
    driver.close()
