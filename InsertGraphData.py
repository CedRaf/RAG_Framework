import os
import re
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()  # load environment variables

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASS")

# Use your absolute directory path
JSON_DIR = r"C:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG\edgar-tools"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def load_10K_from_json(driver, directory=JSON_DIR):
    year_pattern = re.compile(r"(\d{4})-\d{2}-\d{2}")  # YYYY
    # Fix regex to allow -10K.json OR .json-10K.json patterns
    filename_pattern = re.compile(r"([a-z]+)_.*[-_]?(10K|10Q)\.json$", re.IGNORECASE)

    with driver.session() as session:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

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

            table_key = next((k for k in data.keys() if k.startswith("multi_")), None)
            if not table_key:
                print(f"No multi_* key found in {filename}")
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
                            "value": value
                        })

    print("Finished loading 10-K financials into Neo4j.")

def load_10Q_from_json(driver, directory=JSON_DIR):
    year_pattern = re.compile(r"(\d{4})-(\d{2})-\d{2}")  # YYYY-MM
    filename_pattern = re.compile(r"([a-z]+)_.*[-_]?(10K|10Q)\.json$", re.IGNORECASE)
    
    with driver.session() as session:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            if not filename.endswith(".json"):
                continue

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
            
            # Only process 10-Qs for this function
            if filing_type != "10-Q":
                print(f"Skipping 10-K file: {filename}")
                continue

            print(f"Processing {filename} for {ticker} ({filing_type})")

            with open(filepath, "r") as f:
                data = json.load(f)

            table_key = next((k for k in data.keys() if k.startswith("multi_")), None)
            if not table_key:
                print(f"No multi_* key found in {filename}")
                continue

            # Determine table type
            table_type = (
                "IncomeStatement" if "income" in table_key else
                "BalanceSheet" if "balance" in table_key else
                "CashFlowStatement" if "cash" in table_key else "FinancialTable"
            )

            rows = data[table_key]

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

                for key, value in row.items():
                    year_match = year_pattern.match(key)
                    if year_match:
                        filing_year = year_match.group(1)
                        month = year_match.group(2)
                        
                        # Determine quarter based on month position in sorted list for that year
                        sorted_months = year_month_data[filing_year]
                        quarter_index = sorted_months.index(month) + 1
                        quarter = f"Q{quarter_index}"

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
                            "value": value
                        })

    print("Finished loading 10-Q financials into Neo4j.")

if __name__ == "__main__":
   # load_10K_from_json(driver, directory=JSON_DIR)
    load_10Q_from_json(driver, directory=JSON_DIR)
    driver.close()
