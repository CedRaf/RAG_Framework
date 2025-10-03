from edgar import Company, MultiFinancials
from edgar import *
import json
import os
import pandas as pd


set_identity("rhenzgerard0@gmail.com")
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG/edgar-tools")
os.environ['PYTHONTZPATH'] = r"C:\Users\Rhenz\Downloads\tzdata"

companies = ["AAPL"]

for ticker in companies:
    company = Company(ticker)
    # Fetch all 10-Q filings (no year filter here)
    filings = company.get_filings(form="10-Q")
    # Convert filings to a pandas DataFrame and filter for 2023
    df = filings.to_pandas()
    print(df.columns)

    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df_2023 = df[df['filing_date'].dt.year == 2023]
    # Get list of accession numbers for these filings
    accession_nos = df_2023['accession_number'].tolist()
    # Reconstruct the filtered filings object
    filtered_filings = [f for f in filings if f.accession_no in accession_nos]
    # If you have access to MultiFinancials (sometimes requires an object with .head/.tail), you may need:
    # filtered_filings = filings.filter(lambda f: f.accession_no in accession_nos)
    multi_financials = MultiFinancials.extract(filtered_filings)
    
    # Extract financial statements
    balance_sheet_df = multi_financials.balance_sheet().to_dataframe()
    income_statement_df = multi_financials.income_statement().to_dataframe()
    cash_flow_df = multi_financials.cashflow_statement().to_dataframe()
    
    # Save to JSON as before
    with open(f"{ticker.lower()}_multi_balance_sheet-10Q-testlocal.json", 'w') as f:
        json.dump({"ticker": ticker, "multi_balance_sheet": json.loads(balance_sheet_df.to_json(orient='records'))}, f, indent=2)
    with open(f"{ticker.lower()}_multi_income_statement-10Q-testlocal.json", 'w') as f:
        json.dump({"ticker": ticker, "multi_income_statement": json.loads(income_statement_df.to_json(orient='records'))}, f, indent=2)
    with open(f"{ticker.lower()}_multi_cash_flow.json-10Q-testlocal.json", 'w') as f:
        json.dump({"ticker": ticker, "multi_cash_flow": json.loads(cash_flow_df.to_json(orient='records'))}, f, indent=2)
