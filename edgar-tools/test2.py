from edgar import Company, MultiFinancials
from edgar import *
import json
import os

set_identity("rhenzgerard0@gmail.com")
os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG/edgar-tools")
os.environ['PYTHONTZPATH'] = r"C:\Users\Rhenz\Downloads\tzdata"

companies = ["AAPL"]

for ticker in companies:
    # Get financial data
    company = Company(ticker)
    financials = company.get_financials()
    filings = company.get_filings(form="10-Q").head(15)  # Last 5 annual reports
    multi_financials = MultiFinancials.extract(filings)

    # Access statements spanning multiple years
    balance_sheet_df = multi_financials.balance_sheet().to_dataframe()
    income_statement_df = multi_financials.income_statement().to_dataframe()
    cash_flow_df = multi_financials.cashflow_statement().to_dataframe()

    # Convert to dictionary (records)
    balance_json = json.loads(balance_sheet_df.to_json(orient='records'))
    income_json = json.loads(income_statement_df.to_json(orient='records'))
    cashflow_json = json.loads(cash_flow_df.to_json(orient='records'))

    # Save as JSON files
    with open(f"{ticker.lower()}_multi_balance_sheet-10Q.json", 'w') as f:
        json.dump({"ticker": ticker, "multi_balance_sheet": balance_json}, f, indent=2)
    with open(f"{ticker.lower()}_multi_income_statement-10Q.json", 'w') as f:
        json.dump({"ticker": ticker, "multi_income_statement": income_json}, f, indent=2)
    with open(f"{ticker.lower()}_multi_cash_flow.json-10Q.json", 'w') as f:
        json.dump({"ticker": ticker, "multi_cash_flow": cashflow_json}, f, indent=2)