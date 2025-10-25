from edgar import Company, MultiFinancials
from edgar import *
import json
import os

set_identity("rhenzgerard0@gmail.com")
#os.chdir(r"c:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG/edgar-tools")
#os.environ['PYTHONTZPATH'] = r"C:\Users\Rhenz\Downloads\tzdata"

output_dir = "10-K"
balance_dir = os.path.join(output_dir, "multi_balance_sheet")
income_dir = os.path.join(output_dir, "multi_income_statement")
cashflow_dir = os.path.join(output_dir, "multi_cash_flow")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(balance_dir, exist_ok=True)
os.makedirs(income_dir, exist_ok=True)
os.makedirs(cashflow_dir, exist_ok=True)


companies = ["LIN","AMZN","NFLX","COST","MELI","SBUX","ORLY","BKR","PEP","MDLZ",
             "GILD","VRTX","BIIB","CSX","VRSK","ODFL","CSGP","AAPL","GOOGL","META",
             "AVGO","NVDA","APP","CRWD","INTU","ADBE","MSTR","MRVL","NXPI","ZS",
             "MCHP","CTSH","ON","TMUS","CEG"]

for ticker in companies:
    # Get financial data
    company = Company(ticker)
    financials = company.get_financials()
    filings = company.get_filings(form="10-K").head(5)  # Last 5 annual reports
    multi_financials = MultiFinancials.extract(filings)

    # Access statements spanning multiple years
    balance_sheet_df = multi_financials.balance_sheet().to_dataframe()
    income_statement_df = multi_financials.income_statement().to_dataframe()
    cash_flow_df = multi_financials.cashflow_statement().to_dataframe()

    # Convert to dictionary (records)
    balance_json = json.loads(balance_sheet_df.to_json(orient='records'))
    income_json = json.loads(income_statement_df.to_json(orient='records'))
    cashflow_json = json.loads(cash_flow_df.to_json(orient='records'))

    # Save as JSON files in their respective directories
    with open(os.path.join(balance_dir, f"{ticker.lower()}_multi_balance_sheet-10K.json"), "w") as f:
        json.dump({"ticker": ticker, "multi_balance_sheet": balance_json}, f, indent=2)
    with open(os.path.join(income_dir, f"{ticker.lower()}_multi_income_statement-10K.json"), "w") as f:
        json.dump({"ticker": ticker, "multi_income_statement": income_json}, f, indent=2)
    with open(os.path.join(cashflow_dir, f"{ticker.lower()}_multi_cash_flow-10K.json"), "w") as f:
        json.dump({"ticker": ticker, "multi_cash_flow": cashflow_json}, f, indent=2)