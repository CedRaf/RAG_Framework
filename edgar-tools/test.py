from edgar import Company
from edgar import *
import json

set_identity("rhenzgerard0@gmail.com")

companies = ["AAPL", "AMZN", "MSFT", "NVDA"]

for ticker in companies:
    # Get financial data
    company = Company(ticker)
    financials = company.get_financials()

    # Income Statement
    income_df = financials.income_statement().to_dataframe()
    income_json = json.loads(income_df.to_json(orient='records'))
    filename = f"{ticker.lower()}_income_statement.json"
    company_income_statement = {
        "ticker" : ticker,
        "income_statement" : income_json
    }
    with open(filename, 'w') as f:
        json.dump(company_income_statement, f, indent=2)

    # Balance Sheet  
    balance_df = financials.balance_sheet().to_dataframe()
    balance_json = json.loads(balance_df.to_json(orient='records'))
    filename = f"{ticker.lower()}_balance_sheet.json"
    company_balance_sheet = {
        "ticker" : ticker,
        "balance_sheet": balance_json
    }
    with open(filename, 'w') as f:
        json.dump(company_balance_sheet, f, indent=2)

    # Cash Flow
    cashflow_df = financials.cashflow_statement().to_dataframe() 
    cashflow_json = json.loads(cashflow_df.to_json(orient='records'))
    filename = f"{ticker.lower()}_cash_flow.json"
    company_cash_flow = {
        "ticker": ticker,
        "cash_flow": cashflow_json
    }
    with open(filename, 'w') as f:
        json.dump(company_cash_flow, f, indent=2)

