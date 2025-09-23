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

    # # Income Statement
    # income_df = financials.income_statement().to_dataframe()
    # income_json = json.loads(income_df.to_json(orient='records'))
    # filename = f"{ticker.lower()}_income_statement.json"
    # company_income_statement = {
    #     "ticker" : ticker,
    #     "income_statement" : income_json
    # }
    # with open(filename, 'w') as f:
    #     json.dump(company_income_statement, f, indent=2)

    # # Balance Sheet  
    # balance_df = financials.balance_sheet().to_dataframe()
    # balance_json = json.loads(balance_df.to_json(orient='records'))
    # filename = f"{ticker.lower()}_balance_sheet.json"
    # company_balance_sheet = {
    #     "ticker" : ticker,
    #     "balance_sheet": balance_json
    # }
    # with open(filename, 'w') as f:
    #     json.dump(company_balance_sheet, f, indent=2)

    # # Cash Flow
    # cashflow_df = financials.cashflow_statement().to_dataframe() 
    # cashflow_json = json.loads(cashflow_df.to_json(orient='records'))
    # filename = f"{ticker.lower()}_cash_flow.json"
    # company_cash_flow = {
    #     "ticker": ticker,
    #     "cash_flow": cashflow_json
    # }
    # with open(filename, 'w') as f:
    #     json.dump(company_cash_flow, f, indent=2)


    # Get comprehensive metrics dictionary
    # metrics = financials.get_financial_metrics()

    # # All the standard metrics are available
    # print(f"Revenue: ${metrics['revenue']:,.0f}")
    # print(f"Net Income: ${metrics['net_income']:,.0f}")
    # print(f"Total Assets: ${metrics['total_assets']:,.0f}")

    # # Plus calculated ratios
    # print(f"Current Ratio: {metrics['current_ratio']:.2f}")
    # print(f"Debt to Assets: {metrics['debt_to_assets']:.2f}")
    # print(f"Free Cash Flow: ${metrics['free_cash_flow']:,.0f}")

    filings = company.get_filings(form="10-Q").head(15)  # Last 5 annual reports
    # filings = company.get_filings(year=[2023], form="10-Q")
    # filings = company.get_filings(form="10-Q")
    # filings = filings.filter(date="2020-01-01:2021-12-31")
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