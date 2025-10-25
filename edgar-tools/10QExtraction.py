from edgar import Company
from edgar import *
import json
import pandas as pd
import os

set_identity("rhenzgerard0@gmail.com")

output_dir = "10-Q"
balance_dir = os.path.join(output_dir, "balance_sheet")
income_dir = os.path.join(output_dir, "income_statement")
cashflow_dir = os.path.join(output_dir, "cash_flow")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(balance_dir, exist_ok=True)
os.makedirs(income_dir, exist_ok=True)
os.makedirs(cashflow_dir, exist_ok=True)

companies = ["LIN","AMZN","NFLX","COST","MELI","SBUX","ORLY","BKR","PEP","MDLZ",
             "GILD","VRTX","BIIB","CSX","VRSK","ODFL","CSGP","AAPL","GOOGL","META",
             "AVGO","NVDA","APP","CRWD","INTU","ADBE","MSTR","MRVL","NXPI","ZS",
             "MCHP","CTSH","ON","TMUS","CEG"]

for ticker in companies:

    company = Company(ticker)

    filings = company.get_filings(form="10-Q")
    filings_recent = [f for f in filings if f.filing_date.year >= 2020]
    # There is a limit to how far back you can go because of the way the logic works.
    # Earlier 10-Q filings don't follow the same format of having a 'statements' dict, so the logic won't work

    for filing in filings_recent:
        try:
            xbrl = filing.xbrl()
            statements = xbrl.statements
            

            for name, method, subdir in [
                ('balance_sheet', statements.balance_sheet, balance_dir),
                ('income_statement', statements.income_statement, income_dir),
                ('cashflow_statement', statements.cashflow_statement, cashflow_dir)
            ]:
                try:
                    statement = method()
                    df = statement.to_dataframe()
                    json_file = os.path.join(subdir, f"{ticker}_{name}_{filing.filing_date}-10Q.json")
                    df.to_json(json_file, orient="index", indent=2)
                    print(f"Saved {json_file}")
                except Exception as e:
                    print(f"Could not extract {name}: {e}")
                    
        except Exception as e:
            print(f"Skipped {filing.filing_date}, error: {e}")