from edgar import Company
from edgar import *
import json
import pandas as pd
import os

set_identity("rhenzgerard0@gmail.com")

output_dir = "10-Q"
os.makedirs(output_dir, exist_ok=True)

companies = ['AAPL']

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
            
            for name, method in [
                ('balance_sheet', statements.balance_sheet),
                ('income_statement', statements.income_statement), 
                ('cashflow_statement', statements.cashflow_statement)
            ]:
                try:
                    statement = method()
                    df = statement.to_dataframe()
                    json_file = os.path.join(output_dir, f"{ticker}_{name}_{filing.filing_date}-10Q.json") 
                    df.to_json(json_file, orient="index", indent=2)
                    print(f"Saved {json_file}")
                except Exception as e:
                    print(f"Could not extract {name}: {e}")
                    
        except Exception as e:
            print(f"Skipped {filing.filing_date}, error: {e}")