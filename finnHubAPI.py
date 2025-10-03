import finnhub
import requests
import json
import time
import os
import pandas as pd
import numpy as np

# New folder to store retreived data from SEC EDGAR

output_dir = "C:\\Users\\Rhenz\\Documents\\School\\CodeFolders\\Thesis\\RAG\\finnhub_news"
os.makedirs(output_dir, exist_ok=True)
# print(os.path.abspath(output_dir))
# print(os.getcwd())
finnhub_client = finnhub.Client(api_key="d10m0n1r01qlsac9ukpgd10m0n1r01qlsac9ukq0")

nasdaq_100_tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'ADBE', 'AMD',
    'ABNB', 'AEP', 'AMGN', 'ADI', 'ANSS', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM',
    'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR',
    'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD',
    'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC',
    'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC',
    'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSTR',
    'MDLZ', 'MNST', 'NFLX', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW',
    'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX',
    'SNPS', 'TTWO', 'TMUS', 'TSLA', 'TXN', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY',
    'XEL', 'ZS'
]

for ticker in nasdaq_100_tickers:
    try:
        # Fetch company news for the ticker
        news_data = finnhub_client.company_news(ticker, _from="2024-04-01", to="2025-04-01")
        
        print(f"Retrieved news for {ticker}")

        # Save news data to a JSON file
        file_path = os.path.join(output_dir, f"{ticker}_news.json")
        with open(file_path, "w") as f:
            json.dump(news_data, f, indent=4)

    except Exception as e:
        print(f"Failed to fetch news for {ticker}: {str(e)}")

    # Rate limit of 1 per second
    time.sleep(1)