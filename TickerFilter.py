import re 

company_to_ticker = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "amazon": "AMZN", "meta": "META", "google": "GOOGL",   # could also map "GOOG"
    "tesla": "TSLA", "adobe": "ADBE", "amd": "AMD", "airbnb": "ABNB", "american electric power": "AEP", "amgen": "AMGN",
    "analog devices": "ADI", "ansys": "ANSS", "applovin": "APP", "arm": "ARM", "asml": "ASML", "astrazeneca": "AZN",
    "atlassian": "TEAM", "autodesk": "ADSK", "adp": "ADP", "axon": "AXON", "baker hughes": "BKR", "biogen": "BIIB",
    "booking": "BKNG", "broadcom": "AVGO", "cadence": "CDNS", "cdw": "CDW", "charter": "CHTR", "cintas": "CTAS",
    "cisco": "CSCO", "coca cola": "CCEP", "cognizant": "CTSH", "comcast": "CMCSA", "constellation energy": "CEG",
    "copart": "CPRT", "costar": "CSGP", "costco": "COST", "crowdstrike": "CRWD", "csx": "CSX", "datadog": "DDOG",
    "dexcom": "DXCM", "diamondback": "FANG", "doordash": "DASH", "ea": "EA", "exelon": "EXC", "fastenal": "FAST",
    "fortinet": "FTNT", "ge healthcare": "GEHC", "gilead": "GILD", "globalfoundries": "GFS", "honeywell": "HON",
    "idexx": "IDXX", "intel": "INTC", "intuit": "INTU", "intuitive surgical": "ISRG", "keurig": "KDP", "kla": "KLAC",
    "kraft heinz": "KHC", "lam research": "LRCX", "linde": "LIN", "lululemon": "LULU", "marriott": "MAR", "marvell": "MRVL",
    "mercadolibre": "MELI", "microchip": "MCHP", "micron": "MU", "microstrategy": "MSTR", "mondelez": "MDLZ", "monster": "MNST",
    "netflix": "NFLX", "nxp": "NXPI", "o'reilly": "ORLY", "old dominion": "ODFL", "on semiconductor": "ON", "paccar": "PCAR",
    "palantir": "PLTR", "palo alto": "PANW", "paychex": "PAYX", "paypal": "PYPL", "pdd": "PDD", "pepsi": "PEP", "qualcomm": "QCOM",
    "regeneron": "REGN", "roper": "ROP", "ross": "ROST", "shopify": "SHOP", "starbucks": "SBUX", "synopsys": "SNPS", "take two": "TTWO",
    "t mobile": "TMUS", "texas instruments": "TXN", "trade desk": "TTD", "verisk": "VRSK", "vertex": "VRTX", "warner bros": "WBD",
    "workday": "WDAY", "xcel": "XEL", "zscaler": "ZS"
}

# def get_tickers(query):
#     detected = []
#     lower_query = query.lower()
#     for company, ticker in company_to_ticker.items():
#         if re.search(rf"\b{re.escape(company)}\b", lower_query):
#             detected.append(ticker)
#     return detected


def get_tickers(query):
    # Extract ticker from parentheses at the start of the query
    match = re.match(r'^\(([A-Z]+)\)', query.strip())
    if match:
        return [match.group(1)]
    return []