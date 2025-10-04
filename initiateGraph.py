from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import os

load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")

company_data = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG": "Alphabet Inc. (Class C)",
    "TSLA": "Tesla, Inc.",
    "ADBE": "Adobe Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "ABNB": "Airbnb, Inc.",
    "AEP": "American Electric Power",
    "AMGN": "Amgen Inc.",
    "ADI": "Analog Devices, Inc.",
    "ANSS": "ANSYS, Inc.",
    "APP": "AppLovin Corporation Class A",
    "ARM": "Arm Holdings plc",
    "ASML": "ASML Holding N.V.",
    "AZN": "AstraZeneca plc",
    "TEAM": "Atlassian Corporation",
    "ADSK": "Autodesk, Inc.",
    "ADP": "Automatic Data Processing, Inc.",
    "AXON": "Axon Enterprise, Inc.",
    "BKR": "Baker Hughes Company",
    "BIIB": "Biogen Inc.",
    "BKNG": "Booking Holdings Inc.",
    "AVGO": "Broadcom Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    "CDW": "CDW Corporation",
    "CHTR": "Charter Communications, Inc.",
    "CTAS": "Cintas Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "CCEP": "Coca‑Cola Europacific Partners plc",
    "CTSH": "Cognizant Technology Solutions Corporation",
    "CMCSA": "Comcast Corporation",
    "CEG": "Constellation Energy Corporation",
    "CPRT": "Copart, Inc.",
    "CSGP": "CoStar Group, Inc.",
    "COST": "Costco Wholesale Corporation",
    "CRWD": "CrowdStrike Holdings, Inc.",
    "CSX": "CSX Corporation",
    "DDOG": "Datadog, Inc.",
    "DXCM": "DexCom, Inc.",
    "FANG": "Diamondback Energy, Inc.",
    "DASH": "DoorDash, Inc.",
    "EA": "Electronic Arts Inc.",
    "EXC": "Exelon Corporation",
    "FAST": "Fastenal Company",
    "FTNT": "Fortinet, Inc.",
    "GEHC": "GE HealthCare Technologies Inc.",
    "GILD": "Gilead Sciences, Inc.",
    "GFS": "GlobalFoundries Inc.",
    "HON": "Honeywell International Inc.",
    "IDXX": "IDEXX Laboratories, Inc.",
    "INTC": "Intel Corporation",
    "INTU": "Intuit Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "KDP": "Keurig Dr Pepper Inc.",
    "KLAC": "KLA Corporation",
    "KHC": "The Kraft Heinz Company",
    "LRCX": "Lam Research Corporation",
    "LIN": "Linde plc",
    "LULU": "Lululemon Athletica Inc.",
    "MAR": "Marriott International, Inc.",
    "MRVL": "Marvell Technology, Inc.",
    "MELI": "Mercado Libre, Inc.",
    "MCHP": "Microchip Technology Inc.",
    "MU": "Micron Technology, Inc.",
    "MSTR": "MicroStrategy Incorporated",
    "MDLZ": "Mondelez International, Inc.",
    "MNST": "Monster Beverage Corporation",
    "NFLX": "Netflix, Inc.",
    "NXPI": "NXP Semiconductors N.V.",
    "ORLY": "O'Reilly Automotive, Inc.",
    "ODFL": "Old Dominion Freight Line, Inc.",
    "ON": "ON Semiconductor Corporation",
    "PCAR": "PACCAR Inc",
    "PLTR": "Palantir Technologies Inc.",
    "PANW": "Palo Alto Networks, Inc.",
    "PAYX": "Paychex, Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "PDD": "PDD Holdings Inc.",
    "PEP": "PepsiCo, Inc.",
    "QCOM": "Qualcomm Incorporated",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "ROP": "Roper Technologies, Inc.",
    "ROST": "Ross Stores, Inc.",
    "SHOP": "Shopify Inc.",
    "SBUX": "Starbucks Corporation",
    "SNPS": "Synopsys, Inc.",
    "TTWO": "Take‑Two Interactive Software, Inc.",
    "TMUS": "T‑Mobile US, Inc.",
    "TXN": "Texas Instruments Incorporated",
    "TTD": "The Trade Desk, Inc.",
    "VRSK": "Verisk Analytics, Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "WBD": "Warner Bros. Discovery, Inc.",
    "WDAY": "Workday, Inc.",
    "XEL": "Xcel Energy Inc.",
    "ZS": "Zscaler, Inc."
}


filing_types = ["10-K", "10-Q"]
years = ["2020", "2021", "2022", "2023", "2024", "2025"]

driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))

def create_company_structure():
    with driver.session() as session:
        for ticker, name in company_data.items():
            # Central Company Node
            session.run("""
                MERGE (c:Company {company_ticker: $company_ticker})
                SET c.company_name = $company_name
            """, {"company_ticker": ticker, "company_name": name})

            for form_type in filing_types:
                # FilingType Node
                session.run("""
                    MATCH (c:Company {company_ticker: $company_ticker})
                    MERGE (f:FilingType {
                        company_ticker: $company_ticker,
                        form_type: $form_type
                    })
                    MERGE (c)-[:HAS_FILING]->(f)
                """, {
                    "company_ticker": ticker,
                    "form_type": form_type
                })

                for filing_year in years:
                    # Year Node
                    session.run("""
                        MATCH (f:FilingType {
                            company_ticker: $company_ticker,
                            form_type: $form_type
                        })
                        MERGE (y:Year {
                            company_ticker: $company_ticker,
                            form_type: $form_type,
                            filing_year: $filing_year
                        })
                        MERGE (f)-[:HAS_YEAR]->(y)
                    """, {
                        "company_ticker": ticker,
                        "form_type": form_type,
                        "filing_year": filing_year
                    })

    print("Company structure created.")

if __name__ == "__main__":
    create_company_structure()
    driver.close()
