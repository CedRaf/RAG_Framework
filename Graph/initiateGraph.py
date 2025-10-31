from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import os

load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")

company_data = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "GOOGL": "Alphabet Inc.",
    "ADBE": "Adobe Inc.",
    "APP": "AppLovin Corporation Class A",
    "BKR": "Baker Hughes Company",
    "BIIB": "Biogen Inc.",
    "AVGO": "Broadcom Inc.",
    "CTSH": "Cognizant Technology Solutions Corporation",
    "CEG": "Constellation Energy Corporation",
    "CSGP": "CoStar Group, Inc.",
    "COST": "Costco Wholesale Corporation",
    "CRWD": "CrowdStrike Holdings, Inc.",
    "CSX": "CSX Corporation",
    "GILD": "Gilead Sciences, Inc.",
    "INTU": "Intuit Inc.",
    "LIN": "Linde plc",
    "MRVL": "Marvell Technology, Inc.",
    "MELI": "Mercado Libre, Inc.",
    "MCHP": "Microchip Technology Inc.",
    "MSTR": "MicroStrategy Incorporated",
    "MDLZ": "Mondelez International, Inc.",
    "NFLX": "Netflix, Inc.",
    "NXPI": "NXP Semiconductors N.V.",
    "ORLY": "O'Reilly Automotive, Inc.",
    "ODFL": "Old Dominion Freight Line, Inc.",
    "ON": "ON Semiconductor Corporation",
    "PEP": "PepsiCo, Inc.",
    "SBUX": "Starbucks Corporation",
    "TMUS": "T‑Mobile US, Inc.",
    "VRSK": "Verisk Analytics, Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "ZS": "Zscaler, Inc."
}


filing_types = ["10-K", "10-Q"]
years = ["2021", "2022", "2023", "2024"]

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
