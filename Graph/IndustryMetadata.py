from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Industry mappings
TICKER_TO_INDUSTRY = {
    # Technology
    'AAPL': 'Technology',
    'GOOGL': 'Technology',
    'META': 'Technology',
    'AVGO': 'Technology',
    'NVDA': 'Technology',
    'APP': 'Technology',
    'CRWD': 'Technology',
    'INTU': 'Technology',
    'ADBE': 'Technology',
    'MSTR': 'Technology',
    'MRVL': 'Technology',
    'NXPI': 'Technology',
    'ZS': 'Technology',
    'MCHP': 'Technology',
    'CTSH': 'Technology',
    'ON': 'Technology',

    #Consumer Discretionary
    'AMZN': 'Consumer Discretionary',
    'NFLX': 'Consumer Discretionary',
    'COST': 'Consumer Discretionary',
    'MELI': 'Consumer Discretionary',
    'SBUX': 'Consumer Discretionary',
    'ORLY': 'Consumer Discretionary',
    'BKR': 'Consumer Discretionary',
    
    #Healthcare
    'GILD': 'Healthcare',
    'VRTX': 'Healthcare',
    'BIIB': 'Healthcare',

    #Industrials
    'CSX': 'Industrials',
    'VRSK': 'Industrials',
    'ODFL': 'Industrials',

    #Consumer Staples
    'PEP': 'Consumer Staples',
    'MDLZ': 'Consumer Staples',

    #Real Estate
    'CSGP': 'Real Estate',

    #Telecomm
    'TMUS': 'Telecommunications',

    #Utilities
    'CEG': 'Utilities',

    #Basic Materials
    'LIN': 'Basic Materials'
}

def add_industry_to_companies(session):
    """Add industry property to all Company nodes"""
    for ticker, industry in TICKER_TO_INDUSTRY.items():
        session.run("""
            MATCH (c:Company {company_ticker: $ticker})
            SET c.industry = $industry
        """, {"ticker": ticker, "industry": industry})
        print(f"Updated {ticker} -> {industry}")
    
    # Set "Other" for any remaining companies without industry
    result = session.run("""
        MATCH (c:Company)
        WHERE c.industry IS NULL
        SET c.industry = 'Other'
        RETURN count(c) AS updated_count
    """)
    count = result.single()["updated_count"]
    print(f"Set {count} remaining companies to 'Other' industry")

if __name__ == "__main__":
    with driver.session() as session:
        add_industry_to_companies(session)
    
    driver.close()
    print("\nAdded metadata")