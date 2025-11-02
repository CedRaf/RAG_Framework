from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import os
from company_data import company_data

load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")

filing_types = ["10-K", "10-Q"]
years = ["2020", "2021", "2022", "2023", "2024"]

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
