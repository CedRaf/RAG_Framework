import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# LLM setup
chat = ChatPerplexity(
    temperature=0,
    model="sonar-pro",
    pplx_api_key=PERPLEXITY_API_KEY,
)

# Neo4j setup
driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))

# Prompt template
QUERY_PROMPT_TEMPLATE = """
You are a Cypher expert. Given a user query that includes:

- A company name (may not be exact),
- A form type (e.g., 10-K),
- A year,
- And a financial question (e.g., about revenue, net income, etc.),

Generate a Cypher query to retrieve relevant data using the following graph schema:

Graph Schema:
- (Company)-[:HAS_FILING]->(FilingType)-[:HAS_YEAR]->(Year)-[:HAS_TABLE]->(Table)-[:HAS_DATA]->(DataRow)

Node properties:
- Company: company_name, company_ticker
- FilingType: form_type
- Year: filing_year
- Table: title
- DataRow: column names vary (e.g., Category, year_2020, etc.)

Rules:
- Match company nodes using:
  `toLower(c.company_name) CONTAINS ...` or `toLower(c.company_ticker) = ...`
- If the user query **does not explicitly mention** the form type or year:
  - Do not include those conditions in the query
  - Do not assume or guess a default form_type like "10-K"
  - Use OPTIONAL MATCH if needed, or sort years using `ORDER BY y.filing_year DESC LIMIT 1`
- When form_type or filing_year are known, match them exactly. Wrap filing_year in double quotes (e.g., "2020").
- Use `toLower(toString(...))` for fields like Category or table values to ensure compatibility.
- Return table.title, row, or specific fields as needed.
- Output the Cypher query only — do NOT include markdown formatting, backticks, or the word "cypher".

User Query:
\"\"\"{user_query}\"\"\"

Cypher Query:
"""

def generate_cypher_from_query(user_query: str) -> str:
    prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    cypher = chain.invoke({"user_query": user_query})
    return cypher.strip()

def run_cypher_query(cypher_query: str):
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            records = result.data()
            if not records:
                records = [{"message": "No Data Found"}]
            
            return records

        except Exception as e:
            print("Error running query:", e)
            print("Query was:", cypher_query)

# This function is for the 'JointRetrieval' file
def execute_graph_retrieval_process(user_query: str):
    cypher_query = generate_cypher_from_query(user_query)
    result = run_cypher_query(cypher_query)
    return {
        "cypher_query" : cypher_query,
        "results": result
    }

def close_drivers():
    driver.close()

if __name__ == "__main__":
    user_query = input("Ask your financial graph question (e.g., 'Show Apple’s net income in 2020 10-K'): ")
    cypher_query = generate_cypher_from_query(user_query)
    print("\nGenerated Cypher query:\n", cypher_query)
    print("\nQuery Results:\n")
    run_cypher_query(cypher_query)
    driver.close()
