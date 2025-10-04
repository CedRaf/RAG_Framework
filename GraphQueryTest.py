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

- A company name (may not be exact) OR ticker,
- A year (always provided),
- Optionally a quarter (if quarter mentioned → use "10-Q", otherwise default to "10-K"),
- And a financial metric (e.g., revenue, net income, assets, liabilities, cash flow, etc.),

Generate a Cypher query to retrieve the relevant data using the following graph schema:

Graph Schema:
- (Company)-[:HAS_FILING]->(FilingType)-[:HAS_YEAR]->(Year)-[:HAS_TABLE]->(IncomeStatement|BalanceSheet|CashFlowStatement)-[:HAS_ITEM]->(LineItem)

Node properties:
- Company: company_name, company_ticker
- FilingType: form_type, company_ticker
- Year: filing_year, form_type, company_ticker
- IncomeStatement/BalanceSheet/CashFlowStatement: table_type, filing_year, company_ticker, form_type
- LineItem: label, value, filing_year, company_ticker, form_type, table_type

Rules:
1. Always normalize case when comparing strings (use `toLower()` or `toUpper()`).
   - Match companies with:  
     `(toLower(c.company_name) CONTAINS toLower("company_input")) OR (toUpper(c.company_ticker) = toUpper("ticker_input"))`
   - Match `form_type` using `f.form_type = "10-K"` or `f.form_type = "10-Q"`.
   - Match `LineItem.label` with `CONTAINS` (not exact match), e.g., `toUpper(li.label) CONTAINS toUpper("net income")`.
2. Always filter by the given `filing_year` as a string.
3. If no quarter is mentioned in the query, default to form_type = "10-K".
4. Decide which table(s) to search based on the financial metric:
   - Revenue, income, profit, loss, operating expenses → `IncomeStatement`
   - Assets, liabilities, equity, debt, inventory → `BalanceSheet` 
   - Cash flow, operating cash, investing cash, financing cash → `CashFlowStatement`
   - If unclear, search across all three table types.
5. Use appropriate table labels in the MATCH clause:
   - For IncomeStatement: `(t:IncomeStatement)`
   - For BalanceSheet: `(t:BalanceSheet)`
   - For CashFlowStatement: `(t:CashFlowStatement)`
   - For all tables: `(t)` where `t:IncomeStatement OR t:BalanceSheet OR t:CashFlowStatement`
6. Return only:
   c.company_ticker AS ticker,
   c.company_name AS name,
   f.form_type AS filing_type,
   y.filing_year AS year,
   li.label AS line_item,
   li.value AS value

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
