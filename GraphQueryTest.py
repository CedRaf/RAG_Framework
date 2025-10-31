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
   - **If the user query mixes metrics from different tables (e.g., "compare assets and net income"), search across multiple table types simultaneously.**
5. When multiple table types are required:
    - Use a `UNION` of subqueries, one per relevant table type, OR
    - Use a single MATCH with multiple optional patterns to collect data across related tables.
    - Each subquery must still respect the same company, form_type, and year filters.
6. Use appropriate table labels in the MATCH clause:
   - For IncomeStatement: `(t:IncomeStatement)`
   - For BalanceSheet: `(t:BalanceSheet)`
   - For CashFlowStatement: `(t:CashFlowStatement)`
   - For all tables: `(t)` where `t:IncomeStatement OR t:BalanceSheet OR t:CashFlowStatement`
7. If performing a cross-table metric (e.g., "net income divided by total assets" or "operating cash flow to net income ration"), retrieve both values in a single query using OPTIONAL MATCH or subqueries, then compute the ratio in the RETURN clause when possible. 
8. Return only:
   c.company_ticker AS ticker,
   c.company_name AS name,
   f.form_type AS filing_type,
   y.filing_year AS year,
   li.label AS line_item,
   li.value AS value
   t.table_type AS source_table

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

# This is another optimized prompt for reference, can try using this if the one above is not satisfactory.
#
# QUERY_PROMPT_TEMPLATE = """
# You are a Cypher expert for financial data queries. Generate a Cypher query based on the user's natural language question.

# ## Graph Schema:
# (Company)-[:HAS_FILING]->(FilingType)-[:HAS_YEAR]->(Year)-[:HAS_TABLE]->(IncomeStatement|BalanceSheet|CashFlowStatement)-[:HAS_ITEM]->(LineItem)

# ## Node Properties:
# - Company: company_name, company_ticker
# - FilingType: form_type, company_ticker
# - Year: filing_year, form_type, company_ticker
# - IncomeStatement/BalanceSheet/CashFlowStatement: table_type, filing_year, company_ticker, form_type
# - LineItem: label, value, filing_year, company_ticker, form_type, table_type

# ## Query Generation Rules:
# 1. **Case Normalization**: Always use toLower() or toUpper() for string comparisons
#    - Company: `toLower(c.company_name) CONTAINS toLower("input") OR toUpper(c.company_ticker) = toUpper("input")`
#    - Form type: `f.form_type = "10-K"` or `"10-Q"`
#    - Line items: `toUpper(li.label) CONTAINS toUpper("metric")`
# 2. **Year Filtering**: Always filter by filing_year as a string (e.g., `"2023"`)
# 3. **Form Type Selection**:
#    - If quarter mentioned → `"10-Q"`
#    - Otherwise → `"10-K"` (default)
# 4. **Table Selection**:
#    - Revenue, income, profit, expenses → IncomeStatement
#    - Assets, liabilities, equity, debt → BalanceSheet
#    - Cash flows → CashFlowStatement
# 5. **Multi-Table Queries**: Use OPTIONAL MATCH or UNION for cross-table metrics
# 6. **Return Format** (always in this order):
#    ```
#    RETURN c.company_ticker AS ticker,
#           c.company_name AS name,
#           f.form_type AS filing_type,
#           y.filing_year AS year,
#           li.label AS line_item,
#           li.value AS value,
#           t.table_type AS source_table
#    ```
# 7. **Output**: Return ONLY the Cypher query with no explanations, markdown, or code blocks
# 
# ## Examples:
#
# **Example 1 (Single-Table):**
# **Query**: "What was Apple's net income in 2023?"
# **Cypher**:
# MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)-[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:IncomeStatement)-[:HAS_ITEM]->(li:LineItem)
# WHERE (toLower(c.company_name) CONTAINS toLower("apple") OR toUpper(c.company_ticker) = "AAPL")
#   AND f.form_type = "10-K"
#   AND y.filing_year = "2023"
#   AND toUpper(li.label) CONTAINS toUpper("net income")
# RETURN c.company_ticker AS ticker, c.company_name AS name, f.form_type AS filing_type, y.filing_year AS year, li.label AS line_item, li.value AS value, t.table_type AS source_table
#
# **Example 2 (Multi-Table):**
# **Query**: "Show Tesla's revenue and total assets for 2023"
# **Cypher**:
# MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)-[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t)-[:HAS_ITEM]->(li:LineItem)
# WHERE (toLower(c.company_name) CONTAINS toLower("tesla") OR toUpper(c.company_ticker) = "TSLA")
#   AND f.form_type = "10-K"
#   AND y.filing_year = "2023"
#   AND (
#     (t:IncomeStatement AND toUpper(li.label) CONTAINS toUpper("revenue"))
#     OR
#     (t:BalanceSheet AND toUpper(li.label) CONTAINS toUpper("total assets"))
#   )
# RETURN c.company_ticker AS ticker, c.company_name AS name, f.form_type AS filing_type, y.filing_year AS year, li.label AS line_item, li.value AS value, t.table_type AS source_table
#
# ---
#
# User Query: {user_query}
#
# Cypher Query:
# """