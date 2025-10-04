import os
from dotenv import load_dotenv
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

# Prompt template
QUERY_PROMPT_TEMPLATE = """
You are a classification assistant designed to decide the best source for answering a financial question. You must classify the following **user query** into one of two categories:

1. **TEXTUAL** — Narrative sections from SEC filings (10-K, 10-Q, 8-K), such as risk factors, business overviews, management discussions, disclosures, etc.  
2. **TABLE** — Structured financial data from statements (e.g., income statements, balance sheets, cash flow statements, or segment revenue tables).

### Rules:
- Classify as **TABLE** only if the query explicitly requests numerical/structured data found in income statements, balance sheets, cash flow statements, or related financial tables.  
- Otherwise, classify as **TEXTUAL**.  

### Classification Examples:

**Example 1:**  
Q: "What risks did Apple mention in its 2020 10-K filing?"  
A: TEXTUAL  

**Example 2:**  
Q: "Show me the net sales of iPhones from 2018 to 2020."  
A: TABLE  

**Example 3:**  
Q: "List all the subsidiaries of Microsoft mentioned in its latest 10-K."  
A: TEXTUAL  

**Example 4:**  
Q: "Give me the total R&D expenses over the last three years."  
A: TABLE  

---

### Now classify this query and return only the classification. NO ADDITIONAL TEXT:

Q: "{user_query}"

"""

def classify_query(user_query: str)-> str:
    prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    classification = chain.invoke({"user_query":user_query})
    return classification.strip()

if __name__ == "__main__":
    user_query = input("Input the user query: ")
    classification = classify_query(user_query)
    print(classification)