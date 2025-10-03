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

QUERY_PROMPT_TEMPLATE = """
You are a financial analysis assistant. Your task is to generate an accurate and well-structured answer to the **user_query** using the retrieved passages provided.

**Instructions:**

1. Carefully read all retrieved passages.
2. Identify which passage(s) most directly and reliably answer the **user_query**.
3. Use only those passages to formulate a clear, comprehensive response.
4. If the passages do not contain the answer, state that the information is not available in the retrieved context. You may provide a cautious, general explanation if it is widely known and reliable, but never speculate or invent specific figures.
5. Present the answer in a professional, financial-analysis tone.
6. RETURN ONLY THE FINAL ANSWER, NO ADDITIONAL TEXTS

**user_query:** {user_query}

**retrieved_passages:**
{retrieved_passages}

**Final Answer:**

"""

def GenerateFinalAnswer(passages, query):
    prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    finalAnswer = chain.invoke({"user_query": query, "retrieved_passages": passages})
    return finalAnswer