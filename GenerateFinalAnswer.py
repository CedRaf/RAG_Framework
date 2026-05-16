import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# LLM setup using OpenAI 'gpt-4.1-nano'
chat_client = OpenAI(api_key=OPENAI_API_KEY)
CHAT_MODEL_NAME = "gpt-4.1-nano"

QUERY_PROMPT_TEMPLATE = """
You are a financial analysis assistant. Your task is to generate an accurate and well-structured answer to the *user_query* using the retrieved passages provided.

*Instructions:*

1. Carefully read all retrieved passages.
2. Identify which passage(s) most directly and reliably answer the *user_query*.
3. Use only those passages to formulate a clear, comprehensive response.
4. If the passages do not contain the answer, state that the information is not available in the retrieved context. You may provide a cautious, general explanation if it is widely known and reliable, but never speculate or invent specific figures.
5. Present the answer in a professional, financial-analysis tone.
6. RETURN ONLY THE FINAL ANSWER, NO ADDITIONAL TEXTS

*user_query:* {user_query}

*retrieved_passages:*
{retrieved_passages}

*Final Answer:*

"""

def GenerateFinalAnswer(passages, query):
    # Build prompt text from template and call OpenAI chat completion
    prompt_text = QUERY_PROMPT_TEMPLATE.format(user_query=query, retrieved_passages=passages)
    messages = [{"role": "user", "content": prompt_text}]

    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        messages=messages,
        temperature=OPENAI_TEMPERATURE,
    )

    finalAnswer = resp.choices[0].message.content
    return finalAnswer