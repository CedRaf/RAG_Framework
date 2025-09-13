import random
import requests
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

ds = load_dataset("intfloat/query2doc_msmarco", split="train")





def expand_query(query, k=4, model="sonar-pro"):
    few_shot_examples = random.sample(list(ds), k)
    
    prompt = "Write only the answer to the final query, in the same style as the examples.\n\n"
    for example in few_shot_examples:
        prompt += f"Q: {example['query']}\nA: {example['pseudo_doc']}\n\n"
    prompt += f"Q: {query}\nA:"

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are asked to write a passage that answers the given query. Do not ask the user for clarification."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,
        "max_tokens": 128
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        print("Error details:", response.text)
        raise
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()





def sparse_formatting(og_query, pseudo_doc, n=5):
    result = " ".join([og_query] * n) + " " + pseudo_doc
    
    return result

# Dense Retrieval Format
def dense_formatting(og_query, pseudo_doc):
    result = og_query + " [SEP] " + pseudo_doc
    
    return result


# query = "how many shares of common stock were issued and outstanding as of October 16, 2020"
# psuedo_doc= expand_query(query)
# final_output= sparse_formatting(query, psuedo_doc)
# print(final_output)
__all__ = ["expand_query", "sparse_formatting", "dense_formatting"]



