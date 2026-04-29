import random
import requests
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

def sparse_formatting(og_query,n=5):
    result = " ".join([og_query] * n)
    
    return result

# Dense Retrieval Format
def dense_formatting(og_query):
    result = og_query
    
    return result


# query = "how many shares of common stock were issued and outstanding as of October 16, 2020"
# psuedo_doc= expand_query(query)
# final_output= sparse_formatting(query, psuedo_doc)
# print(final_output)
__all__ = ["expand_query", "sparse_formatting", "dense_formatting"]



