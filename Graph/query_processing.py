import json
import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

from .prompts import (INFORMATION_EXTRACTION_PROMPT, QUERY_PROMPT_TEMPLATE, ANSWER_GENERATION_PROMPT)

# Import company data for ticker mapping
try:
    from .company_data import company_data
except ImportError:
    company_data = {}

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM with GPT-4
chat = ChatOpenAI(
    temperature=0,
    model="gpt-5-mini",
    api_key=OPENAI_API_KEY,
)

# Fix path to conceptMap10K.json
CONCEPT_MAP_PATH = Path(__file__).resolve().parent / "conceptMap10K.json"

def extract_companies_from_query(user_query: str) -> List[str]:
    """
    Extract company tickers from user query using regex and company_data mapping
    """
    query_upper = user_query.upper()
    mentioned_tickers = []
    
    # Check all companies for both ticker and name matches
    for ticker, company_name in company_data.items():
        # Skip if already found
        if ticker in mentioned_tickers:
            continue
        
        # Check 1: Exact ticker match (e.g., AAPL, GOOGL)
        # Skip tickers that are common English words - they'll be matched by company name only
        if ticker not in ['ON', 'COST']:
            if re.search(rf'\b{ticker}\b', query_upper):
                mentioned_tickers.append(ticker)
                continue
        
        # Check 2: Full company name match
        # Remove legal suffixes and trailing punctuation for matching
        clean_name = re.sub(r'\s+(Inc\.|Corp\.|Corporation|Ltd\.|LLC|PLC|Class [A-Z]).*$', '', company_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'[,\.]+$', '', clean_name)  # Remove trailing commas and periods
        
        if clean_name and re.search(rf'\b{re.escape(clean_name)}\b', user_query, re.IGNORECASE):
            mentioned_tickers.append(ticker)
            continue
        
        # Check 2b: Common alternate names (e.g., "Google" for "Alphabet Inc.", "Amazon" for "Amazon.com")
        alternate_names = {
            'GOOGL': ['Google', 'Alphabet'],
            'AMZN': ['Amazon']
        }
        if ticker in alternate_names:
            for alt_name in alternate_names[ticker]:
                if re.search(rf'\b{alt_name}\'?s?\b', user_query, re.IGNORECASE):
                    mentioned_tickers.append(ticker)
                    break
            if ticker in mentioned_tickers:
                continue
        
        # Check 3: Single-word company names (e.g., Apple, Amazon, Netflix, Microsoft)
        first_word = clean_name.split()[0] if clean_name else ""
        if first_word and len(first_word) > 3 and re.search(rf'\b{first_word}\'?s?\b', user_query, re.IGNORECASE):
            mentioned_tickers.append(ticker)
            continue
        
        # Check 4: Two-word company names (e.g., "Old Dominion")
        words = clean_name.split()
        if len(words) >= 2:
            first_two = f"{words[0]} {words[1]}"
            if re.search(rf'\b{re.escape(first_two)}\'?s?\b', user_query, re.IGNORECASE):
                mentioned_tickers.append(ticker)
                continue
    
    return mentioned_tickers

def load_concept_map() -> List[Dict[str, Any]]:
    try:
        with open(CONCEPT_MAP_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Concept map not found at {CONCEPT_MAP_PATH}")
        return []

def get_company_concepts(tickers: List[str]) -> Dict[str, Any]:
    concept_map = load_concept_map()
    company_concepts = {}
    
    for company_info in concept_map:
        ticker = company_info.get("ticker", "").upper()
        
        if ticker not in tickers:
            continue
        
        company_concepts[ticker] = {
            "balance_sheet": [],
            "cash_flow": [],
            "income_statement": []
        }
        
        # Extract balance sheet concepts
        if "multi_balance_sheet" in company_info:
            for item in company_info["multi_balance_sheet"]:
                concept = item.get("concept", "")
                label = item.get("label", "")
                if concept and label:
                    company_concepts[ticker]["balance_sheet"].append({
                        "concept": concept,
                        "label": label
                    })
        
        # Extract cash flow concepts
        if "multi_cash_flow" in company_info:
            for item in company_info["multi_cash_flow"]:
                concept = item.get("concept", "")
                label = item.get("label", "")
                if concept and label:
                    company_concepts[ticker]["cash_flow"].append({
                        "concept": concept,
                        "label": label
                    })
        
        # Extract income statement concepts
        if "multi_income_statement" in company_info:
            for item in company_info["multi_income_statement"]:
                concept = item.get("concept", "")
                label = item.get("label", "")
                if concept and label:
                    company_concepts[ticker]["income_statement"].append({
                        "concept": concept,
                        "label": label
                    })
    
    return company_concepts

def format_available_concepts(company_concepts: Dict[str, Any], filter_by_keywords: List[str] = None) -> str:
    """
    Format company concepts into a readable string for the LLM
    
    Args:
        company_concepts: Dictionary of company concepts
        filter_by_keywords: Optional list of keywords to determine which statement types to include
    """
    if not company_concepts:
        return "No companies identified in query."
    
    # Determine which statements to include based on keywords
    include_statements = {
        "balance_sheet": True,
        "cash_flow": True,
        "income_statement": True
    }
    
    if filter_by_keywords:
        # Reset to False if we have specific keywords
        include_statements = {
            "balance_sheet": False,
            "cash_flow": False,
            "income_statement": False
        }
        
        keywords_lower = [k.lower() for k in filter_by_keywords]
        
        # Balance sheet keywords
        bs_keywords = ["asset", "liability", "equity", "debt", "balance", "stockholder", "shareholder"]
        if any(kw in word for kw in bs_keywords for word in keywords_lower):
            include_statements["balance_sheet"] = True
        
        # Cash flow keywords
        cf_keywords = ["cash", "operating", "investing", "financing", "flow"]
        if any(kw in word for kw in cf_keywords for word in keywords_lower):
            include_statements["cash_flow"] = True
        
        # Income statement keywords
        is_keywords = ["revenue", "income", "profit", "earnings", "expense", "margin", "sales", "cost", "ebitda"]
        if any(kw in word for kw in is_keywords for word in keywords_lower):
            include_statements["income_statement"] = True
        
        # If no matches, include all (safer default)
        if not any(include_statements.values()):
            include_statements = {
                "balance_sheet": True,
                "cash_flow": True,
                "income_statement": True
            }
    
    formatted = ""
    
    for ticker, data in company_concepts.items():
        formatted += f"\n=== {ticker} ===\n"
        
        if include_statements["balance_sheet"] and data["balance_sheet"]:
            formatted += f"\nBalance Sheet ({len(data['balance_sheet'])} items):\n"
            for item in data["balance_sheet"][:15]:  # Show first 15
                formatted += f"  • Label: \"{item['label']}\"\n    Concept: {item['concept']}\n"
            if len(data["balance_sheet"]) > 15:
                formatted += f"  ... and {len(data['balance_sheet']) - 15} more\n"
        
        if include_statements["cash_flow"] and data["cash_flow"]:
            formatted += f"\nCash Flow Statement ({len(data['cash_flow'])} items):\n"
            for item in data["cash_flow"][:15]:
                formatted += f"  • Label: \"{item['label']}\"\n    Concept: {item['concept']}\n"
            if len(data["cash_flow"]) > 15:
                formatted += f"  ... and {len(data['cash_flow']) - 15} more\n"
        
        if include_statements["income_statement"] and data["income_statement"]:
            formatted += f"\nIncome Statement ({len(data['income_statement'])} items):\n"
            for item in data["income_statement"][:15]:
                formatted += f"  • Label: \"{item['label']}\"\n    Concept: {item['concept']}\n"
            if len(data["income_statement"]) > 15:
                formatted += f"  ... and {len(data['income_statement']) - 15} more\n"
        
        formatted += "\n"
    
    return formatted

def extract_information_needs(user_query: str, available_concepts: str) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(INFORMATION_EXTRACTION_PROMPT)
    chain = prompt | chat | StrOutputParser()
    
    response = chain.invoke({
        "user_query": user_query,
        "available_concepts": available_concepts
    })
    
    try:
        information_needs = json.loads(response)
        # Ensure tickers are uppercase
        information_needs["tickers"] = [t.upper() for t in information_needs.get("tickers", [])]
    except json.JSONDecodeError:
        print(f"Failed to parse LLM response: {response}")
        information_needs = {
            "tickers": [],
            "selected_concepts": [],
            "selected_labels": [],
            "time_period": "",
            "calculation_needed": False,
            "calculation_formula": ""
        }
    
    return information_needs

def filter_concepts_by_selection(
    company_concepts: Dict[str, Any], 
    selected_concepts: List[str], 
    selected_labels: List[str]
) -> str:
    """
    Filter company concepts to only include the ones selected by LLM
    This reduces context size for Cypher generation
    
    Args:
        company_concepts: Full company concepts dictionary
        selected_concepts: List of concept names selected by LLM
        selected_labels: List of label names selected by LLM
    
    Returns:
        Formatted string with only selected concepts
    """
    if not company_concepts:
        return "No concepts available."
    
    formatted = "Selected Financial Concepts (Filtered for Cypher Generation):\n"
    
    for ticker, data in company_concepts.items():
        formatted += f"\n=== {ticker} ===\n"
        
        # Filter balance sheet
        bs_filtered = [
            item for item in data["balance_sheet"]
            if item["concept"] in selected_concepts or item["label"] in selected_labels
        ]
        if bs_filtered:
            formatted += f"\nBalance Sheet ({len(bs_filtered)} items):\n"
            for item in bs_filtered:
                formatted += f"  • Label: \"{item['label']}\"\n    Concept: {item['concept']}\n"
        
        # Filter cash flow
        cf_filtered = [
            item for item in data["cash_flow"]
            if item["concept"] in selected_concepts or item["label"] in selected_labels
        ]
        if cf_filtered:
            formatted += f"\nCash Flow Statement ({len(cf_filtered)} items):\n"
            for item in cf_filtered:
                formatted += f"  • Label: \"{item['label']}\"\n    Concept: {item['concept']}\n"
        
        # Filter income statement
        is_filtered = [
            item for item in data["income_statement"]
            if item["concept"] in selected_concepts or item["label"] in selected_labels
        ]
        if is_filtered:
            formatted += f"\nIncome Statement ({len(is_filtered)} items):\n"
            for item in is_filtered:
                formatted += f"  • Label: \"{item['label']}\"\n    Concept: {item['concept']}\n"
        
        formatted += "\n"
    
    return formatted

def generate_cypher_from_query(user_query: str, extracted_info: Dict[str, Any], company_concepts: Dict[str, Any] = None) -> str:
    """
    Generate Cypher query with filtered concepts (only those selected by LLM)
    
    Args:
        user_query: User's natural language query
        extracted_info: Information needs extracted by LLM
        company_concepts: Optional full company concepts for filtering
    """
    # If company_concepts provided, filter to only selected concepts/labels
    if company_concepts:
        selected_concepts = extracted_info.get("selected_concepts", [])
        selected_labels = extracted_info.get("selected_labels", [])
        
        filtered_concepts = filter_concepts_by_selection(
            company_concepts, 
            selected_concepts, 
            selected_labels
        )
        
        print(f"\n📝 Filtered to {len(selected_concepts)} concepts and {len(selected_labels)} labels")
        print(f"Preview of filtered data:\n{filtered_concepts[:300]}...\n")
        
        # Add filtered concepts to extracted_info for the prompt
        extracted_info_with_context = {
            **extracted_info,
            "filtered_financial_data": filtered_concepts
        }
    else:
        extracted_info_with_context = extracted_info
    
    prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
    chain = prompt | chat | StrOutputParser()
    
    cypher = chain.invoke({
        "user_query": user_query,
        "extracted_info": json.dumps(extracted_info_with_context, indent=2)
    })
    
    # Extract just the Cypher query (remove markdown code blocks if present)
    cypher = cypher.strip()
    if cypher.startswith("```"):
        lines = cypher.split("\n")
        cypher = "\n".join(lines[1:-1])  # Remove first and last lines
    
    return cypher.strip()

def generate_answer(user_query: str, query_results: str) -> str:
    prompt = ChatPromptTemplate.from_template(ANSWER_GENERATION_PROMPT)
    chain = prompt | chat | StrOutputParser()
    
    response = chain.invoke({
        "user_query": user_query,
        "query_results": query_results
    })
    
    return response

def process_query(user_query: str) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("Step 1: Extracting companies from query using regex...")
    print("=" * 60)
    
    mentioned_tickers = extract_companies_from_query(user_query)
    print(f"Identified companies: {mentioned_tickers}\n")
    
    if not mentioned_tickers:
        print("⚠ No companies identified in query. Please mention at least one company.\n")
        return {
            "user_query": user_query,
            "mentioned_tickers": [],
            "available_concepts": "",
            "information_needs": {},
            "error": "No companies identified"
        }
    
    print("=" * 60)
    print("Step 2: Loading available concepts for companies...")
    print("=" * 60)
    
    company_concepts = get_company_concepts(mentioned_tickers)
    
    # For information extraction: Include ALL statements to give LLM full context
    # (User queries can be ambiguous, so we don't want to miss relevant concepts)
    available_concepts_full = format_available_concepts(company_concepts, filter_by_keywords=None)
    print(f"Loaded concepts for: {list(company_concepts.keys())}")
    print(f"All statements included for information extraction")
    print(f"Preview:\n{available_concepts_full[:500]}...\n")
    
    print("=" * 60)
    print("Step 3: Asking LLM to select relevant concepts...")
    print("=" * 60)
    
    information_needs = extract_information_needs(user_query, available_concepts_full)
    print(f"Selected information:\n{json.dumps(information_needs, indent=2)}\n")
    
    return {
        "user_query": user_query,
        "mentioned_tickers": mentioned_tickers,
        "available_concepts": available_concepts_full,
        "company_concepts": company_concepts,
        "information_needs": information_needs
    }