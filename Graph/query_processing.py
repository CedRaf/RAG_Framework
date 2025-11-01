import json
import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

# Import company data for ticker mapping
try:
    from company_data import company_data
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

# Step 1: Extract information needs with available concepts shown
INFORMATION_EXTRACTION_PROMPT = """
You are a financial data analyst. Given a user query and the available financial concepts and labels for the mentioned companies, identify exactly which data fields are needed to answer the question.

User Query:
{user_query}

Available Financial Concepts for Mentioned Companies:
{available_concepts}

Guidelines:
1. **Statement Prioritization**
   - Income Statement → revenue, net income, profit, earnings, EPS, operating income, expenses.
   - Balance Sheet → assets, liabilities, equity, debt, cash, receivables, inventory.
   - Cash Flow Statement → operating cash flow, cash from operations, investing, financing, dividends, capital expenditures.

2. **Common Phrase Interpretation**
   - “Generated from operations” → Cash Flow Statement → Net Cash Provided by Operating Activities.
   - “Income from operations” → Income Statement → Operating Income.
   - “Cash position” → Balance Sheet → Cash and Cash Equivalents.
   - “Financial performance” → Income Statement (revenue, net income).
   - “Liquidity” → Balance Sheet (current assets/liabilities).
   - “Leverage” → Balance Sheet (debt, equity).
   - “Profitability” → Income Statement (margins, income).
   - “Growth” → year-over-year change.
   - “Expenditures” → Cash Flow Statement.

3. **Time Period**
   - Use the explicit year or range (e.g., “2023” or “2020–2024”).
   - If none specified, default to the most recent available year.

4. **Calculations**
   - Set `"calculation_needed": true` if the query asks for:
     ratios, percentages, comparisons, changes, or trends.
   - Describe the computation clearly in `"calculation_formula"`.
     Example: "(Net Income 2023 - Net Income 2022) / Net Income 2022".

5. **Multi-Company Comparisons**
   - Include one metric per company for consistent comparison.
   - Example: both companies’ “Net Income” for 2023.

6. **Missing or Irrelevant Data**
   - If a required metric isn’t in available_concepts, mark as “Data unavailable”.
   - Ignore qualitative or narrative topics (strategy, management, sentiment).

7. **Concept Selection Rules**
   - Prefer standardized US-GAAP/IFRS labels.
   - Prefer company-level totals (not segments).
   - Prefer numeric values (no abstracts/members).

8. **Language Patterns**
   - “How much did X generate/spend/pay” → Cash Flow (Operating/Investing/Financing).
   - “What were X’s total…” → Aggregate field.
   - “By how much did X increase/decrease…” → Difference between years.
   - “Compare X and Y” → same metric for both.
   - "What is X's ratio…" → extract both numerator and denominator.

Respond ONLY with valid JSON (no explanations, no extra text) in this structure:

{{
  "tickers": ["AAPL", "NVDA"],
  "selected_concepts": ["us-gaap_NetCashProvidedByUsedInOperatingActivities"],
  "selected_labels": ["Net Cash Provided by Operating Activities"],
  "time_period": "2023",
  "calculation_needed": false,
  "calculation_formula": ""
}}
"""

# Step 2: Cypher Generation Prompt (with financial data context)
QUERY_PROMPT_TEMPLATE = """
You are a Cypher expert for querying financial data from a Neo4j graph database.

**Graph Schema:**

Nodes:
- (Company) - properties: company_name, company_ticker, industry
- (FilingType) - properties: form_type ("10-K" or "10-Q"), company_ticker
- (Year) - properties: filing_year, form_type, company_ticker
- (Quarter) - properties: quarter ("Q1", "Q2", "Q3", "Q4"), filing_year, company_ticker
- (IncomeStatement|BalanceSheet|CashFlowStatement) - properties: table_type, filing_year, company_ticker, form_type
- (LineItem) - properties: label, value, concept, filing_year, company_ticker, form_type, table_type
- (AssetToDebtRatio) - properties: ratio, company_ticker, year, form_type
- (GrossProfitMargin) - properties: margin_percent, company_ticker, year, form_type
- (EquityToAssetRatio) - properties: ratio, company_ticker, year, form_type

Relationships:
- (Company)-[:HAS_FILING]->(FilingType)
- (FilingType)-[:HAS_YEAR]->(Year)
- (FilingType)-[:HAS_QUARTER]->(Quarter)
- (Year)-[:HAS_TABLE]->(IncomeStatement|BalanceSheet|CashFlowStatement)
- (Quarter)-[:HAS_TABLE]->(IncomeStatement|BalanceSheet|CashFlowStatement)
- (Table)-[:HAS_ITEM]->(LineItem)
- (LineItem)-[:NEXT_YEAR]->(LineItem) // Links same metric across consecutive years
- (Company)-[:PEER_IN_INDUSTRY {{industry: "..."}}>-(Company) // Links peer companies
- (Company)-[:HAS_METRIC]->(AssetToDebtRatio|GrossProfitMargin|EquityToAssetRatio)
- (Year)-[:HAS_METRIC]->(AssetToDebtRatio|GrossProfitMargin|EquityToAssetRatio)

**Extracted Information:**
{extracted_info}

**User Query:** 
{user_query}

**Critical Instructions:**

1. **Company Matching:** Use `c.company_ticker = "TICKER"` or `toUpper(c.company_ticker) = "TICKER"`

2. **LineItem Matching (Choose ONE approach):**
   - PRIMARY: Match by concept (most reliable): `li.concept = "us-gaap_ConceptName"`
   - FALLBACK: If concept not in extracted info, use label with flexible matching:
     `(li.label = "Exact Label" OR toUpper(li.label) CONTAINS toUpper("keyword"))`

3. **Year Matching:** 
   - Use `y.filing_year = "2023"` (string comparison)
   - For comparisons, fetch multiple years: `y.filing_year IN ["2022", "2023"]`

4. **Table Types:** 
   - Net Income, Revenue, Gross Profit → IncomeStatement
   - Assets, Liabilities, Equity → BalanceSheet  
   - Operating Cash Flow, Free Cash Flow → CashFlowStatement

5. **Form Type:** Default to `form_type: "10-K"` unless quarters mentioned

6. **Calculated Metrics:**
   - Gross Profit Margin → Check GrossProfitMargin nodes first
   - Asset to Debt Ratio → Check AssetToDebtRatio nodes first
   - Free Cash Flow = Operating Cash Flow - Capital Expenditures
   - Free Cash Flow Yield = Free Cash Flow / Market Cap (NOTE: Market cap is NOT in database - answer with absolute FCF instead)
   - If pre-calculated nodes don't exist, fetch LineItems and calculate in query

7. **Data Limitations - IMPORTANT:**
   - Market capitalization is NOT stored in the database (external market data)
   - Stock prices, P/E ratios, and market-based metrics are NOT available
   - If query asks for market-based ratios (like FCF yield), return the underlying metrics instead
   - Let the answer generation explain what data is available vs unavailable

8. **WITH Clauses - CRITICAL:**
   - When using WITH, you MUST carry forward ALL variables you need later
   - If you match a node (like `t:CashFlowStatement`) and use it in RETURN, include it in WITH
   - Example: `WITH c, f, y, t, li` NOT just `WITH c, f, y, li`
   - Each WITH clause drops variables not explicitly listed

9. **Return Clause:** Always return:
   ```
   c.company_ticker AS ticker, c.company_name AS name,
   f.form_type AS filing_type, y.filing_year AS year,
   li.label AS line_item, li.concept AS concept, li.value AS value,
   t.table_type AS source_table
   ```

10. **Multi-Company Queries:** Use UNION or multiple MATCH clauses, ensure both companies return data

11. **Complex Queries with Multiple Metrics:**
    - AVOID using multiple WITH clauses that need to reference early nodes
    - PREFER: Use OPTIONAL MATCH for additional data instead of WITH chains
    - OR: Simplify to return just the essential LineItems and let the answer generation do calculations

**Example Query Pattern (Net Income for multiple companies):**

```
MATCH (c:Company)-[:HAS_FILING]->(f:FilingType {{form_type: "10-K"}})
      -[:HAS_YEAR]->(y:Year {{filing_year: "2023"}})
      -[:HAS_TABLE]->(t:IncomeStatement)-[:HAS_ITEM]->(li:LineItem)
WHERE c.company_ticker IN ["AAPL", "NVDA"]
  AND li.concept = "us-gaap_NetIncomeLoss"
RETURN c.company_ticker AS ticker, c.company_name AS name,
       f.form_type AS filing_type, y.filing_year AS year,
       li.label AS line_item, li.concept AS concept, li.value AS value,
       t.table_type AS source_table
ORDER BY c.company_ticker
```

**Example Query Pattern (Gross Profit Margin from pre-calculated node):**

```
MATCH (c:Company)-[:HAS_METRIC]->(m:GrossProfitMargin {{year: "2024"}})
WHERE c.company_ticker = "NVDA"
RETURN c.company_ticker AS ticker, c.company_name AS name,
       m.year AS year, m.margin_percent AS value,
       "GrossProfitMargin" AS line_item,
       "Pre-calculated metric" AS source_table
```

**Example Query Pattern (Multiple metrics - using OPTIONAL MATCH, NOT WITH chains):**

```
MATCH (c:Company)-[:HAS_FILING]->(f:FilingType {{form_type: "10-K"}})
      -[:HAS_YEAR]->(y:Year {{filing_year: "2022"}})
      -[:HAS_TABLE]->(t1:CashFlowStatement)-[:HAS_ITEM]->(li1:LineItem)
WHERE c.company_ticker IN ["MCHP", "MRVL"]
  AND li1.concept = "us-gaap_NetCashProvidedByUsedInOperatingActivities"
OPTIONAL MATCH (c)-[:HAS_FILING]->(:FilingType {{form_type: "10-K"}})
      -[:HAS_YEAR]->(:Year {{filing_year: "2022"}})
      -[:HAS_TABLE]->(t2:BalanceSheet)-[:HAS_ITEM]->(li2:LineItem)
WHERE li2.concept = "us-gaap_Assets"
RETURN c.company_ticker AS ticker, c.company_name AS name,
       y.filing_year AS year,
       li1.label AS operating_cash_flow_label, li1.value AS operating_cash_flow,
       li2.label AS assets_label, li2.value AS total_assets
ORDER BY c.company_ticker
```

**Example Query Pattern (Free Cash Flow - Operating CF minus CapEx):**

```
MATCH (c:Company)-[:HAS_FILING]->(f:FilingType {{form_type: "10-K"}})
      -[:HAS_YEAR]->(y:Year {{filing_year: "2022"}})
      -[:HAS_TABLE]->(t_op:CashFlowStatement)-[:HAS_ITEM]->(li_op:LineItem)
WHERE c.company_ticker IN ["MCHP", "MRVL"]
  AND li_op.concept = "us-gaap_NetCashProvidedByUsedInOperatingActivities"
OPTIONAL MATCH (c)-[:HAS_FILING]->(f)-[:HAS_YEAR]->(y)
      -[:HAS_TABLE]->(:CashFlowStatement)-[:HAS_ITEM]->(li_cap:LineItem)
WHERE li_cap.concept = "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment"
RETURN c.company_ticker AS ticker, c.company_name AS name,
       y.filing_year AS year,
       li_op.value AS operating_cash_flow,
       li_cap.value AS capex,
       (toFloat(li_op.value) - toFloat(li_cap.value)) AS free_cash_flow
ORDER BY c.company_ticker
```

**Example Query Pattern (Temporal comparison - multiple years):**

```
MATCH (c:Company)-[:HAS_METRIC]->(m:GrossProfitMargin)
WHERE c.company_ticker = "NVDA"
  AND m.year IN ["2022", "2023"]
RETURN c.company_ticker AS ticker, c.company_name AS name,
       m.year AS year, m.margin_percent AS margin
ORDER BY m.year
```

Generate ONLY the Cypher query, no explanations or markdown formatting.

Cypher Query:
"""

# Step 3: Generate final answer
ANSWER_GENERATION_PROMPT = """
You are a financial analyst expert. Using the retrieved financial data, answer the user's query directly, clearly, and with visible numeric reasoning.

User Query:
{user_query}

Graph Database Query Results:
{query_results}

Instructions:
1. Write your answer in the style of a concise financial report statement.
2. When the answer involves **derived values or ratios**, always show the **calculation explicitly** in the text.
   - Example formats:
     - "84,946M / 30,425M ≈ 2.79"
     - "$96,995,000,000 - $72,361,000,000 = $24,634,000,000 higher"
     - "($3,485M – $2,970M) / $2,970M ≈ 17.3% increase"
3. Use the same clear, factual tone as these examples:
   - "$10,742,000,000"
   - "$391,000,000"
   - "Decreased"
   - "The ratio was above 1 in 2020, dipped below 1 in 2021, then stayed slightly above 1 through 2024."
   - "Payments of Dividends is the largest."
4. If multiple companies are mentioned, perform an **explicit comparison**:
   - State who had the higher or lower value.
   - Include the difference or ratio, e.g.:
     - "Apple's net income ($96,995M) was higher than NVIDIA's ($72,361M) by $24,634M in 2023."
5. **Handling Partial Data - CRITICAL:**
   - If the query asks for a market-based metric (like "FCF yield") but only operating metrics are available, answer with the available data
   - Example: Query asks "Which had higher FCF yield?" but only FCF values available → Compare FCF values and note yield requires market cap
   - DO NOT say "Data unavailable" if you have the underlying metrics needed to answer the comparison
   - Focus on answering what CAN be answered from the available data
6. Avoid commentary, repetition of the question, or qualitative opinions.
7. Use correct units (M = million, B = billion) consistently.
8. Only say "Data unavailable" if NO relevant data exists at all (not if just one component is missing).

Final Output:
Provide only the concise, calculation-backed answer text (no JSON, no commentary).
"""

# ANSWER_GENERATION_PROMPT = """
# You are a financial analyst expert. Using the retrieved financial data, answer the user's query comprehensively.

# User Query: {user_query}

# Graph Database Query Results:
# {query_results}

# Instructions:
# 1. Provide specific numbers and calculations
# 2. Compare across companies if multiple are mentioned
# 3. Highlight trends and insights
# 4. Include data sources (tickers, years, filing types)
# 5. If calculations are needed, show your work
# 6. Be concise but thorough
# 7. Format numbers appropriately (use commas, dollar signs, percentages)

# Generate a clear, professional answer based on the data.
# """

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
        if re.search(rf'\b{ticker}\b', query_upper):
            mentioned_tickers.append(ticker)
            continue
        
        # Check 2: Full company name match
        # Remove legal suffixes for matching
        clean_name = re.sub(r'\s+(Inc\.|Corp\.|Corporation|Ltd\.|LLC|PLC|Class [A-Z]).*$', '', company_name, flags=re.IGNORECASE)
        
        if clean_name and re.search(rf'\b{re.escape(clean_name)}\b', user_query, re.IGNORECASE):
            mentioned_tickers.append(ticker)
            continue
        
        # Check 3: Single-word company names (e.g., Apple, Amazon, Netflix, Microsoft)
        first_word = clean_name.split()[0] if clean_name else ""
        if first_word and len(first_word) > 4 and re.search(rf'\b{first_word}\b', user_query, re.IGNORECASE):
            mentioned_tickers.append(ticker)
    
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