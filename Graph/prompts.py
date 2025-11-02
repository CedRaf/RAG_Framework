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
   - "Generated from operations" → Cash Flow Statement → Net Cash Provided by Operating Activities.
   - "Income from operations" → Income Statement → Operating Income.
   - "Cash position" → Balance Sheet → Cash and Cash Equivalents.
   - "Financial performance" → Income Statement (revenue, net income).
   - "Liquidity" → Balance Sheet (current assets/liabilities).
   - "Leverage" → Balance Sheet (debt, equity).
   - "Profitability" → Income Statement (margins, income).
   - "Growth" → year-over-year change.
   - "Expenditures" → Cash Flow Statement.

3. **Time Period**
   - **ALWAYS extract ALL years mentioned** in temporal comparisons:
     * "from 2020 to 2022" → time_period: ["2020", "2022"] (NOT just "2022")
     * "between 2019 and 2023" → time_period: ["2019", "2020", "2021", "2022", "2023"]
     * "over the last 3 years" → time_period: [most_recent - 2, most_recent - 1, most_recent]
   
   - **Keywords indicating multiple years needed:**
     * increase, decrease, change, growth, decline, trend, over time
     * from...to..., between...and..., compared to, versus
     * trajectory, progression, evolution
   
   - **For single-point queries:**
     * "What is X's revenue in 2023?" → time_period: "2023" (single year OK)
   
   - **Default behavior:**
     * If NO time indicators AND NOT a comparison → most recent year
     * If comparison keywords BUT no years → last 2 years: [most_recent - 1, most_recent]

4. **Calculations**
   - Set `"calculation_needed": true` if the query asks for:
     ratios, percentages, comparisons, changes, trends, increases, decreases.
   
   - **For temporal calculations**, specify BOTH time points:
     * "increase from 2020 to 2022" → 
       calculation_formula: "(Value_2022 - Value_2020) / Value_2020 * 100"
       time_period: ["2020", "2022"]
     
     * "trend over 5 years" →
       calculation_formula: "Year-over-year changes for each consecutive year"
       time_period: ["2019", "2020", "2021", "2022", "2023"]
   
   - **For cross-statement calculations**, list ALL required metrics:
     * "Calculate ROE" →
       selected_concepts: ["us-gaap_NetIncomeLoss", "us-gaap_StockholdersEquity"]
       calculation_formula: "Net Income / Shareholders Equity"
       calculation_needed: true

5. **Multi-Company Comparisons**
   - **CRITICAL**: For comparison questions, extract ALL mentioned companies in tickers array
   - **"Which company had higher X: A or B?"** → tickers: ["A", "B"]
   - **"Between A and B, who..."** → tickers: ["A", "B"]
   - **"Compare A and B"** → tickers: ["A", "B"]
   - **"A vs B"** → tickers: ["A", "B"]
   - Include one metric per company for consistent comparison.
   - Example: both companies' "Net Income" for 2023.
   - If comparing over time: same metric, same years, multiple companies.
   - **NEVER extract only one company when question mentions multiple**

6. **Missing or Irrelevant Data**
   - If a required metric isn't in available_concepts, mark as "Data unavailable".
   - Ignore qualitative or narrative topics (strategy, management, sentiment).

7. **Concept Selection Rules**
   - Prefer standardized US-GAAP/IFRS labels.
   - Prefer company-level totals (not segments).
   - Prefer numeric values (no abstracts/members).

8. **Language Patterns**
   - "How much did X generate/spend/pay" → Cash Flow (Operating/Investing/Financing).
   - "What were X's total…" → Aggregate field.
   - **"By how much did X increase/decrease from Y1 to Y2"** → Extract BOTH years, set calculation_needed: true
   - **"Show the trend/progression/evolution"** → Extract year range, set calculation_needed: true
   - **"Compare X and Y"** → Extract BOTH companies in tickers array, same metric, same time period
   - **"Between X and Y, which/who"** → Extract BOTH companies in tickers array
   - **"X vs Y"** → Extract BOTH companies in tickers array
   - "What is X's ratio…" → extract both numerator and denominator.
   - **"What was the change between"** → Extract both time points
   - **"Growth rate from X to Y"** → Extract both years for calculation
   
   **CASH FLOW STATEMENT KEYWORDS - ALWAYS CashFlowStatement:**
   - **"Operating cash flow/activities"** → CashFlowStatement, us-gaap_NetCashProvidedByUsedInOperatingActivities
   - **"Cash from operations"** → CashFlowStatement, us-gaap_NetCashProvidedByUsedInOperatingActivities
   - **"Investing activities/cash"** → CashFlowStatement, us-gaap_NetCashProvidedByUsedInInvestingActivities
   - **"Financing activities/cash"** → CashFlowStatement, us-gaap_NetCashProvidedByUsedInFinancingActivities
   - **"Dividends paid"** → CashFlowStatement (Financing), us-gaap_PaymentsOfDividends
   - **"Share repurchases/buybacks"** → CashFlowStatement (Financing), us-gaap_PaymentsForRepurchaseOfCommonStock
   - **"Capital expenditures/CapEx"** → CashFlowStatement (Investing), us-gaap_PaymentsToAcquirePropertyPlantAndEquipment
   - **"Purchase of equipment/property"** → CashFlowStatement (Investing), NOT Balance Sheet
   
   **VALIDATION RULE**: If question contains ["paid", "spent", "purchased", "repurchased"] + ["dividends", "shares", "stock", "equipment", "property"] → **Cash Flow Statement**, NOT Balance Sheet

9. **Question Type Classification**
   Identify the query type explicitly:
   
   - **temporal_comparison**: Asks for change/difference between time periods
     → Requires multiple years in time_period
   
   - **trend_analysis**: Asks for pattern over time
     → Requires year range in time_period
   
   - **cross_statement_calculation**: Requires metrics from multiple statements
     → Requires multiple concepts in selected_concepts
   
   - **multi_company_comparison**: Compares same metric across companies
     → Requires multiple tickers
   
   - **simple_lookup**: Single metric, single year, single company
     → Standard retrieval

Respond ONLY with valid JSON (no explanations, no extra text) in this structure:

{{
  "query_type": "temporal_comparison",  // NEW: One of the 5 types above
  "tickers": ["AAPL", "NVDA"],
  "selected_concepts": ["us-gaap_NetCashProvidedByUsedInOperatingActivities"],
  "selected_labels": ["Net Cash Provided by Operating Activities"],
  "time_period": ["2020", "2022"],  // CHANGED: Can be array for temporal queries
  "calculation_needed": true,  // Set true for comparisons/calculations
  "calculation_formula": "(Value_2022 - Value_2020) / Value_2020 * 100"
}}

EXAMPLES:

Input: "What was the increase in Meta's cash from 2020 to 2022?"
Output: {{
  "query_type": "temporal_comparison",
  "tickers": ["META"],
  "selected_concepts": ["us-gaap_CashAndCashEquivalents"],
  "selected_labels": ["Cash and Cash Equivalents"],
  "time_period": ["2020", "2022"],
  "calculation_needed": true,
  "calculation_formula": "(Cash_2022 - Cash_2020)"
}}

Input: "Show the revenue trend for Apple from 2019 to 2023"
Output: {{
  "query_type": "trend_analysis",
  "tickers": ["AAPL"],
  "selected_concepts": ["us-gaap_RevenueFromContractWithCustomer"],
  "selected_labels": ["Revenue"],
  "time_period": ["2019", "2020", "2021", "2022", "2023"],
  "calculation_needed": true,
  "calculation_formula": "Year-over-year percentage changes"
}}

Input: "Calculate Tesla's ROE for 2023"
Output: {{
  "query_type": "cross_statement_calculation",
  "tickers": ["TSLA"],
  "selected_concepts": ["us-gaap_NetIncomeLoss", "us-gaap_StockholdersEquity"],
  "selected_labels": ["Net Income", "Stockholders Equity"],
  "time_period": "2023",
  "calculation_needed": true,
  "calculation_formula": "Net Income / Stockholders Equity"
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

Relationships:
- (Company)-[:HAS_FILING]->(FilingType)
- (FilingType)-[:HAS_YEAR]->(Year)
- (Year)-[:HAS_TABLE]->(IncomeStatement|BalanceSheet|CashFlowStatement)
- (Table)-[:HAS_ITEM]->(LineItem)

**Extracted Information:**
{extracted_info}

**User Query:** 
{user_query}

**Critical Instructions:**

1. **Query Type Routing**
   Based on extracted_info query_type, use appropriate pattern:
   - temporal_comparison: Retrieve ALL years in time_period array
   - trend_analysis: Retrieve full year range
   - cross_statement_calculation: Use OPTIONAL MATCH for multiple statements
   - multi_company_comparison: Use IN for ticker array
   - simple_lookup: Standard single MATCH

2. **Temporal Query Pattern**
   
   For queries with time_period as ARRAY (temporal_comparison, trend_analysis):
   
   MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
         -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:TableType)
         -[:HAS_ITEM]->(li:LineItem)
   WHERE c.company_ticker = "TICKER"
     AND y.filing_year IN [YEAR_ARRAY]
     AND li.concept = "us-gaap_Concept"
   RETURN c.company_ticker AS ticker,
          c.company_name AS name,
          y.filing_year AS year,
          li.label AS line_item,
          li.value AS value,
          t.table_type AS source_table
   ORDER BY y.filing_year

3. **Multi-Year Retrieval Example**
   
   Query: "What was the increase in Meta's cash from 2020 to 2022?"
   time_period: ["2020", "2022"]
   
   WHERE y.filing_year IN ["2020", "2022"]
   // Returns TWO rows: one for 2020, one for 2022
   // Both values needed to calculate increase

4. **Cross-Statement Calculation Pattern**
   
   When extracted_info has multiple selected_concepts:
   
   MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
         -[:HAS_YEAR]->(y:Year)
         -[:HAS_TABLE]->(t1:Statement1)-[:HAS_ITEM]->(li1:LineItem)
   WHERE c.company_ticker = "TICKER"
     AND y.filing_year = "YEAR"
     AND li1.concept = "us-gaap_Concept1"
   
   OPTIONAL MATCH (c)-[:HAS_FILING]->(f)-[:HAS_YEAR]->(y)
         -[:HAS_TABLE]->(t2:Statement2)-[:HAS_ITEM]->(li2:LineItem)
   WHERE li2.concept = "us-gaap_Concept2"
   
   RETURN c.company_ticker AS ticker,
          y.filing_year AS year,
          li1.label AS metric1_label,
          li1.value AS metric1_value,
          li2.label AS metric2_label,
          li2.value AS metric2_value

5. **Multi-Company Pattern**
   
   WHERE c.company_ticker IN ["TICKER1", "TICKER2"]
   ORDER BY c.company_ticker
   // Returns one row per company

6. **Company Matching**
   Use: c.company_ticker = "TICKER"
   Or: toUpper(c.company_ticker) = toUpper("TICKER")

7. **LineItem Matching - Choose ONE**
   - PRIMARY: By concept - li.concept = "us-gaap_ConceptName"
   - FALLBACK: By label - toUpper(li.label) CONTAINS toUpper("keyword")

8. **Form Type Selection**
   Default to: form_type = "10-K"
   Only use "10-Q" if query mentions quarters

9. **Table Type Routing**
   - Revenue, Net Income, Operating Income -> IncomeStatement
   - Assets, Liabilities, Equity, Debt, Cash -> BalanceSheet
   - Operating Cash Flow, Free Cash Flow -> CashFlowStatement

10. **Return Clause - Must Include**
    
    RETURN c.company_ticker AS ticker,
           c.company_name AS name,
           f.form_type AS filing_type,
           y.filing_year AS year,
           li.label AS line_item,
           li.concept AS concept,
           li.value AS value,
           t.table_type AS source_table

11. **Ordering**
    - For temporal queries: ORDER BY y.filing_year
    - For company comparisons: ORDER BY c.company_ticker
    - For multiple metrics: Order by company then year

**EXAMPLE 1: Temporal Comparison (Multi-Year)**

Query: "What was the increase in Meta's cash from 2020 to 2022?"
Extracted: query_type=temporal_comparison, time_period=["2020", "2022"]

MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
      -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:BalanceSheet)
      -[:HAS_ITEM]->(li:LineItem)
WHERE c.company_ticker = "META"
  AND f.form_type = "10-K"
  AND y.filing_year IN ["2020", "2022"]
  AND li.concept = "us-gaap_CashAndCashEquivalents"
RETURN c.company_ticker AS ticker,
       c.company_name AS name,
       y.filing_year AS year,
       li.label AS line_item,
       li.value AS value,
       t.table_type AS source_table
ORDER BY y.filing_year

**EXAMPLE 2: Trend Analysis (Year Range)**

Query: "Show revenue trend for Apple from 2019 to 2023"
Extracted: query_type=trend_analysis, time_period=["2019","2020","2021","2022","2023"]

MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
      -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:IncomeStatement)
      -[:HAS_ITEM]->(li:LineItem)
WHERE c.company_ticker = "AAPL"
  AND f.form_type = "10-K"
  AND y.filing_year IN ["2019", "2020", "2021", "2022", "2023"]
  AND li.concept = "us-gaap_RevenueFromContractWithCustomer"
RETURN c.company_ticker AS ticker,
       c.company_name AS name,
       y.filing_year AS year,
       li.label AS line_item,
       li.value AS value
ORDER BY y.filing_year

**EXAMPLE 3: Cross-Statement Calculation**

Query: "Calculate Tesla's ROE for 2023"
Extracted: query_type=cross_statement_calculation, selected_concepts=["us-gaap_NetIncomeLoss", "us-gaap_StockholdersEquity"]

MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
      -[:HAS_YEAR]->(y:Year)
      -[:HAS_TABLE]->(t1:IncomeStatement)-[:HAS_ITEM]->(li1:LineItem)
WHERE c.company_ticker = "TSLA"
  AND f.form_type = "10-K"
  AND y.filing_year = "2023"
  AND li1.concept = "us-gaap_NetIncomeLoss"

OPTIONAL MATCH (c)-[:HAS_FILING]->(f)-[:HAS_YEAR]->(y)
      -[:HAS_TABLE]->(t2:BalanceSheet)-[:HAS_ITEM]->(li2:LineItem)
WHERE li2.concept = "us-gaap_StockholdersEquity"

RETURN c.company_ticker AS ticker,
       y.filing_year AS year,
       li1.label AS net_income_label,
       li1.value AS net_income,
       li2.label AS equity_label,
       li2.value AS equity

**EXAMPLE 4: Multi-Company Comparison**

Query: "Compare cash positions of Meta and Google in 2022"
Extracted: query_type=multi_company_comparison, tickers=["META", "GOOGL"]

MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
      -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:BalanceSheet)
      -[:HAS_ITEM]->(li:LineItem)
WHERE c.company_ticker IN ["META", "GOOGL"]
  AND f.form_type = "10-K"
  AND y.filing_year = "2022"
  AND li.concept = "us-gaap_CashAndCashEquivalents"
RETURN c.company_ticker AS ticker,
       c.company_name AS name,
       y.filing_year AS year,
       li.label AS line_item,
       li.value AS value
ORDER BY c.company_ticker

**EXAMPLE 5: Simple Lookup**

Query: "What is Apple's 2023 revenue?"
Extracted: query_type=simple_lookup, tickers=["AAPL"], time_period="2023"

MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
      -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:IncomeStatement)
      -[:HAS_ITEM]->(li:LineItem)
WHERE c.company_ticker = "AAPL"
  AND f.form_type = "10-K"
  AND y.filing_year = "2023"
  AND li.concept = "us-gaap_RevenueFromContractWithCustomer"
RETURN c.company_ticker AS ticker,
       c.company_name AS name,
       y.filing_year AS year,
       li.label AS line_item,
       li.value AS value

**VALIDATION CHECKLIST:**

Before returning the query, verify:
1. If time_period is ARRAY -> Used IN [array] syntax
2. If time_period is STRING -> Used = "year" syntax
3. If temporal_comparison or trend_analysis -> Included y.filing_year in RETURN
4. If temporal query -> Included ORDER BY y.filing_year
5. If multiple concepts -> Used OPTIONAL MATCH pattern
6. If multiple tickers -> Used IN [ticker_array] syntax
7. Set correct table type (Income/Balance/CashFlow)
8. Set correct form_type ("10-K" or "10-Q")
9. Returned all required fields (ticker, name, year, line_item, value, source_table)

**GENERATE ONLY THE CYPHER QUERY - NO EXPLANATIONS, NO MARKDOWN**

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