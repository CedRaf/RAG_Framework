import re
import os
import sys
import json
from typing import Dict, Optional, List
from pathlib import Path

try:
    from company_data import company_data
except ImportError:
    # Fallback if import fails
    company_data = {}

# Load label mappings from JSON
def load_label_maps():
    try:
        label_maps_path = Path(__file__).parent / "label_maps.json"
        with open(label_maps_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

LABEL_MAPS = load_label_maps()

class SmartQueryRouter:
    # Single predefined template for simple lookups
    TEMPLATE = """
        MATCH (c:Company)-[:HAS_FILING]->(f:FilingType {{form_type: "{form_type}"}})
              -[:HAS_YEAR]->(y:Year {{filing_year: "{year}"}})
              -[:HAS_TABLE]->(t:{table_type})-[:HAS_ITEM]->(li:LineItem)
        WHERE (toLower(c.company_name) CONTAINS toLower("{company}") 
               OR toUpper(c.company_ticker) = toUpper("{ticker}"))
          AND (li.label = "{normalized_label}" OR toUpper(li.label) CONTAINS toUpper("{metric}"))
        RETURN c.company_ticker AS ticker, c.company_name AS name,
               li.label AS line_item, li.value AS value, 
               y.filing_year AS year, f.form_type AS filing_type,
               t.table_type AS source_table
    """
    
    # Metric to table type mapping (used for both label_map categories and keyword fallback)
    METRIC_TABLE_MAP = {
        # Label map categories
        'revenue': 'IncomeStatement',
        'gross_profit': 'IncomeStatement',
        'net_income': 'IncomeStatement',
        'operating_income': 'IncomeStatement',
        'cost': 'IncomeStatement',
        'operating_expense': 'IncomeStatement',
        'rd_expense': 'IncomeStatement',
        'sga_expense': 'IncomeStatement',
        'cash': 'BalanceSheet',
        'marketable_securities': 'BalanceSheet',
        'receivables': 'BalanceSheet',
        'inventory': 'BalanceSheet',
        'ppe': 'BalanceSheet',
        'intangibles': 'BalanceSheet',
        'assets': 'BalanceSheet',
        'payables': 'BalanceSheet',
        'liabilities': 'BalanceSheet',
        'debt': 'BalanceSheet',
        'equity': 'BalanceSheet',
        'cash_flow': 'CashFlowStatement',
        'other': 'IncomeStatement',
        # Keyword fallbacks
        'sales': 'IncomeStatement',
        'income': 'IncomeStatement',
        'profit': 'IncomeStatement',
        'loss': 'IncomeStatement',
        'earnings': 'IncomeStatement',
        'expense': 'IncomeStatement',
        'property': 'BalanceSheet',
        'equipment': 'BalanceSheet',
        'goodwill': 'BalanceSheet',
        'intangible': 'BalanceSheet',
        'stock': 'BalanceSheet',
        'operating cash': 'CashFlowStatement',
        'investing cash': 'CashFlowStatement',
        'financing cash': 'CashFlowStatement',
    }
    
    @staticmethod
    def extract_entities(query: str) -> Dict[str, Optional[str]]:
        """Extract key entities from natural language query"""
        query_lower = query.lower()
        
        # Build company name to ticker mapping dynamically from company_data
        company_patterns = {}
        for ticker_code, company_name in company_data.items():
            # Extract keywords from company name for matching
            name_lower = company_name.lower()
            
            # Remove legal suffixes and punctuation to get clean name
            legal_suffixes = ['.com, inc.', ', inc.', ' inc.', ' inc', ', inc', 
                            ' corporation', ' corp.', ' corp', 
                            ' plc', ' class a', ' class b', ' class c', 
                            ' limited', ' ltd.', ' ltd',
                            ' company', ' co.', ' co', 
                            ' n.v.', ' s.a.', ' llc', 
                            '.com', ',']
            clean_name = name_lower
            for suffix in legal_suffixes:
                clean_name = clean_name.replace(suffix, '')
            clean_name = clean_name.strip()
            
            # Add full clean name (always add this)
            company_patterns[clean_name] = ticker_code
            
            no_space_name = clean_name.replace(' ', '')
            if no_space_name != clean_name:
                company_patterns[no_space_name] = ticker_code
            
            no_punct_name = clean_name.replace("'", "").replace("-", " ").replace(".", "").strip()
            if no_punct_name != clean_name:
                company_patterns[no_punct_name] = ticker_code
                company_patterns[no_punct_name.replace(' ', '')] = ticker_code
            
            # For multi-word names, add variations with common suffixes removed
            common_suffixes = ['technologies', 'technology', 'platforms', 'solutions', 
                             'systems', 'services', 'group', 'holdings', 'international']
            for suffix in common_suffixes:
                if suffix in clean_name:
                    short_name = clean_name.replace(suffix, '').strip()
                    if short_name and len(short_name) > 2:
                        company_patterns[short_name] = ticker_code
            
            # Add first significant word for multi-word companies (or single word if appropriate)
            words = clean_name.split()
            if len(words) >= 1:
                # Find first word that's not a common word
                stop_words = {'the', 'and', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an'}
                for word in words:
                    if len(word) > 3 and word not in stop_words:
                        # Only add if not too generic and not already mapped to different company
                        if word not in company_patterns or company_patterns[word] == ticker_code:
                            company_patterns[word] = ticker_code
                        break
        
        # Extract company from query first (before ticker extraction)
        company = None
        ticker = None
        # Sort by length to match longer, more specific names first
        sorted_patterns = sorted(company_patterns.items(), key=lambda x: len(x[0]), reverse=True)
        for name, tick in sorted_patterns:
            if name in query_lower:
                company = name
                ticker = tick
                break
        
        # Extract ticker from query (2-5 uppercase letters)
        # Only extract if we haven't already found a company match
        # Exclude common words that might look like tickers
        if not ticker:
            excluded_words = {'HOW', 'THE', 'AND', 'FOR', 'ARE', 'WAS', 'WERE', 'WHAT', 'WHEN', 'WHERE', 'WHO', 'WHY', 'DID'}
            ticker_match = re.search(r'\b([A-Z]{2,5})\b', query)
            if ticker_match:
                potential_ticker = ticker_match.group(1).upper()
                if potential_ticker not in excluded_words:
                    ticker = potential_ticker
        
        # Extract years (4 digits)
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        year = year_matches[0] if year_matches else None
        
        # Determine form type
        form_type = "10-K"  # Default
        if any(q in query_lower for q in ['quarter', 'q1', 'q2', 'q3', 'q4', '10-q']):
            form_type = "10-Q"
        
        # Extract metric using label_maps for precise matching
        metric = None
        normalized_label = None
        metric_category = None
        
        # Build a flat list of all label variations with their normalized forms
        label_variations = []
        for category, mappings in LABEL_MAPS.items():
            for label_variant, normalized_form in mappings.items():
                label_variations.append((label_variant, normalized_form, category))
        
        # Sort by length (longest first) to match most specific labels first
        label_variations.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Try to find a matching label in the query
        for label_variant, normalized_form, category in label_variations:
            if label_variant in query_lower:
                metric = label_variant
                normalized_label = normalized_form
                metric_category = category
                break
        
        # Fallback to keyword matching if no label map match found
        if not metric:
            sorted_metrics = sorted(SmartQueryRouter.METRIC_TABLE_MAP.keys(), key=len, reverse=True)
            for metric_keyword in sorted_metrics:
                if metric_keyword in query_lower:
                    metric = metric_keyword
                    break
        
        # Determine table type from metric category or keyword
        table_type = "BalanceSheet"  # Default
        if metric_category:
            # Use the unified METRIC_TABLE_MAP
            table_type = SmartQueryRouter.METRIC_TABLE_MAP.get(metric_category, "BalanceSheet")
        elif metric:
            # Fallback to keyword matching
            table_type = SmartQueryRouter.METRIC_TABLE_MAP.get(metric, "BalanceSheet")
        
        return {
            'ticker': ticker,
            'year': year,
            'years': year_matches,
            'form_type': form_type,
            'metric': metric,
            'normalized_label': normalized_label,
            'table_type': table_type,
            'company': company,
            'query_lower': query_lower
        }
    
    @classmethod # Routes only for single metric queries cuz its hard to maintain templates for complex ones
    def route(cls, query: str) -> Optional[Dict[str, str]]:

        entities = cls.extract_entities(query)
        query_lower = entities['query_lower']
        
        # Ensure normalized_label has a fallback value
        if not entities.get('normalized_label'):
            entities['normalized_label'] = ''
        
        # Check if this is a simple single-metric query
        if entities['ticker'] and entities['year'] and entities['metric']:
            # Ensure it's truly simple - single year only
            if len(entities['years']) != 1:
                return None
            
            # Exclude complex query patterns
            complex_indicators = [
                'compare', 'versus', 'vs', 'compared to', 'peer', 'competitor',
                'higher', 'lower', 'more', 'less', 'better', 'worse',
                'which company', 'which one', ' or ', 'between',
                'trend', 'change', 'growth', 'over time',
                'rank', 'percentile', 'position',
                'component', 'breakdown', 'composition', 'made up', 'composed of',
                'ratio', 'calculate', 'compute', 'average', 'mean', 'median'
            ]
            
            if any(indicator in query_lower for indicator in complex_indicators):
                return None
            
            cypher = cls.TEMPLATE.format(**entities)
            return {'cypher': cypher, 'method': 'template_single_metric'}
        
        return None


def test_router():

    test_queries = [
        "What was Apple’s total assets in 2023?",
        "How much cash and cash equivalents did Amazon report in 2022?",
        "How much did MercadoLibre report as property and equipment, net in 2022?",
        "How did NVIDIA’s total assets change from 2020 to 2024?",
        "Which company had a higher asset-to-debt ratio in 2023, Apple or Google?"
    ]
    
    print("="*80)
    print("Smart Query Router Test")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = SmartQueryRouter.route(query)
        if result:
            print(f"✓ Routed via: {result['method']}")
            print(f"Cypher (truncated): {result['cypher']}")
        else:
            print("✗ No template match - would use LLM")


if __name__ == "__main__":
    test_router()
