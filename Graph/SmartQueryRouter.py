import re
import os
import sys
import json
from typing import Dict, Optional, List
from pathlib import Path

# Import company data from centralized file
try:
    from company_data import company_data
except ImportError:
    # Fallback if import fails
    company_data = {}

# Load label mappings for precise metric extraction
def load_label_maps():
    """Load label mappings from JSON file"""
    try:
        label_maps_path = Path(__file__).parent / "label_maps.json"
        with open(label_maps_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

LABEL_MAPS = load_label_maps()

class SmartQueryRouter:
    """
    Route queries to predefined Cypher templates when possible.
    This reduces reliance on LLM-generated queries and improves consistency.
    """
    
    # Predefined Cypher templates for common query patterns
    TEMPLATES = {
        "single_metric": """
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
        """,
        
        "year_over_year_trend": """
            MATCH (c:Company {{company_ticker: "{ticker}"}})
                  -[:HAS_FILING]->(:FilingType {{form_type: "{form_type}"}})
                  -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:{table_type})
                  -[:HAS_ITEM]->(li:LineItem)
            WHERE (li.label = "{normalized_label}" OR toUpper(li.label) CONTAINS toUpper("{metric}"))
              AND toInteger(y.filing_year) >= {start_year}
              AND toInteger(y.filing_year) <= {end_year}
            WITH c, li, y
            ORDER BY y.filing_year
            WITH c.company_ticker AS ticker, collect({{
                year: y.filing_year,
                label: li.label,
                value: li.value
            }}) AS yearly_data
            UNWIND range(0, size(yearly_data)-1) AS idx
            WITH ticker, yearly_data[idx] AS current,
                 CASE WHEN idx > 0 
                      THEN yearly_data[idx-1]
                      ELSE null 
                 END AS previous
            RETURN ticker,
                   current.label AS line_item,
                   current.year AS year,
                   current.value AS value,
                   CASE WHEN previous IS NOT NULL AND toFloat(previous.value) <> 0
                        THEN round((toFloat(current.value) - toFloat(previous.value)) / toFloat(previous.value) * 100, 2)
                        ELSE null
                   END AS yoy_change_percent
            ORDER BY year
        """,
        
        "peer_comparison": """
            MATCH (c:Company {{company_ticker: "{ticker}"}})-[:SAME_INDUSTRY]->(peer:Company)
            WITH collect(DISTINCT peer.company_ticker) + ["{ticker}"] AS all_tickers
            UNWIND all_tickers AS ticker
            MATCH (comp:Company {{company_ticker: ticker}})
                  -[:HAS_FILING]->(:FilingType {{form_type: "{form_type}"}})
                  -[:HAS_YEAR]->(:Year {{filing_year: "{year}"}})
                  -[:HAS_TABLE]->(t:{table_type})-[:HAS_ITEM]->(li:LineItem)
            WHERE (li.label = "{normalized_label}" OR toUpper(li.label) CONTAINS toUpper("{metric}"))
            RETURN comp.company_ticker AS ticker, 
                   comp.company_name AS name,
                   li.label AS line_item,
                   li.value AS value,
                   t.table_type AS source_table
            ORDER BY toFloat(li.value) DESC
        """,
        
        "percentile_rank": """
            MATCH (target:Company {{company_ticker: "{ticker}"}})
                  -[:HAS_FILING]->(:FilingType {{form_type: "{form_type}"}})
                  -[:HAS_YEAR]->(:Year {{filing_year: "{year}"}})
                  -[:HAS_TABLE]->(t:{table_type})-[:HAS_ITEM]->(target_li:LineItem)
            WHERE (target_li.label = "{normalized_label}" OR toUpper(target_li.label) CONTAINS toUpper("{metric}"))
            WITH target, target_li.value AS target_value, target_li.label AS metric_label
            
            MATCH (c:Company)-[:SAME_INDUSTRY]->(target)
            MATCH (c)-[:HAS_FILING]->(:FilingType {{form_type: "{form_type}"}})
                  -[:HAS_YEAR]->(:Year {{filing_year: "{year}"}})
                  -[:HAS_TABLE]->()-[:HAS_ITEM]->(li:LineItem)
            WHERE (li.label = "{normalized_label}" OR toUpper(li.label) CONTAINS toUpper("{metric}"))
            WITH target, target_value, metric_label, collect(toFloat(li.value)) AS all_values
            WITH target, target_value, metric_label, all_values,
                 size([v IN all_values WHERE v <= toFloat(target_value)]) AS rank,
                 size(all_values) AS total
            RETURN target.company_ticker AS ticker,
                   target.company_name AS name,
                   metric_label AS line_item,
                   target_value AS value,
                   toFloat(rank) / total * 100 AS percentile,
                   rank AS rank_position,
                   total AS total_companies
        """,
        
        "hierarchical_breakdown": """
            MATCH (parent:LineItem)-[:COMPOSED_OF]->(child:LineItem)
            WHERE parent.company_ticker = "{ticker}"
              AND parent.filing_year = "{year}"
              AND parent.form_type = "{form_type}"
              AND (parent.label = "{normalized_label}" OR toUpper(parent.label) CONTAINS toUpper("{metric}"))
            RETURN parent.company_ticker AS ticker,
                   parent.label AS parent_metric,
                   parent.value AS parent_value,
                   collect({{
                       component: child.label,
                       value: child.value,
                       percentage: toFloat(child.value) / toFloat(parent.value) * 100
                   }}) AS components
        """,
        
        "multi_year_comparison": """
            MATCH (c:Company {{company_ticker: "{ticker}"}})
                  -[:HAS_FILING]->(:FilingType {{form_type: "{form_type}"}})
                  -[:HAS_YEAR]->(y:Year)-[:HAS_TABLE]->(t:{table_type})
                  -[:HAS_ITEM]->(li:LineItem)
            WHERE (li.label = "{normalized_label}" OR toUpper(li.label) CONTAINS toUpper("{metric}"))
              AND toInteger(y.filing_year) >= {start_year}
              AND toInteger(y.filing_year) <= {end_year}
            RETURN c.company_ticker AS ticker,
                   c.company_name AS name,
                   li.label AS line_item,
                   y.filing_year AS year,
                   li.value AS value
            ORDER BY y.filing_year
        """
    }
    
    # Metric to table type mapping
    METRIC_TABLE_MAP = {
        'revenue': 'IncomeStatement',
        'sales': 'IncomeStatement',
        'income': 'IncomeStatement',
        'profit': 'IncomeStatement',
        'loss': 'IncomeStatement',
        'earnings': 'IncomeStatement',
        'expense': 'IncomeStatement',
        'cost': 'IncomeStatement',
        'assets': 'BalanceSheet',
        'liabilities': 'BalanceSheet',
        'equity': 'BalanceSheet',
        'debt': 'BalanceSheet',
        'inventory': 'BalanceSheet',
        'receivables': 'BalanceSheet',
        'payables': 'BalanceSheet',
        'cash': 'BalanceSheet',
        'property': 'BalanceSheet',
        'equipment': 'BalanceSheet',
        'goodwill': 'BalanceSheet',
        'intangible': 'BalanceSheet',
        'stock': 'BalanceSheet',
        'cash flow': 'CashFlowStatement',
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
        
        # Determine table type based on metric category or metric keyword
        table_type = "BalanceSheet"  # Default
        if metric_category:
            # Map categories to table types
            category_table_map = {
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
                'other': 'IncomeStatement'
            }
            table_type = category_table_map.get(metric_category, "BalanceSheet")
        elif metric:
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
    
    @classmethod
    def route(cls, query: str) -> Optional[Dict[str, str]]:
        """
        Route query to appropriate template or return None for LLM fallback.
        Returns dict with 'cypher' and 'method' keys if routed successfully.
        """
        entities = cls.extract_entities(query)
        query_lower = entities['query_lower']
        
        # Ensure normalized_label has a fallback value
        if not entities.get('normalized_label'):
            entities['normalized_label'] = ''
        
        # Check for comparison indicators
        is_comparison = any(word in query_lower for word in [
            'compare', 'versus', 'vs', 'compared to', 'peer', 'competitor',
            'higher', 'lower', 'more', 'less', 'better', 'worse',
            'which company', 'which one', ' or ', 'between'
        ])
        
        # Pattern 1: Simple single metric query
        # Match if we have ticker, year, and metric - don't require specific trigger words
        if entities['ticker'] and entities['year'] and entities['metric']:
            # Exclude if it's clearly asking about trends or comparisons
            if not any(word in query_lower for word in ['trend', 'change', 'growth']) and not is_comparison:
                # Exclude component/breakdown queries
                if not any(word in query_lower for word in ['component', 'breakdown', 'composition', 'rank', 'percentile']):
                    # Also check it's not a multi-year query
                    if len(entities['years']) == 1:
                        cypher = cls.TEMPLATES['single_metric'].format(**entities)
                        return {'cypher': cypher, 'method': 'template_single_metric'}
        
        # Pattern 2: Year-over-year trend (multi-year with "trend" or "change")
        if ('trend' in query_lower or 'change' in query_lower or 'growth' in query_lower):
            if len(entities['years']) >= 2 and entities['ticker'] and entities['metric']:
                entities['start_year'] = min(entities['years'])
                entities['end_year'] = max(entities['years'])
                
                cypher = cls.TEMPLATES['year_over_year_trend'].format(**entities)
                return {'cypher': cypher, 'method': 'template_yoy_trend'}
        
        # Pattern 3: Multi-year comparison (no trend keyword, just range)
        if len(entities['years']) >= 2 and entities['ticker'] and entities['metric']:
            if not any(word in query_lower for word in ['trend', 'growth', 'change']):
                entities['start_year'] = min(entities['years'])
                entities['end_year'] = max(entities['years'])
                cypher = cls.TEMPLATES['multi_year_comparison'].format(**entities)
                return {'cypher': cypher, 'method': 'template_multi_year'}
        
        # Pattern 4: Peer comparison
        if is_comparison:
            if entities['ticker'] and entities['year'] and entities['metric']:
                cypher = cls.TEMPLATES['peer_comparison'].format(**entities)
                return {'cypher': cypher, 'method': 'template_peer_comparison'}
        
        # Pattern 5: Percentile/ranking
        if any(word in query_lower for word in ['rank', 'percentile', 'where does', 'how does', 'position']):
            if entities['ticker'] and entities['year'] and entities['metric']:
                cypher = cls.TEMPLATES['percentile_rank'].format(**entities)
                return {'cypher': cypher, 'method': 'template_percentile'}
        
        # Pattern 6: Hierarchical breakdown (components, breakdown, composition)
        if any(word in query_lower for word in ['component', 'breakdown', 'composition', 'made up', 'composed of']):
            if entities['ticker'] and entities['year'] and entities['metric']:
                cypher = cls.TEMPLATES['hierarchical_breakdown'].format(**entities)
                return {'cypher': cypher, 'method': 'template_hierarchical'}
        
        # No template matched - return None to trigger LLM fallback
        return None


def test_router():
    """Test the router with various queries"""
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
