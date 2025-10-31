import os
import json
from neo4j import GraphDatabase
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def load_label_mappings():
    json_path = Path(__file__).resolve().parent / "label_maps.json"
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        LABEL_NORMALIZATION = {}
        for category, mappings in label_data.items():
            LABEL_NORMALIZATION.update(mappings)
        return LABEL_NORMALIZATION
    except FileNotFoundError:
        print(f"Warning: label_maps.json not found at {json_path}")
        return {}

LABEL_NORMALIZATION = load_label_mappings()

def normalize_label(label):
    """
    Normalize financial statement labels to standard format.
    Returns the normalized label or original if no mapping exists.
    """
    if not label:
        return label
    normalized_key = label.strip().lower()
    return LABEL_NORMALIZATION.get(normalized_key, label)

def normalize_existing_labels(session):
    """
    Normalize existing LineItem labels in-place using the normalization mapping.
    Preserves original labels in 'original_label' property for reference.
    """
    print("Normalizing existing LineItem labels...")
    
    total_updated = 0
    
    # Process each label mapping individually to avoid syntax issues
    for original_label, normalized_label in LABEL_NORMALIZATION.items():
        query = """
            MATCH (li:LineItem)
            WHERE toLower(trim(li.label)) = $original
            SET li.original_label = CASE WHEN li.original_label IS NULL THEN li.label ELSE li.original_label END,
                li.label = $normalized
            RETURN count(li) AS updated_count
        """
        
        result = session.run(query, {"original": original_label.lower(), "normalized": normalized_label})
        count = result.single()["updated_count"]
        total_updated += count
    
    print(f"✓ Normalized {total_updated} LineItem labels")


def create_year_over_year_relationships(session):
    """
    Create temporal relationships between same line items across consecutive years.
    This enables trend analysis and YoY growth queries that JSONs cannot do efficiently.
    """
    print("Creating year-over-year relationships...")
    
    result = session.run("""
        // Link same line items from consecutive years
        MATCH (c:Company)-[:HAS_FILING]->(f:FilingType)
              -[:HAS_YEAR]->(y1:Year)-[:HAS_TABLE]->(t1)
              -[:HAS_ITEM]->(li1:LineItem)
        MATCH (c)-[:HAS_FILING]->(f)
              -[:HAS_YEAR]->(y2:Year)-[:HAS_TABLE]->(t2)
              -[:HAS_ITEM]->(li2:LineItem)
        WHERE toInteger(y2.filing_year) = toInteger(y1.filing_year) + 1
          AND li1.label = li2.label
          AND t1.table_type = t2.table_type
          AND li1.value IS NOT NULL
          AND li2.value IS NOT NULL
          AND toFloat(li1.value) <> 0
        WITH li1, li2, 
             ((toFloat(li2.value) - toFloat(li1.value)) / toFloat(li1.value) * 100) AS change_pct
        MERGE (li1)-[r:NEXT_YEAR]->(li2)
        SET r.change_percent = change_pct,
            r.change_absolute = toFloat(li2.value) - toFloat(li1.value)
        RETURN count(r) AS relationships_created
    """)
    
    count = result.single()["relationships_created"]
    print(f"✓ Created {count} year-over-year relationships")


def create_peer_company_relationships(session):
    """
    Link companies in same industry/sector for peer comparisons.
    Enables "how does X compare to its peers?" queries.
    """
    print("Creating peer company relationships...")
    
    result = session.run("""
        MATCH (c1:Company), (c2:Company)
        WHERE c1.industry IS NOT NULL
          AND c2.industry IS NOT NULL
          AND c1.industry = c2.industry
          AND c1 <> c2
        MERGE (c1)-[r:SAME_INDUSTRY {industry: c1.industry}]->(c2)
        RETURN count(r) AS relationships_created
    """)
    
    count = result.single()["relationships_created"]
    print(f"✓ Created {count} peer company relationships")


def create_metric_hierarchy_relationships(session):
    """
    Link parent-child financial metrics (e.g., Total Revenue = Product + Service Revenue).
    Enables drill-down analysis: "which component of revenue grew fastest?"
    """
    print("Creating metric hierarchy relationships...")
    
    # Total Revenue composition
    result1 = session.run("""
        MATCH (parent:LineItem)
        WHERE toUpper(parent.label) CONTAINS 'TOTAL' 
          AND toUpper(parent.label) CONTAINS 'REVENUE'
        MATCH (child:LineItem)
        WHERE (toUpper(child.label) CONTAINS 'PRODUCT' 
               OR toUpper(child.label) CONTAINS 'SERVICE')
          AND NOT toUpper(child.label) CONTAINS 'TOTAL'
          AND child.filing_year = parent.filing_year
          AND child.company_ticker = parent.company_ticker
          AND child.form_type = parent.form_type
          AND child.table_type = parent.table_type
        MERGE (parent)-[:COMPOSED_OF]->(child)
        RETURN count(*) AS revenue_relationships
    """)
    
    # Total Assets composition
    result2 = session.run("""
        MATCH (parent:LineItem)
        WHERE toUpper(parent.label) = 'TOTAL ASSETS'
        MATCH (child:LineItem)
        WHERE (toUpper(child.label) CONTAINS 'CURRENT ASSETS' 
               OR toUpper(child.label) CONTAINS 'NON-CURRENT ASSETS'
               OR toUpper(child.label) CONTAINS 'NONCURRENT ASSETS')
          AND NOT toUpper(child.label) CONTAINS 'TOTAL'
          AND child.filing_year = parent.filing_year
          AND child.company_ticker = parent.company_ticker
          AND child.form_type = parent.form_type
        MERGE (parent)-[:COMPOSED_OF]->(child)
        RETURN count(*) AS asset_relationships
    """)
    
    # Total Equity composition
    result3 = session.run("""
        MATCH (parent:LineItem)
        WHERE toUpper(parent.label) CONTAINS 'TOTAL' 
          AND toUpper(parent.label) CONTAINS 'EQUITY'
        MATCH (child:LineItem)
        WHERE (toUpper(child.label) CONTAINS 'COMMON STOCK'
               OR toUpper(child.label) CONTAINS 'RETAINED EARNINGS'
               OR toUpper(child.label) CONTAINS 'ACCUMULATED')
          AND NOT toUpper(child.label) CONTAINS 'TOTAL'
          AND child.filing_year = parent.filing_year
          AND child.company_ticker = parent.company_ticker
          AND child.form_type = parent.form_type
        MERGE (parent)-[:COMPOSED_OF]->(child)
        RETURN count(*) AS equity_relationships
    """)

    # Can add more results here for other metrics as needed
    
    count1 = result1.single()["revenue_relationships"]
    count2 = result2.single()["asset_relationships"]
    count3 = result3.single()["equity_relationships"]
    
    print(f"✓ Created {count1} revenue hierarchy relationships")
    print(f"✓ Created {count2} asset hierarchy relationships")
    print(f"✓ Created {count3} equity hierarchy relationships")


def create_performance_benchmarks(session):
    """
    Create industry benchmark nodes for percentile ranking.
    Enables "where does company X rank in its industry?" queries.
    """
    print("Creating performance benchmarks...")
    
    # Get available years and industries
    years_result = session.run("""
        MATCH (y:Year)
        WHERE y.form_type = "10-K"
        RETURN DISTINCT y.filing_year AS year
        ORDER BY year
    """)
    
    years = [record["year"] for record in years_result]
    benchmark_count = 0
    relationship_count = 0
    
    # Get all industries
    industries_result = session.run("""
        MATCH (c:Company)
        WHERE c.industry IS NOT NULL
        RETURN DISTINCT c.industry AS industry
    """)
    industries = [record["industry"] for record in industries_result]
    
    for year in years:
        for industry in industries:
            # Revenue benchmarks for each industry
            result = session.run("""
                MATCH (c:Company)-[:HAS_FILING]->(:FilingType {form_type: "10-K"})
                      -[:HAS_YEAR]->(y:Year {filing_year: $year})-[:HAS_TABLE]->(:IncomeStatement)
                      -[:HAS_ITEM]->(li:LineItem)
                WHERE c.industry = $industry
                  AND toUpper(li.label) CONTAINS 'TOTAL'
                  AND toUpper(li.label) CONTAINS 'REVENUE'
                WITH y.filing_year AS year,
                     percentileCont(toFloat(li.value), 0.5) AS median_revenue,
                     percentileCont(toFloat(li.value), 0.25) AS p25_revenue,
                     percentileCont(toFloat(li.value), 0.75) AS p75_revenue,
                     count(li) AS sample_size
                WHERE sample_size >= 3
                MERGE (b:IndustryBenchmark {
                    metric: 'Total Revenue',
                    industry: $industry,
                    year: year
                })
                SET b.median = median_revenue,
                    b.p25 = p25_revenue,
                    b.p75 = p75_revenue,
                    b.sample_size = sample_size,
                    b.computed_at = datetime()
                RETURN 1 AS created
            """, {"year": year, "industry": industry})
            
            if result.single():
                benchmark_count += 1
            
            # Net Income benchmarks for each industry
            result = session.run("""
                MATCH (c:Company)-[:HAS_FILING]->(:FilingType {form_type: "10-K"})
                      -[:HAS_YEAR]->(y:Year {filing_year: $year})-[:HAS_TABLE]->(:IncomeStatement)
                      -[:HAS_ITEM]->(li:LineItem)
                WHERE c.industry = $industry
                  AND toUpper(li.label) CONTAINS 'NET INCOME'
                  AND NOT toUpper(li.label) CONTAINS 'PER SHARE'
                WITH y.filing_year AS year,
                     percentileCont(toFloat(li.value), 0.5) AS median_income,
                     percentileCont(toFloat(li.value), 0.25) AS p25_income,
                     percentileCont(toFloat(li.value), 0.75) AS p75_income,
                     count(li) AS sample_size
                WHERE sample_size >= 3
                MERGE (b:IndustryBenchmark {
                    metric: 'Net Income',
                    industry: $industry,
                    year: year
                })
                SET b.median = median_income,
                    b.p25 = p25_income,
                    b.p75 = p75_income,
                    b.sample_size = sample_size,
                    b.computed_at = datetime()
                RETURN 1 AS created
            """, {"year": year, "industry": industry})
            
            if result.single():
                benchmark_count += 1
    
    # Create relationships from companies to their industry benchmarks
    result = session.run("""
        MATCH (c:Company), (b:IndustryBenchmark)
        WHERE c.industry = b.industry
        MERGE (c)-[r:BENCHMARKED_AGAINST]->(b)
        RETURN count(r) AS relationships_created
    """)
    
    relationship_count = result.single()["relationships_created"]
    
    print(f"✓ Created {benchmark_count} industry benchmark nodes")
    print(f"✓ Created {relationship_count} relationships to benchmarks")


def create_indices(session):
    """Create indices for better query performance"""
    print("Creating indices...")
    
    indices = [
        "CREATE INDEX line_item_label IF NOT EXISTS FOR (li:LineItem) ON (li.label)",
        "CREATE INDEX line_item_year IF NOT EXISTS FOR (li:LineItem) ON (li.filing_year)",
        "CREATE INDEX line_item_ticker IF NOT EXISTS FOR (li:LineItem) ON (li.company_ticker)",
        "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.company_ticker)",
        "CREATE INDEX company_industry IF NOT EXISTS FOR (c:Company) ON (c.industry)",
        "CREATE TEXT INDEX line_item_text IF NOT EXISTS FOR (li:LineItem) ON (li.label)"
    ]
    
    for index_query in indices:
        try:
            session.run(index_query)
        except Exception as e:
            print(f"Note: {e}")
    
    print("✓ Indices created/verified")


def main():
    """Run all relationship creation steps"""
    print("="*80)
    print("Creating Graph Relationships")
    print("="*80)
    
    with driver.session() as session:
        try:
            # Normalize existing labels first
            normalize_existing_labels(session)
            print()
            
            # Create indices for better performance
            create_indices(session)
            print()
            
            # Create semantic relationships
            create_year_over_year_relationships(session)
            print()
            
            create_peer_company_relationships(session)
            print()
            
            create_metric_hierarchy_relationships(session)
            print()
            
            create_performance_benchmarks(session)
            print()
            
            print("All graph relationships created successfully!")

        except Exception as e:
            print(f"\nError: {e}")
            raise
    
    driver.close()


if __name__ == "__main__":
    main()
