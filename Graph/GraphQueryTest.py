import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth
from SmartQueryRouter import SmartQueryRouter
from query_processing import process_query, generate_cypher_from_query, generate_answer

load_dotenv()

URI = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASS")

driver = GraphDatabase.driver(URI, auth=basic_auth(username, password))

def run_cypher_query(cypher_query: str):
    """Execute a Cypher query against the Neo4j database"""
    with driver.session() as session:
        try:
            result = session.run(cypher_query)
            records = result.data()
            if not records:
                return [{"message": "No matching data found in the graph database"}]
            return records
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            print(f"Query was:\n{cypher_query}")
            return [{"error": error_msg}]

def execute_graph_retrieval_with_rag(user_query: str):
    """
    Enhanced retrieval with smart routing + RAG pipeline with fallback.
    
    Flow:
    1. Try SmartQueryRouter template (fast path)
    2. If template succeeds with data → return result
    3. If template matches but returns no data → fallback to LLM pipeline
    4. If no template match → use LLM pipeline directly
    
    This ensures robustness: templates handle common queries efficiently,
    but LLM pipeline catches edge cases where templates fail.
    """
    # Step 1: Try smart routing first (fast path)
    print("=" * 60)
    print("Step 1: Smart Query Routing")
    print("=" * 60)
    
    routed_result = SmartQueryRouter.route(user_query)
    
    if routed_result:
        print(f"✅ Using rule-based template (FAST PATH)\n")
        cypher_query = routed_result['cypher']
        print(f"Generated Cypher Query:\n{cypher_query}\n")
        
        # Execute query directly
        print("=" * 60)
        print("Step 2: Executing Query")
        print("=" * 60)
        query_results = run_cypher_query(cypher_query)
        print(f"Query returned {len(query_results)} records\n")
        
        # Generate final answer
        print("=" * 60)
        print("Step 3: Generating Final Answer")
        print("=" * 60)
        final_answer = generate_answer(user_query, str(query_results))
        
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(final_answer)
        
        # Fallback mechanism: Check if template route failed
        # Conditions for fallback:
        # - Answer says "data unavailable"
        # - No query results
        # - Query results contain error or no-data message
        if "data unavailable" in final_answer.lower() or not query_results or (len(query_results) == 1 and ("error" in query_results[0] or "message" in query_results[0])):
            print("\n" + "⚠️" * 30)
            print("⚠️  Template route returned no data - falling back to LLM pipeline")
            print("⚠️" * 30 + "\n")
            # Continue to LLM pipeline below (don't return yet)
        else:
            # Template route succeeded - return result
            return {
                "user_query": user_query,
                "routing_method": "template",
                "cypher_query": cypher_query,
                "query_results": query_results,
                "final_answer": final_answer
            }
    else:
        # No template match - use full RAG pipeline
        print(f"❌ No template match, using LLM RAG pipeline\n")
    
    # Step 2-4: Extract companies and their concepts
    rag_context = process_query(user_query)
    
    if "error" in rag_context:
        print(f"ERROR: {rag_context['error']}")
        return rag_context
    
    information_needs = rag_context["information_needs"]
    company_concepts = rag_context.get("company_concepts", {})
    
    # Step 5: Generate Cypher with LLM (using filtered concepts)
    print("=" * 60)
    print("Step 5: Generating Cypher with extracted information")
    print("=" * 60)
    cypher_query = generate_cypher_from_query(user_query, information_needs, company_concepts)
    
    print(f"\nGenerated Cypher Query:")
    print("-" * 60)
    print(cypher_query)
    print("-" * 60)
    print()
    
    # Step 6: Execute the query
    print("=" * 60)
    print("Step 6: Executing Query")
    print("=" * 60)
    query_results = run_cypher_query(cypher_query)
    print(f"Query returned {len(query_results)} records\n")
    
    # If no results, print the error or issue
    if len(query_results) == 0 or (len(query_results) == 1 and ("error" in query_results[0] or "message" in query_results[0])):
        print("⚠️  WARNING: Query returned no valid data!")
        print(f"Results: {query_results}\n")
    
    # Step 7: Generate final answer
    print("=" * 60)
    print("Step 7: Generating Final Answer")
    print("=" * 60)
    final_answer = generate_answer(user_query, str(query_results))
    
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    
    return {
        **rag_context,
        "routing_method": "llm" if not routed_result else "template_fallback_to_llm",
        "cypher_query": cypher_query,
        "query_results": query_results,
        "final_answer": final_answer
    }

if __name__ == "__main__":
    test_queries = []
    
    for query in test_queries:
        print(f"\n\n{'='*80}")
        print(f"USER QUERY: {query}")
        print('='*80)
        result = execute_graph_retrieval_with_rag(query)
    
    driver.close()