from QueryRouting import classify_query
from GraphQueryTest import execute_graph_retrieval_process, close_drivers
from QueryTest import execute_vector_retrieval
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel): 
    user_query: str

class QueryResponse(BaseModel):
     classification: str
     cypher_query: str | None = None
     graph_results: list | None = None
     textual_results: dict | None = None


def handle_tabular(query):
    print("Routing to Neo4j (GRAPH)")
    result = execute_graph_retrieval_process(query)
    close_drivers()
    return result

def handle_textual(query):
    print("Routing to Milvus (TEXTUAL)")
    result = execute_vector_retrieval(query)
    return result

def handle_both(query): # This function still iffy, cuz im not sure how the retrieval process will work
    graph_result = handle_tabular(query)
    text_result = handle_textual(query)
    return {
        "type" : "BOTH",
        "cypher_query": graph_result["cypher_query"],
        "graph_results": graph_result["results"],
        "textual_results": text_result["results"]
    }

classification_handlers = {
    "TEXTUAL": handle_textual,
    "TABLE": handle_tabular,
    "BOTH": handle_both,
}

def route_based_on_classification(classification: str, query: str):
    handler = classification_handlers.get(classification.upper())
    if handler:
        return handler(query)
    raise ValueError(f"No handler for classification: {classification}")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        classification = classify_query(request.user_query)
        result = route_based_on_classification(classification, request.user_query)

        response_data = {
            "classification": classification,
            "cypher_query": result.get("cypher_query"),
            "graph_results": result.get("results") if classification.upper() == "TABLE" else result.get("graph_results"),
            "textual_results": result.get("results") if classification.upper() == "TEXTUAL" else result.get("textual_results"),
        }

        return QueryResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


