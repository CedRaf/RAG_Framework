from QueryRouting import classify_query
from GraphQueryTest import execute_graph_retrieval_process, close_drivers
from QueryTest import execute_vector_retrieval
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
app = FastAPI()

class QueryRequest(BaseModel): 
    user_query: str

class QueryResponse(BaseModel):
     classification: str
     cypher_query: str | None = None
     graph_results: list | None = None
     textual_results: dict | None = None

FORCE_TEXTUAL_ONLY = True

def handle_tabular(query):
    print("Routing to Neo4j (GRAPH)")
    result = execute_graph_retrieval_process(query)
    close_drivers()
    return result

def handle_textual(query):
    print("Routing to Milvus (TEXTUAL)")
    result = execute_vector_retrieval(query)
    return result

classification_handlers = {
    "TEXTUAL": handle_textual,
    "TABLE": handle_tabular
}

def route_based_on_classification(classification: str, query: str):
    orig_classification = classification
    if FORCE_TEXTUAL_ONLY:
        print(f"TEMP: Original classification was {classification}, forcing TEXTUAL")
        classification = "TEXTUAL"
    handler = classification_handlers.get(classification.upper())
    if handler:
        return handler(query), classification
    raise ValueError(f"No handler for classification: {classification}")

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        classification = classify_query(request.user_query)
        result = route_based_on_classification(classification, request.user_query)

        result, used_classification = route_based_on_classification(classification, request.user_query)

        response_data = {
            "classification": used_classification,
            "cypher_query": result.get("cypher_query"),
            "graph_results": result.get("results") if used_classification.upper() == "TABLE" else result.get("graph_results"),
            "textual_results": result if used_classification.upper() == "TEXTUAL" else result.get("textual_results"),
        }
        #print("response data:", response_data)        
        return QueryResponse(**response_data)
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("JointRetrieval:app", host="0.0.0.0", port=8000, reload=True)


