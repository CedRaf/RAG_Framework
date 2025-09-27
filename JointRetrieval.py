from QueryRouting import classify_query
from GraphQueryTest import execute_graph_retrieval_process, close_drivers
from QueryTest import execute_vector_retrieval
from GenerateFinalAnswer import GenerateFinalAnswer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel): 
    user_query: str

class QueryResponse(BaseModel):
    classification: str
    results: list[any] | None = None

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
            "results": result.get("results")
        }

        # Commented out for now because still haven't tested if it works
        #finalAnswer = GenerateFinalAnswer(result, request.user_query)
        #return(finalAnswer)

        return QueryResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "_main_":
    import uvicorn
    uvicorn.run("JointRetrieval:app", host="127.0.0.1", port=8000, reload=True)


