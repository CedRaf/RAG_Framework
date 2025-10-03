from pymilvus import MilvusClient

# Connect
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

# Load collection into memory
client.load_collection("news_articles")

def sparse_retrieval_test_finnhub(sparse_query: str):
    sparse_results = client.search(
        collection_name="news_articles",
        data=[sparse_query],
        anns_field="sparse",  # Make sure this matches your schema field name
        limit=50,
        output_fields=["content", "ticker", "datetime", "news_id"],
        search_params={
            "metric_type": "BM25",
            "params": {"inverted_index_algo": "DAAT_MAXSCORE"}
        }
    )

    print("Sparse Results for finnhub/news_articles:\n")
    for i, result in enumerate(sparse_results[0]):
        print(f"Result {i+1}:")
        print(f"Score: {result['distance']}")
        print(f"Ticker: {result['entity']['ticker']}")
        print(f"Datetime: {result['entity']['datetime']}")
        print(f"Article ID: {result['entity']['news_id']}")
        print(f"Content: {result['entity']['content'][:300]}...")
        print("-" * 50)

    return sparse_results


sparse_retrieval_test_finnhub("How will Amazon enhance it's streaming services")
