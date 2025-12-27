# RAG_Framework Setup Guide

>This guide will help you set up and run the RAG_Framework for financial document retrieval and analysis.

## 1. Virtual Environment Setup

Open Command Prompt and run:

```sh
cd "RAG_Framework\edgar-crawler" # Navigate to the edgar-crawler directory
venv312\Scripts\activate  # Activate the Python 3.12 virtual environment
# To deactivate:
deactivate
```

## 2. Environment Variables

Create a `.env` file in the project root with the following keys:

```
NEO4J_URI=
NEO4J_USER=
NEO4J_PASS=
PERPLEXITY_API_KEY=
FINNHUB_API_KEY=
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
```

**Note:** The `MILVUS_URI` and `MILVUS_TOKEN` values shown above are default values for local development. Adjust them according to your Milvus configuration.

## 3. Initial Framework Setup

1. **Activate the virtual environment** (see above).
2. **Edit `config.json`** in the `edgar-crawler` folder to match your requirements.
3. **Run data extraction scripts:**
    - `download_filings.py`
    - `extract_items.py`
4. **Deactivate the virtual environment** and switch to your main Python terminal.
5. **Navigate to `edgar-tools` and run:**
    - `10KExtraction.py`
    - `10QExtraction.py`
6. **Run chunking and embedding scripts in the main folder:**
    - `ChunkData.py`
    - `EmbedDenseChunks.py`
7. **Start Neo4j and Milvus containers:**
    - Run `docker-compose.yml`
    - *Note:* If Milvus collections already exist, run `resetConnections.py` to clear them.
8. **Insert data into databases:**
    - `InsertDenseData.py`
    - `InsertSparseData.py`
    - `InsertGraphData.py`
9. **Test the API endpoint:**
    - Run `JointRetrieval.py` (this launches a FastAPI server)
    - Use Postman or similar tools to send requests to the API

---

**Note:**
- Always follow the order above for reliable setup.
- For API testing, ensure Neo4j and Milvus containers are running.
- More instructions may follow as the framework evolves.

---

For troubleshooting or further instructions, refer to the code comments or contact the project maintainer.
