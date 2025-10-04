HOW TO ACTIVATE VIRTUAL ENVIRONMENT:
1. go to CMD and type the following commands:
    - cd "C:\Users\Cedric\Desktop\RAG THESIS\RAG_Framework\edgar-crawler"
    - venv312\Scripts\activate ---> to activate/enter the venv
    - deactive ---> to deactivate the venv

ENVIRONMENT (ENV) FILE REQUIREMENTS
1. Listed below are the details needed to be placed inside the .env file
    - NEO4J_URI=
    - NEO4J_USER=
    - NEO4J_PASS=
    - PERPLEXITY_API_KEY=

INITIAL FRAMEWORK SETUP:
1. Enter a virtual environment with an older Python version (3.12) ---> FOLLOW INSTRUCTIONS ABOVE
2. Once inside the edgar-crawler folder, edit the config.json file for your specifics
3. After editing, run the following files IN ORDER: download_filings.py and then extract_items.py
4. After successfully running these 2 files, exit the vevn and use the python terminal
5. Navigate into edgar-tools directory and run 2 files: 10KExtraction.py and 10QExtraction.py
6. Run ChunkData.py file found in main folder
7. Run EmbedDenseChunks.py
8. Run docker-compose.yml file to setup Neo4j and Milvus
    - (NOTE) run the resetConnections.py file if collections already exist in Milvus
9. Once the containers are up (Neo4j and Milvus), Run the following 3 files: InsertDenseData.py, InsertSparseData.py, InsertGraphData.py
10. Once all data is inserted, you can begin testing using JointRetrieval.py
    - (NOTE) This is an API endpoint, so you need to use it through postman or something
<----- REST OF THE INSTRUCTIONS TO FOLLOW ----->

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
```

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
