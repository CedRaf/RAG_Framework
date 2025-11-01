import os
import json

BASE_DIR = r"C:\Users\Rhenz\Documents\School\CodeFolders\Thesis\RAG\RAGAS\10-Q (TABLES)\10-Q"
SUBDIRS = ["multi_balance_sheet", "multi_cash_flow", "multi_income_statement"]
STATEMENT_KEYS = {
    "multi_balance_sheet": "multi_balance_sheet",
    "multi_cash_flow": "multi_cash_flow",
    "multi_income_statement": "multi_income_statement"
}

concept_map = {}

for subdir in SUBDIRS:
    dir_path = os.path.join(BASE_DIR, subdir)
    if not os.path.isdir(dir_path):
        continue
    for fname in os.listdir(dir_path):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ticker = data.get("ticker")
        key = STATEMENT_KEYS[subdir]
        items = data.get(key, [])
        reduced = [{"label": x["label"], "concept": x["concept"]} for x in items if "label" in x and "concept" in x]
        if ticker not in concept_map:
            concept_map[ticker] = {"ticker": ticker}
        concept_map[ticker][key] = reduced

# Write output
out_path = os.path.join(BASE_DIR, "conceptMap.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(list(concept_map.values()), f, indent=2)
