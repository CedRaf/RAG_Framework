import os
import pickle

# Classifier globals (will be loaded from sec_classifier_model.pkl)
_vectorizer = None
_classifier = None
_reverse_mapping = None


def _load_classifier(model_path: str | None = None):
    global _vectorizer, _classifier, _reverse_mapping
    if model_path is None:
        cwd_path = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(cwd_path, "sec_classifier_model.pkl"),
            os.path.join(script_dir, "sec_classifier_model.pkl"),
            os.path.join(script_dir, "..", "sec_classifier_model.pkl"),
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"sec_classifier_model.pkl not found. Checked locations: cwd and {os.path.dirname(os.path.abspath(__file__))}"
        )

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    _vectorizer = model_data.get("vectorizer")
    _classifier = model_data.get("classifier")
    # reverse_mapping maps numeric label -> label name
    _reverse_mapping = model_data.get("reverse_mapping") or model_data.get("label_mapping")


# Attempt to load classifier at import time; swallow errors to keep module importable
try:
    _load_classifier()
except Exception:
    _vectorizer = None
    _classifier = None
    _reverse_mapping = None



def classify_query(user_query: str) -> str:
    """Classify a user query using the saved scikit-learn classifier.

    Returns the label name (e.g., 'TEXTUAL' or 'TABLE').
    """
    if _classifier is None or _vectorizer is None or _reverse_mapping is None:
        raise RuntimeError("Classifier model not loaded. Ensure 'sec_classifier_model.pkl' is available.")

    X = _vectorizer.transform([user_query])
    pred = _classifier.predict(X)[0]
    # if reverse mapping keyed by int labels
    label = _reverse_mapping.get(pred, str(pred)) if isinstance(_reverse_mapping, dict) else str(pred)
    return label

if __name__ == "__main__":
    try:
        user_query = input("Input the user query: ")
        classification = classify_query(user_query)
        print(classification)
    except Exception as e:
        print("Error classifying query:", e)