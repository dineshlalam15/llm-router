# ML-Based LLM Router

A genuine Machine Learning classification system utilizing dense SentenceTransformer embeddings (`all-MiniLM-L6-v2`) and an SVM / KNN architecture. It dynamically routes user queries to the optimal **LLM Model** (e.g., `gpt-4o-mini`, `gpt-4o`, `claude-3-5-sonnet`, `gemini-1-5-pro`, `gemini-1-5-flash`, `llama-3-1-8b-instruct`, `llama-3-1-70b-instruct`) based on learned semantic patterns with **zero LLM API calls**.

### Setup (Virtual Environment)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Generation & Training
```bash
# 1. Ingest benchmark datasets and generate profile-evaluated routing labels:
PYTHONPATH=. python -m router_engine.cli generate-data

# 2. Train and evaluate dense embeddings, SVM primary router, and KNN fallback:
PYTHONPATH=. python -m router_engine.cli train
```

### Configuration (`.env`)
You can configure default fallbacks and confidence thresholds directly in `.env`:
```env
CONFIDENCE_THRESHOLD=0.65
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
```
If ML confidence falls below `CONFIDENCE_THRESHOLD`, the router seamlessly falls back to your configured `DEFAULT_MODEL` and `DEFAULT_PROVIDER`.

### Running the API Service
```bash
uvicorn app.main:app --reload
```

### Running Tests
```bash
PYTHONPATH=. pytest -v
```