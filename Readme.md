## Concept of the Router: 
A genuine Machine Learning classification system (TF-IDF + Logistic Regression). It dynamically routes user queries to the optimal LLM provider based on learned textual patterns.

### Recommended Setup (Virtual Environment)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload