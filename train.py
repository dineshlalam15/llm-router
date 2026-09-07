import pandas as pd

def clean_text(text: str):
    return str(text).lower().strip()

def main():
    print("Loading Dataset...")
    final_data = pd.read_csv('data/llm-router-training.csv')
    final_data = final_data.dropna()
    final_data['query_clean'] = final_data['query'].apply(clean_text)
