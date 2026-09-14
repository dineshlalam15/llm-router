import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text):
    """
    This function fixes the messy text. 
    For Example: "Analyze" (capital A) & "analyze" (lowercase a) 
        are seen as two completely different words by the computer.  
    """
    return str(text).lower().strip()


def fetch_top_features(model, vectorizer, n=5):
    """
    This function helps us to fetch the heavy weights which influence the selection of the provider. 

    args:
        1. model -  Logistic Regression Object 
        2. vectorizer - TF/IDF Vector Object
        3. n - top n features
    """
    feature_names = vectorizer.get_feature_names_out()
    classes = model.classes_ 
    for i, element in enumerate(classes):
        top_weights = np.argsort(model.coef_[i][-n:])
        top_features = [(feature_names[j], float(model.coef_[i][j])) for j in top_weights]
        print(f"\nProvider: {element.upper()}")
        for feature, weight in reversed(top_features):
            print(f"  + {feature}: {weight:.4f}")
    print("=" * 50 + "\n")


def main():
    print("Loading Dataset...")
    final_data = pd.read_csv('data/llm-router-training.csv')
    final_data = final_data.dropna()
    final_data['query_clean'] = final_data['query'].apply(clean_text)

    """
    We are going to allot 80% of our data to train the model & rest 20% to test.
    x = query field of the data 
    y = provider field of the data 
    """
    x_train, x_test, y_train, y_test = train_test_split(
        final_data['query_clean'],
        final_data['provider'],
        test_size=0.2, 
        random_state=42,
        stratify=final_data['provider']
    )

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    x_train_vector = vectorizer.fit_transform(x_train)
    x_test_vector = vectorizer.transform(x_test)

    print("Training our model...")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(x_train_vector,y_train)

    print("\nEvaluating Model...")
    y_pred = model.predict(x_test_vector)

    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test,y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y_test,y_pred,labels=model.classes_))

    fetch_top_features(model,vectorizer)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/router_model.joblib')
    joblib.dump(vectorizer, 'model/vectorizer.joblib')
    print("Model and vectorizer saved to 'model/' directory.")


if __name__ == "__main__":
    main()
