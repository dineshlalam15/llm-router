import joblib
from sklearn.svm import SVC

class SVMRouter:
    def __init__(self, kernel="rbf"):
        self.model = SVC(kernel=kernel, probability=True, class_weight="balanced", random_state=42)
    
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        best_idx = probs.argmax()
        return classes[best_idx], float(probs[best_idx]), dict(zip(classes, probs))
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)