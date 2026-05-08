import numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib
import os

def train_recommender(scaled_features: np.ndarray, model_save_path: str = "models/knn_model.pkl"):
    print('Training kNN Recommender...')

    model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    model.fit(scaled_features)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    return model