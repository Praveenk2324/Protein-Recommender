import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="High-Protein Food Recommender API",
    description="Send target macros to receive mathematically similar food recommendations."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any frontend to talk to your API
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, "data", 'processed', 'cleaned_food_data.csv')

scaler = None
knn_model = None
df_lookup = None

@app.on_event("startup")
def load_artifacts():
    global scaler, knn_model, df_lookup
    try:
        scaler  = joblib.load(SCALER_PATH)
        knn_model = joblib.load(MODEL_PATH)
        df_lookup = pd.read_csv(DATA_PATH)
        print("ML Artifacts Loaded Succesfully")
    except Exception as e:
        print(f"ERROR Loading artifacts: {e}")

class MacroRequest(BaseModel):
    calories: float
    protein: float
    carbs: float
    fat: float
    iron: float
    vitamin_c: float

@app.post("/recommend")
def recommend_post(req:MacroRequest):

    if scaler is None or knn_model is None or df_lookup is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")
    if req.calories <=0:
        raise HTTPException(status_code=400, detail="Calories must be greater than 0 to calculate density.")
    
    protein_per_100_cal = (req.protein / req.calories) * 100

    features = np.array([[
        req.calories, req.protein, req.carbs, req.fat, 
        req.iron, req.vitamin_c, protein_per_100_cal
    ]])

    scaled_features = scaler.transform(features)
    distances, indices = knn_model.kneighbors(scaled_features, n_neighbors=5)

    recommendations = []
    for i in range(len(indices[0])):
        match_idx = indices[0][i]
        dist = float(distances[0][i])
        food_data = df_lookup.iloc[match_idx]

        recommendations.append({
            "rank": i + 1,
            "food_name": food_data['food_name'],
            "category": food_data['category'],
            "mathematical_distance": round(dist, 4),
            "macros": {
                "calories": float(food_data['calories']),
                "protein": float(food_data['protein']),
                "carbs": float(food_data['carbs']),
                "fat": float(food_data['fat'])
            }
        })
    
    return {
        "target_request": req.dict(),
        "recommendations": recommendations
    }

@app.get("/")
def serve_frontend():
    """Serves the HTML UI when someone visits the base URL."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))