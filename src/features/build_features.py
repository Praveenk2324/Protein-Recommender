import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def scale_features(df: pd.DataFrame, scaler_save_path: str = "models/scaler.pkl") -> tuple[np.ndarray, list]:
    print('Scaling features...')

    feature_cols = ['calories', 'protein', 'carbs', 'fat', 'iron', 'vitamin_c', 'protein_per_100_cal']
    features = df[feature_cols]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f'Scaler saved to {scaler_save_path}')

    return scaled_features, feature_cols