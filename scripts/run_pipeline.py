import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import load_and_clean_data
from src.features.build_features import scale_features
from src.models.train import train_recommender

def main():
    print("=== Starting MLOps Pipeline ===\n")

    raw_data_path = r"data/raw/Food_Nutrition_Dataset.csv"
    processed_data_path = r"data/processed/cleaned_food_data.csv"

    df_clean = load_and_clean_data(raw_data_path)

    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    df_clean.to_csv(processed_data_path, index=False)
    print(f"Cleaned dataset saved to {processed_data_path}\n")

    scaled_data, _ = scale_features(df_clean)

    train_recommender(scaled_data)
    print("\n=== Pipeline Execution Complete ===")

if __name__ == "__main__":
    main()