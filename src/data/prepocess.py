import pandas as pd

def load_and_clean_data(file_path: str) -> pd.DataFrame:

    print(f"Loading data from {file_path}..." )
    df = pd.read_csv(file_path)

    columns_to_check = ['calories', 'protein', 'carbs', 'fat', 'iron', 'vitamin_c']
    df.dropna(subset=columns_to_check, inplace=True)

    df = df[df['calories'] > 0].copy()

    df['protein_per_100_cal'] = (df['protein'] / df['calories']) * 100

    print(f"Data cleaned. Final shape: {df.shape}")
    return df