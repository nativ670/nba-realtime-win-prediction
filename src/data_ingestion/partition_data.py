import pandas as pd
import os
import shutil

# --- Configuration ---
RAW_FILE = "data/processed/training_data.parquet"
OUTPUT_DIR = "data/processed/seasons"

def partition_data():
    if not os.path.exists(RAW_FILE):
        print(f"Error: {RAW_FILE} not found.")
        return

    print(f"Loading {RAW_FILE}...")
    df = pd.read_parquet(RAW_FILE)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract season from GAME_ID
    # Format: 002YYnnnnn -> YY is index 3, 4
    print("Extracting seasons and partitioning...")
    df['season_year'] = df['GAME_ID'].str[3:5].astype(int)
    # Convert to full year (assuming 20xx for the last 10 years)
    df['season_year'] = df['season_year'].apply(lambda x: 2000 + x if x < 50 else 1900 + x)

    seasons = df['season_year'].unique()
    for season in seasons:
        season_df = df[df['season_year'] == season].drop(columns=['season_year'])
        file_path = os.path.join(OUTPUT_DIR, f"pbp_{season}.parquet")
        print(f"  - Saving Season {season} to {file_path} ({len(season_df)} rows)")
        season_df.to_parquet(file_path, index=False)

    # Delete the original massive file
    print(f"Deleting original file: {RAW_FILE}")
    os.remove(RAW_FILE)
    print("Partitioning complete!")

if __name__ == "__main__":
    partition_data()
