import pandas as pd
import io
import requests

def fetch_elo_data():
    """
    Downloads Neil Paine's Elo Data from GitHub.
    This dataset includes historical Elo ratings and playoff markers for NBA games.
    """
    url = "https://raw.githubusercontent.com/Neil-Paine-1/NBA-elo/main/nba_elo.csv"
    try:
        print(f"Downloading Elo data from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        elo_df = pd.read_csv(io.StringIO(response.text))
        
        # Ensure date is datetime
        elo_df['date'] = pd.to_datetime(elo_df['date'])
        
        return elo_df
    except Exception as e:
        print(f"Error fetching Elo data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Lightweight test: fetch and display the first 5 rows
    elo_data = fetch_elo_data()
    if not elo_data.empty:
        print("Elo data fetch successful!")
        print(elo_data.head())
    else:
        print("Elo data fetch failed.")
