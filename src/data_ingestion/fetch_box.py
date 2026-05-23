import pandas as pd
import time
import os
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import SeasonTypeAllStar
from requests.exceptions import ReadTimeout, ConnectionError

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Align with 10-year window
CURRENT_YEAR = 2024
START_YEAR = CURRENT_YEAR - 10
SEASONS = [f"{year}-{str(year+1)[2:]}" for year in range(START_YEAR, CURRENT_YEAR + 1)]

RAW_DATA_DIR = "data/raw"
BOX_FILE_PATH = os.path.join(RAW_DATA_DIR, "nba_box_scores_10y.parquet")

# API Request Settings
REQUEST_DELAY = 0.6 
MAX_RETRIES = 3

# ==============================================================================
# DATA INGESTION FUNCTIONS
# ==============================================================================

def fetch_box_scores(seasons):
    """
    Retrieves box scores (game results) for the specified seasons with retry logic.
    """
    all_games = []
    season_types = [SeasonTypeAllStar.regular, SeasonTypeAllStar.playoffs]
    
    print(f"Fetching box scores for seasons: {seasons}...")
    
    for season in seasons:
        for s_type in season_types:
            for attempt in range(MAX_RETRIES):
                try:
                    time.sleep(REQUEST_DELAY)
                    game_finder = leaguegamefinder.LeagueGameFinder(
                        season_nullable=season,
                        league_id_nullable='00', # NBA
                        season_type_nullable=s_type,
                        timeout=30
                    )
                    games = game_finder.get_data_frames()[0]
                    
                    # Basic cleaning: add season year for easier filtering later
                    games['season'] = int(season[:4])
                    
                    all_games.append(games)
                    print(f"  - {season} ({s_type}): Found {len(games)} team-game rows")
                    break
                except (ReadTimeout, ConnectionError) as e:
                    print(f"  - ⚠️ Timeout for {season} {s_type}! Retrying {attempt+1}/{MAX_RETRIES} in 5s...")
                    time.sleep(5)
                except Exception as e:
                    print(f"  - Error fetching box scores for {season} {s_type}: {e}")
                    break
                
    if not all_games:
        return pd.DataFrame()
    
    return pd.concat(all_games)

def run_box_ingestion():
    """
    Orchestrates the box score download and save process.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    df_box = fetch_box_scores(SEASONS)
    
    if not df_box.empty:
        # Standardize column names to lowercase for consistency
        df_box.columns = [col.lower() for col in df_box.columns]
        
        # We need to ensure we have opponent information for pregame features.
        # LeagueGameFinder returns two rows per game (one for each team).
        # We'll need to join them later or here. For now, let's save the raw team-game rows.
        
        print(f"Saving {len(df_box)} rows to {BOX_FILE_PATH}...")
        df_box.to_parquet(BOX_FILE_PATH, index=False)
        print("Success!")
    else:
        print("No box score data retrieved.")

# ==============================================================================
# MAIN EXECUTION (LIGHTWEIGHT TEST)
# ==============================================================================

if __name__ == "__main__":
    print("RUNNING LIGHTWEIGHT BOX SCORE TEST...")
    # Test with just the current season
    TEST_SEASONS = ["2023-24"]
    df_test = fetch_box_scores(TEST_SEASONS)
    
    if not df_test.empty:
        print("\nSample Data:")
        print(df_test[['TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP', 'WL']].head())
        print(f"\nTotal rows fetched for test: {len(df_test)}")
        assert 'GAME_ID' in df_test.columns, "Missing GAME_ID"
    else:
        print("Test failed: No data retrieved.")
