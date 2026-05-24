import pandas as pd
import time
import os

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import SeasonTypeAllStar


from src.config import get_current_season_year, RAW_DIR, NBA_API_SLEEP
from src.utils.nba_client import nba_api_call

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Dynamically compute the season range — no more hardcoded year!
CURRENT_YEAR = get_current_season_year()
START_YEAR = CURRENT_YEAR - 10
SEASONS = [f"{year}-{str(year+1)[2:]}" for year in range(START_YEAR, CURRENT_YEAR + 1)]

RAW_DATA_DIR = str(RAW_DIR)
from src.config import RAW_BOX_PATH
BOX_FILE_PATH = str(RAW_BOX_PATH)

# API Request Settings
REQUEST_DELAY = NBA_API_SLEEP

# ==============================================================================
# DATA INGESTION FUNCTIONS
# ==============================================================================

def fetch_box_scores(seasons):
    """
    Retrieves box scores (game results) for the specified seasons.
    Uses shared nba_client for retry logic.
    """
    all_games = []
    season_types = [SeasonTypeAllStar.regular, SeasonTypeAllStar.playoffs]
    
    print(f"Fetching box scores for seasons: {seasons}...")
    
    for season in seasons:
        for s_type in season_types:
            games = nba_api_call(
                leaguegamefinder.LeagueGameFinder,
                season_nullable=season,
                league_id_nullable='00',
                season_type_nullable=s_type
            )
            if not games.empty:
                # Add season year for easier filtering later
                games['season'] = int(season[:4])
                all_games.append(games)
                print(f"  - {season} ({s_type}): Found {len(games)} team-game rows")
            else:
                print(f"  - {season} ({s_type}): No data or API error")
                
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
