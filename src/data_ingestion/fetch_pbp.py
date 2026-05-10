import pandas as pd
import time
import os
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2
from nba_api.stats.library.parameters import SeasonTypeAllStar

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Fetch last 10 years of data
CURRENT_YEAR = 2024 # Adjust based on current season context
START_YEAR = CURRENT_YEAR - 10
SEASONS = [f"{year}-{str(year+1)[2:]}" for year in range(START_YEAR, CURRENT_YEAR + 1)]

RAW_DATA_DIR = "data/raw"
PBP_FILE_PATH = os.path.join(RAW_DATA_DIR, "nba_pbp_10y.parquet")
LOG_FILE_PATH = os.path.join(RAW_DATA_DIR, "pbp_fetch_progress.log")

# API Request Settings
REQUEST_DELAY = 0.6  # Seconds to wait between requests to avoid rate limits
MAX_RETRIES = 3

# ==============================================================================
# DATA INGESTION FUNCTIONS
# ==============================================================================

def get_game_ids(seasons):
    """
    Retrieves all regular season game IDs for a list of seasons.
    """
    all_games = []
    print(f"Fetching game IDs for seasons: {seasons}...")
    
    for season in seasons:
        try:
            # Query LeagueGameFinder for NBA games in the specific season
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00', # NBA
                season_type_nullable=SeasonTypeAllStar.regular # Regular Season
            )
            games = game_finder.get_data_frames()[0]
            all_games.append(games)
            print(f"  - {season}: Found {len(games)} games")
            time.sleep(REQUEST_DELAY) # Rate limiting
        except Exception as e:
            print(f"  - Error fetching IDs for {season}: {e}")
            
    if not all_games:
        return []
    
    combined_games = pd.concat(all_games).drop_duplicates(subset='GAME_ID')
    return combined_games['GAME_ID'].unique().tolist()

def fetch_pbp_for_game(game_id):
    """
    Fetches play-by-play data for a single game ID with retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            return pbp.get_data_frames()[0]
        except Exception as e:
            wait_time = (attempt + 1) * 2
            print(f"    [!] Error for Game {game_id} (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            print(f"    Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            
    return pd.DataFrame()

def run_ingestion(limit_games=None):
    """
    Orchestrates the PBP data download process.
    limit_games: Optional integer to limit the number of games (for testing).
    """
    # 1. Ensure directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 2. Get the list of Game IDs
    game_ids = get_game_ids(SEASONS)
    
    if limit_games:
        game_ids = game_ids[:limit_games]
        print(f"DEBUG: Limited to first {limit_games} games for testing.")

    total_games = len(game_ids)
    
    if total_games == 0:
        print("No games found. Exiting.")
        return

    print(f"Starting PBP download for {total_games} games...")
    
    all_pbp_data = []
    
    # 3. Loop through game IDs and fetch PBP
    for i, game_id in enumerate(game_ids):
        # Progress Tracking
        if i % 50 == 0:
            print(f"Progress: {i}/{total_games} games processed ({(i/total_games)*100:.1f}%)")
            
        df_pbp = fetch_pbp_for_game(game_id)
        
        if not df_pbp.empty:
            all_pbp_data.append(df_pbp)
        
        # Consistent rate limiting
        time.sleep(REQUEST_DELAY)
        
        # Partial save every 500 games to avoid memory/crash loss
        if i > 0 and i % 500 == 0:
            temp_df = pd.concat(all_pbp_data)
            temp_df.to_parquet(PBP_FILE_PATH, index=False)
            print(f"--- Checkpoint saved (parquet) at game {i} ---")

    # 4. Final consolidation and save
    if all_pbp_data:
        print("Finalizing and saving data...")
        final_df = pd.concat(all_pbp_data)
        final_df.to_parquet(PBP_FILE_PATH, index=False)
        print(f"Success! Data saved to {PBP_FILE_PATH}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("No PBP data was successfully retrieved.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Lightweight test: Fetch 2 games from the current season
    print("RUNNING LIGHTWEIGHT TEST (2 GAMES)...")
    TEST_SEASONS = ["2023-24"]
    # Temporarily override SEASONS for testing
    original_seasons = SEASONS
    SEASONS = TEST_SEASONS
    
    start_time = time.time()
    run_ingestion(limit_games=2)
    
    duration = (time.time() - start_time) / 60
    print(f"Test job completed in {duration:.2f} minutes.")
    
    # Reset SEASONS for future imports
    SEASONS = original_seasons
