import pandas as pd
import glob
import os
import sys
from tqdm import tqdm

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.in_game import calculate_in_game_features
from src.data_ingestion.update_pbp import _compute_rest_days, _compute_travel_distance, _get_home_away_abbrevs
from src.utils.nba_client import nba_api_call
from nba_api.stats.endpoints import leaguegamefinder

SEASONS_DIR = "data/processed/seasons"

def get_full_season_schedule(season_year_str):
    """
    Fetches the full season schedule using LeagueGameFinder.
    Format of season_year_str is 'YYYY-YY' e.g. '2023-24'
    """
    print(f"  Fetching full season schedule for {season_year_str}...")
    games = nba_api_call(
        leaguegamefinder.LeagueGameFinder,
        df_index=0,
        season_nullable=season_year_str,
        league_id_nullable='00'
    )
    if not games.empty:
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], errors='coerce')
    return games

def reprocess_all_seasons():
    files = sorted(glob.glob(os.path.join(SEASONS_DIR, "pbp_*.parquet")))
    
    if not files:
        print(f"No files found in {SEASONS_DIR}")
        return

    print(f"🚀 Reprocessing {len(files)} seasons to fix in-game features (fouls, bonus, timeouts)...")
    
    for file_path in files:
        year = os.path.basename(file_path).split('_')[1].split('.')[0]
        print(f"--- Processing Season {year} ---")
        
        try:
            df = pd.read_parquet(file_path)
            
            # Re-run the engine per game to ensure context is correct
            print(f"  Engineering features for {df['GAME_ID'].nunique()} games...")
            
            # Fetch schedule for the season to reconstruct pregame features
            season_start_year = int(year)
            season_str = f"{season_start_year}-{str(season_start_year + 1)[2:]}"
            games_df = get_full_season_schedule(season_str)
            
            if games_df.empty:
                print(f"  ❌ Failed to fetch schedule for {season_str}. Skipping pregame fixes.")
            
            processed_games = []
            
            for gid, df_game in tqdm(df.groupby('GAME_ID'), desc=f"Season {year}"):
                if 'EVENTNUM' in df_game.columns:
                    df_game = df_game.sort_values('EVENTNUM')
                
                df_fixed_game = calculate_in_game_features(df_game)
                
                if 'GAME_ID' not in df_fixed_game.columns:
                    df_fixed_game['GAME_ID'] = gid
                
                # --- RECALCULATE PREGAME FEATURES ---
                if not games_df.empty:
                    game_rows = games_df[games_df['GAME_ID'] == gid]
                    if not game_rows.empty:
                        game_date = game_rows['GAME_DATE'].iloc[0]
                        home_abbr, away_abbr = _get_home_away_abbrevs(game_rows)
                        
                        home_rest = _compute_rest_days(home_abbr, game_date, games_df) if home_abbr else 3
                        away_rest = _compute_rest_days(away_abbr, game_date, games_df) if away_abbr else 3
                        rest_advantage = home_rest - away_rest
                        
                        distance_traveled = _compute_travel_distance(
                            away_abbr, home_abbr, is_home=False
                        ) if (away_abbr and home_abbr) else 0.0
                        
                        df_fixed_game['rest_advantage'] = rest_advantage
                        df_fixed_game['distance_traveled'] = distance_traveled
                
                processed_games.append(df_fixed_game)
            
            df_fixed = pd.concat(processed_games, ignore_index=True)
            
            # Restore other metadata columns (elo, playoffs, etc.) but EXCLUDE rest/distance since we fixed them
            pregame_cols = [
                'GAME_ID', 'elo_advantage', 
                'is_playoffs', 'home_win', 'live_raptor_advantage',
                'my_elo_pre', 'opp_elo_pre'
            ]
            actual_pregame_cols = [c for c in pregame_cols if c in df.columns]
            
            if actual_pregame_cols:
                df_meta = df[actual_pregame_cols].drop_duplicates('GAME_ID')
                cols_to_drop = [c for c in actual_pregame_cols if c in df_fixed.columns and c != 'GAME_ID']
                df_fixed = df_fixed.drop(columns=cols_to_drop)
                df_fixed = df_fixed.merge(df_meta, on='GAME_ID', how='left')

            # Save back
            df_fixed.to_parquet(file_path, index=False)
            print(f"  ✅ Saved updated {file_path} ({len(df_fixed)} rows)")
            
        except Exception as e:
            print(f"  ❌ Error processing {year}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    reprocess_all_seasons()
