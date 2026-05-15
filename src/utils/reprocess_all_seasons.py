import pandas as pd
import glob
import os
import sys
from tqdm import tqdm

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.in_game import calculate_in_game_features

SEASONS_DIR = "data/processed/seasons"

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
            
            # Use a list to collect processed game dataframes
            processed_games = []
            
            # Group by GAME_ID and process each game individually
            for gid, df_game in tqdm(df.groupby('GAME_ID'), desc=f"Season {year}"):
                # Ensure the game is sorted by EVENTNUM if available, or PCTIMESTRING
                if 'EVENTNUM' in df_game.columns:
                    df_game = df_game.sort_values('EVENTNUM')
                
                df_fixed_game = calculate_in_game_features(df_game)
                
                # Ensure GAME_ID is present in the result
                if 'GAME_ID' not in df_fixed_game.columns:
                    df_fixed_game['GAME_ID'] = gid
                
                processed_games.append(df_fixed_game)
            
            df_fixed = pd.concat(processed_games, ignore_index=True)
            
            # Restore any metadata columns that were in the original but not in the fixed output
            # (Specifically pre-game context and labels)
            pregame_cols = [
                'GAME_ID', 'elo_advantage', 'rest_advantage', 'distance_traveled', 
                'is_playoffs', 'home_win', 'live_raptor_advantage',
                'my_elo_pre', 'opp_elo_pre', 'days_rest_capped'
            ]
            actual_pregame_cols = [c for c in pregame_cols if c in df.columns]
            
            if actual_pregame_cols:
                # Get unique metadata per game from the original dataframe
                df_meta = df[actual_pregame_cols].drop_duplicates('GAME_ID')
                
                # Drop columns from df_fixed if they already exist to avoid duplicates during merge
                cols_to_drop = [c for c in actual_pregame_cols if c in df_fixed.columns and c != 'GAME_ID']
                df_fixed = df_fixed.drop(columns=cols_to_drop)
                
                # Merge metadata back
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
