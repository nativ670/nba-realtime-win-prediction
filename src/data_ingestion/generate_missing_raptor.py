import pandas as pd
import numpy as np
import os
import sys
import time
from nba_api.stats.endpoints import leaguedashplayerstats

# --- Path Injection ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# --- Configuration & Weights (Reused from raptor_engine.py) ---
RAW_DATA_DIR = "data/raw"

OFF_WEIGHTS = {
    'intercept': -3.88704,
    'MPG': 0.026112,
    'PTS': 0.662784,
    'TSA': -0.51622,
    'AST': 0.430454,
    'TOV': -0.893465,
    'ORB': 0.303023,
    'DRB': -0.085637,
    'STL': 0.418092,
    'BLK': -0.230734,
    'PF': -0.108369
}

DEF_WEIGHTS = {
    'intercept': -3.079144,
    'MPG': 0.033637,
    'PTS': -0.081412,
    'TSA': 0.025422,
    'AST': -0.025109,
    'TOV': -0.055809,
    'ORB': -0.099034,
    'DRB': 0.191569,
    'STL': 1.150891,
    'BLK': 0.611107,
    'PF': 0.010649
}

def fetch_season_stats(season_str):
    """
    Fetches player stats for a specific season (Per 100 Possessions).
    """
    print(f"Fetching stats for {season_str}...")
    try:
        # Get Per 100 Possessions
        p100 = leaguedashplayerstats.LeagueDashPlayerStats(
            per_mode_detailed='Per100Possessions',
            season=season_str,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        
        time.sleep(1.0) # Rate limit protection
        
        # Get Per Game (for MPG)
        pg = leaguedashplayerstats.LeagueDashPlayerStats(
            per_mode_detailed='PerGame',
            season=season_str,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        
        # Select only MPG from PG and rename before merge to avoid conflicts
        df_pg_min = pg[['PLAYER_ID', 'MIN']].rename(columns={'MIN': 'MPG'})
        
        # Combine
        df = p100.merge(df_pg_min, on='PLAYER_ID', how='inner')
        
        return df
    except Exception as e:
        print(f"Error fetching stats for {season_str}: {e}")
        return pd.DataFrame()

def calculate_proxy_raptor(df):
    """
    Applies the proxy RAPTOR weights to the dataframe.
    """
    if df.empty:
        return df

    # TSA = FGA + 0.44 * FTA
    df['TSA'] = df['FGA'] + 0.44 * df['FTA']
    
    # Offensive RAPTOR
    df['raptor_off'] = (
        OFF_WEIGHTS['intercept'] +
        OFF_WEIGHTS['MPG'] * df['MPG'] +
        OFF_WEIGHTS['PTS'] * df['PTS'] +
        OFF_WEIGHTS['TSA'] * df['TSA'] +
        OFF_WEIGHTS['AST'] * df['AST'] +
        OFF_WEIGHTS['TOV'] * df['TOV'] +
        OFF_WEIGHTS['ORB'] * df['OREB'] +
        OFF_WEIGHTS['DRB'] * df['DREB'] +
        OFF_WEIGHTS['STL'] * df['STL'] +
        OFF_WEIGHTS['BLK'] * df['BLK'] +
        OFF_WEIGHTS['PF']  * df['PF']
    )
    
    # Defensive RAPTOR
    df['raptor_def'] = (
        DEF_WEIGHTS['intercept'] +
        DEF_WEIGHTS['MPG'] * df['MPG'] +
        DEF_WEIGHTS['PTS'] * df['PTS'] +
        DEF_WEIGHTS['TSA'] * df['TSA'] +
        DEF_WEIGHTS['AST'] * df['AST'] +
        DEF_WEIGHTS['TOV'] * df['TOV'] +
        DEF_WEIGHTS['ORB'] * df['OREB'] +
        DEF_WEIGHTS['DRB'] * df['DREB'] +
        DEF_WEIGHTS['STL'] * df['STL'] +
        DEF_WEIGHTS['BLK'] * df['BLK'] +
        DEF_WEIGHTS['PF']  * df['PF']
    )
    
    df['raptor_total'] = df['raptor_off'] + df['raptor_def']
    return df

def run_backfill():
    """
    Main loop to generate proxy RAPTOR for 2022-23, 2023-24, and 2024-25.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Map season to its string format
    seasons_to_fetch = {
        2023: "2022-23",
        2024: "2023-24",
        2025: "2024-25"
    }
    
    for season_year, season_str in seasons_to_fetch.items():
        print(f"\n--- Processing Season: {season_str} ---")
        df_stats = fetch_season_stats(season_str)
        
        if not df_stats.empty:
            df_raptor = calculate_proxy_raptor(df_stats)
            
            # Prepare final columns: player_name, season, raptor_total
            df_final = df_raptor[['PLAYER_NAME', 'raptor_total']].copy()
            df_final['season'] = season_year
            df_final = df_final.rename(columns={'PLAYER_NAME': 'player_name'})
            
            # Reorder columns
            df_final = df_final[['player_name', 'season', 'raptor_total']]
            
            output_path = os.path.join(RAW_DATA_DIR, f"proxy_raptor_{season_year}.csv")
            print(f"Saving {len(df_final)} rows to {output_path}...")
            df_final.to_csv(output_path, index=False)
        else:
            print(f"Skipping {season_str} due to fetch failure.")
            
    print("\n✅ Backfill complete!")

if __name__ == "__main__":
    run_backfill()
