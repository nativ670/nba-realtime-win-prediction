import pandas as pd
import numpy as np
import os

import time
from nba_api.stats.endpoints import leaguedashplayerstats


# --- Configuration & Weights (Reused from src.config) ---
from src.config import RAPTOR_OFF_WEIGHTS, RAPTOR_DEF_WEIGHTS, RAW_DIR
from src.utils.nba_client import nba_api_call

RAW_DATA_DIR = str(RAW_DIR)

def fetch_season_stats(season_str):
    """
    Fetches player stats for a specific season (Per 100 Possessions).
    """
    print(f"Fetching stats for {season_str}...")
    try:
        # Get Per 100 Possessions
        p100 = nba_api_call(
            leaguedashplayerstats.LeagueDashPlayerStats,
            df_index=0,
            per_mode_detailed='Per100Possessions',
            season=season_str,
            season_type_all_star='Regular Season'
        )
        
        # Get Per Game (for MPG)
        pg = nba_api_call(
            leaguedashplayerstats.LeagueDashPlayerStats,
            df_index=0,
            per_mode_detailed='PerGame',
            season=season_str,
            season_type_all_star='Regular Season'
        )
        
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
        RAPTOR_OFF_WEIGHTS['intercept'] +
        RAPTOR_OFF_WEIGHTS['MPG'] * df['MPG'] +
        RAPTOR_OFF_WEIGHTS['PTS'] * df['PTS'] +
        RAPTOR_OFF_WEIGHTS['TSA'] * df['TSA'] +
        RAPTOR_OFF_WEIGHTS['AST'] * df['AST'] +
        RAPTOR_OFF_WEIGHTS['TOV'] * df['TOV'] +
        RAPTOR_OFF_WEIGHTS['ORB'] * df['OREB'] +
        RAPTOR_OFF_WEIGHTS['DRB'] * df['DREB'] +
        RAPTOR_OFF_WEIGHTS['STL'] * df['STL'] +
        RAPTOR_OFF_WEIGHTS['BLK'] * df['BLK'] +
        RAPTOR_OFF_WEIGHTS['PF']  * df['PF']
    )
    
    # Defensive RAPTOR
    df['raptor_def'] = (
        RAPTOR_DEF_WEIGHTS['intercept'] +
        RAPTOR_DEF_WEIGHTS['MPG'] * df['MPG'] +
        RAPTOR_DEF_WEIGHTS['PTS'] * df['PTS'] +
        RAPTOR_DEF_WEIGHTS['TSA'] * df['TSA'] +
        RAPTOR_DEF_WEIGHTS['AST'] * df['AST'] +
        RAPTOR_DEF_WEIGHTS['TOV'] * df['TOV'] +
        RAPTOR_DEF_WEIGHTS['ORB'] * df['OREB'] +
        RAPTOR_DEF_WEIGHTS['DRB'] * df['DREB'] +
        RAPTOR_DEF_WEIGHTS['STL'] * df['STL'] +
        RAPTOR_DEF_WEIGHTS['BLK'] * df['BLK'] +
        RAPTOR_DEF_WEIGHTS['PF']  * df['PF']
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
