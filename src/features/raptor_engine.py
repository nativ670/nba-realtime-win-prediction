import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
from nba_api.stats.endpoints import leaguedashplayerstats

# --- Path Injection ---
# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from requests.exceptions import ReadTimeout, ConnectionError

from src.config import RAPTOR_OFF_WEIGHTS, RAPTOR_DEF_WEIGHTS, RAPTOR_PATH
from src.utils.nba_client import nba_api_call

# --- Configuration ---
RAPTOR_OUTPUT_PATH = str(RAPTOR_PATH)

def fetch_player_stats():
    """
    Fetches current season player stats: 
    - Fetches both Regular Season and Playoff stats.
    - Joins them and creates a weighted average based on total minutes.
    """
    # Determine current season
    now = datetime.now()
    if now.month >= 10:
        season = f"{now.year}-{str(now.year + 1)[2:]}"
    else:
        season = f"{now.year - 1}-{str(now.year)[2:]}"
    
    print(f"Fetching player stats for season {season} (Regular + Playoffs) from nba_api...")
    
    try:
        # Helper to fetch a specific type
        def get_phase_stats(phase):
            p100 = nba_api_call(
                leaguedashplayerstats.LeagueDashPlayerStats,
                df_index=0,
                per_mode_detailed='Per100Possessions',
                season=season,
                season_type_all_star=phase
            )
            
            pg = nba_api_call(
                leaguedashplayerstats.LeagueDashPlayerStats,
                df_index=0,
                per_mode_detailed='PerGame',
                season=season,
                season_type_all_star=phase
            )
            
            if p100.empty or pg.empty:
                return pd.DataFrame()
            
            # Combine Per100 and PerGame (for MPG/MIN)
            df_pg_min = pg[['PLAYER_ID', 'MIN', 'GP']].rename(columns={'MIN': 'MPG', 'GP': 'GP_PHASE'})
            # Also need total minutes to weight correctly: Total MIN = MPG * GP
            df_pg_min['TOTAL_MIN_PHASE'] = df_pg_min['MPG'] * df_pg_min['GP_PHASE']
            
            return p100.merge(df_pg_min, on='PLAYER_ID', how='inner')

        df_reg = get_phase_stats('Regular Season')
        df_ply = get_phase_stats('Playoffs')

        if df_reg.empty and df_ply.empty:
            return pd.DataFrame()
        if df_reg.empty: return df_ply
        if df_ply.empty: return df_reg

        # --- Weighted Merge Logic ---
        # We want to combine them so that if a player is in both, we take the weighted average
        common_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID']
        stat_cols = [
            'PTS', 'FGA', 'FTA', 'AST', 'TOV', 'OREB', 'DREB', 'STL', 'BLK', 'PF', 'MPG'
        ]
        
        # Merge the two sets
        df_combined = df_reg.merge(df_ply, on=common_cols, how='outer', suffixes=('_REG', '_PLY'))
        
        # Fill NaNs with 0 for players only in one phase
        df_combined = df_combined.fillna(0)
        
        # Calculate weighted averages
        df_combined['TOTAL_MIN'] = df_combined['TOTAL_MIN_PHASE_REG'] + df_combined['TOTAL_MIN_PHASE_PLY']
        
        # Avoid division by zero
        df_combined = df_combined[df_combined['TOTAL_MIN'] > 0].copy()

        for col in stat_cols:
            # Weighted average: (Stat_Reg * Min_Reg + Stat_Ply * Min_Ply) / Total_Min
            reg_val = df_combined[f'{col}_REG' if col != 'MPG' else 'MPG_REG']
            ply_val = df_combined[f'{col}_PLY' if col != 'MPG' else 'MPG_PLY']
            
            df_combined[col] = (
                (reg_val * df_combined['TOTAL_MIN_PHASE_REG']) + 
                (ply_val * df_combined['TOTAL_MIN_PHASE_PLY'])
            ) / df_combined['TOTAL_MIN']

        return df_combined
    except Exception as e:
        print(f"Error fetching player stats: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def calculate_estimated_raptor(df):
    """
    Applies Neil Paine's linear regression weights to calculate box-score RAPTOR.
    """
    if df.empty:
        return df

    # Calculate True Shooting Attempts (TSA) per 100
    # TSA = FGA + 0.44 * FTA
    df['TSA'] = df['FGA'] + 0.44 * df['FTA']
    
    # Map columns to match weights
    # Note: Stats in df are already per 100 possessions due to 'Per100Possessions' mode
    
    # Offensive RAPTOR
    df['RAPTOR_OFF'] = (
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
    df['RAPTOR_DEF'] = (
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
    
    # Positional Adjustments (Default to SF 0/0 for V1)
    # PG: +0.3/-0.3, SG: +0.2/-0.2, SF: 0/0, PF: -0.2/+0.2, C: -0.5/+0.5
    # Future enhancement: scrape position data
    df['RAPTOR_OFF'] += 0.0
    df['RAPTOR_DEF'] += 0.0
    
    # Total RAPTOR
    df['RAPTOR_TOTAL'] = df['RAPTOR_OFF'] + df['RAPTOR_DEF']
    
    return df

def run_raptor_pipeline():
    """
    Orchestrates the RAPTOR calculation and export.
    """
    os.makedirs(os.path.dirname(RAPTOR_OUTPUT_PATH), exist_ok=True)
    
    df = fetch_player_stats()
    if not df.empty:
        df_raptor = calculate_estimated_raptor(df)
        
        # Select final columns
        final_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'MPG', 
            'RAPTOR_OFF', 'RAPTOR_DEF', 'RAPTOR_TOTAL'
        ]
        # Rename MPG back to MIN for final CSV as per requirements
        df_export = df_raptor[final_cols].rename(columns={'MPG': 'MIN'})
        
        print(f"Saving RAPTOR data to {RAPTOR_OUTPUT_PATH}...")
        df_export.to_csv(RAPTOR_OUTPUT_PATH, index=False)
        return df_export
    else:
        print("Pipeline failed: No player stats retrieved.")
        return pd.DataFrame()

if __name__ == "__main__":
    print("--- NBA RAPTOR ENGINE V1 ---")
    df_results = run_raptor_pipeline()
    
    if not df_results.empty:
        print("\n🏆 RAPTOR LEADERBOARD (Top 10)")
        # Sort by Total RAPTOR, minimum minutes filter for quality
        leaderboard = df_results[df_results['MIN'] > 15].sort_values('RAPTOR_TOTAL', ascending=False).head(10)
        print(leaderboard[['PLAYER_NAME', 'MIN', 'RAPTOR_OFF', 'RAPTOR_DEF', 'RAPTOR_TOTAL']].to_string(index=False))
