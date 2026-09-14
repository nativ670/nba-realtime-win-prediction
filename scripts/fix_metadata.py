import pandas as pd
import numpy as np
import os
import sys
import glob
from nba_api.stats.endpoints import leaguegamefinder
from tqdm import tqdm


from src.features.pregame import calculate_pregame_features, add_external_features
from src.utils.helpers import ELO_TEAM_MAP

from src.config import LIVE_ELO_PATH, SEASONS_DIR
from src.utils.nba_client import nba_api_call
LIVE_ELO_PATH = str(LIVE_ELO_PATH)
SEASONS_DIR = str(SEASONS_DIR)

def fix_file_metadata(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"🚀 Fixing metadata for {os.path.basename(file_path)}...")
    
    # 1. Detect Season from GAME_IDs
    df_pbp = pd.read_parquet(file_path)
    reg_games = df_pbp[df_pbp["GAME_ID"].str.startswith("002")]["GAME_ID"].unique()
    if len(reg_games) == 0:
        print(f"  ❌ No regular season games found in {file_path}. Skipping.")
        return
        
    sample_id = reg_games[0]
    yy = int(sample_id[3:5]) # '24' for 2024-25, '25' for 2025-26
    api_year = 2000 + yy + 1 # 2025 for 2024-25, 2026 for 2025-26
    season_str = f"{api_year-1}-{str(api_year)[2:]}"
    
    print(f"  Detected Season: {season_str} (API Year: {api_year})")
    
    # 2. Fetch Box Scores for the season
    print(f"  Fetching {season_str} box scores from NBA API...")
    try:
        df_box_raw = nba_api_call(
            leaguegamefinder.LeagueGameFinder,
            df_index=0,
            season_nullable=season_str,
            league_id_nullable='00'
        )
        print(f"  Found {len(df_box_raw)} team-game rows.")
    except Exception as e:
        print(f"  ❌ Error fetching box scores: {e}")
        return

    if df_box_raw.empty:
        print(f"  ❌ No box scores found for {season_str}.")
        return

    # 3. Prepare for Pregame Engine
    df_box = df_box_raw.copy()
    df_box.columns = [col.lower() for col in df_box.columns]
    df_box['season'] = api_year
    df_box['game_date'] = pd.to_datetime(df_box['game_date'])
    
    df_box = df_box.rename(columns={
        'fga': 'field_goals_attempted',
        'fta': 'free_throws_attempted',
        'oreb': 'offensive_rebounds',
        'tov': 'turnovers',
        'pts': 'team_score'
    })
    
    print("  Pairing games for opponent context...")
    df_box_paired = []
    df_box = df_box.sort_values(['game_id', 'team_id'])
    
    for gid, group in df_box.groupby('game_id'):
        if len(group) == 2:
            row1 = group.iloc[0].to_dict()
            row2 = group.iloc[1].to_dict()
            
            r1 = row1.copy()
            r1['opponent_team_id'] = row2['team_id']
            r1['opponent_team_score'] = row2['team_score']
            r1['team_home_away'] = 'home' if 'vs.' in r1['matchup'] else 'away'
            df_box_paired.append(r1)
            
            r2 = row2.copy()
            r2['opponent_team_id'] = row1['team_id']
            r2['opponent_team_score'] = row1['team_score']
            r2['team_home_away'] = 'home' if 'vs.' in r2['matchup'] else 'away'
            df_box_paired.append(r2)
            
    if not df_box_paired:
        print("  ❌ No paired games found.")
        return
        
    df_box_pre = pd.DataFrame(df_box_paired)
    
    # 4. Calculate Pregame Features (Rest, Distance)
    print("  Calculating rest and distance...")
    df_pre_features = calculate_pregame_features(df_box_pre)
    
    # 5. Add Elo from local live_elo.csv
    print("  Adding Elo ratings from live_elo.csv...")
    if os.path.exists(LIVE_ELO_PATH):
        elo_df = pd.read_parquet(LIVE_ELO_PATH)
        df_final_pre = add_external_features(df_pre_features, elo_df)
        
        print(f"  External features join result: {len(df_final_pre)} rows (from {len(df_pre_features)})")
    else:
        print(f"  ❌ {LIVE_ELO_PATH} not found. Cannot add Elo.")
        return

    # 6. Merge into PBP
    print(f"  Merging features into {os.path.basename(file_path)}...")
    df_pbp['GAME_ID'] = df_pbp['GAME_ID'].astype(str).str.strip()
    
    cols_to_drop = [
        'elo_advantage', 'rest_advantage', 'distance_traveled', 'is_playoffs',
        'my_elo_pre', 'opp_elo_pre', 'days_rest_capped'
    ]
    df_pbp = df_pbp.drop(columns=[c for c in cols_to_drop if c in df_pbp.columns])
    
    df_meta_home = df_final_pre[df_final_pre['team_home_away'] == 'home'][[
        'game_id', 'elo_advantage', 'rest_advantage', 'distance_traveled', 
        'is_playoffs', 'my_elo_pre', 'opp_elo_pre', 'days_rest_capped'
    ]].copy()
    df_meta_home['GAME_ID'] = df_meta_home['game_id'].astype(str).str.strip()
    
    df_fixed = df_pbp.merge(df_meta_home.drop(columns=['game_id']), on='GAME_ID', how='left')
    
    fill_cols = ['elo_advantage', 'rest_advantage', 'distance_traveled', 'is_playoffs']
    df_fixed[fill_cols] = df_fixed[fill_cols].fillna(0)
    
    # 7. Save
    df_fixed.to_parquet(file_path, index=False)
    print(f"  ✅ Successfully updated {os.path.basename(file_path)}.")
    print(f"     Mean Elo Adv: {df_fixed['elo_advantage'].mean():.4f}")
    print(f"     Non-zero Elo Rows: {(df_fixed['elo_advantage'] != 0).sum()} / {len(df_fixed)}")

if __name__ == "__main__":
    fix_file_metadata(os.path.join(SEASONS_DIR, "pbp_2024.parquet"))
    fix_file_metadata(os.path.join(SEASONS_DIR, "pbp_2025.parquet"))
