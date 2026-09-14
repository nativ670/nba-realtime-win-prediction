import pandas as pd
import numpy as np
import os
import sys
import time
from nba_api.stats.endpoints import leaguegamefinder


from src.features.pregame import calculate_pregame_features

from src.config import SEASONS_DIR, LIVE_ELO_PATH
from src.utils.nba_client import nba_api_call
PBP_2025_PATH = str(SEASONS_DIR / "pbp_2025.parquet")
LIVE_ELO_PATH = str(LIVE_ELO_PATH)

def fix_2025_features():
    if not os.path.exists(PBP_2025_PATH):
        print(f"Error: {PBP_2025_PATH} not found.")
        return

    print("🚀 Fixing features for 2025-26 season...")
    
    # 1. Fetch 2025-26 Box Scores
    print("  Fetching 2025-26 box scores from NBA API...")
    try:
        df_box_raw = nba_api_call(
            leaguegamefinder.LeagueGameFinder,
            df_index=0,
            season_nullable="2025-26",
            league_id_nullable='00'
        )
        print(f"  Found {len(df_box_raw)} team-game rows for 2025.")
    except Exception as e:
        print(f"  ❌ Error fetching box scores: {e}")
        return

    if df_box_raw.empty:
        print("  ❌ No box scores found for 2025.")
        return

    # 2. Prepare for Pregame Engine
    df_box = df_box_raw.copy()
    df_box.columns = [col.lower() for col in df_box.columns]
    df_box['season'] = 2025
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
    
    # 3. Calculate Pregame Features (Rest, Distance)
    print("  Calculating rest and distance...")
    df_pre_features = calculate_pregame_features(df_box_pre)
    
    # Manually calculate rest_advantage (as add_external_features does)
    opp_rest_df = df_pre_features[['game_id', 'team_id', 'days_rest_capped']].rename(
        columns={'team_id': 'opponent_team_id', 'days_rest_capped': 'opp_rest'}
    )
    df_pre_features = df_pre_features.merge(opp_rest_df, on=['game_id', 'opponent_team_id'], how='left')
    df_pre_features['rest_advantage'] = df_pre_features['days_rest_capped'] - df_pre_features['opp_rest']
    
    # 4. Add Elo from local live_elo.csv
    print("  Adding Elo ratings from live_elo.csv...")
    if os.path.exists(LIVE_ELO_PATH):
        elo_df = pd.read_parquet(LIVE_ELO_PATH)
        df_pre_features['game_date'] = pd.to_datetime(df_pre_features['game_date'])
        elo_df['date'] = pd.to_datetime(elo_df['date'], format='mixed')
        
        elo_clean = elo_df[['date', 'team1', 'team2', 'elo1_pre', 'elo2_pre', 'is_home']].copy()
        elo_home = elo_clean[elo_clean['is_home'] == 1].copy()
        
        df_home = df_pre_features[df_pre_features['team_home_away'] == 'home'].copy()
        df_home = df_home.merge(
            elo_home,
            left_on=['game_date', 'team_abbreviation'],
            right_on=['date', 'team1'],
            how='left'
        )
        df_home['elo_advantage'] = (df_home['elo1_pre'] - df_home['elo2_pre']) / 100
        df_home['is_playoffs'] = df_home['game_id'].str.startswith('004').astype(int)
        
        df_meta = df_home[['game_id', 'elo_advantage', 'rest_advantage', 'distance_traveled', 'is_playoffs']]
        df_meta = df_meta.rename(columns={'game_id': 'GAME_ID'})
    else:
        print(f"  ❌ {LIVE_ELO_PATH} not found. Cannot add Elo.")
        return

    # 5. Merge into PBP
    print(f"  Merging features into {PBP_2025_PATH}...")
    df_pbp = pd.read_parquet(PBP_2025_PATH)
    
    df_pbp['GAME_ID'] = df_pbp['GAME_ID'].astype(str)
    df_meta['GAME_ID'] = df_meta['GAME_ID'].astype(str)
    
    cols_to_drop = ['elo_advantage', 'rest_advantage', 'distance_traveled', 'is_playoffs']
    df_pbp = df_pbp.drop(columns=[c for c in cols_to_drop if c in df_pbp.columns])
    
    df_fixed = df_pbp.merge(df_meta, on='GAME_ID', how='left')
    
    df_fixed[['elo_advantage', 'rest_advantage', 'distance_traveled', 'is_playoffs']] = \
        df_fixed[['elo_advantage', 'rest_advantage', 'distance_traveled', 'is_playoffs']].fillna(0)
    
    # 6. Save
    df_fixed.to_parquet(PBP_2025_PATH, index=False)
    print(f"  ✅ Successfully updated 2025 features.")
    print(f"     Non-zero Elos: {(df_fixed['elo_advantage'] != 0).sum()}")
    print(f"     Mean Elo Adv: {df_fixed['elo_advantage'].mean():.4f}")

if __name__ == "__main__":
    fix_2025_features()
