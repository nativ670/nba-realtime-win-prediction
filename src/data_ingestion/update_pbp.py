import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

# --- Path Injection ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features.in_game import calculate_in_game_features
from src.features.pregame import calculate_pregame_features, add_external_features
from src.utils.helpers import standardize_pbp_v3
from src.data_ingestion.fetch_elo import fetch_elo_data

# --- Configuration ---
SEASONS_DIR = "data/processed/seasons"
LIVE_ELO_PATH = "data/processed/live_elo.csv"

def get_yesterday_games():
    """Fetches games played yesterday from nba_api."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Checking for games on {yesterday}...")
    try:
        game_finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=yesterday,
            date_to_nullable=yesterday,
            league_id_nullable='00'
        )
        games = game_finder.get_data_frames()[0]
        return games, yesterday
    except Exception as e:
        print(f"Error fetching games: {e}")
        return pd.DataFrame(), yesterday

def update_pbp():
    # 1. Get yesterday's games
    games, date_str = get_yesterday_games()
    if games.empty:
        print(f"No games found for {date_str}. Nothing to update.")
        return

    # 2. Determine Season and File Path
    # Use first game to determine season
    sample_gid = games.iloc[0]['GAME_ID']
    season_suffix = int(sample_gid[3:5])
    season_year = 2000 + season_suffix if season_suffix < 50 else 1900 + season_suffix
    season_file = os.path.join(SEASONS_DIR, f"pbp_{season_year}.parquet")

    if not os.path.exists(season_file):
        print(f"Season file {season_file} not found. Initializing new season...")
        df_season = pd.DataFrame()
    else:
        df_season = pd.read_parquet(season_file)

    # 3. Fetch and Process PBP for each game
    game_ids = games['GAME_ID'].unique()
    all_new_pbp = []

    for gid in game_ids:
        # Check if already processed
        if not df_season.empty and gid in df_season['GAME_ID'].values:
            print(f"  - Game {gid} already in dataset. Skipping.")
            continue

        print(f"  - Processing PBP for Game {gid}...")
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=gid)
            df_pbp_raw = pbp.get_data_frames()[0]
            if df_pbp_raw.empty: continue
            
            # Standardize and calculate in-game features
            df_pbp = standardize_pbp_v3(df_pbp_raw)
            df_features = calculate_in_game_features(df_pbp)
            
            # Calculate Target (home_win)
            last_play = df_pbp.iloc[-1]
            home_final = float(last_play['SCORE_HOME'])
            away_final = float(last_play['SCORE_AWAY'])
            df_features['home_win'] = int(home_final > away_final)
            
            all_new_pbp.append(df_features)
            time.sleep(0.6) # Rate limit
        except Exception as e:
            print(f"Error processing {gid}: {e}")

    if not all_new_pbp:
        print("No new PBP data to append.")
        return

    df_new_pbp = pd.concat(all_new_pbp, ignore_index=True)

    # 4. Add Pre-Game Features
    print("Calculating pre-game context for new games...")
    # We need to calculate pre-game features using our shared engine.
    # To do this correctly (especially for rest/dist), we'd need the previous games.
    # For now, let's look up the Elo from our live_elo.csv and mock the rest/dist 
    # if we don't want to re-run the whole pregame engine.
    # Actually, let's use the live_elo.csv which we just updated.
    
    elo_df = pd.read_csv(LIVE_ELO_PATH)
    elo_df['date'] = pd.to_datetime(elo_df['date'])
    
    # We'll merge Elo and set defaults for rest/dist for the daily update
    # (In a full rebuild, these would be precise).
    
    # Neil Paine dataset uses Home=team1, Away=team2
    # We need elo_advantage = (Home_Elo - Away_Elo) / 100
    
    new_game_meta = []
    for gid in game_ids:
        game_elo = elo_df[elo_df['team1'].isin(games[games['GAME_ID']==gid]['TEAM_ABBREVIATION'])] # Roughly
        # Better: get the specific row from elo_df for this gid/date
        # update_elo.py saves rows with the same date_str
        match_elo = elo_df[(elo_df['date'] == date_str) & (elo_df['is_home'] == 1)]
        # This might match multiple games, let's filter by team
        home_team = games[(games['GAME_ID'] == gid) & (~games['MATCHUP'].str.contains('@'))]['TEAM_ABBREVIATION'].iloc[0]
        game_elo_row = match_elo[match_elo['team1'] == home_team]
        
        if not game_elo_row.empty:
            row = game_elo_row.iloc[0]
            new_game_meta.append({
                'GAME_ID': gid,
                'elo_advantage': (row['elo1_pre'] - row['elo2_pre']) / 100,
                'rest_advantage': 0, # Default for now
                'distance_traveled': 0, # Default for now
                'is_playoffs': 1 if gid.startswith('004') else 0
            })

    df_meta = pd.DataFrame(new_game_meta)
    df_new_final = df_new_pbp.merge(df_meta, on='GAME_ID', how='left')

    # 5. Append and Save
    df_updated = pd.concat([df_season, df_new_final], ignore_index=True)
    df_updated.to_parquet(season_file, index=False)
    print(f"Successfully updated {season_file} with {len(game_ids)} new games.")

if __name__ == "__main__":
    update_pbp()
