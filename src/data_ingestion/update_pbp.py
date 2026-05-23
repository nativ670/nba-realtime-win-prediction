import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3
from requests.exceptions import ReadTimeout, ConnectionError

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
    """Fetches games played yesterday from nba_api with retry logic."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Checking for games on {yesterday}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Tell the script to take a breath
            time.sleep(2) 
            
            game_finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=yesterday,
                date_to_nullable=yesterday,
                league_id_nullable='00',
                timeout=30
            )
            games = game_finder.get_data_frames()[0]
            return games, yesterday
            
        except (ReadTimeout, ConnectionError) as e:
            print(f"⚠️ NBA API timed out! Retrying {attempt + 1}/{max_retries} in 10 seconds...")
            time.sleep(10)
            
        except Exception as e:
            print(f"An unexpected error occurred fetching games: {e}")
            break
            
    print(f"❌ Failed to fetch games for {yesterday} after {max_retries} attempts.")
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
        
        df_pbp_raw = pd.DataFrame()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                time.sleep(0.6) # Rate limit
                pbp = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=30)
                df_pbp_raw = pbp.get_data_frames()[0]
                break
            except (ReadTimeout, ConnectionError) as e:
                print(f"⚠️ PBP Timeout for {gid}! Retrying {attempt+1}/{max_retries} in 5s...")
                time.sleep(5)
            except Exception as e:
                print(f"    Unexpected error for {gid}: {e}")
                break

        if df_pbp_raw.empty:
            print(f"    [!] Skipping {gid} due to empty PBP data.")
            continue
            
        try:
            # Standardize and calculate in-game features
            df_pbp = standardize_pbp_v3(df_pbp_raw)
            df_features = calculate_in_game_features(df_pbp)
            
            # Calculate Target (home_win)
            last_play = df_pbp.iloc[-1]
            home_final = float(last_play['SCORE_HOME'])
            away_final = float(last_play['SCORE_AWAY'])
            df_features['home_win'] = int(home_final > away_final)
            
            all_new_pbp.append(df_features)
        except Exception as e:
            print(f"Error processing features for {gid}: {e}")

    if not all_new_pbp:
        print("No new PBP data to append.")
        return

    df_new_pbp = pd.concat(all_new_pbp, ignore_index=True)

    # 4. Add Pre-Game Features
    print("Calculating pre-game context for new games...")
    elo_df = pd.read_csv(LIVE_ELO_PATH)
    elo_df['date'] = pd.to_datetime(elo_df['date'], format='mixed')
    
    new_game_meta = []
    for gid in game_ids:
        # Determine Elo Advantage
        match_elo = elo_df[(elo_df['date'] == date_str)]
        home_team = games[(games['GAME_ID'] == gid) & (~games['MATCHUP'].str.contains('@'))]['TEAM_ABBREVIATION'].iloc[0]
        game_elo_row = match_elo[(match_elo['team1'] == home_team) & (match_elo['is_home'] == 1)]
        
        elo_adv = 0
        if not game_elo_row.empty:
            row = game_elo_row.iloc[0]
            elo_adv = (row['elo1_pre'] - row['elo2_pre']) / 100

        # Estimate Rest and Distance (Heuristic for daily update)
        # For a truly accurate calculation, we'd search the last game of each team in df_season
        rest_adv = 0
        dist_trav = 0
        
        try:
            away_team = games[(games['GAME_ID'] == gid) & (games['MATCHUP'].str.contains('@'))]['TEAM_ABBREVIATION'].iloc[0]
            
            # Simple lookback for home team rest
            if not df_season.empty:
                home_last = df_season[df_season['teamTricode'] == home_team].tail(1)
                # (This is complex to do purely here, usually we re-run pregame.py logic)
                # For now, let's just keep the elo fix which is the most critical.
        except:
            pass

        new_game_meta.append({
            'GAME_ID': gid,
            'elo_advantage': elo_adv,
            'rest_advantage': 0, # Placeholder (ideally re-run pregame.py)
            'distance_traveled': 0, # Placeholder
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
