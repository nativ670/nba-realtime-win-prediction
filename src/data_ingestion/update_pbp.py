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

def get_games_for_range(start_date, end_date):
    """Fetches games played in a date range from nba_api with retry logic."""
    print(f"Checking for games between {start_date} and {end_date}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Tell the script to take a breath
            time.sleep(2) 
            
            game_finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=start_date,
                date_to_nullable=end_date,
                league_id_nullable='00',
                timeout=30
            )
            games = game_finder.get_data_frames()[0]
            return games
            
        except (ReadTimeout, ConnectionError) as e:
            print(f"⚠️ NBA API timed out! Retrying {attempt + 1}/{max_retries} in 10 seconds...")
            time.sleep(10)
            
        except Exception as e:
            print(f"An unexpected error occurred fetching games: {e}")
            break
            
    print(f"❌ Failed to fetch games for range after {max_retries} attempts.")
    return pd.DataFrame()

def update_pbp():
    # 1. Get games from the last 7 days (Catch-up window)
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    games = get_games_for_range(start_date, end_date)
    if games.empty:
        print(f"No games found in range {start_date} to {end_date}. Nothing to update.")
        return

    # 2. Determine Season and File Path
    # Group games by season suffix to handle season boundaries if they happen in the 7-day window
    games['SEASON_YEAR'] = games['GAME_ID'].apply(lambda x: 2000 + int(x[3:5]) if int(x[3:5]) < 50 else 1900 + int(x[3:5]))
    
    for season_year, season_games in games.groupby('SEASON_YEAR'):
        season_file = os.path.join(SEASONS_DIR, f"pbp_{season_year}.parquet")
        
        if not os.path.exists(season_file):
            print(f"Season file {season_file} not found. Initializing new season...")
            df_season = pd.DataFrame()
        else:
            df_season = pd.read_parquet(season_file)

        # 3. Fetch and Process PBP for each game
        game_ids = season_games['GAME_ID'].unique()
        all_new_pbp = []
        processed_count = 0

        for gid in game_ids:
            # Check if already processed
            if not df_season.empty and gid in df_season['GAME_ID'].values:
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
                processed_count += 1
            except Exception as e:
                print(f"Error processing features for {gid}: {e}")

        if not all_new_pbp:
            print(f"No new PBP data to append for season {season_year}.")
            continue

        df_new_pbp = pd.concat(all_new_pbp, ignore_index=True)

        # 4. Add Pre-Game Features
        print(f"Calculating pre-game context for {processed_count} new games...")
        elo_df = pd.read_csv(LIVE_ELO_PATH)
        elo_df['date'] = pd.to_datetime(elo_df['date'], format='mixed')
        
        new_game_meta = []
        for gid in game_ids:
            if not df_season.empty and gid in df_season['GAME_ID'].values:
                continue
                
            # Find game date from season_games
            game_date_str = season_games[season_games['GAME_ID'] == gid]['GAME_DATE'].iloc[0]
            
            # Determine Elo Advantage
            match_elo = elo_df[(elo_df['date'] == game_date_str)]
            
            try:
                home_team = season_games[(season_games['GAME_ID'] == gid) & (~season_games['MATCHUP'].str.contains('@'))]['TEAM_ABBREVIATION'].iloc[0]
                game_elo_row = match_elo[(match_elo['team1'] == home_team) & (match_elo['is_home'] == 1)]
                
                elo_adv = 0
                if not game_elo_row.empty:
                    row = game_elo_row.iloc[0]
                    elo_adv = (row['elo1_pre'] - row['elo2_pre']) / 100
            except:
                elo_adv = 0

            new_game_meta.append({
                'GAME_ID': gid,
                'elo_advantage': elo_adv,
                'rest_advantage': 0, # Placeholder
                'distance_traveled': 0, # Placeholder
                'is_playoffs': 1 if gid.startswith('004') else 0
            })

        df_meta = pd.DataFrame(new_game_meta)
        df_new_final = df_new_pbp.merge(df_meta, on='GAME_ID', how='left')

        # 5. Append and Save
        df_updated = pd.concat([df_season, df_new_final], ignore_index=True)
        df_updated.to_parquet(season_file, index=False)
        print(f"Successfully updated {season_file} with {processed_count} new games.")


if __name__ == "__main__":
    update_pbp()
