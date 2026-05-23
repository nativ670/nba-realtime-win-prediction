import pandas as pd
import numpy as np
import os
import time
import sys
from datetime import datetime, timedelta

# --- Path Injection ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nba_api.stats.endpoints import leaguegamefinder
from src.data_ingestion.fetch_elo import fetch_elo_data
from requests.exceptions import ReadTimeout, ConnectionError

# --- Configuration ---
LIVE_ELO_PATH = "data/processed/live_elo.csv"
K_FACTOR = 20
HOME_ADVANTAGE = 100
OFFSEASON_REGRESSION_FACTOR = 0.75
OFFSEASON_MEAN = 1505

def calculate_expected_score(elo1, elo2, hca=0):
    """Calculates the expected score for team1."""
    elo_diff = (elo1 + hca) - elo2
    return 1 / (10**(-elo_diff / 400) + 1)

def calculate_new_elos(elo1_pre, elo2_pre, score1, score2, is_home=True):
    """
    Calculates post-game Elo ratings using the 538 formula.
    As per user instructions:
    Margin of Victory Multiplier = ((MOV + 3)^0.8) / (7.5 + 0.006 * elo_diff)
    elo_diff is the difference including HCA, and should be negative if the underdog wins.
    """
    hca = HOME_ADVANTAGE if is_home else -HOME_ADVANTAGE
    
    # elo_diff for the formula is (Favorite Elo - Underdog Elo) including HCA?
    # User says: "elo_diff is the Elo difference including home-court advantage, 
    # and should be negative if the underdog wins."
    
    elo1_eff = elo1_pre + (HOME_ADVANTAGE if is_home else 0)
    elo2_eff = elo2_pre + (HOME_ADVANTAGE if not is_home else 0)
    
    # Probability for team 1
    exp1 = 1 / (10**(-(elo1_eff - elo2_eff) / 400) + 1)
    
    # Result for team 1
    s1 = 1 if score1 > score2 else 0
    
    # Margin of Victory
    mov = abs(score1 - score2)
    
    # Elo difference from perspective of the winner
    # "elo_diff should be negative if the underdog wins"
    # This implies elo_diff = (Winner Elo - Loser Elo)
    winner_elo_eff = elo1_eff if s1 == 1 else elo2_eff
    loser_elo_eff = elo2_eff if s1 == 1 else elo1_eff
    elo_diff_winner = winner_elo_eff - loser_elo_eff
    
    # Multiplier
    multiplier = ((mov + 3)**0.8) / (7.5 + 0.006 * elo_diff_winner)
    
    # Shift
    shift = K_FACTOR * (s1 - exp1) * multiplier
    
    return elo1_pre + shift, elo2_pre - shift

def get_games_for_date(target_date):
    """Fetches games played on a specific date from nba_api with retry logic."""
    print(f"Fetching games for {target_date}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Tell the script to take a breath
            time.sleep(2) 
            
            game_finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=target_date,
                date_to_nullable=target_date,
                league_id_nullable='00',
                timeout=30
            )
            games = game_finder.get_data_frames()[0]
            return games
            
        except (ReadTimeout, ConnectionError) as e:
            print(f"⚠️ NBA API timed out! Retrying {attempt + 1}/{max_retries} in 10 seconds...")
            time.sleep(10)
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
            
    print(f"❌ Failed to fetch games for {target_date} after {max_retries} attempts.")
    return pd.DataFrame()

def update_elo():
    # 1. Load existing Elo
    if os.path.exists(LIVE_ELO_PATH):
        df_live = pd.read_csv(LIVE_ELO_PATH)
        df_live['date'] = pd.to_datetime(df_live['date'], errors='coerce')
    else:
        print("Initializing live_elo.csv from historical data...")
        df_live = fetch_elo_data()
        if df_live.empty:
            print("Failed to fetch historical data. Exiting.")
            return
        # Ensure date is datetime
        df_live['date'] = pd.to_datetime(df_live['date'], errors='coerce')
        # Ensure directories exist
        os.makedirs(os.path.dirname(LIVE_ELO_PATH), exist_ok=True)
        df_live.to_csv(LIVE_ELO_PATH, index=False)

    # 2. Determine Catch-up Range
    last_date = df_live['date'].max()
    yesterday = datetime.now() - timedelta(days=1)
    
    # Floor both to midnight for comparison
    last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

    if last_date >= yesterday:
        print(f"Data is up to date (Last entry: {last_date.strftime('%Y-%m-%d')}).")
        return

    # Iterate through each missing day
    current_date_dt = last_date + timedelta(days=1)
    while current_date_dt <= yesterday:
        target_date = current_date_dt.strftime('%Y-%m-%d')
        
        # 3. Get the latest Elo for each team (Recalculate for each day in loop)
        latest_elos = {}
        teams = pd.concat([df_live['team1'], df_live['team2']]).unique()
        
        # Optimize: only look at the most recent 2000 rows to find latest elos
        recent_df = df_live.tail(2000) 
        for team in teams:
            t1_rows = recent_df[recent_df['team1'] == team]
            t2_rows = recent_df[recent_df['team2'] == team]
            
            last_t1 = t1_rows.iloc[-1] if not t1_rows.empty else None
            last_t2 = t2_rows.iloc[-1] if not t2_rows.empty else None
            
            if last_t1 is not None and last_t2 is not None:
                if last_t1['date'] >= last_t2['date']:
                    latest_elos[team] = last_t1['elo1_post']
                else:
                    latest_elos[team] = last_t2['elo2_post']
            elif last_t1 is not None:
                latest_elos[team] = last_t1['elo1_post']
            elif last_t2 is not None:
                latest_elos[team] = last_t2['elo2_post']
            else:
                # Fallback to full search
                t1_rows_full = df_live[df_live['team1'] == team]
                t2_rows_full = df_live[df_live['team2'] == team]
                last_t1 = t1_rows_full.iloc[-1] if not t1_rows_full.empty else None
                last_t2 = t2_rows_full.iloc[-1] if not t2_rows_full.empty else None
                if last_t1 is not None and last_t2 is not None:
                    latest_elos[team] = last_t1['elo1_post'] if last_t1['date'] >= last_t2['date'] else last_t2['elo2_post']
                elif last_t1 is not None: latest_elos[team] = last_t1['elo1_post']
                elif last_t2 is not None: latest_elos[team] = last_t2['elo2_post']
                else: latest_elos[team] = 1500

        # 4. Fetch games for target date
        games = get_games_for_date(target_date)
        if games.empty:
            print(f"No games found for {target_date}. Adding empty entry to prevent re-scan.")
            # We don't actually add empty entries to the CSV, but the loop advances
            current_date_dt += timedelta(days=1)
            continue

        # 5. Process games...
        game_ids = games['GAME_ID'].unique()
        new_rows = []
        last_season = df_live['season'].max()

        # Determine season for target date
        game_dt = datetime.strptime(target_date, '%Y-%m-%d')
        current_season = game_dt.year + 1 if game_dt.month >= 10 else game_dt.year

        # Offseason regression check
        if current_season > last_season:
            print(f"New season detected ({current_season}). Applying regression...")
            for t in latest_elos:
                latest_elos[t] = (OFFSEASON_REGRESSION_FACTOR * latest_elos[t]) + ((1 - OFFSEASON_REGRESSION_FACTOR) * OFFSEASON_MEAN)
            last_season = current_season

        for gid in game_ids:
            game_pair = games[games['GAME_ID'] == gid]
            if len(game_pair) != 2: continue
                
            row_a, row_b = game_pair.iloc[0], game_pair.iloc[1]
            home_row, away_row = (row_b, row_a) if '@' in row_a['MATCHUP'] else (row_a, row_b)
                
            team_h, team_a = home_row['TEAM_ABBREVIATION'], away_row['TEAM_ABBREVIATION']
            score_h, score_a = home_row['PTS'], away_row['PTS']

            elo_h_pre, elo_a_pre = latest_elos.get(team_h, 1500), latest_elos.get(team_a, 1500)
            prob_h = calculate_expected_score(elo_h_pre, elo_a_pre, hca=HOME_ADVANTAGE)
            elo_h_post, elo_a_post = calculate_new_elos(elo_h_pre, elo_a_pre, score_h, score_a, is_home=True)
            
            new_rows.append({
                'date': target_date, 'season': current_season, 'neutral': 0, 'playoff': np.nan,
                'team1': team_h, 'team2': team_a, 'elo1_pre': elo_h_pre, 'elo2_pre': elo_a_pre,
                'elo_prob1': prob_h, 'elo_prob2': 1-prob_h, 'elo1_post': elo_h_post, 'elo2_post': elo_a_post,
                'score1': score_h, 'score2': score_a, 'is_home': 1
            })
            new_rows.append({
                'date': target_date, 'season': current_season, 'neutral': 0, 'playoff': np.nan,
                'team1': team_a, 'team2': team_h, 'elo1_pre': elo_a_pre, 'elo2_pre': elo_h_pre,
                'elo_prob1': 1-prob_h, 'elo_prob2': prob_h, 'elo1_post': elo_a_post, 'elo2_post': elo_h_post,
                'score1': score_a, 'score2': score_h, 'is_home': 0
            })
            latest_elos[team_h], latest_elos[team_a] = elo_h_post, elo_a_post

        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_live = pd.concat([df_live, df_new], ignore_index=True)
            print(f"Successfully processed {target_date}.")
        
        current_date_dt += timedelta(days=1)

    # 6. Save final updated file
    df_live.to_csv(LIVE_ELO_PATH, index=False)
    print(f"Elo update complete. Final date: {df_live['date'].max().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    update_elo()
