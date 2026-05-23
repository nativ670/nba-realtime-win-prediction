import pandas as pd
import numpy as np
import os
import time
import sys
from datetime import datetime, timedelta

# --- Path Injection ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nba_api.stats.endpoints import leaguegamefinder
from src.config import (
    LIVE_ELO_PATH, K_FACTOR, HOME_ADVANTAGE, 
    OFFSEASON_REGRESSION_FACTOR, OFFSEASON_MEAN
)
from src.utils.nba_client import nba_api_call
from src.data_ingestion.fetch_elo import fetch_elo_data

def calculate_expected_score(elo1, elo2, hca=0):
    elo_diff = (elo1 + hca) - elo2
    return 1 / (10**(-elo_diff / 400) + 1)

def calculate_new_elos(elo1_pre, elo2_pre, score1, score2, is_home=True):
    hca = HOME_ADVANTAGE if is_home else -HOME_ADVANTAGE
    elo1_eff = elo1_pre + (HOME_ADVANTAGE if is_home else 0)
    elo2_eff = elo2_pre + (HOME_ADVANTAGE if not is_home else 0)
    exp1 = 1 / (10**(-(elo1_eff - elo2_eff) / 400) + 1)
    s1 = 1 if score1 > score2 else 0
    mov = abs(score1 - score2)
    winner_elo_eff = elo1_eff if s1 == 1 else elo2_eff
    loser_elo_eff = elo2_eff if s1 == 1 else elo1_eff
    elo_diff_winner = winner_elo_eff - loser_elo_eff
    multiplier = ((mov + 3)**0.8) / (7.5 + 0.006 * elo_diff_winner)
    shift = K_FACTOR * (s1 - exp1) * multiplier
    return elo1_pre + shift, elo2_pre - shift

def get_games_for_date_range(start_date, end_date):
    """Fetches games played in a date range from nba_api with retry logic."""
    print(f"Fetching games from {start_date} to {end_date}...")
    games = nba_api_call(
        leaguegamefinder.LeagueGameFinder,
        date_from_nullable=start_date,
        date_to_nullable=end_date,
        league_id_nullable='00'
    )
    return games

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
        df_live['date'] = pd.to_datetime(df_live['date'], errors='coerce')
        os.makedirs(os.path.dirname(LIVE_ELO_PATH), exist_ok=True)
        df_live.to_csv(LIVE_ELO_PATH, index=False)

    # 2. Determine Catch-up Range
    last_date = df_live['date'].max()
    yesterday = datetime.now() - timedelta(days=1)
    
    last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)

    if last_date >= yesterday:
        print(f"Data is up to date (Last entry: {last_date.strftime('%Y-%m-%d')}).")
        return

    start_date_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date_str = yesterday.strftime('%Y-%m-%d')

    # Fetch all games in the missing date range at once (Phase 1 Fix)
    all_games = get_games_for_date_range(start_date_str, end_date_str)
    
    if all_games.empty:
        print(f"No games found from {start_date_str} to {end_date_str}.")
        return

    # Create a dictionary for O(1) Elo lookups
    latest_elos = {}
    teams = pd.concat([df_live['team1'], df_live['team2']]).dropna().unique()
    recent_df = df_live.dropna(subset=['team1', 'team2']).tail(3000)
    
    for team in teams:
        team_games = recent_df[(recent_df['team1'] == team) | (recent_df['team2'] == team)]
        if not team_games.empty:
            last_game = team_games.iloc[-1]
            if last_game['team1'] == team:
                latest_elos[team] = last_game['elo1_post']
            else:
                latest_elos[team] = last_game['elo2_post']
        else:
            latest_elos[team] = 1500

    last_season = df_live['season'].max()
    new_rows = []
    days_processed = 0

    # Group games by date and sort chronologically
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    all_games = all_games.sort_values('GAME_DATE')

    for game_date, daily_games in all_games.groupby('GAME_DATE'):
        target_date_str = game_date.strftime('%Y-%m-%d')
        current_season = game_date.year + 1 if game_date.month >= 10 else game_date.year
        
        if current_season > last_season:
            print(f"New season detected ({current_season}). Applying regression...")
            for t in latest_elos:
                latest_elos[t] = (OFFSEASON_REGRESSION_FACTOR * latest_elos[t]) + ((1 - OFFSEASON_REGRESSION_FACTOR) * OFFSEASON_MEAN)
            last_season = current_season

        game_ids = daily_games['GAME_ID'].unique()
        for gid in game_ids:
            game_pair = daily_games[daily_games['GAME_ID'] == gid]
            if len(game_pair) != 2: continue
                
            row_a, row_b = game_pair.iloc[0], game_pair.iloc[1]
            home_row, away_row = (row_b, row_a) if '@' in row_a['MATCHUP'] else (row_a, row_b)
                
            team_h, team_a = home_row['TEAM_ABBREVIATION'], away_row['TEAM_ABBREVIATION']
            score_h, score_a = home_row['PTS'], away_row['PTS']

            if pd.isna(score_h) or pd.isna(score_a):
                continue

            elo_h_pre = latest_elos.get(team_h, 1500)
            elo_a_pre = latest_elos.get(team_a, 1500)
            prob_h = calculate_expected_score(elo_h_pre, elo_a_pre, hca=HOME_ADVANTAGE)
            elo_h_post, elo_a_post = calculate_new_elos(elo_h_pre, elo_a_pre, score_h, score_a, is_home=True)
            
            new_rows.append({
                'date': game_date, 'season': current_season, 'neutral': 0, 'playoff': np.nan,
                'team1': team_h, 'team2': team_a, 'elo1_pre': elo_h_pre, 'elo2_pre': elo_a_pre,
                'elo_prob1': prob_h, 'elo_prob2': 1-prob_h, 'elo1_post': elo_h_post, 'elo2_post': elo_a_post,
                'score1': score_h, 'score2': score_a, 'is_home': 1
            })
            new_rows.append({
                'date': game_date, 'season': current_season, 'neutral': 0, 'playoff': np.nan,
                'team1': team_a, 'team2': team_h, 'elo1_pre': elo_a_pre, 'elo2_pre': elo_h_pre,
                'elo_prob1': 1-prob_h, 'elo_prob2': prob_h, 'elo1_post': elo_a_post, 'elo2_post': elo_h_post,
                'score1': score_a, 'score2': score_h, 'is_home': 0
            })
            latest_elos[team_h], latest_elos[team_a] = elo_h_post, elo_a_post
            
        days_processed += 1
        
        # Periodic saves
        if days_processed % 30 == 0 and new_rows:
            df_live = pd.concat([df_live, pd.DataFrame(new_rows)], ignore_index=True)
            df_live.to_csv(LIVE_ELO_PATH, index=False)
            new_rows = []
            print(f"Checkpoint saved at {target_date_str}.")

    # Final save
    if new_rows:
        df_live = pd.concat([df_live, pd.DataFrame(new_rows)], ignore_index=True)
    
    # End of Elo update
    
    df_live.to_csv(LIVE_ELO_PATH, index=False)
    print(f"Elo update complete. Final date: {yesterday.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    update_elo()
