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

def get_yesterday_games():
    """Fetches games played yesterday from nba_api."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    # For testing/initial run, we might want a specific date or range
    # But for the cron job, it's yesterday.
    
    try:
        print(f"Fetching games for {yesterday}...")
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

def update_elo():
    # 1. Load existing Elo
    if os.path.exists(LIVE_ELO_PATH):
        df_live = pd.read_csv(LIVE_ELO_PATH)
        df_live['date'] = pd.to_datetime(df_live['date'])
    else:
        print("Initializing live_elo.csv from historical data...")
        df_live = fetch_elo_data()
        if df_live.empty:
            print("Failed to fetch historical data. Exiting.")
            return
        # Ensure directories exist
        os.makedirs(os.path.dirname(LIVE_ELO_PATH), exist_ok=True)
        df_live.to_csv(LIVE_ELO_PATH, index=False)

    # 2. Get the latest Elo for each team
    # We need to find the latest elo_post for every team in the league
    latest_elos = {}
    
    # Get all teams
    teams = pd.concat([df_live['team1'], df_live['team2']]).unique()
    
    # Sort by date and get last occurrences
    # A team could be team1 or team2. The post-game Elo is elo1_post or elo2_post.
    # It's easier to reconstruct a 'last_elo' series.
    for team in teams:
        # Check rows where team is team1
        t1_rows = df_live[df_live['team1'] == team]
        # Check rows where team is team2
        t2_rows = df_live[df_live['team2'] == team]
        
        last_t1 = t1_rows.sort_values('date').iloc[-1] if not t1_rows.empty else None
        last_t2 = t2_rows.sort_values('date').iloc[-1] if not t2_rows.empty else None
        
        if last_t1 is not None and last_t2 is not None:
            if last_t1['date'] >= last_t2['date']:
                latest_elos[team] = last_t1['elo1_post']
            else:
                latest_elos[team] = last_t2['elo2_post']
        elif last_t1 is not None:
            latest_elos[team] = last_t1['elo1_post']
        elif last_t2 is not None:
            latest_elos[team] = last_t2['elo2_post']

    # 3. Fetch yesterday's games
    games, date_str = get_yesterday_games()
    if games.empty:
        print(f"No games found for {date_str}.")
        return

    # nba_api returns two rows per game. Let's pair them.
    game_ids = games['GAME_ID'].unique()
    new_rows = []
    
    # Detect current season (heuristic: 2023-24 -> 2024)
    # Most recent season in df_live
    last_season = df_live['season'].max()

    for gid in game_ids:
        game_pair = games[games['GAME_ID'] == gid]
        if len(game_pair) != 2:
            continue
            
        row_a = game_pair.iloc[0]
        row_b = game_pair.iloc[1]
        
        # Identify Home/Away
        if '@' in row_a['MATCHUP']:
            away_row, home_row = row_a, row_b
        else:
            home_row, away_row = row_a, row_b
            
        team_h = home_row['TEAM_ABBREVIATION']
        team_a = away_row['TEAM_ABBREVIATION']
        score_h = home_row['PTS']
        score_a = away_row['PTS']
        
        # Check for season regression
        # (Simplified: if current date is after October and last_season < current_year)
        # Better: check the 'season' in the API if available, or use the date.
        # NBA regular season usually starts late Oct.
        game_date = datetime.strptime(date_str, '%Y-%m-%d')
        current_season = game_date.year if game_date.month >= 10 else game_date.year
        # Wait, if month is Jan-June, it's current year's season. If Oct-Dec, it's next year's season.
        current_season = game_date.year + 1 if game_date.month >= 10 else game_date.year
        
        if current_season > last_season:
            print(f"New season detected ({current_season}). Applying regression...")
            for t in latest_elos:
                latest_elos[t] = (OFFSEASON_REGRESSION_FACTOR * latest_elos[t]) + ((1 - OFFSEASON_REGRESSION_FACTOR) * OFFSEASON_MEAN)
            last_season = current_season

        # Get pre-game Elos (fallback to 1500 if new team, though unlikely)
        elo_h_pre = latest_elos.get(team_h, 1500)
        elo_a_pre = latest_elos.get(team_a, 1500)
        
        # Calculate probabilities for the CSV (538 style)
        prob_h = calculate_expected_score(elo_h_pre, elo_a_pre, hca=HOME_ADVANTAGE)
        prob_a = 1 - prob_h
        
        # Calculate post-game Elos
        elo_h_post, elo_a_post = calculate_new_elos(elo_h_pre, elo_a_pre, score_h, score_a, is_home=True)
        
        # Create the two rows (one for team1=home, one for team1=away)
        # Row 1: Home perspective
        new_rows.append({
            'date': date_str,
            'season': current_season,
            'neutral': 0,
            'playoff': np.nan,
            'team1': team_h,
            'team2': team_a,
            'elo1_pre': elo_h_pre,
            'elo2_pre': elo_a_pre,
            'elo_prob1': prob_h,
            'elo_prob2': prob_a,
            'elo1_post': elo_h_post,
            'elo2_post': elo_a_post,
            'score1': score_h,
            'score2': score_a,
            'is_home': 1
        })
        # Row 2: Away perspective
        new_rows.append({
            'date': date_str,
            'season': current_season,
            'neutral': 0,
            'playoff': np.nan,
            'team1': team_a,
            'team2': team_h,
            'elo1_pre': elo_a_pre,
            'elo2_pre': elo_h_pre,
            'elo_prob1': prob_a,
            'elo_prob2': prob_h,
            'elo1_post': elo_a_post,
            'elo2_post': elo_h_post,
            'score1': score_a,
            'score2': score_h,
            'is_home': 0
        })
        
        # Update latest_elos for next games in the same day (if any, though rare to have same team twice)
        latest_elos[team_h] = elo_h_post
        latest_elos[team_a] = elo_a_post

    # 4. Append and Save
    df_new = pd.DataFrame(new_rows)
    df_updated = pd.concat([df_live, df_new], ignore_index=True)
    df_updated.to_csv(LIVE_ELO_PATH, index=False)
    print(f"Successfully added {len(new_rows)} rows to {LIVE_ELO_PATH}.")

if __name__ == "__main__":
    update_elo()
