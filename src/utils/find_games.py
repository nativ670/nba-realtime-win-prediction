from nba_api.stats.endpoints import scoreboardv3
from datetime import datetime, timedelta
import pandas as pd
import sys
import os
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def find_games(target_date=None):
    """
    Fetches NBA games for a specific date and prints their Game IDs, matchups, and status.
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
        
    try:
        print(f"[*] Fetching NBA games for {target_date}...")
        
        # Query ScoreboardV3
        sb = scoreboardv3.ScoreboardV3(game_date=target_date)
        df_games = sb.get_data_frames()[1] # Scoreboard
        df_teams = sb.get_data_frames()[2] # TeamScoreboard
        
        if df_games.empty:
            print(f"[?] No games found for {target_date}.")
            return

        print(f"\n{'GAME_ID':<12} | {'MATCHUP':<25} | {'STATUS'}")
        print("-" * 55)
        
        for _, row in df_games.iterrows():
            game_id = row['gameId']
            teams = df_teams[df_teams['gameId'] == game_id]
            
            if len(teams) >= 2:
                # In ScoreboardV3 Table 2, the two rows per game are away then home (usually)
                # But let's be safe and just show both.
                team_abbrevs = teams['teamTricode'].tolist()
                matchup = f"{team_abbrevs[0]} @ {team_abbrevs[1]}"
            else:
                matchup = "Unknown Matchup"
                
            status = row['gameStatusText']
            
            print(f"{game_id:<12} | {matchup:<25} | {status}")
            
    except Exception as e:
        print(f"[!] Error fetching games: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find NBA Game IDs for a given date.")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()
    
    find_games(args.date)
