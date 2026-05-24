"""
src/data_ingestion/update_pbp.py — Daily PBP Updater
=====================================================
Fetches recent NBA games, processes play-by-play data,
computes in-game and pre-game features, and appends
them to the per-season parquet files.

Flow:  get_games_for_range → update_pbp → process each game → save
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

# --- Project Imports (shared utilities, no duplication) ---
from src.config import (
    SEASONS_DIR, LIVE_ELO_PATH, NBA_API_SLEEP,
    DAILY_UPDATE_LOOKBACK_DAYS, get_season_year_from_game_id
)
from src.utils.nba_client import nba_api_call
from src.features.in_game import calculate_in_game_features
from src.features.pregame import haversine_vectorized
from src.utils.helpers import standardize_pbp_v3, TEAM_COORDS


# ==============================================================================
# HELPER: Fetch games in a date range via shared NBA API wrapper
# ==============================================================================

def get_games_for_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches games played in a date range from nba_api using the shared
    retry wrapper (nba_api_call) with 30-day chunking to avoid timeouts.

    Parameters
    ----------
    start_date, end_date : str
        Date strings in 'YYYY-MM-DD' (or 'MM/DD/YYYY' — the API accepts both).

    Returns
    -------
    pd.DataFrame  — LeagueGameFinder results, or empty DataFrame on failure.
    """
    print(f"Checking for games between {start_date} and {end_date}...")
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    all_chunks = []
    curr_start = start_dt
    
    while curr_start <= end_dt:
        curr_end = min(curr_start + timedelta(days=30), end_dt)
        print(f"  -> Chunk: {curr_start.strftime('%Y-%m-%d')} to {curr_end.strftime('%Y-%m-%d')}")
        games = nba_api_call(
            leaguegamefinder.LeagueGameFinder,
            df_index=0,
            date_from_nullable=curr_start.strftime('%Y-%m-%d'),
            date_to_nullable=curr_end.strftime('%Y-%m-%d'),
            league_id_nullable='00'
        )
        if not games.empty:
            all_chunks.append(games)
            
        curr_start = curr_end + timedelta(days=1)
        
    if all_chunks:
        return pd.concat(all_chunks, ignore_index=True)
    return pd.DataFrame()


# ==============================================================================
# HELPER: Compute actual rest advantage for a game
# ==============================================================================

def _compute_rest_days(team_abbr: str, game_date: pd.Timestamp,
                       games_df: pd.DataFrame) -> int:
    """
    Looks up a team's previous game in the schedule DataFrame and
    returns the number of rest days (capped at 5).  Defaults to 5
    if no prior game is found in the current window.

    Parameters
    ----------
    team_abbr : str
        Team abbreviation (e.g. 'BOS').
    game_date : pd.Timestamp
        Date of the current game.
    games_df : pd.DataFrame
        Full games DataFrame from get_games_for_range(), must have
        'TEAM_ABBREVIATION' and 'GAME_DATE' columns (datetime).

    Returns
    -------
    int  — rest days, capped at 5.
    """
    team_games = games_df.loc[
        (games_df['TEAM_ABBREVIATION'] == team_abbr) &
        (games_df['GAME_DATE'] < game_date),
        'GAME_DATE'
    ]
    if team_games.empty:
        return 5  # No prior game in window → assume well-rested

    prev_date = team_games.max()
    rest = (game_date - prev_date).days - 1
    return int(min(max(rest, 0), 5))


# ==============================================================================
# HELPER: Compute travel distance between consecutive games
# ==============================================================================

def _compute_travel_distance(team_abbr: str, opponent_abbr: str,
                              is_home: bool) -> float:
    """
    Estimates travel distance (miles) for the *away* team using
    the Haversine formula and TEAM_COORDS.  Returns 0 for the
    home team (they didn't travel).

    Parameters
    ----------
    team_abbr : str   — The team whose perspective we're computing.
    opponent_abbr : str
    is_home : bool    — True if `team_abbr` is the home team.

    Returns
    -------
    float — distance in miles (0 for home team or if coords missing).
    """
    if is_home:
        return 0.0  # Home team didn't travel for this game

    coords_team = TEAM_COORDS.get(team_abbr)
    coords_opp = TEAM_COORDS.get(opponent_abbr)
    if coords_team is None or coords_opp is None:
        return 0.0

    return float(haversine_vectorized(
        coords_team[0], coords_team[1],
        coords_opp[0], coords_opp[1]
    ))


# ==============================================================================
# HELPER: Extract home & away abbreviations from a game's rows
# ==============================================================================

def _get_home_away_abbrevs(game_rows: pd.DataFrame):
    """
    Given all rows for a single GAME_ID from LeagueGameFinder,
    returns (home_abbr, away_abbr).

    Home team is the one whose MATCHUP does NOT contain '@'.
    """
    home_rows = game_rows[~game_rows['MATCHUP'].str.contains('@')]
    away_rows = game_rows[game_rows['MATCHUP'].str.contains('@')]

    home_abbr = home_rows['TEAM_ABBREVIATION'].iloc[0] if not home_rows.empty else None
    away_abbr = away_rows['TEAM_ABBREVIATION'].iloc[0] if not away_rows.empty else None
    return home_abbr, away_abbr


# ==============================================================================
# MAIN: update_pbp — fetch, process, and save new PBP data
# ==============================================================================

def update_pbp(start_date=None, end_date=None):
    """
    End-to-end daily updater.

    1. Determines the catch-up date range.
    2. Fetches game list via LeagueGameFinder.
    3. Downloads PBP for each new game.
    4. Computes in-game features + pre-game context (Elo, rest, travel).
    5. Appends to the correct per-season parquet file.
    """

    # ------------------------------------------------------------------
    # 1. Determine catch-up range
    # ------------------------------------------------------------------
    if start_date is None:
        # Try to infer from live_elo.csv (our source of truth for finished games)
        if LIVE_ELO_PATH.exists():
            df_elo_dates = pd.read_csv(LIVE_ELO_PATH, usecols=['date', 'score1', 'score2'])
            df_elo_dates['date'] = pd.to_datetime(df_elo_dates['date'], errors='coerce')
            finished = df_elo_dates.dropna(subset=['score1', 'score2'])
            if not finished.empty:
                last_elo_date = finished['date'].max()
                start_date = (last_elo_date - timedelta(days=DAILY_UPDATE_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # ------------------------------------------------------------------
    # 2. Fetch game list for the date range
    # ------------------------------------------------------------------
    games = get_games_for_range(start_date, end_date)
    if games.empty:
        print(f"No games found in range {start_date} to {end_date}. Nothing to update.")
        return

    # Ensure GAME_DATE is datetime for rest-day calculations
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], errors='coerce')

    # Derive season year from GAME_ID using shared config helper
    games['SEASON_YEAR'] = games['GAME_ID'].apply(get_season_year_from_game_id)

    # ------------------------------------------------------------------
    # 3. Load Elo data once (used for all seasons in this run)
    # ------------------------------------------------------------------
    elo_df = pd.DataFrame()
    if LIVE_ELO_PATH.exists():
        elo_df = pd.read_csv(LIVE_ELO_PATH)
        elo_df['date'] = pd.to_datetime(elo_df['date'], format='mixed')

    # ------------------------------------------------------------------
    # 4. Process each season group
    # ------------------------------------------------------------------
    for season_year, season_games in games.groupby('SEASON_YEAR'):
        season_file = SEASONS_DIR / f"pbp_{season_year}.parquet"

        # Load existing season parquet (or start fresh)
        if season_file.exists():
            df_season = pd.read_parquet(season_file)
        else:
            print(f"Season file {season_file} not found. Initializing new season...")
            df_season = pd.DataFrame()

        # Unique game IDs in this season window
        game_ids = season_games['GAME_ID'].unique()

        # ------------------------------------------------------------------
        # 4a. Download & compute in-game features for each NEW game
        # ------------------------------------------------------------------
        all_new_pbp = []
        processed_game_ids = set()  # Track actually-processed IDs (fix #7)

        for gid in game_ids:
            # Skip already-processed games
            if not df_season.empty and gid in df_season['GAME_ID'].values:
                continue

            print(f"  - Processing PBP for Game {gid}...")

            # Use shared nba_api_call wrapper (fix #5)
            df_pbp_raw = nba_api_call(
                playbyplayv3.PlayByPlayV3,
                df_index=0,
                game_id=gid
            )

            if df_pbp_raw.empty:
                print(f"    [!] Skipping {gid} due to empty PBP data.")
                continue

            try:
                # Standardize V3 → legacy format, then calculate in-game features
                df_pbp = standardize_pbp_v3(df_pbp_raw)
                df_features = calculate_in_game_features(df_pbp)

                # Calculate target variable (home_win)
                last_play = df_pbp.iloc[-1]
                home_final = float(last_play['SCORE_HOME'])
                away_final = float(last_play['SCORE_AWAY'])
                df_features['home_win'] = int(home_final > away_final)

                all_new_pbp.append(df_features)
                processed_game_ids.add(gid)  # Record success
            except Exception as e:
                print(f"    Error processing features for {gid}: {e}")

        if not all_new_pbp:
            print(f"No new PBP data to append for season {season_year}.")
            continue

        df_new_pbp = pd.concat(all_new_pbp, ignore_index=True)

        # ------------------------------------------------------------------
        # 4b. Build pre-game context (Elo, rest, travel) for processed games
        # ------------------------------------------------------------------
        print(f"Calculating pre-game context for {len(processed_game_ids)} new games...")

        new_game_meta = []
        for gid in processed_game_ids:  # Only iterate actually-processed IDs (fix #7)
            # Look up game rows from the LeagueGameFinder results
            game_rows = season_games[season_games['GAME_ID'] == gid]
            game_date = game_rows['GAME_DATE'].iloc[0]  # Already datetime

            # --- Identify home / away teams ---
            home_abbr, away_abbr = _get_home_away_abbrevs(game_rows)

            # --- Elo Advantage (fix #3: compare datetime to datetime) ---
            elo_adv = 0.0
            if not elo_df.empty and home_abbr is not None:
                try:
                    # Convert game_date to datetime for comparison (fix #3)
                    game_date_dt = pd.Timestamp(game_date)
                    match_elo = elo_df[elo_df['date'] == game_date_dt]
                    game_elo_row = match_elo[
                        (match_elo['team1'] == home_abbr) &
                        (match_elo['is_home'] == 1)
                    ]
                    if not game_elo_row.empty:
                        row = game_elo_row.iloc[0]
                        elo_adv = (row['elo1_pre'] - row['elo2_pre']) / 100.0
                except Exception as e:  # fix #1: no bare except
                    print(f"    ⚠️ Elo lookup failed for {gid}: {e}")
                    elo_adv = 0.0

            # --- Rest Advantage (fix #2: actual computation) ---
            home_rest = _compute_rest_days(home_abbr, game_date, games) if home_abbr else 3
            away_rest = _compute_rest_days(away_abbr, game_date, games) if away_abbr else 3
            rest_advantage = home_rest - away_rest

            # --- Travel Distance (fix #2: actual computation) ---
            # Away team travels; home team distance = 0
            distance_traveled = _compute_travel_distance(
                away_abbr, home_abbr, is_home=False
            ) if (away_abbr and home_abbr) else 0.0

            # --- Playoff flag from GAME_ID prefix ---
            is_playoffs = 1 if gid.startswith('004') else 0

            new_game_meta.append({
                'GAME_ID': gid,
                'elo_advantage': elo_adv,
                'rest_advantage': rest_advantage,
                'distance_traveled': distance_traveled,
                'is_playoffs': is_playoffs
            })

        # Merge pre-game context onto the new PBP rows
        df_meta = pd.DataFrame(new_game_meta)
        df_new_final = df_new_pbp.merge(df_meta, on='GAME_ID', how='left')

        # ------------------------------------------------------------------
        # 5. Append and save
        # ------------------------------------------------------------------
        SEASONS_DIR.mkdir(parents=True, exist_ok=True)
        df_updated = pd.concat([df_season, df_new_final], ignore_index=True)
        df_updated.to_parquet(season_file, index=False)
        print(f"✅ Updated {season_file} with {len(processed_game_ids)} new games.")


# ==============================================================================
# MAIN (Self-Test)
# ==============================================================================

if __name__ == "__main__":
    """
    Lightweight test: runs update_pbp for yesterday only.
    Safe to execute — it will only fetch one day of games and
    skip any that are already in the season parquet.
    """
    print("=== update_pbp Self-Test ===")

    # Test the helper functions with mock data
    print("\n--- Testing _compute_rest_days ---")
    mock_games = pd.DataFrame({
        'TEAM_ABBREVIATION': ['BOS', 'BOS', 'LAL'],
        'GAME_DATE': pd.to_datetime(['2025-01-10', '2025-01-13', '2025-01-12']),
        'GAME_ID': ['001', '002', '003'],
        'MATCHUP': ['BOS vs. LAL', 'BOS vs. MIA', 'LAL @ BOS']
    })
    rest = _compute_rest_days('BOS', pd.Timestamp('2025-01-15'), mock_games)
    print(f"  BOS rest before 2025-01-15: {rest} days (expected 1)")

    print("\n--- Testing _compute_travel_distance ---")
    dist = _compute_travel_distance('BOS', 'LAL', is_home=False)
    print(f"  BOS → LAL (away): {dist:.0f} miles (expected ~2600)")
    dist_home = _compute_travel_distance('LAL', 'BOS', is_home=True)
    print(f"  LAL at home: {dist_home:.0f} miles (expected 0)")

    print("\n--- Testing get_games_for_range (yesterday) ---")
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    df_test = get_games_for_range(yesterday, yesterday)
    if df_test.empty:
        print(f"  No games found for {yesterday} (possibly offseason).")
    else:
        print(f"  Found {df_test['GAME_ID'].nunique()} games for {yesterday}.")

    print("\nSelf-test complete ✅")
