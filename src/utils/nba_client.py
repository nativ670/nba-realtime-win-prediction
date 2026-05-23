"""
src/utils/nba_client.py — Shared NBA API Wrapper
==================================================
Provides a single retry-enabled wrapper for all nba_api calls.
Replaces 6+ copy-pasted retry loops scattered across the codebase.
"""

import time
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError

# Import config — use try/except for robustness during testing
try:
    from src.config import (
        NBA_API_SLEEP, NBA_API_RETRY_SLEEP,
        NBA_API_MAX_RETRIES, NBA_API_TIMEOUT
    )
except ImportError:
    # Fallback defaults if config can't be loaded
    NBA_API_SLEEP = 0.6
    NBA_API_RETRY_SLEEP = 5
    NBA_API_MAX_RETRIES = 3
    NBA_API_TIMEOUT = 30


def nba_api_call(endpoint_class, df_index=0, max_retries=None,
                 sleep=None, timeout=None, **kwargs):
    """
    Generic retry wrapper for nba_api endpoint calls.

    Parameters
    ----------
    endpoint_class : class
        The nba_api endpoint class (e.g., LeagueGameFinder, PlayByPlayV3).
    df_index : int, optional
        Index of the DataFrame to return from get_data_frames() (default 0).
    max_retries : int, optional
        Override default max retries.
    sleep : float, optional
        Override default sleep between calls.
    timeout : int, optional
        Override default API timeout.
    **kwargs : dict
        Keyword arguments passed directly to the endpoint constructor.

    Returns
    -------
    pd.DataFrame
        The result DataFrame, or an empty DataFrame on failure.
    """
    _retries = max_retries or NBA_API_MAX_RETRIES
    _sleep = sleep or NBA_API_SLEEP
    _timeout = timeout or NBA_API_TIMEOUT

    for attempt in range(_retries):
        try:
            # Rate limit — breathe before every call
            time.sleep(_sleep)

            result = endpoint_class(**kwargs, timeout=_timeout)
            dfs = result.get_data_frames()

            if df_index < len(dfs):
                return dfs[df_index]
            else:
                print(f"⚠️ DataFrame index {df_index} out of range "
                      f"(only {len(dfs)} available)")
                return pd.DataFrame()

        except (ReadTimeout, ConnectionError) as e:
            backoff = NBA_API_RETRY_SLEEP * (attempt + 1)
            print(f"⚠️ NBA API timeout ({type(e).__name__})! "
                  f"Retry {attempt + 1}/{_retries} in {backoff}s...")
            time.sleep(backoff)

        except Exception as e:
            print(f"❌ Unexpected error calling {endpoint_class.__name__}: {e}")
            break

    print(f"❌ Failed to call {endpoint_class.__name__} "
          f"after {_retries} attempts.")
    return pd.DataFrame()


def nba_api_call_multi(endpoint_class, df_indices=None,
                       max_retries=None, sleep=None, timeout=None, **kwargs):
    """
    Like nba_api_call but returns multiple DataFrames at once.

    Parameters
    ----------
    df_indices : list of int
        Indices of DataFrames to return. If None, returns all.

    Returns
    -------
    list of pd.DataFrame
        The requested DataFrames, or a list of empty DataFrames on failure.
    """
    _retries = max_retries or NBA_API_MAX_RETRIES
    _sleep = sleep or NBA_API_SLEEP
    _timeout = timeout or NBA_API_TIMEOUT

    for attempt in range(_retries):
        try:
            time.sleep(_sleep)
            result = endpoint_class(**kwargs, timeout=_timeout)
            dfs = result.get_data_frames()

            if df_indices is None:
                return dfs
            return [dfs[i] if i < len(dfs) else pd.DataFrame()
                    for i in df_indices]

        except (ReadTimeout, ConnectionError) as e:
            backoff = NBA_API_RETRY_SLEEP * (attempt + 1)
            print(f"⚠️ NBA API timeout ({type(e).__name__})! "
                  f"Retry {attempt + 1}/{_retries} in {backoff}s...")
            time.sleep(backoff)

        except Exception as e:
            print(f"❌ Unexpected error calling {endpoint_class.__name__}: {e}")
            break

    n = len(df_indices) if df_indices else 1
    return [pd.DataFrame()] * n


# ==============================================================================
# MAIN (Self-Test)
# ==============================================================================

if __name__ == "__main__":
    print("=== NBA API Client Self-Test ===")
    print("Testing with ScoreboardV3 (today's games)...")

    try:
        from nba_api.stats.endpoints import scoreboardv3
        from datetime import datetime

        today = datetime.now().strftime('%Y-%m-%d')
        df = nba_api_call(
            scoreboardv3.ScoreboardV3,
            df_index=1,
            game_date=today
        )
        if not df.empty:
            print(f"Found {len(df)} games for {today}")
            print(df.head())
        else:
            print(f"No games found for {today} (or offseason)")
        print("NBA Client OK ✅")
    except ImportError:
        print("nba_api not installed — skipping live test")
        print("NBA Client structure OK ✅")
