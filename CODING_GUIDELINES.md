# NBA Real-Time WP — Coding Guidelines

To ensure the codebase remains clean, maintainable, and free of the spaghetti patterns and bugs we recently cleaned up, all future development must adhere to these structural guidelines.

## 1. Centralized Configuration (Single Source of Truth)
Never hardcode file paths, mathematical constants (like `K_FACTOR`), or the `MODEL_FEATURES` array directly into individual scripts.
- **Rule:** Import everything from `src.config`.
- **Why:** If a path changes or we add a 13th feature, we only want to update one file, not ten. 
- **Example:**
  ```python
  # ❌ BAD
  df.to_csv("data/processed/live_elo.csv")
  
  # ✅ GOOD
  from src.config import LIVE_ELO_PATH
  df.to_csv(LIVE_ELO_PATH)
  ```

## 2. API Interactions & Rate Limiting
Never write custom `time.sleep()` retry loops for NBA API endpoints. The `nba_api` package aggressively throttles requests, especially in CI environments like GitHub Actions.
- **Rule:** Use the `nba_api_call` wrapper located in `src.utils.nba_client`.
- **Why:** It contains hardened exponential backoff logic and standardizes how we catch `ReadTimeout` and `ConnectionError`.
- **Example:**
  ```python
  # ❌ BAD
  try:
      pbp = playbyplayv3.PlayByPlayV3(game_id=gid).get_data_frames()[0]
  except:
      time.sleep(5)
      
  # ✅ GOOD
  from src.utils.nba_client import nba_api_call
  pbp = nba_api_call(playbyplayv3.PlayByPlayV3, df_index=0, game_id=gid)
  ```

## 3. Running Scripts & Imports
Never use `sys.path.append(os.path.abspath(".."))` to force Python to find the `src` directory.
- **Rule:** Run scripts from the project root as Python modules using the `-m` flag.
- **Why:** It eliminates path-hacking boilerplate and prevents "ModuleNotFoundError" bugs.
- **Example:**
  ```bash
  # ❌ BAD
  python src/features/build_sequences.py
  
  # ✅ GOOD
  python -m src.features.build_sequences
  ```

## 4. Incremental Processing (Efficiency)
Avoid fetching or re-engineering the entire history of a game (or season) every time a script runs.
- **Rule:** Track the high-water mark (e.g., `last_action_number`, or the last processed date) and only process *new* rows.
- **Why:** Running 400+ rows through feature engineering every 15 seconds causes timeouts on the live tracker. 

## 5. Script Organization
Do not place one-time utility scripts (like backfilling missing data or restructuring parquet files) in the `src/` directory.
- **Rule:** Production logic goes in `src/`. Ad-hoc, exploratory, or one-time maintenance scripts go in `scripts/` or `notebooks/`.

## 6. Error Handling
Do not swallow errors silently, especially in automated pipelines.
- **Rule:** Avoid bare `except:` blocks. Log the specific exception. If a critical script (like `daily_update.py`) fails, it must call `sys.exit(1)`.
- **Why:** GitHub Actions relies on exit codes to know if a daily run succeeded or failed. If Python exits with `0` after swallowing an error, we won't get alerted when the pipeline breaks.
