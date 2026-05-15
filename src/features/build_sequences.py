import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

# ==============================================================================
# LSTM SEQUENCE BUILDER
# ==============================================================================
# This script transforms partitioned play-by-play data into 3D NumPy arrays
# suitable for training an LSTM Neural Network.

# Constants
SEQUENCE_LENGTH = 15
FEATURES = [
    'score_differential', 
    'seconds_remaining_in_game', 
    'possession_team_id', 
    'elo_advantage', 
    'rest_advantage', 
    'distance_traveled', 
    'momentum_differential', 
    'home_timeouts_remaining', 
    'away_timeouts_remaining', 
    'home_in_bonus', 
    'away_in_bonus'
]
TARGET = 'home_win'

def build_lstm_sequences():
    """
    Iterates through seasonal parquet files, groups by game, and creates
    3D sequences using a sliding window approach.
    """
    data_dir = 'data/processed/seasons/'
    output_dir = 'data/processed/'
    
    # Efficiently find all partitioned season files
    files = sorted(glob.glob(os.path.join(data_dir, 'pbp_*.parquet')))
    
    if not files:
        print(f"Error: No .parquet files found in {data_dir}.")
        print("Please ensure the data ingestion and partitioning steps are complete.")
        return

    all_X = []
    all_y = []
    
    print(f"🚀 Starting sequence building from {len(files)} season files...")
    
    for file_path in tqdm(files, desc="Processing Seasons"):
        try:
            # Load one season at a time to manage RAM
            df_season = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Warning: Failed to read {file_path}. Skipping. Error: {e}")
            continue
            
        # Verify required columns exist in this partition
        required_cols = FEATURES + [TARGET, 'GAME_ID', 'EVENTNUM']
        missing = [c for c in required_cols if c not in df_season.columns]
        if missing:
            print(f"Warning: Skipping {file_path} due to missing columns: {missing}")
            continue

        # Ensure chronological order within each game
        # Sorting by GAME_ID first, then EVENTNUM ensures proper sequencing
        df_season = df_season.sort_values(['GAME_ID', 'EVENTNUM'])
        
        # Process each game individually
        for game_id, df_game in df_season.groupby('GAME_ID'):
            # Extract feature matrix and handle any potential NaNs (ffill then 0)
            # This ensures the LSTM receives clean numerical input
            game_data = df_game[FEATURES].ffill().fillna(0).values
            
            # The target (win/loss) is taken from the last row of the sequence context
            # In our case, home_win is a terminal result, constant for the game.
            game_target = df_game[TARGET].iloc[-1]
            
            # Only process games long enough to form at least one sequence
            if len(game_data) >= SEQUENCE_LENGTH:
                # Calculate the number of possible windows
                num_sequences = len(game_data) - SEQUENCE_LENGTH + 1
                
                # Vectorized sliding window view (Zero-copy until concatenation)
                # Shape: (num_sequences, 1, SEQUENCE_LENGTH, num_features)
                try:
                    sequences = np.lib.stride_tricks.sliding_window_view(
                        game_data, (SEQUENCE_LENGTH, len(FEATURES))
                    )
                    
                    # Reshape to (Batch, Timesteps, Features)
                    sequences = sequences.reshape(num_sequences, SEQUENCE_LENGTH, len(FEATURES))
                    
                    all_X.append(sequences.astype(np.float32))
                    all_y.append(np.full((num_sequences, 1), game_target, dtype=np.float32))
                except AttributeError:
                    # Fallback for older NumPy versions (< 1.20)
                    sequences = []
                    for i in range(num_sequences):
                        sequences.append(game_data[i : i + SEQUENCE_LENGTH])
                    all_X.append(np.array(sequences).astype(np.float32))
                    all_y.append(np.full((num_sequences, 1), game_target, dtype=np.float32))
                
    if not all_X:
        print("❌ No sequences were generated. Check your data filters and sequence length.")
        return

    # Combine all season/game data into final master arrays
    print("\n📦 Concatenating sequences into master arrays...")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Persistence
    os.makedirs(output_dir, exist_ok=True)
    x_file = os.path.join(output_dir, 'X_sequences.npy')
    y_file = os.path.join(output_dir, 'y_sequences.npy')
    
    np.save(x_file, X)
    np.save(y_file, y)
    
    print(f"✅ Success! Saved LSTM training data to {output_dir}")
    print(f"📊 Final X_sequences Shape: {X.shape}")
    print(f"📊 Final y_sequences Shape: {y.shape}")

if __name__ == "__main__":
    build_lstm_sequences()
