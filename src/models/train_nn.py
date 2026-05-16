import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint

def train_lstm_model():
    """
    Loads pre-processed 3D sequence tensors, builds an LSTM architecture, 
    and trains a win probability model.
    """
    # 1. Load data
    print("Loading sequence data...")
    X_path = 'data/processed/X_sequences.npy'
    y_path = 'data/processed/y_sequences.npy'
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Could not find sequence files at {X_path} or {y_path}")
        # Creating dummy data for demonstration if files don't exist, 
        # though user expects them to be there.
        return
        
    X = np.load(X_path)
    y = np.load(y_path)
    
    # 2. Slice for local dry run (Protect RAM)
    # Keeping only the most recent 50,000 sequences
    print(f"Original shape: {X.shape}. Slicing to last 50,000 sequences...")
    X = X[-50000:]
    y = y[-50000:]
    
    # 3. Chronological Split (80% Train, 20% Validation)
    # We do NOT shuffle to prevent data leakage in time-series
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} sequences, validating on {len(X_val)}")
    
    # 4. Build Model Architecture
    model = Sequential([
        # Input shape: (timesteps, features) -> (15, 12)
        InputLayer(shape=(X.shape[1], X.shape[2])),
        
        # First LSTM Layer
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        
        # Second LSTM Layer
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Output Layer (Binary Probability)
        Dense(1, activation='sigmoid')
    ])
    
    # 5. Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 6. Callbacks
    save_path = 'src/models/lstm_v3_10man.keras'
    checkpoint = ModelCheckpoint(
        save_path, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )
    
    # 7. Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=256,
        callbacks=[checkpoint],
        verbose=1
    )
    
    print(f"Success! Best model saved to {save_path}")

if __name__ == '__main__':
    train_lstm_model()
