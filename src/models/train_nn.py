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
    from src.config import PROCESSED_DIR, LSTM_MODEL_PATH
    import joblib
    from sklearn.preprocessing import StandardScaler
    
    X_path = str(PROCESSED_DIR / 'X_sequences.npy')
    y_path = str(PROCESSED_DIR / 'y_sequences.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Could not find sequence files. Generating dummy data for dry run...")
        X = np.random.rand(1000, 15, 12)
        y = np.random.randint(0, 2, 1000)
    else:
        # Load with mmap_mode to prevent RAM explosion
        X = np.load(X_path, mmap_mode='r')
        y = np.load(y_path, mmap_mode='r')
        
        # 2. Slice for local dry run (Protect RAM)
        print(f"Original shape: {X.shape}. Slicing to last 50,000 sequences...")
        X = X[-50000:].copy() # Copy to RAM after slicing
        y = y[-50000:].copy()
    
    # 3. Chronological Split (80% Train, 20% Validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Scale features across all timesteps (prevent 9-orders-of-magnitude destabilization)
    scaler = StandardScaler()
    num_features = X_train.shape[2]
    
    # Reshape to 2D for scaling, then back to 3D
    X_train_2d = X_train.reshape(-1, num_features)
    X_train_2d_scaled = scaler.fit_transform(X_train_2d)
    X_train = X_train_2d_scaled.reshape(X_train.shape)
    
    X_val_2d = X_val.reshape(-1, num_features)
    X_val_2d_scaled = scaler.transform(X_val_2d)
    X_val = X_val_2d_scaled.reshape(X_val.shape)
    
    # Save the scaler
    scaler_path = str(PROCESSED_DIR / 'lstm_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to {scaler_path}")
    
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
    save_path = str(LSTM_MODEL_PATH)
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
    import mlflow
    import mlflow.keras
    
    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NBA_RealTime_WP")
    mlflow.keras.autolog()
    
    train_lstm_model()
