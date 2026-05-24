from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model as load_keras_model
from typing import List

from src.config import (
    MODEL_FEATURES, XGB_MODEL_PATH, LSTM_MODEL_PATH,
    XGB_WEIGHT, LSTM_WEIGHT, LSTM_SEQUENCE_LENGTH
)

# Pydantic model for individual game state
class GameFeatureList(BaseModel):
    score_differential: int = Field(..., description="Home Score - Away Score")
    seconds_remaining_in_game: float = Field(..., description="Seconds remaining in the game")
    possession_team_id: int = Field(..., description="ID of the team currently in possession")
    elo_advantage: float = Field(..., description="Home Elo - Away Elo")
    rest_advantage: int = Field(..., description="Home Rest Days - Away Rest Days")
    distance_traveled: float = Field(..., description="Distance traveled by the away team")
    momentum_differential: float = Field(..., description="Momentum metric differential")
    home_timeouts_remaining: int = Field(..., description="Home team timeouts left")
    away_timeouts_remaining: int = Field(..., description="Away team timeouts left")
    home_in_bonus: int = Field(..., description="1 if home team is in bonus, 0 otherwise")
    away_in_bonus: int = Field(..., description="1 if away team is in bonus, 0 otherwise")
    live_raptor_advantage: float = Field(..., description="Real-time RAPTOR advantage on the floor")

# Pydantic model for ensemble input
class EnsemblePayload(BaseModel):
    sequence: List[GameFeatureList] = Field(..., max_items=15, description="A sequence of up to 15 game feature states")

    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [
                    {
                        "score_differential": 5,
                        "seconds_remaining_in_game": 300.0,
                        "possession_team_id": 1610612737,
                        "elo_advantage": 50.5,
                        "rest_advantage": 1,
                        "distance_traveled": 450.0,
                        "momentum_differential": 2.5,
                        "home_timeouts_remaining": 3,
                        "away_timeouts_remaining": 2,
                        "home_in_bonus": 0,
                        "away_in_bonus": 1,
                        "live_raptor_advantage": 2.5
                    }
                ]
            }
        }

# Global variables to hold the models
xgb_model = None
lstm_model = None

app = FastAPI(title="NBA Real-Time Win Probability API - The Blender")

@app.on_event("startup")
def load_models():
    """Loads both XGBoost and LSTM models on application startup."""
    global xgb_model, lstm_model
    xgb_path = XGB_MODEL_PATH
    lstm_path = LSTM_MODEL_PATH
    
    # Load XGBoost
    if not xgb_path.exists():
        print(f"Warning: XGBoost model not found at {xgb_path}")
    else:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(xgb_path))
        print(f"XGBoost model loaded successfully from {xgb_path}")
    
    # Load LSTM
    if not lstm_path.exists():
        print(f"Warning: LSTM model not found at {lstm_path}")
    else:
        lstm_model = load_keras_model(str(lstm_path))
        print(f"LSTM model loaded successfully from {lstm_path}")

@app.get("/")
def read_root():
    return {"message": "NBA Real-Time Win Probability API (The Blender) is running."}

@app.post("/predict_win_prob")
async def predict_win_prob(payload: EnsemblePayload):
    """
    Predicts the home team win probability using an ensemble of XGBoost and LSTM models.
    """
    if xgb_model is None or lstm_model is None:
        raise HTTPException(status_code=500, detail="Models are not fully loaded.")
    
    if not payload.sequence:
        raise HTTPException(status_code=400, detail="Sequence cannot be empty.")

    try:
        # --- 1. PREPARE DATA ---
        # Convert Pydantic sequence to a list of dicts
        raw_sequence = [item.dict() for item in payload.sequence]
        
        # Create a DataFrame for feature extraction and consistency
        df_seq = pd.DataFrame(raw_sequence)[MODEL_FEATURES]
        
        # --- 2. XGBOOST PREDICTION ---
        # XGBoost expects the most recent state (last item in sequence)
        last_state = df_seq.tail(1)
        xgb_probs = xgb_model.predict_proba(last_state)
        xgb_prob = float(xgb_probs[0][1])
        
        # --- 3. LSTM PREDICTION ---
        # LSTM expects a 3D array of shape (1, 15, feature_count)
        # Convert sequence to numpy array
        seq_array = df_seq.values
        
        # Handle padding if sequence length < LSTM_SEQUENCE_LENGTH
        if len(seq_array) < LSTM_SEQUENCE_LENGTH:
            padding_len = LSTM_SEQUENCE_LENGTH - len(seq_array)
            # Pad the beginning with copies of the first play (or zeros, but first play is often more representative)
            padding = np.repeat(seq_array[0:1], padding_len, axis=0)
            seq_array = np.vstack([padding, seq_array])
        elif len(seq_array) > LSTM_SEQUENCE_LENGTH:
            # Should not happen due to Pydantic max_items validation, but for safety:
            seq_array = seq_array[-LSTM_SEQUENCE_LENGTH:]
            
        # Reshape for LSTM: (1, LSTM_SEQUENCE_LENGTH, feature_count)
        lstm_input = seq_array.reshape(1, LSTM_SEQUENCE_LENGTH, len(MODEL_FEATURES))
        lstm_probs = lstm_model.predict(lstm_input, verbose=0)
        lstm_prob = float(lstm_probs[0][0])
        
        # --- 4. THE BLEND (Weighted Average) ---
        # XGBoost: {XGB_WEIGHT*100}%, LSTM: {LSTM_WEIGHT*100}%
        final_win_probability = (XGB_WEIGHT * xgb_prob) + (LSTM_WEIGHT * lstm_prob)
        
        return {
            "xgb_prob": round(xgb_prob, 4),
            "lstm_prob": round(lstm_prob, 4),
            "final_win_probability": round(final_win_probability, 4)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
