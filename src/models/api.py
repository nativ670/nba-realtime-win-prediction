from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import uvicorn
import os
import json

# Define the features in the exact order they were trained
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

# Pydantic model for input validation
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

    class Config:
        json_schema_extra = {
            "example": {
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
                "away_in_bonus": 1
            }
        }

# Global variable to hold the model
model = None

app = FastAPI(title="NBA Real-Time Win Probability API")

@app.on_event("startup")
def load_model():
    """Loads the XGBoost model on application startup."""
    global model
    model_path = os.path.join("src", "models", "xgb_wp_model.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first.")
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

@app.get("/")
def read_root():
    return {"message": "NBA Real-Time Win Probability API is running."}

@app.post("/predict_win_prob")
async def predict_win_prob(features: GameFeatureList):
    """
    Predicts the home team win probability based on current game state.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Convert Pydantic model to dictionary
        data_dict = features.dict()
        
        # Create a DataFrame with a single row, ensuring column order matches training
        df = pd.DataFrame([data_dict])[FEATURES]
        
        # Get probability prediction (class 1 is home win)
        # Note: model.predict_proba returns [[prob_0, prob_1]]
        probabilities = model.predict_proba(df)
        home_win_prob = float(probabilities[0][1])
        
        return {"home_win_probability": round(home_win_prob, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Mandatory Shootaround: Run the app using uvicorn
    # Note: In a production environment, you might use 'uvicorn api:app --host 0.0.0.0 --port 8000'
    uvicorn.run(app, host="0.0.0.0", port=8000)
