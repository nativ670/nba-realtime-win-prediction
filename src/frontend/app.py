import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import os
import sys
import glob

# --- Path Injection ---
# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nba_api.stats.endpoints import playbyplayv3, boxscoresummaryv2
from src.features.in_game import calculate_in_game_features
from src.utils.helpers import standardize_pbp_v3

# --- Configuration ---
st.set_page_config(page_title="NBA Live Win Probability", page_icon="🏀", layout="wide")

# Custom CSS for a cleaner, light theme
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    /* Standard Streamlit Metrics Visibility */
    [data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-size: 1.1rem !important;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    /* Make metrics much larger in Live Mode */
    .live-metric-container {
        display: flex;
        justify-content: space-around;
        text-align: center;
        margin-bottom: 30px;
        gap: 20px;
    }
    .live-metric-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #3b82f6;
        min-width: 250px;
        flex: 1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .live-metric-label {
        font-size: 1.3rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }
    .live-metric-value {
        font-size: 3.8rem;
        font-weight: 800;
        color: #111827;
    }
    .live-metric-delta {
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 5px;
    }
    .live-status {
        color: #059669;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 20px;
        animation: blinker 1.5s linear infinite;
    }
    .game-header {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-bottom: 20px;
        text-align: center;
    }
    .game-header h2 {
        margin: 0;
        color: #1f2937;
    }
    .game-header p {
        margin: 5px 0 0 0;
        color: #6b7280;
        font-size: 1rem;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏀 NBA Real-Time Win Probability")

# API Endpoint (Make sure api.py is running!)
API_URL = "http://127.0.0.1:8000/predict_win_prob"

# --- Session State Management ---
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()
if "current_game_id" not in st.session_state:
    st.session_state.current_game_id = ""
if "game_meta" not in st.session_state:
    st.session_state.game_meta = {}
if "is_polling" not in st.session_state:
    st.session_state.is_polling = False
if "trigger_fetch" not in st.session_state:
    st.session_state.trigger_fetch = False
if "params" not in st.session_state:
    st.session_state.params = {
        'elo_adv': 50.0,
        'rest_adv': 0,
        'dist_trav': 300.0,
        'poll_interval': 15
    }

# --- Data Loading ---
@st.cache_data
def load_historical_metadata():
    """Loads pre-game context from the most recent partitioned season file."""
    seasons_dir = "data/processed/seasons"
    if os.path.exists(seasons_dir):
        try:
            # Get the most recent season file
            files = glob.glob(os.path.join(seasons_dir, "*.parquet"))
            if not files:
                return None
            latest_file = max(files, key=os.path.getmtime)
            
            # Load only necessary columns to save memory
            df = pd.read_parquet(latest_file, columns=['GAME_ID', 'elo_advantage', 'rest_advantage', 'distance_traveled'])
            # Keep only one row per game
            return df.drop_duplicates('GAME_ID').set_index('GAME_ID')
        except Exception as e:
            st.warning(f"Could not load historical metadata: {e}")
    return None

historical_meta = load_historical_metadata()

# --- Core Functions ---

def fetch_game_metadata(game_id):
    """Fetches game metadata like team names, date, and time."""
    try:
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
        df_summary = summary.get_data_frames()[0] # GameSummary
        df_linescore = summary.get_data_frames()[5] # LineScore

        if df_summary.empty or df_linescore.empty:
            return {}

        game_date_raw = df_summary.iloc[0]['GAME_DATE_EST']
        # Convert date to a nicer format if possible
        try:
            game_date = pd.to_datetime(game_date_raw).strftime("%B %d, %Y")
        except:
            game_date = game_date_raw

        # Get Team Names
        def get_full_name(row):
            if 'TEAM_NAME' in row and pd.notna(row['TEAM_NAME']):
                return row['TEAM_NAME']
            city = row.get('TEAM_CITY_NAME', '')
            nickname = row.get('TEAM_NICKNAME', '')
            if city and nickname:
                return f"{city} {nickname}"
            return row.get('TEAM_ABBREVIATION', 'Unknown')

        home_team = get_full_name(df_linescore.iloc[1]) if len(df_linescore) > 1 else "Home Team"
        away_team = get_full_name(df_linescore.iloc[0]) if len(df_linescore) > 1 else "Away Team"
        
        return {
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "matchup": f"{away_team} @ {home_team}"
        }
    except Exception as e:
        st.error(f"Error fetching game metadata: {e}")
        return {}

def fetch_and_process(game_id):
    """Fetches PBP data and runs it through the feature engineering engine."""
    try:
        # Reduced noise for live polling
        if st.session_state.is_polling:
             pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        else:
            with st.spinner(f"Fetching Play-by-Play for {game_id}..."):
                pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        
        df_pbp_raw = pbp.get_data_frames()[0]
        
        if df_pbp_raw.empty:
            st.warning(f"No play-by-play data found for Game ID {game_id}.")
            return None
        
        # Standardize columns for the engine
        df_pbp = standardize_pbp_v3(df_pbp_raw)
        
        # Process features using our shared engine
        df_features = calculate_in_game_features(df_pbp)
        return df_features
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_predictions(df_features, elo_adv, rest_adv, dist_trav):
    """Sends feature rows to the FastAPI server to get Win Probabilities."""
    
    existing_history = st.session_state.history
    
    # Identify only new plays based on EVENTNUM to avoid redundant API calls
    if not existing_history.empty:
        new_plays = df_features[~df_features['EVENTNUM'].isin(existing_history['EVENTNUM'])].copy()
    else:
        new_plays = df_features.copy()
        
    if new_plays.empty and not existing_history.empty:
        return existing_history
    
    # Progress bar only for historical re-calculation
    progress_bar = st.empty()
    if not st.session_state.is_polling and len(new_plays) > 20:
        progress_bar = st.progress(0)
    
    total_new = len(new_plays)
    new_results = []
    
    # We'll use a session to speed up requests
    with requests.Session() as session:
        for i, (idx, row) in enumerate(new_plays.iterrows()):
            payload = {
                "score_differential": int(row['score_differential']),
                "seconds_remaining_in_game": float(row['seconds_remaining_in_game']),
                "possession_team_id": int(row['possession_team_id']) if pd.notna(row['possession_team_id']) else 0,
                "elo_advantage": float(elo_adv),
                "rest_advantage": int(rest_adv),
                "distance_traveled": float(dist_trav),
                "momentum_differential": float(row['momentum_differential']),
                "home_timeouts_remaining": int(row['home_timeouts_remaining']),
                "away_timeouts_remaining": int(row['away_timeouts_remaining']),
                "home_in_bonus": int(row['home_in_bonus']),
                "away_in_bonus": int(row['away_in_bonus'])
            }
            
            try:
                response = session.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    wp = response.json()["home_win_probability"]
                    row_data = row.to_dict()
                    row_data['home_win_probability'] = wp
                    new_results.append(row_data)
                else:
                    st.error(f"API Error at EVENT {row['EVENTNUM']}: {response.text}")
                    break
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Is the FastAPI server running at http://127.0.0.1:8000?")
                return existing_history
            
            if not st.session_state.is_polling and total_new > 20:
                progress_bar.progress((i + 1) / total_new)
                
    progress_bar.empty()
    
    if new_results:
        combined = pd.concat([existing_history, pd.DataFrame(new_results)], ignore_index=True)
        st.session_state.history = combined
        return combined
    
    return existing_history

def render_scoreboard(latest, is_live=False):
    """Renders a high-visibility scoreboard."""
    score_diff = int(latest['score_differential'])
    diff_color = "#10b981" if score_diff >= 0 else "#ef4444"
    diff_sign = "+" if score_diff > 0 else ""
    
    wp_pct = latest['home_win_probability'] * 100
    wp_color = "#3b82f6" # Neutral blue for WP

    if is_live:
        # High-Visibility "Live" Dashboard
        st.markdown(f"""
            <div class="live-metric-container">
                <div class="live-metric-box">
                    <div class="live-metric-label">Score</div>
                    <div class="live-metric-value">{int(latest['home_score'])} - {int(latest['away_score'])}</div>
                    <div class="live-metric-delta" style="color: {diff_color}">{diff_sign}{score_diff}</div>
                </div>
                <div class="live-metric-box">
                    <div class="live-metric-label">Time / Period</div>
                    <div class="live-metric-value">{latest['PCTIMESTRING']}</div>
                    <div class="live-metric-delta">Period {int(latest['PERIOD'])}</div>
                </div>
                <div class="live-metric-box">
                    <div class="live-metric-label">Win Probability</div>
                    <div class="live-metric-value" style="color: {wp_color}">{wp_pct:.1f}%</div>
                    <div class="live-metric-delta">Home Team</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<p class="live-status">● LIVE POLLING ACTIVE</p>', unsafe_allow_html=True)
    else:
        # Standard Metrics for historical analysis
        col1, col2, col3 = st.columns(3)
        col1.metric("Score", f"{int(latest['home_score'])} - {int(latest['away_score'])}", f"{score_diff}")
        col2.metric("Clock", f"{latest['PCTIMESTRING']}", f"Period {int(latest['PERIOD'])}")
        col3.metric("Home Win Prob", f"{wp_pct:.1f}%")

# --- Sidebar Inputs ---
if not st.session_state.is_polling:
    st.sidebar.header("⚙️ Game Settings")
    game_id = st.sidebar.text_input("Enter GAME_ID (e.g., 0022300001)", st.session_state.current_game_id or "0022300001")

    # Handle GAME_ID Change
    if game_id != st.session_state.current_game_id:
        st.session_state.history = pd.DataFrame()
        st.session_state.current_game_id = game_id
        st.session_state.game_meta = {} # Clear old meta
        st.session_state.is_polling = False

        # Automatic Context Lookup on GAME_ID change
        if historical_meta is not None and game_id in historical_meta.index:
            meta = historical_meta.loc[game_id]
            st.session_state.params['elo_adv'] = float(meta['elo_advantage'] * 100)
            st.session_state.params['rest_adv'] = int(meta['rest_advantage'])
            st.session_state.params['dist_trav'] = float(meta['distance_traveled'])
            st.sidebar.success(f"✅ Found historical context for {game_id}")

    st.sidebar.subheader("📋 Pre-Game Context")
    st.session_state.params['elo_adv'] = st.sidebar.number_input(
        "Elo Advantage (Home - Away)", 
        value=st.session_state.params['elo_adv'], 
        help="Difference in Elo ratings between home and away team."
    )
    st.session_state.params['rest_adv'] = st.sidebar.number_input(
        "Rest Advantage (Days)", 
        value=st.session_state.params['rest_adv'], 
        help="Days of rest difference."
    )
    st.session_state.params['dist_trav'] = st.sidebar.number_input(
        "Away Distance Traveled (Miles)", 
        value=st.session_state.params['dist_trav'], 
        help="Distance the away team traveled to the venue."
    )

    st.sidebar.markdown("---")
    st.session_state.params['poll_interval'] = st.sidebar.slider(
        "Poll Interval (seconds)", 5, 60, st.session_state.params['poll_interval']
    )

    # Action Buttons
    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("📊 Analyze History"):
        st.session_state.is_polling = False
        st.session_state.history = pd.DataFrame() # Clear to force full re-fetch
        st.session_state.game_meta = fetch_game_metadata(game_id)
        st.session_state.trigger_fetch = True

    if col_btn2.button("📡 Start Live"):
        st.session_state.is_polling = True
        st.session_state.history = pd.DataFrame() # Start fresh for live
        st.session_state.game_meta = fetch_game_metadata(game_id)
        st.rerun()
else:
    # Minimalist Sidebar for Live Mode
    st.sidebar.header("📡 Live Mode Active")
    st.sidebar.write(f"**Game ID:** {st.session_state.current_game_id}")
    if st.session_state.game_meta:
        st.sidebar.write(f"**Matchup:** {st.session_state.game_meta.get('matchup')}")
        st.sidebar.write(f"**Date:** {st.session_state.game_meta.get('game_date')}")
    st.sidebar.write(f"**Polling Interval:** {st.session_state.params['poll_interval']}s")
    
    if st.sidebar.button("🛑 Stop Polling"):
        st.session_state.is_polling = False
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("Settings are hidden during Live Mode for a cleaner view. Stop polling to change settings.")

# Use params from session state
current_elo = st.session_state.params['elo_adv']
current_rest = st.session_state.params['rest_adv']
current_dist = st.session_state.params['dist_trav']
poll_interval = st.session_state.params['poll_interval']
game_id = st.session_state.current_game_id

# --- Display Game Header ---
if st.session_state.game_meta:
    meta = st.session_state.game_meta
    st.markdown(f"""
        <div class="game-header">
            <h2>{meta.get('matchup')}</h2>
            <p>{meta.get('game_date')}</p>
        </div>
    """, unsafe_allow_html=True)

# --- Main Dashboard Logic ---

# 1. Trigger Data Fetching & Prediction
df_features = None
history = None
if st.session_state.is_polling or st.session_state.trigger_fetch or (not st.session_state.history.empty):
    df_features = fetch_and_process(game_id if not st.session_state.is_polling else st.session_state.current_game_id)
    if df_features is not None:
        # Use current parameters (from session state if polling)
        history = get_predictions(df_features, current_elo, current_rest, current_dist)
        st.session_state.trigger_fetch = False
        
        if not history.empty:
            latest = history.iloc[-1]
            
            # Use specialized scoreboard renderer
            render_scoreboard(latest, is_live=st.session_state.is_polling)

            # --- Chart ---
            st.subheader("📈 Win Probability Trend")
            
            # Clean data for plotting: remove NaNs and ensure numeric
            chart_data = history.copy()
            chart_data = chart_data.dropna(subset=['elapsed_time', 'home_win_probability'])
            
            if not chart_data.empty:
                # Use elapsed_time from the feature engine
                x_vals = pd.to_numeric(chart_data['elapsed_time']).tolist()
                y_vals = pd.to_numeric(chart_data['home_win_probability']).tolist()
                
                # Prepare hover data
                chart_data['game_result'] = chart_data['home_score'].fillna(0).astype(int).astype(str) + " - " + chart_data['away_score'].fillna(0).astype(int).astype(str)
                chart_data['time_classic'] = chart_data['PCTIMESTRING'].fillna("") + " Q" + chart_data['PERIOD'].fillna(0).astype(int).astype(str)
                # Fix for Overtime labeling
                chart_data.loc[chart_data['PERIOD'] > 4, 'time_classic'] = \
                    chart_data['PCTIMESTRING'].fillna("") + " OT" + (chart_data['PERIOD'] - 4).fillna(0).astype(int).astype(str)

                # Split into above and below 50% for color switching
                x_above, y_above = [], []
                x_below, y_below = [], []
                
                for i in range(len(y_vals)):
                    if y_vals[i] >= 0.5:
                        if i > 0 and y_vals[i-1] < 0.5:
                            # Intersect at 0.5
                            t = x_vals[i-1] + (0.5 - y_vals[i-1]) * (x_vals[i] - x_vals[i-1]) / (y_vals[i] - y_vals[i-1])
                            x_above.append(t); y_above.append(0.5)
                            x_below.append(t); y_below.append(0.5)
                            # Break red line
                            x_below.append(None); y_below.append(None)
                        
                        x_above.append(x_vals[i])
                        y_above.append(y_vals[i])
                    else:
                        if i > 0 and y_vals[i-1] >= 0.5:
                            # Intersect at 0.5
                            t = x_vals[i-1] + (0.5 - y_vals[i-1]) * (x_vals[i] - x_vals[i-1]) / (y_vals[i] - y_vals[i-1])
                            x_above.append(t); y_above.append(0.5)
                            x_below.append(t); y_below.append(0.5)
                            # Break blue line
                            x_above.append(None); y_above.append(None)
                        
                        x_below.append(x_vals[i])
                        y_below.append(y_vals[i])

                fig = go.Figure()

                # Background Regions
                fig.add_hrect(y0=0.5, y1=1.0, fillcolor="#3b82f6", opacity=0.05, line_width=0, layer="below")
                fig.add_hrect(y0=0.0, y1=0.5, fillcolor="#ef4444", opacity=0.05, line_width=0, layer="below")

                # 50% Baseline
                fig.add_hline(y=0.5, line_dash="solid", line_color="rgba(0,0,0,0.1)", line_width=1)

                # Quarter/OT Lines
                max_period = int(chart_data['PERIOD'].max()) if not chart_data['PERIOD'].empty else 4
                for p in range(1, max_period + 1):
                    line_x = p * 720 if p <= 4 else 2880 + (p - 4) * 300
                    style = "solid" if p == 4 else "dash"
                    color = "rgba(0,0,0,0.3)" if p == 4 else "rgba(0,0,0,0.1)"
                    fig.add_vline(x=line_x, line_dash=style, line_color=color)

                # Blue Trace (Above 50%)
                fig.add_trace(go.Scatter(
                    x=x_above, y=y_above,
                    mode='lines',
                    line=dict(color='#3b82f6', width=3),
                    name='Home Advantage',
                    hoverinfo='skip',
                    connectgaps=False
                ))

                # Red Trace (Below 50%)
                fig.add_trace(go.Scatter(
                    x=x_below, y=y_below,
                    mode='lines',
                    line=dict(color='#ef4444', width=3),
                    name='Away Advantage',
                    hoverinfo='skip',
                    connectgaps=False
                ))
                
                # Transparent Trace for Hover (Full Data)
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers' if len(chart_data) < 50 else 'lines',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    marker=dict(size=4, color='rgba(0,0,0,0)'),
                    name='Win Prob',
                    hovertemplate="<b>Time:</b> %{customdata[0]}<br>" +
                                "<b>Win Prob:</b> %{y:.1%}<br>" +
                                "<b>Score:</b> %{customdata[1]}<extra></extra>",
                    customdata=chart_data[['time_classic', 'game_result']].values
                ))

                # Layout adjustments
                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=500,
                    showlegend=False,
                    xaxis=dict(
                        title="Game Clock (seconds elapsed)",
                        showgrid=False,
                        range=[0, max(2880, max(x_vals) + 60 if x_vals else 2880)]
                    ),
                    yaxis=dict(
                        title="Home Win Probability",
                        tickformat=".0%",
                        range=[0, 1],
                        showgrid=False
                    ),
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data to plot trend line.")
            
            if not st.session_state.is_polling:
                with st.expander("🔍 Detailed In-Game Features"):
                    st.json({
                        "Possession ID": int(latest['possession_team_id']) if pd.notna(latest['possession_team_id']) else "N/A",
                        "Home Timeouts": int(latest['home_timeouts_remaining']),
                        "Away Timeouts": int(latest['away_timeouts_remaining']),
                        "Home in Bonus": bool(latest['home_in_bonus']),
                        "Away in Bonus": bool(latest['away_in_bonus']),
                        "Momentum Diff": round(latest['momentum_differential'], 2)
                    })

# 2. Polling Logic
if st.session_state.is_polling:
    st.caption(f"Next update in {poll_interval} seconds...")
    time.sleep(poll_interval)
    st.rerun()
else:
    if history is None:
        st.info("👋 Welcome! Enter a Game ID and click 'Analyze History' or 'Start Live' to begin.")
        st.image("nba_feature_importance.png", caption="Model Feature Importance", use_container_width=True)

# Footer
st.markdown("---")
st.caption("Data provided by nba_api. Predictions powered by XGBoost. Built for the NBA Real-Time WP Predictor project.")
