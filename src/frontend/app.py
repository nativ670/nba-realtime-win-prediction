import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import os
import sys
import glob

from src.config import MODEL_FEATURES
from src.utils.nba_client import nba_api_call
from nba_api.stats.endpoints import playbyplayv3, boxscoresummaryv3
from src.features.in_game import calculate_in_game_features
from src.features.substitution_tracker import SubstitutionTracker
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
        margin-bottom: 20px;
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
    .sub-metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
    }
    .sub-metric-box {
        background-color: #f3f4f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #d1d5db;
        min-width: 180px;
        text-align: center;
    }
    .live-metric-label {
        font-size: 1.1rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 5px;
    }
    .live-metric-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #111827;
    }
    .sub-metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #374151;
    }
    .live-metric-delta {
        font-size: 1.4rem;
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

# Initialize parameters directly in session state for widget binding
if 'elo_adv' not in st.session_state:
    st.session_state.elo_adv = 50.0
if 'rest_adv' not in st.session_state:
    st.session_state.rest_adv = 0
if 'dist_trav' not in st.session_state:
    st.session_state.dist_trav = 300.0
if 'poll_interval' not in st.session_state:
    st.session_state.poll_interval = 15
if 'last_action_number' not in st.session_state:
    st.session_state.last_action_number = 0
if 'accumulated_features' not in st.session_state:
    st.session_state.accumulated_features = pd.DataFrame()

# --- Data Loading ---
@st.cache_data
def load_historical_metadata():
    """Loads pre-game context from all partitioned season files."""
    from src.config import SEASONS_DIR
    seasons_dir = str(SEASONS_DIR)
    if os.path.exists(seasons_dir):
        try:
            files = glob.glob(os.path.join(seasons_dir, "*.parquet"))
            if not files:
                return None
            dfs = []
            for f in files:
                # Load only required columns to save memory
                df = pd.read_parquet(f, columns=['GAME_ID', 'elo_advantage', 'rest_advantage', 'distance_traveled'])
                dfs.append(df.drop_duplicates('GAME_ID'))
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                return combined.drop_duplicates('GAME_ID').set_index('GAME_ID')
        except Exception as e:
            st.warning(f"Could not load historical metadata: {e}")
    return None

historical_meta = load_historical_metadata()

# --- Core Functions ---

def fetch_game_metadata(game_id):
    """Fetches game metadata like team names, date, and time using BoxScoreSummaryV3."""
    try:
        df_summary = nba_api_call(boxscoresummaryv3.BoxScoreSummaryV3, df_index=1, game_id=game_id)
        df_linescore = nba_api_call(boxscoresummaryv3.BoxScoreSummaryV3, df_index=4, game_id=game_id)

        if df_summary.empty or df_linescore.empty:
            return {}

        game_date_raw = df_summary.iloc[0]['gameDate']
        try:
            game_date = pd.to_datetime(game_date_raw).strftime("%B %d, %Y")
        except Exception as e:
            game_date = game_date_raw

        def get_full_name(row):
            city = row.get('teamCity', '')
            nickname = row.get('teamName', '')
            if city and nickname:
                return f"{city} {nickname}"
            return row.get('teamTricode', 'Unknown')

        # In BoxScoreSummaryV3 LineScore: 0 is usually Away, 1 is Home
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
    """Fetches PBP data and incrementally runs it through the feature engineering engine."""
    try:
        # Initialize Substitution Tracker if not present or game changed
        if "sub_tracker" not in st.session_state or st.session_state.get("tracker_game_id") != game_id:
            st.session_state.sub_tracker = SubstitutionTracker(game_id)
            st.session_state.tracker_game_id = game_id
            st.session_state.last_action_number = 0
            st.session_state.accumulated_features = pd.DataFrame()

        if not st.session_state.is_polling:
            st.spinner(f"Fetching Play-by-Play for {game_id}...")
            
        df_pbp_raw = nba_api_call(
            playbyplayv3.PlayByPlayV3,
            df_index=0,
            timeout=30,
            game_id=game_id
        )
        
        if df_pbp_raw.empty:
            return None
            
        # Figure out the action number column
        action_col = 'actionNumber' if 'actionNumber' in df_pbp_raw.columns else 'EVENTNUM'
        max_action = int(df_pbp_raw[action_col].max()) if action_col in df_pbp_raw.columns else 0
        
        # Incremental filter
        if st.session_state.last_action_number > 0 and action_col in df_pbp_raw.columns:
            df_new_raw = df_pbp_raw[df_pbp_raw[action_col] > st.session_state.last_action_number].copy()
        else:
            df_new_raw = df_pbp_raw.copy()
            
        if df_new_raw.empty:
            return st.session_state.accumulated_features
        
        df_new = standardize_pbp_v3(df_new_raw)
        df_new_features = calculate_in_game_features(df_new)
        
        # Add Live RAPTOR Advantage
        df_new_features = st.session_state.sub_tracker.process_pbp(df_new_features)
        
        st.session_state.last_action_number = max_action
        st.session_state.accumulated_features = pd.concat(
            [st.session_state.accumulated_features, df_new_features], 
            ignore_index=True
        )
        
        return st.session_state.accumulated_features
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_predictions(df_features, elo_adv, rest_adv, dist_trav):
    """Sends feature sequences to the FastAPI server to get Win Probabilities."""
    
    existing_history = st.session_state.history
    if not existing_history.empty:
        new_plays = df_features[~df_features['EVENTNUM'].isin(existing_history['EVENTNUM'])].copy()
    else:
        new_plays = df_features.copy()
        
    if new_plays.empty and not existing_history.empty:
        return existing_history
    
    progress_bar = st.empty()
    if not st.session_state.is_polling and len(new_plays) > 20:
        progress_bar = st.progress(0)
    
    total_new = len(new_plays)
    new_results = []
    
    # Track the rolling window of features for the LSTM
    all_features_list = df_features.to_dict('records')
    
    with requests.Session() as session:
        for i, (idx, row) in enumerate(new_plays.iterrows()):
            # Find the index in the full feature list to build the sequence
            current_idx = df_features.index.get_loc(idx)
            start_idx = max(0, current_idx - 14)
            sequence_slice = all_features_list[start_idx : current_idx + 1]
            
            # Construct Sequence Payload
            payload_sequence = []
            for s_row in sequence_slice:
                payload_sequence.append({
                    "score_differential": int(s_row['score_differential']),
                    "seconds_remaining_in_game": float(s_row['seconds_remaining_in_game']),
                    "is_home_possession": float(s_row['is_home_possession']) if pd.notna(s_row['is_home_possession']) else 0.5,
                    "elo_advantage": float(elo_adv),
                    "rest_advantage": int(rest_adv),
                    "distance_traveled": float(dist_trav),
                    "momentum_differential": float(s_row['momentum_differential']),
                    "home_timeouts_remaining": int(s_row['home_timeouts_remaining']),
                    "away_timeouts_remaining": int(s_row['away_timeouts_remaining']),
                    "home_in_bonus": int(s_row['home_in_bonus']),
                    "away_in_bonus": int(s_row['away_in_bonus']),
                    "live_raptor_advantage": float(s_row.get('live_raptor_advantage', 0.0))
                })
            
            payload = {"sequence": payload_sequence}
            
            try:
                response = session.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    row_data = row.to_dict()
                    row_data['home_win_probability'] = data["final_win_probability"]
                    row_data['xgb_prob'] = data["xgb_prob"]
                    row_data['lstm_prob'] = data["lstm_prob"]
                    new_results.append(row_data)
                else:
                    st.error(f"API Error: {response.text}")
                    break
            except Exception as e:
                st.error(f"Connection Error: {e}")
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
    """Renders the updated high-visibility ensemble scoreboard."""
    score_diff = int(latest['score_differential'])
    diff_color = "#10b981" if score_diff >= 0 else "#ef4444"
    diff_sign = "+" if score_diff > 0 else ""
    
    final_wp = latest['home_win_probability'] * 100
    xgb_wp = latest.get('xgb_prob', 0) * 100
    lstm_wp = latest.get('lstm_prob', 0) * 100

    # Main Row
    st.markdown(f"""
        <div class="live-metric-container">
            <div class="live-metric-box" style="border-color: {diff_color}">
                <div class="live-metric-label">Score Differential</div>
                <div class="live-metric-value" style="color: {diff_color}">{diff_sign}{score_diff}</div>
                <div class="live-metric-delta">{int(latest['home_score'])} - {int(latest['away_score'])}</div>
            </div>
            <div class="live-metric-box">
                <div class="live-metric-label">Final Win Probability</div>
                <div class="live-metric-value">{final_wp:.1f}%</div>
                <div class="live-metric-delta">{latest['PCTIMESTRING']} Period {int(latest['PERIOD'])}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sub Row (The Debate)
    st.markdown(f"""
        <div class="sub-metric-container">
            <div class="sub-metric-box">
                <div class="live-metric-label">XGBoost (Math)</div>
                <div class="sub-metric-value">{xgb_wp:.0f}%</div>
            </div>
            <div class="sub-metric-box">
                <div class="live-metric-label">LSTM (Momentum)</div>
                <div class="sub-metric-value">{lstm_wp:.0f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if is_live:
        st.markdown('<p class="live-status">● LIVE POLLING ACTIVE</p>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
def on_game_id_change():
    new_id = st.session_state.game_id_input
    if new_id != st.session_state.current_game_id:
        st.session_state.current_game_id = new_id
        st.session_state.history = pd.DataFrame()
        st.session_state.game_meta = {}
        st.session_state.is_polling = False

        # Automatic Context Lookup
        if historical_meta is not None and new_id in historical_meta.index:
            meta = historical_meta.loc[new_id]
            st.session_state.elo_adv = float(meta['elo_advantage'] * 100)
            st.session_state.rest_adv = int(meta['rest_advantage'])
            st.session_state.dist_trav = float(meta['distance_traveled'])

if not st.session_state.is_polling:
    st.sidebar.header("⚙️ Game Settings")
    
    if "game_id_input" not in st.session_state:
        st.session_state.game_id_input = st.session_state.current_game_id or "0022300001"

    st.sidebar.text_input("Enter GAME_ID", key="game_id_input", on_change=on_game_id_change)
    
    # Initialize current_game_id if it's empty
    if not st.session_state.current_game_id:
        st.session_state.current_game_id = st.session_state.game_id_input

    st.sidebar.subheader("📋 Pre-Game Context")
    st.sidebar.number_input("Elo Advantage", key="elo_adv")
    st.sidebar.number_input("Rest Advantage", key="rest_adv")
    st.sidebar.number_input("Distance Traveled", key="dist_trav")

    st.sidebar.markdown("---")
    st.sidebar.slider("Poll Interval (s)", 5, 60, key="poll_interval")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    if col_btn1.button("📊 Analyze History"):
        st.session_state.is_polling = False
        st.session_state.history = pd.DataFrame()
        st.session_state.game_meta = fetch_game_metadata(st.session_state.current_game_id)
        st.session_state.trigger_fetch = True
    if col_btn2.button("📡 Start Live"):
        st.session_state.is_polling = True
        st.session_state.history = pd.DataFrame()
        st.session_state.game_meta = fetch_game_metadata(st.session_state.current_game_id)
        st.rerun()
else:
    st.sidebar.header("📡 Live Mode Active")
    st.sidebar.write(f"**Game ID:** {st.session_state.current_game_id}")
    if st.sidebar.button("🛑 Stop Polling"):
        st.session_state.is_polling = False
        st.rerun()

# --- Main Dashboard Logic ---
if st.session_state.game_meta:
    meta = st.session_state.game_meta
    st.markdown(f'<div class="game-header"><h2>{meta.get("matchup")}</h2><p>{meta.get("game_date")}</p></div>', unsafe_allow_html=True)

if st.session_state.is_polling or st.session_state.trigger_fetch or (not st.session_state.history.empty):
    df_features = fetch_and_process(st.session_state.current_game_id)
    if df_features is not None:
        # Use session state keys
        history = get_predictions(
            df_features, 
            st.session_state.elo_adv, 
            st.session_state.rest_adv, 
            st.session_state.dist_trav
        )
        st.session_state.trigger_fetch = False
        
        if not history.empty:
            # Ensure history is strictly chronological to prevent zig-zag plots
            # The NBA API sometimes inserts plays retroactively
            history = history.sort_values(['elapsed_time', 'EVENTNUM']).reset_index(drop=True)
            st.session_state.history = history

            latest = history.iloc[-1]
            render_scoreboard(latest, is_live=st.session_state.is_polling)

            # --- Plotly Chart ---
            st.subheader("📈 Win Probability Trend")
            chart_data = history.dropna(subset=['elapsed_time', 'home_win_probability'])
            
            if not chart_data.empty:
                x_vals = pd.to_numeric(chart_data['elapsed_time']).tolist()
                y_vals = pd.to_numeric(chart_data['home_win_probability']).tolist()
                
                # Prepare hover data
                chart_data['game_result'] = chart_data['home_score'].fillna(0).astype(int).astype(str) + " - " + chart_data['away_score'].fillna(0).astype(int).astype(str)
                chart_data['time_classic'] = chart_data['PCTIMESTRING'].fillna("") + " Q" + chart_data['PERIOD'].fillna(0).astype(int).astype(str)
                # Overtime labeling
                chart_data.loc[chart_data['PERIOD'] > 4, 'time_classic'] = \
                    chart_data['PCTIMESTRING'].fillna("") + " OT" + (chart_data['PERIOD'] - 4).fillna(0).astype(int).astype(str)

                # Split color logic
                x_above, y_above = [], []
                x_below, y_below = [], []
                
                for i in range(len(y_vals)):
                    if y_vals[i] >= 0.5:
                        if i > 0 and y_vals[i-1] < 0.5:
                            # Intersect
                            t = x_vals[i-1] + (0.5 - y_vals[i-1]) * (x_vals[i] - x_vals[i-1]) / (y_vals[i] - y_vals[i-1])
                            x_above.append(t); y_above.append(0.5)
                            x_below.append(t); y_below.append(0.5); x_below.append(None); y_below.append(None)
                        x_above.append(x_vals[i]); y_above.append(y_vals[i])
                    else:
                        if i > 0 and y_vals[i-1] >= 0.5:
                            # Intersect
                            t = x_vals[i-1] + (0.5 - y_vals[i-1]) * (x_vals[i] - x_vals[i-1]) / (y_vals[i] - y_vals[i-1])
                            x_above.append(t); y_above.append(0.5); x_above.append(None); y_above.append(None)
                            x_below.append(t); y_below.append(0.5)
                        x_below.append(x_vals[i]); y_below.append(y_vals[i])

                fig = go.Figure()
                fig.add_hrect(y0=0.5, y1=1.0, fillcolor="#3b82f6", opacity=0.05, line_width=0, layer="below")
                fig.add_hrect(y0=0.0, y1=0.5, fillcolor="#ef4444", opacity=0.05, line_width=0, layer="below")
                fig.add_hline(y=0.5, line_dash="solid", line_color="rgba(0,0,0,0.1)", line_width=1)

                # Quarter/OT Lines
                max_period = int(chart_data['PERIOD'].max())
                for p in range(1, max_period + 1):
                    line_x = p * 720 if p <= 4 else 2880 + (p - 4) * 300
                    fig.add_vline(x=line_x, line_dash="dash", line_color="rgba(0,0,0,0.2)")

                # Blue Trace (Above 50%)
                fig.add_trace(go.Scatter(
                    x=x_above, y=y_above, mode='lines', 
                    line=dict(color='#3b82f6', width=3), 
                    connectgaps=False, showlegend=False,
                    hoverinfo='skip'
                ))
                # Red Trace (Below 50%)
                fig.add_trace(go.Scatter(
                    x=x_below, y=y_below, mode='lines', 
                    line=dict(color='#ef4444', width=3), 
                    connectgaps=False, showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Invisible Trace for Hover
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, 
                    mode='lines+markers' if len(chart_data) < 50 else 'lines', 
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    marker=dict(size=4, color='rgba(0,0,0,0)'),
                    hovertemplate="<b>Time:</b> %{customdata[0]}<br><b>Win Prob:</b> %{y:.1%}<br><b>Score:</b> %{customdata[1]}<extra></extra>",
                    customdata=chart_data[['time_classic', 'game_result']].values,
                    showlegend=False
                ))

                fig.update_layout(
                    template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), height=500,
                    xaxis=dict(title="Game Clock (s)", range=[0, max(2880, max(x_vals)+60)], showgrid=False),
                    yaxis=dict(title="Win Prob", tickformat=".0%", range=[0,1], showgrid=False),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, width='stretch')

if st.session_state.is_polling:
    time.sleep(st.session_state.poll_interval)
    st.rerun()
