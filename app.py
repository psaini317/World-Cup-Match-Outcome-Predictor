import streamlit as st
import pandas as pd
import joblib
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from flags import get_flag

# load model and data
model = joblib.load("models/best_model.joblib")
with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)
matches = pd.read_csv("data/processed/matches_featured.csv")

all_teams = sorted(set(
    matches["home_team"].unique().tolist() +
    matches["away_team"].unique().tolist()
))




# FRONTEND

# Page config
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="https://flagcdn.com/16x12/un.png",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@500&display=swap');

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .main { background-color: #080b12; }

        .app-header {
            padding: 32px 0 8px 0;
            border-bottom: 1px solid #1e2433;
            margin-bottom: 32px;
        }
        .app-title {
            font-size: 26px;
            font-weight: 700;
            color: #f0f2f8;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .app-subtitle {
            font-size: 13px;
            color: #4e576e;
            margin-top: 4px;
            font-weight: 400;
        }

        .label {
            font-size: 11px;
            font-weight: 600;
            color: #4e576e;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 6px;
        }
        .flag-name {
            font-size: 28px;
            font-weight: 700;
            color: #f0f2f8;
            letter-spacing: -0.02em;
            margin: 10px 0 2px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .rank-chip {
            display: inline-block;
            background: #131825;
            border: 1px solid #1e2433;
            border-radius: 6px;
            padding: 4px 10px;
            font-family: 'DM Mono', monospace;
            font-size: 12px;
            color: #6b7a99;
            margin-top: 6px;
        }

        .divider { border: none; border-top: 1px solid #1e2433; margin: 28px 0; }

        .result-wrap { border-radius: 10px; padding: 22px 24px; margin: 20px 0 28px 0; }
        .result-home { background: #0d1f16; border: 1px solid #1a4028; }
        .result-draw { background: #1a1608; border: 1px solid #3d2f00; }
        .result-away { background: #0d1525; border: 1px solid #1a2d4a; }
        .result-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }
        .result-label-home { color: #2d6a4f; }
        .result-label-draw { color: #8a6500; }
        .result-label-away { color: #2d5080; }
        .result-text { font-size: 22px; font-weight: 700; color: #f0f2f8; letter-spacing: -0.02em; }

        .prob-section { margin-top: 8px; }
        .prob-row { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; }
        .prob-label { font-size: 13px; color: #6b7a99; width: 80px; white-space: nowrap; }
        .prob-track { flex: 1; background: #131825; border-radius: 4px; height: 8px; overflow: hidden; }
        .prob-fill { height: 8px; border-radius: 4px; transition: width 0.4s ease; }
        .prob-pct { font-family: 'DM Mono', monospace; font-size: 13px; color: #f0f2f8; width: 44px; text-align: right; }

        .context-row { display: flex; gap: 24px; margin-top: 20px; }
        .context-card {
            flex: 1;
            background: #0d1020;
            border: 1px solid #1e2433;
            border-radius: 8px;
            padding: 14px 16px;
        }
        .context-team { font-size: 13px; font-weight: 600; color: #c9d0e3; margin-bottom: 6px; }
        .context-stat { font-size: 12px; color: #4e576e; line-height: 1.8; }

        .stButton > button {
            background: #1a4028 !important;
            color: #52b788 !important;
            border: 1px solid #2d6a4f !important;
            border-radius: 8px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            padding: 12px 0 !important;
            width: 100% !important;
            letter-spacing: 0.02em !important;
            transition: all 0.2s !important;
        }
        .stButton > button:hover {
            background: #2d6a4f !important;
            color: #ffffff !important;
        }

        #MainMenu, footer, header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)



# Header
st.markdown("""
    <div class='app-header'>
        <div class='app-title'>World Cup Match Predictor</div>
        <div class='app-subtitle'>XGBoost model trained on 17,000+ FIFA matches</div>
    </div>
""", unsafe_allow_html=True)

# Team selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='label'>Home Team</div>", unsafe_allow_html=True)
    home_team = st.selectbox("", all_teams, key="home", label_visibility="collapsed")
    home_rank_data = matches[matches["home_team"] == home_team]["home_rank"]
    home_rank = home_rank_data.iloc[-1] if len(home_rank_data) > 0 else 50
    flag = get_flag(home_team)
    st.markdown(f"""
        <div class='flag-name'>{flag} {home_team}</div>
        <div class='rank-chip'>FIFA #{int(home_rank)}</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='label'>Away Team</div>", unsafe_allow_html=True)
    away_team = st.selectbox("", all_teams, key="away", label_visibility="collapsed")
    away_rank_data = matches[matches["away_team"] == away_team]["away_rank"]
    away_rank = away_rank_data.iloc[-1] if len(away_rank_data) > 0 else 50
    flag = get_flag(away_team)
    st.markdown(f"""
        <div class='flag-name'>{flag} {away_team}</div>
        <div class='rank-chip'>FIFA #{int(away_rank)}</div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
neutral = st.checkbox("Neutral venue")
if neutral:
    st.caption("Note: home/away order may still affect predictions at neutral venues.")
st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Predict Match Outcome", use_container_width=True)

# Prediction
if predict_button:
    if home_team == away_team:
        st.error("Select two different teams.")
    else:
        home_form_data = matches[matches["home_team"] == home_team]["home_form"]
        away_form_data = matches[matches["away_team"] == away_team]["away_form"]
        home_form = home_form_data.iloc[-1] if len(home_form_data) > 0 else 0.5
        away_form = away_form_data.iloc[-1] if len(away_form_data) > 0 else 0.5
        rank_difference = home_rank - away_rank

        input_data = pd.DataFrame([[
            home_rank, away_rank, rank_difference,
            home_form, away_form, int(neutral)
        ]], columns=feature_columns)

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        home_pct = round(float(probabilities[0]) * 100, 1)
        draw_pct = round(float(probabilities[1]) * 100, 1)
        away_pct = round(float(probabilities[2]) * 100, 1)

        # Result box
        if prediction == 0:
            rc = "result-home"
            rl = "result-label-home"
            label = "Predicted Winner"
            text = f"{get_flag(home_team)} {home_team}"
        elif prediction == 1:
            rc = "result-draw"
            rl = "result-label-draw"
            label = "Predicted Result"
            text = "Match ends in a Draw"
        else:
            rc = "result-away"
            rl = "result-label-away"
            label = "Predicted Winner"
            text = f"{get_flag(away_team)} {away_team}"

        st.markdown(f"""
            <div class='result-wrap {rc}'>
                <div class='result-label {rl}'>{label}</div>
                <div class='result-text'>{text}</div>
            </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("<div class='label'>Win Probabilities</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='prob-section'>
                <div class='prob-row'>
                    <div class='prob-label'>Home Win</div>
                    <div class='prob-track'><div class='prob-fill' style='width:{home_pct}%; background:#2d6a4f;'></div></div>
                    <div class='prob-pct'>{home_pct}%</div>
                </div>
                <div class='prob-row'>
                    <div class='prob-label'>Draw</div>
                    <div class='prob-track'><div class='prob-fill' style='width:{draw_pct}%; background:#8a6500;'></div></div>
                    <div class='prob-pct'>{draw_pct}%</div>
                </div>
                <div class='prob-row'>
                    <div class='prob-label'>Away Win</div>
                    <div class='prob-track'><div class='prob-fill' style='width:{away_pct}%; background:#2d5080;'></div></div>
                    <div class='prob-pct'>{away_pct}%</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Match context cards
        st.markdown(f"""
            <div class='context-row'>
                <div class='context-card'>
                    <div class='context-team'>{get_flag(home_team)} {home_team}</div>
                    <div class='context-stat'>
                        FIFA Rank: #{int(home_rank)}<br>
                        Recent Form: {home_form:.0%} win rate
                    </div>
                </div>
                <div class='context-card'>
                    <div class='context-team'>{get_flag(away_team)} {away_team}</div>
                    <div class='context-stat'>
                        FIFA Rank: #{int(away_rank)}<br>
                        Recent Form: {away_form:.0%} win rate
                    </div>
                </div>
            </div>
            <div style='font-size:12px; color:#4e576e; margin-top:14px;'>
                Rank gap: {int(abs(rank_difference))} places &nbsp;·&nbsp; Neutral venue: {'Yes' if neutral else 'No'}
            </div>
        """, unsafe_allow_html=True)