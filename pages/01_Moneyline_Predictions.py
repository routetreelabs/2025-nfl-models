#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.title("Moneyline Model – Logistic Regression")
st.markdown("**Week 1 Record:** 14–2 ✅")

# Controls
debug = st.checkbox("Debug mode (print intermediate variables)")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_FINAL.csv")

# Load dataset
df = pd.read_csv(csv_path)

if debug:
    st.subheader("Head()")
    st.dataframe(df.head())

# Binary target
df['Win_Binary'] = df['Win']

# Feature engineering
df['Tm_3DConv_Rate'] = df['Tm_3DConv'] / df['Tm_3DAtt'].replace(0, 1)
df['Opp_3DConv_Rate'] = df['Opp_3DConv'] / df['Opp_3DAtt'].replace(0, 1)
df['Turnover_Diff'] = df['Opp_TO'] - df['Tm_TO']

stat_cols = [
    'Tm_pY/A', 'Tm_rY/A', 'Tm_Y/P',
    'Opp_pY/A', 'Opp_rY/A', 'Opp_Y/P',
    'Tm_TO', 'Opp_TO', 'Tm_PenYds', 'Opp_PenYds',
    'Tm_3DConv_Rate', 'Opp_3DConv_Rate',
    'Turnover_Diff'
]

# Leakage-free rolling averages
for col in stat_cols:
    df[f'{col}_avg'] = (
        df.groupby(['Season', 'Team'])[col]
          .apply(lambda x: x.shift().expanding().mean())
          .reset_index(level=[0,1], drop=True)
    )

# Fill NaN (only Week 1 rows) with league average
for col in stat_cols:
    mask = df[f'{col}_avg'].isna()
    league_avg = df[col].mean()
    df.loc[mask, f'{col}_avg'] = league_avg

# Features
features_avg = ['Spread', 'Total', 'Home'] + [f'{col}_avg' for col in stat_cols]
df_clean = df.dropna(subset=features_avg + ['Win_Binary'])

if debug:
    st.write("Dataset shape:", df_clean.shape)
    st.write("Missing values after dropna():")
    st.write(df_clean.isna().sum())

X = df_clean[features_avg]
y = df_clean['Win_Binary']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Evaluation
st.write(f"**Train Accuracy:** {model.score(X_train, y_train):.2%}")
st.write(f"**Test Accuracy:** {model.score(X_test, y_test):.2%}")

if debug:
    y_pred = model.predict(X_test)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# Helper function to build feature rows for predictions
def get_team_features(df, season, team, spread, total, home):
    # Grab the most recent row for this team (latest game played)
    team_row = df[(df['Season'] == season) & (df['Team'] == team)].iloc[-1]

    # Start with Vegas info
    features = {"Spread": spread, "Total": total, "Home": home}

    # Add rolling averages for all stat columns
    for c in stat_cols:
        features[f"{c}_avg"] = team_row[f"{c}_avg"]

    return features


# Shared team list
week2_teams = [
    {"Home": "Packers", "Away": "Commanders"},
    {"Home": "Cowboys", "Away": "Giants"},
    {"Home": "Jets", "Away": "Bills"},
    {"Home": "Saints", "Away": "49ers"},
    {"Home": "Titans", "Away": "Rams"},
    {"Home": "Steelers", "Away": "Seahawks"},
    {"Home": "Bengals", "Away": "Jaguars"},
    {"Home": "Ravens", "Away": "Browns"},
    {"Home": "Lions", "Away": "Bears"},
    {"Home": "Dolphins", "Away": "Patriots"},
    {"Home": "Cardinals", "Away": "Panthers"},
    {"Home": "Colts", "Away": "Broncos"},
    {"Home": "Chiefs", "Away": "Eagles"},
    {"Home": "Vikings", "Away": "Falcons"},
    {"Home": "Texans", "Away": "Buccaneers"},
    {"Home": "Raiders", "Away": "Chargers"}
]

# FanDuel Week 2 Predictions
st.markdown("---")
st.subheader("Week 2 Predictions - FanDuel Lines")


# Example FanDuel lines for Week 2 (replace with real spreads/totals)
week2_games_fd = [
    get_team_features(df, 2025, "GNB", spread=-3.5, total=48.5, home=1),
    get_team_features(df, 2025, "WAS", spread=+3.5, total=48.5, home=0),
    get_team_features(df, 2025, "DAL", spread=-4.5, total=44.5, home=1),
    get_team_features(df, 2025, "NYG", spread=+4.5, total=44.5, home=0),
    get_team_features(df, 2025, "NYJ", spread=+6.5, total=47.5, home=1),
    get_team_features(df, 2025, "BUF", spread=-6.5, total=47.5, home=0),
    get_team_features(df, 2025, "OTI", spread=+5.5, total=41.5, home=1),
    get_team_features(df, 2025, "RAM", spread=-5.5, total=41.5, home=0),
    get_team_features(df, 2025, "PIT", spread=-3.0, total=40.5, home=1),
    get_team_features(df, 2025, "SEA", spread=+3.0, total=40.5, home=0),
    get_team_features(df, 2025, "NOR", spread=+3.0, total=40.5, home=1),
    get_team_features(df, 2025, "SFO", spread=-3.0, total=40.5, home=0),
    get_team_features(df, 2025, "CIN", spread=-3.5, total=49.5, home=1),
    get_team_features(df, 2025, "JAX", spread=+3.5, total=49.5, home=0),
    get_team_features(df, 2025, "RAV", spread=-12.5, total=45.5, home=1),
    get_team_features(df, 2025, "CLE", spread=+12.5, total=45.5, home=0),
    get_team_features(df, 2025, "DET", spread=-6.5, total=46.5, home=1),
    get_team_features(df, 2025, "CHI", spread=+6.5, total=46.5, home=0),
    get_team_features(df, 2025, "MIA", spread=-2.5, total=43.5, home=1),
    get_team_features(df, 2025, "NWE", spread=+2.5, total=43.5, home=0),
    get_team_features(df, 2025, "CRD", spread=-7.0, total=44.5, home=1),
    get_team_features(df, 2025, "CAR", spread=+7.0, total=44.5, home=0),
    get_team_features(df, 2025, "CLT", spread=+2.5, total=43.5, home=1),
    get_team_features(df, 2025, "DEN", spread=-2.5, total=43.5, home=0),
    get_team_features(df, 2025, "KAN", spread=+1.5, total=46.5, home=1),
    get_team_features(df, 2025, "PHI", spread=-1.5, total=46.5, home=0),
    get_team_features(df, 2025, "MIN", spread=-3.5, total=44.5, home=1),
    get_team_features(df, 2025, "ATL", spread=+3.5, total=44.5, home=0),
    get_team_features(df, 2025, "HTX", spread=-2.5, total=42.5, home=1),
    get_team_features(df, 2025, "TAM", spread=+2.5, total=42.5, home=0),
    get_team_features(df, 2025, "RAI", spread=+3.0, total=46.5, home=1),
    get_team_features(df, 2025, "SDG", spread=-3.0, total=46.5, home=0)
]
week2_df_fd = pd.DataFrame(week2_games_fd)

if st.button("Run Week 2 Predictions - FanDuel"):
    probs = model.predict_proba(week2_df_fd[features_avg])[:, 1]
    preds = (probs >= 0.5).astype(int)

    results_fd = []
    for i in range (0, len(probs),2): # Step by 2
        game_index = i // 2
        home = week2_teams[game_index]["Home"]
        away = week2_teams[game_index]["Away"]

        prob = probs[i] # Home team
        pred = preds[i] # 1 = Home wins, 0 = Away wins

        winner = home if pred == 1 else away
        results_fd.append({
            "Matchup": f"{away} @ {home}",
            "Home Win Probability": prob,
            "Predicted Winner": winner
        })

    out_fd = pd.DataFrame(results_fd)
    st.dataframe(out_fd.style.format({"Home Win Probability": "{:.2%}"}))

# DraftKings Week 2 Predictions
st.markdown("---")
st.subheader("Week 2 Predictions - DraftKings Lines")

week2_games_dk = [
    get_team_features(df, 2025, "GNB", spread=-3.5, total=48.5, home=1),
    get_team_features(df, 2025, "WAS", spread=+3.5, total=48.5, home=0),
    get_team_features(df, 2025, "DAL", spread=-4.5, total=44.5, home=1),
    get_team_features(df, 2025, "NYG", spread=+4.5, total=44.5, home=0),
    get_team_features(df, 2025, "NYJ", spread=+6.5, total=47.5, home=1),
    get_team_features(df, 2025, "BUF", spread=-6.5, total=47.5, home=0),
    get_team_features(df, 2025, "OTI", spread=+5.5, total=41.5, home=1),
    get_team_features(df, 2025, "RAM", spread=-5.5, total=41.5, home=0),
    get_team_features(df, 2025, "PIT", spread=-3.5, total=40.5, home=1),
    get_team_features(df, 2025, "SEA", spread=+3.5, total=40.5, home=0),
    get_team_features(df, 2025, "NOR", spread=+3.0, total=40.5, home=1),
    get_team_features(df, 2025, "SFO", spread=-3.0, total=40.5, home=0),
    get_team_features(df, 2025, "CIN", spread=-3.5, total=49.5, home=1),
    get_team_features(df, 2025, "JAX", spread=+3.5, total=49.5, home=0),
    get_team_features(df, 2025, "RAV", spread=-12.5, total=45.5, home=1),
    get_team_features(df, 2025, "CLE", spread=+12.5, total=45.5, home=0),
    get_team_features(df, 2025, "DET", spread=-6.0, total=46.5, home=1),
    get_team_features(df, 2025, "CHI", spread=+6.0, total=46.5, home=0),
    get_team_features(df, 2025, "MIA", spread=-1.5, total=43.5, home=1),
    get_team_features(df, 2025, "NWE", spread=+1.5, total=43.5, home=0),
    get_team_features(df, 2025, "CRD", spread=-7.0, total=44.5, home=1),
    get_team_features(df, 2025, "CAR", spread=+7.0, total=44.5, home=0),
    get_team_features(df, 2025, "CLT", spread=+2.5, total=43.5, home=1),
    get_team_features(df, 2025, "DEN", spread=-2.5, total=43.5, home=0),
    get_team_features(df, 2025, "KAN", spread=+1.5, total=46.5, home=1),
    get_team_features(df, 2025, "PHI", spread=-1.5, total=46.5, home=0),
    get_team_features(df, 2025, "MIN", spread=-3.5, total=44.5, home=1),
    get_team_features(df, 2025, "ATL", spread=+3.5, total=44.5, home=0),
    get_team_features(df, 2025, "HTX", spread=-2.5, total=42.5, home=1),
    get_team_features(df, 2025, "TAM", spread=+2.5, total=42.5, home=0),
    get_team_features(df, 2025, "RAI", spread=+3.5, total=46.5, home=1),
    get_team_features(df, 2025, "SDG", spread=-3.5, total=46.5, home=0)
]

week2_df_dk = pd.DataFrame(week2_games_dk)

if st.button("Run Week 2 Predictions - DraftKings"):
    probs = model.predict_proba(week2_df_dk[features_avg])[:, 1]
    preds = (probs >= 0.5).astype(int)

    results_dk = []
    for i in range (0, len(probs),2): # Step by 2
        game_index = i // 2
        home = week2_teams[game_index]["Home"]
        away = week2_teams[game_index]["Away"]

        prob = probs[i] # Home team
        pred = preds[i] # 1 = Home wins, 0 = Away wins

        winner = home if pred == 1 else away
        results_dk.append({
            "Matchup": f"{away} @ {home}",
            "Home Win Probability": prob,
            "Predicted Winner": winner
        })

    out_dk = pd.DataFrame(results_dk)
    st.dataframe(out_dk.style.format({"Home Win Probability": "{:.2%}"}))
