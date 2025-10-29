#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)

# UI Header / Weekly Records

st.title("Moneyline Model – Logistic Regression (Leak-Safe, Rolling Context)")
st.markdown("**Week 1 Record:** Both models 14–2 ✅")
st.markdown("**Week 2 Record:** Both models 11–5 ✅")
st.markdown("**Week 3 Record:** FanDuel 12–4 ✅")
st.markdown("**Week 3 Record:** DraftKings 11–5 ✅")
st.markdown("**Week 4 Record:** Both models 10–5-1 ✅")
st.markdown("**Week 5 Record:** Both models 7–7 ➖")
st.markdown("**Week 6 Record:** Both models 9–6 ✅")
st.markdown("**Week 7 Record:** FanDuel 6–9 ❌")
st.markdown("**Week 7 Record:** DraftKings 8–7 ✅")
st.markdown("**Week 8 Record:** Both models 11–2 ✅")

# Controls
debug = st.checkbox("Debug mode (print intermediate variables)")

# Data
DATA_DIR = os.path.join(os.getcwd(), "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_ML_week7.csv")

st.write("Looking for file at:", csv_path)
st.write("File exists?", os.path.exists(csv_path))

if not os.path.exists(csv_path):
    st.error(f"Dataset not found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

if debug:
    st.subheader("Raw Head()")
    st.dataframe(df.head())

# Helper
def find_col(candidates):
    """Return the first column name that exists in df from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def time_to_minutes(t):
    if isinstance(t, str) and ":" in t:
        try:
            m, s = map(int, t.split(":"))
            return m + s / 60.0
        except Exception:
            return np.nan
    return np.nan

def expanding_mean_leak_safe(frame, keys, col):
    """
    Shift by 1 game (so we don't use current game stats),
    then take expanding mean within (Season, Team).
    """
    return (
        frame.groupby(keys)[col]
             .apply(lambda x: x.shift().expanding().mean())
             .reset_index(level=list(range(len(keys))), drop=True)
    )

# Feature Engineering

# Binary target
df["Win_Binary"] = df["Win"]

# 3rd down conversion rates
df["Tm_3DConv_Rate"] = df["Tm_3DConv"] / df["Tm_3DAtt"].replace(0, 1)
df["Opp_3DConv_Rate"] = df["Opp_3DConv"] / df["Opp_3DAtt"].replace(0, 1)

# Turnover differential (opponent turnovers forced - team turnovers)
df["Turnover_Diff"] = df["Opp_TO"] - df["Tm_TO"]

# Time of Possession (minutes) + differential
tm_top_col  = find_col(["Tm_ToP", "Tm_ToP", "Tm_ToP"])  # wide fallback list from notebook
opp_top_col = find_col(["Opp_ToP", "Opp_ToP", "Opp_ToP"])
if tm_top_col and opp_top_col:
    df["Tm_ToP_min"]  = df[tm_top_col].apply(time_to_minutes)
    df["Opp_ToP_min"] = df[opp_top_col].apply(time_to_minutes)
    df["ToP_Diff"]    = df["Tm_ToP_min"] - df["Opp_ToP_min"]

# Scoring efficiency & yards-per-point
if "Tm_Ply" in df.columns and "Tm_Pts" in df.columns:
    df["Tm_PtsPerPlay"] = df["Tm_Pts"] / df["Tm_Ply"].replace(0, 1)
if "Opp_Ply" in df.columns and "Opp_Pts" in df.columns:
    df["Opp_PtsPerPlay"] = df["Opp_Pts"] / df["Opp_Ply"].replace(0, 1)

if "Tm_Tot" in df.columns and "Tm_Pts" in df.columns:
    df["Tm_YdsPerPt"] = df["Tm_Tot"] / df["Tm_Pts"].replace(0, 1)
if "Opp_Tot" in df.columns and "Opp_Pts" in df.columns:
    df["Opp_YdsPerPt"] = df["Opp_Tot"] / df["Opp_Pts"].replace(0, 1)

# Pass rush / protection (sack rate)
if all(c in df.columns for c in ["Tm_Sk", "Opp_pAtt", "Opp_Sk", "Tm_pAtt"]):
    df["Tm_SackRate"]   = df["Tm_Sk"]  / df["Opp_pAtt"].replace(0, 1)
    df["Opp_SackRate"]  = df["Opp_Sk"] / df["Tm_pAtt"].replace(0, 1)
    df["SackRate_Diff"] = df["Tm_SackRate"] - df["Opp_SackRate"]

# Special Teams + field goals
if all(c in df.columns for c in ["Tm_PntYds", "Tm_Pnt"]):
    df["Tm_PuntAvgYds"] = df["Tm_PntYds"] / df["Tm_Pnt"].replace(0, 1)
if all(c in df.columns for c in ["Opp_PntYds", "Opp_Pnt"]):
    df["Opp_PuntAvgYds"] = df["Opp_PntYds"] / df["Opp_Pnt"].replace(0, 1)
if "Tm_PuntAvgYds" in df.columns and "Opp_PuntAvgYds" in df.columns:
    df["Punt_Diff"] = df["Tm_PuntAvgYds"] - df["Opp_PuntAvgYds"]

if all(c in df.columns for c in ["Tm_FGM", "Tm_FGA"]):
    df["Tm_FG_Pct"] = df["Tm_FGM"] / df["Tm_FGA"].replace(0, 1)
if all(c in df.columns for c in ["Opp_FGM", "Opp_FGA"]):
    df["Opp_FG_Pct"] = df["Opp_FGM"] / df["Opp_FGA"].replace(0, 1)

# Penalty differential
if all(c in df.columns for c in ["Opp_PenYds", "Tm_PenYds"]):
    df["PenYds_Diff"] = df["Opp_PenYds"] - df["Tm_PenYds"]

# Momentum features: rolling win % and rolling point differential (3 games)
df["Tm_PtDiff"] = df["Tm_Pts"] - df["Opp_Pts"]
df["Tm_WinRate_Roll3"] = (
    df.groupby("Team")["Win_Binary"].shift().rolling(3).mean()
)
df["Tm_PtDiff_Roll3"] = (
    df.groupby("Team")["Tm_PtDiff"].shift().rolling(3).mean()
)

# Build Stat Cols List
base_stat_cols = [
    "Tm_pY/A", "Tm_rY/A", "Tm_Y/P",
    "Opp_pY/A", "Opp_rY/A", "Opp_Y/P",
    "Tm_TO", "Opp_TO", "Tm_PenYds", "Opp_PenYds",
    "Tm_3DConv_Rate", "Opp_3DConv_Rate",
    "Turnover_Diff",
]

new_candidates = [
    "ToP_Diff",
    "Tm_PtsPerPlay", "Opp_PtsPerPlay",
    "Tm_YdsPerPt", "Opp_YdsPerPt",
    "SackRate_Diff",
    "Punt_Diff",
    "Tm_FG_Pct", "Opp_FG_Pct",
    "PenYds_Diff",
    "Tm_WinRate_Roll3", "Tm_PtDiff_Roll3",
]

# Keep only columns that actually exist in df
stat_cols = [c for c in base_stat_cols if c in df.columns] + \
            [c for c in new_candidates if c in df.columns]

# Leak-Safe Expanding Means
for col in stat_cols:
    df[f"{col}_avg"] = expanding_mean_leak_safe(
        df, keys=["Season", "Team"], col=col
    )

# Backfill early-season NaN using league average of the raw stat
for col in stat_cols:
    league_avg = df[col].mean()
    df[f"{col}_avg"] = df[f"{col}_avg"].fillna(league_avg)

# Final model features
context_feats = ["Spread", "Total", "Home"]
features_avg = context_feats + [f"{c}_avg" for c in stat_cols]

# Drop rows missing anything important
df_clean = df.dropna(subset=features_avg + ["Win_Binary"])

if debug:
    st.subheader("Post-Feature Engineering Shape / Nulls")
    st.write("Dataset shape after feature engineering:", df_clean.shape)
    st.write(
        df_clean[features_avg + ["Win_Binary"]]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

# Train/Eval Model
X = df_clean[features_avg]
y = df_clean["Win_Binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=2000, solver="liblinear")
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test, model.predict(X_test))
test_auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

st.write(f"**Train Accuracy:** {train_acc:.2%}")
st.write(f"**Test Accuracy:** {test_acc:.2%}")
st.write(f"**Test ROC-AUC:** {test_auc:.3f}")

if debug:
    st.subheader("Holdout Classification Report / Confusion Matrix")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
    st.write("Confusion Matrix (Test):")
    st.write(confusion_matrix(y_test, y_pred))

    # Cross-val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    st.subheader("5-Fold CV Performance")
    st.write("Accuracy mean ± std:", f"{cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    st.write("ROC-AUC  mean ± std:", f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # Feature importances
    coefs = np.abs(model.coef_[0])
    feat_imp = (
        pd.DataFrame({"Feature": X.columns, "Importance": coefs})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    st.subheader("Top 20 Features by |coef|")
    st.dataframe(feat_imp.head(20))

# Prediction Helper
def get_team_features(latest_df, season, team, spread, total, home, stat_cols_for_avg):
    """
    Pull most recent row for (season, team). If empty (early season edge case),
    fall back to latest row for that team across all seasons.
    Then build a feature row that matches `features_avg`.
    """
    subset = latest_df[(latest_df["Season"] == season) & (latest_df["Team"] == team)]
    if subset.empty:
        subset = latest_df[latest_df["Team"] == team]

    team_row = subset.iloc[-1]

    f = {"Spread": spread, "Total": total, "Home": home}
    for c in stat_cols_for_avg:
        colname = f"{c}_avg"
        # use the stored avg if we have it for this team_row
        val = team_row.get(colname, np.nan)

        if pd.notna(val):
            f[colname] = val
        else:
            # fallback: league average of the raw stat col
            f[colname] = latest_df[c].mean() if c in latest_df.columns else 0.0

    return f

# Shared team list
week8_teams = [
    {"Home": "Chargers", "Away": "Vikings"},
    {"Home": "Panthers", "Away": "Bills"},
    {"Home": "Eagles", "Away": "Giants"},
    {"Home": "Bengals", "Away": "Jets"},
    {"Home": "Patriots", "Away": "Browns"},
    {"Home": "Ravens", "Away": "Bears"},
    {"Home": "Falcons", "Away": "Dolphins"},
    {"Home": "Texans", "Away": "49ers"},
    {"Home": "Saints", "Away": "Buccaneers"},
    {"Home": "Broncos", "Away": "Cowboys"},
    {"Home": "Colts", "Away": "Titans"},
    {"Home": "Steelers", "Away": "Packers"},
    {"Home": "Chiefs", "Away": "Commanders"},
]

# Week x - FANDUEL

st.markdown("---")
st.subheader("Week 8 Predictions - FanDuel Lines")

week8_games_fd = [
    get_team_features(df, 2025, "SDG", spread=-3.0,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "MIN", spread=+3.0,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CAR", spread=+7.5,  total=47.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "BUF", spread=-7.5,  total=47.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CIN", spread=-5.5,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NYJ", spread=+5.5,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "RAV", spread=-2.5,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CHI", spread=+2.5,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NWE", spread=-7.0,  total=40.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CLE", spread=+7.0,  total=40.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "HTX", spread=-2.5,  total=41.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "SFO", spread=+2.5,  total=41.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "ATL", spread=-6.5,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "MIA", spread=+6.5,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "PHI", spread=-7.5,  total=43.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NYG", spread=+7.5,  total=43.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NOR", spread=+3.5,  total=46.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "TAM", spread=-3.5,  total=46.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CLT", spread=-14.5, total=47.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "OTI", spread=+14.5, total=47.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "DEN", spread=-3.5,  total=51.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "DAL", spread=+3.5,  total=51.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "PIT", spread=+2.5,  total=45.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "GNB", spread=-2.5,  total=45.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "KAN", spread=-11.5, total=47.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "WAS", spread=+11.5, total=47.5, home=0, stat_cols_for_avg=stat_cols),
]

week8_df_fd = pd.DataFrame(week8_games_fd)

if st.button("Run Week 8 Predictions - FanDuel"):
    probs = model.predict_proba(week8_df_fd[features_avg])[:, 1]
    preds = (probs >= 0.5).astype(int)

    results_fd = []
    for i in range(0, len(probs), 2):  # pair home/away
        game_index = i // 2
        home = week8_teams[game_index]["Home"]
        away = week8_teams[game_index]["Away"]

        prob = probs[i]  # Home team row
        pred = preds[i]  # 1 = home wins, 0 = away wins
        winner = home if pred == 1 else away

        results_fd.append({
            "Matchup": f"{away} @ {home}",
            "Home Win Probability": prob,
            "Predicted Winner": winner,
        })

    out_fd = pd.DataFrame(results_fd)
    st.dataframe(out_fd.style.format({"Home Win Probability": "{:.2%}"}))

# Week x - DRAFTKINGS
st.markdown("---")
st.subheader("Week 8 Predictions - DraftKings Lines")

week8_games_dk = [
    get_team_features(df, 2025, "SDG", spread=-3.0,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "MIN", spread=+3.0,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CAR", spread=+7.0,  total=47.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "BUF", spread=-7.0,  total=47.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CIN", spread=-5.5,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NYJ", spread=+5.5,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "RAV", spread=-2.5,  total=45.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CHI", spread=+2.5,  total=45.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NWE", spread=-7.0,  total=40.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CLE", spread=+7.0,  total=40.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "HTX", spread=-2.5,  total=41.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "SFO", spread=+2.5,  total=41.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "ATL", spread=-7.0,  total=44.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "MIA", spread=+7.0,  total=44.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "PHI", spread=-7.0,  total=43.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NYG", spread=+7.0,  total=43.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "NOR", spread=+4.5,  total=46.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "TAM", spread=-4.5,  total=46.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "CLT", spread=-14.5, total=47.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "OTI", spread=+14.5, total=47.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "DEN", spread=-3.5,  total=51.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "DAL", spread=+3.5,  total=51.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "PIT", spread=+3.0,  total=45.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "GNB", spread=-3.0,  total=45.5, home=0, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "KAN", spread=-11.5, total=47.5, home=1, stat_cols_for_avg=stat_cols),
    get_team_features(df, 2025, "WAS", spread=+11.5, total=47.5, home=0, stat_cols_for_avg=stat_cols),
]

week8_df_dk = pd.DataFrame(week8_games_dk)

if st.button("Run Week 8 Predictions - DraftKings"):
    probs = model.predict_proba(week8_df_dk[features_avg])[:, 1]
    preds = (probs >= 0.5).astype(int)

    results_dk = []
    for i in range(0, len(probs), 2):  # pair home/away
        game_index = i // 2
        home = week8_teams[game_index]["Home"]
        away = week8_teams[game_index]["Away"]

        prob = probs[i]  # Home team row
        pred = preds[i]  # 1 = home wins, 0 = away wins
        winner = home if pred == 1 else away

        results_dk.append({
            "Matchup": f"{away} @ {home}",
            "Home Win Probability": prob,
            "Predicted Winner": winner,
        })

    out_dk = pd.DataFrame(results_dk)
    st.dataframe(out_dk.style.format({"Home Win Probability": "{:.2%}"}))
