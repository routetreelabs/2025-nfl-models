import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Header
st.title("NFL Points Predictor – Linear Regression")

debug = st.checkbox("Debug mode (print intermediate variables)")

# Data
DATA_DIR = os.path.join(os.getcwd(), "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_Points_week7_copy.csv")

if not os.path.exists(csv_path):
    st.error(f"Dataset not found: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)
df = df.sort_values(by=["Team", "Season", "Week"])
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Helper Functions
def time_to_minutes(t):
    """Convert time of possession 'MM:SS' to float minutes"""
    if isinstance(t, str) and ":" in t:
        try:
            m, s = map(int, t.split(":"))
            return m + s / 60
        except Exception:
            return np.nan
    return np.nan

def rolling_stat(df, group_col, target_col, window):
    """Leakage-free rolling mean (shifted by 1 game)"""
    return (
        df.groupby(group_col)[target_col]
          .shift(1)
          .rolling(window)
          .mean()
          .reset_index(level=0, drop=True)
    )

# Feature Engineering

# Base scoring rolls
df["Tm_Pts_Last1"] = df.groupby("Team")["Tm_Pts"].shift(1)
df["Tm_Pts_Roll3"] = rolling_stat(df, "Team", "Tm_Pts", 3)
df["Tm_Pts_Roll5"] = rolling_stat(df, "Team", "Tm_Pts", 5)
df[["Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"]] = df[
    ["Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"]
].fillna(0)

# Opponent defensive context
df["Opp_Pts_Allowed_Last1"] = df.groupby("Team")["Opp_Pts"].shift(1)
df["Opp_Pts_Allowed_Roll3"] = rolling_stat(df, "Team", "Opp_Pts", 3)
df["Opp_Pts_Allowed_Roll5"] = rolling_stat(df, "Team", "Opp_Pts", 5)
df[["Opp_Pts_Allowed_Last1", "Opp_Pts_Allowed_Roll3", "Opp_Pts_Allowed_Roll5"]] = df[
    ["Opp_Pts_Allowed_Last1", "Opp_Pts_Allowed_Roll3", "Opp_Pts_Allowed_Roll5"]
].fillna(0)

# Efficiency metrics
df["Tm_YdsPerPlay"] = df["Tm_Tot"] / df["Tm_Ply"].replace(0, 1)
df["Opp_YdsPerPlay"] = df["Opp_Tot"] / df["Opp_Ply"].replace(0, 1)
df["Tm_TO_Rate"] = df["Tm_TO"] / df["Tm_Ply"].replace(0, 1)
df["Opp_TO_Rate"] = df["Opp_TO"] / df["Opp_Ply"].replace(0, 1)
df["Tm_3DConv_Rate"] = df["Tm_3DConv"] / df["Tm_3DAtt"].replace(0, 1)
df["Opp_3DConv_Rate"] = df["Opp_3DConv"] / df["Opp_3DAtt"].replace(0, 1)

for col in ["Tm_YdsPerPlay","Opp_YdsPerPlay","Tm_TO_Rate","Opp_TO_Rate","Tm_3DConv_Rate","Opp_3DConv_Rate"]:
    df[f"{col}_Roll3"] = rolling_stat(df, "Team", col, 3)
    df[f"{col}_Roll3"].fillna(df[col].mean(), inplace=True)

# Time of possession & pace
df["Tm_ToP_min"] = df["Tm_ToP"].apply(time_to_minutes)
df["Opp_ToP_min"] = df["Opp_ToP"].apply(time_to_minutes)
df["ToP_Diff"] = df["Tm_ToP_min"] - df["Opp_ToP_min"]
df["Tm_Pace"] = df["Tm_Ply"] / df["Tm_ToP_min"].replace(0, np.nan)
df["Opp_Pace"] = df["Opp_Ply"] / df["Opp_ToP_min"].replace(0, np.nan)

for col in ["ToP_Diff", "Tm_Pace", "Opp_Pace"]:
    df[f"{col}_Roll3"] = rolling_stat(df, "Team", col, 3)
    df[f"{col}_Roll3"].fillna(df[col].mean(), inplace=True)

# Home/Away scoring trends
df["Tm_Home_Roll3"] = (
    df[df["Home"] == 1]
    .groupby("Team")["Tm_Pts"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)
df["Tm_Away_Roll3"] = (
    df[df["Home"] == 0]
    .groupby("Team")["Tm_Pts"]
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)
df["Tm_Home_Roll3"].fillna(0, inplace=True)
df["Tm_Away_Roll3"].fillna(0, inplace=True)

# Interaction terms
df["Spread_x_Roll3"] = df["Spread"] * df["Tm_Pts_Roll3"]
df["Total_x_Roll3"] = df["Total"] * df["Tm_Pts_Roll3"]

# Model Training
features = [
    "Season", "Week", "Home", "Team", "Opp",
    "Spread", "Total",
    "Tm_Pts_Last1","Tm_Pts_Roll3","Tm_Pts_Roll5",
    "Opp_Pts_Allowed_Last1","Opp_Pts_Allowed_Roll3","Opp_Pts_Allowed_Roll5",
    "Tm_YdsPerPlay_Roll3","Opp_YdsPerPlay_Roll3",
    "Tm_3DConv_Rate_Roll3","Opp_3DConv_Rate_Roll3",
    "ToP_Diff_Roll3","Tm_Pace_Roll3","Opp_Pace_Roll3",
    "Tm_Home_Roll3","Tm_Away_Roll3",
    "Spread_x_Roll3","Total_x_Roll3"
]
target = "Tm_Pts"

X = df[features]
y = df[target]

categorical_features = ["Team", "Opp"]
numerical_features = [c for c in features if c not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"**R²:** {r2:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# Prediction Functions
def predict_team_points(season, week, home, team, opponent, spread, total, last1, roll3, roll5):
    input_df = pd.DataFrame([{
        "Season": season, "Week": week, "Home": home,
        "Team": team, "Opp": opponent, "Spread": spread, "Total": total,
        "Tm_Pts_Last1": last1, "Tm_Pts_Roll3": roll3, "Tm_Pts_Roll5": roll5,
        "Opp_Pts_Allowed_Last1": 0, "Opp_Pts_Allowed_Roll3": 0, "Opp_Pts_Allowed_Roll5": 0,
        "Tm_YdsPerPlay_Roll3": 0, "Opp_YdsPerPlay_Roll3": 0,
        "Tm_3DConv_Rate_Roll3": 0, "Opp_3DConv_Rate_Roll3": 0,
        "ToP_Diff_Roll3": 0, "Tm_Pace_Roll3": 0, "Opp_Pace_Roll3": 0,
        "Tm_Home_Roll3": 0, "Tm_Away_Roll3": 0,
        "Spread_x_Roll3": spread * roll3, "Total_x_Roll3": total * roll3
    }])
    input_processed = preprocessor.transform(input_df)
    return model.predict(input_processed)[0]

def predict_matchups(matchups):
    team_results, game_results = [], []
    for g in matchups:
        team_pred = predict_team_points(**{
            "season": g["Season"], "week": g["Week"], "home": g["Home"],
            "team": g["Team"], "opponent": g["Opp"], "spread": g["Spread"],
            "total": g["Total"], "last1": g["Last1"], "roll3": g["Roll3"], "roll5": g["Roll5"]
        })
        opp_pred = predict_team_points(**{
            "season": g["Season"], "week": g["Week"], "home": 1-g["Home"],
            "team": g["Opp"], "opponent": g["Team"], "spread": -g["Spread"], "total": g["Total"],
            "last1": g.get("Opp_Last1", 0), "roll3": g.get("Opp_Roll3", 0), "roll5": g.get("Opp_Roll5", 0)
        })
        team_results += [
            {"Team": g["Team"], "Pred_Pts": round(team_pred, 2), "Opp": g["Opp"], "Home": g["Home"], "Spread": g["Spread"]},
            {"Team": g["Opp"], "Pred_Pts": round(opp_pred, 2), "Opp": g["Team"], "Home": 1-g["Home"], "Spread": -g["Spread"]}
        ]
        game_results.append({
            "Matchup": f"{g['Team']} vs {g['Opp']}",
            "Pred_Total": round(team_pred + opp_pred, 2),
            "Vegas_Total": g["Total"],
            "Diff": round(team_pred + opp_pred - g["Total"], 2)
        })
    return pd.DataFrame(team_results), pd.DataFrame(game_results)

# FanDuel
week8_games_fd = [
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "SDG", "Opp": "MIN", "Spread": -3.0, "Total": 44.5, "Last1": 24, "Roll3": 21.0, "Roll5": 20.8, "Opp_Last1": 22, "Opp_Roll3": 21.3, "Opp_Roll5": 23.6},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "CAR", "Opp": "BUF", "Spread": +7.5, "Total": 47.5, "Last1": 13, "Roll3": 23.3, "Roll5": 22.6, "Opp_Last1": 14, "Opp_Roll3": 21.6, "Opp_Roll5": 25.2},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "PHI", "Opp": "NYG", "Spread": -7.5, "Total": 43.5, "Last1": 28, "Roll3": 20.6, "Roll5": 25.2, "Opp_Last1": 32, "Opp_Roll3": 26.6, "Opp_Roll5": 22.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "CIN", "Opp": "NYJ", "Spread": -5.5, "Total": 44.5, "Last1": 33, "Roll3": 25.0, "Roll5": 17.6, "Opp_Last1": 6, "Opp_Roll3": 13.0, "Opp_Roll5": 17.4},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "NWE", "Opp": "CLE", "Spread": -7.0, "Total": 40.5, "Last1": 31, "Roll3": 26.3, "Roll5": 27.0, "Opp_Last1": 31, "Opp_Roll3": 19.0, "Opp_Roll5": 16.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "RAV", "Opp": "CHI", "Spread": -2.5, "Total": 44.5, "Last1": 3, "Roll3": 11.0, "Roll5": 20.8, "Opp_Last1": 26, "Opp_Roll3": 25.3, "Opp_Roll5": 25.6},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "ATL", "Opp": "MIA", "Spread": -6.5, "Total": 44.5, "Last1": 10, "Roll3": 22.6, "Roll5": 18.0, "Opp_Last1": 6, "Opp_Roll3": 19.0, "Opp_Roll5": 21.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "HTX", "Opp": "SFO", "Spread": -2.5, "Total": 41.5, "Last1": 19, "Roll3": 29.6, "Roll5": 23.6, "Opp_Last1": 20, "Opp_Roll3": 21.6, "Opp_Roll5": 20.4},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "NOR", "Opp": "TAM", "Spread": +4.5, "Total": 46.5, "Last1": 14, "Roll3": 19.6, "Roll5": 18.2, "Opp_Last1": 9, "Opp_Roll3": 25.6, "Opp_Roll5": 26.2},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "DEN", "Opp": "DAL", "Spread": -3.5, "Total": 51.5, "Last1": 33, "Roll3": 22.3, "Roll5": 23.0, "Opp_Last1": 44, "Opp_Roll3": 36.0, "Opp_Roll5": 32.4},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "CLT", "Opp": "OTI", "Spread": -14.5, "Total": 47.5, "Last1": 38, "Roll3": 36.3, "Roll5": 34.0, "Opp_Last1": 13, "Opp_Roll3": 15.0, "Opp_Roll5": 13.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "PIT", "Opp": "GNB", "Spread": +2.5, "Total": 45.5, "Last1": 31, "Roll3": 26.0, "Roll5": 23.2, "Opp_Last1": 27, "Opp_Roll3": 31.3, "Opp_Roll5": 26.2},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "KAN", "Opp": "WAS", "Spread": -10.5, "Total": 47.5, "Last1": 31, "Roll3": 29.6, "Roll5": 29.6, "Opp_Last1": 22, "Opp_Roll3": 24.3, "Opp_Roll5": 28.2}
]

# DraftKings
week8_games_dk = [
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "SDG", "Opp": "MIN", "Spread": -3.0, "Total": 44.5, "Last1": 24, "Roll3": 21.0, "Roll5": 20.8, "Opp_Last1": 22, "Opp_Roll3": 21.3, "Opp_Roll5": 23.6},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "CAR", "Opp": "BUF", "Spread": +7.0, "Total": 47.5, "Last1": 13, "Roll3": 23.3, "Roll5": 22.6, "Opp_Last1": 14, "Opp_Roll3": 21.6, "Opp_Roll5": 25.2},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "PHI", "Opp": "NYG", "Spread": -7.5, "Total": 43.5, "Last1": 28, "Roll3": 20.6, "Roll5": 25.2, "Opp_Last1": 32, "Opp_Roll3": 26.6, "Opp_Roll5": 22.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "CIN", "Opp": "NYJ", "Spread": -5.5, "Total": 44.5, "Last1": 33, "Roll3": 25.0, "Roll5": 17.6, "Opp_Last1": 6, "Opp_Roll3": 13.0, "Opp_Roll5": 17.4},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "NWE", "Opp": "CLE", "Spread": -7.0, "Total": 40.5, "Last1": 31, "Roll3": 26.3, "Roll5": 27.0, "Opp_Last1": 31, "Opp_Roll3": 19.0, "Opp_Roll5": 16.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "RAV", "Opp": "CHI", "Spread": -2.5, "Total": 45.5, "Last1": 3, "Roll3": 11.0, "Roll5": 20.8, "Opp_Last1": 26, "Opp_Roll3": 25.3, "Opp_Roll5": 25.6},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "ATL", "Opp": "MIA", "Spread": -7.0, "Total": 44.5, "Last1": 10, "Roll3": 22.6, "Roll5": 18.0, "Opp_Last1": 6, "Opp_Roll3": 19.0, "Opp_Roll5": 21.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "HTX", "Opp": "SFO", "Spread": -2.5, "Total": 41.5, "Last1": 19, "Roll3": 29.6, "Roll5": 23.6, "Opp_Last1": 20, "Opp_Roll3": 21.6, "Opp_Roll5": 20.4},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "NOR", "Opp": "TAM", "Spread": +4.5, "Total": 46.5, "Last1": 14, "Roll3": 19.6, "Roll5": 18.2, "Opp_Last1": 9, "Opp_Roll3": 25.6, "Opp_Roll5": 26.2},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "DEN", "Opp": "DAL", "Spread": -3.5, "Total": 51.5, "Last1": 33, "Roll3": 22.3, "Roll5": 23.0, "Opp_Last1": 44, "Opp_Roll3": 36.0, "Opp_Roll5": 32.4},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "CLT", "Opp": "OTI", "Spread": -14.5, "Total": 47.5, "Last1": 38, "Roll3": 36.3, "Roll5": 34.0, "Opp_Last1": 13, "Opp_Roll3": 15.0, "Opp_Roll5": 13.0},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "PIT", "Opp": "GNB", "Spread": +3.0, "Total": 45.5, "Last1": 31, "Roll3": 26.0, "Roll5": 23.2, "Opp_Last1": 27, "Opp_Roll3": 31.3, "Opp_Roll5": 26.2},
    {"Season": 2025, "Week": 8, "Home": 1, "Team": "KAN", "Opp": "WAS", "Spread": -10.5, "Total": 48.5, "Last1": 31, "Roll3": 29.6, "Roll5": 29.6, "Opp_Last1": 22, "Opp_Roll3": 24.3, "Opp_Roll5": 28.2}
]

# Streamlit Display
st.markdown("---")
st.subheader("Week 8 Predictions – FanDuel Lines")
if st.button("Run Week 8 Predictions – FanDuel"):
    team_preds_fd, game_preds_fd = predict_matchups(week8_games_fd)
    team_preds_fd["Home"] = team_preds_fd["Home"].apply(lambda x: "Yes" if x == 1 else "No")
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_fd.style.format({"Spread": "{:.1f}", "Pred_Pts": "{:.2f}"}))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_fd.style.format({"Vegas_Total": "{:.1f}", "Pred_Total": "{:.2f}", "Diff": "{:.2f}"}))

st.markdown("---")
st.subheader("Week 8 Predictions – DraftKings Lines")
if st.button("Run Week 8 Predictions – DraftKings"):
    team_preds_dk, game_preds_dk = predict_matchups(week8_games_dk)
    team_preds_dk["Home"] = team_preds_dk["Home"].apply(lambda x: "Yes" if x == 1 else "No")
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_dk.style.format({"Spread": "{:.1f}", "Pred_Pts": "{:.2f}"}))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_dk.style.format({"Vegas_Total": "{:.1f}", "Pred_Total": "{:.2f}", "Diff": "{:.2f}"}))
