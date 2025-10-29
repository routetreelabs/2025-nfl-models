import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Points Predictor – Linear Regression")

# Controls
debug = st.checkbox("Debug mode (print intermediate variables)")

# Data
DATA_DIR = os.path.join(os.getcwd(), "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_Points_week7_copy.csv")


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

# Base rolling scoring metrics
df["Tm_Pts_Last1"] = df.groupby("Team")["Tm_Pts"].shift(1)
df["Tm_Pts_Roll3"] = rolling_stat(df, "Team", "Tm_Pts", 3)
df["Tm_Pts_Roll5"] = rolling_stat(df, "Team", "Tm_Pts", 5)
df[["Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"]] = df[
    ["Tm_Pts_Last1", "Tm_Pts_Roll3", "Tm_Pts_Roll5"]
].fillna(0)

# Opponent defensive rolling context
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

# Time of possession and pace
df["Tm_ToP_min"] = df["Tm_ToP"].apply(time_to_minutes)
df["Opp_ToP_min"] = df["Opp_ToP"].apply(time_to_minutes)
df["ToP_Diff"] = df["Tm_ToP_min"] - df["Opp_ToP_min"]
df["Tm_Pace"] = df["Tm_Ply"] / df["Tm_ToP_min"].replace(0, np.nan)
df["Opp_Pace"] = df["Opp_Ply"] / df["Opp_ToP_min"].replace(0, np.nan)

for col in ["ToP_Diff", "Tm_Pace", "Opp_Pace"]:
    df[f"{col}_Roll3"] = rolling_stat(df, "Team", col, 3)
    df[f"{col}_Roll3"].fillna(df[col].mean(), inplace=True)

# Home / away scoring trends
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

# Final features/target
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

# Preprocessor with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

X_processed = preprocessor.fit_transform(X)

# Model training/evaluation
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

if debug:
    st.subheader("Feature Engineering Preview")
    st.dataframe(df.head())
    st.subheader("Model Coefficients (Top 20)")
    encoded_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_columns = np.concatenate([encoded_columns, numerical_features])
    coef_df = (
        pd.DataFrame({"Feature": all_columns, "Coefficient": model.coef_})
        .sort_values("Coefficient", ascending=False)
    )
    st.dataframe(coef_df.head(20))

# Prediction helpers
def predict_team_points(season, week, home, team, opponent, spread, total, last1, roll3, roll5):
    """Predict single team points using engineered features"""
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
        team_pred = predict_team_points(
            season=g["Season"], week=g["Week"], home=g["Home"],
            team=g["Team"], opponent=g["Opp"], spread=g["Spread"],
            total=g["Total"], last1=g["Last1"], roll3=g["Roll3"], roll5=g["Roll5"]
        )
        opp_pred = predict_team_points(
            season=g["Season"], week=g["Week"], home=1-g["Home"],
            team=g["Opp"], opponent=g["Team"], spread=-g["Spread"], total=g["Total"],
            last1=g.get("Opp_Last1", 0), roll3=g.get("Opp_Roll3", 0), roll5=g.get("Opp_Roll5", 0)
        )

        team_results.append({
            "Team": g["Team"], "Pred_Pts": round(team_pred, 2),
            "Opp": g["Opp"], "Home": g["Home"], "Spread": g["Spread"]
        })
        team_results.append({
            "Team": g["Opp"], "Pred_Pts": round(opp_pred, 2),
            "Opp": g["Team"], "Home": 1-g["Home"], "Spread": -g["Spread"]
        })
        game_results.append({
            "Matchup": f"{g['Team']} vs {g['Opp']}",
            "Pred_Total": round(team_pred + opp_pred, 2),
            "Vegas_Total": g["Total"],
            "Diff": round(team_pred + opp_pred - g["Total"], 2)
        })
    return pd.DataFrame(team_results), pd.DataFrame(game_results)

# FanDuel Predictions
st.markdown("---")
st.subheader("Week 8 Predictions – FanDuel Lines")
if st.button("Run Week 8 Predictions – FanDuel"):
    team_preds_fd, game_preds_fd = predict_matchups(week8_games_fd)
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_fd.style.format({
        "Spread": "{:.1f}", "Total": "{:.1f}", "Roll3": "{:.2f}", "Roll5": "{:.2f}", "Pred_Points": "{:.2f}"
    }))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_fd.style.format({
        "Spread": "{:.1f}", "Vegas_Total": "{:.1f}", "Home_Pred": "{:.2f}", "Away_Pred": "{:.2f}", "Predicted_Total": "{:.2f}"
    }))

# DraftKings Predictions
st.markdown("---")
st.subheader("Week 8 Predictions – DraftKings Lines")
if st.button("Run Week 8 Predictions – DraftKings"):
    team_preds_dk, game_preds_dk = predict_matchups(week8_games_dk)
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_dk.style.format({
        "Spread": "{:.1f}", "Total": "{:.1f}", "Roll3": "{:.2f}", "Roll5": "{:.2f}", "Pred_Points": "{:.2f}"
    }))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_dk.style.format({
        "Spread": "{:.1f}", "Vegas_Total": "{:.1f}", "Home_Pred": "{:.2f}", "Away_Pred": "{:.2f}", "Predicted_Total": "{:.2f}"
    }))
