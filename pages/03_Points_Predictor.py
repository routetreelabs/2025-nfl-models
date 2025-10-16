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
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_Points_week6_copy.csv")

# Load and Prepare the Data
df = pd.read_csv(csv_path)
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Sort so rolling stats are correct
df = df.sort_values(by=["Team", "Season", "Week"])

# Rolling features
df["Tm_Pts_Last1"] = df.groupby("Team")["Tm_Pts"].shift(1)
df["Tm_Pts_Rolling3"] = df.groupby("Team")["Tm_Pts"].shift(1).rolling(3).mean().reset_index(0, drop=True)
df["Tm_Pts_Rolling5"] = df.groupby("Team")["Tm_Pts"].shift(1).rolling(5).mean().reset_index(0, drop=True)

# Keep your existing NA strategy (fill with 0s)
df[["Tm_Pts_Last1", "Tm_Pts_Rolling3", "Tm_Pts_Rolling5"]] = df[
    ["Tm_Pts_Last1", "Tm_Pts_Rolling3", "Tm_Pts_Rolling5"]
].fillna(0)

# Features & target (add Roll5 ONLY)
features_early = [
    "Season", "Week", "Home", "Team", "Opp",
    "Spread", "Total", "Tm_Pts_Last1", "Tm_Pts_Rolling3", "Tm_Pts_Rolling5"
]
target_early = "Tm_Pts"
X_early = df[features_early]
y_early = df[target_early]

# Preprocessing (add Roll5 to numerical list ONLY)
categorical_early = ["Team", "Opp"]
numerical_early = [
    "Season", "Week", "Home", "Spread", "Total",
    "Tm_Pts_Last1", "Tm_Pts_Rolling3", "Tm_Pts_Rolling5"
]

preprocessor_early = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_early)],
    remainder="passthrough"
)

X_processed_early = preprocessor_early.fit_transform(X_early)

# Train/test split (unchanged)
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_processed_early, y_early, test_size=0.2, random_state=42
)

# Train model (unchanged)
model_early = LinearRegression()
model_early.fit(X_train_early, y_train_early)

# Evaluate (unchanged)
y_pred_early = model_early.predict(X_test_early)
mae_early = mean_absolute_error(y_test_early, y_pred_early)
rmse_early = np.sqrt(mean_squared_error(y_test_early, y_pred_early))
r2_early = r2_score(y_test_early, y_pred_early)

st.write(f"[Early Model] **R²:** {r2_early:.2f}")
st.write(f"[Early Model] **MAE:** {mae_early:.2f}")
st.write(f"[Early Model] **RMSE:** {rmse_early:.2f}")

if debug:
    st.write("Sample of training data after preprocessing:")
    st.write(pd.DataFrame(X_early).head())

# Prediction Functions (ONLY change: add roll5 through)
def predict_team_points(season, week, home, team, opponent, spread, total, last1, roll3, roll5):
    input_df = pd.DataFrame([{
        "Season": season, "Week": week, "Home": home,
        "Team": team, "Opp": opponent, "Spread": spread, "Total": total,
        "Tm_Pts_Last1": last1, "Tm_Pts_Rolling3": roll3, "Tm_Pts_Rolling5": roll5
    }])
    input_processed = preprocessor_early.transform(input_df)
    prediction = model_early.predict(input_processed)[0]
    return prediction

def predict_matchups(matchups):
    team_results, game_results = [], []
    for game in matchups:
        team_pred = predict_team_points(
            season=game["Season"], week=game["Week"], home=game["Home"],
            team=game["Team"], opponent=game["Opp"], spread=game["Spread"], total=game["Total"],
            last1=game["Last1"], roll3=game["Roll3"], roll5=game["Roll5"]
        )
        team_results.append({
            "Season": game["Season"], "Week": game["Week"], "Team": game["Team"], "Opponent": game["Opp"],
            "Home": game["Home"], "Spread": game["Spread"], "Total": game["Total"],
            "Last1": game["Last1"], "Roll3": game["Roll3"], "Roll5": game["Roll5"],
            "Predicted_Points": round(team_pred, 2)
        })

        opp_pred = predict_team_points(
            season=game["Season"], week=game["Week"], home=1-game["Home"],
            team=game["Opp"], opponent=game["Team"], spread=-game["Spread"], total=game["Total"],
            last1=game.get("Opp_Last1", 0), roll3=game.get("Opp_Roll3", 0), roll5=game.get("Opp_Roll5", 0)
        )
        team_results.append({
            "Season": game["Season"], "Week": game["Week"], "Team": game["Opp"], "Opponent": game["Team"],
            "Home": 1-game["Home"], "Spread": -game["Spread"], "Total": game["Total"],
            "Last1": game.get("Opp_Last1", 0), "Roll3": game.get("Opp_Roll3", 0), "Roll5": game.get("Opp_Roll5", 0),
            "Predicted_Points": round(opp_pred, 2)
        })

        game_results.append({
            "Season": game["Season"], "Week": game["Week"], "Matchup": f"{game['Team']} vs {game['Opp']}",
            "Home_Team": game["Team"] if game["Home"]==1 else game["Opp"],
            "Away_Team": game["Opp"] if game["Home"]==1 else game["Team"],
            "Home_Pred": round(team_pred, 2) if game["Home"]==1 else round(opp_pred, 2),
            "Away_Pred": round(opp_pred, 2) if game["Home"]==1 else round(team_pred, 2),
            "Predicted_Total": round(team_pred + opp_pred, 2),
            "Vegas_Total": game["Total"], "Spread": game["Spread"]
        })
    return pd.DataFrame(team_results), pd.DataFrame(game_results)

# FanDuel
week7_games_fd = [
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CIN", "Opp": "PIT", "Spread": +5.5, "Total": 44.5, "Last1": 18, "Roll3": 15.0, "Roll5": 17.2, "Opp_Last1": 23, "Opp_Roll3": 22.6, "Opp_Roll5": 23.8},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "JAX", "Opp": "RAM", "Spread": +3.0, "Total": 44.5, "Last1": 12, "Roll3": 23.0, "Roll5": 22.6, "Opp_Last1": 17, "Opp_Roll3": 22.3, "Opp_Roll5": 25.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "MIN", "Opp": "PHI", "Spread": +2.5, "Total": 43.5, "Last1": 21, "Roll3": 30.0, "Roll5": 24.6, "Opp_Last1": 17, "Opp_Roll3": 21.6, "Opp_Roll5": 23.6},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "OTI", "Opp": "NWE", "Spread": +7.0, "Total": 41.5, "Last1": 10, "Roll3": 10.6, "Roll5": 14.2, "Opp_Last1": 25, "Opp_Roll3": 30.0, "Opp_Roll5": 27.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "KAN", "Opp": "RAI", "Spread": -11.5, "Total": 45.5, "Last1": 30, "Roll3": 31.6, "Roll5": 26.8, "Opp_Last1": 20, "Opp_Roll3": 16.6, "Opp_Roll5": 16.6},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "NYJ", "Opp": "CAR", "Spread": +1.5, "Total": 42.5, "Last1": 11, "Roll3": 18.0, "Roll5": 18.2, "Opp_Last1": 30, "Opp_Roll3": 23.3, "Opp_Roll5": 24.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CHI", "Opp": "NOR", "Spread": -5.5, "Total": 46.5, "Last1": 25, "Roll3": 27.0, "Roll5": 25.2, "Opp_Last1": 19, "Opp_Roll3": 21.3, "Opp_Roll5": 19.6},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CLE", "Opp": "MIA", "Spread": -2.5, "Total": 39.5, "Last1": 9, "Roll3": 12.0, "Roll5": 13.2, "Opp_Last1": 27, "Opp_Roll3": 26.0, "Opp_Roll5": 25.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "DEN", "Opp": "NYG", "Spread": -7.0, "Total": 40.5, "Last1": 13, "Roll3": 20.6, "Roll5": 22.0, "Opp_Last1": 34, "Opp_Roll3": 23.0, "Opp_Roll5": 23.0},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "SDG", "Opp": "CLT", "Spread": -1.5, "Total": 48.5, "Last1": 29, "Roll3": 19.0, "Roll5": 20.0, "Opp_Last1": 31, "Opp_Roll3": 30.3, "Opp_Roll5": 32.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CRD", "Opp": "GNB", "Spread": +6.5, "Total": 44.5, "Last1": 27, "Roll3": 22.6, "Roll5": 22.0, "Opp_Last1": 27, "Opp_Roll3": 25.6, "Opp_Roll5": 26.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "DAL", "Opp": "WAS", "Spread": +1.5, "Total": 54.5, "Last1": 27, "Roll3": 34.6, "Roll5": 31.6, "Opp_Last1": 24, "Opp_Roll3": 26.0, "Opp_Roll5": 27.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "SFO", "Opp": "ATL", "Spread": -1.5, "Total": 46.5, "Last1": 19, "Roll3": 22.0, "Roll5": 21.6, "Opp_Last1": 24, "Opp_Roll3": 19.3, "Opp_Roll5": 20.0},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "DET", "Opp": "TAM", "Spread": -5.5, "Total": 52.5, "Last1": 17, "Roll3": 29.3, "Roll5": 35.6, "Opp_Last1": 30, "Opp_Roll3": 31.0, "Opp_Roll5": 28.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "SEA", "Opp": "HTX", "Spread": -3.5, "Total": 40.5, "Last1": 20, "Roll3": 23.3, "Roll5": 26.8, "Opp_Last1": 44, "Opp_Roll3": 26.6, "Opp_Roll5": 21.6}
]

# DraftKings
week7_games_dk = [
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CIN", "Opp": "PIT", "Spread": +5.5, "Total": 44.5, "Last1": 18, "Roll3": 15.0, "Roll5": 17.2, "Opp_Last1": 23, "Opp_Roll3": 22.6, "Opp_Roll5": 23.8},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "JAX", "Opp": "RAM", "Spread": +3.0, "Total": 44.5, "Last1": 12, "Roll3": 23.0, "Roll5": 22.6, "Opp_Last1": 17, "Opp_Roll3": 22.3, "Opp_Roll5": 25.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "MIN", "Opp": "PHI", "Spread": +1.5, "Total": 43.5, "Last1": 21, "Roll3": 30.0, "Roll5": 24.6, "Opp_Last1": 17, "Opp_Roll3": 21.6, "Opp_Roll5": 23.6},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "OTI", "Opp": "NWE", "Spread": +7.0, "Total": 42.5, "Last1": 10, "Roll3": 10.6, "Roll5": 14.2, "Opp_Last1": 25, "Opp_Roll3": 30.0, "Opp_Roll5": 27.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "KAN", "Opp": "RAI", "Spread": -11.5, "Total": 45.5, "Last1": 30, "Roll3": 31.6, "Roll5": 26.8, "Opp_Last1": 20, "Opp_Roll3": 16.6, "Opp_Roll5": 16.6},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "NYJ", "Opp": "CAR", "Spread": +1.5, "Total": 42.5, "Last1": 11, "Roll3": 18.0, "Roll5": 18.2, "Opp_Last1": 30, "Opp_Roll3": 23.3, "Opp_Roll5": 24.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CHI", "Opp": "NOR", "Spread": -4.5, "Total": 46.5, "Last1": 25, "Roll3": 27.0, "Roll5": 25.2, "Opp_Last1": 19, "Opp_Roll3": 21.3, "Opp_Roll5": 19.6},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CLE", "Opp": "MIA", "Spread": -2.5, "Total": 40.5, "Last1": 9, "Roll3": 12.0, "Roll5": 13.2, "Opp_Last1": 27, "Opp_Roll3": 26.0, "Opp_Roll5": 25.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "DEN", "Opp": "NYG", "Spread": -7.0, "Total": 39.5, "Last1": 13, "Roll3": 20.6, "Roll5": 22.0, "Opp_Last1": 34, "Opp_Roll3": 23.0, "Opp_Roll5": 23.0},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "SDG", "Opp": "CLT", "Spread": -1.5, "Total": 48.5, "Last1": 29, "Roll3": 19.0, "Roll5": 20.0, "Opp_Last1": 31, "Opp_Roll3": 30.3, "Opp_Roll5": 32.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "CRD", "Opp": "GNB", "Spread": +6.5, "Total": 45.5, "Last1": 27, "Roll3": 22.6, "Roll5": 22.0, "Opp_Last1": 27, "Opp_Roll3": 25.6, "Opp_Roll5": 26.2},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "DAL", "Opp": "WAS", "Spread": +2.5, "Total": 54.5, "Last1": 27, "Roll3": 34.6, "Roll5": 31.6, "Opp_Last1": 24, "Opp_Roll3": 26.0, "Opp_Roll5": 27.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "SFO", "Opp": "ATL", "Spread": -2.5, "Total": 47.5, "Last1": 19, "Roll3": 22.0, "Roll5": 21.6, "Opp_Last1": 24, "Opp_Roll3": 19.3, "Opp_Roll5": 20.0},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "DET", "Opp": "TAM", "Spread": -5.5, "Total": 52.5, "Last1": 17, "Roll3": 29.3, "Roll5": 35.6, "Opp_Last1": 30, "Opp_Roll3": 31.0, "Opp_Roll5": 28.4},
    {"Season": 2025, "Week": 7, "Home": 1, "Team": "SEA", "Opp": "HTX", "Spread": -3.0, "Total": 41.5, "Last1": 20, "Roll3": 23.3, "Roll5": 26.8, "Opp_Last1": 44, "Opp_Roll3": 26.6, "Opp_Roll5": 21.6}
]

# FanDuel Predictions
st.markdown("---")
st.subheader("Week 7 Predictions – FanDuel Lines")
if st.button("Run Week 7 Predictions – FanDuel"):
    team_preds_fd, game_preds_fd = predict_matchups(week7_games_fd)
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_fd.style.format({
        "Spread": "{:.1f}",
        "Total": "{:.1f}",
        "Roll3": "{:.2f}",
        "Roll5": "{:.2f}",
        "Predicted_Points": "{:.2f}"
    }))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_fd.style.format({
        "Spread": "{:.1f}",
        "Vegas_Total": "{:.1f}",
        "Home_Pred": "{:.2f}",
        "Away_Pred": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))

# DraftKings Predictions
st.markdown("---")
st.subheader("Week 7 Predictions – DraftKings Lines")
if st.button("Run Week 7 Predictions – DraftKings"):
    team_preds_dk, game_preds_dk = predict_matchups(week7_games_dk)
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_dk.style.format({
        "Spread": "{:.1f}",
        "Total": "{:.1f}",
        "Roll3": "{:.2f}",
        "Roll5": "{:.2f}",
        "Predicted_Points": "{:.2f}"
    }))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_dk.style.format({
        "Spread": "{:.1f}",
        "Vegas_Total": "{:.1f}",
        "Home_Pred": "{:.2f}",
        "Away_Pred": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))
