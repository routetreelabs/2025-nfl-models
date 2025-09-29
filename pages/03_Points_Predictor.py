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

# --- Controls
debug = st.checkbox("Debug mode (print intermediate variables)")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_Points_Week3_copy.csv")

# === Load and Prepare the Data ===
df = pd.read_csv(csv_path)
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Sort so rolling stats are correct
df = df.sort_values(by=["Team", "Season", "Week"])

# Rolling features
df["Tm_Pts_Last1"] = df.groupby("Team")["Tm_Pts"].shift(1)
df["Tm_Pts_Rolling3"] = df.groupby("Team")["Tm_Pts"].shift(1).rolling(3).mean().reset_index(0, drop=True)
df[["Tm_Pts_Last1", "Tm_Pts_Rolling3"]] = df[["Tm_Pts_Last1", "Tm_Pts_Rolling3"]].fillna(0)

# Features & target
features_early = ["Season", "Week", "Home", "Team", "Opp",
                  "Spread", "Total", "Tm_Pts_Last1", "Tm_Pts_Rolling3"]
target_early = "Tm_Pts"
X_early = df[features_early]
y_early = df[target_early]

# Preprocessing
categorical_early = ["Team", "Opp"]
numerical_early = ["Season", "Week", "Home", "Spread", "Total", "Tm_Pts_Last1", "Tm_Pts_Rolling3"]

preprocessor_early = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_early)],
    remainder="passthrough"
)
X_processed_early = preprocessor_early.fit_transform(X_early)

# Train/test split
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_processed_early, y_early, test_size=0.2, random_state=42
)

# Train model
model_early = LinearRegression()
model_early.fit(X_train_early, y_train_early)

# Evaluate
y_pred_early = model_early.predict(X_test_early)
mae_early = mean_absolute_error(y_test_early, y_pred_early)
rmse_early = np.sqrt(mean_squared_error(y_test_early, y_pred_early))
r2_early = r2_score(y_test_early, y_pred_early)

st.write(f"[Early Model] **R²:** {r2_early:.2f}")
st.write(f"[Early Model] **MAE:** {mae_early:.2f}")
st.write(f"[Early Model] **RMSE:** {rmse_early:.2f}")

# --- Prediction Functions ---
def predict_team_points(season, week, home, team, opponent, spread, total, last1, roll3):
    input_df = pd.DataFrame([{
        "Season": season, "Week": week, "Home": home,
        "Team": team, "Opp": opponent, "Spread": spread, "Total": total,
        "Tm_Pts_Last1": last1, "Tm_Pts_Rolling3": roll3
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
            last1=game["Last1"], roll3=game["Roll3"]
        )
        team_results.append({
            "Season": game["Season"], "Week": game["Week"], "Team": game["Team"], "Opponent": game["Opp"],
            "Home": game["Home"], "Spread": game["Spread"], "Total": game["Total"],
            "Last1": game["Last1"], "Roll3": game["Roll3"], "Predicted_Points": round(team_pred, 2)
        })

        opp_pred = predict_team_points(
            season=game["Season"], week=game["Week"], home=1-game["Home"],
            team=game["Opp"], opponent=game["Team"], spread=-game["Spread"], total=game["Total"],
            last1=game.get("Opp_Last1", 0), roll3=game.get("Opp_Roll3", 0)
        )
        team_results.append({
            "Season": game["Season"], "Week": game["Week"], "Team": game["Opp"], "Opponent": game["Team"],
            "Home": 1-game["Home"], "Spread": -game["Spread"], "Total": game["Total"],
            "Last1": game.get("Opp_Last1", 0), "Roll3": game.get("Opp_Roll3", 0),
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
week4_games_fd = [
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CRD", "Opp": "SEA", "Spread": +1.5, "Total": 43.5, "Last1": 15, "Roll3": 20.6, "Opp_Last1": 44, "Opp_Roll3": 29.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "PIT", "Opp": "MIN", "Spread": +2.5, "Total": 40.5, "Last1": 21, "Roll3": 24.0, "Opp_Last1": 48, "Opp_Roll3": 27.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "TAM", "Opp": "PHI", "Spread": +3.5, "Total": 44.5, "Last1": 29, "Roll3": 24.0, "Opp_Last1": 33, "Opp_Roll3": 25.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NYG", "Opp": "SDG", "Spread": +6.5, "Total": 42.5, "Last1": 6, "Roll3": 16.3, "Opp_Last1": 23, "Opp_Roll3": 23.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "BUF", "Opp": "NOR", "Spread": -14.5, "Total": 48.5, "Last1": 31, "Roll3": 34.0, "Opp_Last1": 13, "Opp_Roll3": 15.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "ATL", "Opp": "WAS", "Spread": -2.5, "Total": 43.5, "Last1": 0, "Roll3": 14.0, "Opp_Last1": 41, "Opp_Roll3": 26.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "DET", "Opp": "CLE", "Spread": -9.5, "Total": 44.5, "Last1": 38, "Roll3": 34.3, "Opp_Last1": 13, "Opp_Roll3": 15.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "HTX", "Opp": "OTI", "Spread": -7.5, "Total": 39.5, "Last1": 10, "Roll3": 12.6, "Opp_Last1": 20, "Opp_Roll3": 17.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NWE", "Opp": "CAR", "Spread": -5.5, "Total": 43.5, "Last1": 14, "Roll3": 20.0, "Opp_Last1": 30, "Opp_Roll3": 20.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "RAM", "Opp": "CLT", "Spread": -3.5, "Total": 49.5, "Last1": 26, "Roll3": 24.3, "Opp_Last1": 41, "Opp_Roll3": 34.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SFO", "Opp": "JAX", "Spread": -3.5, "Total": 45.5, "Last1": 16, "Roll3": 19.6, "Opp_Last1": 17, "Opp_Roll3": 23.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "RAI", "Opp": "CHI", "Spread": -1.5, "Total": 47.5, "Last1": 24, "Roll3": 17.6, "Opp_Last1": 31, "Opp_Roll3": 25.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "KAN", "Opp": "RAV", "Spread": +2.5, "Total": 48.5, "Last1": 22, "Roll3": 20.0, "Opp_Last1": 30, "Opp_Roll3": 37.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "DAL", "Opp": "GNB", "Spread": +6.5, "Total": 46.5, "Last1": 14, "Roll3": 24.6, "Opp_Last1": 10, "Opp_Roll3": 21.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "MIA", "Opp": "NYJ", "Spread": -2.5, "Total": 43.5, "Last1": 21, "Roll3": 18.6, "Opp_Last1": 27, "Opp_Roll3": 23.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "DEN", "Opp": "CIN", "Spread": -7.5, "Total": 44.5, "Last1": 20, "Roll3": 22.6, "Opp_Last1": 10, "Opp_Roll3": 19.3}
]

# DraftKings
week4_games_dk = [
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CRD", "Opp": "SEA", "Spread": +1.5, "Total": 43.5, "Last1": 15, "Roll3": 20.6, "Opp_Last1": 44, "Opp_Roll3": 29.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "PIT", "Opp": "MIN", "Spread": +2.5, "Total": 41.5, "Last1": 21, "Roll3": 24.0, "Opp_Last1": 48, "Opp_Roll3": 27.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "TAM", "Opp": "PHI", "Spread": +3.5, "Total": 44.5, "Last1": 29, "Roll3": 24.0, "Opp_Last1": 33, "Opp_Roll3": 25.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NYG", "Opp": "SDG", "Spread": +6.0, "Total": 43.5, "Last1": 6, "Roll3": 16.3, "Opp_Last1": 23, "Opp_Roll3": 23.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "BUF", "Opp": "NOR", "Spread": -15.5, "Total": 48.5, "Last1": 31, "Roll3": 34.0, "Opp_Last1": 13, "Opp_Roll3": 15.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "ATL", "Opp": "WAS", "Spread": -2.5, "Total": 43.5, "Last1": 0, "Roll3": 14.0, "Opp_Last1": 41, "Opp_Roll3": 26.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "DET", "Opp": "CLE", "Spread": -10.0, "Total": 44.5, "Last1": 38, "Roll3": 34.3, "Opp_Last1": 13, "Opp_Roll3": 15.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "HTX", "Opp": "OTI", "Spread": -7.5, "Total": 39.5, "Last1": 10, "Roll3": 12.6, "Opp_Last1": 20, "Opp_Roll3": 17.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NWE", "Opp": "CAR", "Spread": -5.5, "Total": 42.5, "Last1": 14, "Roll3": 20.0, "Opp_Last1": 30, "Opp_Roll3": 20.6},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "RAM", "Opp": "CLT", "Spread": -3.5, "Total": 49.5, "Last1": 26, "Roll3": 24.3, "Opp_Last1": 41, "Opp_Roll3": 34.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SFO", "Opp": "JAX", "Spread": -3.5, "Total": 45.5, "Last1": 16, "Roll3": 19.6, "Opp_Last1": 17, "Opp_Roll3": 23.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "RAI", "Opp": "CHI", "Spread": -1.5, "Total": 47.5, "Last1": 24, "Roll3": 17.6, "Opp_Last1": 31, "Opp_Roll3": 25.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "KAN", "Opp": "RAV", "Spread": +2.5, "Total": 48.5, "Last1": 22, "Roll3": 20.0, "Opp_Last1": 30, "Opp_Roll3": 37.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "DAL", "Opp": "GNB", "Spread": +7.0, "Total": 47.5, "Last1": 14, "Roll3": 24.6, "Opp_Last1": 10, "Opp_Roll3": 21.3},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "MIA", "Opp": "NYJ", "Spread": -2.5, "Total": 44.5, "Last1": 21, "Roll3": 18.6, "Opp_Last1": 27, "Opp_Roll3": 23.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "DEN", "Opp": "CIN", "Spread": -7.5, "Total": 44.5, "Last1": 20, "Roll3": 22.6, "Opp_Last1": 10, "Opp_Roll3": 19.3}
]

# FanDuel Predictions
st.markdown("---")
st.subheader("Week 4 Predictions – FanDuel Lines")
if st.button("Run Week 4 Predictions – FanDuel"):
    team_preds_fd, game_preds_fd = predict_matchups(week4_games_fd)
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_fd.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_fd.style.format({
        "Home_Pred": "{:.2f}",
        "Away_Pred": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))

# DraftKings Predictions
st.markdown("---")
st.subheader("Week 4 Predictions – DraftKings Lines")
if st.button("Run Week 4 Predictions – DraftKings"):
    team_preds_dk, game_preds_dk = predict_matchups(week4_games_dk)
    st.write("**Team-Level Predictions**")
    st.dataframe(team_preds_dk.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Game-Level Predictions**")
    st.dataframe(game_preds_dk.style.format({
        "Home_Pred": "{:.2f}",
        "Away_Pred": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))



