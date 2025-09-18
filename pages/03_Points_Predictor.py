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
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_Points_Week2_copy.csv")

# 2. Load and Prepare the Data
df = pd.read_csv(csv_path)
df["Total_Points_Scored"] = df["Tm_Pts"] + df["Opp_Pts"]

# Sort and create rolling average feature
df = df.sort_values(by=["Team", "Season", "Week"])
df["Tm_Pts_Last1"] = df.groupby("Team")["Tm_Pts"].shift(1)
df["Tm_Pts_Last1"] = df["Tm_Pts_Last1"].fillna(0)

# 3. Select Features
features_early = ["Season", "Week", "Home", "Team", "Opp", "Spread", "Total", "Tm_Pts_Last1"]
target_early = "Tm_Pts"

X_early = df[features_early]
y_early = df[target_early]

# 4. Preprocessing: One-hot encode 'Team' and 'Opp'
categorical_early = ["Team", "Opp"]
numerical_early = ["Season", "Week", "Home", "Spread", "Total", "Tm_Pts_Last1"]

preprocessor_early = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_early)],
    remainder="passthrough"
)

X_processed_early = preprocessor_early.fit_transform(X_early)

# 5. Split the Data
X_train_early, X_test_early, y_train_early, y_test_early = train_test_split(
    X_processed_early, y_early, test_size=0.2, random_state=42
)

# 6. Train Model
model_early = LinearRegression()
model_early.fit(X_train_early, y_train_early)

# 7. Evaluate
y_pred_early = model_early.predict(X_test_early)
mae_early = mean_absolute_error(y_test_early, y_pred_early)
rmse_early = np.sqrt(mean_squared_error(y_test_early, y_pred_early))
r2_early = r2_score(y_test_early, y_pred_early)

st.write(f"[Early Model] **R²:** {r2_early:.2f}")
st.write(f"[Early Model] **MAE:** {mae_early:.2f}")
st.write(f"[Early Model] **RMSE:** {rmse_early:.2f}")

# 8. Optional: Feature Importance
if st.checkbox("Show top features"):
    encoded_columns = preprocessor_early.named_transformers_["cat"].get_feature_names_out(categorical_early)
    all_columns = np.concatenate([encoded_columns, numerical_early])
    coef_df = pd.DataFrame({
        "Feature": all_columns,
        "Coefficient": model_early.coef_
    }).sort_values(by="Coefficient", ascending=False)
    st.dataframe(coef_df.head(10))

# 9. Prediction Function
def predict_week_points_early(games):
    input_df = pd.DataFrame(games)

    if "Last1" in input_df.columns:
        input_df["Tm_Pts_Last1"] = input_df["Last1"]

    # Make sure all required columns are present in the right order
    input_df = input_df[features_early]

    input_processed = preprocessor_early.transform(input_df)
    predictions = model_early.predict(input_processed)

    input_df["Predicted_Points"] = predictions

    return input_df


# 10. Helpers for correct row selection in grouped apply
def get_home_team(x):
    return x.loc[x["Home"] == 1, "Team"].values[0] if (x["Home"] == 1).any() else x["Team"].values[0]

def get_away_team(x):
    return x.loc[x["Home"] == 0, "Team"].values[0] if (x["Home"] == 0).any() else x["Team"].values[0]

def get_home_pred(x):
    return x.loc[x["Home"] == 1, "Predicted_Points"].values[0] if (x["Home"] == 1).any() else x["Predicted_Points"].values[0]

def get_away_pred(x):
    return x.loc[x["Home"] == 0, "Predicted_Points"].values[0] if (x["Home"] == 0).any() else x["Predicted_Points"].values[0]

# Helper function to generate both sides of each game
def add_reverse_games(game_list):
    doubled = []
    for game in game_list:
        # Original (home team)
        doubled.append(game)
        # Reverse (away team)
        reversed_game = {
            "Season": game["Season"],
            "Week": game["Week"],
            "Home": 0,
            "Team": game["Opp"],
            "Opp": game["Team"],
            "Spread": -game["Spread"],
            "Total": game["Total"],
            "Last1": game["Opp_Last1"],
            "Opp_Last1": game["Last1"],
        }
        doubled.append(reversed_game)
    return doubled

# FanDuel
week3_games_fd = [
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "BUF", "Opp": "MIA", "Spread": -11.5, "Total": 50.5, "Last1": 30, "Last2": 35.5, "Opp_Last1": 27, "Opp_Last2": 17.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CLE", "Opp": "GNB", "Spread": +7.5, "Total": 41.5, "Last1": 17, "Last2": 16.5, "Opp_Last1": 27, "Opp_Last2": 27.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "TAM", "Opp": "NYJ", "Spread": -6.5, "Total": 44.5, "Last1": 20, "Last2": 21.5, "Opp_Last1": 10, "Opp_Last2": 21.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "PHI", "Opp": "RAM", "Spread": -3.5, "Total": 44.5, "Last1": 20, "Last2": 22.0, "Opp_Last1": 33, "Opp_Last2": 23.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CAR", "Opp": "ATL", "Spread": +5.5, "Total": 43.5, "Last1": 22, "Last2": 16.0, "Opp_Last1": 22, "Opp_Last2": 21.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "OTI", "Opp": "CLT", "Spread": +4.5, "Total": 43.5, "Last1": 19, "Last2": 15.5, "Opp_Last1": 29, "Opp_Last2": 31.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NWE", "Opp": "PIT", "Spread": +1.5, "Total": 44.5, "Last1": 13, "Last2": 11.5, "Opp_Last1": 17, "Opp_Last2": 25.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "WAS", "Opp": "RAI", "Spread": -3.5, "Total": 44.5, "Last1": 18, "Last2": 19.5, "Opp_Last1": 9, "Opp_Last2": 14.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "MIN", "Opp": "CIN", "Spread": -3.0, "Total": 42.5, "Last1": 6, "Last2": 16.5, "Opp_Last1": 31, "Opp_Last2": 24.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "JAX", "Opp": "HTX", "Spread": -1.5, "Total": 44.5, "Last1": 27, "Last2": 26.5, "Opp_Last1": 19, "Opp_Last2": 14.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SEA", "Opp": "NOR", "Spread": -7.5, "Total": 41.5, "Last1": 31, "Last2": 22.0, "Opp_Last1": 21, "Opp_Last2": 17.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SDG", "Opp": "DEN", "Spread": -2.5, "Total": 45.5, "Last1": 20, "Last2": 23.5, "Opp_Last1": 28, "Opp_Last2": 24.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CHI", "Opp": "DAL", "Spread": +1.5, "Total": 50.5, "Last1": 21, "Last2": 22.5, "Opp_Last1": 40, "Opp_Last2": 30.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SFO", "Opp": "CRD", "Spread": -2.5, "Total": 44.5, "Last1": 26, "Last2": 21.5, "Opp_Last1": 27, "Opp_Last2": 23.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NYG", "Opp": "KAN", "Spread": +5.5, "Total": 44.5, "Last1": 37, "Last2": 21.5, "Opp_Last1": 17, "Opp_Last2": 19.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "RAV", "Opp": "DET", "Spread": -5.5, "Total": 52.5, "Last1": 41, "Last2": 40.5, "Opp_Last1": 52, "Opp_Last2": 32.5}
]

# DraftKings
week3_games_dk = [
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "BUF", "Opp": "MIA", "Spread": -12.5, "Total": 49.5, "Last1": 30, "Last2": 35.5, "Opp_Last1": 27, "Opp_Last2": 17.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CLE", "Opp": "GNB", "Spread": +8.5, "Total": 41.5, "Last1": 17, "Last2": 16.5, "Opp_Last1": 27, "Opp_Last2": 27.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "TAM", "Opp": "NYJ", "Spread": -7.0, "Total": 43.5, "Last1": 20, "Last2": 21.5, "Opp_Last1": 10, "Opp_Last2": 21.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "PHI", "Opp": "RAM", "Spread": -3.5, "Total": 44.5, "Last1": 20, "Last2": 22.0, "Opp_Last1": 33, "Opp_Last2": 23.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CAR", "Opp": "ATL", "Spread": +5.5, "Total": 43.5, "Last1": 22, "Last2": 16.0, "Opp_Last1": 22, "Opp_Last2": 21.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "OTI", "Opp": "CLT", "Spread": +4.5, "Total": 43.5, "Last1": 19, "Last2": 15.5, "Opp_Last1": 29, "Opp_Last2": 31.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NWE", "Opp": "PIT", "Spread": +1.5, "Total": 44.5, "Last1": 13, "Last2": 11.5, "Opp_Last1": 17, "Opp_Last2": 25.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "WAS", "Opp": "RAI", "Spread": -3.5, "Total": 44.5, "Last1": 18, "Last2": 19.5, "Opp_Last1": 9, "Opp_Last2": 14.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "MIN", "Opp": "CIN", "Spread": -3.0, "Total": 42.5, "Last1": 6, "Last2": 16.5, "Opp_Last1": 31, "Opp_Last2": 24.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "JAX", "Opp": "HTX", "Spread": -1.5, "Total": 44.5, "Last1": 27, "Last2": 26.5, "Opp_Last1": 19, "Opp_Last2": 14.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SEA", "Opp": "NOR", "Spread": -7.5, "Total": 41.5, "Last1": 31, "Last2": 22.0, "Opp_Last1": 21, "Opp_Last2": 17.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SDG", "Opp": "DEN", "Spread": -2.5, "Total": 45.5, "Last1": 20, "Last2": 23.5, "Opp_Last1": 28, "Opp_Last2": 24.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "CHI", "Opp": "DAL", "Spread": +1.5, "Total": 50.5, "Last1": 21, "Last2": 22.5, "Opp_Last1": 40, "Opp_Last2": 30.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "SFO", "Opp": "CRD", "Spread": -3.0, "Total": 46.5, "Last1": 26, "Last2": 21.5, "Opp_Last1": 27, "Opp_Last2": 23.5},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "NYG", "Opp": "KAN", "Spread": +6.0, "Total": 45.5, "Last1": 37, "Last2": 21.5, "Opp_Last1": 17, "Opp_Last2": 19.0},
    {"Season": 2025, "Week": 3, "Home": 1, "Team": "RAV", "Opp": "DET", "Spread": -4.5, "Total": 53.5, "Last1": 41, "Last2": 40.5, "Opp_Last1": 52, "Opp_Last2": 32.5}
]

# FanDuel Predictions
st.markdown("---")
st.subheader("Week 3 Predictions – FanDuel Lines")
if st.button("Run Week 2 Predictions – FanDuel"):
    week3_games_fd_doubled = add_reverse_games(week3_games_fd)
    week3_predictions_fd = predict_week_points_early(week3_games_fd_doubled)

    week3_predictions_fd["Opponent"] = week3_predictions_fd["Opp"]
    week3_predictions_fd = week3_predictions_fd.drop(columns="Opp")

    # Better matchup label: home vs away
    week3_predictions_fd["Matchup"] = week3_predictions_fd.apply(
        lambda row: f"{row['Team']} vs {row['Opponent']}" if row["Home"] == 1 else f"{row['Opponent']} vs {row['Team']}",
        axis=1
    )

    # Fix home/away grouping using apply()
    totals_fd = week3_predictions_fd.groupby("Matchup").apply(lambda x: pd.Series({
        "Home_Team": get_home_team(x),
        "Away_Team": get_away_team(x),
        "Home_Predicted": get_home_pred(x),
        "Away_Predicted": get_away_pred(x),
        "Predicted_Total": x["Predicted_Points"].sum()
    })).reset_index()

    st.dataframe(week3_predictions_fd.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Predicted Totals (FanDuel):**")
    st.dataframe(totals_fd.style.format({
        "Home_Predicted": "{:.2f}",
        "Away_Predicted": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))


# --- DraftKings Predictions
st.markdown("---")
st.subheader("Week 3 Predictions – DraftKings Lines")
if st.button("Run Week 3 Predictions – DraftKings"):
    week3_games_dk_doubled = add_reverse_games(week3_games_dk)
    week3_predictions_dk = predict_week_points_early(week3_games_dk_doubled)

    week3_predictions_dk["Opponent"] = week3_predictions_dk["Opp"]
    week3_predictions_dk = week3_predictions_dk.drop(columns="Opp")

    week3_predictions_dk["Matchup"] = week3_predictions_dk.apply(
        lambda row: f"{row['Team']} vs {row['Opponent']}" if row["Home"] == 1 else f"{row['Opponent']} vs {row['Team']}",
        axis=1
    )

    totals_dk = week3_predictions_dk.groupby("Matchup").apply(lambda x: pd.Series({
        "Home_Team": get_home_team(x),
        "Away_Team": get_away_team(x),
        "Home_Predicted": get_home_pred(x),
        "Away_Predicted": get_away_pred(x),
        "Predicted_Total": x["Predicted_Points"].sum()
    })).reset_index()

    st.dataframe(week3_predictions_dk.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Predicted Totals (DraftKings):**")
    st.dataframe(totals_dk.style.format({
        "Home_Predicted": "{:.2f}",
        "Away_Predicted": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))


