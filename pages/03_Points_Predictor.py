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
csv_path = os.path.join(DATA_DIR, "nfl_gamelogs_vegas_2015-2025_FINAL.csv")

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

# Week 2 Games (FanDuel)
# FanDuel Week 2
week2_games_fd = [
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "GNB", "Opp": "WAS", "Spread": -3.5, "Total": 48.5, "Last1": 27, "Opp_Last1": 21},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "DAL", "Opp": "NYG", "Spread": -5.5, "Total": 44.5, "Last1": 20, "Opp_Last1": 6},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "NYJ", "Opp": "BUF", "Spread": +7.0, "Total": 45.5, "Last1": 32, "Opp_Last1": 41},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "OTI", "Opp": "RAM", "Spread": +5.5, "Total": 42.5, "Last1": 12, "Opp_Last1": 14},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "PIT", "Opp": "SEA", "Spread": -2.5, "Total": 39.5, "Last1": 34, "Opp_Last1": 13},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "NOR", "Opp": "SFO", "Spread": +4.5, "Total": 42.5, "Last1": 13, "Opp_Last1": 17},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "BAL", "Opp": "CLE", "Spread": -11.5, "Total": 45.5, "Last1": 40, "Opp_Last1": 16},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "CIN", "Opp": "JAX", "Spread": -3.5, "Total": 49.5, "Last1": 17, "Opp_Last1": 26},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "DET", "Opp": "CHI", "Spread": -5.5, "Total": 47.5, "Last1": 13, "Opp_Last1": 24},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "MIA", "Opp": "NWE", "Spread": -1.5, "Total": 43.5, "Last1": 8, "Opp_Last1": 13},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "CRD", "Opp": "CAR", "Spread": -6.5, "Total": 43.5, "Last1": 20, "Opp_Last1": 10},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "CLT", "Opp": "DEN", "Spread": +2.5, "Total": 42.5, "Last1": 33, "Opp_Last1": 20},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "KAN", "Opp": "PHI", "Spread": +1.5, "Total": 46.5, "Last1": 21, "Opp_Last1": 24},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "MIN", "Opp": "ATL", "Spread": -4.5, "Total": 45.5, "Last1": 27, "Opp_Last1": 20},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "HTX", "Opp": "TAM", "Spread": -2.5, "Total": 42.5, "Last1": 9, "Opp_Last1": 23},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "RAI", "Opp": "SDG", "Spread": +3.0, "Total": 46.5, "Last1": 20, "Opp_Last1": 27},
]

# Week 2 Games (DraftKings)
week2_games_dk = [
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "GNB", "Opp": "WAS", "Spread": -3.5, "Total": 48.5, "Last1": 27, "Opp_Last1": 21},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "DAL", "Opp": "NYG", "Spread": -5.5, "Total": 44.5, "Last1": 20, "Opp_Last1": 6},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "NYJ", "Opp": "BUF", "Spread": +7.0, "Total": 46.5, "Last1": 32, "Opp_Last1": 41},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "OTI", "Opp": "RAM", "Spread": +5.5, "Total": 41.5, "Last1": 12, "Opp_Last1": 14},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "PIT", "Opp": "SEA", "Spread": -3.0, "Total": 40.5, "Last1": 34, "Opp_Last1": 13},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "NOR", "Opp": "SFO", "Spread": +4.5, "Total": 42.5, "Last1": 13, "Opp_Last1": 17},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "BAL", "Opp": "CLE", "Spread": -11.5, "Total": 45.5, "Last1": 40, "Opp_Last1": 16},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "CIN", "Opp": "JAX", "Spread": -3.5, "Total": 49.5, "Last1": 17, "Opp_Last1": 26},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "DET", "Opp": "CHI", "Spread": -5.5, "Total": 46.5, "Last1": 13, "Opp_Last1": 24},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "MIA", "Opp": "NWE", "Spread": -1.5, "Total": 43.5, "Last1": 8, "Opp_Last1": 13},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "CRD", "Opp": "CAR", "Spread": -6.5, "Total": 44.5, "Last1": 20, "Opp_Last1": 10},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "CLT", "Opp": "DEN", "Spread": +2.5, "Total": 42.5, "Last1": 33, "Opp_Last1": 20},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "KAN", "Opp": "PHI", "Spread": +1.5, "Total": 46.5, "Last1": 21, "Opp_Last1": 24},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "MIN", "Opp": "ATL", "Spread": -4.5, "Total": 44.5, "Last1": 27, "Opp_Last1": 20},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "HTX", "Opp": "TAM", "Spread": -2.5, "Total": 42.5, "Last1": 9, "Opp_Last1": 23},
    {"Season": 2025, "Week": 2, "Home": 1, "Team": "RAI", "Opp": "SDG", "Spread": +3.5, "Total": 46.5, "Last1": 20, "Opp_Last1": 27},
]

# FanDuel Predictions
st.markdown("---")
st.subheader("Week 2 Predictions – FanDuel Lines")
if st.button("Run Week 2 Predictions – FanDuel"):
    week2_games_fd_doubled = add_reverse_games(week2_games_fd)
    week2_predictions_fd = predict_week_points_early(week2_games_fd_doubled)

    week2_predictions_fd["Opponent"] = week2_predictions_fd["Opp"]
    week2_predictions_fd = week2_predictions_fd.drop(columns="Opp")

    # Better matchup label: home vs away
    week2_predictions_fd["Matchup"] = week2_predictions_fd.apply(
        lambda row: f"{row['Team']} vs {row['Opponent']}" if row["Home"] == 1 else f"{row['Opponent']} vs {row['Team']}",
        axis=1
    )

    # Fix home/away grouping using apply()
    totals_fd = week2_predictions_fd.groupby("Matchup").apply(lambda x: pd.Series({
        "Home_Team": get_home_team(x),
        "Away_Team": get_away_team(x),
        "Home_Predicted": get_home_pred(x),
        "Away_Predicted": get_away_pred(x),
        "Predicted_Total": x["Predicted_Points"].sum()
    })).reset_index()

    st.dataframe(week2_predictions_fd.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Predicted Totals (FanDuel):**")
    st.dataframe(totals_fd.style.format({
        "Home_Predicted": "{:.2f}",
        "Away_Predicted": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))


# --- DraftKings Predictions
st.markdown("---")
st.subheader("Week 2 Predictions – DraftKings Lines")
if st.button("Run Week 2 Predictions – DraftKings"):
    week2_games_dk_doubled = add_reverse_games(week2_games_dk)
    week2_predictions_dk = predict_week_points_early(week2_games_dk_doubled)

    week2_predictions_dk["Opponent"] = week2_predictions_dk["Opp"]
    week2_predictions_dk = week2_predictions_dk.drop(columns="Opp")

    week2_predictions_dk["Matchup"] = week2_predictions_dk.apply(
        lambda row: f"{row['Team']} vs {row['Opponent']}" if row["Home"] == 1 else f"{row['Opponent']} vs {row['Team']}",
        axis=1
    )

    totals_dk = week2_predictions_dk.groupby("Matchup").apply(lambda x: pd.Series({
        "Home_Team": get_home_team(x),
        "Away_Team": get_away_team(x),
        "Home_Predicted": get_home_pred(x),
        "Away_Predicted": get_away_pred(x),
        "Predicted_Total": x["Predicted_Points"].sum()
    })).reset_index()

    st.dataframe(week2_predictions_dk.style.format({"Predicted_Points": "{:.2f}"}))
    st.write("**Predicted Totals (DraftKings):**")
    st.dataframe(totals_dk.style.format({
        "Home_Predicted": "{:.2f}",
        "Away_Predicted": "{:.2f}",
        "Predicted_Total": "{:.2f}"
    }))


