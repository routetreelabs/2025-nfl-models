# pages/Against_the_Spread.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

st.title("Against the Spread (ATS) Model - Logistic Regression")
st.markdown("**Week 12 Record:** Both models 10–4 ✅")
st.markdown("**Week 13 Record:** Both models 9–7 ✅")
st.markdown("**Week 14 Record:** Both models 7–7 ➖")
st.markdown("**Week 15 Record:** Both models 7–9 ❌")
st.markdown("**Week 16 Record:** Both models 11–4 ✅")
st.markdown("**Week 17 Record:** Both models 6–10 ❌")
st.markdown("**TEST")

# Load data
df = pd.read_csv("datasets/nfl_gamelogs_vegas_2015-2025_ML_week17_copy.csv")

# Feature Engineering
df['Cover_Binary'] = df['Cover']
df['Tm_3DConv_Rate'] = df['Tm_3DConv'] / df['Tm_3DAtt'].replace(0, 1)
df['Opp_3DConv_Rate'] = df['Opp_3DConv'] / df['Opp_3DAtt'].replace(0, 1)
df['Turnover_Diff'] = df['Opp_TO'] - df['Tm_TO']
df['Pt_Diff'] = df['Tm_Pts'] - df['Opp_Pts']
df['Margin_vs_Spread'] = df['Pt_Diff'] - df['Spread']
df["Tm_WinRate_Roll3"] = df.groupby("Team")["Win"].shift().rolling(3).mean()
df["Tm_PtDiff_Roll3"] = df.groupby("Team")["Pt_Diff"].shift().rolling(3).mean()

# Expanding averages
stat_cols = [
    "Tm_pY/A", "Tm_rY/A", "Tm_Y/P",
    "Opp_pY/A", "Opp_rY/A", "Opp_Y/P",
    "Tm_TO", "Opp_TO",
    "Tm_PenYds", "Opp_PenYds",
    "Tm_3DConv_Rate", "Opp_3DConv_Rate",
    "Turnover_Diff", "Tm_WinRate_Roll3", "Tm_PtDiff_Roll3"
]

for col in stat_cols:
    df[f"{col}_avg"] = (
        df.groupby(["Season", "Team"], group_keys=False)[col]
          .transform(lambda x: x.shift().expanding().mean())
    )
    league_avg = df[col].mean()
    df[f"{col}_avg"] = df[f"{col}_avg"].fillna(league_avg)

# Train Model
features_avg = ["Spread", "Total", "Home"] + [f"{c}_avg" for c in stat_cols]
df_clean = df.dropna(subset=features_avg + ["Cover_Binary"])

X = df_clean[features_avg]
y = df_clean["Cover_Binary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

ats_model = LogisticRegression(max_iter=2000, solver="liblinear")
ats_model.fit(X_train, y_train)

# Team names map/dictionary
team_name_map = {
    "KAN": "Chiefs", "CLT": "Colts", "CIN": "Bengals", "NWE": "Patriots",
    "CHI": "Bears", "PIT": "Steelers", "RAV": "Ravens", "NYJ": "Jets",
    "OTI": "Titans", "SEA": "Seahawks", "DET": "Lions", "NYG": "Giants",
    "GNB": "Packers", "MIN": "Vikings", "RAI": "Raiders", "CLE": "Browns",
    "CRD": "Cardinals", "JAX": "Jaguars", "NOR": "Saints", "ATL": "Falcons",
    "DAL": "Cowboys", "PHI": "Eagles", "RAM": "Rams", "TAM": "Buccaneers",
    "SFO": "49ers", "CAR": "Panthers", "HTX": "Texans", "BUF": "Bills", "WAS": "Commanders",
    "DEN": "Broncos", "SDG": "Chargers", "MIA": "Dolphins"
}


# Prediction Helpers
def get_team_features(latest_df, season, team, spread, total, home, stat_cols_for_avg):
    subset = latest_df[(latest_df["Season"] == season) & (latest_df["Team"] == team)]
    if subset.empty:
        subset = latest_df[latest_df["Team"] == team]
    team_row = subset.iloc[-1]

    f = {"Spread": spread, "Total": total, "Home": home}
    for c in stat_cols_for_avg:
        colname = f"{c}_avg"
        if colname in latest_df.columns and pd.notna(team_row.get(colname, np.nan)):
            f[colname] = team_row[colname]
        else:
            f[colname] = latest_df[c].mean() if c in latest_df.columns else 0.0
    return f

def run_weekly_ats_predictions(model, df, season, week, matchups, stat_cols_for_avg):
    all_preds = []
    for game in matchups:
        home_team, away_team, spread, total = game

        home_features = get_team_features(df, season, home_team, spread, total, 1, stat_cols_for_avg)
        away_features = get_team_features(df, season, away_team, -spread, total, 0, stat_cols_for_avg)

        temp_df = pd.DataFrame([home_features, away_features])
        probs = model.predict_proba(temp_df[["Spread", "Total", "Home"] + [f"{c}_avg" for c in stat_cols_for_avg]])[:, 1]
        preds = (probs >= 0.5).astype(int)

        home_prob = probs[0]
        pred = preds[0]
        ats_winner = home_team if pred == 1 else away_team

        all_preds.append({
            "Matchup": f"{team_name_map.get(away_team, away_team)} @ {team_name_map.get(home_team, home_team)}",
            "Spread (Home Team)": spread,
            "Home Cover Probability": f"{home_prob:.2%}",
            "Predicted ATS Winner": team_name_map.get(ats_winner, ats_winner)
        })
    return pd.DataFrame(all_preds)


# Week 12 Matchups
season, week = 2025, 18

week18_games_fd = [
    ("TAM", "CAR", -3.0, 43.5),
    ("SFO", "SEA", +2.5, 47.5),
    ("ATL", "NOR", -3.5, 44.5),
    ("CIN", "CLE", -7.5, 44.5),
    ("NYG", "DAL", +3.5, 49.5),
    ("MIN", "GNB", -8.5, 36.5),
    ("HTX", "CLT", -10.5, 38.5),
    ("JAX", "OTI", -12.5, 47.5),
    ("RAI", "KAN", +5.5, 36.5),
    ("CHI", "DET", -3.0, 50.5),
    ("DEN", "SDG", -13.5, 37.5),
    ("NWE", "MIA", -11.5, 45.5),
    ("PHI", "WAS", -3.5, 39.5),
    ("RAM", "CRD", -8.5, 45.5),
    ("BUF", "NYJ", -7.5, 37.5),
    ("PIT", "RAV", +3.5, 41.5),
]

week18_games_dk = [
    ("TAM", "CAR", -3.0, 43.5),
    ("SFO", "SEA", +2.5, 47.5),
    ("ATL", "NOR", -3.5, 43.5),
    ("CIN", "CLE", -7.5, 44.5),
    ("NYG", "DAL", +3.5, 49.5),
    ("MIN", "GNB", -9.5, 36.5),
    ("HTX", "CLT", -10.0, 38.5),
    ("JAX", "OTI", -13.5, 47.5),
    ("RAI", "KAN", +5.5, 36.5),
    ("CHI", "DET", -3.0, 50.5),
    ("DEN", "SDG", -13.5, 37.5),
    ("NWE", "MIA", -11.5, 45.5),
    ("PHI", "WAS", -4.5, 38.5),
    ("RAM", "CRD", -8.5, 46.5),
    ("BUF", "NYJ", -7.5, 37.5),
    ("PIT", "RAV", +3.5, 41.5),
]

# Display Predictions
st.subheader("FanDuel Lines - Week 18 ATS Predictions")
ats_fd = run_weekly_ats_predictions(ats_model, df, season, week, week18_games_fd, stat_cols)
st.dataframe(ats_fd, use_container_width=True)

st.subheader("DraftKings Lines - Week 18 ATS Predictions")
ats_dk = run_weekly_ats_predictions(ats_model, df, season, week, week18_games_dk, stat_cols)
st.dataframe(ats_dk, use_container_width=True)
