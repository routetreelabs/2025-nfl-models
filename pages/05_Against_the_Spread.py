"""
ats_model.py

Logistic Regression ATS (Against the Spread) model for NFL games.
- Loads & feature-engineers historical data
- Trains a probabilistic ATS model
- Provides helpers for weekly predictions (FanDuel, DraftKings, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# TEAM NAME MAP 

TEAM_NAMES = {
    "HTX": "Texans",
    "BUF": "Bills",
    "KAN": "Chiefs",
    "CLT": "Colts",
    "CIN": "Bengals",
    "NWE": "Patriots",
    "CHI": "Bears",
    "PIT": "Steelers",
    "RAV": "Ravens",
    "NYJ": "Jets",
    "OTI": "Titans",
    "SEA": "Seahawks",
    "DET": "Lions",
    "NYG": "Giants",
    "GNB": "Packers",
    "MIN": "Vikings",
    "RAI": "Raiders",
    "CLE": "Browns",
    "CRD": "Cardinals",
    "JAX": "Jaguars",
    "NOR": "Saints",
    "ATL": "Falcons",
    "DAL": "Cowboys",
    "PHI": "Eagles",
    "RAM": "Rams",
    "TAM": "Buccaneers",
    "SFO": "49ers",
    "CAR": "Panthers",
}


# 1. DATA LOADING & FEATURE ENGINEERING

def load_and_engineer_ats_data(csv_path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Load the NFL game logs CSV and perform ATS-specific feature engineering.

    Returns
    -------
    df : pd.DataFrame
        Full engineered dataframe.
    stat_cols : list[str]
        Raw stat columns used for expanding averages.
    features_avg : list[str]
        Final feature columns used for training/prediction.
    """
    df = pd.read_csv(csv_path)

    # ----- Target -----
    df["Cover_Binary"] = df["Cover"]

    # ----- Basic rates -----
    df["Tm_3DConv_Rate"] = df["Tm_3DConv"] / df["Tm_3DAtt"].replace(0, 1)
    df["Opp_3DConv_Rate"] = df["Opp_3DConv"] / df["Opp_3DAtt"].replace(0, 1)

    # Turnover differential
    df["Turnover_Diff"] = df["Opp_TO"] - df["Tm_TO"]

    # Point differential + margin vs spread
    df["Pt_Diff"] = df["Tm_Pts"] - df["Opp_Pts"]
    df["Margin_vs_Spread"] = df["Pt_Diff"] - df["Spread"]

    # Rolling 3-game win rate and point diff (per Team)
    df["Tm_WinRate_Roll3"] = df.groupby("Team")["Win"].shift().rolling(3).mean()
    df["Tm_PtDiff_Roll3"] = df.groupby("Team")["Pt_Diff"].shift().rolling(3).mean()

    # Stat columns for expanding means
    stat_cols = [
        "Tm_pY/A", "Tm_rY/A", "Tm_Y/P",
        "Opp_pY/A", "Opp_rY/A", "Opp_Y/P",
        "Tm_TO", "Opp_TO",
        "Tm_PenYds", "Opp_PenYds",
        "Tm_3DConv_Rate", "Opp_3DConv_Rate",
        "Turnover_Diff", "Tm_WinRate_Roll3", "Tm_PtDiff_Roll3",
    ]

    # Leakage-free expanding averages: per (Season, Team), shifted by 1
    for col in stat_cols:
        df[f"{col}_avg"] = (
            df.groupby(["Season", "Team"], group_keys=False)[col]
              .transform(lambda x: x.shift().expanding().mean())
        )
        league_avg = df[col].mean()
        df[f"{col}_avg"] = df[f"{col}_avg"].fillna(league_avg)

    # Final feature list
    features_avg = ["Spread", "Total", "Home"] + [f"{c}_avg" for c in stat_cols]

    return df, stat_cols, features_avg


# 2. MODEL TRAINING


def train_ats_model(
    df: pd.DataFrame,
    features_avg: list[str],
    target_col: str = "Cover_Binary",
    random_state: int = 42,
) -> tuple[LogisticRegression, dict]:
    """
    Train a logistic regression ATS model.

    Returns
    -------
    ats_model : LogisticRegression
        Fitted logistic regression model.
    metrics : dict
        Dictionary of training metrics (train/test accuracy, ROC-AUC, CV accuracy, etc.).
    """
    df_clean = df.dropna(subset=features_avg + [target_col])

    X = df_clean[features_avg]
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000, solver="liblinear")
    model.fit(X_train, y_train)

    # Holdout metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    cls_report = classification_report(y_test, y_test_pred, output_dict=False)
    conf_mat = confusion_matrix(y_test, y_test_pred)

    # CV metrics (accuracy)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_roc_auc": test_auc,
        "classification_report": cls_report,
        "confusion_matrix": conf_mat,
        "cv_accuracy_mean": cv_acc.mean(),
        "cv_accuracy_std": cv_acc.std(),
    }

    return model, metrics


# 3. HELPERS FOR WEEKLY PREDICTIONS

def get_team_features(
    latest_df: pd.DataFrame,
    season: int,
    team: str,
    spread: float,
    total: float,
    home: int,
    stat_cols_for_avg: list[str],
) -> dict:
    """
    Build a single feature row for a given team using the latest available
    expanding-average stats. If no rows are found for this team/season,
    falls back to all seasons for that team. If still empty, uses league averages.
    """
    subset = latest_df[(latest_df["Season"] == season) & (latest_df["Team"] == team)]
    if subset.empty:
        subset = latest_df[latest_df["Team"] == team]

    f = {"Spread": spread, "Total": total, "Home": home}

    if subset.empty:
        # Full fallback: league averages
        for c in stat_cols_for_avg:
            colname = f"{c}_avg"
            if colname in latest_df.columns:
                f[colname] = latest_df[colname].mean()
            elif c in latest_df.columns:
                f[colname] = latest_df[c].mean()
            else:
                f[colname] = 0.0
        return f

    team_row = subset.iloc[-1]

    for c in stat_cols_for_avg:
        colname = f"{c}_avg"
        if (colname in latest_df.columns) and pd.notna(team_row.get(colname, np.nan)):
            f[colname] = team_row[colname]
        else:
            # Fallback to league avg of the raw column if available
            if c in latest_df.columns:
                f[colname] = latest_df[c].mean()
            elif colname in latest_df.columns:
                f[colname] = latest_df[colname].mean()
            else:
                f[colname] = 0.0

    return f


def run_weekly_ats_predictions(
    model: LogisticRegression,
    df: pd.DataFrame,
    season: int,
    matchups: list[tuple[str, str, float, float]],
    stat_cols_for_avg: list[str],
) -> pd.DataFrame:
    """
    Run ATS predictions for a slate of games.

    Parameters
    ----------
    model : fitted LogisticRegression
        The trained ATS model.
    df : pd.DataFrame
        Full historical dataframe with engineered features.
    season : int
        Season (e.g., 2025).
    matchups : list of tuples
        Each tuple is (home_team_abbr, away_team_abbr, home_spread, total).
        Example: [("HTX", "BUF", +5.5, 43.5), ...]
    stat_cols_for_avg : list[str]
        Raw stat columns used for the *_avg features.

    Returns
    -------
    results_df : pd.DataFrame
        One row per game with model probabilities and full team names.
    """
    results = []

    feature_cols = ["Spread", "Total", "Home"] + [f"{c}_avg" for c in stat_cols_for_avg]

    for home_team, away_team, spread, total in matchups:
        # Home and away feature rows
        home_features = get_team_features(
            latest_df=df,
            season=season,
            team=home_team,
            spread=spread,
            total=total,
            home=1,
            stat_cols_for_avg=stat_cols_for_avg,
        )
        away_features = get_team_features(
            latest_df=df,
            season=season,
            team=away_team,
            spread=-spread,  # from away POV
            total=total,
            home=0,
            stat_cols_for_avg=stat_cols_for_avg,
        )

        temp_df = pd.DataFrame([home_features, away_features])

        probs = model.predict_proba(temp_df[feature_cols])[:, 1]
        preds = (probs >= 0.5).astype(int)

        home_prob = probs[0]  # probability that "home team covers"
        home_pred = preds[0]  # 1 = home covers, 0 = away covers

        # Map abbreviations to full names for display
        home_full = TEAM_NAMES.get(home_team, home_team)
        away_full = TEAM_NAMES.get(away_team, away_team)
        winner_full = home_full if home_pred == 1 else away_full

        results.append(
            {
                "Matchup": f"{away_full} @ {home_full}",
                "Home_Team_Abbr": home_team,
                "Away_Team_Abbr": away_team,
                "Home_Team": home_full,
                "Away_Team": away_full,
                "Spread_Home": spread,
                "Total": total,
                "Home_Cover_Prob": home_prob,
                "Predicted_ATS_Winner": winner_full,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


# 4. OPTIONAL: EXAMPLE USAGE (for testing locally)

if __name__ == "__main__":
    # Example: train and run Week 12 predictions with hardcoded lines
    csv_path = "nfl_gamelogs_vegas_2015-2025_ML_week11.csv"

    df, stat_cols, features_avg = load_and_engineer_ats_data(csv_path)
    model, metrics = train_ats_model(df, features_avg)

    print("=== ATS Model Metrics (Quick View) ===")
    print("Train Acc:", metrics["train_accuracy"])
    print("Test  Acc:", metrics["test_accuracy"])
    print("Test  ROC-AUC:", metrics["test_roc_auc"])
    print("CV Acc  mean±std:", f"{metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])

    week12_games_fd = [
        ("HTX", "BUF", +5.5, 43.5),
        ("KAN", "CLT", -3.5, 49.5),
        ("CIN", "NWE", +7.5, 49.5),
        ("CHI", "PIT", -2.5, 44.5),
        ("RAV", "NYJ", -13.5, 44.5),
        ("OTI", "SEA", +13.5, 40.5),
        ("DET", "NYG", -10.5, 49.5),
        ("GNB", "MIN", -6.5, 41.5),
        ("RAI", "CLE", -3.5, 36.5),
        ("CRD", "JAX", +2.5, 47.5),
        ("NOR", "ATL", -1.5, 39.5),
        ("DAL", "PHI", +3.5, 47.5),
        ("RAM", "TAM", -6.5, 49.5),
        ("SFO", "CAR", -7.0, 48.5),
    ]

    season = 2025
    ats_results_fd = run_weekly_ats_predictions(model, df, season, week12_games_fd, stat_cols)
    print("\n=== Week 12 FanDuel ATS Predictions ===")
    print(ats_results_fd.to_string(index=False))
