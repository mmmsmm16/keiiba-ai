"""
Deep Value Model (Hard Weighted + No Yoso)
==========================================
Trains a LambdaRank model for value hunting.
"""
import os

import joblib
import lightgbm as lgb
import pandas as pd

DATA_PATH = os.environ.get("V13_DATA_PATH", "data/processed/preprocessed_data_v12.parquet")
BASELINE_FEATURES_PATH = os.environ.get(
    "V13_BASELINE_FEATURES_PATH",
    "models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted/features.csv",
)
OUTPUT_DIR = os.environ.get("V13_OUTPUT_DIR", "models/experiments/exp_lambdarank_hard_weighted")
ODDS_MODE = os.environ.get("V13_ODDS_MODE", "LEGACY").upper()  # LEGACY / REDUCED / NO_ODDS


def _is_odds_like_feature(name: str) -> bool:
    s = str(name).lower()
    return ("odds" in s) or ("popularity" in s) or ("ninki" in s)


print("Loading data...")
print(f"Data path: {DATA_PATH}")
print(f"Baseline feature list: {BASELINE_FEATURES_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Odds mode: {ODDS_MODE}")

df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df = df[(df["odds"] > 0) & (df["odds"].notna())].reset_index(drop=True)

print("Adding features...")
if "odds_10min" not in df.columns or df["odds_10min"].isna().all():
    print("Warning: odds_10min missing in parquet. Falling back to odds for pre-race proxy.")
    df["odds_feature"] = df["odds"]
else:
    df["odds_feature"] = df["odds_10min"]

df["odds_rank_pre"] = df.groupby("race_id")["odds_feature"].rank(ascending=True, method="min")
if "relative_horse_elo_z" in df.columns:
    df["elo_rank"] = df.groupby("race_id")["relative_horse_elo_z"].rank(ascending=False, method="min")
    df["odds_rank_vs_elo"] = df["odds_rank_pre"] - df["elo_rank"]
else:
    df["odds_rank_vs_elo"] = 0.0

df["is_high_odds"] = (df["odds_feature"] >= 10).astype(int)
df["is_mid_odds"] = ((df["odds_feature"] >= 5) & (df["odds_feature"] < 10)).astype(int)

baseline_features = pd.read_csv(BASELINE_FEATURES_PATH)["0"].astype(str).tolist()
filtered_features = [f for f in baseline_features if "yoso" not in f.lower()]

if ODDS_MODE in {"REDUCED", "NO_ODDS"}:
    before = len(filtered_features)
    filtered_features = [f for f in filtered_features if not _is_odds_like_feature(f)]
    print(f"Removed odds/popularity-like baseline features: {before - len(filtered_features)}")

if ODDS_MODE == "LEGACY":
    new_features = ["odds_rank_vs_elo", "is_high_odds", "is_mid_odds"]
elif ODDS_MODE == "REDUCED":
    new_features = ["odds_rank_vs_elo"]
else:
    new_features = []

all_features = [c for c in (filtered_features + new_features) if c in df.columns]
print(f"Total selected features: {len(all_features)}")

df_train = df[df["year"] <= 2022].copy()
df_valid = df[df["year"] == 2023].copy()
df_test = df[df["year"] == 2024].copy()

train_groups = df_train.groupby("race_id").size().values
valid_groups = df_valid.groupby("race_id").size().values

X_train = df_train[all_features].values
X_valid = df_valid[all_features].values
X_test = df_test[all_features].values

y_train = df_train["rank"].apply(lambda r: max(0, 4 - r)).values
y_valid = df_valid["rank"].apply(lambda r: max(0, 4 - r)).values
w_train = df_train["odds"].clip(1.0, 50.0).values

train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=w_train, feature_name=all_features)
valid_data = lgb.Dataset(X_valid, label=y_valid, group=valid_groups, reference=train_data, feature_name=all_features)

print("\nTraining Deep Value Model...")
params = {
    "num_leaves": 31,
    "learning_rate": 0.01,
    "n_estimators": 2000,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5],
    "boosting_type": "gbdt",
    "verbosity": -1,
    "seed": 42,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=params["n_estimators"],
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
)

print("\n=== Evaluation (Deep Value) ===")
scores = model.predict(X_test)
df_test["pred_score"] = scores
df_test["pred_rank"] = df_test.groupby("race_id")["pred_score"].rank(ascending=False, method="first")

rank1 = df_test[df_test["pred_rank"] == 1].copy()
wins = rank1[rank1["rank"] == 1]
roi = wins["odds"].sum() / len(rank1) * 100.0 if len(rank1) > 0 else 0.0
hit_rate = len(wins) / len(rank1) * 100.0 if len(rank1) > 0 else 0.0
top3 = df_test[df_test["pred_rank"] <= 3]
top3_wins = top3[top3["rank"] == 1]
top3_roi = top3_wins["odds"].sum() / len(top3) * 100.0 if len(top3) > 0 else 0.0
top1_pop123 = (rank1["popularity"] <= 3).mean() * 100.0 if "popularity" in rank1.columns else float("nan")

print("\nRank 1 Stats:")
print(rank1[["odds", "rank"]].describe())
print(f"Top 1 ROI: {roi:.2f}% (Hit Rate: {hit_rate:.1f}%)")
print(f"Top 3 Win ROI: {top3_roi:.2f}%")
if top1_pop123 == top1_pop123:
    print(f"Top1 predicted horses in popularity 1-3: {top1_pop123:.1f}%")

os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")
pd.DataFrame({"0": all_features}).to_csv(f"{OUTPUT_DIR}/features.csv", index=False)
print(f"Model saved to {OUTPUT_DIR}")
