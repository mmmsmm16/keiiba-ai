import json
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

sys.path.append("/workspace")

from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
from src.preprocessing.loader import JraVanDataLoader


def _to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype.name == "category":
            X[c] = X[c].cat.codes
        elif X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0).astype(float)


def _eval_top1(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    t = df.copy()
    t["pred_rank"] = t.groupby("race_id")[score_col].rank(ascending=False, method="first")
    top1 = t[t["pred_rank"] == 1].copy()
    wins = top1[top1["rank"] == 1]
    out = {
        "races": int(top1["race_id"].nunique()),
        "hit": float((top1["rank"] == 1).mean()),
        "roi": float(wins["odds"].sum() / len(top1) * 100.0 if len(top1) > 0 else 0.0),
        "avg_odds": float(top1["odds"].mean()) if "odds" in top1.columns else np.nan,
    }
    if "popularity" in top1.columns:
        out["pop3"] = float((pd.to_numeric(top1["popularity"], errors="coerce") <= 3).mean())
    return out


def _weights() -> List[float]:
    return [i / 20.0 for i in range(21)]


def _prepare_v13() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/preprocessed_data_v12.parquet").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == 2024].copy()
    df = df[(pd.to_numeric(df["odds"], errors="coerce") > 0) & (pd.to_numeric(df["odds"], errors="coerce").notna())]

    if "odds_10min" in df.columns and not df["odds_10min"].isna().all():
        df["odds_feature"] = pd.to_numeric(df["odds_10min"], errors="coerce").fillna(pd.to_numeric(df["odds"], errors="coerce"))
    else:
        df["odds_feature"] = pd.to_numeric(df["odds"], errors="coerce")
    df["odds_rank_pre"] = df.groupby("race_id")["odds_feature"].rank(ascending=True, method="min")
    if "relative_horse_elo_z" in df.columns:
        df["elo_rank"] = df.groupby("race_id")["relative_horse_elo_z"].rank(ascending=False, method="min")
        df["odds_rank_vs_elo"] = df["odds_rank_pre"] - df["elo_rank"]
    else:
        df["odds_rank_vs_elo"] = 0.0
    df["is_high_odds"] = (df["odds_feature"] >= 10).astype(int)
    df["is_mid_odds"] = ((df["odds_feature"] >= 5) & (df["odds_feature"] < 10)).astype(int)

    m_base = joblib.load("models/experiments/exp_lambdarank_hard_weighted/model.pkl")
    f_base = pd.read_csv("models/experiments/exp_lambdarank_hard_weighted/features.csv")["0"].tolist()
    X_base = _to_numeric(df.reindex(columns=f_base, fill_value=0.0))
    df["s_base"] = m_base.predict(X_base.values)

    m_enh = joblib.load("models/experiments/exp_lambdarank_hard_weighted_enhanced/model.pkl")
    f_enh = pd.read_csv("models/experiments/exp_lambdarank_hard_weighted_enhanced/features.csv")["0"].tolist()
    X_enh = _to_numeric(df.reindex(columns=f_enh, fill_value=0.0))
    df["s_enh"] = m_enh.predict(X_enh.values)
    return df


def _prepare_v14() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/preprocessed_data_v13_active.parquet").copy()
    need_compute = ("odds_10min" not in df.columns) or df["odds_10min"].isna().all()

    if need_compute:
        if "start_time_str" not in df.columns:
            loader = JraVanDataLoader()
            dates = df["date"].astype(str).unique()
            info = loader.load(
                history_start_date=min(dates),
                end_date=max(dates),
                skip_odds=True,
                skip_training=True,
            )
            info["race_id"] = info["race_id"].astype(str)
            df["race_id"] = df["race_id"].astype(str)
            if "start_time_str" in info.columns:
                df = df.merge(info[["race_id", "start_time_str"]].drop_duplicates(), on="race_id", how="left")

        odds = compute_odds_fluctuation(df)
        if not odds.empty:
            df["race_id"] = df["race_id"].astype(str)
            df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0).astype(int)
            odds = odds.drop_duplicates(subset=["race_id", "horse_number"])
            for c in ["odds_10min", "odds_final", "odds_60min", "odds_ratio_10min", "rank_diff_10min", "odds_log_ratio_10min", "odds_ratio_60_10"]:
                if c in df.columns:
                    df = df.drop(columns=[c])
            df = df.merge(odds.drop(columns=["horse_id"], errors="ignore"), on=["race_id", "horse_number"], how="left")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["odds_10min"]).copy()
    df["odds_10min"] = pd.to_numeric(df["odds_10min"], errors="coerce")
    df = df.dropna(subset=["odds_10min"]).copy()
    df["odds_rank_10min"] = df.groupby("race_id")["odds_10min"].rank(method="min")
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    df = df[df["date"] >= cutoff].copy()

    m_base = joblib.load("models/experiments/exp_gap_v14_production/model_v14.pkl")
    f_base = pd.read_csv("models/experiments/exp_gap_v14_production/features.csv")["feature"].tolist()
    X_base = _to_numeric(df.reindex(columns=f_base, fill_value=0.0))
    df["s_base"] = m_base.predict(X_base)

    m_enh = joblib.load("models/experiments/exp_gap_v14_production_enhanced/model_v14.pkl")
    f_enh = pd.read_csv("models/experiments/exp_gap_v14_production_enhanced/features.csv")["feature"].tolist()
    X_enh = _to_numeric(df.reindex(columns=f_enh, fill_value=0.0))
    df["s_enh"] = m_enh.predict(X_enh)
    return df


def _scan(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in _weights():
        d = df.copy()
        d["s_mix"] = w * d["s_base"] + (1.0 - w) * d["s_enh"]
        m = _eval_top1(d, "s_mix")
        m["w_base"] = w
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    v13 = _scan(_prepare_v13())
    v14 = _scan(_prepare_v14())

    out = {
        "v13_all": v13.sort_values(["roi", "hit"], ascending=False).to_dict(orient="records"),
        "v13_practical": v13[(v13["hit"] >= 0.29) & (v13["pop3"] <= 0.97)].sort_values(["roi", "hit"], ascending=False).to_dict(orient="records"),
        "v14_all": v14.sort_values(["hit", "roi"], ascending=False).to_dict(orient="records"),
        "v14_practical": v14[v14["hit"] >= 0.06].sort_values(["roi", "hit"], ascending=False).to_dict(orient="records"),
    }

    with open("reports/blend_weight_scan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved reports/blend_weight_scan.json")
    print("\nv13 practical top5:")
    print(v13[(v13["hit"] >= 0.29) & (v13["pop3"] <= 0.97)].sort_values(["roi", "hit"], ascending=False).head(5).to_string(index=False))
    print("\nv14 practical top5:")
    print(v14[v14["hit"] >= 0.06].sort_values(["roi", "hit"], ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
