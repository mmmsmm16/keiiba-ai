import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

sys.path.append("/workspace")

from src.preprocessing.features.odds_fluctuation import compute_odds_fluctuation
from src.preprocessing.loader import JraVanDataLoader


V13_DATA_PATH = "data/processed/preprocessed_data_v12.parquet"
V14_DATA_PATH = "data/processed/preprocessed_data_v13_active.parquet"

V13_MODELS = {
    "BASE": {
        "model": "models/experiments/exp_lambdarank_hard_weighted/model.pkl",
        "features": "models/experiments/exp_lambdarank_hard_weighted/features.csv",
    },
    "ENHANCED": {
        "model": "models/experiments/exp_lambdarank_hard_weighted_enhanced/model.pkl",
        "features": "models/experiments/exp_lambdarank_hard_weighted_enhanced/features.csv",
    },
}

V14_MODELS = {
    "BASE": {
        "model": "models/experiments/exp_gap_v14_production/model_v14.pkl",
        "features": "models/experiments/exp_gap_v14_production/features.csv",
    },
    "ENHANCED": {
        "model": "models/experiments/exp_gap_v14_production_enhanced/model_v14.pkl",
        "features": "models/experiments/exp_gap_v14_production_enhanced/features.csv",
    },
}


def _load_feature_list(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "feature" in df.columns:
        return df["feature"].tolist()
    if "0" in df.columns:
        return df["0"].tolist()
    return df.iloc[:, 0].tolist()


def _to_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype.name == "category":
            out[col] = out[col].cat.codes
        elif out[col].dtype == "object":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.fillna(0.0).astype(float)


def _eval_top_metrics(df_pred: pd.DataFrame) -> Dict[str, float]:
    top1 = df_pred[df_pred["pred_rank"] == 1].copy()
    top3 = df_pred[df_pred["pred_rank"] <= 3].copy()
    races = int(top1["race_id"].nunique())
    top1_hit = float((top1["rank"] == 1).mean()) if len(top1) > 0 else 0.0
    top1_roi = float(top1.loc[top1["rank"] == 1, "odds"].sum() / len(top1) * 100.0) if len(top1) > 0 else 0.0
    top3_win_roi = float(top3.loc[top3["rank"] == 1, "odds"].sum() / len(top3) * 100.0) if len(top3) > 0 else 0.0
    out = {
        "races": races,
        "top1_hit_rate": top1_hit,
        "top1_win_roi": top1_roi,
        "top3_win_roi": top3_win_roi,
    }
    if "popularity" in top1.columns:
        out["top1_pop3_rate"] = float((pd.to_numeric(top1["popularity"], errors="coerce") <= 3).mean())
    if "odds" in top1.columns:
        out["top1_avg_odds"] = float(pd.to_numeric(top1["odds"], errors="coerce").mean())
    if "odds_10min" in top1.columns:
        out["top1_avg_odds_10min"] = float(pd.to_numeric(top1["odds_10min"], errors="coerce").mean())
    return out


def evaluate_v13() -> Dict[str, Dict[str, float]]:
    df = pd.read_parquet(V13_DATA_PATH).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
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

    test_df = df[df["year"] == 2024].copy()
    results: Dict[str, Dict[str, float]] = {}

    for name, conf in V13_MODELS.items():
        model = joblib.load(conf["model"])
        feats = _load_feature_list(conf["features"])
        X = _to_numeric_matrix(test_df.reindex(columns=feats, fill_value=0.0))
        score = model.predict(X.values)

        keep = ["race_id", "horse_number", "rank", "odds"]
        if "popularity" in test_df.columns:
            keep.append("popularity")
        out = test_df[keep].copy()
        out["pred_score"] = score
        out["pred_rank"] = out.groupby("race_id")["pred_score"].rank(ascending=False, method="first")
        results[name] = _eval_top_metrics(out)

    return results


def _prepare_v14_data() -> pd.DataFrame:
    df = pd.read_parquet(V14_DATA_PATH).copy()
    need_compute = ("odds_10min" not in df.columns) or df["odds_10min"].isna().all()

    if need_compute:
        if "start_time_str" not in df.columns:
            loader = JraVanDataLoader()
            dates = df["date"].astype(str).unique()
            df_info = loader.load(
                history_start_date=min(dates),
                end_date=max(dates),
                skip_odds=True,
                skip_training=True,
            )
            df["race_id"] = df["race_id"].astype(str)
            df_info["race_id"] = df_info["race_id"].astype(str)
            if "start_time_str" in df_info.columns:
                df = pd.merge(
                    df,
                    df_info[["race_id", "start_time_str"]].drop_duplicates(),
                    on="race_id",
                    how="left",
                )

        df_odds = compute_odds_fluctuation(df)
        if not df_odds.empty:
            df["race_id"] = df["race_id"].astype(str)
            df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0).astype(int)
            df_odds = df_odds.drop_duplicates(subset=["race_id", "horse_number"])
            for c in [
                "odds_10min",
                "odds_final",
                "odds_60min",
                "odds_ratio_10min",
                "rank_diff_10min",
                "odds_log_ratio_10min",
                "odds_ratio_60_10",
            ]:
                if c in df.columns:
                    df = df.drop(columns=[c])
            df = pd.merge(
                df,
                df_odds.drop(columns=["horse_id"], errors="ignore"),
                on=["race_id", "horse_number"],
                how="left",
            )

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["odds_10min"]).copy()
    df["odds_10min"] = pd.to_numeric(df["odds_10min"], errors="coerce")
    df = df.dropna(subset=["odds_10min"]).copy()
    df["odds_rank_10min"] = df.groupby("race_id")["odds_10min"].rank(method="min")
    df["gap_score"] = df["odds_rank_10min"] - pd.to_numeric(df["rank"], errors="coerce")
    return df


def evaluate_v14() -> Dict[str, Dict[str, float]]:
    df = _prepare_v14_data()
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    test_df = df[df["date"] >= cutoff].copy()

    results: Dict[str, Dict[str, float]] = {
        "_split": {
            "test_start": str(cutoff.date()),
            "test_end": str(df["date"].max().date()),
            "rows": int(len(test_df)),
            "races": int(test_df["race_id"].nunique()),
        }
    }

    for name, conf in V14_MODELS.items():
        model = joblib.load(conf["model"])
        feats = _load_feature_list(conf["features"])
        X = _to_numeric_matrix(test_df.reindex(columns=feats, fill_value=0.0))
        pred = model.predict(X)

        keep = ["race_id", "horse_number", "rank", "odds", "odds_10min", "gap_score"]
        if "popularity" in test_df.columns:
            keep.append("popularity")
        out = test_df[keep].copy()
        out["pred_gap"] = pred
        out["pred_rank"] = out.groupby("race_id")["pred_gap"].rank(ascending=False, method="first")

        top_metrics = _eval_top_metrics(out)
        target = out[(out["pred_rank"] <= 3) & (out["odds_10min"] >= 10.0) & (out["odds_10min"] <= 50.0)]
        place_hit_rate = float((target["rank"] <= 3).mean()) if len(target) > 0 else 0.0
        rmse = float(np.sqrt(np.mean((out["pred_gap"] - out["gap_score"]) ** 2)))

        top_metrics.update(
            {
                "target_picks": int(len(target)),
                "target_place_hit_rate": place_hit_rate,
                "rmse_gap_score": rmse,
            }
        )
        results[name] = top_metrics

    return results


def _delta(enh: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in enh.items():
        if k in base and isinstance(v, (int, float)) and isinstance(base[k], (int, float)):
            out[k] = float(v - base[k])
    return out


def main() -> None:
    v13 = evaluate_v13()
    v14 = evaluate_v14()
    result = {
        "v13": {
            "BASE": v13["BASE"],
            "ENHANCED": v13["ENHANCED"],
            "DELTA_ENH_MINUS_BASE": _delta(v13["ENHANCED"], v13["BASE"]),
        },
        "v14": {
            "split": v14["_split"],
            "BASE": v14["BASE"],
            "ENHANCED": v14["ENHANCED"],
            "DELTA_ENH_MINUS_BASE": _delta(v14["ENHANCED"], v14["BASE"]),
        },
    }

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/base_vs_enhanced_v13_v14_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
