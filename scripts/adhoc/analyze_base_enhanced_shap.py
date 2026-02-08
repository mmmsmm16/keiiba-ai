import json
import os
import sys
from typing import Dict, List, Tuple

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

# Added feature candidates in this workstream
NEW_FEATURES = {
    "nige_score",
    "race_nige_pressure_score_sum",
    "race_nige_count_weighted",
    "nige_score_interaction",
    "nicks_ctx_count",
    "nicks_ctx_win_rate_hier",
    "nicks_ctx_top3_rate_hier",
    "sire_top3_std_50",
    "sire_top3_iqr_50",
    "sire_course_win_rate_decay",
    "sire_dist_win_rate_decay",
    "sire_surface_win_rate_decay",
    "sire_win_rate_std_50",
    "sire_win_rate_iqr_50",
    "sire_variant_pref",
    "blood_variant_fit",
    "pace_diff_z_ctx",
    "horse_pace_elasticity_20",
    "pace_high_prob",
    "pace_fit_expected",
    "front_congestion_idx",
    "track_variant_robust",
    "track_variant_uncertainty",
    "track_variant_confidence",
    "rest_success_rate_smoothed",
    "rest_success_rate_decay",
    "rest_optimality_score",
    "rotation_stress",
    "horse_variant_pref",
    "horse_variant_match",
    "going_shift",
    "going_shift_up_top3_rate",
    "going_shift_down_top3_rate",
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


def _sample_df(df: pd.DataFrame, max_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _predict_contrib(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "booster_") and model.booster_ is not None:
        return model.booster_.predict(X, pred_contrib=True)
    return model.predict(X, pred_contrib=True)


def _mean_abs_contrib(contrib: np.ndarray) -> np.ndarray:
    # Last column is expected value (bias term)
    return np.abs(contrib[:, :-1]).mean(axis=0)


def _feature_rank_table(features: List[str], mean_abs: np.ndarray) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": features, "mean_abs_contrib": mean_abs})
    imp = imp.sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def _new_feature_summary(imp_df: pd.DataFrame, top_k: int = 15) -> List[Dict]:
    sub = imp_df[imp_df["feature"].isin(NEW_FEATURES)].copy()
    sub = sub.sort_values("mean_abs_contrib", ascending=False).head(top_k)
    return sub.to_dict(orient="records")


def _direction_stats(X: pd.DataFrame, contrib: np.ndarray, imp_df: pd.DataFrame, k: int = 8) -> List[Dict]:
    out: List[Dict] = []
    cand = imp_df[imp_df["feature"].isin(NEW_FEATURES)].head(k)["feature"].tolist()
    for f in cand:
        idx = X.columns.get_loc(f)
        x = X[f].values
        s = contrib[:, idx]
        if np.std(x) < 1e-12 or np.std(s) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, s)[0, 1])
        q = pd.qcut(pd.Series(x), q=4, duplicates="drop")
        q_mean = pd.DataFrame({"q": q.astype(str), "s": s}).groupby("q", observed=False)["s"].mean().to_dict()
        out.append(
            {
                "feature": f,
                "corr_feature_vs_contrib": corr,
                "quartile_mean_contrib": q_mean,
            }
        )
    return out


def prepare_v13_test() -> pd.DataFrame:
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
    return df[df["year"] == 2024].copy()


def prepare_v14_test() -> Tuple[pd.DataFrame, Dict]:
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
        odds = compute_odds_fluctuation(df)
        if not odds.empty:
            df["race_id"] = df["race_id"].astype(str)
            df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").fillna(0).astype(int)
            odds = odds.drop_duplicates(subset=["race_id", "horse_number"])
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
            df = pd.merge(df, odds.drop(columns=["horse_id"], errors="ignore"), on=["race_id", "horse_number"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["odds_10min"]).copy()
    cutoff = df["date"].max() - pd.Timedelta(days=90)
    test_df = df[df["date"] >= cutoff].copy()
    meta = {
        "test_start": str(cutoff.date()),
        "test_end": str(df["date"].max().date()),
        "rows": int(len(test_df)),
        "races": int(test_df["race_id"].nunique()),
    }
    return test_df, meta


def analyze_family(name: str, model_conf: Dict[str, Dict[str, str]], test_df: pd.DataFrame) -> Dict:
    out: Dict = {}
    for prof, conf in model_conf.items():
        model = joblib.load(conf["model"])
        feats = _load_feature_list(conf["features"])
        X_all = _to_numeric_matrix(test_df.reindex(columns=feats, fill_value=0.0))
        X = _sample_df(X_all, max_rows=12000, seed=42)
        contrib = _predict_contrib(model, X)
        mean_abs = _mean_abs_contrib(contrib)
        imp_df = _feature_rank_table(list(X.columns), mean_abs)

        out[prof] = {
            "n_rows_used": int(len(X)),
            "top20_mean_abs_contrib": imp_df.head(20).to_dict(orient="records"),
            "new_feature_summary": _new_feature_summary(imp_df, top_k=20),
            "new_feature_direction": _direction_stats(X, contrib, imp_df, k=8),
        }
    return out


def main() -> None:
    v13_test = prepare_v13_test()
    v14_test, v14_split = prepare_v14_test()

    result = {
        "v13": analyze_family("v13", V13_MODELS, v13_test),
        "v14": {
            "split": v14_split,
            "analysis": analyze_family("v14", V14_MODELS, v14_test),
        },
    }

    os.makedirs("reports", exist_ok=True)
    out_path = "reports/base_vs_enhanced_shap_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
