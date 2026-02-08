"""
Exp v15: HIT/ROI Max Search
===========================

Goal:
- Explore model/feature/strategy combinations.
- Maximize ROI while keeping moderate HIT and minimum bet volume.
- Train final best model and output SHAP summary.
"""

from __future__ import annotations

import argparse
import json
import os
import gc
from datetime import datetime
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


DEFAULT_DATA_PATH = "data/processed/preprocessed_data_v12.parquet"
DEFAULT_BASE_FEATURES_PATH = "models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted/features.csv"
DEFAULT_OUTPUT_DIR = "models/experiments/exp_v15_hit_roi_max"
DEFAULT_REPORT_PATH = "reports/exp_v15_hit_roi_max.json"


A_BLOODLINE_FEATURES = [
    "sire_variant_pref",
    "blood_variant_fit",
    "sire_course_win_rate_decay",
    "sire_dist_win_rate_decay",
    "sire_surface_win_rate_decay",
    "sire_win_rate_std_50",
    "sire_win_rate_iqr_50",
    "horse_variant_pref",
    "horse_variant_match",
]

B_PACE_FEATURES = [
    "pace_diff_z_ctx",
    "horse_pace_elasticity_20",
    "pace_high_prob",
    "pace_fit_expected",
    "front_congestion_idx",
    "nige_score",
    "nige_score_interaction",
    "race_nige_pressure_score_sum",
    "race_nige_count_weighted",
]

C_TRACK_REST_FEATURES = [
    "track_variant_robust",
    "track_variant_uncertainty",
    "track_variant_confidence",
    "going_shift",
    "going_shift_up_top3_rate",
    "going_shift_down_top3_rate",
    "rest_success_rate_smoothed",
    "rest_success_rate_decay",
    "rest_optimality_score",
    "rotation_stress",
]

DERIVED_MARKET_FEATURES = [
    "market_prob",
    "log_odds",
    "odds_rank_pre",
    "is_mid_odds",
    "is_high_odds",
]


@dataclass
class CandidateSpec:
    name: str
    family: str  # binary / ranker
    feature_mode: str  # NO_MARKET / LIGHT_MARKET / BALANCED_MARKET
    weight_mode: str  # none / value


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def make_iter_logger(label: str, period: int = 50):
    def _callback(env):
        iteration = env.iteration + 1
        end_iter = env.end_iteration
        if iteration == 1 or iteration % period == 0 or iteration == end_iter:
            if env.evaluation_result_list:
                parts = []
                for data_name, eval_name, result, _ in env.evaluation_result_list:
                    parts.append(f"{data_name}.{eval_name}={result:.5f}")
                eval_text = ", ".join(parts)
            else:
                eval_text = "no_eval"
            log(f"[{label}] iter {iteration}/{end_iter} {eval_text}")

    _callback.order = 20
    return _callback


def _coalesce_columns(df: pd.DataFrame, target: str, candidates: List[str]) -> pd.DataFrame:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return df
    s = pd.to_numeric(df[cols[0]], errors="coerce")
    for c in cols[1:]:
        s = s.combine_first(pd.to_numeric(df[c], errors="coerce"))
    df[target] = s
    for c in cols:
        if c != target and c in df.columns:
            df = df.drop(columns=[c], errors="ignore")
    return df


def normalize_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "odds_10min": ["odds_10min", "odds_10min_x", "odds_10min_y"],
        "odds_60min": ["odds_60min", "odds_60min_x", "odds_60min_y"],
        "odds_final": ["odds_final", "odds_final_x", "odds_final_y"],
        "odds_ratio_10min": ["odds_ratio_10min", "odds_ratio_10min_x", "odds_ratio_10min_y"],
        "odds_ratio_60_10": ["odds_ratio_60_10", "odds_ratio_60_10_x", "odds_ratio_60_10_y"],
        "rank_diff_10min": ["rank_diff_10min", "rank_diff_10min_x", "rank_diff_10min_y"],
        "odds_log_ratio_10min": ["odds_log_ratio_10min", "odds_log_ratio_10min_x", "odds_log_ratio_10min_y"],
    }
    for target, candidates in mapping.items():
        df = _coalesce_columns(df, target, candidates)
    return df


def read_dataset_minimal(data_path: str, base_features_path: str) -> pd.DataFrame:
    base = pd.read_csv(base_features_path).iloc[:, 0].astype(str).tolist()
    required = set(
        [
            "race_id",
            "horse_number",
            "date",
            "rank",
            "odds",
            "popularity",
            "odds_10min",
            "odds_10min_x",
            "odds_10min_y",
            "odds_60min",
            "odds_60min_x",
            "odds_60min_y",
            "odds_final",
            "odds_final_x",
            "odds_final_y",
        ]
        + base
        + A_BLOODLINE_FEATURES
        + B_PACE_FEATURES
        + C_TRACK_REST_FEATURES
    )

    try:
        import pyarrow.parquet as pq

        all_cols = set(pq.ParquetFile(data_path).schema.names)
        use_cols = sorted(c for c in required if c in all_cols)
        if use_cols:
            log(f"Using {len(use_cols)} columns (minimal read).")
            return pd.read_parquet(data_path, columns=use_cols)
    except Exception as e:
        log(f"Minimal read fallback to full read due to: {e}")

    return pd.read_parquet(data_path)


def to_numeric_matrix(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    x = df.reindex(columns=features, fill_value=0).copy()
    for col in x.columns:
        s = x[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            x[col] = s.cat.codes
        elif s.dtype == "object":
            x[col] = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32, copy=False)


def race_softmax_prob(race_ids: pd.Series, scores: np.ndarray) -> np.ndarray:
    s = pd.Series(scores, index=race_ids.index, dtype=float)
    grp = race_ids.astype(str)
    max_per_race = s.groupby(grp).transform("max")
    exp_s = np.exp((s - max_per_race).clip(-30, 30))
    sum_exp = exp_s.groupby(grp).transform("sum")
    prob = (exp_s / sum_exp).fillna(0.0)
    return prob.to_numpy(dtype=float)


def race_normalize_prob(race_ids: pd.Series, probs: np.ndarray) -> np.ndarray:
    p = pd.Series(probs, index=race_ids.index, dtype=float).clip(lower=1e-8)
    grp = race_ids.astype(str)
    s = p.groupby(grp).transform("sum")
    norm = (p / s).fillna(0.0)
    return norm.to_numpy(dtype=float)


def build_top1_table(pred_df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    work = pred_df[["race_id", "rank", "odds", "pred_prob", "popularity"]].copy()
    work["value_score"] = work["pred_prob"] * np.power(work["odds"].clip(lower=1.01), gamma)
    idx = work.groupby("race_id")["value_score"].idxmax()
    top = work.loc[idx].copy()
    top["ev"] = top["pred_prob"] * top["odds"]
    return top


def evaluate_top1_with_filter(
    top_df: pd.DataFrame,
    min_odds: float,
    max_odds: float,
    ev_thr: float,
    prob_floor: float,
    min_hit: float,
    min_coverage: float,
    min_bets: int,
) -> Dict[str, float]:
    races = int(top_df["race_id"].nunique())
    m = (
        (top_df["odds"] >= min_odds)
        & (top_df["odds"] <= max_odds)
        & (top_df["ev"] >= ev_thr)
        & (top_df["pred_prob"] >= prob_floor)
    )
    bets = top_df[m].copy()
    n_bets = int(len(bets))
    wins = bets[bets["rank"] == 1]
    n_wins = int(len(wins))

    hit = float(n_wins / n_bets) if n_bets > 0 else 0.0
    roi = float(wins["odds"].sum() / n_bets * 100.0) if n_bets > 0 else 0.0
    coverage = float(n_bets / races) if races > 0 else 0.0
    pop3 = float((pd.to_numeric(bets["popularity"], errors="coerce") <= 3).mean()) if n_bets > 0 else 0.0
    avg_odds = float(pd.to_numeric(bets["odds"], errors="coerce").mean()) if n_bets > 0 else 0.0

    # Strong penalties keep strategy from collapsing into tiny-bet lucky pockets.
    penalty_hit = max(0.0, min_hit - hit) * 600.0
    penalty_cov = max(0.0, min_coverage - coverage) * 1800.0
    penalty_bets = max(0.0, float(min_bets - n_bets)) * 1.1
    utility = roi - penalty_hit - penalty_cov - penalty_bets

    return {
        "races": races,
        "bets": n_bets,
        "wins": n_wins,
        "hit_rate": hit,
        "roi": roi,
        "coverage": coverage,
        "pop3_rate": pop3,
        "avg_odds": avg_odds,
        "utility": float(utility),
    }


def search_best_strategy(
    pred_df: pd.DataFrame,
    min_hit: float,
    min_coverage: float,
    min_bets: int,
    progress_label: str = "",
    progress_every: int = 300,
) -> Dict[str, float]:
    gammas = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75]
    ev_thrs = [0.00, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50]
    min_odds_list = [1.0, 2.0, 3.0, 5.0]
    max_odds_list = [20.0, 30.0, 50.0, 80.0]
    prob_floors = [0.00, 0.02, 0.03, 0.05]

    best_any: Dict[str, float] | None = None
    best_feasible: Dict[str, float] | None = None
    total_steps = (
        len(gammas)
        * len(min_odds_list)
        * len(max_odds_list)
        * len(ev_thrs)
        * len(prob_floors)
    )
    step = 0
    label = progress_label.strip() if progress_label else "strategy"
    log(f"[{label}] strategy search start: {total_steps} combinations")

    for gamma in gammas:
        top = build_top1_table(pred_df, gamma=gamma)
        for min_odds in min_odds_list:
            for max_odds in max_odds_list:
                if min_odds >= max_odds:
                    continue
                for ev_thr in ev_thrs:
                    for prob_floor in prob_floors:
                        step += 1
                        metrics = evaluate_top1_with_filter(
                            top,
                            min_odds=min_odds,
                            max_odds=max_odds,
                            ev_thr=ev_thr,
                            prob_floor=prob_floor,
                            min_hit=min_hit,
                            min_coverage=min_coverage,
                            min_bets=min_bets,
                        )
                        candidate = {
                            "gamma": gamma,
                            "min_odds": min_odds,
                            "max_odds": max_odds,
                            "ev_thr": ev_thr,
                            "prob_floor": prob_floor,
                            **metrics,
                        }
                        if best_any is None or candidate["utility"] > best_any["utility"]:
                            best_any = candidate
                        feasible = (
                            candidate["bets"] >= min_bets
                            and candidate["hit_rate"] >= min_hit
                            and candidate["coverage"] >= min_coverage
                        )
                        if feasible and (best_feasible is None or candidate["roi"] > best_feasible["roi"]):
                            best_feasible = candidate
                            log(
                                f"[{label}] new feasible best roi={best_feasible['roi']:.2f} "
                                f"hit={best_feasible['hit_rate']:.3f} bets={best_feasible['bets']} "
                                f"(step {step}/{total_steps})"
                            )
                        if step % progress_every == 0 or step == total_steps:
                            pct = (step / total_steps) * 100.0
                            current = best_feasible if best_feasible is not None else best_any
                            best_u = current["utility"] if current is not None else float("nan")
                            log(f"[{label}] strategy progress {step}/{total_steps} ({pct:.1f}%), best_utility={best_u:.2f}")
    return (best_feasible or best_any or {})


def apply_strategy(pred_df: pd.DataFrame, strategy: Dict[str, float]) -> Dict[str, float]:
    top = build_top1_table(pred_df, gamma=float(strategy["gamma"]))
    metrics = evaluate_top1_with_filter(
        top,
        min_odds=float(strategy["min_odds"]),
        max_odds=float(strategy["max_odds"]),
        ev_thr=float(strategy["ev_thr"]),
        prob_floor=float(strategy["prob_floor"]),
        min_hit=float(strategy.get("min_hit_constraint", 0.0)),
        min_coverage=float(strategy.get("min_coverage_constraint", 0.0)),
        min_bets=int(strategy.get("min_bets_constraint", 0)),
    )
    return metrics


def make_feature_modes(df: pd.DataFrame, base_features_path: str) -> Dict[str, List[str]]:
    base = pd.read_csv(base_features_path).iloc[:, 0].astype(str).tolist()
    base = [f for f in base if f in df.columns]
    enhanced = [f for f in (A_BLOODLINE_FEATURES + B_PACE_FEATURES + C_TRACK_REST_FEATURES) if f in df.columns]

    modes = {
        "NO_MARKET": sorted(set(base + enhanced)),
        "LIGHT_MARKET": sorted(set(base + enhanced + ["market_prob", "log_odds", "odds_rank_pre"])),
        "BALANCED_MARKET": sorted(set(base + enhanced + DERIVED_MARKET_FEATURES + ["popularity"])),
    }
    return modes


def build_candidates() -> List[CandidateSpec]:
    out: List[CandidateSpec] = []
    for mode in ["NO_MARKET", "LIGHT_MARKET", "BALANCED_MARKET"]:
        out.append(CandidateSpec(name=f"bin_{mode.lower()}_plain", family="binary", feature_mode=mode, weight_mode="none"))
        out.append(CandidateSpec(name=f"bin_{mode.lower()}_valuew", family="binary", feature_mode=mode, weight_mode="value"))
        out.append(CandidateSpec(name=f"rank_{mode.lower()}", family="ranker", feature_mode=mode, weight_mode="none"))
    return out


def train_and_predict_candidate(
    spec: CandidateSpec,
    features: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    progress_label: str,
) -> Tuple[object, np.ndarray, np.ndarray]:
    x_train = to_numeric_matrix(train_df, features)
    x_valid = to_numeric_matrix(valid_df, features)

    if spec.family == "binary":
        y_train = (pd.to_numeric(train_df["rank"], errors="coerce") == 1).astype(int).to_numpy()
        y_valid = (pd.to_numeric(valid_df["rank"], errors="coerce") == 1).astype(int).to_numpy()

        sample_weight = None
        if spec.weight_mode == "value":
            odds_train = pd.to_numeric(train_df["odds"], errors="coerce").fillna(10.0).clip(lower=1.01, upper=100.0)
            sample_weight = (1.0 + y_train * np.log1p(odds_train)).to_numpy(dtype=float)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 1400,
            "learning_rate": 0.03,
            "num_leaves": 95 if spec.feature_mode != "NO_MARKET" else 63,
            "min_child_samples": 80,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.3,
            "reg_lambda": 1.0,
            "scale_pos_weight": 14.0,
            "random_state": seed,
            "n_jobs": 4,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[make_iter_logger(progress_label, period=50), lgb.early_stopping(80)],
        )

        valid_raw = model.predict_proba(x_valid)[:, 1]
        valid_prob = race_normalize_prob(valid_df["race_id"], valid_raw)

        del x_train, x_valid
        gc.collect()

        x_test = to_numeric_matrix(test_df, features)
        test_raw = model.predict_proba(x_test)[:, 1]
        test_prob = race_normalize_prob(test_df["race_id"], test_raw)
        del x_test
        gc.collect()
        return model, valid_prob, test_prob

    y_train = np.clip(4 - pd.to_numeric(train_df["rank"], errors="coerce").fillna(99).astype(int), 0, 3).to_numpy()
    y_valid = np.clip(4 - pd.to_numeric(valid_df["rank"], errors="coerce").fillna(99).astype(int), 0, 3).to_numpy()
    grp_train = train_df.groupby("race_id", sort=False).size().to_numpy()
    grp_valid = valid_df.groupby("race_id", sort=False).size().to_numpy()

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 80,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": 4,
        "verbosity": -1,
    }
    model = lgb.LGBMRanker(**params)
    model.fit(
        x_train,
        y_train,
        group=grp_train,
        eval_set=[(x_valid, y_valid)],
        eval_group=[grp_valid],
        eval_at=[1, 3, 5],
        callbacks=[make_iter_logger(progress_label, period=50), lgb.early_stopping(80)],
    )

    valid_score = model.predict(x_valid)
    valid_prob = race_softmax_prob(valid_df["race_id"], valid_score)
    del x_train, x_valid
    gc.collect()

    x_test = to_numeric_matrix(test_df, features)
    test_score = model.predict(x_test)
    test_prob = race_softmax_prob(test_df["race_id"], test_score)
    del x_test
    gc.collect()
    return model, valid_prob, test_prob


def retrain_best_model(
    best_spec: CandidateSpec,
    features: List[str],
    train_valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    seed: int,
    progress_label: str = "retrain",
) -> Tuple[object, np.ndarray, np.ndarray]:
    x_train = to_numeric_matrix(train_valid_df, features)

    if best_spec.family == "binary":
        y_train = (pd.to_numeric(train_valid_df["rank"], errors="coerce") == 1).astype(int).to_numpy()
        sample_weight = None
        if best_spec.weight_mode == "value":
            odds_train = pd.to_numeric(train_valid_df["odds"], errors="coerce").fillna(10.0).clip(lower=1.01, upper=100.0)
            sample_weight = (1.0 + y_train * np.log1p(odds_train)).to_numpy(dtype=float)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "num_leaves": 95 if best_spec.feature_mode != "NO_MARKET" else 63,
            "min_child_samples": 80,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.3,
            "reg_lambda": 1.0,
            "scale_pos_weight": 14.0,
            "random_state": seed,
            "n_jobs": 4,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train, sample_weight=sample_weight, callbacks=[make_iter_logger(progress_label, period=100)])

        x_test = to_numeric_matrix(test_df, features)
        test_raw = model.predict_proba(x_test)[:, 1]
        test_prob = race_normalize_prob(test_df["race_id"], test_raw)
        del x_train, x_test
        gc.collect()
        hold_prob = np.array([])
        if len(holdout_df) > 0:
            x_hold = to_numeric_matrix(holdout_df, features)
            hold_raw = model.predict_proba(x_hold)[:, 1]
            hold_prob = race_normalize_prob(holdout_df["race_id"], hold_raw)
            del x_hold
            gc.collect()
        return model, test_prob, hold_prob

    y_train = np.clip(4 - pd.to_numeric(train_valid_df["rank"], errors="coerce").fillna(99).astype(int), 0, 3).to_numpy()
    grp_train = train_valid_df.groupby("race_id", sort=False).size().to_numpy()
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[1, 3, 5],
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x_train, y_train, group=grp_train, callbacks=[make_iter_logger(progress_label, period=100)])

    x_test = to_numeric_matrix(test_df, features)
    test_score = model.predict(x_test)
    test_prob = race_softmax_prob(test_df["race_id"], test_score)
    del x_train, x_test
    gc.collect()
    hold_prob = np.array([])
    if len(holdout_df) > 0:
        x_hold = to_numeric_matrix(holdout_df, features)
        hold_score = model.predict(x_hold)
        hold_prob = race_softmax_prob(holdout_df["race_id"], hold_score)
        del x_hold
        gc.collect()
    return model, test_prob, hold_prob


def make_pred_df(df: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "race_id": df["race_id"].astype(str).values,
            "rank": pd.to_numeric(df["rank"], errors="coerce").fillna(99).astype(int).values,
            "odds": pd.to_numeric(df["odds"], errors="coerce").fillna(0.0).clip(lower=1.01, upper=200.0).values,
            "popularity": pd.to_numeric(df.get("popularity", np.nan), errors="coerce"),
            "pred_prob": prob,
        }
    )
    return out


def compute_shap_summary(
    model: object,
    df_test: pd.DataFrame,
    features: List[str],
    output_dir: str,
    seed: int,
) -> Dict[str, object]:
    if len(df_test) == 0:
        return {"top_features": []}

    try:
        sample_n = min(1500, len(df_test))
        sample_df = df_test.sample(sample_n, random_state=seed)
        x_sample_df = pd.DataFrame(to_numeric_matrix(sample_df, features), columns=features)

        booster = model.booster_ if hasattr(model, "booster_") else model
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(x_sample_df, check_additivity=False)
        if isinstance(shap_values, list):
            shap_arr = np.asarray(shap_values[-1])
        else:
            shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, -1]

        mean_abs = np.abs(shap_arr).mean(axis=0)
        summary = (
            pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(30)
            .reset_index(drop=True)
        )
        out_csv = os.path.join(output_dir, "shap_summary_top30.csv")
        summary.to_csv(out_csv, index=False)
        return {"top_features": summary.to_dict(orient="records"), "path": out_csv}
    except Exception as e:
        return {"top_features": [], "path": None, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="HIT/ROI maximization experiment.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--base-features-path", default=DEFAULT_BASE_FEATURES_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--train-end-year", type=int, default=2023)
    parser.add_argument("--valid-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--min-hit", type=float, default=0.12)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-bets", type=int, default=300)
    parser.add_argument("--families", default="binary,ranker", help="Comma separated: binary,ranker")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    log(f"Loading data: {args.data_path}")
    df = read_dataset_minimal(args.data_path, args.base_features_path)
    df = normalize_odds_columns(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df = df[df["date"].notna()].copy()
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df[df["rank"].notna()].copy()

    if "odds_10min" in df.columns and pd.to_numeric(df["odds_10min"], errors="coerce").notna().any():
        odds_feature = pd.to_numeric(df["odds_10min"], errors="coerce")
    else:
        odds_feature = pd.to_numeric(df["odds"], errors="coerce")

    df["odds_feature"] = odds_feature.fillna(pd.to_numeric(df["odds"], errors="coerce")).clip(lower=1.01)
    df["market_prob"] = (1.0 / df["odds_feature"]).clip(upper=1.0)
    df["log_odds"] = np.log1p(df["odds_feature"])
    df["odds_rank_pre"] = df.groupby("race_id")["odds_feature"].rank(method="min", ascending=True)
    df["is_mid_odds"] = ((df["odds_feature"] >= 5.0) & (df["odds_feature"] < 15.0)).astype(int)
    df["is_high_odds"] = (df["odds_feature"] >= 15.0).astype(int)

    df = df.sort_values(["date", "race_id", "horse_number"], kind="mergesort").reset_index(drop=True)

    train_df = df[df["year"] <= args.train_end_year].copy()
    valid_df = df[df["year"] == args.valid_year].copy()
    test_df = df[df["year"] == args.test_year].copy()
    holdout_df = df[df["year"] > args.test_year].copy()

    log(f"Split rows -> train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}, holdout: {len(holdout_df)}")
    log(
        f"Split races -> train: {train_df['race_id'].nunique()}, "
        f"valid: {valid_df['race_id'].nunique()}, test: {test_df['race_id'].nunique()}, holdout: {holdout_df['race_id'].nunique()}"
    )

    if train_df.empty or valid_df.empty or test_df.empty:
        raise RuntimeError("Train/Valid/Test split is empty. Check years or data.")

    feature_modes = make_feature_modes(df, args.base_features_path)
    log("Feature counts:")
    for k, v in feature_modes.items():
        log(f"  {k}: {len(v)}")

    family_set = {s.strip() for s in str(args.families).split(",") if s.strip()}
    candidates = [c for c in build_candidates() if c.family in family_set]
    if not candidates:
        raise RuntimeError(f"No candidates for families={args.families}")
    candidate_rows: List[Dict[str, object]] = []
    best_by_valid: Dict[str, object] | None = None

    n_candidates = len(candidates)
    for i, spec in enumerate(candidates, start=1):
        feats = feature_modes[spec.feature_mode]
        log(
            f"[Candidate {i}/{n_candidates}] {spec.name} "
            f"family={spec.family} mode={spec.feature_mode} features={len(feats)}"
        )
        model, valid_prob, test_prob = train_and_predict_candidate(
            spec=spec,
            features=feats,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            seed=args.seed,
            progress_label=f"{spec.name} {i}/{n_candidates}",
        )

        valid_pred = make_pred_df(valid_df, valid_prob)
        test_pred = make_pred_df(test_df, test_prob)

        best_strategy = search_best_strategy(
            valid_pred,
            min_hit=args.min_hit,
            min_coverage=args.min_coverage,
            min_bets=args.min_bets,
            progress_label=f"{spec.name} {i}/{n_candidates}",
        )
        best_strategy["min_hit_constraint"] = args.min_hit
        best_strategy["min_coverage_constraint"] = args.min_coverage
        best_strategy["min_bets_constraint"] = args.min_bets

        valid_metrics = apply_strategy(valid_pred, best_strategy)
        test_metrics = apply_strategy(test_pred, best_strategy)

        row = {
            "candidate": spec.name,
            "family": spec.family,
            "feature_mode": spec.feature_mode,
            "weight_mode": spec.weight_mode,
            "n_features": len(feats),
            "strategy_gamma": best_strategy["gamma"],
            "strategy_ev_thr": best_strategy["ev_thr"],
            "strategy_min_odds": best_strategy["min_odds"],
            "strategy_max_odds": best_strategy["max_odds"],
            "strategy_prob_floor": best_strategy["prob_floor"],
            "valid_utility": valid_metrics["utility"],
            "valid_roi": valid_metrics["roi"],
            "valid_hit": valid_metrics["hit_rate"],
            "valid_bets": valid_metrics["bets"],
            "valid_coverage": valid_metrics["coverage"],
            "test_utility": test_metrics["utility"],
            "test_roi": test_metrics["roi"],
            "test_hit": test_metrics["hit_rate"],
            "test_bets": test_metrics["bets"],
            "test_coverage": test_metrics["coverage"],
        }
        candidate_rows.append(row)
        log(
            f"  valid ROI={valid_metrics['roi']:.2f} HIT={valid_metrics['hit_rate']:.3f} bets={valid_metrics['bets']} "
            f"| test ROI={test_metrics['roi']:.2f} HIT={test_metrics['hit_rate']:.3f} bets={test_metrics['bets']}"
        )

        progress_csv = os.path.join(args.output_dir, "candidate_results_progress.csv")
        pd.DataFrame(candidate_rows).sort_values(["valid_utility", "test_roi"], ascending=False).to_csv(progress_csv, index=False)
        log(f"  progress saved: {progress_csv} ({len(candidate_rows)}/{n_candidates})")

        if best_by_valid is None or row["valid_utility"] > best_by_valid["row"]["valid_utility"]:
            best_by_valid = {
                "row": row,
                "spec": spec,
                "features": feats,
                "strategy": best_strategy,
            }
            log(f"  candidate best updated -> {spec.name} (valid utility={row['valid_utility']:.2f})")

        del model, valid_prob, test_prob, valid_pred, test_pred
        gc.collect()

    if best_by_valid is None:
        raise RuntimeError("No valid candidate was produced.")

    candidate_df = pd.DataFrame(candidate_rows).sort_values(["valid_utility", "test_roi"], ascending=False)
    candidate_csv = os.path.join(args.output_dir, "candidate_results.csv")
    candidate_df.to_csv(candidate_csv, index=False)

    best_spec: CandidateSpec = best_by_valid["spec"]
    best_features: List[str] = best_by_valid["features"]
    best_strategy = best_by_valid["strategy"]

    log("Retraining best candidate on train+valid...")
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    final_model, final_test_prob, final_hold_prob = retrain_best_model(
        best_spec=best_spec,
        features=best_features,
        train_valid_df=train_valid_df,
        test_df=test_df,
        holdout_df=holdout_df,
        seed=args.seed,
        progress_label="retrain_best",
    )

    final_test_pred = make_pred_df(test_df, final_test_prob)
    final_test_metrics = apply_strategy(final_test_pred, best_strategy)
    final_hold_metrics = None
    if len(holdout_df) > 0:
        final_hold_pred = make_pred_df(holdout_df, final_hold_prob)
        final_hold_metrics = apply_strategy(final_hold_pred, best_strategy)

    model_path = os.path.join(args.output_dir, "model.pkl")
    features_path = os.path.join(args.output_dir, "features.csv")
    strategy_path = os.path.join(args.output_dir, "best_strategy.json")
    summary_path = os.path.join(args.output_dir, "summary.json")

    joblib.dump(final_model, model_path)
    pd.DataFrame({"feature": best_features}).to_csv(features_path, index=False)
    with open(strategy_path, "w", encoding="utf-8") as f:
        json.dump(best_strategy, f, ensure_ascii=False, indent=2)

    shap_summary = compute_shap_summary(
        model=final_model,
        df_test=test_df,
        features=best_features,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    report = {
        "experiment": "exp_v15_hit_roi_max",
        "data_path": args.data_path,
        "split": {
            "train_end_year": args.train_end_year,
            "valid_year": args.valid_year,
            "test_year": args.test_year,
            "holdout_years_gt_test": True,
        },
        "constraints": {
            "min_hit": args.min_hit,
            "min_coverage": args.min_coverage,
            "min_bets": args.min_bets,
        },
        "best_candidate_by_valid": best_by_valid["row"],
        "best_spec": asdict(best_spec),
        "best_strategy": best_strategy,
        "final_test_metrics": final_test_metrics,
        "final_holdout_metrics": final_hold_metrics,
        "artifacts": {
            "model": model_path,
            "features": features_path,
            "strategy": strategy_path,
            "candidate_results": candidate_csv,
            "shap_summary": shap_summary.get("path"),
            "summary": summary_path,
        },
        "shap_top_features": shap_summary.get("top_features", []),
        "top_candidates_valid": candidate_df.head(10).to_dict(orient="records"),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log("=== FINAL RESULT ===")
    print(json.dumps(report["best_candidate_by_valid"], ensure_ascii=False, indent=2), flush=True)
    print("Final Test:", json.dumps(final_test_metrics, ensure_ascii=False), flush=True)
    if final_hold_metrics is not None:
        print("Final Holdout:", json.dumps(final_hold_metrics, ensure_ascii=False), flush=True)
    log(f"Saved report: {args.report_path}")


if __name__ == "__main__":
    main()
