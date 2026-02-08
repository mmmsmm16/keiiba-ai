"""
Exp v16: multi-bet ROI/HIT optimization with multi-model comparison.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CAT = True
except Exception:
    HAS_CAT = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
    TABNET_IMPORT_ERROR = ""
except Exception as e:
    HAS_TABNET = False
    TABNET_IMPORT_ERROR = f"{type(e).__name__}: {e}"

DATA_PATH = "data/processed/preprocessed_data_v12.parquet"
BASE_FEATURES_PATH = "models/experiments/exp_lambdarank_v12_batch4_optuna_odds10_weighted/features.csv"
OUT_DIR = "models/experiments/exp_v16_multi_bet_max"
REPORT_PATH = "reports/exp_v16_multi_bet_max.json"
BET_TYPES = ["win", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"]

A_FEATS = ["sire_variant_pref", "blood_variant_fit", "sire_course_win_rate_decay", "sire_dist_win_rate_decay", "sire_surface_win_rate_decay", "sire_win_rate_std_50", "sire_win_rate_iqr_50", "horse_variant_pref", "horse_variant_match"]
B_FEATS = ["pace_diff_z_ctx", "horse_pace_elasticity_20", "pace_high_prob", "pace_fit_expected", "front_congestion_idx", "nige_score", "nige_score_interaction", "race_nige_pressure_score_sum", "race_nige_count_weighted"]
C_FEATS = ["track_variant_robust", "track_variant_uncertainty", "track_variant_confidence", "going_shift", "going_shift_up_top3_rate", "going_shift_down_top3_rate", "rest_success_rate_smoothed", "rest_success_rate_decay", "rest_optimality_score", "rotation_stress"]


@dataclass
class Spec:
    name: str
    algo: str
    objective: str
    mode: str
    wmode: str


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def parse_str_list(s: str) -> List[str]:
    return [p.strip() for p in str(s).split(",") if p.strip()]


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _coalesce(df: pd.DataFrame, target: str, cands: List[str]) -> pd.DataFrame:
    cols = [c for c in cands if c in df.columns]
    if not cols:
        return df
    s = pd.to_numeric(df[cols[0]], errors="coerce")
    for c in cols[1:]:
        s = s.combine_first(pd.to_numeric(df[c], errors="coerce"))
    df[target] = s
    for c in cols:
        if c != target:
            df = df.drop(columns=[c], errors="ignore")
    return df


def normalize_odds(df: pd.DataFrame) -> pd.DataFrame:
    mp = {
        "odds_10min": ["odds_10min", "odds_10min_x", "odds_10min_y"],
        "odds_60min": ["odds_60min", "odds_60min_x", "odds_60min_y"],
        "odds_final": ["odds_final", "odds_final_x", "odds_final_y"],
    }
    for t, c in mp.items():
        df = _coalesce(df, t, c)
    return df


def read_data(data_path: str, feat_path: str) -> pd.DataFrame:
    base = pd.read_csv(feat_path).iloc[:, 0].astype(str).tolist()
    req = set(["race_id", "horse_number", "date", "rank", "odds", "popularity", "odds_10min", "odds_60min"] + base + A_FEATS + B_FEATS + C_FEATS)
    try:
        import pyarrow.parquet as pq
        allc = set(pq.ParquetFile(data_path).schema.names)
        use = sorted(c for c in req if c in allc)
        if use:
            log(f"Using {len(use)} columns")
            return pd.read_parquet(data_path, columns=use)
    except Exception as e:
        log(f"minimal read fallback: {e}")
    return pd.read_parquet(data_path)


def to_matrix(df: pd.DataFrame, feats: List[str]) -> np.ndarray:
    x = df.reindex(columns=feats, fill_value=0).copy()
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            x[c] = x[c].cat.codes
        elif x[c].dtype == "object":
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32, copy=False)


def race_norm_prob(race_ids: pd.Series, p: np.ndarray) -> np.ndarray:
    s = pd.Series(p, index=race_ids.index).clip(lower=1e-8)
    den = s.groupby(race_ids.astype(str)).transform("sum")
    return (s / den).fillna(0).to_numpy(dtype=float)


def race_softmax_prob(race_ids: pd.Series, score: np.ndarray) -> np.ndarray:
    s = pd.Series(score, index=race_ids.index, dtype=float)
    grp = race_ids.astype(str)
    mx = s.groupby(grp).transform("max")
    ex = np.exp((s - mx).clip(-30, 30))
    den = ex.groupby(grp).transform("sum")
    return (ex / den).fillna(0).to_numpy(dtype=float)


def make_modes(df: pd.DataFrame, feat_path: str) -> Dict[str, List[str]]:
    base = [c for c in pd.read_csv(feat_path).iloc[:, 0].astype(str).tolist() if c in df.columns]
    enh = [c for c in (A_FEATS + B_FEATS + C_FEATS) if c in df.columns]
    return {
        "NO_MARKET": sorted(set(base + enh)),
        "LIGHT_MARKET": sorted(set(base + enh + ["market_prob", "log_odds", "odds_rank_pre"])),
    }

def fit_and_predict(spec: Spec, feats: List[str], tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame, seed: int):
    xtr = to_matrix(tr, feats)
    xva = to_matrix(va, feats)
    if spec.objective == "binary":
        ytr = (pd.to_numeric(tr["rank"], errors="coerce") == 1).astype(int).to_numpy()
        yva = (pd.to_numeric(va["rank"], errors="coerce") == 1).astype(int).to_numpy()
        sw = None
        if spec.wmode in {"value", "value_hard"}:
            o = pd.to_numeric(tr["odds"], errors="coerce").fillna(10).clip(lower=1.01, upper=120)
            if spec.wmode == "value_hard":
                # Emphasize undervalued winners more aggressively than the standard value mode.
                sw = (1 + ytr * np.power(np.log1p(o), 2.2)).to_numpy(dtype=float)
            else:
                sw = (1 + ytr * np.log1p(o)).to_numpy(dtype=float)

        if spec.algo == "lgbm":
            m = lgb.LGBMClassifier(
                objective="binary", metric="binary_logloss", n_estimators=1200, learning_rate=0.03,
                num_leaves=95 if spec.mode != "NO_MARKET" else 63, min_child_samples=80,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.0,
                scale_pos_weight=14.0, random_state=seed, n_jobs=4, verbosity=-1
            )
            m.fit(xtr, ytr, sample_weight=sw, eval_set=[(xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(80)])
        elif spec.algo == "xgb":
            m = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", n_estimators=500, learning_rate=0.04,
                max_depth=6, min_child_weight=4, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.2, reg_lambda=1.0, random_state=seed, n_jobs=4, tree_method="hist"
            )
            m.fit(xtr, ytr, sample_weight=sw, eval_set=[(xva, yva)], verbose=False)
        elif spec.algo == "mlp":
            m = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            alpha=1e-4,
                            learning_rate_init=1e-3,
                            batch_size=2048,
                            max_iter=40,
                            early_stopping=True,
                            n_iter_no_change=8,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            fit_kwargs = {"clf__sample_weight": sw} if sw is not None else {}
            try:
                m.fit(xtr, ytr, **fit_kwargs)
            except TypeError:
                # Compatibility fallback for sklearn versions without sample_weight support.
                m.fit(xtr, ytr)
        elif spec.algo == "tabnet":
            m = TabNetClassifier(
                n_d=32,
                n_a=32,
                n_steps=4,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-4,
                mask_type="entmax",
                seed=seed,
                verbose=0,
            )
            fit_params = dict(
                X_train=xtr,
                y_train=ytr,
                eval_set=[(xva, yva)],
                eval_name=["valid"],
                eval_metric=["logloss"],
                max_epochs=40,
                patience=8,
                batch_size=8192,
                virtual_batch_size=1024,
                num_workers=0,
                drop_last=False,
            )
            if sw is not None:
                fit_params["weights"] = sw
            m.fit(**fit_params)
        else:
            m = cb.CatBoostClassifier(
                loss_function="Logloss", eval_metric="Logloss", iterations=1200, learning_rate=0.03,
                depth=8, l2_leaf_reg=3, random_seed=seed, verbose=False
            )
            m.fit(xtr, ytr, sample_weight=sw, eval_set=(xva, yva), use_best_model=True)

        pva = race_norm_prob(va["race_id"], m.predict_proba(xva)[:, 1])
        xte = to_matrix(te, feats)
        pte = race_norm_prob(te["race_id"], m.predict_proba(xte)[:, 1])
        del xtr, xva, xte
        gc.collect()
        return m, pva, pte

    ytr = np.clip(4 - pd.to_numeric(tr["rank"], errors="coerce").fillna(99).astype(int), 0, 3).to_numpy()
    yva = np.clip(4 - pd.to_numeric(va["rank"], errors="coerce").fillna(99).astype(int), 0, 3).to_numpy()
    gtr = tr.groupby("race_id", sort=False).size().to_numpy()
    gva = va.groupby("race_id", sort=False).size().to_numpy()

    if spec.algo == "lgbm":
        m = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 3, 5],
            n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=80,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, n_jobs=4, verbosity=-1
        )
        m.fit(xtr, ytr, group=gtr, eval_set=[(xva, yva)], eval_group=[gva], eval_at=[1, 3, 5], callbacks=[lgb.early_stopping(80)])
    elif spec.algo == "xgb":
        m = xgb.XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg@5", n_estimators=500, learning_rate=0.04,
            max_depth=6, min_child_weight=4, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=4, tree_method="hist"
        )
        m.fit(xtr, ytr, group=gtr, eval_set=[(xva, yva)], eval_group=[gva], verbose=False)
    elif spec.algo == "cat":
        tr_gid = np.repeat(np.arange(len(gtr)), gtr)
        va_gid = np.repeat(np.arange(len(gva)), gva)
        tr_pool = cb.Pool(xtr, ytr, group_id=tr_gid)
        va_pool = cb.Pool(xva, yva, group_id=va_gid)
        m = cb.CatBoostRanker(
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=5",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=seed,
            verbose=False,
        )
        m.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    else:
        raise ValueError(f"ranker objective not supported for algo={spec.algo}")

    pva = race_softmax_prob(va["race_id"], m.predict(xva))
    xte = to_matrix(te, feats)
    pte = race_softmax_prob(te["race_id"], m.predict(xte))
    del xtr, xva, xte
    gc.collect()
    return m, pva, pte


def pred_table(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({
        "race_id": df["race_id"].astype(str).values,
        "horse_number": pd.to_numeric(df["horse_number"], errors="coerce").fillna(0).astype(int).values,
        "rank": pd.to_numeric(df["rank"], errors="coerce").fillna(99).astype(int).values,
        "odds": pd.to_numeric(df["odds"], errors="coerce").fillna(0).clip(lower=1.01, upper=200).values,
        "popularity": pd.to_numeric(df.get("popularity", np.nan), errors="coerce"),
        "pred_prob": p,
    })
    out["market_prob"] = (1.0 / out["odds"].clip(lower=1.01)).clip(upper=1.0)
    out["edge_ratio"] = (out["pred_prob"] / out["market_prob"].clip(lower=1e-9)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _si(v) -> Optional[int]:
    if pd.isna(v):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _pair(v, ordered: bool) -> Optional[Tuple[int, int]]:
    x = _si(v)
    if x is None:
        return None
    s = f"{x:04d}"
    a, b = int(s[:2]), int(s[2:4])
    if a <= 0 or b <= 0:
        return None
    return (a, b) if ordered else tuple(sorted((a, b)))


def _trio(v, ordered: bool) -> Optional[Tuple[int, int, int]]:
    x = _si(v)
    if x is None:
        return None
    s = f"{x:06d}"
    a, b, c = int(s[:2]), int(s[2:4]), int(s[4:6])
    if a <= 0 or b <= 0 or c <= 0:
        return None
    return (a, b, c) if ordered else tuple(sorted((a, b, c)))

def load_payout_map(years: List[int], db_url: Optional[str]) -> Dict[str, Dict[str, Dict]]:
    if not years:
        return {}
    if db_url is None:
        u = os.environ.get("POSTGRES_USER", "postgres")
        p = os.environ.get("POSTGRES_PASSWORD", "postgres")
        h = os.environ.get("POSTGRES_HOST", "db")
        pt = os.environ.get("POSTGRES_PORT", "5432")
        d = os.environ.get("POSTGRES_DB", "pckeiba")
        db_url = f"postgresql://{u}:{p}@{h}:{pt}/{d}"
    eng = create_engine(db_url)
    ys = ",".join([f"'{int(y)}'" for y in sorted(set(years))])
    q = text(f"""
    SELECT CONCAT(kaisai_nen,LPAD(keibajo_code::text,2,'0'),LPAD(kaisai_kai::text,2,'0'),LPAD(kaisai_nichime::text,2,'0'),LPAD(race_bango::text,2,'0')) AS race_id,
           haraimodoshi_tansho_1a, haraimodoshi_tansho_1b,
           haraimodoshi_umaren_1a, haraimodoshi_umaren_1b, haraimodoshi_umaren_2a, haraimodoshi_umaren_2b, haraimodoshi_umaren_3a, haraimodoshi_umaren_3b,
           haraimodoshi_wide_1a, haraimodoshi_wide_1b, haraimodoshi_wide_2a, haraimodoshi_wide_2b, haraimodoshi_wide_3a, haraimodoshi_wide_3b,
           haraimodoshi_wide_4a, haraimodoshi_wide_4b, haraimodoshi_wide_5a, haraimodoshi_wide_5b, haraimodoshi_wide_6a, haraimodoshi_wide_6b, haraimodoshi_wide_7a, haraimodoshi_wide_7b,
           haraimodoshi_umatan_1a, haraimodoshi_umatan_1b, haraimodoshi_umatan_2a, haraimodoshi_umatan_2b, haraimodoshi_umatan_3a, haraimodoshi_umatan_3b,
           haraimodoshi_umatan_4a, haraimodoshi_umatan_4b, haraimodoshi_umatan_5a, haraimodoshi_umatan_5b, haraimodoshi_umatan_6a, haraimodoshi_umatan_6b,
           haraimodoshi_sanrenpuku_1a, haraimodoshi_sanrenpuku_1b, haraimodoshi_sanrenpuku_2a, haraimodoshi_sanrenpuku_2b, haraimodoshi_sanrenpuku_3a, haraimodoshi_sanrenpuku_3b,
           haraimodoshi_sanrentan_1a, haraimodoshi_sanrentan_1b, haraimodoshi_sanrentan_2a, haraimodoshi_sanrentan_2b, haraimodoshi_sanrentan_3a, haraimodoshi_sanrentan_3b,
           haraimodoshi_sanrentan_4a, haraimodoshi_sanrentan_4b, haraimodoshi_sanrentan_5a, haraimodoshi_sanrentan_5b, haraimodoshi_sanrentan_6a, haraimodoshi_sanrentan_6b
    FROM jvd_hr WHERE kaisai_nen IN ({ys})
    """)
    df = pd.read_sql(q, eng)
    pm: Dict[str, Dict[str, Dict]] = {}
    for _, r in df.iterrows():
        rid = str(r["race_id"])
        pm[rid] = {"win": {}, "umaren": {}, "wide": {}, "umatan": {}, "sanrenpuku": {}, "sanrentan": {}}
        wh, wp = _si(r.get("haraimodoshi_tansho_1a")), _si(r.get("haraimodoshi_tansho_1b"))
        if wh and wp:
            pm[rid]["win"][int(wh)] = int(wp)
        for i in range(1, 4):
            k, pay = _pair(r.get(f"haraimodoshi_umaren_{i}a"), False), _si(r.get(f"haraimodoshi_umaren_{i}b"))
            if k and pay:
                pm[rid]["umaren"][k] = int(pay)
        for i in range(1, 8):
            k, pay = _pair(r.get(f"haraimodoshi_wide_{i}a"), False), _si(r.get(f"haraimodoshi_wide_{i}b"))
            if k and pay:
                pm[rid]["wide"][k] = int(pay)
        for i in range(1, 7):
            k, pay = _pair(r.get(f"haraimodoshi_umatan_{i}a"), True), _si(r.get(f"haraimodoshi_umatan_{i}b"))
            if k and pay:
                pm[rid]["umatan"][k] = int(pay)
        for i in range(1, 4):
            k, pay = _trio(r.get(f"haraimodoshi_sanrenpuku_{i}a"), False), _si(r.get(f"haraimodoshi_sanrenpuku_{i}b"))
            if k and pay:
                pm[rid]["sanrenpuku"][k] = int(pay)
        for i in range(1, 7):
            k, pay = _trio(r.get(f"haraimodoshi_sanrentan_{i}a"), True), _si(r.get(f"haraimodoshi_sanrentan_{i}b"))
            if k and pay:
                pm[rid]["sanrentan"][k] = int(pay)
    log(f"Loaded payout map races={len(pm)} for years={sorted(set(years))}")
    return pm


def policy_grid(
    profile: str = "jit_safe",
    axis_min_edges: Optional[List[float]] = None,
    axis_modes: Optional[List[str]] = None,
    partner_modes: Optional[List[str]] = None,
    pair_options: Optional[List[int]] = None,
    trio_options: Optional[List[int]] = None,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    axis_min_edges = axis_min_edges or [0.0]
    axis_modes = axis_modes or ["value"]
    partner_modes = partner_modes or ["value"]
    if profile == "roi_aggressive":
        gammas = [0.3, 0.6, 0.9]
        bonuses = [0.2, 0.4, 0.6]
        axis_ps = [0.03, 0.05, 0.08]
        min_odds = [1.0, 2.0, 3.0]
        win_evs = [0.9, 1.1, 1.3]
        pair_ns = [2, 3, 4]
        trio_ns = [3, 4]
    elif profile == "roi_balanced":
        gammas = [0.0, 0.3, 0.6]
        bonuses = [0.0, 0.2, 0.4]
        axis_ps = [0.05, 0.08]
        min_odds = [1.0, 2.5]
        win_evs = [0.9, 1.1, 1.3]
        pair_ns = [2, 3]
        trio_ns = [3, 4]
    else:
        # jit_safe: compact search, predictable run time.
        gammas = [0.0, 0.4]
        bonuses = [0.0, 0.3]
        axis_ps = [0.05, 0.08]
        min_odds = [1.0, 2.5]
        win_evs = [0.9, 1.1]
        pair_ns = [2, 3]
        trio_ns = [3]
    if pair_options:
        pair_ns = [int(v) for v in pair_options]
    if trio_options:
        trio_ns = [int(v) for v in trio_options]

    for g in gammas:
        for b in bonuses:
            for pp in axis_ps:
                for mo in min_odds:
                    for ev in win_evs:
                        for pn in pair_ns:
                            for tn in trio_ns:
                                for edge_thr in axis_min_edges:
                                    for axis_mode in axis_modes:
                                        for partner_mode in partner_modes:
                                            out.append(
                                                {
                                                    "gamma": g,
                                                    "bonus": b,
                                                    "axis_p": pp,
                                                    "axis_min_odds": mo,
                                                    "axis_min_edge": float(edge_thr),
                                                    "axis_mode": str(axis_mode),
                                                    "partner_mode": str(partner_mode),
                                                    "win_ev": ev,
                                                    "pair_n": float(pn),
                                                    "trio_n": float(tn),
                                                }
                                            )
    return out


def eval_policy(pred: pd.DataFrame, pm: Dict[str, Dict[str, Dict]], pol: Dict[str, object], min_hit: float, min_cov: float, min_tickets: int) -> Dict[str, object]:
    st = {bt: {"cost": 0.0, "ret": 0.0, "tickets": 0, "ticket_hits": 0, "races_bet": 0, "races_hit": 0} for bt in BET_TYPES}
    total_races, races_bet_any, races_hit_any = 0, 0, 0
    axis_pop, axis_odds = [], []

    for rid, gdf in pred.groupby("race_id", sort=False):
        pay = pm.get(str(rid))
        if pay is None:
            continue
        total_races += 1
        w = gdf[["horse_number", "odds", "popularity", "pred_prob", "market_prob", "edge_ratio"]].copy()
        longflag = ((pd.to_numeric(w["popularity"], errors="coerce") >= 6) | (pd.to_numeric(w["odds"], errors="coerce") >= 10)).astype(float)
        w["value"] = w["pred_prob"] * np.power(w["odds"].clip(lower=1.01), float(pol["gamma"])) * (1.0 + float(pol["bonus"]) * longflag)
        axis_mode = str(pol.get("axis_mode", "value"))
        if axis_mode == "edge":
            w["axis_score"] = w["edge_ratio"] * np.sqrt(w["pred_prob"].clip(lower=0.0))
        elif axis_mode == "hybrid":
            w["axis_score"] = w["value"] * np.power(w["edge_ratio"].clip(lower=1e-6), 0.40)
        else:
            w["axis_score"] = w["value"]
        w = w.sort_values(["axis_score", "pred_prob"], ascending=False)
        if w.empty:
            continue
        ax = w.iloc[0]
        ah, ap, ao = int(ax["horse_number"]), float(ax["pred_prob"]), float(ax["odds"])
        apop = float(pd.to_numeric(ax.get("popularity", np.nan), errors="coerce"))
        aedge = float(ax.get("edge_ratio", 0.0))
        if ap < float(pol["axis_p"]) or ao < float(pol["axis_min_odds"]) or aedge < float(pol.get("axis_min_edge", 0.0)):
            continue

        if str(pol.get("partner_mode", "value")) == "prob":
            wp = w.sort_values(["pred_prob", "value"], ascending=False)
        else:
            wp = w.sort_values(["value", "pred_prob"], ascending=False)
        pair = wp[wp["horse_number"] != ah].head(int(pol["pair_n"]))["horse_number"].astype(int).tolist()
        trio = wp[wp["horse_number"] != ah].head(int(pol["trio_n"]))["horse_number"].astype(int).tolist()
        race_ret, race_cost = 0.0, 0.0

        if ap * ao >= float(pol["win_ev"]):
            st["win"]["races_bet"] += 1
            st["win"]["cost"] += 100; st["win"]["tickets"] += 1; race_cost += 100
            r = float(pay["win"].get(ah, 0)); st["win"]["ret"] += r; race_ret += r
            if r > 0:
                st["win"]["ticket_hits"] += 1; st["win"]["races_hit"] += 1

        if len(pair) > 0:
            for bt in ["umaren", "wide", "umatan"]:
                st[bt]["races_bet"] += 1
            for h2 in pair:
                k_umaren = tuple(sorted((ah, int(h2))))
                k_wide = k_umaren
                k_umatan = (ah, int(h2))
                for bt, k in [("umaren", k_umaren), ("wide", k_wide), ("umatan", k_umatan)]:
                    st[bt]["cost"] += 100; st[bt]["tickets"] += 1; race_cost += 100
                    r = float(pay[bt].get(k, 0)); st[bt]["ret"] += r; race_ret += r
                    if r > 0:
                        st[bt]["ticket_hits"] += 1
            for bt in ["umaren", "wide", "umatan"]:
                if st[bt]["ret"] > 0:
                    pass

        if len(trio) >= 2:
            st["sanrenpuku"]["races_bet"] += 1
            st["sanrentan"]["races_bet"] += 1
            hit_trio, hit_tan = False, False
            for h2, h3 in itertools.combinations(trio, 2):
                ku = tuple(sorted((ah, int(h2), int(h3))))
                st["sanrenpuku"]["cost"] += 100; st["sanrenpuku"]["tickets"] += 1; race_cost += 100
                r = float(pay["sanrenpuku"].get(ku, 0)); st["sanrenpuku"]["ret"] += r; race_ret += r
                if r > 0:
                    st["sanrenpuku"]["ticket_hits"] += 1; hit_trio = True
                for p2, p3 in [(h2, h3), (h3, h2)]:
                    kt = (ah, int(p2), int(p3))
                    st["sanrentan"]["cost"] += 100; st["sanrentan"]["tickets"] += 1; race_cost += 100
                    r2 = float(pay["sanrentan"].get(kt, 0)); st["sanrentan"]["ret"] += r2; race_ret += r2
                    if r2 > 0:
                        st["sanrentan"]["ticket_hits"] += 1; hit_tan = True
            if hit_trio:
                st["sanrenpuku"]["races_hit"] += 1
            if hit_tan:
                st["sanrentan"]["races_hit"] += 1

        if race_cost > 0:
            races_bet_any += 1
            if race_ret > 0:
                races_hit_any += 1
            axis_pop.append(apop); axis_odds.append(ao)

    tot_cost = float(sum(st[bt]["cost"] for bt in BET_TYPES))
    tot_ret = float(sum(st[bt]["ret"] for bt in BET_TYPES))
    tickets = int(sum(st[bt]["tickets"] for bt in BET_TYPES))
    roi = (tot_ret / tot_cost * 100.0) if tot_cost > 0 else 0.0
    hit = (races_hit_any / races_bet_any) if races_bet_any > 0 else 0.0
    cov = (races_bet_any / total_races) if total_races > 0 else 0.0
    ap = pd.Series(axis_pop, dtype=float); ao = pd.Series(axis_odds, dtype=float)
    pop3 = float((ap <= 3).mean()) if len(ap) > 0 else 0.0
    longshot = float(((ap >= 6) | (ao >= 10)).mean()) if len(ap) > 0 else 0.0

    utility = roi - max(0, min_hit - hit) * 550 - max(0, min_cov - cov) * 1500 - max(0, min_tickets - tickets) * 0.03 + min(longshot, 0.5) * 10 - max(0, pop3 - 0.75) * 10

    bm = {}
    for bt in BET_TYPES:
        c, r, t = st[bt]["cost"], st[bt]["ret"], st[bt]["tickets"]
        rb, rh = st[bt]["races_bet"], st[bt]["races_hit"]
        bm[bt] = {"roi": (r / c * 100) if c > 0 else 0.0, "tickets": int(t), "race_hit": (rh / rb) if rb > 0 else 0.0}

    return {
        "blended_roi": roi,
        "blended_hit": hit,
        "coverage": cov,
        "total_tickets": tickets,
        "axis_longshot_rate": longshot,
        "axis_pop3_rate": pop3,
        "utility": float(utility),
        "bet_type_metrics": bm,
    }


def search_policy(
    pred: pd.DataFrame,
    pm: Dict[str, Dict[str, Dict]],
    min_hit: float,
    min_cov: float,
    min_tickets: int,
    label: str,
    profile: str = "jit_safe",
    axis_min_edges: Optional[List[float]] = None,
    axis_modes: Optional[List[str]] = None,
    partner_modes: Optional[List[str]] = None,
    pair_options: Optional[List[int]] = None,
    trio_options: Optional[List[int]] = None,
    max_cases: int = 0,
) -> Dict[str, object]:
    grid = policy_grid(
        profile=profile,
        axis_min_edges=axis_min_edges,
        axis_modes=axis_modes,
        partner_modes=partner_modes,
        pair_options=pair_options,
        trio_options=trio_options,
    )
    if max_cases > 0:
        grid = grid[:max_cases]
    best_any, best_ok = None, None
    log(f"[{label}] policy search start: {len(grid)}")
    for i, pol in enumerate(grid, start=1):
        m = eval_policy(pred, pm, pol, min_hit, min_cov, min_tickets)
        cand = {"policy": pol, "metrics": m}
        if best_any is None or m["utility"] > best_any["metrics"]["utility"]:
            best_any = cand
        ok = m["total_tickets"] >= min_tickets and m["blended_hit"] >= min_hit and m["coverage"] >= min_cov
        if ok and (best_ok is None or m["blended_roi"] > best_ok["metrics"]["blended_roi"]):
            best_ok = cand
            log(f"[{label}] new feasible roi={m['blended_roi']:.2f} hit={m['blended_hit']:.3f} tickets={m['total_tickets']} ({i}/{len(grid)})")
        if i % 40 == 0 or i == len(grid):
            cur = best_ok if best_ok is not None else best_any
            log(f"[{label}] progress {i}/{len(grid)} bestU={cur['metrics']['utility']:.2f}")
    return best_ok or best_any or {}

def flatten(prefix: str, m: Dict[str, object]) -> Dict[str, object]:
    out = {}
    bm = m.get("bet_type_metrics", {})
    for bt in BET_TYPES:
        out[f"{prefix}_{bt}_roi"] = float(bm.get(bt, {}).get("roi", 0.0))
        out[f"{prefix}_{bt}_tickets"] = int(bm.get(bt, {}).get("tickets", 0))
        out[f"{prefix}_{bt}_race_hit"] = float(bm.get(bt, {}).get("race_hit", 0.0))
    return out


def shap_summary(model, df_test: pd.DataFrame, feats: List[str], out_dir: str, seed: int) -> Dict[str, object]:
    if len(df_test) == 0:
        return {"top_features": []}
    try:
        smp = df_test.sample(min(1500, len(df_test)), random_state=seed)
        x = pd.DataFrame(to_matrix(smp, feats), columns=feats)
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(x, check_additivity=False)
        arr = np.asarray(sv[-1] if isinstance(sv, list) else sv)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        imp = np.abs(arr).mean(axis=0)
        s = pd.DataFrame({"feature": feats, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False).head(30).reset_index(drop=True)
        p = os.path.join(out_dir, "shap_summary_top30.csv")
        s.to_csv(p, index=False)
        return {"top_features": s.to_dict(orient="records"), "path": p}
    except Exception as e:
        return {"top_features": [], "path": None, "error": str(e)}


def build_specs(
    algos: List[str],
    objectives: List[str],
    modes: List[str],
    wmods: List[str],
    skip_cat_ranker: bool = True,
) -> List[Spec]:
    sp = []
    for a in algos:
        if a == "xgb" and not HAS_XGB:
            continue
        if a == "cat" and not HAS_CAT:
            continue
        if a == "tabnet" and not HAS_TABNET:
            continue
        for o in objectives:
            if o == "ranker" and a in {"mlp", "tabnet"}:
                continue
            if skip_cat_ranker and a == "cat" and o == "ranker":
                continue
            for m in modes:
                if o == "binary":
                    for w in wmods:
                        sp.append(Spec(name=f"{a}_{o}_{m.lower()}_{w}", algo=a, objective=o, mode=m, wmode=w))
                else:
                    sp.append(Spec(name=f"{a}_{o}_{m.lower()}", algo=a, objective=o, mode=m, wmode="none"))
    return sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=DATA_PATH)
    ap.add_argument("--base-features-path", default=BASE_FEATURES_PATH)
    ap.add_argument("--output-dir", default=OUT_DIR)
    ap.add_argument("--report-path", default=REPORT_PATH)
    ap.add_argument("--train-end-year", type=int, default=2023)
    ap.add_argument("--valid-year", type=int, default=2024)
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--min-hit", type=float, default=0.17)
    ap.add_argument("--min-coverage", type=float, default=0.12)
    ap.add_argument("--min-tickets", type=int, default=1600)
    ap.add_argument("--algos", default="lgbm,xgb,cat")
    ap.add_argument("--objectives", default="binary,ranker")
    ap.add_argument("--feature-modes", default="NO_MARKET,LIGHT_MARKET")
    ap.add_argument("--binary-weight-modes", default="none,value")
    ap.add_argument("--policy-profile", default="jit_safe", choices=["jit_safe", "roi_balanced", "roi_aggressive"])
    ap.add_argument("--policy-axis-min-edges", default="0.0,1.05")
    ap.add_argument("--policy-axis-modes", default="value,hybrid")
    ap.add_argument("--policy-partner-modes", default="value,prob")
    ap.add_argument("--policy-pair-options", default="", help="comma separated ints, empty=profile default")
    ap.add_argument("--policy-trio-options", default="", help="comma separated ints, empty=profile default")
    ap.add_argument("--policy-max-cases", type=int, default=0, help="0 means full grid")
    ap.add_argument("--skip-cat-ranker", action="store_true", default=False)
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    log(f"Loading data: {args.data_path}")
    df = read_data(args.data_path, args.base_features_path)
    df = normalize_odds(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df = df[df["date"].notna()].copy()
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df[df["odds"].notna() & (df["odds"] > 0)].copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df[df["rank"].notna()].copy()

    oref = pd.to_numeric(df["odds_10min"], errors="coerce") if "odds_10min" in df.columns else pd.to_numeric(df["odds"], errors="coerce")
    df["odds_feature"] = oref.fillna(pd.to_numeric(df["odds"], errors="coerce")).clip(lower=1.01)
    df["market_prob"] = (1.0 / df["odds_feature"]).clip(upper=1.0)
    df["log_odds"] = np.log1p(df["odds_feature"])
    df["odds_rank_pre"] = df.groupby("race_id")["odds_feature"].rank(method="min", ascending=True)

    tr = df[df["year"] <= args.train_end_year].copy()
    va = df[df["year"] == args.valid_year].copy()
    te = df[df["year"] == args.test_year].copy()
    ho = df[df["year"] > args.test_year].copy()
    log(f"rows train={len(tr)} valid={len(va)} test={len(te)} holdout={len(ho)}")

    modes_map = make_modes(df, args.base_features_path)
    modes = [m.strip() for m in str(args.feature_modes).split(",") if m.strip()]
    for m in modes:
        if m not in modes_map:
            raise RuntimeError(f"unknown mode: {m}")
    axis_min_edges = parse_float_list(args.policy_axis_min_edges)
    axis_modes = parse_str_list(args.policy_axis_modes)
    partner_modes = parse_str_list(args.policy_partner_modes)
    pair_options = parse_int_list(args.policy_pair_options)
    trio_options = parse_int_list(args.policy_trio_options)

    specs = build_specs(
        [s.strip() for s in str(args.algos).split(",") if s.strip()],
        [s.strip() for s in str(args.objectives).split(",") if s.strip()],
        modes,
        [s.strip() for s in str(args.binary_weight_modes).split(",") if s.strip()],
        skip_cat_ranker=bool(args.skip_cat_ranker),
    )
    requested_algos = [s.strip() for s in str(args.algos).split(",") if s.strip()]
    if "tabnet" in requested_algos and not HAS_TABNET:
        log(f"tabnet skipped: {TABNET_IMPORT_ERROR}")
    if not specs:
        raise RuntimeError("no specs")
    log(
        f"candidates={len(specs)} policy_edges={axis_min_edges} "
        f"axis_modes={axis_modes} partner_modes={partner_modes} "
        f"pair_options={pair_options or 'default'} trio_options={trio_options or 'default'} "
        f"policy_max_cases={args.policy_max_cases}"
    )

    pva = load_payout_map([args.valid_year], args.db_url)
    pte = load_payout_map([args.test_year], args.db_url)

    rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []
    best = None
    for i, sp in enumerate(specs, start=1):
        feats = modes_map[sp.mode]
        log(f"[Candidate {i}/{len(specs)}] {sp.name} feats={len(feats)}")
        try:
            m, vprob, tprob = fit_and_predict(sp, feats, tr, va, te, args.seed)
            vpred = pred_table(va, vprob)
            tpred = pred_table(te, tprob)

            best_pol_pack = search_policy(
                vpred,
                pva,
                args.min_hit,
                args.min_coverage,
                args.min_tickets,
                f"{sp.name} {i}/{len(specs)}",
                profile=args.policy_profile,
                axis_min_edges=axis_min_edges,
                axis_modes=axis_modes,
                partner_modes=partner_modes,
                pair_options=pair_options,
                trio_options=trio_options,
                max_cases=int(args.policy_max_cases),
            )
            pol = best_pol_pack["policy"]
            vm = best_pol_pack["metrics"]
            tm = eval_policy(tpred, pte, pol, args.min_hit, args.min_coverage, args.min_tickets)

            row: Dict[str, object] = {
                "candidate": sp.name,
                "algo": sp.algo,
                "objective": sp.objective,
                "feature_mode": sp.mode,
                "weight_mode": sp.wmode,
                "n_features": len(feats),
                "policy_gamma": pol["gamma"],
                "policy_bonus": pol["bonus"],
                "policy_axis_p": pol["axis_p"],
                "policy_axis_min_odds": pol["axis_min_odds"],
                "policy_axis_min_edge": pol.get("axis_min_edge", 0.0),
                "policy_axis_mode": pol.get("axis_mode", "value"),
                "policy_partner_mode": pol.get("partner_mode", "value"),
                "policy_win_ev": pol["win_ev"],
                "policy_pair_n": int(pol["pair_n"]),
                "policy_trio_n": int(pol["trio_n"]),
                "valid_utility": float(vm["utility"]),
                "valid_blended_roi": float(vm["blended_roi"]),
                "valid_blended_hit": float(vm["blended_hit"]),
                "valid_tickets": int(vm["total_tickets"]),
                "valid_coverage": float(vm["coverage"]),
                "test_utility": float(tm["utility"]),
                "test_blended_roi": float(tm["blended_roi"]),
                "test_blended_hit": float(tm["blended_hit"]),
                "test_tickets": int(tm["total_tickets"]),
                "test_coverage": float(tm["coverage"]),
            }
            row.update(flatten("valid", vm)); row.update(flatten("test", tm))
            rows.append(row)

            pd.DataFrame(rows).sort_values(["valid_utility", "test_blended_roi"], ascending=False).to_csv(
                os.path.join(args.output_dir, "candidate_results_progress.csv"),
                index=False,
            )
            log(
                f"  valid ROI={vm['blended_roi']:.2f} HIT={vm['blended_hit']:.3f} tickets={vm['total_tickets']} "
                f"| test ROI={tm['blended_roi']:.2f} HIT={tm['blended_hit']:.3f} tickets={tm['total_tickets']}"
            )
            if best is None or row["valid_utility"] > best["row"]["valid_utility"]:
                best = {"row": row, "spec": sp, "feats": feats, "model": m}
                log(f"  best updated -> {sp.name}")

            del vpred, tpred, vprob, tprob
            gc.collect()
        except Exception as e:
            emsg = f"{type(e).__name__}: {e}"
            log(f"  candidate failed -> {sp.name} :: {emsg}")
            log(traceback.format_exc())
            failed_rows.append(
                {
                    "candidate": sp.name,
                    "algo": sp.algo,
                    "objective": sp.objective,
                    "feature_mode": sp.mode,
                    "weight_mode": sp.wmode,
                    "n_features": len(feats),
                    "error": emsg,
                }
            )
            if rows:
                pd.DataFrame(rows).sort_values(["valid_utility", "test_blended_roi"], ascending=False).to_csv(
                    os.path.join(args.output_dir, "candidate_results_progress.csv"),
                    index=False,
                )
            pd.DataFrame(failed_rows).to_csv(os.path.join(args.output_dir, "candidate_failures.csv"), index=False)
            gc.collect()
            continue

    if not rows or best is None:
        raise RuntimeError("all candidates failed; check candidate_failures.csv")

    cand = pd.DataFrame(rows).sort_values(["valid_utility", "test_blended_roi"], ascending=False)
    cand_path = os.path.join(args.output_dir, "candidate_results.csv")
    cand.to_csv(cand_path, index=False)

    shp = shap_summary(best["model"], te, best["feats"], args.output_dir, args.seed)
    model_path = os.path.join(args.output_dir, "model.pkl")
    feat_path = os.path.join(args.output_dir, "features.csv")
    joblib.dump(best["model"], model_path)
    pd.DataFrame({"feature": best["feats"]}).to_csv(feat_path, index=False)

    rep = {
        "experiment": "exp_v16_multi_bet_max",
        "split": {"train_end_year": args.train_end_year, "valid_year": args.valid_year, "test_year": args.test_year},
        "constraints": {"min_hit": args.min_hit, "min_coverage": args.min_coverage, "min_tickets": args.min_tickets},
        "policy_profile": args.policy_profile,
        "policy_axis_min_edges": axis_min_edges,
        "policy_axis_modes": axis_modes,
        "policy_partner_modes": partner_modes,
        "policy_pair_options": pair_options,
        "policy_trio_options": trio_options,
        "policy_max_cases": int(args.policy_max_cases),
        "skip_cat_ranker": bool(args.skip_cat_ranker),
        "best_candidate_by_valid": best["row"],
        "best_spec": best["spec"].__dict__,
        "artifacts": {"model": model_path, "features": feat_path, "candidate_results": cand_path, "shap_summary": shp.get("path")},
        "shap_top_features": shp.get("top_features", []),
        "top_candidates_valid": cand.head(15).to_dict(orient="records"),
        "failed_candidates": failed_rows,
    }
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    log("=== FINAL RESULT ===")
    print(json.dumps(best["row"], ensure_ascii=False, indent=2), flush=True)
    log(f"Saved report: {args.report_path}")


if __name__ == "__main__":
    main()
