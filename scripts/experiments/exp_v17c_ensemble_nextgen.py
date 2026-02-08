from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.append("/workspace")

from scripts.experiments.exp_v16_multi_bet_max import (
    DATA_PATH,
    BASE_FEATURES_PATH,
    REPORT_PATH,
    OUT_DIR,
    eval_policy,
    load_payout_map,
    normalize_odds,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    pred_table,
    race_norm_prob,
    read_data,
    search_policy,
    to_matrix,
)


@dataclass
class ModelPack:
    name: str
    model_path: str
    feat_path: str


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_models(spec: str) -> List[ModelPack]:
    packs: List[ModelPack] = []
    for token in [t.strip() for t in str(spec).split(",") if t.strip()]:
        if "=" in token:
            name, path = token.split("=", 1)
            path = path.strip()
            name = name.strip()
        else:
            path = token
            name = os.path.basename(path.rstrip("/")).replace(" ", "_")
        mp = os.path.join(path, "model.pkl")
        fp = os.path.join(path, "features.csv")
        packs.append(ModelPack(name=name, model_path=mp, feat_path=fp))
    return packs


def load_feature_list(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "feature" in df.columns:
        return df["feature"].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def model_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        p = np.asarray(p)
        if p.ndim == 2:
            if p.shape[1] >= 2:
                return p[:, 1].astype(float)
            return p[:, 0].astype(float)
        return p.reshape(-1).astype(float)
    raw = np.asarray(model.predict(x)).reshape(-1)
    return (1.0 / (1.0 + np.exp(-raw))).astype(float)


def weight_vectors(n_models: int, step: float, max_vectors: int) -> List[np.ndarray]:
    if n_models <= 0:
        return []
    if n_models == 1:
        return [np.array([1.0], dtype=float)]
    units = int(round(1.0 / step))
    vecs: List[np.ndarray] = []
    for comb in itertools.combinations_with_replacement(range(n_models), units):
        c = np.bincount(comb, minlength=n_models).astype(float)
        w = c / c.sum()
        vecs.append(w)
    vecs.sort(key=lambda x: float(np.max(x)))
    if max_vectors > 0:
        return vecs[:max_vectors]
    return vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list: name=dir,name2=dir2 (dir has model.pkl/features.csv)")
    ap.add_argument("--data-path", default=DATA_PATH)
    ap.add_argument("--base-features-path", default=BASE_FEATURES_PATH)
    ap.add_argument("--output-dir", default="models/experiments/exp_v17c_ensemble_nextgen")
    ap.add_argument("--report-path", default="reports/exp_v17c_ensemble_nextgen.json")
    ap.add_argument("--train-end-year", type=int, default=2023)
    ap.add_argument("--valid-year", type=int, default=2024)
    ap.add_argument("--test-year", type=int, default=2025)
    ap.add_argument("--min-hit", type=float, default=0.14)
    ap.add_argument("--min-coverage", type=float, default=0.08)
    ap.add_argument("--min-tickets", type=int, default=1200)
    ap.add_argument("--policy-profile", default="jit_safe", choices=["jit_safe", "roi_balanced", "roi_aggressive"])
    ap.add_argument("--policy-axis-min-edges", default="0.0,1.05")
    ap.add_argument("--policy-axis-modes", default="value,hybrid")
    ap.add_argument("--policy-partner-modes", default="value,prob")
    ap.add_argument("--policy-pair-options", default="0,2,3")
    ap.add_argument("--policy-trio-options", default="0,3")
    ap.add_argument("--policy-max-cases", type=int, default=96)
    ap.add_argument("--weight-step", type=float, default=0.1)
    ap.add_argument("--max-weight-vectors", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    packs = parse_models(args.models)
    if not packs:
        raise RuntimeError("no models")
    for p in packs:
        if not os.path.exists(p.model_path):
            raise FileNotFoundError(p.model_path)
        if not os.path.exists(p.feat_path):
            raise FileNotFoundError(p.feat_path)

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
    log(f"rows train={len(tr)} valid={len(va)} test={len(te)}")

    pva = load_payout_map([args.valid_year], None)
    pte = load_payout_map([args.test_year], None)

    model_probs_valid: List[np.ndarray] = []
    model_probs_test: List[np.ndarray] = []
    model_meta: List[Dict[str, str]] = []

    for i, mp in enumerate(packs, start=1):
        log(f"[{i}/{len(packs)}] loading model {mp.name}")
        model = joblib.load(mp.model_path)
        feats = [c for c in load_feature_list(mp.feat_path) if c in df.columns]
        if not feats:
            raise RuntimeError(f"no usable features for {mp.name}")
        xva = to_matrix(va, feats)
        xte = to_matrix(te, feats)
        pva_raw = model_scores(model, xva)
        pte_raw = model_scores(model, xte)
        pva_norm = race_norm_prob(va["race_id"], pva_raw)
        pte_norm = race_norm_prob(te["race_id"], pte_raw)
        model_probs_valid.append(pva_norm)
        model_probs_test.append(pte_norm)
        model_meta.append({"name": mp.name, "model_path": mp.model_path, "feat_path": mp.feat_path, "n_features": len(feats)})

    axis_min_edges = parse_float_list(args.policy_axis_min_edges)
    axis_modes = parse_str_list(args.policy_axis_modes)
    partner_modes = parse_str_list(args.policy_partner_modes)
    pair_options = parse_int_list(args.policy_pair_options)
    trio_options = parse_int_list(args.policy_trio_options)

    wvecs = weight_vectors(len(packs), args.weight_step, args.max_weight_vectors)
    log(f"weight vectors={len(wvecs)}")

    rows: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None

    for i, w in enumerate(wvecs, start=1):
        bva = np.zeros_like(model_probs_valid[0], dtype=float)
        bte = np.zeros_like(model_probs_test[0], dtype=float)
        for j in range(len(w)):
            bva += float(w[j]) * model_probs_valid[j]
            bte += float(w[j]) * model_probs_test[j]
        bva = race_norm_prob(va["race_id"], bva)
        bte = race_norm_prob(te["race_id"], bte)

        vpred = pred_table(va, bva)
        tpred = pred_table(te, bte)

        best_pol = search_policy(
            vpred,
            pva,
            args.min_hit,
            args.min_coverage,
            args.min_tickets,
            f"blend {i}/{len(wvecs)}",
            profile=args.policy_profile,
            axis_min_edges=axis_min_edges,
            axis_modes=axis_modes,
            partner_modes=partner_modes,
            pair_options=pair_options,
            trio_options=trio_options,
            max_cases=int(args.policy_max_cases),
        )
        pol = best_pol["policy"]
        vm = best_pol["metrics"]
        tm = eval_policy(tpred, pte, pol, args.min_hit, args.min_coverage, args.min_tickets)

        row: Dict[str, object] = {
            "blend_id": i,
            "weights": {packs[k].name: float(w[k]) for k in range(len(w))},
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
            "policy": pol,
        }
        rows.append(row)
        if best is None or row["valid_utility"] > best["valid_utility"]:
            best = row
            log(
                f"best update blend={i} validROI={row['valid_blended_roi']:.2f} "
                f"testROI={row['test_blended_roi']:.2f}"
            )
        if i % 5 == 0 or i == len(wvecs):
            pd.DataFrame(rows).sort_values(["valid_utility", "test_blended_roi"], ascending=False).to_csv(
                os.path.join(args.output_dir, "ensemble_results_progress.csv"),
                index=False,
            )
            log(f"progress {i}/{len(wvecs)}")

    if not rows or best is None:
        raise RuntimeError("no ensemble results")

    res = pd.DataFrame(rows).sort_values(["valid_utility", "test_blended_roi"], ascending=False)
    res_path = os.path.join(args.output_dir, "ensemble_results.csv")
    res.to_csv(res_path, index=False)

    rep = {
        "experiment": "exp_v17c_ensemble_nextgen",
        "split": {
            "train_end_year": args.train_end_year,
            "valid_year": args.valid_year,
            "test_year": args.test_year,
        },
        "constraints": {
            "min_hit": args.min_hit,
            "min_coverage": args.min_coverage,
            "min_tickets": args.min_tickets,
        },
        "model_packs": model_meta,
        "weight_step": args.weight_step,
        "n_weight_vectors": len(wvecs),
        "policy_profile": args.policy_profile,
        "policy_axis_min_edges": axis_min_edges,
        "policy_axis_modes": axis_modes,
        "policy_partner_modes": partner_modes,
        "policy_pair_options": pair_options,
        "policy_trio_options": trio_options,
        "policy_max_cases": int(args.policy_max_cases),
        "best_candidate_by_valid": best,
        "top_candidates_valid": res.head(20).to_dict(orient="records"),
        "artifacts": {
            "ensemble_results": res_path,
            "ensemble_results_progress": os.path.join(args.output_dir, "ensemble_results_progress.csv"),
        },
    }

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    log("=== FINAL RESULT ===")
    print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    log(f"Saved report: {args.report_path}")


if __name__ == "__main__":
    main()
