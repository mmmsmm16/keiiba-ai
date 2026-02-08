"""
Run focused ROI campaign for exp_v16.

Usage:
  docker compose exec app python scripts/experiments/run_v16_roi_campaign.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_one(name: str, policy_profile: str, objective: str = "binary") -> dict:
    out_dir = f"models/experiments/{name}"
    report_path = f"reports/{name}.json"
    stdout_path = f"reports/{name}.stdout.txt"

    cmd = [
        "python",
        "-u",
        "scripts/experiments/exp_v16_multi_bet_max.py",
        "--algos",
        "xgb",
        "--objectives",
        objective,
        "--feature-modes",
        "LIGHT_MARKET",
        "--binary-weight-modes",
        "value",
        "--policy-profile",
        policy_profile,
        "--min-hit",
        "0.14",
        "--min-coverage",
        "0.08",
        "--min-tickets",
        "1200",
        "--output-dir",
        out_dir,
        "--report-path",
        report_path,
    ]
    log(f"Run start: {name} ({policy_profile}, {objective})")
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed: {name}, rc={proc.returncode}")
    log(f"Run done: {name}")

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    best = rep.get("best_candidate_by_valid", {})
    return {
        "name": name,
        "policy_profile": policy_profile,
        "objective": objective,
        "valid_blended_roi": best.get("valid_blended_roi"),
        "valid_blended_hit": best.get("valid_blended_hit"),
        "test_blended_roi": best.get("test_blended_roi"),
        "test_blended_hit": best.get("test_blended_hit"),
        "test_tickets": best.get("test_tickets"),
        "test_coverage": best.get("test_coverage"),
    }


def main() -> None:
    runs = [
        ("exp_v16_xgb_binary_jit_safe", "jit_safe", "binary"),
        ("exp_v16_xgb_binary_roi_balanced", "roi_balanced", "binary"),
        ("exp_v16_xgb_binary_roi_aggressive", "roi_aggressive", "binary"),
        ("exp_v16_xgb_ranker_roi_balanced", "roi_balanced", "ranker"),
    ]
    results = []
    for name, profile, objective in runs:
        try:
            results.append(run_one(name, profile, objective))
        except Exception as e:
            log(f"ERROR: {e}")
            results.append({"name": name, "policy_profile": profile, "objective": objective, "error": str(e)})

    out = Path("reports/exp_v16_roi_campaign_summary.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved summary: {out}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
