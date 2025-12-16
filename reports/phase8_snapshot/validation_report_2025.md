# Phase 8: Time-Series Odds Validation Report (FINAL)

**Date**: 2025-12-16
**Model**: v13 market_residual
**Subject**: Evaluation of Final Odds vs Snapshot (T-10m) Odds

---

## Executive Summary

| Metric | Final Odds | Snapshot Odds | Δ |
|--------|------------|---------------|---|
| **Win EV Strategy ROI** | 224.8% | 220.0% | **+4.85%** |
| **95% CI of Δ** | - | - | **[+0.75%, +8.46%]** |
| **P(Final > Snapshot)** | - | - | **98.8%** |

> [!IMPORTANT]
> **統計的に有意**: 95% CIが0を含まないため、Final Odds を使った方が snapshot より約5%高いROIを示すことが統計的に確認されました。これは「Final Odds Bias」の存在を支持します。

---

## 1. Population/Denominator Definition

> [!NOTE]
> **比較母集団は 2,951レース** (v13予測とSnapshotオッズの intersection) に固定。  
> Bet Races（ベット発生レース数）とは区別し、全レースを分母として計算しています。

| Metric | Final | Snapshot |
|--------|-------|----------|
| **Total Races (分母)** | 2,951 | 2,951 |
| Bet Races (EV>1 あり) | 2,908 (98.5%) | 2,919 (98.9%) |
| No-Bet Races | 43 | 32 |
| Total Stake | ¥957,800 | ¥1,003,000 |
| Total Payout | ¥2,153,230 | ¥2,206,340 |
| **ROI** | **224.81%** | **219.97%** |
| 95% CI | [206.5%, 244.6%] | [202.4%, 238.9%] |

### Bet Race Differences

| Category | Count | 説明 |
|----------|-------|------|
| Both | 2,905 | 両条件でベット発生 |
| Only Final | 3 | Finalのみ EV>1 達成 |
| Only Snapshot | 14 | Snapshotのみ EV>1 達成 |
| Neither | 29 | どちらも EV>1 なし |

---

## 2. Bootstrap Statistical Analysis

**方法**: レース単位でブートストラップ (n=1,000)。全2,951レースを対象とし、ベット無しレースは profit=0 として含む。

```
Final ROI:    225.1% [95% CI: 206.5%, 244.6%]
Snapshot ROI: 220.3% [95% CI: 202.4%, 238.9%]

ROI Difference: +4.85%
95% CI:         [+0.75%, +8.46%]  ← 0を含まない ✅
P(Final > Snapshot): 98.8%
```

**結論**: Final Odds Bias は統計的に有意 (p < 0.02)。

---

## 3. Timestamp Validation

| Metric | Value |
|--------|-------|
| Snapshot取得時刻 (発走前) | 10-11分 |
| Post-race contamination | 0件 (0.00%) |
| <5分前取得 | 0件 (0.00%) |

✅ Snapshotは正しく発走前オッズを取得しています。

---

## 4. Prediction Drift Analysis

| Metric | Value |
|--------|-------|
| Mean absolute prob diff | 0.0125 |
| **Rank change rate** | **39.2%** |
| Top-1 overlap | 86.9% |

p_market をsnapshotオッズに差し替えると、約13%の Top-1 予測が変化します。

---

## 5. Monthly ROI Breakdown

| Month | Final ROI | Snapshot ROI | Δ |
|-------|-----------|--------------|---|
| 01 | 184.7% | 179.4% | +5.3% |
| 02 | 235.6% | 225.0% | +10.6% |
| 03 | 312.5% | 299.8% | +12.7% |
| 04 | 200.5% | 194.2% | +6.3% |
| 05 | 216.6% | 211.6% | +5.0% |
| 06 | 250.5% | 247.4% | +3.1% |
| 07 | 203.9% | 200.7% | +3.3% |
| 08 | 228.3% | 231.3% | -3.1% |
| 09 | 195.4% | 186.1% | +9.3% |

8/9ヶ月で Final > Snapshot。8月のみ逆転。

---

## 6. Leakage Evidence

| Check | Result |
|-------|--------|
| Training excludes 2025 | ✅ (folds: 2022,2023,2024) |
| Forbidden columns | ✅ None |
| Current-race results | ✅ Not in features |

---

## Conclusions

### 確定事項
1. ✅ **Final Odds Bias は統計的に有意** (95% CI: [+0.75%, +8.46%], p<0.02)
2. ✅ **運用時ROI見積り**: 約220% (T-10mオッズ基準)
3. ✅ **バックテストROI**: 約225% (確定オッズ基準)
4. ✅ **Bias幅**: 約+5% (Final が有利)
5. ✅ **リーク無し**: 学習期間・特徴量に問題なし

### 解釈
- 確定オッズを使ったバックテストは、実運用ROIを約5%過大評価する
- 運用時は T-10m オッズ基準の 220% が現実的な期待値
- ただし、これは「単勝 EV>1.0 戦略」のみの結果であり、他の券種には外挿できない

---

## Files Created

| File | Purpose |
|------|---------|
| `data/odds_snapshots/2025_win_T-10m_jra_only.parquet` | Snapshot odds |
| `data/predictions/v13_market_residual_2025_snapshot_recalc.parquet` | Recalculated predictions |
| `reports/phase8_snapshot/win_ev/aligned_ledger_final.parquet` | Aligned ledger (Final) |
| `reports/phase8_snapshot/win_ev/aligned_ledger_snapshot.parquet` | Aligned ledger (Snapshot) |

---

## Reproducibility Commands

```bash
# Population alignment analysis
docker compose exec app python scripts/phase8_population_align.py

# Full audit
docker compose exec app python scripts/phase8_audit.py

# Win EV backtest
docker compose exec app python scripts/phase8_win_ev_backtest.py --ev_threshold 1.0
```
