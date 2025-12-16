# Phase 5: Slippage Sweep Summary

**Date**: 2025-12-16 12:28
**Prob Column**: `prob_residual_softmax`
**Odds Source**: final (with --allow_final_odds flag)
**Purpose**: Evaluate ROI sensitivity to odds slippage (conservative evaluation)

## Results

| Slippage Factor | Expected Odds | Bets | Hit% | ROI | MaxDD |
|-----------------|---------------|------|------|-----|-------|
| 1.00 | 最終オッズそのまま | 12,492 | 36.5% | **230.2%** | ¥2,120 |
| 0.95 | ×0.95 (5%不利) | 11,443 | 36.4% | **227.4%** | ¥1,883 |
| 0.90 | ×0.90 (10%不利) | 10,352 | 36.5% | **225.5%** | ¥1,649 |

## Key Findings

1. **戦略は非常に堅牢**: 10% slippage でも ROI 225% を維持
2. **Bet数減少**: slippage が高いほど EV 閾値を超えにくくなり、Bet数が減少
3. **MaxDD改善**: slippage が高いほど MaxDD が減少（より保守的な選択）

## Interpretation

| Slippage | Meaning |
|----------|---------|
| 1.00 | 最終オッズそのまま（楽観シナリオ） |
| 0.95 | 5%悪いオッズで購入できた想定（標準保守） |
| 0.90 | 10%悪いオッズで購入できた想定（保守的） |

## Recommendation

✅ **保守シナリオ (slippage=0.90) でも ROI 225.5%** → 戦略は堅牢

実運用では:
- 締切前オッズデータがあれば `--odds_source pre_close` を使用
- 最終オッズを使う場合は `--slippage_factor 0.90` で保守評価
- 結果が slippage に対してロバストであることを確認

## Command Reference

```bash
# 標準（最終オッズ、slippage=1.0）
docker compose exec app python src/phase5/prob_sweep_roi.py \
  --prob_cols prob_residual_softmax \
  --odds_source final \
  --allow_final_odds \
  --slippage_factor 1.0

# 保守（10% slippage）
docker compose exec app python src/phase5/prob_sweep_roi.py \
  --prob_cols prob_residual_softmax \
  --odds_source final \
  --allow_final_odds \
  --slippage_factor 0.90
```
