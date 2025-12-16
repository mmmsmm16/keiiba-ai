# Phase 7 (v2): Multi-Ticket Backtest Report

**Date**: 2025-12-16 01:12
**Year**: 2024
**Filter**: JRA-only

## Bankroll Settings

| Parameter | Value |
|-----------|-------|
| Initial Bankroll | ¥100,000 |
| Max Bet Fraction | 5.0% |
| Min Equity Threshold | ¥100 |
| Rescale Mode | scale |

## Results

| Ticket | TopN | Races | ROI | Max DD | Rescales | Final Equity |
|--------|------|-------|-----|--------|----------|--------------|
| umaren | 3 | 2,096 | 84.1% | 98.1% | 75 | ¥1,960 |
| umaren | 4 | 872 | 78.9% | 98.1% | 146 | ¥1,940 |
| umaren | 5 | 523 | 77.2% | 98.1% | 133 | ¥1,942 |
| sanrenpuku | 4 | 2,303 | 88.8% | 98.1% | 195 | ¥1,955 |
| sanrenpuku | 5 | 646 | 81.2% | 98.0% | 179 | ¥1,977 |
| sanrenpuku | 6 | 344 | 68.9% | 98.1% | 268 | ¥1,936 |
| sanrentan | 4 | 239 | 67.0% | 98.1% | 147 | ¥1,931 |
| sanrentan | 5 | 284 | 52.2% | 98.1% | 284 | ¥1,920 |
| sanrentan | 6 | 273 | 70.3% | 98.4% | 273 | ¥1,900 |

## Validation

- **Max DD < 100%**: ✅ PASS

## Bankroll Constraint Diagnostics

| Ticket | TopN | Planned Bet | Executed Bet | Rescale Count | Avg Ratio | Skip | Bankrupt Stops |
|--------|------|-------------|--------------|---------------|-----------|------|----------------|
| umaren | 3 | ¥949,800 | ¥617,900 | 75 | 0.516 | 1070 | 0 |
| umaren | 4 | ¥1,899,600 | ¥464,500 | 146 | 0.330 | 2294 | 0 |
| umaren | 5 | ¥3,165,000 | ¥430,400 | 133 | 0.304 | 2642 | 0 |
| sanrenpuku | 4 | ¥1,266,400 | ¥878,600 | 195 | 0.454 | 863 | 0 |
| sanrenpuku | 5 | ¥3,165,000 | ¥521,700 | 179 | 0.306 | 2519 | 0 |
| sanrenpuku | 6 | ¥6,314,000 | ¥315,300 | 268 | 0.305 | 2813 | 0 |
| sanrentan | 4 | ¥7,598,400 | ¥297,600 | 147 | 0.218 | 2927 | 0 |
| sanrentan | 5 | ¥18,990,000 | ¥205,300 | 284 | 0.120 | 2881 | 0 |
| sanrentan | 6 | ¥37,884,000 | ¥330,800 | 273 | 0.101 | 2884 | 0 |

## Best Strategy (Max DD < 100%)

**sanrenpuku BOX4**: ROI **88.84%**, Max DD 98.06%, Final Equity ¥1,955
