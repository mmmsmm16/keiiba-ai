# Experiment Leaderboard

## Active Experiments (V13+)
| Exp ID | Features | Model | Valid Period | AUC | LogLoss | Brier | PR-AUC | NDCG@5 | Recall@5 | ROI | Memo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **exp_q12_bloodline** | +Sire Aptitude | LGBM (Win) | 2024 | **0.7745** | 0.2433 | - | - | 0.6031 | 0.6689 | - | Phase Q Iteration 12 (Best AUC, but Ranking dropped. Rejected.) |
| **exp_q11_dynamics** | +Race Dynamics | LGBM (Win) | 2024 | 0.7743 | **0.2433** | - | - | **0.6041** | **0.6704** | - | Phase Q Iteration 11 (Identical to Q8. No Gain. Rejected.) |
| **exp_q10_strategy** | +Jockey Strategy | LGBM (Win) | 2024 | 0.7738 | 0.2435 | - | - | 0.6033 | 0.6700 | - | Phase Q Iteration 10 (Slight Drop across all metrics. Rejected.) |
| **exp_q9_physique** | +Physique/Training | LGBM (Win) | 2024 | 0.7743 | 0.2435 | - | - | 0.6028 | 0.6680 | - | Phase Q Iteration 9 (AUC tie, but Ranking dropped. Rejected.) | 
| **exp_r3_ensemble_v2** | +Ens V2 (LongTrain) | Avg Ensemble | 2024 | **0.7914** | **0.2368** | - | - | **0.6232** | **0.6842** | 77.3% | Phase R Final V2 (Longer Training. Best Accuracy/Recall.) |
| **exp_r3_ensemble** | +Ens (LGBM+MLP+Tab) | Avg Ensemble | 2024 | 0.7909 | 0.2369 | - | - | 0.6224 | 0.6824 | **78.8%** | Phase R Final V1 (AUC > 0.79) |
| **exp_r2_tabnet** | +TabNet (Attentive) | TabNet | 2024 | 0.7885 | 0.2370 | - | - | - | - | TBD | Phase R Iteration 2 (Re-trained 200 Epochs) |
| **exp_r1_mlp** | +NN (MLP) Baseline | MLP (Embed) | 2024 | 0.7834 | 0.2392 | - | - | 0.6164 | 0.6758 | TBD | Phase R Iteration 1 (Deep Learning Breakthrough. All metrics UP.) |
| **exp_q8_freshness** | +Freshness | LGBM (Win) | 2024 | 0.7743 | 0.2433 | - | - | 0.6041 | 0.6704 | 78.3% | Phase Q Iteration 8 (Best All-Round, Recall > 0.67) |
| **exp_q7_golden_combos** | +Jockey-Trainer | LGBM (Win) | 2024 | 0.7742 | 0.2434 | - | - | 0.6039 | 0.6698 | - | Phase Q Iteration 7 (All Metrics Improved) |
| **exp_q6_runstyle_fit** | +RunStyle Fit | LGBM (Win) | 2024 | 0.7740 | **0.2434** | - | - | 0.6036 | 0.6695 | - | Phase Q Iteration 6 (Best NDCG ever, Small Field dominance) |
| **exp_q5_extended_aptitude** | +Sire/Jockey Apt | LGBM (Win) | 2024 | **0.7740** | **0.2434** | - | - | 0.6026 | 0.6682 | - | Phase Q Iteration 5 (Best AUC, Class Change improved) |
| **exp_q4_course_aptitude** | +Course Aptitude | LGBM (Win) | 2024 | 0.7736 | 0.2435 | - | - | 0.6030 | 0.6693 | - | Phase Q Iteration 4 (High AUC + NDCG Recovery) |
| **exp_q3_risk_stats** | +Risk Stats | LGBM (Win) | 2024 | **0.7740** | **0.2434** | - | - | 0.6019 | 0.6694 | - | Phase Q Iteration 3 (Best AUC/LL, Recall drop) |
| **exp_q2_relative_enhanced** | +Relative Diff | LGBM (Win) | 2024 | 0.7716 | 0.2444 | - | - | 0.6020 | 0.6687 | - | Phase Q Iteration 2 (Rejected: Recall/NDCG drop) |
| **exp_q1_class_stats** | +Class Stats | LGBM (Win) | 2024 | 0.7716 | 0.2460 | - | - | 0.6031 | **0.6908** | - | Phase Q Iteration 1 (Drift Fix, AUC +3.3%) |
| **phase_m_win (Repro)** | Baseline | LGBM (Win) | 2024 | 0.7380 | 0.2592 | - | - | 0.5823 | 0.6720 | - | Baseline Reproduction on Recalculated Features |
| **v13_top3_lgbm** | All + SpeedIndex(Safe) | LGBM (Top3) | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | 0.6368 | 0.7166 | **0.7166** | - | V13 Top3 (頑健化+RLE) |
| **v24_m5_final** | M4-B (Class Stats) | Ensemble (W1) | 2024 | - | - | - | - | 0.6047 | 0.6704 | 61.7% | Phase O Test (Overfitting/Regime Shift) |
| **v13_d1_pace** | All + SpeedIndex(Safe) + PacePressure(D1) | LGBM (Top3) | 2024 | 0.7621 | 0.5637 | 0.1928 | 0.4675 | 0.6356 | 0.7173 | 0.0% | ペースプレッシャー(D1)追加 |
| **v13_top3_lgbm** | All + SpeedIndex(Safe) | LGBM (Top3) | 2023 | 0.7560 | 0.5661 | 0.1939 | 0.4540 | - | 0.6781 | - | V13 Top3 Specialist (Leakage Fixed) |
| **v13_d1_pace** | All + SpeedIndex(Safe) + PacePressure(D1) | LGBM (Top3) | 2023 | 0.7563 | 0.5630 | 0.1919 | 0.4530 | 0.6263 | **0.7053** | 0.0% | Pace Pressure (D1) Added |
| **v13_top3_lgbm** | All + SpeedIndex(Safe) | LGBM (Top3) | 2022 | 0.7544 | 0.5781 | 0.1990 | 0.4503 | 0.5748 | **0.6793** | - | V13 Top3 Specialist (Leakage Fixed) |
| **v13_d1_pace** | All + SpeedIndex(Safe) + PacePressure(D1) | LGBM (Top3) | 2022 | 0.7541 | 0.5654 | 0.1982 | 0.4518 | **0.6295** | **0.7121** | 0.0% | ペースプレッシャー(D1)追加 |

<br>

## M4 Adhoc Leaderboard (2024, Split Pipeline)
**Eval Mode**: Adhoc (Features generated via Split Pipeline, Evaluated on `valid_preds`)

| モデル | Recall@5 | NDCG@5 | Race-Hit@5 | Precision@5 | 備考 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M5-Ensemble** | **0.6704** | **0.6047** | **0.9728** | **0.4019** | **最終採用**。Win/Top2/Top3 アンサンブル。 |
| **M4-B (Class)** | 0.6590 | 0.5601 | 0.9669 | 0.3944 | M4-B 単体モデル (Top3バイナリ) |
| **M4-C (Segment)** | 0.6590 | 0.5616 | 0.9672 | 0.3944 | **不採用**。小頭数で Recall 悪化 (-0.42pt)。 |
| **M4-A (Core)** | 0.6015 | 0.4961 | 0.9385 | 0.3599 | 実験用ベースライン (Core特徴量のみ) |
| **M3 Baseline** | 0.6061 | 0.5000 | 0.9392 | 0.3627 | 比較基準 (全特徴量) |

## Archive (Past Experiments)
| Exp ID | Feature Ver | Model | Valid Period | AUC | LogLoss | NDCG@5 | ROI (Flat) | Memo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Benchmark** | Odds | - | 2025 | 0.720 (Est) | - | - | 78-80% | 単勝人気順 |
| **v12_win_lgbm** | V12 Full | binary | 2024 | **0.7684** | 0.5145 | **0.5810** | 0.0% | v12 単勝特化型 (LGBM) |
| **v12_win_lgbm_ablation** | No Speed Index | binary | 2024 | 0.7671 | 0.5106 | 0.5783 | 0.0% | アブレーション・スタディ (SpeedIndex除外) |
| **v12_win_cat** | V12 Full | binary | 2024 | 0.6850 | 14.04 | 0.4726 | 0.0% | v12 単勝特化型 (CatBoost) |
| **v11_lgbm_enhanced** | V11 Full | lambdarank | 2024 | 0.7478 | 4.3319 | 0.5816 | 0.0% | LGBMでのNDCG最高記録 |
| **v11_cat_enhanced** | V11 Full | lambdarank | 2024 | 0.7428 | 4.5381 | 0.5708 | 0.0% | v10から改善したCatBoost |
| **v10_ensemble** | LGBM + Cat | rank_avg | 2024 | - | - | 0.5735 | 0.0% | チューニング済みアンサンブル |
| **v10_lgbm_competitor** | +Class/Grade | lambdarank | 2024 | - | - | 0.5786 | 0.0% | クラス/グレード特徴量追加 |
| **v09_ensemble** | LGBM + Cat | rank_avg | 2024 | - | - | 0.5766 | 0.0% | ハイブリッド・アンサンブル |
| **v08_return_tweedie** | +Odds | tweedie | 2024 | 0.7989 | 1.8844 | 0.6345 | 80.2% | **NDCG最高記録。** オッズリークの疑いあり |
| **v07_win_binary** | All | binary | 2024 | 0.7260 | 0.2505 | 0.5152 | 72.1% | ROIが72%で飽和 |
| **v05_sire** | +Bloodline | binary | 2024 | 0.7506 | 0.5900 | 0.5830 | 72.3% | 高いAUCを維持 |
| **v03_jockey** | +Jockey | binary | 2024 | 0.7479 | 0.6006 | 0.5802 | 0.0% | 騎手特徴量でAUC +1.7% |
| **v02_binary** | +History | binary | 2024 | 0.7310 | 0.6099 | - | 0.0% | ベースラインの修正 |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats'] | binary | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | - | 0.7166 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats'] | binary | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | - | 0.7166 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats'] | binary | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | - | 0.7166 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats'] | binary | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | 0.5809 | 0.7166 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats'] | binary | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | 0.6368 | 0.7166 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats'] | binary | 2024 | 0.7620 | 0.5669 | 0.1941 | 0.4671 | 0.6368 | 0.7166 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats'] | binary | 2024 | 0.7621 | 0.5637 | 0.1928 | 0.4675 | 0.6356 | 0.7173 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats'] | binary | 2023 | 0.7563 | 0.5617 | 0.1919 | 0.4530 | 0.6263 | 0.7053 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_top3_lgbm | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats'] | binary | 2022 | 0.7542 | 0.5801 | 0.1998 | 0.4518 | 0.6295 | 0.7121 | 0.0% | v13 Top3 Specialist (LGBM, is_unbalance=True, Leakage Fixed) |
| v13_d2_relative | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats'] | binary | 2024 | 0.7706 | 0.5524 | 0.1879 | 0.4848 | 0.6342 | 0.7140 | 0.0% | v13 D2 Relative Stats (Z-Score/Pct) |
| v13_d2_relative | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats'] | binary | 2023 | 0.7639 | 0.5539 | 0.1899 | 0.4708 | 0.6274 | 0.7085 | 0.0% | v13 D2 Relative Stats (Z-Score/Pct) |
| v13_d2_relative | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats'] | binary | 2022 | 0.7573 | 0.5739 | 0.1982 | 0.4571 | 0.6314 | 0.7115 | 0.0% | v13 D2 Relative Stats (Z-Score/Pct) |
| v13_d3_jockey_trainer | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7709 | 0.5516 | 0.1896 | 0.4860 | 0.6347 | 0.7129 | 0.0% | v13 D3 Jockey x Trainer Interaction Stats |

| v13_d3_jockey_trainer | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2023 | 0.7637 | 0.5532 | 0.1893 | 0.4701 | 0.6247 | 0.7073 | 0.0% | v13 D3 Jockey x Trainer Interaction Stats |
| v13_d3_jockey_trainer | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2022 | 0.7637 | 0.5719 | 0.1964 | 0.4726 | 0.6318 | 0.7113 | 0.0% | v13 D3 Jockey x Trainer Interaction Stats |


| v13_d2_relative | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats'] | binary | 2023 | 0.7640 | 0.5571 | 0.1899 | 0.4708 | 0.6274 | 0.7085 | 0.0% | v13 D2 Relative Stats (Z-Score/Pct) |
| v13_d2_relative | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats'] | binary | 2022 | 0.7637 | 0.5719 | 0.1962 | 0.4720 | 0.6314 | 0.7115 | 0.0% | v13 D2 Relative Stats (Z-Score/Pct) |
| v13_d3_jockey_trainer | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7709 | 0.5560 | 0.1896 | 0.4860 | 0.6347 | 0.7129 | 0.0% | v13 D3 Jockey x Trainer Interaction Stats |
| v13_d3_jockey_trainer | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2023 | 0.7637 | 0.5556 | 0.1893 | 0.4701 | 0.6247 | 0.7073 | 0.0% | v13 D3 Jockey x Trainer Interaction Stats |
| v13_d3_jockey_trainer | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2022 | 0.7637 | 0.5723 | 0.1964 | 0.4726 | 0.6318 | 0.7113 | 0.0% | v13 D3 Jockey x Trainer Interaction Stats |
| v13_e1_calib_2024 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7713 | 0.4441 | 0.1433 | 0.4881 | 0.6070 | 0.7144 | 0.0% | v13 Phase E: Platt Scaling Calibration (2024) |
| v13_e1_calib_2023 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2023 | 0.7654 | 0.4470 | 0.1443 | 0.4756 | 0.6008 | 0.7080 | 0.0% | v13 Phase E: Platt Scaling Calibration (2023) |
| v13_e1_calib_2023 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2023 | 0.7654 | 0.4470 | 0.1443 | 0.4756 | 0.6008 | 0.7080 | 0.0% | v13 Phase E: Platt Scaling Calibration (2023) |
| v13_e1_calib_2022 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2022 | 0.7651 | 0.4500 | 0.1453 | 0.4770 | 0.6011 | 0.7117 | 0.0% | v13 Phase E: Platt Scaling Calibration (2022) |
| m3_top3_decay_002 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7653 | 0.4493 | 0.1454 | 0.4791 | 0.6058 | 0.7168 | 0.0% | M3: Exponential Decay (lambda=0.002) |
| m3_top3_decay_001 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7705 | 0.4453 | 0.1439 | 0.4899 | 0.6121 | 0.7227 | 0.0% | M3: Exponential Decay (lambda=0.001) |
| m3_top3_decay_0005 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7735 | 0.4432 | 0.1431 | 0.4962 | 0.6154 | 0.7229 | 0.0% | M3: Exponential Decay (lambda=0.0005) |
| m3_top3_decay_piecewise | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7725 | 0.4439 | 0.1434 | 0.4947 | 0.6136 | 0.7234 | 0.0% | M3: Piecewise Decay |
| m3_top3_base | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7738 | 0.4430 | 0.1431 | 0.4957 | 0.6145 | 0.7209 | 0.0% | M3 Base: Top3 Model (No Decay) |
| phase_m_top2 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7770 | 0.3548 | 0.1089 | 0.3827 | 0.6138 | 0.7525 | 0.0% | Phase M: Top2 Probability Model |
| phase_m_top3 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7738 | 0.4436 | 0.1433 | 0.4902 | 0.6130 | 0.7199 | 0.0% | Phase M: Top3 Probability Model |
| exp_20251222_031150 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7053 | 0.4990 | 0.1649 | 0.4214 | 0.5337 | 0.6615 | 0.0% |  |
| exp_20251222_031151 | ['base_attributes', 'history_stats', 'jockey_stats', 'pace_stats', 'bloodline_stats', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats', 'jockey_trainer_stats'] | binary | 2024 | 0.7049 | 0.4990 | 0.1649 | 0.4214 | 0.5346 | 0.6607 | 0.0% |  |
| exp_20251222_040356 | ['base_attributes', 'history_stats', 'jockey_stats', 'temporal_jockey_stats', 'temporal_trainer_stats', 'pace_stats', 'bloodline_stats', 'odds_features', 'training_stats', 'burden_stats', 'changes_stats', 'aptitude_stats', 'speed_index_stats', 'pace_pressure_stats', 'relative_stats'] | binary | 2024 | 0.7052 | 0.4989 | 0.1649 | 0.4212 | 0.5329 | 0.6617 | 0.0% |  |
| exp_t2_refined_v3 | 31 features | binary | 2023 | 0.7779 | 0.2289 | 0.0626 | 0.2177 | 0.5246 | 0.7609 | 0.0% | T2 Refined v3 (Expanded Features: Conditions, Structure, Elo, Trends) |
| exp_t2_refined_v3 | 31 features | binary | 2023 | 0.7959 | 0.2248 | 0.0618 | 0.2376 | 0.5454 | 0.7846 | 0.0% | T2 Refined v3 (Expanded Features: Conditions, Structure, Elo, Trends) |
