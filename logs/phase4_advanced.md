# Phase 4: 高度化 (Advanced Modeling) 開発ログ

## 方針
- 特徴量エンジニアリングの深化と、複数モデルのアンサンブルにより、予測精度の限界突破を目指す。
- CatBoost (YetiRank) を導入し、LightGBMとの補完関係を利用する。
- 展開予測（Pace）などのドメイン知識に基づいた特徴量を実装する。

## 実装ステップ
1. **AdvancedFeatureEngineer:** 展開予測（逃げ率、ペース予測）の実装。
2. **KeibaCatBoost:** CatBoost Rankerのラッパークラス。
3. **EnsembleModel:** LightGBMとCatBoostの予測値をメタモデル（LinearRegression）で統合するBlendingの実装。

## 変更履歴
- **[2024-05-21]**: Phase 4 計画策定。
- **[2024-05-21]**: `src/preprocessing/advanced_features.py` 実装完了。`passing_rank` からの逃げ判定と、過去の逃げ率集計、レースごとの逃げ馬数カウントを実装。
- **[2024-05-21]**: `src/model/catboost_model.py` 実装完了。YetiRankを用いたランキング学習クラス。
- **[2024-05-21]**: `src/model/ensemble.py` 実装完了。Validセットを用いたBlendingロジックを実装。

## 今後の課題
- アンサンブルモデルを `train.py` や `evaluate.py` に組み込み、単体LightGBMと比較検証する。
- 血統データの詳細分析（5代血統表など）は未着手。
