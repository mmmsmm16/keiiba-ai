# データ更新・運用パイプライン実行手順書

本ドキュメントでは、新しいレース開催日が追加された際（または過去の欠損データを補完する際）に行う、データ更新から予測・精算までの一連のオペレーション手順を記載します。

## 全体フロー
1. **データ取込**: PC-KEIBAデータベースから学習・予測用データをインクリメンタル更新する。
2. **オッズ更新**: オッズの時系列スナップショットファイル（Parquet）を更新する。
3. **予測実行**: 本番パイプラインを回し、推奨買い目（Orders）を生成する。
4. **収支精算**: レース終了後、確定結果に基づいてROIを計算する。

---

## 1. データ取込 (Preprocessing)

PC-KEIBAのデータベースに入っている最新のレース情報・馬毎データを、学習・予測用の形式（Parquet）に取り込みます。`--incremental` オプションを使用することで、既存データに不足している日付のみを高速に追加します。

**実行タイミング**: 新しいレース枠順が確定した後、またはレース翌日（成績データ反映後）。

```powershell
# インクリメンタルモードで実行（不足している日付のみ追加）
docker compose exec app python src/preprocessing/run_preprocessing.py --incremental
```

**確認事項**:
- ログに `Detected N missing dates: ['2025-12-06', ...]` のように追加対象の日付が表示されること。
- `Saved merged dataframe to data/preprocessed_data.parquet` と表示されれば完了。

---

## 2. オッズ更新 (Odds Snapshots)

予測パイプラインで高速かつ正確にオッズを参照するために、データベースのオッズデータを専用の時系列スナップショット形式（Parquet）に変換します。

**実行タイミング**: オッズデータが蓄積された後（リアルタイム予測の場合は不要だが、過去シミュレーションやバックテスト前には必須）。

```powershell
# 2025年分のオッズスナップショットを更新（既存ファイルは上書き・マージ更新されます）
docker compose exec app python src/data/build_time_series_odds.py --start_year 2025 --end_year 2025
```

**確認事項**:
- ログに `Saved 2025 T-10: ... rows` などが表示されること。
- これにより `data/odds_snapshots/2025/odds_T-10.parquet` などが最新化されます。

---

## 3. 予測実行 (Production Pipeline)

更新されたデータとオッズを使用して、予測とベット生成を行います。
オッズスナップショット（Step 2）が存在する場合、高速なParquetモードで実行されます。存在しない場合でもDBフォールバック機能が働きますが、Step 2を実施しておくことを推奨します。

**実行タイミング**: レース発走前（またはシミュレーション時）。

```powershell
# 日付を指定してパイプラインを実行 (例: 2025-12-07)
docker compose exec app python src/scripts/run_production_pipeline.py --date 2025-12-07 --mode paper --force
```

**オプション**:
- `--mode paper`: ペーパートレードモード（実際の発注は行わない）。
- `--force`: 既に出力ファイルが存在しても上書き実行する。

**出力物**:
- `outputs/reports/YYYYMMDD_prediction.md`: 予測レポート（買い目リスト含む）。
- `outputs/orders/YYYYMMDD_orders.csv`: 発注ファイル。

---

## 4. 収支精算 (Settlement)

レース確定後、またはシミュレーション実行後に、予測結果の答え合わせ（ROI計算）を行います。

**実行タイミング**: 全レース終了後・成績確定後。

```powershell
# 日付を指定して精算を実行
docker compose exec app python src/scripts/settle_paper_trades.py --date 2025-12-07
```

**出力物**:
- `reports/phase13/daily/YYYYMMDD_report.md`: 確定収支レポート。
- コンソールに `ROI: 158.11%` のように結果が表示される。

---

## トラブルシューティング

**Q: "No odds in snapshot" という警告が出る**
A: Step 2のオッズ更新が行われていません。自動的にDBフォールバック機能が作動して実行は継続されますが、処理速度が低下します。Step 2を実行してスナップショットを作成することを推奨します。

**Q: ROIが 0.00% になる**
A: 以下を確認してください。
1. ベットが1件も生成されていない（`No bets generated`）→ Step 3のログを確認（オッズ不足や予測スコア低迷など）。
2. DBに払戻データがない → PC-KEIBAのデータ取り込み状況を確認してください。
