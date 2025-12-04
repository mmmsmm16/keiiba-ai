# Phase 1: データ収集基盤 (Scraping Pipeline) 開発ログ

## 方針
- `netkeiba.com` からレース結果を取得する。
- サーバー負荷を考慮し、必ず `time.sleep(1)` を挟む。
- 取得したデータは `pandas.DataFrame` に変換し、PostgreSQL に保存する。

## 実装ステップ
1. **Scraper:** HTML取得担当。リトライ処理や待機処理を含む。
2. **Parser:** HTML解析担当。`BeautifulSoup` を使用。
3. **Loader:** DB保存担当。`SQLAlchemy` を使用。
4. **Bulk Loader:** 過去データの大量取得担当。ID生成とループ処理。

## 変更履歴
- **[2024-05-21]**: 開発開始。
- **[2024-05-21]**: `src/scraping/netkeiba.py` 実装完了。`requests` + `time.sleep`。
- **[2024-05-21]**: `src/scraping/parser.py` 実装完了。レース情報、結果、払い戻しのパースを確認。
    - 払い戻しテーブルのクラス名が `pay_block` ではなく `pay_table_01` であったため修正。
- **[2024-05-21]**: `src/scraping/loader.py` 実装完了。データの冪等性を担保するため、保存前に当該レースの既存データを削除するロジックを採用。
- **[2024-05-21]**: `src/scraping/run_sample.py` による結合テスト（スクレイピング→パース）成功。
- **[2024-05-21]**: `src/scraping/bulk_loader.py` 実装完了。`--dry_run` オプションを追加し、単体テスト (`src/scraping/test_bulk_loader.py`) にてロジックを検証済み。
- **[2024-05-21]**: リファクタリング実施。ソースコード内のコメントを英語から日本語へ翻訳 (AGENTS.md準拠)。

## 次のステップ
- Phase 2: データ前処理の実装へ移行。
