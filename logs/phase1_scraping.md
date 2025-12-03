# Phase 1: データ収集基盤 (Scraping Pipeline) 開発ログ

## 方針
- `netkeiba.com` からレース結果を取得する。
- サーバー負荷を考慮し、必ず `time.sleep(1)` を挟む。
- 取得したデータは `pandas.DataFrame` に変換し、PostgreSQL に保存する。

## 実装ステップ
1. **Scraper:** HTML取得担当。リトライ処理や待機処理を含む。
2. **Parser:** HTML解析担当。`BeautifulSoup` を使用。
3. **Loader:** DB保存担当。`SQLAlchemy` を使用。

## 変更履歴
- **[2024-05-21]**: 開発開始。
- **[2024-05-21]**: `src/scraping/netkeiba.py` 実装完了。`requests` + `time.sleep`。
- **[2024-05-21]**: `src/scraping/parser.py` 実装完了。レース情報、結果、払い戻しのパースを確認。
    - 払い戻しテーブルのクラス名が `pay_block` ではなく `pay_table_01` であったため修正。
- **[2024-05-21]**: `src/scraping/loader.py` 実装完了。データの冪等性を担保するため、保存前に当該レースの既存データを削除するロジックを採用。
- **[2024-05-21]**: `src/scraping/run_sample.py` による結合テスト（スクレイピング→パース）成功。

## 次のステップ
- 過去10年分のレースIDリストを作成し、バッチ処理で取得するスクリプト (`src/scraping/bulk_loader.py` 等) の作成。
