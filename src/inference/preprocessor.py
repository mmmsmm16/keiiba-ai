import pandas as pd
import numpy as np
import logging
import os
import sys

# src pass
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.aggregators import HistoryAggregator
from preprocessing.category_aggregators import CategoryAggregator
from preprocessing.advanced_features import AdvancedFeatureEngineer
from preprocessing.cleansing import DataCleanser

logger = logging.getLogger(__name__)

class InferencePreprocessor:
    """
    推論用：データの前処理を行うクラス。
    過去データと結合して特徴量を再生成し、学習時と同じフォーマットの入力を作成します。
    """
    def __init__(self, history_path: str = None):
        if history_path is None:
            # デフォルトは src/../../data/processed/preprocessed_data.parquet
            base_dir = os.path.dirname(__file__)
            history_path = os.path.join(base_dir, '../../data/processed/preprocessed_data.parquet')
        
        self.history_path = history_path

    def preprocess(self, new_df: pd.DataFrame, history_df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        推論用データを前処理します (高速化版)。
        
        高速化戦略:
        1. 馬の過去走集計 (Lag/Rolling):
           - 予測対象の馬のデータのみを過去データから抽出して結合・再計算する。
           - 全件再計算を回避し、計算量を O(N_horses) に削減。
        2. カテゴリ集計 (Jockey/Trainer/Sire):
           - 過去データの「最終状態」をルックアップテーブルとして作成。
           - 予測対象行に対してマージし、前走結果をもとに値を更新（インクリメンタル更新）する。
        """
        logger.info("推論用前処理を開始します (Incremental Mode)...")

        # 1. 過去データのロード (引数で渡された場合はそれを使用、なければファイルからロード)
        if history_df is None:
            if not os.path.exists(self.history_path):
                raise FileNotFoundError(f"過去データが見つかりません: {self.history_path}")
            
            # Parquet読み込み
            history_df = pd.read_parquet(self.history_path)
        else:
            logger.info("キャッシュされた過去データを使用します。")
        
        # 日付変換
        new_df['date'] = pd.to_datetime(new_df['date'])
        
        # --- 基本特徴量生成 (new_dfのみ) ---
        feat_engineer = FeatureEngineer()
        new_df = feat_engineer.add_features(new_df)

        # ターゲット馬のIDリスト
        target_horse_ids = new_df['horse_id'].unique()
        
        # =================================================================
        # Strategy 1: Horse History Features (Filtering)
        # 予測対象馬の過去データを抽出
        # =================================================================
        relevant_history = history_df[history_df['horse_id'].isin(target_horse_ids)].copy()
        
        # 重複防止: 履歴データから今回の予測対象日のデータを除外
        # new_dfと同じ日付のデータが履歴に含まれている場合、concat後に重複してしまう
        target_dates = new_df['date'].unique()
        relevant_history = relevant_history[~relevant_history['date'].isin(target_dates)]
        
        # 結合 (Rolling計算のため、対象馬の全履歴 + 今回のデータ)
        # カテゴリ系カラムはhistory_dfには既に計算済みが入っているが、
        # new_dfには入っていない。ここで結合するとnew_dfのカテゴリ列はNaNになる。
        # -> HistoryAggregatorはカテゴリ列を使わないのでOK。
        
        # 必要なカラムに絞って結合することでメモリ節約
        # rank, last_3f などのraw columnsが必要
        combined_horse_df = pd.concat([relevant_history, new_df], axis=0, ignore_index=True)
        
        # 過去走集計実行 (対象馬のみなので高速)
        hist_agg = HistoryAggregator()
        combined_horse_df = hist_agg.aggregate(combined_horse_df)
        
        # 計算結果のうち、今回のnew_dfに対応する行だけを取り出したいが、
        # 後でAdvancedFeaturesでレース全体が必要になるため、一旦保持。
        
        # =================================================================
        # Strategy 2: Category Features (Lookup & Update)
        # 全履歴から最新の状態を取得し、new_dfにマッピングする
        # =================================================================
        logger.info("カテゴリ特徴量のインクリメンタル更新中...")
        
        # カテゴリごとに処理
        # CategoryAggregatorは f'{col}_n_races' の形式で保存している (例: jockey_id_n_races)
        for cat_col in ['jockey_id', 'trainer_id', 'sire_id']:
            prefix = cat_col # prefix = 'jockey_id' etc.
            # 1. ヒストリカルデータから各IDの「最新行」を取得
            # date, race_idでソート済みと仮定 (load時にsortされていない場合はsortが必要)
            # history_dfは通常時系列ソートされているはず。
            
            # 各カテゴリIDごとに最後の1行を取得
            # 必要なカラム: {prefix}_n_races, {prefix}_win_rate, {prefix}_top3_rate, rank (前走結果)
            cols_to_keep = [cat_col, f'{prefix}_n_races', f'{prefix}_win_rate', f'{prefix}_top3_rate', 'rank']
            
            # カラムが存在するかチェック (初回学習前などで見つからない場合のガード)
            available_cols = [c for c in cols_to_keep if c in history_df.columns]
            if len(available_cols) < len(cols_to_keep):
                logger.warning(f"カテゴリ特徴量列が見つかりません。{cat_col} の集計をスキップします。")
                continue

            latest_stats = history_df.drop_duplicates(subset=[cat_col], keep='last')[available_cols].set_index(cat_col)
            
            # 2. new_df にマージ
            # new_dfの各行に対し、そのカテゴリIDの「前走時点の集計値」と「前走ランク」を紐付ける
            merged = new_df[[cat_col]].join(latest_stats, on=cat_col, how='left')
            
            # 3. インクリメンタル更新計算
            # 取得できた値は「前走時点の特徴量」ではなく「前走終了時点の特徴量」にしたいが、
            # preprocessed_dataに入っているのは「そのレースの予測に使われた特徴量 (T-1までの集計)」である。
            # したがって、最新行(T)が持っている特徴量は (0...T-1) の集計値。
            # 最新行(T)の rank が T の結果。
            # 今回(T+1)に使いたい特徴量は (0...T) の集計値。
            # つまり、Loadした特徴量 + LoadしたRank で更新する必要がある。
            
            # n_races (T+1) = n_races (T) + 1
            # wins (T+1) = wins (T) + (1 if rank==1 else 0)
            #            = (n_races(T) * win_rate(T)) + (is_win)
            
            prev_n_races = merged[f'{prefix}_n_races'].fillna(0)
            prev_win_rate = merged[f'{prefix}_win_rate'].fillna(0)
            prev_top3_rate = merged[f'{prefix}_top3_rate'].fillna(0)
            prev_rank = merged['rank'] # NaNならレース未消化扱い
            
            # 未知のID(初出走騎手など)は fillna(0) で 0スタートになる
            
            # 更新ロジック
            is_prev_win = (prev_rank == 1).astype(int)
            is_prev_top3 = (prev_rank <= 3).astype(int)
            
            # 履歴データがあった場合のみカウントアップ (履歴がない=これが初=過去0戦)
            # merged['rank'] が NaN (入っていない) ということはないはず(history由来なので)。
            # ただし、historyに1走もなければNaN。その場合は n_races=0 で正しい。
            # しかし、historyの最後の行が「除外」などでrank NaNの可能性はある。
            # その場合もn_racesはカウントアップすべきか？ -> CategoryAggregatorの実装による(expanding.count()は非NaNを数える)。
            # 簡単のため、rankが有効な場合のみ更新する、あるいは historyがあるなら +1 する。
            
            has_history = prev_n_races > 0
            
            # 新しい値
            new_n_races = prev_n_races + 1
            
            # prev_n_races=0 (初) の場合、prev_runs * rate = 0 になるので式は成立する
            new_wins = (prev_n_races * prev_win_rate) + is_prev_win
            new_top3 = (prev_n_races * prev_top3_rate) + is_prev_top3
            
            # new_df にセット (combined_horse_df 側に入れる必要がある)
            # combined_horse_df は new_df の行を含んでいるので、race_id, horse_numberでキーにして入れるか、
            # 単純に new_df と同じ順序であることを利用するか。
            # combined_horse_df は HistoryAggregator 内で sort_values されている可能性がある。
            # 安全のため、dictionary mapping を作成して map する。
            
            # ID -> 新しい特徴量 のMapを作成したいが、同じ騎手が今回のレースに複数回出ることはない(1レース1騎乗)。
            # しかし new_df 全体では複数回出る。
            # したがって、new_dfのindex に基づいて計算したSeriesを、combined_horse_dfに結合する必要がある。
            
            # 計算したSeriesを一旦 new_df に格納
            new_df[f'{prefix}_n_races'] = new_n_races
            new_df[f'{prefix}_win_rate'] = (new_wins / new_n_races).fillna(0)
            new_df[f'{prefix}_top3_rate'] = (new_top3 / new_n_races).fillna(0)

        # =================================================================
        # Merge Category Features to Combined DF
        # =================================================================
        # combined_horse_df には今回分の行(new_df由来)も含まれているが、そこにはカテゴリ特徴量が入っていない(NaN)。
        # new_df で計算したカテゴリ特徴量を、combined_horse_df の該当行に移す。
        
        # combined_horse_df のうち、推論対象データ(dateがnew_dfの日付)を特定
        # あるいは race_id が new_df にあるもの。
        
        target_ids = new_df['race_id'].unique()
        is_target_row = combined_horse_df['race_id'].isin(target_ids)
        
        # マージ用のキーを作成 (race_id + horse_number)
        combined_horse_df['temp_key'] = combined_horse_df['race_id'].astype(str) + '_' + combined_horse_df['horse_number'].astype(str)
        new_df['temp_key'] = new_df['race_id'].astype(str) + '_' + new_df['horse_number'].astype(str)
        
        # カテゴリ特徴量列
        cat_feature_cols = []
        for cat_col in ['jockey_id', 'trainer_id', 'sire_id']:
            prefix = cat_col
            cat_feature_cols.extend([f'{prefix}_n_races', f'{prefix}_win_rate', f'{prefix}_top3_rate'])
            
        # 存在する列のみ
        cat_feature_cols = [c for c in cat_feature_cols if c in new_df.columns]
        
        # new_dfから辞書作成
        for col in cat_feature_cols:
            val_map = new_df.set_index('temp_key')[col].to_dict()
            # combined_horse_df の該当列を更新
            # mapを使って、target行のみ更新。history行は元の値を維持したいが、
            # history行には既に値が入っているはず。
            # fillnaを使って、NaNになっている(new_df由来の)行だけを埋める
            combined_horse_df[col] = combined_horse_df[col].fillna(combined_horse_df['temp_key'].map(val_map))
            
        combined_horse_df.drop('temp_key', axis=1, inplace=True)
        new_df.drop('temp_key', axis=1, inplace=True) # clean up

        # =================================================================
        # 3. 高度特徴量生成 (AdvancedFeatureEngineer)
        # =================================================================
        # 展開予測などはレース内のメンバー構成依存なので、結合後データに対して実行する必要がある
        # あるいは new_df (とその履歴) だけで閉じて計算できるか？
        # AdvancedFeatureEngineer は「past N races」を見るものもあるかもしれない。
        # 実装を確認すると、AdvancedFeatureEngineerは主に「同レース内の比較」や「脚質判定」を行う。
        # 脚質判定は過去走を参照する。よって combined_horse_df に対して行うのが正解。
        
        adv_engineer = AdvancedFeatureEngineer()
        combined_horse_df = adv_engineer.add_features(combined_horse_df)


        # =================================================================
        # 4. 推論対象行の抽出 & クリーニング
        # =================================================================
        # 今回のrace_idの行のみ抽出
        inference_df = combined_horse_df[combined_horse_df['race_id'].isin(target_ids)].copy()
        
        # 順序戻し
        inference_df = inference_df.sort_values(['race_id', 'horse_number'])
        
        logger.info(f"特徴量生成完了。推論対象: {len(inference_df)} 件")

        # 5. カラム選択 (学習時と同じ入力形式にする)
        # DatasetSplitter._create_lgbm_dataset のロジックを参照
        
        drop_cols = [
            # ID・メタデータ
            'race_id', 'date', 'title', 'horse_id', 'horse_name',
            'jockey_id', 'trainer_id', 'sire_id', 'mare_id',
            # 目的変数 (存在すれば)
            'rank', 'target', 'rank_str',
            # 未来情報 
            'time', 'raw_time',
            'passing_rank',
            'last_3f',
            'odds', 'popularity',
            'weight', 'weight_diff', # weightは補完していないので削除対象（学習時も削除している）
            'weight_diff_val', 'weight_diff_sign',
            'winning_numbers', 'payout', 'ticket_type',
            # PC-KEIBA specific
            'pass_1', 'pass_2', 'pass_3', 'pass_4',
            # Temporary
            'is_nige_temp'
        ]

        # ID情報は返却用に確保（レース情報も含める）
        id_cols = ['race_id', 'date', 'venue', 'race_number', 'horse_number', 'horse_name', 
                   'jockey_id', 'odds', 'popularity', 'title', 'distance', 'surface', 'state', 'weather']
        ids = inference_df[id_cols].copy()

        # X の作成
        X = inference_df.drop(columns=drop_cols, errors='ignore')
        
        # カテゴリ変数の除外 (数値のみ)
        X = X.select_dtypes(exclude=['object'])

        return X, ids
