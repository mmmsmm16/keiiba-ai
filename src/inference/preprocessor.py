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
from preprocessing.disadvantage_detector import DisadvantageDetector
from preprocessing.relative_features import RelativeFeatureEngineer
from preprocessing.cleansing import DataCleanser
from scipy.stats import entropy

logger = logging.getLogger(__name__)

def calculate_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース単位の特徴量を計算 (Betting Model用)
    Args:
        df: 'race_id', 'prob', 'odds' カラムを持つDataFrame
    Returns:
        DataFrame: 'entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses' を持つ1行のDF (1レース分の場合)
    """
    race_feats = []
    
    # 複数レースが含まれている場合に対応するため groupby
    for race_id, group in df.groupby('race_id'):
        probs = group['prob'].values
        odds = group['odds'].fillna(0).values
        
        # 1. Entropy (Confusion)
        ent = entropy(probs)
        
        # 2. Odds Volatility (Standard Deviation)
        odds_std = np.std(odds)
        
        # 3. Model Confidence (Max Prob)
        max_prob = np.max(probs)
        
        # 4. Confidence Gap (1st - 2nd)
        sorted_probs = sorted(probs, reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0
        
        # 5. Number of horses
        n_horses = len(group)
        
        race_feats.append({
            'race_id': race_id,
            'entropy': ent,
            'odds_std': odds_std,
            'max_prob': max_prob,
            'confidence_gap': gap,
            'n_horses': n_horses
        })
        
    return pd.DataFrame(race_feats)

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
        
        # =================================================================
        # Strategy 2a: Bloodline Features (Mapping)
        # =================================================================
        from preprocessing.bloodline_features import BloodlineFeatureEngineer
        bl_engineer = BloodlineFeatureEngineer(data_loader=None) # Loader不要(Mappingだけならhistory不要だが内部で使うかも?)
        # 実際にはMappingだけ欲しい。Loaderなしか、MockLoaderで...
        # BloodlineFeatureEngineerは内部でJraVanDataLoaderを作る。
        # ここではシンプルに、history_dfにある 'sire_id', 'bms_id' などを利用したいが、
        # new_df には horse_id しかない。
        # したがって、new_dfに対しても マッピングが必要。
        # BloodlineFeatureEngineerのadd_featuresの前半(mapping)だけ利用する。
        
        # history_df には既に sire_id, bms_id があるはず。
        # new_df にはないので、結合する必要がある。
        new_df = bl_engineer.add_features(new_df) 
        # add_featuresは集計も行うが、historyがないnew_df単体では集計値はNaN/0になる。
        # しかしマッピング(sire_id列などの追加)は行われる。
        # 集計カラム (sire_avg_rankなど) は後で上書きする。

        # =================================================================
        # Strategy 2b: Category & Context Features (Lookup & Update)
        # =================================================================
        logger.info("カテゴリ・コンテキスト特徴量のインクリメンタル更新中...")

        # ----------------------------------------------------------------
        # 準備: history_df 側の条件カラム復元 (保存時にdropされているため)
        # ----------------------------------------------------------------
        # distance_cat
        if 'distance' in history_df.columns and 'distance_cat' not in history_df.columns:
            # history_dfは巨大な可能性があるので、必要な時だけ計算するか、
            # ここで一括でやるか。一括の方が安全。
            # SettingWithCopyWarning回避のためcopyするか、直接代入するか。
            # 推論用なのでメモリ余裕あればcopy推奨だが、遅くなる。
            # ここでは参照渡しされたhistory_dfを直接変更しないよう、ローカルで扱うが、
            # Pandasの挙動として列追加はcopy発生することが多い。
            # 暫定的に: distance_catが必要なのは sire_dist のみ。
            pass # 個別のupdate関数内で処理、あるいはここでやる。

        # new_df 側
        if 'distance' in new_df.columns:
            new_df['distance_cat'] = pd.cut(
                new_df['distance'], 
                bins=[0, 1399, 1899, 2399, 9999], 
                labels=['Sprint', 'Mile', 'Intermediate', 'Long']
            )
            new_df['distance_cat'] = new_df['distance_cat'].astype(str)

        # -----------------------------------------------------
        # Helper: Generic Lookup & Update
        # -----------------------------------------------------
        def update_incremental_stats(target_df, source_history, keys, prefix, metrics, is_bloodline=False):
            # 必要なキーがソースにあるか確認
            # distance_catなどの動的カラム対応
            temp_history = source_history
            
            # distance_cat がキーに含まれていて、ソースにない場合は一時的に作成
            if 'distance_cat' in keys and 'distance_cat' not in source_history.columns:
                if 'distance' in source_history.columns:
                    # viewへの代入を避けるため、必要な列だけ切り出して作成
                    # メモリ効率のため、全体コピーは避ける
                    # しかし drop_duplicates するので、結局 subset を作る。
                    # ここでは「必要な列 + distance」でsubsetを作って計算する。
                    cols_needed = list(set(keys + [f'{prefix}_{m}' for m in metrics] + ['distance']))
                    # 存在する列だけ (distanceは確認済み、statsはあるか？)
                    cols_needed = [c for c in cols_needed if c in source_history.columns]
                    
                    temp_history = source_history[cols_needed].copy()
                    temp_history['distance_cat'] = pd.cut(
                        temp_history['distance'], 
                        bins=[0, 1399, 1899, 2399, 9999], 
                        labels=['Sprint', 'Mile', 'Intermediate', 'Long']
                    ).astype(str)
                else:
                    logger.warning(f"[{prefix}] distanceカラムがないため distance_cat を生成できず、スキップします。")
                    return

            # Define exact column names expected in history
            hist_cols = keys + [f'{prefix}_{m}' for m in metrics]
            missing = [c for c in hist_cols if c not in temp_history.columns]
            if missing:
                logger.warning(f"[{prefix}] Missing columns in history: {missing}. Skiping.")
                return
                
            # Drop duplicates keeping last
            latest_stats = temp_history.drop_duplicates(subset=keys, keep='last')[hist_cols].set_index(keys)
            
            # Merge to target_df (new_df)
            merged = target_df[keys].join(latest_stats, on=keys, how='left')
            
            # Update/Copy
            if is_bloodline:
                 # Bloodline Logic (Direct Copy as explained)
                 target_df[f'{prefix}_count'] = merged[f'{prefix}_count'].fillna(0)
                 target_df[f'{prefix}_win_rate'] = merged[f'{prefix}_win_rate'].fillna(0)
                 target_df[f'{prefix}_roi_rate'] = merged[f'{prefix}_roi_rate'].fillna(0)
                 target_df[f'{prefix}_avg_rank'] = merged[f'{prefix}_avg_rank'].fillna(0)
            else:
                 # Category Logic (Direct Copy)
                 target_df[f'{prefix}_n_races'] = merged[f'{prefix}_n_races'].fillna(0)
                 target_df[f'{prefix}_win_rate'] = merged[f'{prefix}_win_rate'].fillna(0)
                 target_df[f'{prefix}_top3_rate'] = merged[f'{prefix}_top3_rate'].fillna(0)
            
            # Check success
            new_col = f'{prefix}_win_rate'
            if new_col in target_df.columns:
                pass # logger.info(f"[{prefix}] Added features successfully.")
            else:
                logger.warning(f"[{prefix}] Failed to add features.")

        # -----------------------------------------------------
        # Execution
        # -----------------------------------------------------
        
        # 1. Basic Categories
        # Jockey, Trainer, Sire
        update_incremental_stats(new_df, history_df, ['jockey_id'], 'jockey_id', ['n_races', 'win_rate', 'top3_rate'])
        update_incremental_stats(new_df, history_df, ['trainer_id'], 'trainer_id', ['n_races', 'win_rate', 'top3_rate'])
        update_incremental_stats(new_df, history_df, ['sire_id'], 'sire_id', ['n_races', 'win_rate', 'top3_rate'])
        
        # 2. Context Categories
        # (1) Jockey x Course
        if 'venue' in new_df.columns:
            update_incremental_stats(new_df, history_df, ['jockey_id', 'venue'], 'jockey_course', ['n_races', 'win_rate', 'top3_rate'])
            update_incremental_stats(new_df, history_df, ['sire_id', 'venue'], 'sire_course', ['n_races', 'win_rate', 'top3_rate'])
            update_incremental_stats(new_df, history_df, ['trainer_id', 'venue'], 'trainer_course', ['n_races', 'win_rate', 'top3_rate'])
            
        # (2) Sire x Distance
        if 'distance_cat' in new_df.columns:
            update_incremental_stats(new_df, history_df, ['sire_id', 'distance_cat'], 'sire_dist', ['n_races', 'win_rate', 'top3_rate'])
            
        # (3) Sire x Track
        if 'surface' in new_df.columns:
             update_incremental_stats(new_df, history_df, ['sire_id', 'surface'], 'sire_track', ['n_races', 'win_rate', 'top3_rate'])
             # NEW v3
             update_incremental_stats(new_df, history_df, ['jockey_id', 'surface'], 'jockey_surface', ['n_races', 'win_rate', 'top3_rate'])
             update_incremental_stats(new_df, history_df, ['trainer_id', 'surface'], 'trainer_surface', ['n_races', 'win_rate', 'top3_rate'])

        # NEW v3: Distance Context
        if 'distance_cat' in new_df.columns:
             # sire_dist is already handled above at line 226, but let's regroup if needed. 
             # Actually line 226 handles sire_dist. I'll add jockey/trainer here.
             update_incremental_stats(new_df, history_df, ['jockey_id', 'distance_cat'], 'jockey_dist', ['n_races', 'win_rate', 'top3_rate'])
             update_incremental_stats(new_df, history_df, ['trainer_id', 'distance_cat'], 'trainer_dist', ['n_races', 'win_rate', 'top3_rate'])
             
        # (4) Jockey x Trainer
        update_incremental_stats(new_df, history_df, ['jockey_id', 'trainer_id'], 'jockey_trainer', ['n_races', 'win_rate', 'top3_rate'])

        # 3. Bloodline Features
        # Sire, BMS
        # Prefix: sire, bms (metrics: avg_rank, win_rate, roi_rate, count)
        # Note: 'rank' access required in generic func? No, generic func removed 'rank' dependency for direct copy.
        update_incremental_stats(new_df, history_df, ['sire_id'], 'sire', ['avg_rank', 'win_rate', 'roi_rate', 'count'], is_bloodline=True)
        update_incremental_stats(new_df, history_df, ['bms_id'], 'bms', ['avg_rank', 'win_rate', 'roi_rate', 'count'], is_bloodline=True)


        # =================================================================
        # Merge Category Features to Combined DF
        # =================================================================
        # combined_horse_df is composed of [history; new_df].
        # The history part ALREADY has these features calculated (static in parquet).
        # The new_df part (at the end) has NaN.
        # We just calculated the values for new_df in `new_df`.
        # We process the merge as before.
        
        target_ids = new_df['race_id'].unique()
        combined_horse_df['temp_key'] = combined_horse_df['race_id'].astype(str) + '_' + combined_horse_df['horse_number'].astype(str)
        new_df['temp_key'] = new_df['race_id'].astype(str) + '_' + new_df['horse_number'].astype(str)
        
        # Collect all columns we just added
        # Basic
        cols_to_merge = []
        for p in ['jockey_id', 'trainer_id', 'sire_id']:
             cols_to_merge.extend([f'{p}_n_races', f'{p}_win_rate', f'{p}_top3_rate'])
        # Context
        for p in ['jockey_course', 'sire_course', 'trainer_course', 'sire_dist', 'sire_track', 'jockey_trainer',
                  'jockey_surface', 'trainer_surface', 'jockey_dist', 'trainer_dist']:
             cols_to_merge.extend([f'{p}_n_races', f'{p}_win_rate', f'{p}_top3_rate'])
        # Bloodline
        for p in ['sire', 'bms']:
             cols_to_merge.extend([f'{p}_avg_rank', f'{p}_win_rate', f'{p}_roi_rate', f'{p}_count'])
             
        # Only existing ones
        cols_to_merge = [c for c in cols_to_merge if c in new_df.columns]
        
        # Mapping
        for col in cols_to_merge:
            val_map = new_df.set_index('temp_key')[col].to_dict()
            # Update only new_df rows in combined (fill nans)
            combined_horse_df[col] = combined_horse_df[col].fillna(combined_horse_df['temp_key'].map(val_map))
           
        combined_horse_df.drop('temp_key', axis=1, inplace=True)
        new_df.drop('temp_key', axis=1, inplace=True)

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
        # 3.5. 不利検出特徴量生成 (Phase 11.1新規)
        # =================================================================
        disadv_detector = DisadvantageDetector()
        combined_horse_df = disadv_detector.add_features(combined_horse_df)

        # =================================================================
        # 3.6. 相対的特徴量生成 (Phase 11.1新規)
        # =================================================================
        relative_engineer = RelativeFeatureEngineer()
        combined_horse_df = relative_engineer.add_features(combined_horse_df)

        # =================================================================
        # 3.7. リアルタイム特徴量生成 (v9新規 - 当日の傾向)
        # =================================================================
        # 推論時は Loader が全レース(過去レース含む)を取得していることを前提とする
        from preprocessing.realtime_features import RealTimeFeatureEngineer
        realtime_engineer = RealTimeFeatureEngineer()
        combined_horse_df = realtime_engineer.add_features(combined_horse_df)

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
            'weight', 
            # 'weight_diff', # 学習時に使用しているため残す
            'weight_diff_val', 'weight_diff_sign',
            'winning_numbers', 'payout', 'ticket_type',
            # PC-KEIBA specific
            'pass_1', 'pass_2', 'pass_3', 'pass_4',
            # Temporary
            'is_nige_temp',
            
            # --- Leakage Features (Phase 11.1 fix: Sync with dataset.py) ---
            'slow_start_recovery', 'pace_disadvantage', 'wide_run',
            'track_bias_disadvantage', 'outer_frame_disadv',
            'odds_race_rank', 'popularity_race_rank',
            'odds_deviation', 'popularity_deviation'
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
