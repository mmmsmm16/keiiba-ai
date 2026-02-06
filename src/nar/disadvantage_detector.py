import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DisadvantageDetector:
    """
    近走での不利パターンを検出し、特徴量化するクラス。
    
    検出パターン:
    1. 展開不利: 逃げ馬が多すぎて前に行けなかった
    2. 出遅れ: スタートで大きく出遅れた
    3. 外々回し: 常に外枠を走らされた
    4. 馬場不利: 内枠が圧倒的に有利な馬場で外枠を引いた
    """
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """不利検出特徴量を追加"""
        logger.info("不利検出特徴量の生成を開始...")
        
        # 時系列ソート
        df = df.sort_values(['horse_id', 'date'])
        
        # =============================================================
        # 1. 出遅れ検出
        # =============================================================
        df = self._detect_slow_start(df)
        
        # =============================================================
        # 2. 展開不利検出
        # =============================================================
        df = self._detect_pace_disadvantage(df)
        
        # =============================================================
        # 3. 外々回し検出
        # =============================================================
        df = self._detect_wide_run(df)
        
        # =============================================================
        # 4. 馬場不利検出
        # =============================================================
        df = self._detect_track_bias(df)
        
        # =============================================================
        # 5. 総合不利スコア
        # =============================================================
        df = self._calculate_disadvantage_score(df)
        
        logger.info("不利検出特徴量の生成完了")
        return df
    
    def _detect_slow_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        出遅れ検出
        
        ロジック:
        - passing_rank の最初のコーナー順位が極端に悪い
        - しかし最終的な着順はそこまで悪くない
        → 出遅れたが追い込んだ = 本来の実力以上の結果
        """
        def analyze_start(row):
            # passing_rank: "1-2-3-4" or "10-9-8-7"
            passing = row.get('passing_rank')
            rank = row.get('rank')
            
            if pd.isna(passing) or pd.isna(rank):
                return 0
            
            try:
                # 最初のコーナー順位
                corners = passing.split('-')
                if not corners:
                    return 0
                
                first_corner = int(corners[0])
                
                # レース内頭数の推定（race_id ごとのカウント）
                # この関数内では取得できないので、保守的に18頭と仮定
                # または passing_rank から推定
                n_horses = max(int(c) for c in corners if c.isdigit()) if len(corners) > 0 else 18
                
                # 出遅れ判定: 最初のコーナーで後方（下位50%以下）
                if first_corner > n_horses / 2:
                    # しかし最終着順は中団以上（上位60%以内）
                    if rank <= n_horses * 0.6:
                        # 出遅れ挽回フラグ
                        return 1
                
                return 0
            except:
                return 0
        
        df['slow_start_recovery'] = df.apply(analyze_start, axis=1)
        
        # 馬ごとの過去の出遅れ挽回率
        grouped = df.groupby('horse_id')
        df['horse_slow_start_rate'] = grouped['slow_start_recovery'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        return df
    
    def _detect_pace_disadvantage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        展開不利検出
        
        ロジック:
        - 逃げ馬が多すぎる（race_nige_horse_count >= 3）
        - 自分も前に行きたい馬（nige_rate > 0.3）
        → ポジション争いで不利
        """
        # race_nige_horse_count は AdvancedFeatureEngineer で計算済み前提
        # もしなければ、ここで再計算
        if 'race_nige_horse_count' not in df.columns:
            # nige_rate が存在するか確認
            if 'nige_rate' in df.columns:
                df['race_nige_horse_count'] = df.groupby('race_id')['nige_rate'].transform(
                    lambda x: (x >= 0.5).sum()
                )
            else:
                logger.warning("nige_rate カラムが見つかりません。展開不利検出をスキップします。")
                df['pace_disadvantage'] = 0
                df['horse_pace_disadv_rate'] = 0
                return df
        
        # nige_rate が存在しない場合の処理
        if 'nige_rate' not in df.columns:
            logger.warning("nige_rate カラムが見つかりません。展開不利検出をスキップします。")
            df['pace_disadvantage'] = 0
            df['horse_pace_disadv_rate'] = 0
            return df
        
        # 展開不利フラグ
        df['pace_disadvantage'] = (
            (df['race_nige_horse_count'] >= 3) &  # 逃げ馬が多い
            (df['nige_rate'] > 0.3)  # 自分も前に行きたい
        ).astype(int)
        
        # 馬ごとの過去の展開不利経験率
        df['horse_pace_disadv_rate'] = df.groupby('horse_id')['pace_disadvantage'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        return df
    
    def _detect_wide_run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        外々回し検出
        
        ロジック:
        - 通過順位の変動が大きい（内外を行き来）
        - または外枠スタート（frame_number >= 7）で不利
        """
        def analyze_wide_run(row):
            passing = row.get('passing_rank')
            
            if pd.isna(passing):
                return 0
            
            try:
                corners = [int(c) for c in passing.split('-') if c.isdigit()]
                
                if len(corners) < 2:
                    return 0
                
                # 順位変動（標準偏差が大きい = 内外を行き来）
                std = np.std(corners)
                
                # 標準偏差が3以上 = 大きく位置取りが変わった
                if std >= 3:
                    return 1
                
                return 0
            except:
                return 0
        
        df['wide_run'] = df.apply(analyze_wide_run, axis=1)
        
        # 外枠不利フラグ（外枠 + 結果が悪い）
        if 'frame_number' in df.columns:
            df['outer_frame_disadv'] = (
                (df['frame_number'] >= 7) &  # 外枠
                (df['rank'] > 8)  # 結果が悪い
            ).astype(int)
        else:
            df['outer_frame_disadv'] = 0
        
        # 馬ごとの外々回し率
        df['horse_wide_run_rate'] = df.groupby('horse_id')['wide_run'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        return df
    
    def _detect_track_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        馬場不利検出
        
        ロジック:
        - そのレースで内枠/外枠に極端な有利不利があった場合を検出
        - レース内での枠番別平均着順を計算
        """
        if 'frame_number' not in df.columns:
            df['track_bias_disadvantage'] = 0
            df['horse_track_bias_rate'] = 0
            return df
        
        # レースごとの枠番別平均着順
        frame_stats = df.groupby(['race_id', 'frame_number'])['rank'].mean().reset_index()
        frame_stats.columns = ['race_id', 'frame_number', 'frame_avg_rank']
        
        df = df.merge(frame_stats, on=['race_id', 'frame_number'], how='left')
        
        # レース全体の平均着順
        race_avg = df.groupby('race_id')['rank'].transform('mean')
        
        # 自分の枠の平均着順が全体平均より2以上悪い = 馬場不利
        df['track_bias_disadvantage'] = (
            (df['frame_avg_rank'] - race_avg) > 2
        ).astype(int)
        
        # 馬ごとの馬場不利経験率
        df['horse_track_bias_rate'] = df.groupby('horse_id')['track_bias_disadvantage'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0)
        
        # 一時カラム削除
        df.drop(columns=['frame_avg_rank'], inplace=True, errors='ignore')
        
        return df
    
    def _calculate_disadvantage_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        総合不利スコアの計算
        
        直近1走での不利の合計 = 今回はチャンスの可能性
        """
        # 各不利フラグのカラム
        disadv_cols = [
            'slow_start_recovery',
            'pace_disadvantage',
            'wide_run',
            'outer_frame_disadv',
            'track_bias_disadvantage'
        ]
        
        # 存在するカラムのみ使用
        available_cols = [c for c in disadv_cols if c in df.columns]
        
        if not available_cols:
            logger.warning("不利フラグが1つも生成されませんでした。")
            df['prev_disadvantage_score'] = 0
            df['avg_disadvantage_score_3races'] = 0
            return df
        
        # 直近1走での不利スコア（過去走の不利を持ち越し）
        for col in available_cols:
            if col not in df.columns:
                df[col] = 0
        
        # 一時的に合計列を作成
        df['temp_disadv_sum'] = df[available_cols].sum(axis=1)
        
        # shift して前走の不利スコアを取得
        df['prev_disadvantage_score'] = df.groupby('horse_id')['temp_disadv_sum'].shift(1).fillna(0)
        
        # 直近3走での平均不利スコア
        df['avg_disadvantage_score_3races'] = df.groupby('horse_id')['temp_disadv_sum'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        ).fillna(0)
        
        # 一時カラム削除
        df.drop(columns=['temp_disadv_sum'], inplace=True, errors='ignore')
        
        return df
