import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    基本特徴量の生成を行うクラス。
    """
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カテゴリ変数の数値化や日付特徴量の追加を行います。

        Args:
            df (pd.DataFrame): クレンジング済みのデータ。

        Returns:
            pd.DataFrame: 特徴量が追加されたデータ。
        """
        logger.info("基本特徴量の生成を開始...")

        # 1. 性別の数値化
        # 牡->1, 牝->2, セ->3
        sex_map = {'牡': 1, '牝': 2, 'セ': 3}
        df['sex_num'] = df['sex'].map(sex_map).fillna(0).astype(int)

        # 2. 天候の数値化
        # 晴->1, 曇->2, 雨->3, 小雨->4, 雪->5
        weather_map = {'晴': 1, '曇': 2, '雨': 3, '小雨': 4, '雪': 5}
        # マップにないものは0 (Unknown)
        df['weather_num'] = df['weather'].map(weather_map).fillna(0).astype(int)

        # 3. 馬場種別の数値化
        # 芝->1, ダート->2, 障害->3
        surface_map = {'芝': 1, 'ダート': 2, '障害': 3}
        df['surface_num'] = df['surface'].map(surface_map).fillna(0).astype(int)

        # 4. 馬場状態の数値化
        # 良->1, 稍重->2, 重->3, 不良->4
        state_map = {'良': 1, '稍重': 2, '重': 3, '不良': 4}
        df['state_num'] = df['state'].map(state_map).fillna(0).astype(int)

        # 5. 日付情報の抽出
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday # 0:Monday, 6:Sunday

        # 6. クラスレベルの推定 (Class Level)
        # v8: 集計特徴量(CategoryAggregator)でも使用するため、ここで生成する
        def parse_class_level(title):
            if pd.isna(title): return 5 
            title = str(title)
            if 'G1' in title or 'ＧⅠ' in title or 'GⅠ' in title: return 9
            elif 'G2' in title or 'ＧⅡ' in title or 'GⅡ' in title: return 8
            elif 'G3' in title or 'ＧⅢ' in title or 'GⅢ' in title: return 7
            elif 'オープン' in title or 'OP' in title or 'L' in title: return 6
            elif '3勝' in title or '1600万' in title: return 5
            elif '2勝' in title or '1000万' in title: return 4
            elif '1勝' in title or '500万' in title: return 3
            elif '未勝利' in title: return 2
            elif '新馬' in title: return 1
            else: return 5

        if 'title' in df.columns:
            df['class_level'] = df['title'].apply(parse_class_level).astype('int8')
        else:
            logger.warning("titleカラムがないため、class_levelをデフォルト値(5)で埋めます")
            df['class_level'] = 5

        # ================================================================
        # 7. [v11 Extended N1] course_id: 競馬場×芝ダ×距離×回り
        # ================================================================
        # venue × surface × distance_cat × direction
        logger.info("v11: course_id（コース識別子）を生成中...")
        
        # 距離カテゴリ
        if 'distance' in df.columns:
            df['distance_cat'] = pd.cut(
                df['distance'], 
                bins=[0, 1399, 1899, 2399, 9999], 
                labels=['Sprint', 'Mile', 'Intermediate', 'Long']
            ).astype(str)
        else:
            df['distance_cat'] = 'Unknown'
        
        # 回り（mawariカラムがある場合）
        if 'mawari' in df.columns:
            df['direction'] = df['mawari'].fillna('不明').astype(str)
        else:
            df['direction'] = '不明'
        
        # course_id生成
        df['course_id'] = (
            df['venue'].astype(str) + '_' + 
            df['surface'].astype(str) + '_' + 
            df['distance_cat'] + '_' + 
            df['direction']
        )
        
        # カテゴリ型として保持
        df['course_id'] = df['course_id'].astype('category')
        
        logger.info(f"course_id ユニーク数: {df['course_id'].nunique()}")

        logger.info("基本特徴量の生成完了")
        return df

