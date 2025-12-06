import pandas as pd
import logging
from .loader import JraVanDataLoader

logger = logging.getLogger(__name__)

class BloodlineFeatureEngineer:
    """
    血統（種牡馬・繁殖牝馬）に関する特徴量を生成するクラス。
    PC-KEIBAのjvd_um（競走馬マスタ）テーブルから血統情報を取得し、
    種牡馬・繁殖牝馬ごとの集計特徴量を追加します。
    """
    def __init__(self, data_loader: JraVanDataLoader = None):
        """
        Args:
            data_loader: JraVanDataLoaderのインスタンス。
                         指定がない場合は新規に作成しますが、
                         パフォーマンスのため、既存のインスタンスを渡すことを推奨します。
        """
        self.loader = data_loader if data_loader else JraVanDataLoader()
        self.bloodline_map = None

    def _load_bloodline_data(self):
        """
        競走馬マスタ(jvd_um)から血統情報をロードし、マッピング辞書を作成します。
        """
        if self.bloodline_map is not None:
            return

        logger.info("競走馬マスタから血統情報をロード中...")
        # jvd_umから必要なカラムを取得
        # ketto_joho_01a: 父馬コード (Sire ID)
        # ketto_joho_02a: 母馬コード (Mare ID)
        # ketto_joho_03a: 母父馬コード (Broodmare Sire ID)
        query = """
        SELECT
            ketto_touroku_bango AS horse_id,
            ketto_joho_01a AS sire_id,
            ketto_joho_02a AS mare_id,
            ketto_joho_03a AS bms_id
        FROM jvd_um
        """
        try:
            df_um = self.loader.load_data(query)
            # 重複排除（念のため）
            df_um = df_um.drop_duplicates(subset=['horse_id'])

            self.bloodline_map = df_um.set_index('horse_id')[['sire_id', 'mare_id', 'bms_id']]
            logger.info(f"血統情報ロード完了: {len(self.bloodline_map)}頭")
        except Exception as e:
            logger.error(f"血統情報のロードに失敗しました: {e}")
            self.bloodline_map = pd.DataFrame() # 空のDFで続行

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        血統特徴量を追加します。

        Args:
            df (pd.DataFrame): 前処理中のデータセット (horse_idが含まれていること)

        Returns:
            pd.DataFrame: 血統特徴量が追加されたデータセット
        """
        logger.info("血統特徴量の生成を開始...")

        # 1. 血統情報の結合
        self._load_bloodline_data()

        if self.bloodline_map.empty:
            logger.warning("血統情報が空のため、スキップします。")
            return df

        # データセットにsire_id, mare_id, bms_idを結合
        # 左結合で、情報がない馬はNaNになる
        df = df.merge(self.bloodline_map, on='horse_id', how='left')

        # 2. ターゲットエンコーディング的な集計 (リーク防止のため時系列shiftが必要)

        # 集計対象: rank (着順), win_flag (1着かどうか), top3_flag (3着以内)
        # rankはNaN（未来のレース）の場合があるため除外して計算する必要がある

        # NOTE: Datasetクラスで処理される際に、Train/Test分割やリーク防止が行われるが、
        # ここで全期間の集計をしてしまうとリークになる。
        # そのため、CategoryAggregatorと同様のロジック（shift -> expanding -> mean）を
        # sire_id, bms_id に対して適用する。

        # 必要なカラムがあるか確認
        required_cols = ['date', 'rank']
        if not all(col in df.columns for col in required_cols):
             logger.warning(f"必要なカラム {required_cols} が不足しているため、集計特徴量はスキップします。")
             return df

        # 着順の数値化
        df['rank_numeric'] = pd.to_numeric(df['rank'], errors='coerce')
        df['is_win'] = (df['rank_numeric'] == 1).astype(int)
        df['is_top3'] = (df['rank_numeric'] <= 3).astype(int)

        # ソート (CategoryAggregatorと同様)
        df = df.sort_values(['date'])

        # 集計関数
        def calculate_expanding_stats(group_col, prefix):
            # shift(1)して、その行以前（その行は含まない）の累積平均を計算

            # 平均着順
            df[f'{prefix}_avg_rank'] = df.groupby(group_col)['rank_numeric'].transform(
                lambda x: x.shift(1).expanding().mean()
            )

            # 勝率
            df[f'{prefix}_win_rate'] = df.groupby(group_col)['is_win'].transform(
                lambda x: x.shift(1).expanding().mean()
            )

            # 複勝率
            df[f'{prefix}_roi_rate'] = df.groupby(group_col)['is_top3'].transform(
                lambda x: x.shift(1).expanding().mean()
            )

            # 出走回数（信頼度用）
            df[f'{prefix}_count'] = df.groupby(group_col)['rank_numeric'].transform(
                lambda x: x.shift(1).expanding().count()
            )

        # 種牡馬(Sire)の集計
        logger.info("種牡馬(Sire)別の集計特徴量を生成中...")
        calculate_expanding_stats('sire_id', 'sire')

        # 母父(BMS)の集計
        logger.info("母父(BMS)別の集計特徴量を生成中...")
        calculate_expanding_stats('bms_id', 'bms')

        # 一時カラムの削除
        df.drop(columns=['rank_numeric', 'is_win', 'is_top3'], inplace=True)

        # 欠損値埋め: 初出走の種牡馬などはNaNになる -> 0で埋める
        cols_to_fill = [c for c in df.columns if c.startswith('sire_') or c.startswith('bms_')]
        df[cols_to_fill] = df[cols_to_fill].fillna(0)

        logger.info("血統特徴量の生成完了")
        return df
