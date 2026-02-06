
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)

def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    対戦成績特徴量 (Head-to-Head) を計算する (Optimized Iterative Approach)。
    
    以前の global merge 方式 (O(N^2) rows) はメモリ枯渇するため、
    レースを時系列順に処理し、疎な辞書で対戦成績を管理する方式に変更。
    
    Complexity: O(R * M^2)
      R: レース数 (~350k)
      M: 出走頭数 (~14)
      Total ops: ~70M (Python loopで数分〜10分程度)
      
    Features:
    - vs_rival_win_rate: 今回のライバルたちとの過去対戦における通算勝率
    - vs_rival_match_count: 今回のライバルたちとの過去対戦数の合計
    
    Args:
        df: FeaturePipelineから渡されるDataFrame (全期間)
        
    Returns:
        pd.DataFrame: 追加特徴量を持つDataFrame (keys + features)
    """
    logger.info("Computing Head-to-Head features (Iterative Sparse Update)...")
    
    # 必要なカラム確認
    req_cols = ['race_id', 'horse_id', 'horse_number', 'rank', 'date']
    if not all(c in df.columns for c in req_cols):
        logger.warning("Missing required columns for Head-to-Head. Returning empty.")
        return df[['race_id', 'horse_number', 'horse_id']].copy()
    
    # 1. 準備: データセットを軽量化・ソート
    df_base = df[req_cols].copy()
    
    # rank欠損は勝敗不明なので除外するが、予測対象としては残す必要がある。
    # ここでは「学習用データ構築」と「予測用」を兼ねるため、
    # nan rankの行も含めてループするが、state update時には nan rank の馬は勝敗更新に参加させない。
    
    if not np.issubdtype(df_base['date'].dtype, np.datetime64):
         df_base['date'] = pd.to_datetime(df_base['date'])
         
    # 時系列ソート (必須)
    df_base = df_base.sort_values(['date', 'race_id'])
    
    # 2. 高速化: Horse ID to Integer Map
    # 文字列の辞書検索よりIntの方が速くメモリも軽い
    unique_horses = df_base['horse_id'].unique()
    horse_to_idx = {hid: i for i, hid in enumerate(unique_horses)}
    df_base['h_idx'] = df_base['horse_id'].map(horse_to_idx)
    
    # 3. 状態管理用辞書 (Sparse Matrix substitute)
    # Key: (winner_idx, loser_idx) -> counts
    # wins[(hA, hB)] = AがBに勝った回数
    # matches[(hA, hB)] = AとBが対戦した回数 (無向グラフだが、キーは順列で持つか、正規化して持つか)
    # ここでは Directed Key (hA, hB) で管理する方が直感的。
    # matches[(hA, hB)] == matches[(hB, hA)] だが、取得時に query(hA, hB) するので両方向更新or片側管理。
    # 片側管理 (min, max) にして look up 時に sort するのがメモリ効率良い。
    
    pair_matches = defaultdict(int) # Key: tuple(sorted(h1, h2)) -> count
    pair_wins = defaultdict(int)    # Key: (h_winner, h_loser) -> count
    
    # 結果格納用リスト
    result_win_rates = []
    result_match_counts = []
    
    # 4. レースごとのイテレーション
    # race_id でグルーピングしたいが、df.groupby は遅い可能性がある。
    # 既にソート済みなので、itertuples で回しつつ race_id の変化を検知する方が速い、
    # あるいは race_id ごとの index list を取得するか。
    # メモリ節約のため、GroupByオブジェクトを回す。
    
    # Tqdmで進捗表示
    grouped = df_base.groupby('race_id', sort=False)
    
    for race_id, group in tqdm(grouped, desc="Processing Races", mininterval=5.0):
        # group: そのレースの出走馬一覧
        # vectors
        h_indices = group['h_idx'].values
        ranks = group['rank'].values
        
        n_horses = len(h_indices)
        
        # --- A. 特徴量取得 (State Query) ---
        # この時点での過去成績を取得
        
        race_win_rates = np.zeros(n_horses, dtype=np.float32)
        race_match_counts = np.zeros(n_horses, dtype=np.float32)
        
        if n_horses > 1:
            for i in range(n_horses):
                h1 = h_indices[i]
                total_wins = 0
                total_matches = 0
                
                for j in range(n_horses):
                    if i == j: continue
                    h2 = h_indices[j]
                    
                    # Match Count
                    pair_key = (h1, h2) if h1 < h2 else (h2, h1)
                    m_count = pair_matches.get(pair_key, 0)
                    
                    if m_count > 0:
                        total_matches += m_count
                        # Win Count (h1 beats h2)
                        total_wins += pair_wins.get((h1, h2), 0)
                
                if total_matches > 0:
                    race_win_rates[i] = total_wins / total_matches
                    race_match_counts[i] = total_matches
                    
        # 結果を保持 (元のIndex順)
        # 1行ずつ追加するより、一括で保管してあとで結合する方が速いが、
        # ここではリストに追加し、あとでDataFrame化する。
        # group.index を使って紐付ける。
        
        # --- B. 状態更新 (State Update) ---
        # ランクが確定している馬のみ対象
        
        # 有効なランクを持つ馬のインデックス
        valid_mask = ~np.isnan(ranks)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 1:
            # O(M^2) loop
            # 全ペアについて更新
            for k1 in range(len(valid_indices)):
                idx1 = valid_indices[k1]
                h1 = h_indices[idx1]
                r1 = ranks[idx1]
                
                for k2 in range(k1 + 1, len(valid_indices)):
                    idx2 = valid_indices[k2]
                    h2 = h_indices[idx2]
                    r2 = ranks[idx2]
                    
                    # Match Count Update (Shared)
                    pair_key = (h1, h2) if h1 < h2 else (h2, h1)
                    pair_matches[pair_key] += 1
                    
                    # Win Update
                    if r1 < r2:
                        pair_wins[(h1, h2)] += 1
                    elif r2 < r1:
                        pair_wins[(h2, h1)] += 1
                    # 同着 (r1 == r2) はどちらも加算しない (=引き分け)
        
        # 結果リストへの追加 (順序は group の行順)
        result_win_rates.extend(race_win_rates)
        result_match_counts.extend(race_match_counts)

    # 5. DataFrame結合
    # df_base (sorted) の順序でループしたので、そのまま結合可能
    df_base['vs_rival_win_rate'] = result_win_rates
    df_base['vs_rival_match_count'] = result_match_counts
    
    # 欠損値は0埋め (既に0だが念のため)
    df_base['vs_rival_win_rate'] = df_base['vs_rival_win_rate'].fillna(0.0)
    df_base['vs_rival_match_count'] = df_base['vs_rival_match_count'].fillna(0.0)
    
    # 元の識別キーのみ残して返す
    # 注意: 入力の df と並び順が変わっている (sort_valuesしたため)
    # 呼び出し元でマージしやすいよう、キーカラム + 特徴量 を返す
    
    ret_cols = ['race_id', 'horse_number', 'horse_id', 'vs_rival_win_rate', 'vs_rival_match_count']
    return df_base[ret_cols].copy()

def get_feature_names():
    return ['vs_rival_win_rate', 'vs_rival_match_count']
