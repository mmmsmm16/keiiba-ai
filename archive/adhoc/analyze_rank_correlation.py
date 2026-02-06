import pandas as pd
import numpy as np
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def analyze_rank_correlation():
    base_dir = os.path.join(os.path.dirname(__file__), '../../../experiments')
    pred_path = os.path.join(base_dir, 'v7_ensemble_full/reports/predictions.parquet')
    
    print(f"Loading predictions from {pred_path}...")
    df = pd.read_parquet(pred_path)
    
    # 2025年のデータのみ
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year == 2025].copy()
        
    print(f"Data Loaded: {len(df)} rows")
    
    # 実際順位(rank)があるか確認
    if 'rank' not in df.columns:
        print("Error: 'rank' column not found.")
        return
        
    # 分析用データ構造
    # レース単位で処理
    race_ids = df['race_id'].unique()
    
    target_races = 0
    top1_ranks = [] # AI 1位の実際の着順
    
    # 逃したパターンの集計
    miss_axis_winner_rank = [] # 軸抜け時の勝ち馬のAI予測ランク
    miss_opp_2nd_rank = []     # 相手抜け時の実際2着馬のAI予測ランク
    miss_opp_3rd_rank = []     # 相手抜け時の実際3着馬のAI予測ランク
    
    hit_axis_1st = 0 # 軸が1着
    hit_axis_2nd = 0 # 軸が2着
    hit_axis_3rd = 0 # 軸が3着
    hit_axis_out = 0 # 軸が4着以下
    
    hit_perfect = 0 # 今の戦略で的中
    
    potential_hit_multi = 0 # 軸が1-3着で、相手に他の2頭が含まれていたケース（マルチなら当たり）
    
    print(f"Analyzing {len(race_ids)} races...")
    
    for race_id in race_ids:
        race_df = df[df['race_id'] == race_id].copy()
        race_df = race_df.sort_values('score', ascending=False)
        
        if len(race_df) < 5: continue
        
        # AI予測1位の情報
        top1 = race_df.iloc[0]
        top1_pop = top1['popularity'] if pd.notna(top1['popularity']) else 1
        
        # 条件: 予測1位が4番人気以上
        if top1_pop < 4: continue
        
        target_races += 1
        
        # AI 1位の実際の着順
        actual_rank_top1 = top1['rank']
        top1_ranks.append(actual_rank_top1)
        
        if actual_rank_top1 == 1:
            hit_axis_1st += 1
        elif actual_rank_top1 == 2:
            hit_axis_2nd += 1
        elif actual_rank_top1 == 3:
            hit_axis_3rd += 1
        else:
            hit_axis_out += 1
            
        # 実際の1,2,3着馬を特定
        actual_1st = race_df[race_df['rank'] == 1]
        actual_2nd = race_df[race_df['rank'] == 2]
        actual_3rd = race_df[race_df['rank'] == 3]
        
        # AIランク（score順のindex + 1）
        # race_dfはscore降順なので、ilocのindexがそのままランク
        # ただし、行を取り出してindexを見つける必要がある
        
        def get_ai_rank(row_series):
            if row_series.empty: return 99
            # race_df内での位置を探す
            idx = race_df.index.get_loc(row_series.index[0])
            return idx + 1 # 1-based
            
        ai_rank_1st = get_ai_rank(actual_1st)
        ai_rank_2nd = get_ai_rank(actual_2nd)
        ai_rank_3rd = get_ai_rank(actual_3rd)
        
        # 今の相手: 予測2-5位 (AI Rank 2, 3, 4, 5)
        opp_ranks = [2, 3, 4, 5]
        
        # Case分析
        
        # 1. 軸抜け (AI 1位 != 1着)
        if actual_rank_top1 != 1:
            # 勝ち馬はAI予測何位だったか？
            miss_axis_winner_rank.append(ai_rank_1st)
            
        # 2. 完全的中チェック (AI 1位 == 1着 AND 2,3着がOppに含まれる)
        is_hit = False
        if actual_rank_top1 == 1:
            if ai_rank_2nd in opp_ranks and ai_rank_3rd in opp_ranks:
                hit_perfect += 1
                is_hit = True
            else:
                # 相手抜け
                if ai_rank_2nd not in opp_ranks: miss_opp_2nd_rank.append(ai_rank_2nd)
                if ai_rank_3rd not in opp_ranks: miss_opp_3rd_rank.append(ai_rank_3rd)
        
        # 3. マルチ可能性 (AI 1位が1-3着)
        if actual_rank_top1 <= 3:
            # 軸以外の2頭が相手(2-5位)に含まれているか？
            # 実際の1,2,3着のAIランクセット
            actual_top3_ai_ranks = {ai_rank_1st, ai_rank_2nd, ai_rank_3rd}
            # AI 1位(ランク1)を除く
            others = actual_top3_ai_ranks - {1}
            
            # othersの全ての要素が opp_ranks に含まれていればマルチ的中
            if others.issubset(set(opp_ranks)):
                potential_hit_multi += 1

    print("\n" + "="*50)
    print("📊 AI予測ランク相関分析 (pop >= 4 のレース)")
    print("="*50)
    print(f"対象レース数: {target_races}")
    print("-" * 30)
    print("🎯 AI予測1位の着順分布")
    print(f"  1着: {hit_axis_1st}回 ({hit_axis_1st/target_races*100:.1f}%)")
    print(f"  2着: {hit_axis_2nd}回 ({hit_axis_2nd/target_races*100:.1f}%)")
    print(f"  3着: {hit_axis_3rd}回 ({hit_axis_3rd/target_races*100:.1f}%)")
    print(f"  着外: {hit_axis_out}回 ({hit_axis_out/target_races*100:.1f}%)")
    print(f"  (馬券内率: {(hit_axis_1st+hit_axis_2nd+hit_axis_3rd)/target_races*100:.1f}%)")
    print("-" * 30)
    print("❌ 軸抜け時の勝ち馬 (AI予測ランク分布)")
    # 頻度上位を表示
    if miss_axis_winner_rank:
        s = pd.Series(miss_axis_winner_rank)
        print(s.value_counts().sort_index().head(10))
    print("-" * 30)
    print("❌ 相手抜け時の2着/3着馬 (AI予測ランク分布, 軸1着時)")
    # 軸が1着なのに相手が抜けたケースで、誰が2,3着に来たか
    miss_opps = miss_opp_2nd_rank + miss_opp_3rd_rank
    if miss_opps:
        s = pd.Series(miss_opps)
        print(s.value_counts().sort_index().head(10))
    print("-" * 30)
    print("🔍 戦略ポテンシャル比較")
    print(f"  現在の的中数 (1頭軸流し): {hit_perfect}回")
    print(f"  マルチなら的中 (1頭軸マルチ): {potential_hit_multi}回")
    print(f"  Diff: +{potential_hit_multi - hit_perfect}回")
    print("="*50)

if __name__ == "__main__":
    analyze_rank_correlation()
