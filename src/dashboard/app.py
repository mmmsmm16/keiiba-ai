import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import re
import sys

# ページ設定
st.set_page_config(
    page_title="最強AI ダッシュボード",
    layout="wide"
)

st.title("🏇 最強競馬AI: 分析ダッシュボード")

# ディレクトリ設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '../../')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# JRA Venue Mapping
venue_map = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
    '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
}

# タブ作成
tab1, tab2, tab3, tab4, tab5 = st.tabs(["概要 (Overview)", "特徴量重要度 (Feature Importance)", "シミュレーション (ROI)", "分析・振り返り (Analysis)", "予測実行 (Predict)"])

with tab1:
    st.header("実験履歴 (Experiment History)")
    
    history_path = os.path.join(EXPERIMENTS_DIR, 'history.csv')
    if os.path.exists(history_path):
        df_history = pd.read_csv(history_path)
        st.dataframe(df_history)
        
        # メトリクスの推移プロット
        st.subheader("精度メトリクスの推移")
        if not df_history.empty:
            metric = st.selectbox("指標を選択", ["rmse", "ndcg", "map@10"], index=1)
            if metric in df_history.columns:
                st.line_chart(df_history.set_index('timestamp')[metric])
            else:
                st.warning(f"指標 '{metric}' が履歴に見つかりません。")
    else:
        st.warning("実験履歴が見つかりません。まずは学習(train.py)を実行してください。")

with tab2:
    st.header("特徴量重要度 (Feature Importance)")
    
    # 簡易的にTabNetの保存済み画像を表示
    tabnet_imp_path = os.path.join(MODELS_DIR, 'tabnet_importance.png')
    if os.path.exists(tabnet_imp_path):
        st.image(tabnet_imp_path, caption="TabNet 特徴量重要度")
    else:
        st.info("TabNetの重要度プロット画像が見つかりません。")

with tab3:
    st.header("回収率シミュレーション (ROI Simulation)")
    
    sim_path = os.path.join(EXPERIMENTS_DIR, 'latest_simulation.json')
    if os.path.exists(sim_path):
        with open(sim_path, 'r') as f:
            sim_data = json.load(f)
            
        st.markdown(f"**最終更新:** {sim_data.get('timestamp')}")
        
        # 1. 戦略別サマリ
        st.subheader("戦略別サマリ (単純1点買い)")
        strat = sim_data.get('strategies', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("### Max EV (All)")
            max_ev = strat.get('max_ev', {})
            st.metric("回収率", f"{max_ev.get('roi', 0):.2f}%", f"{max_ev.get('accuracy', 0):.2%}")
            
        with col2:
            st.markdown("### Max EV (10-50倍)")
            ev_mid = strat.get('max_ev_odds_10_50', {})
            st.metric("回収率", f"{ev_mid.get('roi', 0):.2f}%", f"{ev_mid.get('accuracy', 0):.2%}")

        with col3:
            st.markdown("### Max EV (50倍+)")
            ev_long = strat.get('max_ev_odds_50plus', {})
            st.metric("回収率", f"{ev_long.get('roi', 0):.2f}%", f"{ev_long.get('accuracy', 0):.2%}")

        with col4:
            st.markdown("### Max Score")
            max_score = strat.get('max_score', {})
            st.metric("回収率", f"{max_score.get('roi', 0):.2f}%", f"{max_score.get('accuracy', 0):.2%}")

        # 2. ROI Curve
        st.subheader("回収率カーブ (期待値閾値ごとの推移)")
        st.markdown("期待値が **閾値** を超えた馬を単勝購入した場合のシミュレーション")
        curve_data = sim_data.get('roi_curve', [])
        
        if curve_data:
            df_curve = pd.DataFrame(curve_data)
            
            # グラフ描画
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.set_xlabel('期待値閾値 (Expected Value Threshold)')
            ax1.set_ylabel('回収率 (%)', color='tab:blue')
            ax1.plot(df_curve['threshold'], df_curve['roi'], color='tab:blue', marker='o', label='回収率', linestyle='-', linewidth=2)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.axhline(100, color='red', linestyle='--', alpha=0.7, label='損益分岐 (100%)') # 100%ライン
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('購入件数 (Bet Count)', color='tab:orange')
            ax2.bar(df_curve['threshold'], df_curve['bet_count'], color='tab:orange', alpha=0.3, width=0.05, label='購入件数')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
            # 凡例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            st.pyplot(fig)
            st.dataframe(df_curve)
        else:
            st.warning("カーブデータが見つかりません。")

        # 3. Complex Betting
        st.subheader("複合馬券シミュレーション")
        st.markdown("Box: スコア上位5頭Box / Nagashi: スコア1位軸・上位2-6位相手")
        
        strategies = sim_data.get('strategies', {})
        complex_keys = [
            'recommended_formation',
            'umaren_box5', 'sanrenpuku_box5', 'sanrentan_box5',
            'umaren_nagashi', 'sanrenpuku_nagashi', 'sanrentan_nagashi'
        ]
        
        complex_data = []
        names = {
            'recommended_formation': '🏆 推奨: 3連単1頭軸流し (条件付)',
            'umaren_box5': '馬連 Box (5頭)', 
            'sanrenpuku_box5': '3連複 Box (5頭)', 
            'sanrentan_box5': '3連単 Box (5頭)',
            'umaren_nagashi': '馬連 流し (軸1頭-相手5頭)',
            'sanrenpuku_nagashi': '3連複 流し (軸1頭-相手5頭)',
            'sanrentan_nagashi': '3連単 流し (軸1頭マルチ-相手5頭)'
        }
        
        for k in complex_keys:
            if k in strategies:
                d = strategies[k]
                complex_data.append({
                    '券種 (Strategy)': names.get(k, k),
                    '回収率 (ROI)': f"{d['roi']:.2f}%",
                    '的中率 (Hit Rate)': f"{d['accuracy']*100:.2f}%",
                    '総投資額': f"{d['bet']:,}円",
                    '払戻総額': f"{d['return']:,}円",
                    '対象レース数': d['races']
                })
        
        if complex_data:
            st.table(pd.DataFrame(complex_data))
        else:
            st.info("複合馬券のシミュレーション結果がありません。")
    else:
        st.warning("シミュレーション結果が見つかりません。先に 'src/model/evaluate.py' を実行してください。")

# --------------------------------------------------------------------------------
# Tab 4: 分析・振り返り (Analysis & Review)
# --------------------------------------------------------------------------------
with tab4:
    st.header("レース振り返り分析 (Race Review)")
    
    # --- Model Selection Scanning ---
    import glob
    pred_files = glob.glob(os.path.join(EXPERIMENTS_DIR, 'predictions_*.parquet'))
    model_options = []
    default_ix = 0
    
    for i, f in enumerate(pred_files):
        fname = os.path.basename(f)
        # Format: predictions_{model}_{version}.parquet -> {model}_{version}
        label = fname.replace('predictions_', '').replace('.parquet', '')
        model_options.append({'label': label, 'path': f})
        if 'catboost_v7' in label: default_ix = i

    st.sidebar.markdown("### 🤖 モデル選択 (Model)")
    if not model_options:
         st.sidebar.error("予測ファイル(predictions_*.parquet)が見つかりません。")
         sel_model_path = None
    else:
         sel_model_item = st.sidebar.selectbox(
             "使用モデル",
             model_options,
             format_func=lambda x: x['label'],
             index=default_ix
         )
         sel_model_path = sel_model_item['path']

    # Load Data (Cached)
    @st.cache_data
    def load_analysis_data(pred_file_path):
        if not pred_file_path or not os.path.exists(pred_file_path):
            return None, None

        payout_path = os.path.join(EXPERIMENTS_DIR, 'payouts_2024_2025.parquet')
        if not os.path.exists(payout_path):
             # Fallback to 2024 if combined not found
             payout_path = os.path.join(EXPERIMENTS_DIR, 'payouts_2024.parquet')
        
        if not os.path.exists(payout_path):
            return None, None
        
        df_pred = pd.read_parquet(pred_file_path)
        df_pay = pd.read_parquet(payout_path)
        
        # Payout Map Construction
        pm = {}
        for _, row in df_pay.iterrows():
            rid = row['race_id']
            pm[rid] = {'sanrentan': {}, 'wide': {}, 'umaren': {}} # Keep minimal for now
            
            # Helper
            def load_p(type_key, col_prefix):
                for i in range(1, 8):
                    k_comb = f'{col_prefix}_{i}a'
                    k_pay = f'{col_prefix}_{i}b'
                    if k_comb in row and row[k_comb] and str(row[k_comb]).strip():
                        try:
                            pay = float(row[k_pay])
                            pm[rid][type_key][str(row[k_comb])] = pay
                        except: pass
            
            load_p('sanrentan', 'haraimodoshi_sanrentan')
            load_p('wide', 'haraimodoshi_wide')
            load_p('wide', 'haraimodoshi_wide')
            load_p('umaren', 'haraimodoshi_umaren')
            
        return df_pred, pm

    if sel_model_path:
        df_an, payout_map = load_analysis_data(sel_model_path)
    else:
        df_an, payout_map = None, None

    if df_an is None:
        st.error(f"データ読み込みエラー: {sel_model_path} または 払戻データが見つかりません。")
    else:
        # Pre-calc Daily Stats for all dates
        if 'date' in df_an.columns:
            df_an['date'] = pd.to_datetime(df_an['date'])
            # FORCE STRING TYPE for race_id
            df_an['race_id'] = df_an['race_id'].astype(str)
            
            # Smart Bet / Strategy Selector
            st.sidebar.markdown("### ⚙️ 分析オプション (Betting Strategy)")
            strategy_mode = st.sidebar.selectbox(
                "運用戦略を選択",
                ["Perfect Portfolio (推奨)", "Smart Bet (旧推奨)", "Normal (全レース)"],
                index=0,
                help="Perfect Portfolio: 最新の最適化戦略（本命フォーメーション＋大穴流し＋見送り）\nSmart Bet: 以前の推奨条件（勝率20%+, EV1.2+, Odds3.0+）"
            )

            # --- 1. Aggregation for Calendar ---
            
            @st.cache_data
            def calc_daily_stats(df_pred, _payout_map, strategy_mode):
                dates_list = sorted(df_pred['date'].unique())
                stats = []
                
                for d in dates_list:
                    day_races = df_pred[df_pred['date'] == d]
                    rids = day_races['race_id'].unique()
                    
                    d_cost = 0
                    d_return = 0
                    d_races = 0
                    
                    for rid in rids:
                        sub = day_races[day_races['race_id'] == rid]
                        if sub.empty: continue
                        
                        top = sub.sort_values('score', ascending=False).iloc[0]
                        odds = top['odds'] if not pd.isna(top['odds']) else 0
                        prob = top['prob']
                        
                        # Use pre-calculated EV if available, else calc
                        if 'expected_value' in top:
                            ev = top['expected_value']
                        else:
                            ev = prob * odds
                        
                        # Betting Logic
                        cost = 0
                        payout = 0
                        bet_type = "-" # For debug
                        
                        if strategy_mode == "Perfect Portfolio (推奨)":
                            # v9 Recommended Stratgy (Aligned with evaluate.py "recommended_formation")
                            # Condition: Prob >= 20%, Odds >= 3.0, EV >= 1.2
                            if (prob >= 0.20 and odds >= 3.0 and ev >= 1.2):
                                bet_type = "Recommended Formation (v9)"
                                # Formation: Axis -> Opps(2-7) -> Opps(2-7) (30 pts)
                                cost = 30 * 100
                                pm = _payout_map.get(rid, {})
                                
                                actual_rank1 = sub[sub['rank'] == 1]
                                actual_rank2 = sub[sub['rank'] == 2]
                                actual_rank3 = sub[sub['rank'] == 3]
                                h1 = int(actual_rank1['horse_number'].iloc[0]) if not actual_rank1.empty else -1
                                h2 = int(actual_rank2['horse_number'].iloc[0]) if not actual_rank2.empty else -1
                                h3 = int(actual_rank3['horse_number'].iloc[0]) if not actual_rank3.empty else -1
                                axis = int(top['horse_number'])
                                
                                opps = sub.sort_values('score', ascending=False).iloc[1:7]['horse_number'].tolist()
                                opp_nums = [int(x) for x in opps if not pd.isna(x)]
                                
                                if h1 == axis and h2 in opp_nums and h3 in opp_nums:
                                    k = f"{h1:02}{h2:02}{h3:02}"
                                    payout = pm.get('sanrentan', {}).get(k, 0)
                            else:
                                pass # Skip

                        else: # Normal
                            # Standard Logic (Odds based)
                            if odds < 3.0:
                                cost = 42 * 100 # 7 Opps
                                opp_count = 7
                                bet_type = "sanrentan"
                            elif odds < 10.0:
                                cost = 30 * 100 # 6 Opps
                                opp_count = 6
                                bet_type = "sanrentan"
                            else:
                                cost = 7 * 100 # Wide 7 Opps
                                opp_count = 7
                                bet_type = "wide"
                            
                            pm = _payout_map.get(rid, {})
                            actual_rank1 = sub[sub['rank'] == 1]
                            actual_rank2 = sub[sub['rank'] == 2]
                            actual_rank3 = sub[sub['rank'] == 3]
                            h1 = int(actual_rank1['horse_number'].iloc[0]) if not actual_rank1.empty else -1
                            h2 = int(actual_rank2['horse_number'].iloc[0]) if not actual_rank2.empty else -1
                            h3 = int(actual_rank3['horse_number'].iloc[0]) if not actual_rank3.empty else -1
                            axis = int(top['horse_number'])
                            opps = sub.sort_values('score', ascending=False).iloc[1:opp_count+1]['horse_number'].tolist()
                            opp_nums = [int(x) for x in opps if not pd.isna(x)]

                            if bet_type == "sanrentan":
                                if h1 == axis and h2 in opp_nums and h3 in opp_nums:
                                    k = f"{h1:02}{h2:02}{h3:02}"
                                    payout = pm.get('sanrentan', {}).get(k, 0)
                            elif bet_type == "wide":
                                winners = [h1, h2, h3]
                                for w in winners:
                                    if w == axis: continue
                                    if w in opp_nums:
                                        pair = sorted([axis, w])
                                        p = pm.get('wide', {}).get(f"{pair[0]:02}{pair[1]:02}", 0)
                                        payout += p
                        
                        d_cost += cost
                        d_return += payout
                        if cost > 0: d_races += 1
                        
                        # Store in day_details list (needs to be defined outside loop?)
                        # WARNING: calc_daily_stats CURRENTLY aggregate stats for whole day.
                        # It doesn't modify the global day_details list.
                        # Wait, day_details used in grid is defined where?
                        # It is defined in the MAIN LOOP (lines 530+).
                        # lines 234+ is just the 'calc_daily_stats' FUNCTION (used for ROI chart).
                        # The GRID uses a separate loop.

                    
                    # Append daily stats
                    d_profit = d_return - d_cost
                    d_roi = (d_return / d_cost * 100) if d_cost > 0 else 0
                    stats.append({
                        'date_obj': d,
                        '日付': d.strftime('%Y-%m-%d'),
                        'レース数': d_races,
                        'total_cost': d_cost,
                        'total_return': d_return,
                        '日次収支': d_profit,
                        'ROI(%)': d_roi
                    })
                
                return pd.DataFrame(stats)

            daily_df = calc_daily_stats(df_an, payout_map, strategy_mode)

            daily_df['Year'] = daily_df['date_obj'].dt.year
            daily_df['Month'] = daily_df['date_obj'].dt.month
            daily_df['年月'] = daily_df['date_obj'].dt.strftime('%Y-%m')
            
            # =========================================================
            # 2. Selectors & Annual/Monthly Stats (Left Column)
            # =========================================================
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.subheader("📅 期間選択")
                
                # --- Year Selection ---
                years = sorted(daily_df['Year'].unique(), reverse=True)
                sel_year = st.selectbox("対象年 (Year)", years)
                
                # Annual Stats
                year_data = daily_df[daily_df['Year'] == sel_year]
                y_cost = year_data['total_cost'].sum()
                y_return = year_data['total_return'].sum()
                y_profit = y_return - y_cost
                y_roi = (y_return / y_cost * 100) if y_cost > 0 else 0
                y_races = year_data['レース数'].sum()

                st.markdown(f"**{sel_year}年 年間成績**")
                yc1, yc2 = st.columns(2)
                yc1.metric("年間 ROI", f"{y_roi:.1f}%", f"{y_profit:+,}")
                yc2.metric("総投資", f"{y_cost:,}")
                
                # DEBUG INFO
                with st.expander("詳細統計 (Debug)", expanded=True):
                    st.write(f"適用フィルタ: {strategy_mode}")
                    st.write(f"対象レース数: {y_races} R")
                    st.write(f"総投資額: {y_cost} 円")
                    st.write(f"総払戻額: {y_return} 円")
                    if y_cost > 0:
                        st.write(f"平均ROI: {y_roi:.2f}%")
                    else:
                        st.warning("投資対象レースがありません (条件に合うレースなし)")

                st.divider()

                # --- Month Selection ---
                months_in_year = sorted(year_data['Month'].unique(), reverse=True)
                sel_month_int = st.selectbox("対象月 (Month)", months_in_year, format_func=lambda x: f"{x}月")
                
                # Filter Monthly Data
                month_data = year_data[year_data['Month'] == sel_month_int].copy()
                
                # Monthly Stats
                m_cost = month_data['total_cost'].sum()
                m_return = month_data['total_return'].sum()
                m_profit = m_return - m_cost
                m_roi = (m_return / m_cost * 100) if m_cost > 0 else 0
                
                st.markdown(f"**{sel_month_int}月 月次成績**")
                mc1, mc2 = st.columns(2)
                mc1.metric("月間 ROI", f"{m_roi:.1f}%", f"{m_profit:+,}")
                mc2.metric("総投資", f"{m_cost:,}")
                
                # Prepare Calendar List
                st.write("日別成績一覧")
                display_cols = ['レース数', 'ROI(%)', '日次収支']
                month_display = month_data.set_index('日付')[display_cols]
                
                event_state = st.dataframe(
                    month_display.style.background_gradient(subset=['ROI(%)'], cmap='RdYlGn', vmin=50, vmax=150),
                    on_select="rerun",
                    selection_mode="single-row",
                    use_container_width=True,
                    height=300
                )
                
                selected_rows = event_state.selection.rows
                if selected_rows:
                    sel_date_str = month_display.index[selected_rows[0]]
                    sel_date = pd.to_datetime(sel_date_str)
                else:
                    st.info("👈 日付を選択してください")
                    sel_date = None

            # =========================================================
            # 3. Race Details (Right Column)
            # =========================================================
            race_ids = []
            if sel_date:
                day_races = df_an[df_an['date'] == sel_date]
                race_ids = sorted(day_races['race_id'].unique())
                
            if not race_ids:
                if sel_date:
                    with col_right:
                        st.warning("この日のレースデータはありません。")
            else:
                # =========================================================
                # 3. Calc Details for Grid
                # =========================================================
                total_cost = 0
                total_return = 0
                total_races = len(race_ids)
                hit_races = 0
                day_details = []
                
                for rid in race_ids:
                    r_df = day_races[day_races['race_id'] == rid].sort_values('score', ascending=False)
                    if r_df.empty: continue
                    
                    top_horse = r_df.iloc[0]
                    top_odds = top_horse['odds'] if not pd.isna(top_horse['odds']) else 0
                    prob = top_horse['prob']
                    
                    # Use pre-calculated EV if available
                    if 'expected_value' in top_horse:
                        ev = top_horse['expected_value']
                    else:
                        ev = prob * top_odds
                    
                    # Logic Refinement: Use 'strategy_mode'
                    cost = 0
                    payout = 0
                    bet_type = "-"

                    # 1. Perfect Portfolio
                    if strategy_mode == "Perfect Portfolio (推奨)":
                        if top_odds < 3.0 and ev >= 1.0:
                            # Solid: Formation (24pts)
                            cost = 2400
                            bet_type = "Solid Formation"
                            
                            pm = payout_map.get(rid, {})
                            actual_rank1 = r_df[r_df['rank'] == 1]
                            actual_rank2 = r_df[r_df['rank'] == 2]
                            actual_rank3 = r_df[r_df['rank'] == 3]
                            h1 = int(actual_rank1['horse_number'].iloc[0]) if not actual_rank1.empty else -1
                            h2 = int(actual_rank2['horse_number'].iloc[0]) if not actual_rank2.empty else -1
                            h3 = int(actual_rank3['horse_number'].iloc[0]) if not actual_rank3.empty else -1
                            axis = int(top_horse['horse_number'])
                            
                            opps = r_df.iloc[1:] # 2nd place and below
                            if len(opps) >= 9:
                                rank_map = {i+2: int(row['horse_number']) for i, row in enumerate(opps.iloc[:9].to_dict('records'))}
                                if h1 == axis:
                                    valid_h2 = [rank_map.get(r) for r in [2,3,4,5] if rank_map.get(r) is not None]
                                    valid_h3 = [rank_map.get(r) for r in [2,3,4,5,6,7,8] if rank_map.get(r) is not None]
                                    if h2 in valid_h2 and h3 in valid_h3:
                                        k = f"{h1:02}{h2:02}{h3:02}"
                                        payout = pm.get('sanrentan', {}).get(k, 0)

                        elif top_odds >= 10.0 and ev >= 1.3:
                            # Longshot: Nagashi (30pts)
                            cost = 3000
                            bet_type = "Longshot Nagashi"
                            # Standard Nagashi Hit Check
                            pm = payout_map.get(rid, {})
                            actual_rank1 = r_df[r_df['rank'] == 1]
                            actual_rank2 = r_df[r_df['rank'] == 2]
                            actual_rank3 = r_df[r_df['rank'] == 3]
                            h1 = int(actual_rank1['horse_number'].iloc[0]) if not actual_rank1.empty else -1
                            h2 = int(actual_rank2['horse_number'].iloc[0]) if not actual_rank2.empty else -1
                            h3 = int(actual_rank3['horse_number'].iloc[0]) if not actual_rank3.empty else -1
                            axis = int(top_horse['horse_number'])
                            opps = r_df.iloc[1:7]['horse_number'].tolist() # Top 6 opps
                            opp_nums = [int(x) for x in opps if not pd.isna(x)]
                            
                            if h1 == axis and h2 in opp_nums and h3 in opp_nums:
                                k = f"{h1:02}{h2:02}{h3:02}"
                                payout = pm.get('sanrentan', {}).get(k, 0)
                        else:
                            bet_type = "SKIP"
                            
                    # 2. Smart Bet (Old)
                    elif strategy_mode == "Smart Bet (旧推奨)":
                        if prob >= 0.20 and top_odds >= 3.0 and ev >= 1.2:
                            cost = 3000
                            bet_type = "Smart Bet Nagashi"
                            # Hit Check
                            pm = payout_map.get(rid, {})
                            actual_rank1 = r_df[r_df['rank'] == 1]
                            actual_rank2 = r_df[r_df['rank'] == 2]
                            actual_rank3 = r_df[r_df['rank'] == 3]
                            h1 = int(actual_rank1['horse_number'].iloc[0]) if not actual_rank1.empty else -1
                            h2 = int(actual_rank2['horse_number'].iloc[0]) if not actual_rank2.empty else -1
                            h3 = int(actual_rank3['horse_number'].iloc[0]) if not actual_rank3.empty else -1
                            axis = int(top_horse['horse_number'])
                            opps = r_df.iloc[1:7]['horse_number'].tolist()
                            opp_nums = [int(x) for x in opps if not pd.isna(x)]
                            
                            if h1 == axis and h2 in opp_nums and h3 in opp_nums:
                                k = f"{h1:02}{h2:02}{h3:02}"
                                payout = pm.get('sanrentan', {}).get(k, 0)
                        else:
                            bet_type = "SKIP"
                    
                    # 3. Normal (Fallback)
                    else:
                        # Reusing calc_daily_stats logic structure?
                        # Simplification for Grid: Just show "Normal" stats if implemented
                        # For now, if "Normal" logic is needed here, paste it.
                        # Assuming user rarely checks "Normal" detailed logic in grid, but for consistency:
                        if top_odds < 3.0:
                             cost = 4200
                             bet_type = "Normal 3T"
                             opp_count = 7
                             b_cat = "sanrentan"
                        elif top_odds < 10.0:
                             cost = 3000
                             bet_type = "Normal 3T"
                             opp_count = 6
                             b_cat = "sanrentan"
                        else:
                             cost = 700
                             bet_type = "Normal Wide"
                             opp_count = 7
                             b_cat = "wide"
                        
                        pm = payout_map.get(rid, {})
                        actual_rank1 = r_df[r_df['rank'] == 1]
                        actual_rank2 = r_df[r_df['rank'] == 2]
                        actual_rank3 = r_df[r_df['rank'] == 3]
                        h1 = int(actual_rank1['horse_number'].iloc[0]) if not actual_rank1.empty else -1
                        h2 = int(actual_rank2['horse_number'].iloc[0]) if not actual_rank2.empty else -1
                        h3 = int(actual_rank3['horse_number'].iloc[0]) if not actual_rank3.empty else -1
                        axis = int(top_horse['horse_number'])
                        opps = r_df.iloc[1:opp_count+1]['horse_number'].tolist()
                        opp_nums = [int(x) for x in opps if not pd.isna(x)]

                        if b_cat == "sanrentan":
                            if h1 == axis and h2 in opp_nums and h3 in opp_nums:
                                k = f"{h1:02}{h2:02}{h3:02}"
                                payout = pm.get('sanrentan', {}).get(k, 0)
                        elif b_cat == "wide":
                             winners = [h1, h2, h3]
                             for w in winners:
                                 if w == axis: continue
                                 if w in opp_nums:
                                     pair = sorted([axis, w])
                                     p = pm.get('wide', {}).get(f"{pair[0]:02}{pair[1]:02}", 0)
                                     payout += p



                    hit = (payout > 0)
                    total_cost += cost
                    total_return += payout
                    if hit: hit_races += 1
                    
                    day_details.append({
                        'race_id': rid,
                        'race_no': r_df.iloc[0].get('race_number', '??'),
                        'bet_type': bet_type,
                        'cost': cost,
                        'return': payout,
                        'profit': payout - cost,
                        'status': '🎯' if hit else ('SKIP' if bet_type == 'SKIP' else 'lose')
                    })

                # =========================================================
                # 4. Render Race Grid (Tabs + Grid)
                # =========================================================
                with col_right:
                    st.subheader(f"📅 {sel_date.strftime('%Y-%m-%d')} - レース選択")
                    
                    # Group Races by Venue
                    venue_groups = {}
                    for rid in race_ids:
                        p_code = str(rid)[4:6]
                        if p_code not in venue_groups: venue_groups[p_code] = []
                        venue_groups[p_code].append(rid)
                    
                    sorted_venues = sorted(venue_groups.keys())
                    if not sorted_venues:
                        st.info("レースデータがありません")
                    else:
                        # Use Tabs for Venues
                        venue_map = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
                        tabs = st.tabs([venue_map.get(v, v) for v in sorted_venues])
                        
                        if 'selected_race_id' not in st.session_state:
                             st.session_state['selected_race_id'] = None
                        
                        for idx, p_code in enumerate(sorted_venues):
                            with tabs[idx]:
                                rids = venue_groups[p_code]
                                # Create a responsive grid (e.g., 3 cols)
                                grid_cols = st.columns(3)
                                
                                for i, rid in enumerate(sorted(rids)):
                                    d = next((item for item in day_details if item['race_id'] == rid), None)
                                    profit = d['profit'] if d else 0
                                    cost_val = d['cost'] if d else 0
                                    ret_val = d['return'] if d else 0
                                    roi_val = (ret_val / cost_val * 100) if cost_val > 0 else 0
                                    status = d['status'] if d else '-'
                                    bet_type_disp = d['bet_type'] if d else '-'
                                    
                                    race_no_int = int(str(rid)[10:12])
                                    r_row = day_races[day_races['race_id'] == rid].iloc[0]
                                    title = r_row.get('race_name', '') # Use race_name (original) or race_name_j
                                    if pd.isna(title): title = ""
                                    display_title = title[:10] + ".." if len(title) > 10 else title
                                    
                                    # Column assignment
                                    with grid_cols[i % 3]:
                                        label_header = f"{race_no_int}R {display_title}"
                                        
                                        if status == 'SKIP':
                                            with st.container(border=True):
                                                st.caption(f"{race_no_int}R (SKIP)")
                                                if st.button("詳細", key=f"btn_{rid}", use_container_width=True):
                                                    st.session_state['selected_race_id'] = rid
                                            continue

                                        if profit > 0:
                                            with st.success(label_header):
                                                st.caption(f"¥{profit:+,}")
                                                if st.button("詳細", key=f"btn_{rid}", use_container_width=True):
                                                    st.session_state['selected_race_id'] = rid
                                        elif profit < 0:
                                            with st.error(label_header):
                                                st.caption(f"¥{profit:+,}")
                                                if st.button("詳細", key=f"btn_{rid}", use_container_width=True):
                                                    st.session_state['selected_race_id'] = rid
                                        else: # draw or 0
                                            with st.container(border=True):
                                                st.markdown(f"**{label_header}**")
                                                st.caption(f"¥{profit:+,}")
                                                if st.button("詳細", key=f"btn_{rid}", use_container_width=True):
                                                    st.session_state['selected_race_id'] = rid
                
                # Daily Summary Metrics
                st.divider()
                st.markdown("### 📊 日次収支サマリー")
                daily_profit = total_return - total_cost
                daily_roi = (total_return / total_cost * 100) if total_cost > 0 else 0
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("総投資", f"{total_cost:,}円", f"{total_races}R")
                m2.metric("総払戻", f"{total_return:,}円", f"的中 {hit_races}R")
                m3.metric("日次収支", f"{daily_profit:+,}円", delta_color="normal")
                m4.metric("回収率 (ROI)", f"{daily_roi:.1f}%")
                
                sel_race_id = st.session_state.get('selected_race_id')
                
                # Check if selected race belongs to current day
                if sel_race_id not in race_ids:
                    sel_race_id = None

                if sel_race_id:
                    race_df = day_races[day_races['race_id'] == sel_race_id].copy()
                    race_df = race_df.sort_values('score', ascending=False)
                    
                    detail = next((d for d in day_details if d['race_id'] == sel_race_id), None)
                    
                    st.subheader(f"📊 レース詳細: {sel_race_id}")
                    
                    # --- Logic Debug Info ---
                    if detail:
                        top_horse = race_df.iloc[0]
                        th_no = top_horse['horse_number']
                        th_odds = top_horse['odds'] if not pd.isna(top_horse['odds']) else 0
                        th_prob = top_horse['prob']
                        # Use loaded EV if available
                        th_ev = top_horse['expected_value'] if 'expected_value' in top_horse else (th_prob * th_odds)

                        with st.expander("🧐 ベッティング判定詳細 (Debug)", expanded=False):
                            st.write(f"**戦略**: {strategy_mode}")
                            st.write(f"**本命 (Axis)**: 馬番 {th_no} (AI 1位) | Odds: {th_odds} | EV: {th_ev:.2f}")
                            
                            is_solid = (th_odds < 3.0 and th_ev >= 1.0)
                            is_long = (th_odds >= 10.0 and th_ev >= 1.3)
                            
                            st.write(f"判定: Solid={is_solid}, Longshot={is_long}")
                            
                            if detail['status'] != 'SKIP':
                                # Re-construct formation for display
                                opps_ordered = race_df.iloc[1:] # 2nd place and below
                                if len(opps_ordered) >= 9:
                                    # Logic mimics calc_daily_stats
                                    rank_map = {i+2: int(row['horse_number']) for i, row in enumerate(opps_ordered.iloc[:9].to_dict('records'))}
                                    
                                    if detail['bet_type'] == "Solid Formation": # Infer from logic
                                        r2_list = [rank_map.get(r) for r in [2,3,4,5] if rank_map.get(r) is not None]
                                        r3_list = [rank_map.get(r) for r in [2,3,4,5,6,7,8] if rank_map.get(r) is not None]
                                        st.write(f"**購入フォーメーション**:")
                                        st.write(f"1着: {th_no}")
                                        st.write(f"2着候補 (AI 2-5位): {r2_list}")
                                        st.write(f"3着候補 (AI 2-8位): {r3_list}")
                                    elif detail['bet_type'] == "Longshot Nagashi":
                                        opp_nums = [int(x) for x in race_df.iloc[1:7]['horse_number'].tolist() if not pd.isna(x)]
                                        st.write(f"**購入流し**:")
                                        st.write(f"軸: {th_no}")
                                        st.write(f"相手 (AI 2-7位): {opp_nums}")
                            else:
                                if not is_solid and not is_long:
                                  st.write("理由: 条件（Odds/EV）を満たさず")

                    if detail:
                         if detail['status'] == 'SKIP':
                             st.info(f"ℹ️ このレースは{strategy_mode}の条件を満たさないためスキップしました。")
                         else:
                             res_color = "red" if detail['profit'] < 0 else "green"
                             st.markdown(f"**結果**: {detail['status']} | 投資: {detail['cost']}円 | 払戻: {detail['return']}円 | 収支: <span style='color:{res_color}'>{detail['profit']:+,}円</span> | 券種: {detail['bet_type']}", unsafe_allow_html=True)

                    # Add helper cols
                    race_df['AI順位'] = range(1, len(race_df) + 1)
                    
                    def get_mark(rank):
                        if rank == 1: return "◎"
                        if rank == 2: return "〇"
                        if rank == 3: return "▲"
                        if 4 <= rank <= 6: return "△"
                        return ""
                    
                    race_df['印'] = race_df['AI順位'].apply(get_mark)
                    
                    def get_rank_icon(r):
                        try:
                            r = int(r)
                            if r == 1: return "🥇 1着"
                            if r == 2: return "🥈 2着"
                            if r == 3: return "🥉 3着"
                            return f"{r}着"
                        except: return "-"

                    race_df['着順'] = race_df['rank'].apply(get_rank_icon)
                    
                    # Calculate EV (Expected Value) only if not exists
                    if 'expected_value' not in race_df.columns:
                        race_df['expected_value'] = race_df['prob'] * race_df['odds']
                    
                    # Ensure EV correct (sometimes loaded from different version)
                    # Force recalc if missing or just trust parquet? Trust parquet v7.

                    # Convert prob to percentage (0-100) for easier handling
                    race_df['prob'] = race_df['prob'] * 100
                    
                    # Combine No and Name for compactness
                    race_df['馬'] = race_df['horse_number'].astype(str) + ". " + race_df['horse_name']
                    
                    # Display Columns (Added expected_value)
                    show_cols = ['AI順位', '印', '着順', '馬', 'expected_value', 'prob', 'odds']
                    # Ensure cols exist
                    avail_cols = [c for c in show_cols if c in race_df.columns]
                    
                    # Formatting DataFrame
                    st.dataframe(
                        race_df[avail_cols],
                        column_config={
                            "AI順位": st.column_config.NumberColumn("AI Rank", format="%d", width="small"),
                            "印": st.column_config.TextColumn("Mark", width="small"),
                            "着順": st.column_config.TextColumn("Result", width="small"),
                            "馬": st.column_config.TextColumn("No. Horse Name", width="medium"),
                            "expected_value": st.column_config.NumberColumn(
                                "期待値 (EV)",
                                help="確率 x オッズ (1.0以上が目安)",
                                format="%.2f",
                                width="small"
                            ),
                            "prob": st.column_config.ProgressColumn(
                                "Confidence",
                                help="AI Model Probability",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "odds": st.column_config.NumberColumn("Odds", format="%.1f", width="small"),
                        },
                        use_container_width=True,
                        height=500,
                        hide_index=True
                    )

# --------------------------------------------------------------------------------
# Tab 5: 予測実行 (Real-time Prediction)
# --------------------------------------------------------------------------------
with tab5:
# ... (Previous Tab 4 content shifted)

    st.header("予測実行 (Real-time Prediction)")
    st.markdown("指定した日付・レースの予測をリアルタイムに実行します。データはPC-KEIBA DBから取得します。")

    # 必要なモジュールのインポート (ここでインポートしてスコープを限定)
    sys.path.append(os.path.join(BASE_DIR, '../'))
    from inference.loader import InferenceDataLoader
    from inference.preprocessor import InferencePreprocessor, calculate_race_features
    from model.ensemble import EnsembleModel
    from model.lgbm import KeibaLGBM
    from model.catboost_model import KeibaCatBoost
    from model.tabnet_model import KeibaTabNet
    from scipy.special import softmax
    import lightgbm as lgb
    import pickle

    @st.cache_resource
    def load_betting_model():
        try:
            model_path = os.path.join(MODELS_DIR, 'betting_model.pkl')
            if not os.path.exists(model_path):
                return None
            with open(model_path, 'rb') as f:
                bst = pickle.load(f)
            return bst
        except Exception as e:
            st.warning(f"Betting Model Load Failed: {e}")
            return None

    # モデルバージョンの取得
    def get_model_versions(model_type):
        if not os.path.exists(MODELS_DIR):
            return ['v1']

        files = os.listdir(MODELS_DIR)
        versions = set()

        # Check for legacy/default files
        if model_type == 'ensemble':
            if 'ensemble_model.pkl' in files:
                versions.add('v1')
        else:
            # lgbm.pkl, catboost.pkl, tabnet.pkl/zip
            base_name = f"{model_type}.pkl"
            if model_type == 'tabnet' and 'tabnet.zip' in files:
                versions.add('v1')
            elif base_name in files:
                versions.add('v1')
        
        # Check for versioned files
        prefix = f"{model_type}_"
        for f in files:
            if f.startswith(prefix):
                # Extract tag
                tag = ""
                if f.endswith('.pkl'):
                    tag = f[len(prefix):-4]
                elif f.endswith('.zip') and model_type == 'tabnet':
                    tag = f[len(prefix):-4]
                
                if tag:
                    if model_type == 'ensemble' and tag == 'model':
                        continue # Already handled as v1
                    versions.add(tag)
        
        # Sort versions
        return sorted(list(versions))

    # キャッシュされたモデルとプリプロセッサのロード関数
    @st.cache_resource
    def load_model_resources_v3(model_type, version):
        model = None
        path = ""
        
        if model_type == 'ensemble':
            model = EnsembleModel()
            # Try specific version first, then default
            path = os.path.join(MODELS_DIR, f'ensemble_{version}.pkl')
            if not os.path.exists(path):
                 path = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
        elif model_type == 'lgbm':
            model = KeibaLGBM()
            path = os.path.join(MODELS_DIR, f'lgbm_{version}.pkl')
            if not os.path.exists(path):
                 path = os.path.join(MODELS_DIR, 'lgbm.pkl')
        elif model_type == 'catboost':
            model = KeibaCatBoost()
            path = os.path.join(MODELS_DIR, f'catboost_{version}.pkl')
            if not os.path.exists(path):
                 path = os.path.join(MODELS_DIR, 'catboost.pkl')
        elif model_type == 'tabnet':
            model = KeibaTabNet()
            # TabNet special case: zip vs pkl
            path_zip = os.path.join(MODELS_DIR, f'tabnet_{version}.zip')
            if os.path.exists(path_zip):
                path = path_zip.replace('.zip', '.pkl')
            else:
                path = os.path.join(MODELS_DIR, 'tabnet.pkl')

        if not os.path.exists(path) and not (model_type == 'tabnet' and os.path.exists(path.replace('.pkl', '.zip'))):
            return None, f"Model file not found: {path} (Type: {model_type}, Ver: {version})"

        try:
            if model_type in ['ensemble', 'tabnet']:
                 model.load_model(path, device_name='cpu')
            else:
                 model.load_model(path)
            return model, f"Loaded: {os.path.basename(path)}"
        except Exception as e:
            return None, f"Error loading model: {e}"

    @st.cache_resource
    def load_preprocessor_resources():
        preprocessor = InferencePreprocessor()
        return preprocessor

    # Data Loader for historical data (Heavy)
    @st.cache_resource
    def get_historical_data():
        data_path = os.path.join(PROJECT_ROOT, 'data/processed/preprocessed_data.parquet')
        if os.path.exists(data_path):
            st.info("Loading historical data to memory (One-time operation)...")
            return pd.read_parquet(data_path)
        return None

    # UI Inputs
    st.subheader("設定 (Settings)")
    
    # Date Selection
    col_date, col_venue, col_race = st.columns(3)
    with col_date:
        selected_date = st.date_input(
            "開催日",
            value=pd.Timestamp.now(),
            min_value=pd.Timestamp('2020-01-01'),
            max_value=pd.Timestamp.now() + pd.Timedelta(days=30)
        )
        target_date = selected_date.strftime('%Y%m%d')

    # Load Race List
    loader = InferenceDataLoader()
    race_list_df = loader.load_race_list(target_date)
    
    venue_map = {
        '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
        '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
    }

    if race_list_df.empty:
        st.warning(f"指定された日付 ({target_date}) のレースデータが見つかりません。")
        venue_code = None
        race_no = None
        selected_race_id = None
    else:
        # Venue Selection
        with col_venue:
            available_venues = race_list_df['venue'].unique()
            # Sort by code
            available_venues.sort()
            
            venue_options = {v: venue_map.get(v, v) for v in available_venues}
            venue_code = st.selectbox("開催場所", options=list(venue_options.keys()), format_func=lambda x: f"{x}: {venue_options[x]}")

        # Race Selection
        with col_race:
            if venue_code:
                venue_races = race_list_df[race_list_df['venue'] == venue_code]
                if not venue_races.empty:
                    # Create label "11R: メインレース名 (15:35)"
                    def race_label(row):
                        time_str = f" ({row['start_time']})" if row['start_time'] else ""
                        title_str = f"{row['title']}" if row['title'] else "レース名なし"
                        return f"{row['race_number']}R: {title_str}{time_str}"
                    
                    race_options = {row['race_number']: race_label(row) for _, row in venue_races.iterrows()}
                    race_no = st.selectbox("レース選択", options=sorted(list(race_options.keys())), format_func=lambda x: race_options[x])
                    selected_race_id = venue_races[venue_races['race_number'] == race_no]['race_id'].iloc[0]
                else:
                    st.warning("この会場のレースがありません")
                    race_no = None
                    selected_race_id = None
            else:
                 race_no = None
                 selected_race_id = None

    # Model Selection
    col_mod1, col_mod2 = st.columns(2)
    with col_mod1:
        model_type = st.selectbox("使用モデル", ['ensemble', 'lgbm', 'catboost', 'tabnet'], index=0)
    
    # Dynamic version loading
    avail_versions = get_model_versions(model_type)
    if not avail_versions:
        avail_versions = ['v1'] # Fallback
        
    with col_mod2:
        # Default to last one
        default_idx = len(avail_versions) - 1
        model_version = st.selectbox("モデルバージョン", avail_versions, index=default_idx)

    # st.info(f"Target: {target_date} / {venue_map.get(venue_code)} / {race_no}R | Model: {model_type} ({model_version})")

    if st.button("予測実行 (Predict)", disabled=(not selected_race_id)):
        with st.spinner('モデルとデータを準備中...'):
            model, msg = load_model_resources_v3(model_type, model_version)
            if model:
                st.success(msg)
            else:
                st.error(msg)
            
            hist_df = get_historical_data() # Cached load
            
            if model is None:
                pass # Already showed error
            elif hist_df is None:
                st.error("過去データが見つかりません (data/processed/preprocessed_data.parquet)")
            else:
                # 1. データロード (ID指定で高速化可能だが、現状のLoader仕様に合わせて日付でロードしてフィルタ)
                try:
                    # RaceListからIDがわかっているので、ID指定ロードしても良いが、Loaderの変更が必要になるため
                    # ここでは既存の load(target_date) を使ってフィルタする (Loader内部で全件ロードしているなら効率悪いが許容)
                    # ★改善: loader.load に race_ids 引数を渡せばSQLレベルで絞り込める
                    new_df = loader.load(target_date=target_date, race_ids=[selected_race_id])
                except Exception as e:
                    new_df = pd.DataFrame()
                    st.error(f"データロードエラー: {e}")

                if new_df.empty:
                    st.warning(f"データ詳細が見つかりませんでした (ID: {selected_race_id})。PC-KEIBAでデータ登録済みか確認してください。")
                else:
                    # st.success(f"データロード成功: {len(new_df)} 頭")
                    
                    # 2. 前処理
                    preprocessor = InferencePreprocessor()
                    
                    try:
                        X, ids = preprocessor.preprocess(new_df, history_df=hist_df)
                        
                        if X.empty:
                            st.error("前処理後の特徴量が生成できませんでした。")
                        else:
                            # レース情報カードの表示
                            race_info = new_df.iloc[0]
                            
                            st.markdown("---")
                            st.subheader(f"📋 {race_info.get('race_number')}R: {race_info.get('title', 'N/A')}")
                            
                            # Loaderで既にデコード済みのため、そのまま表示する
                            info_col1, info_col2, info_col3 = st.columns(3)
                            with info_col1:
                                dist_str = f"{race_info.get('distance', 'N/A')}m" if race_info.get('distance') else 'N/A'
                                st.metric("条件", f"{race_info.get('surface', 'N/A')} {dist_str}")
                            with info_col2:
                                st.metric("馬場 / 天候", f"{race_info.get('state', 'N/A')} / {race_info.get('weather', 'N/A')}")
                            with info_col3:
                                start_time = race_info.get('start_time')
                                start_time_str = start_time if start_time else '--:--'
                                st.metric("発走時刻 / 頭数", f"{start_time_str} / {len(new_df)}頭")
                            
                            st.markdown("---")
                            
                            # 3. 予測
                            # 特徴量のフィルタリング
                            if hasattr(model, 'model') and hasattr(model.model, 'feature_name'): # LightGBM
                                required_features = model.model.feature_name()
                                missing = set(required_features) - set(X.columns)
                                if not missing:
                                    X = X[required_features]
                            elif hasattr(model, 'model') and hasattr(model.model, 'feature_names_'): # CatBoost
                                required_features = model.model.feature_names_
                                missing = set(required_features) - set(X.columns)
                                if not missing:
                                    X = X[required_features]
                            
                            preds = model.predict(X)
                            
                            # 結果整形
                            results = ids.copy()
                            results['score'] = preds
                            results['prob'] = results.groupby('race_id')['score'].transform(lambda x: softmax(x))
                            
                            # 期待値計算
                            results['expected_value'] = results['prob'] * results['odds']
                            results['recommended'] = results['expected_value'] > 1.0
                            
                            # 表示用カラム (整数ランク)
                            results['pred_rank'] = results.groupby('race_id')['score'].rank(ascending=False, method='min').astype(int)
                            
                            # --- Betting Model Prediction ---
                            bet_model = load_betting_model()
                            bet_confidence = None
                            if bet_model:
                                try:
                                    # Feature Calculation
                                    race_feats = calculate_race_features(results)
                                    # Predict
                                    # LightGBM predict returns array
                                    features = ['entropy', 'odds_std', 'max_prob', 'confidence_gap', 'n_horses']
                                    bet_pred = bet_model.predict(race_feats[features])
                                    bet_confidence = bet_pred[0]
                                except Exception as e:
                                    # logger.error(f"Betting pred error: {e}") # Logger not avail in this scope
                                    st.warning(f"勝負度判定エラー: {e}")

                            if bet_confidence is not None:
                                st.markdown("### 🤖 買い時判定 AI")
                                b_col1, b_col2, b_col3 = st.columns(3)
                                b_col1.metric("勝負度 (Confidence)", f"{bet_confidence:.4f}")
                                
                                # Threshold (Using the one from optimizing logic, e.g. 0.5 for now or implicit)
                                # The model targets "is_profitable". So > 0.5 means > 50% chance of being profitable?
                                # Phase 10 Model AUC 0.55 is weak, but let's show visual feedback.
                                if bet_confidence > 0.52: # Slightly conservative
                                    b_col2.success("✅ 期待値高 (Bet)")
                                else:
                                    b_col2.warning("⚠️ 見送り推奨 (Stay)")
                                    
                                # Context Features
                                if not race_feats.empty:
                                    ent = race_feats.iloc[0]['entropy']
                                    b_col3.metric("情報の錯綜度 (Entropy)", f"{ent:.3f}")
                            
                            st.markdown("---")
                            
                            # 詳細情報を表示
                            display_cols = ['pred_rank', 'horse_number', 'horse_name', 'score', 'prob', 'odds', 'popularity', 'expected_value']
                            display_df = results.sort_values('pred_rank')[display_cols]
                            
                            # カラム名日本語化
                            rename_map = {
                                'pred_rank': '予想順位',
                                'horse_number': '馬番',
                                'horse_name': '馬名',
                                'score': '予測スコア',
                                'prob': 'AI勝率',
                                'odds': '単勝オッズ',
                                'popularity': '人気',
                                'expected_value': '期待値'
                            }
                            display_df_renamed = display_df.rename(columns=rename_map)
                            
                            # 色付けとハイライト
                            st.subheader(f"🎯 予測結果")
                            
                            # おすすめ馬の数を表示
                            rec_count = results['recommended'].sum()
                            if rec_count > 0:
                                st.info(f"💡 期待値が1.0を超える「おすすめ馬」が {rec_count} 頭います（黄色・赤色でハイライト）")
                            
                            def highlight_rows(s):
                                rank = s['予想順位']
                                exp_val = s.get('期待値', 0)
                                is_rec = exp_val > 1.0
                                
                                # より淡い色で見やすく
                                if is_rec and rank == 1:
                                    return ['background-color: #ffb3ba; color: black'] * len(s)  # ライトピンク
                                elif is_rec:
                                    return ['background-color: #ffffba; color: black'] * len(s)  # ライトイエロー
                                elif rank == 1:
                                    return ['background-color: #ffe6e6; color: black'] * len(s)  # 極薄ピンク
                                elif rank <= 3:
                                    return ['background-color: #f5f5f5; color: black'] * len(s)  # 薄グレー
                                else:
                                    return [''] * len(s)

                            # style.apply は axis=1 で行ごとに適用 (formatで小数点調整)
                            st.dataframe(
                                display_df_renamed.style.apply(highlight_rows, axis=1).format({
                                    '予想順位': '{:d}',
                                    '予測スコア': '{:.4f}', 
                                    'AI勝率': '{:.2%}',
                                    '期待値': '{:.2f}'
                                })
                            )

                            # --- 具体的な買い目提案 (Betting Suggestions) ---
                            st.markdown("---")
                            st.subheader("🎫 推奨買い目 (Betting Suggestions)")
                            
                            # --- 具体的な買い目提案 (Betting Suggestions) ---
                            st.markdown("---")
                            st.subheader("🎫 推奨買い目 (Betting Suggestions)")
                            
                            # Session State for Odds
                            if 'complex_odds_cache' not in st.session_state:
                                st.session_state.complex_odds_cache = {}
                            if 'last_race_id_odds' not in st.session_state:
                                st.session_state.last_race_id_odds = None

                            # Race change detection to clear cache specific to logic if needed, 
                            # but cache dict with race_id key is safer.
                            
                            # Button to fetch
                            col_btn, col_info = st.columns([1, 2])
                            with col_btn:
                                if st.button("🔴 最新オッズ取得 (Real-time)"):
                                    with st.spinner("複合オッズ(O2-O6)を取得中..."):
                                        c_odds = loader.load_complex_odds(target_date, [selected_race_id])
                                        if selected_race_id not in st.session_state.complex_odds_cache:
                                            st.session_state.complex_odds_cache[selected_race_id] = {}
                                        st.session_state.complex_odds_cache[selected_race_id].update(c_odds.get(selected_race_id, {}))
                                        st.success("オッズ更新完了")
                            
                            # Check availability
                            available_odds_types = []
                            has_odds = False
                            if selected_race_id in st.session_state.complex_odds_cache:
                                has_odds = True
                                available_odds_types = list(st.session_state.complex_odds_cache[selected_race_id].keys())

                            if has_odds:
                                with col_info:
                                    st.caption(f"取得済み: {', '.join(available_odds_types)}")
                            else:
                                with col_info:
                                    st.caption("最新オッズ未取得 (ボタンを押して取得)")

                            # Helper to get odds from session_state
                            def get_odds_val(ticket_type, key):
                                if selected_race_id not in st.session_state.complex_odds_cache: return None
                                odds_data = st.session_state.complex_odds_cache[selected_race_id]
                                if ticket_type not in odds_data: return None
                                return odds_data[ticket_type].get(key)


                            # Top Horses
                            top_df = display_df.head(6) 
                            if len(top_df) >= 1:
                                # 軸馬 (1位)
                                axis_horse = top_df.iloc[0]
                                axis_num = int(axis_horse['horse_number'])
                                axis_name = axis_horse['horse_name']
                                
                                # 相手 (2位〜6位, 最大5頭)
                                opponents = top_df.iloc[1:6]
                                opp_nums = [int(x) for x in opponents['horse_number'].tolist()]
                                opp_nums_str = ", ".join(map(str, opp_nums))
                                
                                # Box用の馬 (上位5頭)
                                box_horses = top_df.head(5)
                                box_nums = [int(x) for x in box_horses['horse_number'].tolist()]
                                box_nums_str = ", ".join(map(str, box_nums))

                                b_col1, b_col2 = st.columns(2)
                                
                                with b_col1:
                                    st.info("🔄 流し (Formation)")
                                    st.markdown(f"**軸**: {axis_num} ({axis_name})")
                                    st.markdown(f"**相手**: {opp_nums_str} ({len(opp_nums)}点)")
                                    
                                    # Odds Calculation for Formation
                                    # Umaren
                                    umaren_odds_list = []
                                    sanren_odds_list = []
                                    
                                    # Umaren
                                    for o in opp_nums:
                                        k = f"{min(axis_num, o):02}{max(axis_num, o):02}"
                                        val = get_odds_val('umaren', k)
                                        if val: umaren_odds_list.append(val)
                                    
                                    # Sanrenpuku (Axis - Opp - Opp)
                                    from itertools import combinations
                                    if len(opp_nums) >= 2:
                                        for c in combinations(opp_nums, 2):
                                            # Axis + 2 Opps
                                            combo = sorted([axis_num, c[0], c[1]])
                                            k = f"{combo[0]:02}{combo[1]:02}{combo[2]:02}"
                                            val = get_odds_val('sanrenpuku', k)
                                            if val: sanren_odds_list.append(val)

                                    uma_range = f"{min(umaren_odds_list)}~{max(umaren_odds_list)}" if umaren_odds_list else "---"
                                    san_range = f"{min(sanren_odds_list)}~{max(sanren_odds_list)}" if sanren_odds_list else "---"

                                    st.code(f"""
単勝     : {axis_num}
馬連     : {axis_num} - {opp_nums_str} (オッズ: {uma_range})
3連複    : {axis_num} - {opp_nums_str} (オッズ: {san_range})
                                    """.strip(), language="text")

                                with b_col2:
                                    st.success("📦 ボックス (Box)")
                                    if len(box_nums) >= 3:
                                        st.markdown(f"**対象**: {box_nums_str} ({len(box_nums)}頭 Box)")
                                        
                                        # Odds Calculation for Box
                                        umaren_box_odds = []
                                        sanren_box_odds = []
                                        
                                        # Umaren Box
                                        for c in combinations(box_nums, 2):
                                            c_sorted = sorted(c)
                                            k = f"{c_sorted[0]:02}{c_sorted[1]:02}"
                                            val = get_odds_val('umaren', k)
                                            if val: umaren_box_odds.append(val)
                                            
                                        # Sanrenpuku Box
                                        for c in combinations(box_nums, 3):
                                            c_sorted = sorted(c)
                                            k = f"{c_sorted[0]:02}{c_sorted[1]:02}{c_sorted[2]:02}"
                                            val = get_odds_val('sanrenpuku', k)
                                            if val: sanren_box_odds.append(val)

                                        u_range = f"{min(umaren_box_odds)}~{max(umaren_box_odds)}" if umaren_box_odds else "---"
                                        s_range = f"{min(sanren_box_odds)}~{max(sanren_box_odds)}" if sanren_box_odds else "---"
                                        
                                        st.code(f"""
馬連     : {box_nums_str} (オッズ: {u_range})
3連複    : {box_nums_str} (オッズ: {s_range})
                                        """.strip(), language="text")
                                    else:
                                        st.write("頭数が少ないためBox推奨はありません。")
                            else:
                                st.warning("予測データが不足しており買い目を生成できません。")

                    except Exception as e:
                        st.error(f"予測実行中にエラーが発生しました: {e}")
                        import traceback
                        st.text(traceback.format_exc())
