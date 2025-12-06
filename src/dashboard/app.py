import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

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

# タブ作成
tab1, tab2, tab3, tab4 = st.tabs(["概要 (Overview)", "特徴量重要度 (Feature Importance)", "シミュレーション (ROI)", "予測実行 (Predict)"])

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
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 最大期待値 (Max EV)")
            max_ev = strat.get('max_ev', {})
            st.metric("回収率 (ROI)", f"{max_ev.get('roi', 0):.2f}%")
            st.metric("的中率 (Hit)", f"{max_ev.get('accuracy', 0):.2%}")
            
        with col2:
            st.markdown("### 最大スコア (Max Score)")
            max_score = strat.get('max_score', {})
            st.metric("回収率 (ROI)", f"{max_score.get('roi', 0):.2f}%")
            st.metric("的中率 (Hit)", f"{max_score.get('accuracy', 0):.2%}")

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
        st.subheader("複合馬券シミュレーション (Box 5)")
        st.markdown("スコア上位5頭をBOX買いした場合の回収率シミュレーション")
        
        strategies = sim_data.get('strategies', {})
        complex_keys = ['umaren_box5', 'sanrenpuku_box5', 'sanrentan_box5']
        
        complex_data = []
        names = {
            'umaren_box5': '馬連 Box5 (10点)', 
            'sanrenpuku_box5': '3連複 Box5 (10点)', 
            'sanrentan_box5': '3連単 Box5 (60点)'
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
# Tab 4: 予測実行 (Real-time Prediction)
# --------------------------------------------------------------------------------
with tab4:
    st.header("予測実行 (Real-time Prediction)")
    st.markdown("指定した日付・レースの予測をリアルタイムに実行します。データはPC-KEIBA DBから取得します。")

    # 必要なモジュールのインポート (ここでインポートしてスコープを限定)
    import sys
    sys.path.append(os.path.join(BASE_DIR, '../'))
    from inference.loader import InferenceDataLoader
    from inference.preprocessor import InferencePreprocessor
    from model.ensemble import EnsembleModel
    from scipy.special import softmax

    # キャッシュされたモデルとプリプロセッサのロード関数
    @st.cache_resource
    def load_model_resources():
        model_path = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
        if not os.path.exists(model_path):
            return None
        model = EnsembleModel()
        model.load_model(model_path)
        return model

    @st.cache_resource
    def load_preprocessor_resources():
        # Preprocessorは過去データをロードするため重い。キャッシュする。
        preprocessor = InferencePreprocessor()
        # ダミー実行でデータをロードさせる（InferencePreprocessorの設計次第だが、
        # preprocess初回呼び出し時にロードされるなら、ここで明示的に呼び出すか、
        # インスタンスをキャッシュして使い回す）
        # constructorではロードしない実装だったため、preprocessメソッド内でロードされる。
        # データフレーム自体をキャッシュする設計に変更しないと毎回ロードされる可能性がある。
        # src/inference/preprocessor.py を見ると、preprocess() 内で pd.read_parquet している。
        # これを回避するには preprocessor 自体にデータを保持させるか、データロード関数を分ける必要がある。
        # ここでは簡易的に、preprocessorインスタンスをキャッシュする（ただしparquet読み込みはキャッシュされないかも）。
        # データロード部分だけキャッシュ関数にするのがベスト。
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
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        selected_date = st.date_input(
            "開催日",
            value=pd.Timestamp.now(),
            min_value=pd.Timestamp('2020-01-01'),
            max_value=pd.Timestamp.now() + pd.Timedelta(days=30)
        )
        target_date = selected_date.strftime('%Y%m%d')
    with col_in2:
        venue_map = {
            '01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京', 
            '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'
        }
        venue_code = st.selectbox("開催場所", options=list(venue_map.keys()), format_func=lambda x: f"{x}: {venue_map[x]}")
    with col_in3:
        race_no = st.number_input("レース番号", min_value=1, max_value=12, value=11)

    # Race ID (Check purpose only, not used for querying)
    # PC-KEIBA DB Key is (Year, Venue, Kai, Nichime, Race), not (Date, Venue, Race).
    # So we load by Date and filter in memory.
    st.info(f"Target: {target_date} / {venue_map.get(venue_code)} / {race_no}R")

    if st.button("予測実行 (Predict)"):
        with st.spinner('モデルとデータを準備中...'):
            model = load_model_resources()
            hist_df = get_historical_data() # Cached load
            
            if model is None:
                st.error("モデルファイルが見つかりません (models/ensemble_model.pkl)")
            elif hist_df is None:
                st.error("過去データが見つかりません (data/processed/preprocessed_data.parquet)")
            else:
                # 1. データロード
                loader = InferenceDataLoader()
                try:
                    # Load all races for the date then filter
                    new_df = loader.load(target_date=target_date)
                except Exception as e:
                    new_df = pd.DataFrame()
                    st.error(f"データロードエラー: {e}")

                # Filter by Venue and Race No
                if not new_df.empty:
                    # Ensure types match for filtering
                    # new_df['venue'] is code string '05', venue_code is '05'
                    # new_df['race_number'] is int, race_no is int
                    new_df = new_df[
                        (new_df['venue'] == venue_code) & 
                        (new_df['race_number'] == race_no)
                    ]

                if new_df.empty:
                    st.warning(f"データが見つかりませんでした (Date: {target_date}, Venue: {venue_code}, Race: {race_no})。PC-KEIBAでデータ登録済みか確認してください。")
                else:
                    st.success(f"データロード成功: {len(new_df)} 頭")
                    
                    # 2. 前処理
                    # ヒストリカルデータのロードを高速化するためにキャッシュ利用
                    preprocessor = InferencePreprocessor()
                    
                    try:
                        # 修正: キャッシュしたhistory_dfを渡す
                        X, ids = preprocessor.preprocess(new_df, history_df=hist_df)
                        
                        if X.empty:
                            st.error("前処理後の特徴量が生成できませんでした。")
                        else:
                            # レース情報カードの表示
                            race_info = new_df.iloc[0]
                            
                            # マッピング辞書
                            surface_map = {'10': '芝', '11': '芝・直線', '20': 'ダート', '21': 'ダート・直線', '30': '障害・芝', '31': '障害・芝直線'}
                            state_map = {'1': '良', '2': '稍重', '3': '重', '4': '不良'}
                            weather_map = {'1': '晴', '2': '曇', '3': '雨', '4': '小雨', '5': '小雪', '6': '雪'}
                            
                            st.markdown("---")
                            st.subheader(f"📋 レース情報")
                            
                            info_col1, info_col2, info_col3 = st.columns(3)
                            with info_col1:
                                st.metric("レース名", race_info.get('title', 'N/A'))
                                st.metric("距離", f"{race_info.get('distance', 'N/A')}m")
                            with info_col2:
                                surf = surface_map.get(str(race_info.get('surface', '')), 'N/A')
                                st.metric("馬場", surf)
                                state = state_map.get(str(race_info.get('state', '')), 'N/A')
                                st.metric("馬場状態", state)
                            with info_col3:
                                weather = weather_map.get(str(race_info.get('weather', '')), 'N/A')
                                st.metric("天候", weather)
                                st.metric("出走頭数", f"{len(new_df)}頭")
                            
                            st.markdown("---")
                            
                            # 3. 予測
                            preds = model.predict(X)
                            
                            # 結果整形
                            results = ids.copy()
                            results['score'] = preds
                            results['prob'] = results.groupby('race_id')['score'].transform(lambda x: softmax(x))
                            
                            # 期待値計算
                            results['expected_value'] = results['prob'] * results['odds']
                            results['recommended'] = results['expected_value'] > 1.0
                            
                            # 表示用カラム
                            results['pred_rank'] = results.groupby('race_id')['score'].rank(ascending=False, method='min')
                            
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
                            display_df = display_df.rename(columns=rename_map)
                            
                            # 色付けとハイライト
                            st.subheader(f"🎯 予測結果: {venue_map.get(venue_code, venue_code)} {race_no}R")
                            
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

                            # style.apply は axis=1 で行ごとに適用
                            st.dataframe(
                                display_df.style.apply(highlight_rows, axis=1).format({
                                    '予測スコア': '{:.4f}', 
                                    'AI勝率': '{:.2%}',
                                    '期待値': '{:.2f}'
                                })
                            )

                    except Exception as e:
                        st.error(f"予測実行中にエラーが発生しました: {e}")
