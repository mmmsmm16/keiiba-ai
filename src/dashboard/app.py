import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# ページ設定
st.set_page_config(
    page_title="Project Strongest Dashboard",
    layout="wide"
)

st.title("🏇 Project Strongest: Analysis Dashboard")

# ディレクトリ設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '../../')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# タブ作成
tab1, tab2, tab3 = st.tabs(["Overview", "Feature Importance", "Simulation (ROI)"])

with tab1:
    st.header("Experiment History")
    
    history_path = os.path.join(EXPERIMENTS_DIR, 'history.csv')
    if os.path.exists(history_path):
        df_history = pd.read_csv(history_path)
        st.dataframe(df_history)
        
        # メトリクスの推移プロット
        st.subheader("Metrics Trend")
        if not df_history.empty:
            metric = st.selectbox("Select Metric", ["rmse", "ndcg", "map@10"], index=1)
            if metric in df_history.columns:
                st.line_chart(df_history.set_index('timestamp')[metric])
            else:
                st.warning(f"Metric '{metric}' not found in history.")
    else:
        st.warning("Experiment history not found. Run training first.")

with tab2:
    st.header("Feature Importance")
    
    # 簡易的にTabNetの保存済み画像を表示
    # 本格的には pickle したモデルをロードしてプロットしたいが、Streamlitでtorch/lgbmをロードするのは重い可能性がある
    
    tabnet_imp_path = os.path.join(MODELS_DIR, 'tabnet_importance.png')
    if os.path.exists(tabnet_imp_path):
        st.image(tabnet_imp_path, caption="TabNet Feature Importance")
    else:
        st.info("TabNet importance plot not found.")
        
    # 将来的には LightGBM / CatBoost の Plot もここに追加

with tab3:
    st.header("ROI Simulation")
    
    sim_path = os.path.join(EXPERIMENTS_DIR, 'latest_simulation.json')
    if os.path.exists(sim_path):
        with open(sim_path, 'r') as f:
            sim_data = json.load(f)
            
        st.markdown(f"**Last Updated:** {sim_data.get('timestamp')}")
        
        # 1. 戦略別サマリ
        st.subheader("Strategy Summary (Single Bet)")
        strat = sim_data.get('strategies', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Max Expected Value")
            max_ev = strat.get('max_ev', {})
            st.metric("ROI", f"{max_ev.get('roi', 0):.2f}%")
            st.metric("Hit Rate", f"{max_ev.get('accuracy', 0):.2%}")
            
        with col2:
            st.markdown("### Max Score")
            max_score = strat.get('max_score', {})
            st.metric("ROI", f"{max_score.get('roi', 0):.2f}%")
            st.metric("Hit Rate", f"{max_score.get('accuracy', 0):.2%}")

        # 2. ROI Curve
        st.subheader("ROI Curve (Threshold Strategy)")
        curve_data = sim_data.get('roi_curve', [])
        
        if curve_data:
            df_curve = pd.DataFrame(curve_data)
            
            # グラフ描画
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.set_xlabel('Expected Value Threshold')
            ax1.set_ylabel('ROI (%)', color='tab:blue')
            ax1.plot(df_curve['threshold'], df_curve['roi'], color='tab:blue', marker='o', label='ROI')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.axhline(100, color='gray', linestyle='--', alpha=0.7) # 100%ライン
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Bet Count', color='tab:orange')
            ax2.bar(df_curve['threshold'], df_curve['bet_count'], color='tab:orange', alpha=0.3, width=0.05, label='Bet Count')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
            st.pyplot(fig)
            
            st.dataframe(df_curve)
        else:
            st.warning("No curve data found.")
            
    else:
        st.warning("Simulation data not found. Run 'src/model/evaluate.py' first.")
