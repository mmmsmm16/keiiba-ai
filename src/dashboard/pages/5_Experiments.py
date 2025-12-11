import streamlit as st
import pandas as pd
import json
import yaml
import os
import glob
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="実験管理 (MLOps)", page_icon="🧪", layout="wide")

st.title("🧪 実験管理 (MLOps)")
st.caption("過去の実験結果を比較・分析します。")

# 実験ディレクトリのパス
EXPERIMENTS_DIR = "experiments"

def load_experiments():
    experiments = []
    
    if not os.path.exists(EXPERIMENTS_DIR):
        return pd.DataFrame()

    # ディレクトリ一覧取得 (タイムスタンプ順)
    dirs = [d for d in glob.glob(os.path.join(EXPERIMENTS_DIR, "*")) if os.path.isdir(d)]
    dirs.sort(key=os.path.getmtime, reverse=True)
    
    for d in dirs:
        exp_name = os.path.basename(d)
        config_path = os.path.join(d, "config_snapshot.yaml")
        metrics_path = os.path.join(d, "reports", "metrics.json")
        
        # 必須ファイルがない場合はスキップ
        if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
            continue
            
        try:
            # Config読み込み
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
            # Metrics読み込み
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            # 代表指標の抽出
            # アンサンブルがある場合はEnsemble、なければ先頭のモデル
            metric_data = {}
            if "metrics" in metrics:
                if "Ensemble" in metrics["metrics"]:
                    metric_data = metrics["metrics"]["Ensemble"]
                    model_name = "Ensemble"
                else:
                    model_name = list(metrics["metrics"].keys())[0]
                    metric_data = metrics["metrics"][model_name]
            else:
                # 互換性維持 (古いフォーマット)
                metric_data = {"roi": 0, "accuracy": 0}
                model_name = "Unknown"

            # タイムスタンプ
            timestamp = datetime.fromtimestamp(os.path.getmtime(d)).strftime('%Y-%m-%d %H:%M')

            experiments.append({
                "Experiment": exp_name,
                "Date": timestamp,
                "Model Type": config.get("model", {}).get("type", "unknown"),
                "Representative Model": model_name,
                "ROI (%)": metric_data.get("roi", 0),
                "Accuracy (%)": metric_data.get("accuracy", 0),
                "Valid Year": config.get("data", {}).get("valid_year", "N/A"),
                "Features": config.get("data", {}).get("features", "N/A"),
                "Dropped Features": str(config.get("data", {}).get("drop_features", [])),
                "Path": d
            })

        except Exception as e:
            st.error(f"Error loading {exp_name}: {e}")
            continue
            
    return pd.DataFrame(experiments)

df = load_experiments()

if df.empty:
    st.warning("実験データが見つかりません。パイプラインを実行してください。")
else:
    # 1. 指標比較（テーブル）
    st.subheader("📊 実験一覧")
    
    # ROIで色分け
    st.dataframe(
        df.style.background_gradient(subset=["ROI (%)"], cmap="Greens").format({"ROI (%)": "{:.2f}", "Accuracy (%)": "{:.2%}"}),
        use_container_width=True
    )
    
    # 2. グラフ比較
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROI 比較")
        fig_roi = px.bar(df, x="Experiment", y="ROI (%)", color="Model Type", 
                         hover_data=["Representative Model", "Valid Year"], title="ROI by Experiment")
        fig_roi.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Break Even (100%)")
        st.plotly_chart(fig_roi, use_container_width=True)
        
    with col2:
        st.subheader("🎯 Accuracy 比較")
        fig_acc = px.bar(df, x="Experiment", y="Accuracy (%)", color="Model Type", 
                         hover_data=["Representative Model", "Valid Year"], title="Accuracy by Experiment")
        st.plotly_chart(fig_acc, use_container_width=True)

    # 3. 詳細確認
    st.subheader("🔍 実験詳細")
    selected_exp = st.selectbox("詳細を確認する実験を選択", df["Experiment"].unique())
    
    if selected_exp:
        exp_row = df[df["Experiment"] == selected_exp].iloc[0]
        exp_dir = exp_row["Path"]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Model**: {exp_row['Model Type']}")
        with c2:
            st.success(f"**ROI**: {exp_row['ROI (%)']:.2f}%")
        with c3:
            st.warning(f"**Target Year**: {exp_row['Valid Year']}")

        # Config表示
        with st.expander("Explore Configuration (config.yaml)"):
            with open(os.path.join(exp_dir, "config_snapshot.yaml"), "r") as f:
                st.code(f.read(), language="yaml")
                
        # Metrics詳細
        with st.expander("Explore Full Metrics (metrics.json)"):
            with open(os.path.join(exp_dir, "reports", "metrics.json"), "r") as f:
                st.json(json.load(f))
                
        # 戦略レポート
        opt_path = os.path.join(exp_dir, "reports", "optimization_report.json")
        if os.path.exists(opt_path):
            with st.expander("Explore Strategy Optimization (optimization_report.json)"):
                with open(opt_path, "r") as f:
                    opt_res = json.load(f)
                    
                # ベスト戦略の表示
                if "best_strategies" in opt_res and opt_res["best_strategies"]:
                    st.markdown("### 🏆 Best Strategies")
                    best_df = pd.json_normalize(opt_res["best_strategies"])
                    st.dataframe(best_df[['name', 'roi', 'total_return', 'bet_count']])
                    
                st.markdown("### JSON Raw")
                st.json(opt_res)
        
        # Plot表示
        img_path = os.path.join(exp_dir, "reports", "lgbm_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Feature Importance (LightGBM)")
