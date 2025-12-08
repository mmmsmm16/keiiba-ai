
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

st.set_page_config(page_title="リスク管理", page_icon="🛡️", layout="wide")

st.title("🛡️ リスク管理 & 資金管理")

# Define Data Paths
BASE_DIR = os.path.dirname(__file__)
EXP_DIR = os.path.join(BASE_DIR, '../../../experiments')
SIM_FILE = os.path.join(EXP_DIR, 'latest_simulation.json')

# --- Helper Functions ---
def load_simulation_results():
    if not os.path.exists(SIM_FILE):
        return None
    try:
        with open(SIM_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"シミュレーションファイルの読み込みエラー: {e}")
        return None

def calculate_kelly_bet(bankroll, win_prob, odds, fraction=0.25):
    """
    Calculate Kelly bet size.
    f* = (bp - q) / b = p - q/b
    where b = odds - 1, p = win_prob, q = 1 - p
    """
    if odds <= 1: return 0
    b = odds - 1
    p = win_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(0, f * fraction * bankroll)

# --- UI Layout ---

# 1. Bankroll Simulation Analysis
st.header("📊 シミュレーション分析")

data = load_simulation_results()

if data:
    st.write(f"**モデル:** {data.get('model', 'Unknown')} | **バージョン:** {data.get('version', 'Unknown')} | **日時:** {data.get('timestamp', '')}")
    
    # ROI Curve Analysis (if available)
    if 'roi_curve' in data and data['roi_curve']:
        roi_df = pd.DataFrame(data['roi_curve'])
        if not roi_df.empty:
            st.subheader("期待値閾値ごとのROI")
            
            # Interactive Chart
            st.line_chart(roi_df.set_index('threshold')[['roi', 'accuracy']])
            
            # Show Table
            st.dataframe(roi_df.style.format({
                "roi": "{:.1f}%", 
                "accuracy": "{:.1%}",
                "bet_count": "{:,}"
            }))
    
    # Strategies Summary
    if 'strategies' in data:
        st.subheader("戦略別パフォーマンス")
        strategies = data['strategies']
        rows = []
        for name, stats in strategies.items():
            rows.append({
                "戦略": name,
                "ROI": stats.get('roi', 0),
                "的中率": stats.get('accuracy', 0),
                "購入数": stats.get('bet_count', stats.get('races', 0)*stats.get('bet', 0)/100), # Approx or exact
                "総払戻": stats.get('return', 0)
            })
        
        st_df = pd.DataFrame(rows)
        # Sort by ROI
        st_df = st_df.sort_values('ROI', ascending=False)
        
        st.dataframe(st_df.style.format({
            "ROI": "{:.1f}%",
            "的中率": "{:.1%}",
            "総払戻": "¥{:,}"
        }))
else:
    st.info("シミュレーション履歴が見つかりません。`src/model/evaluate.py` を実行してデータを生成してください。")

st.divider()

# 2. Bet Size Calculator
st.header("🧮 推奨賭け金計算機 (ケリー基準)")

col1, col2 = st.columns(2)

with col1:
    current_bankroll = st.number_input("現在の資金 (¥)", min_value=1000, value=100000, step=1000)
    risk_tolerance = st.slider("ケリー係数 (リスク許容度)", 0.1, 1.0, 0.25, 0.05, help="目安: 0.25 (クォーターケリー) は安全な資産運用向けです。")

with col2:
    st.info("""
    **ケリー基準の目安:**
    - **フルケリー (1.0)**: 理論上の最大成長。ボラティリティ激高。破産リスクあり。
    - **ハーフケリー (0.5)**: 75%の成長速度で、ボラティリティは半分。
    - **クォーターケリー (0.25)**: スポーツベッティングの標準。安全重視の運用。
    """)

st.subheader("計算機")
c1, c2, c3 = st.columns(3)
with c1:
    odds_input = st.number_input("オッズ", min_value=1.0, value=10.0, step=0.1)
with c2:
    prob_input = st.number_input("勝率 (%)", min_value=1.0, max_value=100.0, value=15.0, step=0.1) / 100.0
with c3:
    ev_input = odds_input * prob_input
    st.metric("期待値 (EV)", f"{ev_input:.2f}")

suggested_wager = calculate_kelly_bet(current_bankroll, prob_input, odds_input, risk_tolerance)
wager_pct = (suggested_wager / current_bankroll) * 100

st.metric("推奨購入額", f"¥{int(suggested_wager):,}", delta=f"資金の {wager_pct:.2f}%")

if ev_input < 1.0:
    st.warning("期待値が1.0未満です。見送りを推奨します。")
elif suggested_wager == 0:
    st.warning("計算された購入額は0です (期待値不足または安全圏外)。")
else:
    st.success(f"この馬への推奨購入額: ¥{int(suggested_wager):,}")

st.divider()
st.caption("Risk Management Module v1.0")
