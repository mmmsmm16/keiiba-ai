import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

def render_radar_chart(row):
    """
    Render a radar chart for a single horse using its features.
    Args:
        row (pd.Series): A row from the dataframe containing feature values.
    """
    # Define Axes and Normalize Values (Dummy logic for now, should use actual features)
    # Mapping actual features to abstract axes:
    # Speed: speed_index, time_index
    # Stamina: distance_suitability
    # Condition: interval, weight_change
    # Jockey: jockey_win_rate
    # Pedigree: sire_win_rate
    
    # Mock scores for demonstration (0-100)
    # In production, these should be calculated from actual features.
    
    # Extract feature values or simulate them
    speed = min(100, max(20, (row.get('speed_index', 50) + 0) * 1)) # Placeholder
    stamina = 50 # Placeholder
    condition = 60 # Placeholder
    jockey = min(100, row.get('jockey_win_rate', 0.1) * 200) # 0.1 -> 20
    pedigree = 50 # Placeholder
    bias = 50 # Placeholder

    # If we have score, use it as 'Total' proxy
    total = row.get('score', 0) * 100
    
    categories = ['Speed (指数)', 'Stamina (距離)', 'Condition (調子)', 'Jockey (騎手)', 'Pedigree (血統)', 'Bias (展開)']
    values = [speed, stamina, condition, jockey, pedigree, bias]
    
    # Close the loop
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=row['horse_name'],
        line_color='#6366f1'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_position_map(df_race):
    """
    Render a Race Position Map (Tenkai-Zu)
    """
    st.markdown("##### 展開予想図 (Position Map)")
    
    # 1. Determine Pace
    # Logic: Average 'nige_rate' or similar.
    # Placeholder: Randomly determine pace
    pace = "Middle" 
    
    # 2. Categorize Horses by Leg Type (Nige, Senko, Sashi, Oikomi)
    # Using 'leg_type' feature if available, or guessing from 'passing_rank' history
    
    nige = []
    senko = []
    sashi = []
    oikomi = []
    
    for i, row in df_race.iterrows():
        # Mock logic: use horse number hash or random
        # In integration, use `last_3f` or `passing_rank_mean` features
        hn = row['horse_number']
        name = row['horse_name'][:5]
        
        # Simple heuristic using horse number for demo (Replace with Feature Logic!)
        val = (int(hn) * 7) % 4
        if val == 0: nige.append(f"#{hn} {name}")
        elif val == 1: senko.append(f"#{hn} {name}")
        elif val == 2: sashi.append(f"#{hn} {name}")
        else: oikomi.append(f"#{hn} {name}")
        
    # 3. Visual Representation (Simple HTML/CSS Grid)
    
    # Style
    st.markdown("""
    <style>
    .pos-container {
        display: flex;
        align-items: center;
        background-color: #e2e8f0;
        border-radius: 50px; /* Oval track look */
        padding: 20px;
        overflow-x: auto;
        margin-bottom: 20px;
    }
    .pos-group {
        display: flex;
        flex-direction: column;
        margin-right: 30px;
        align-items: center;
    }
    .pos-label {
        font-size: 0.8rem;
        font-weight: bold;
        color: #64748b;
        margin-bottom: 5px;
    }
    .horse-badge {
        background-color: white;
        border: 1px solid #cbd5e1;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-bottom: 4px;
        white-space: nowrap;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .arrow {
        font-size: 2rem;
        color: #94a3b8;
        margin-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render
    c_nige = "".join([f"<div class='horse-badge'>{h}</div>" for h in nige])
    c_senko = "".join([f"<div class='horse-badge'>{h}</div>" for h in senko])
    c_sashi = "".join([f"<div class='horse-badge'>{h}</div>" for h in sashi])
    c_oikomi = "".join([f"<div class='horse-badge'>{h}</div>" for h in oikomi])
    
    html = f"""
    <div class='pos-container'>
        <div class='arrow'>Goal ◀</div>
        <div class='pos-group'>
            <div class='pos-label'>逃げ (Lead)</div>
            {c_nige}
        </div>
        <div class='pos-group'>
            <div class='pos-label'>先行 (Forward)</div>
            {c_senko}
        </div>
        <div class='pos-group'>
            <div class='pos-label'>差し (Mid)</div>
            {c_sashi}
        </div>
        <div class='pos-group'>
            <div class='pos-label'>追込 (Back)</div>
            {c_oikomi}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    st.caption(f"Predicted Pace: {pace}")

def render_comparison(df_race):
    """
    Render Head-to-Head Comparison
    """
    st.markdown("##### ⚔️ 2頭比較 (Head-to-Head)")
    
    # Selectors
    opts = df_race.apply(lambda r: f"#{r['horse_number']} {r['horse_name']}", axis=1).tolist()
    
    c1, c2 = st.columns(2)
    with c1:
        h1_str = st.selectbox("Horse A", opts, index=0)
    with c2:
        h2_str = st.selectbox("Horse B", opts, index=1 if len(opts)>1 else 0)
        
    if h1_str == h2_str:
        st.warning("異なる馬を選択してください")
        return

    # Extract rows
    h1_no = int(h1_str.split('#')[1].split(' ')[0])
    h2_no = int(h2_str.split('#')[1].split(' ')[0])
    
    row1 = df_race[df_race['horse_number'] == h1_no].iloc[0]
    row2 = df_race[df_race['horse_number'] == h2_no].iloc[0]
    
    # Comparison Table
    comp_data = {
        'Metric': ['Prediction Score', 'Rank', 'Odds', 'Jockey', 'Weight'],
        f"{row1['horse_name']}": [
            f"{row1['score']:.4f}",
            f"#{row1.get('rank', '-')}",
            f"{row1.get('odds', '-')}",
            f"{row1.get('jockey_id', '-')}", # Should be name if available
            f"{row1.get('weight', '-')}"
        ],
        f"{row2['horse_name']}": [
            f"{row2['score']:.4f}",
            f"#{row2.get('rank', '-')}",
            f"{row2.get('odds', '-')}",
            f"{row2.get('jockey_id', '-')}",
            f"{row2.get('weight', '-')}"
        ]
    }
    
    st.table(pd.DataFrame(comp_data).set_index('Metric'))
    
    # Render Dual Radar Chart?
    # TODO: Implement overlay radar chart for comparison
