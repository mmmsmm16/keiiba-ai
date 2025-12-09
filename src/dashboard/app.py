import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '../../')
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.inference.loader import InferenceDataLoader

# --- Config & Setup ---
st.set_page_config(
    page_title="Keiiba-AI Dashboard 2.0",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Definitions ---
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
IMG_DIR = os.path.join(ASSETS_DIR, 'images') # if any

# --- Helper Functions ---
def load_css(theme='light'):
    """Load Custom CSS based on theme"""
    css_path = os.path.join(ASSETS_DIR, 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            base_css = f.read()
        
        # Inject Dark Mode Overrides directly if active
        if theme == 'dark':
            dark_css = """
            /* Global Dark Overrides */
            .stApp {
                background-color: #0f172a !important;
                color: #f1f5f9 !important;
            }
            
            /* Dark Mode Buttons - Broad & Specific Selectors */
            html body .stApp button,
            html body .stApp .stButton > button,
            html body .stApp [data-testid="stButton"] > button {
                background-color: #1e293b !important; /* Slate-800 */
                color: #e2e8f0 !important; /* Slate-200 */
                border: 1px solid #334155 !important;
            }

            html body .stApp button:hover,
            html body .stApp .stButton > button:hover,
            html body .stApp [data-testid="stButton"] > button:hover {
                background-color: #334155 !important; /* Slate-700 */
                color: #ffffff !important;
                border-color: #475569 !important;
            }
            
            /* Primary Button (Selected) in Dark Mode */
            html body .stApp button[kind="primary"],
            html body .stApp .stButton > button[kind="primary"],
            html body .stApp [data-testid="stButton"] > button[kind="primary"] {
                background-color: #6366f1 !important;
                color: white !important;
                border-color: #6366f1 !important;
            }
            
            /* Sidebar Buttons in Dark */
            section[data-testid="stSidebar"] button,
            section[data-testid="stSidebar"] .stButton > button {
                background-color: transparent !important;
                border: 1px solid #475569 !important;
                color: #94a3b8 !important;
            }
            
            section[data-testid="stSidebar"] button:hover,
            section[data-testid="stSidebar"] .stButton > button:hover {
                background-color: #1e293b !important;
                color: #f1f5f9 !important;
            }

            /* Global Text Fixes */
            h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, .element-container {
                color: #f1f5f9 !important;
            }
            
            /* Metric Labels */
            .metric-label { color: #94a3b8 !important; }
            """
            base_css += dark_css
        
        st.markdown(f'<style>{base_css}</style>', unsafe_allow_html=True)

@st.cache_resource
def get_loader():
    return InferenceDataLoader()

# --- Session State Init ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

# --- Sidebar Navigation ---
def render_sidebar():
    with st.sidebar:
        st.title("🏇 Keiiba-AI")
        st.caption("Professional Racing Analytics")
        
        st.markdown("---")
        
        # Navigation
        st.write("MENU")
        
        # Use buttons for navigation to keep it seamless
        if st.button("🏠 Home (Predictions)", use_container_width=True, type="primary" if st.session_state['page']=='Home' else "secondary"):
            st.session_state['page'] = 'Home'
            st.rerun()
            
        if st.button("📅 Schedule", use_container_width=True, type="primary" if st.session_state['page']=='Schedule' else "secondary"):
            st.session_state['page'] = 'Schedule'
            st.rerun()
            
        if st.button("📈 Performance", use_container_width=True, type="primary" if st.session_state['page']=='Performance' else "secondary"):
            st.session_state['page'] = 'Performance'
            st.rerun()

        st.markdown("---")
        st.write("SETTINGS")
        
        # Theme Toggle
        is_dark = st.session_state['theme'] == 'dark'
        toggle = st.toggle("🌙 Dark Mode", value=is_dark)
        
        new_theme = 'dark' if toggle else 'light'
        if new_theme != st.session_state['theme']:
            st.session_state['theme'] = new_theme
            st.rerun()

        st.markdown("---")
        st.caption(f"v5.0.0 | Theme: {st.session_state['theme'].title()}")
        # Debug Indicator
        # st.code(f"Current State: {st.session_state['theme']}")

# --- Page: Home (Predictions) ---
def render_home():
    st.header("🏁 Predictions (Today)")
    
    # 1. Load Latest Predictions
    # Logic: Look for latest prediction parquet file
    pred_files = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, 'predictions_*.parquet')), reverse=True)
    
    if not pred_files:
        st.info("予測データが見つかりません。まずは 'run_weekly.py' または 'predict.py' を実行してください。")
        return

    # Selector for Model/Date
    options = {os.path.basename(f): f for f in pred_files[:5]} # Top 5 recent
    selected_file_name = st.selectbox("Select Prediction File", list(options.keys()))
    selected_file_path = options[selected_file_name]
    
    try:
        df = pd.read_parquet(selected_file_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    if df.empty:
        st.warning("Empty prediction file.")
        return

    # Extract dates
    if 'date' in df.columns:
        dates = sorted(df['date'].unique(), reverse=True)
        date_opts = [str(d)[:10] for d in dates]
        
        # Check if target_date comes from Schedule
        default_ix = 0
        if 'target_date' in st.session_state and st.session_state['target_date'] in date_opts:
            default_ix = date_opts.index(st.session_state['target_date'])
        
        selected_date_str = st.selectbox("Select Date", date_opts, index=default_ix)
        
        # Filter
        df_display = df[df['date'].astype(str).str.contains(selected_date_str)].copy()
    else:
        df_display = df.copy()

    # --- Hybrid View ---
    # Left: Race List, Right: Analysis Card
    
    # --- Hybrid View (Revised) ---
    # Left: Race List (Compact), Right: Detail Card
    
    # 1:3 Ratio for a narrower list column
    col_list, col_detail = st.columns([1.5, 4.5])
    
    with col_list:
        st.markdown("##### Race List")
        
        # Define races
        races = sorted(df_display['race_id'].unique())
        
        if not races:
            st.info("No races.")
        else:
            # Group by Venue
            venue_groups = {}
            for rid in races:
                row = df_display[df_display['race_id'] == rid].iloc[0]
                venue_code = str(rid)[4:6]
                v_map = {'01':'札幌', '02':'函館', '03':'福島', '04':'新潟', '05':'東京', 
                         '06':'中山', '07':'中京', '08':'京都', '09':'阪神', '10':'小倉'}
                venue = v_map.get(venue_code, f"{venue_code}")
                
                if venue not in venue_groups:
                    venue_groups[venue] = []
                venue_groups[venue].append(rid)
            
            # Sort venues
            sorted_venues = sorted(venue_groups.keys())
            
            # Create Nested Columns within col_list
            # If too many venues, maybe limit? But usually 2-3 venues max per day.
            v_cols = st.columns(len(sorted_venues))
            
            for i, venue in enumerate(sorted_venues):
                with v_cols[i]:
                    st.caption(f"**{venue}**")
                    for rid in venue_groups[venue]:
                        # Get Race Info
                        row = df_display[df_display['race_id'] == rid].iloc[0]
                        race_no = row.get('race_number', int(str(rid)[10:12]))
                        race_name = row.get('race_name', '-')
                        
                        # Compact Label: "1R Name" -> "1R" maybe too short? 
                        # User wants 2 columns in the sidebar space. 
                        # Text might wrap if too long. Let's try "1R" + Name truncated or just "1R" if width is issue
                        # But user said "縦にレースを書く".
                        
                        label = f"{race_no}R" 
                        # Tooltip or separate text for name? 
                        # Providing full name might be too wide for 2 cols in 1.5/6 width.
                        # Let's try "1R Name" first.
                        # Actually "1R" is enough if column header is Venue.
                        
                        label = f"{race_no}R"
                        if race_name and race_name != '-' and race_name != '':
                             label += f" {race_name}"
                        
                        # Checkbox-like button behavior
                        is_sel = (st.session_state['selected_race_id'] == rid)
                        btn_kind = "primary" if is_sel else "secondary"
                        
                        # Add tooltip for potential truncated text
                        tooltip = f"{venue} {race_no}R: {race_name} ({row.get('start_time', '')})"
                        
                        if st.button(label, key=f"lst_{rid}", type=btn_kind, use_container_width=True, help=tooltip):
                            st.session_state['selected_race_id'] = rid
                            st.rerun()

    with col_detail:
        rid = st.session_state['selected_race_id']
        if rid:
             render_race_detail(df_display[df_display['race_id'] == rid])
        elif races:
             # Auto select first
             st.session_state['selected_race_id'] = races[0]
             st.rerun()

from src.dashboard.components.charts import render_radar_chart, render_position_map, render_comparison

def render_race_detail(df_race):
    if df_race.empty: return
    
    meta = df_race.iloc[0]
    st.subheader(f"{meta.get('race_name', 'Race Detail')} (ID: {meta['race_id']})")
    
    # Sort by score
    df_race = df_race.sort_values('score', ascending=False)
    
    # Top Picks Cards
    top3 = df_race.iloc[:3]
    
    c1, c2, c3 = st.columns(3)
    for i, (idx, row) in enumerate(top3.iterrows()):
        cols = [c1, c2, c3]
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"#### #{i+1} {row['horse_name']}")
                st.caption(f"馬番: {row['horse_number']} | 予測Score: {row['score']:.4f}")
                st.markdown(f"**単勝オッズ: {row.get('odds', '-')}**")
                
                # Value Detector
                ev = row.get('expected_value', 0)
                if ev >= 1.2:
                    st.markdown(f"Value: **{ev:.2f}** <span class='badge badge-success'>Overrated</span>", unsafe_allow_html=True) # "Overrated" meaning good value here? or "Undervalued"? usually "Value" means Underpriced. "Good Value"
                    # User asked for "Value Detector". Let's name it "High Value".
                else:
                    st.markdown(f"Value: {ev:.2f}")

    st.markdown("### 📋 全頭リスト (Ranking)")
    
    # Display Table with Styler
    display_cols = ['rank', 'horse_number', 'horse_name', 'jockey_id', 'odds', 'prob', 'score']
    # Filter existing cols
    display_cols = [c for c in display_cols if c in df_race.columns]
    
    st.dataframe(
        df_race[display_cols].style.background_gradient(subset=['score'], cmap='Blues'),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    st.markdown("### 🧠 Deep Analytics")
    
    # Tabs for analytics
    tab_a, tab_b, tab_c = st.tabs(["📊 Radar Chart", "🗺️ Position Map", "⚔️ Head-to-Head"])
    
    with tab_a:
        st.caption("上位馬の戦力分析")
        rc1, rc2 = st.columns(2)
        with rc1:
            if len(top3) > 0:
                render_radar_chart(top3.iloc[0])
        with rc2:
            if len(top3) > 1:
                render_radar_chart(top3.iloc[1])

    with tab_b:
        render_position_map(df_race)

    with tab_c:
        render_comparison(df_race)


# --- Page: Schedule ---
def render_schedule():
    st.header("📅 Race Schedule")
    
    loader = get_loader()
    # Fetch schedule
    try:
        df_sch = loader.get_race_schedule(limit=10) # Next 10 days/venues
    except Exception as e:
        df_sch = pd.DataFrame()
        st.error(f"Failed to load schedule: {e}")

    if df_sch.empty:
        st.warning("開催予定が見つかりませんでした。")
    else:
        # Group by Date
        dates = df_sch['date'].unique()
        for d in dates:
            d_str = str(d)[:10]
            st.markdown(f"### {d_str}")
            
            day_rows = df_sch[df_sch['date'] == d]
            
            cols = st.columns(4)
            for i, (idx, row) in enumerate(day_rows.iterrows()):
                with cols[i % 4]:
                    with st.container(border=True):
                        st.markdown(f"**{row['venue']}** ({row['races_count']}R)")
                        if row['main_race_name']:
                            st.caption(f"Main: {row['main_race_name']}")
                        

                        if st.button("Check", key=f"sch_{d_str}_{row['venue']}"):
                            # Set Page to Home
                            st.session_state['page'] = 'Home'
                            # Pass context (Simple way: query param or session state)
                            # Since render_home reads 'pred_files' etc, we need a way to filter.
                            # For now, let's store 'target_date' in session state
                            st.session_state['target_date'] = d_str
                            st.rerun()

# --- Page: Performance ---
def render_performance():
    st.header("📈 Model Performance")
    st.info("ここにはモデルの精度推移やROIカーブを表示します。（旧ダッシュボードのTab3/4を移植予定）")

# --- Main Entry Point ---
def main():
    # Load Theme CSS
    load_css(st.session_state['theme'])
    
    # Render Layout
    render_sidebar()
    
    # Routing
    if st.session_state['page'] == 'Home':
        render_home()
    elif st.session_state['page'] == 'Schedule':
        render_schedule()
    elif st.session_state['page'] == 'Performance':
        render_performance()
    else:
        render_home()

if __name__ == "__main__":
    main()
