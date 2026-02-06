
import os
import pandas as pd
from sqlalchemy import create_engine, text

def generate_schema_doc():
    # Configuration
    db_configs = [
        "postgresql://postgres:postgres@localhost:5433/pckeiba",
        "postgresql://postgres:postgres@localhost:5432/pckeiba",
        "postgresql://postgres:postgres@db:5432/pckeiba"
    ]
    
    engine = None
    for conn_str in db_configs:
        try:
            print(f"Trying connection: {conn_str} ...")
            eng = create_engine(conn_str)
            with eng.connect() as conn:
                pass
            print("Connection successful!")
            engine = eng
            break
        except Exception as e:
            print(f"Failed: {e}")
    
    if not engine:
        print("Could not connect to database.")
        return

    # Japanese Translations
    table_map = {
        "jvd_ra": "レース詳細 (RA)",
        "jvd_se": "馬毎レース成績 (SE)",
        "jvd_hr": "払戻 (HR)",
        "jvd_um": "競争馬マスタ (UM)",
        "jvd_ks": "騎手マスタ (KS)",
        "jvd_ch": "調教師マスタ (CH)",
        "jvd_br": "生産者マスタ (BR)",
        "jvd_bn": "馬主マスタ (BN)",
        "jvd_hc": "坂路調教 (HC)",
        "jvd_wc": "ウッド調教 (WC)",
        "jvd_bt": "系統情報 (BT)",
        "jvd_hn": "繁殖馬マスタ (HN)",
        "jvd_sk": "産駒マスタ (SK)",
        "jvd_ck": "着差コード (CK)",
        "jvd_tc": "トラックコード (TC)",
        "jvd_wh": "天候コード (WH)",
        "jvd_we": "馬場状態コード (WE)",
        "jvd_is": "異常区分コード",
        "jvd_dm": "開催ダミー",
        "jvd_jg": "除外情報",
        "jvd_ys": "開催予定",
        "jvd_hy": "馬名由来 (HY)",
        
        "apd_sokuho_o1": "速報オッズ (単勝・複勝)",
        "apd_sokuho_o2": "速報オッズ (枠連)",
        "apd_sokuho_o3": "速報オッズ (ワイド)",
        "apd_sokuho_o4": "速報オッズ (馬単)",
        "apd_sokuho_o5": "速報オッズ (三連複)",
        "apd_sokuho_o6": "速報オッズ (三連単)",
        "apd_se_jv": "JV-Data成績拡張",
        "races": "独自定義レーステーブル",
        "results": "独自定義結果テーブル",
        "horses": "独自定義馬テーブル"
    }

    col_map = {
        "kaisai_nen": "開催年",
        "kaisai_tsukihi": "開催月日",
        "keibajo_code": "競馬場コード",
        "kaisai_kai": "開催回",
        "kaisai_nichime": "開催日目",
        "race_bango": "レース番号",
        "wakuban": "枠番",
        "umaban": "馬番",
        "ketto_toroku_bango": "血統登録番号(馬ID)",
        "bamei": "馬名",
        "kishu_code": "騎手コード",
        "chokyoshi_code": "調教師コード",
        "bataiju": "馬体重",
        "zogen_sa": "増減差",
        "kakutei_chakujun": "確定着順",
        "time_sa": "タイム差",
        "soha_time": "走破タイム",
        "corner_1": "第1コーナー順位",
        "corner_2": "第2コーナー順位",
        "corner_3": "第3コーナー順位",
        "corner_4": "第4コーナー順位",
        "tansho_odds": "単勝オッズ",
        "tansho_ninkijun": "単勝人気順",
        "kohan_3f": "後半3Fタイム",
        "kyori": "距離",
        "track_code": "トラックコード(芝/ダート)",
        "tenko_code": "天候コード",
        "babajotai_code_shiba": "馬場状態(芝)",
        "babajotai_code_dirt": "馬場状態(ダート)",
        "shusso_tosu": "出走頭数",
        "lap_time": "ラップタイム",
        "keito_mei": "系統名",
        "keito_setsumei": "系統説明",
        "seibetsu_code": "性別コード",
        "seinengappi": "生年月日",
        "menkyo_kofu_nengappi": "免許交付年月日",
        "tozai_shozoku_code": "東西所属コード",
        "chokyo_nengappi": "調教年月日",
        "time_gokei_4f": "合計タイム(4F)",
        "lap_time_1f": "ラスト1Fタイム",
        "mining_kubun": "マイニング区分",
        "yoso_soha_time": "予想走破タイム",
        "yoso_gosa_plus": "予想誤差(+)",
        "yoso_gosa_minus": "予想誤差(-)",
        "update_timestamp": "最終更新日時",
        "record_id": "レコードID",
        "data_kubun": "データ区分"
    }

    # Get all tables
    query_tables = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name;
    """
    
    try:
        tables_df = pd.read_sql(query_tables, engine)
        tables = tables_df['table_name'].tolist()
    except Exception as e:
        print(f"Error fetching tables: {e}")
        return

    print(f"Found {len(tables)} tables.")
    
    # Output file
    output_path = os.path.join(os.path.dirname(__file__), '../../docs/database_schema_complete_jp.md')
    output_path = os.path.abspath(output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# PC-KEIBA データベース全スキーマ仕様書 (日本語版)\n\n")
        f.write(f"作成日時: {pd.Timestamp.now()}\n")
        f.write(f"テーブル数: {len(tables)}\n\n")
        
        # Table of Contents
        f.write("## 目次\n")
        for t in tables:
            jp_name = table_map.get(t, t)
            f.write(f"- [{jp_name} ({t})](#{t})\n")
        f.write("\n---\n")
        
        # Details
        for t in tables:
            print(f"Processing {t}...")
            jp_name = table_map.get(t, t)
            f.write(f"## {t}\n")
            f.write(f"**和名**: {jp_name}\n\n")
            
            # Get Columns
            query_cols = text(f"""
            SELECT 
                column_name, 
                data_type, 
                character_maximum_length,
                is_nullable, 
                column_default
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = :tname
            ORDER BY ordinal_position;
            """)
            
            try:
                cols_df = pd.read_sql(query_cols, engine, params={"tname": t})
                
                # Format for display
                f.write("| カラム名 (英語) | 型 | Null容認 | デフォルト | 和名 (推定) |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- |\n")
                
                for _, row in cols_df.iterrows():
                    dtype = row['data_type']
                    if row['character_maximum_length'] and not pd.isna(row['character_maximum_length']):
                        dtype += f"({int(row['character_maximum_length'])})"
                    
                    cname = row['column_name']
                    desc = col_map.get(cname, "")
                    if desc == "":
                         # Partial match fallback
                         if 'odds' in cname: desc = "オッズ関連"
                         if 'haraimodoshi' in cname: desc = "払戻金"
                         if 'time' in cname: desc = "タイム関連"
                         if 'code' in cname: desc = "コード"
                         if 'name' in cname or 'mei' in cname: desc = "名称"
                    
                    f.write(f"| `{row['column_name']}` | {dtype} | {row['is_nullable']} | {row['column_default']} | {desc} |\n")
                
                f.write("\n")
                
                # Get Row Count
                try:
                    count_df = pd.read_sql(text(f"SELECT COUNT(*) as cnt FROM {t}"), engine)
                    cnt = count_df.iloc[0]['cnt']
                    f.write(f"**行数**: {cnt:,} 件\n\n")
                except:
                    f.write("**行数**: (取得エラー)\n\n")

                # Sample Data
                f.write("### サンプルデータ (先頭3件)\n")
                try:
                    sample_df = pd.read_sql(text(f"SELECT * FROM {t} LIMIT 3"), engine)
                    if not sample_df.empty:
                        f.write(sample_df.to_markdown(index=False))
                    else:
                        f.write("*データなし*")
                except Exception as e:
                     f.write(f"*取得エラー: {e}*")
                
                f.write("\n\n---\n")

            except Exception as e:
                f.write(f"カラム取得エラー {t}: {e}\n\n")
    
    print(f"Documentation generated at {output_path}")

if __name__ == "__main__":
    generate_schema_doc()
