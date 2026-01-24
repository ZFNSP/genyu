import streamlit as st
import requests
import pandas as pd
import zipfile
import io
from bs4 import BeautifulSoup
import datetime

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="EDINET 受注残高スクリーナー", layout="wide")
st.title("📊 半導体・建設株「受注残高」抽出アプリ")
st.markdown("EDINETから有報・四半期報告書を取得し、**受注残高 (Order Backlog)** を自動抽出します。")

# ==========================================
# ユーザー入力エリア（サイドバー）
# ==========================================
st.sidebar.header("検索設定")

# デフォルトの銘柄リスト
default_tickers = "6368, 6920, 8035, 1969, 6501"
ticker_input = st.sidebar.text_area("銘柄コード (カンマ区切り)", value=default_tickers)
target_tickers = [t.strip() for t in ticker_input.split(',')]

# 日付選択（デフォルトは直近の平日など）
target_date = st.sidebar.date_input("検索する日付", datetime.date(2025, 11, 14))

# 実行ボタン
run_btn = st.sidebar.button("データを取得する")

# ==========================================
# 関数定義
# ==========================================
EDINET_API_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"

def get_document_list(date):
    """指定日の提出書類一覧を取得"""
    params = {'date': date, 'type': 2}
    res = requests.get(f"{EDINET_API_URL}/documents.json", params=params)
    if res.status_code != 200:
        st.error(f"API Error: {res.status_code}")
        return []
    return res.json().get('results', [])

def get_xbrl_data(doc_id):
    """XBRLファイルをダウンロードして解析"""
    url = f"{EDINET_API_URL}/documents/{doc_id}"
    params = {'type': 1} # 1: XBRL
    res = requests.get(url, params=params)
    
    if res.status_code != 200:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            for filename in z.namelist():
                if 'PublicDoc' in filename and filename.endswith('.xbrl'):
                    with z.open(filename) as f:
                        return BeautifulSoup(f, 'lxml-xml')
    except zipfile.BadZipFile:
        return None
    return None

def find_order_backlog(soup):
    """受注残高に関連しそうなタグを探して値を返す"""
    candidates = []
    # タグ名に 'OrderBacklog' を含むものを全検索
    tags = soup.find_all(lambda tag: tag.name and 'OrderBacklog' in tag.name)
    
    for tag in tags:
        context_id = tag.get('contextRef')
        value = tag.text.strip()
        
        # 数値変換できるかチェック
        if value and value.replace(',', '').replace('-', '').isdigit():
            # Current(当期) や Instant(時点) のデータを優先
            if 'Current' in context_id or 'Instant' in context_id:
                try:
                    float_val = float(value)
                    candidates.append({
                        'tag_name': tag.name,
                        'value': float_val,
                        'context': context_id
                    })
                except ValueError:
                    continue
    return candidates

# ==========================================
# メイン処理
# ==========================================
if run_btn:
    st.info(f"📅 {target_date} に提出された書類を検索中...")
    
    # 1. 書類リスト取得
    docs = get_document_list(target_date)
    if not docs:
        st.warning("指定された日に書類は見つかりませんでした。")
        st.stop()
        
    # 2. ターゲット銘柄でフィルタリング
    target_docs = []
    # EDINETコードは末尾0がつくことが多いので調整してマッチング
    search_codes = [t + '0' for t in target_tickers] 
    
    for d in docs:
        if d['secCode'] in search_codes and d['docDescription']:
            if '四半期' in d['docDescription'] or '有価証券' in d['docDescription']:
                target_docs.append(d)
    
    if not target_docs:
        st.warning(f"指定された銘柄の書類はこの日({target_date})には提出されていません。")
        st.stop()
        
    st.write(f"該当書類: {len(target_docs)} 件")
    
    # 3. XBRL解析ループ
    results = []
    progress_bar = st.progress(0)
    
    for i, doc in enumerate(target_docs):
        ticker = doc['secCode'][:-1]
        name = doc['filerName']
        
        with st.spinner(f"解析中: {name} ({ticker})..."):
            soup = get_xbrl_data(doc['docID'])
            
            if soup:
                backlogs = find_order_backlog(soup)
                if backlogs:
                    # 最大値を採用（連結合計の可能性が高いため）
                    best_match = max(backlogs, key=lambda x: x['value'])
                    results.append({
                        'コード': ticker,
                        '企業名': name,
                        '受注残高(百万円)': f"{best_match['value']/1000000:,.0f}", # 百万円単位で整形
                        '生データ': best_match['value'],
                        '抽出タグ': best_match['tag_name']
                    })
                else:
                    results.append({
                        'コード': ticker,
                        '企業名': name,
                        '受注残高(百万円)': "取得失敗 (タグなし)",
                        '生データ': 0,
                        '抽出タグ': "-"
                    })
            else:
                st.error(f"{name}: XBRLダウンロード失敗")
                
        progress_bar.progress((i + 1) / len(target_docs))

    # 4. 結果表示
    if results:
        df = pd.DataFrame(results)
        st.success("解析完了！")
        st.dataframe(df)
        
        # グラフ化（取得できたものだけ）
        valid_df = df[df['生データ'] > 0].copy()
        if not valid_df.empty:
            st.bar_chart(valid_df.set_index('企業名')['生データ'])
    else:
        st.warning("データが抽出できませんでした。")
