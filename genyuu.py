import streamlit as st
import requests
import pandas as pd
import zipfile
import io
import datetime
import traceback

# ==========================================
# 0. ページ設定 (必ず最初に書く)
# ==========================================
st.set_page_config(page_title="EDINET 受注残高スクリーナー", layout="wide")

# ライブラリのインポートチェック（bs4）
try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("ライブラリ 'beautifulsoup4' が見つかりません。requirements.txt を確認してください。")
    st.stop()

# ==========================================
# 1. 関数定義
# ==========================================
EDINET_API_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"

def get_document_list(date):
    """指定日の提出書類一覧を取得"""
    params = {'date': date, 'type': 2}
    try:
        res = requests.get(f"{EDINET_API_URL}/documents.json", params=params, timeout=10)
        if res.status_code != 200:
            return []
        data = res.json()
        return data.get('results', [])
    except Exception as e:
        st.error(f"API接続エラー: {e}")
        return []

def get_xbrl_soup(doc_id):
    """XBRLファイルをダウンロードしてBeautifulSoupオブジェクトを返す"""
    url = f"{EDINET_API_URL}/documents/{doc_id}"
    params = {'type': 1} # 1: XBRL
    try:
        res = requests.get(url, params=params, timeout=30)
        if res.status_code != 200:
            return None
        
        # ZIPファイルを展開してXBRLを探す
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            for filename in z.namelist():
                if 'PublicDoc' in filename and filename.endswith('.xbrl'):
                    with z.open(filename) as f:
                        # lxmlパーサーを試すが、ダメならhtml.parserで逃げる
                        try:
                            return BeautifulSoup(f, 'lxml-xml')
                        except Exception:
                            return BeautifulSoup(f, 'html.parser')
    except Exception:
        return None
    return None

def find_order_backlog(soup):
    """受注残高(OrderBacklog)タグを探す"""
    candidates = []
    if not soup:
        return []

    # タグ名に 'OrderBacklog' を含むものを検索
    # (名前空間プレフィックス対策で name 属性自体を文字列検索)
    tags = soup.find_all(lambda tag: tag.name and 'OrderBacklog' in tag.name)
    
    for tag in tags:
        context_id = tag.get('contextRef', '')
        value_str = tag.text.strip()
        
        # 数値変換できるか
        clean_val = value_str.replace(',', '').replace(' ', '')
        if clean_val.replace('-', '').isdigit():
             # Current(当期) や Instant(時点) のデータを優先
            if 'Current' in context_id or 'Instant' in context_id:
                try:
                    val = float(clean_val)
                    candidates.append({
                        'tag_name': tag.name,
                        'value': val,
                        'context': context_id
                    })
                except ValueError:
                    continue
    return candidates

# ==========================================
# 2. メインアプリ画面
# ==========================================
st.title("📊 半導体・建設株「受注残高」抽出アプリ")
st.info("EDINETから有価証券報告書・四半期報告書を取得し、受注残高 (Order Backlog) を抽出します。")

# サイドバー設定
st.sidebar.header("検索設定")
default_tickers = "6368, 6920, 8035, 1969, 6501"
ticker_input = st.sidebar.text_area("銘柄コード (カンマ区切り)", value=default_tickers)
# 日付：直近で確実にデータがある日(2024/11/14)を初期値に設定
target_date = st.sidebar.date_input("検索する日付", datetime.date(2024, 11, 14))
run_btn = st.sidebar.button("データを取得する")

# 実行処理
if run_btn:
    st.write(f"📅 **{target_date}** のデータを検索します...")
    
    # プログレスバー
    bar = st.progress(0)
    status_text = st.empty()

    # 1. 書類リスト取得
    docs = get_document_list(target_date)
    
    if not docs:
        st.warning("書類が見つかりませんでした。土日祝日ではないか確認してください。")
        st.stop()

    # 2. ターゲット銘柄のフィルタリング
    target_codes = [t.strip() for t in ticker_input.split(',')]
    # EDINETコードは末尾0がつく場合があるので前方一致で対応できるように辞書化
    target_docs = []
    
    for d in docs:
        sec_code = d.get('secCode')
        doc_desc = d.get('docDescription', '')
        
        if sec_code and doc_desc:
            # コードが一致 かつ 「四半期」or「有価証券」報告書
            if sec_code[:-1] in target_codes:
                if '四半期' in doc_desc or '有価証券' in doc_desc:
                    target_docs.append(d)

    if not target_docs:
        st.warning(f"指定された銘柄 ({ticker_input}) の報告書はこの日には見つかりませんでした。")
        st.stop()
    
    st.success(f"{len(target_docs)} 件の書類が見つかりました。解析を開始します。")

    # 3. 解析ループ
    results = []
    
    for i, doc in enumerate(target_docs):
        name = doc.get('filerName', '不明')
        code = doc.get('secCode', '')[:-1]
        
        status_text.text(f"解析中 ({i+1}/{len(target_docs)}): {name}")
        bar.progress((i + 1) / len(target_docs))
        
        try:
            soup = get_xbrl_soup(doc['docID'])
            if soup:
                backlogs = find_order_backlog(soup)
                if backlogs:
                    # 最大値を採用（連結合計の可能性が高い）
                    best = max(backlogs, key=lambda x: x['value'])
                    results.append({
                        '企業名': name,
                        'コード': code,
                        '受注残高(百万円)': best['value'] / 1_000_000,
                        '抽出タグ': best['tag_name']
                    })
                else:
                    results.append({
                        '企業名': name,
                        'コード': code,
                        '受注残高(百万円)': 0,
                        '抽出タグ': 'なし'
                    })
            else:
                 results.append({'企業名': name, 'コード': code, '受注残高(百万円)': 0, '抽出タグ': 'DL失敗'})
                 
        except Exception as e:
            # ここでエラーが起きても止まらないようにする
            st.error(f"{name} の処理中にエラーが発生しましたがスキップします: {e}")
            
    bar.empty()
    status_text.empty()

    # 4. 結果表示
    if results:
        df = pd.DataFrame(results)
        
        # 数値フォーマット
        st.subheader("抽出結果")
        st.dataframe(
            df.style.format({'受注残高(百万円)': '{:,.0f}'})
        )
        
        # グラフ（データがあるものだけ）
        valid_data = df[df['受注残高(百万円)'] > 0]
        if not valid_data.empty:
            st.bar_chart(data=valid_data, x='企業名', y='受注残高(百万円)')
        else:
            st.warning("受注残高の数値タグが見つかりませんでした。")
    else:
        st.warning("結果を取得できませんでした。")
