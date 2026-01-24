import requests
import pandas as pd
import zipfile
import io
from bs4 import BeautifulSoup
import datetime

# ==========================================
# 設定エリア
# ==========================================
# 検索したい銘柄コード（証券コード）のリスト
TARGET_TICKERS = ['6368', '6920', '8035', '1969']  # 例: オルガノ, レーザーテック, TEL, 高砂熱学
# 検索する日付（直近の平日を指定してください）
TARGET_DATE = '2025-11-14' # ※ここを適宜変更してください（決算発表が多い日推奨）
# ==========================================

EDINET_API_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"

def get_document_list(date):
    """指定日の提出書類一覧を取得"""
    params = {'date': date, 'type': 2}
    # ※APIキー（Subscription-Key）がある場合はheaderに追加推奨
    res = requests.get(f"{EDINET_API_URL}/documents.json", params=params)
    if res.status_code != 200:
        print(f"Error: {res.status_code}")
        return []
    return res.json().get('results', [])

def get_xbrl_data(doc_id):
    """XBRLファイルをダウンロードして解析"""
    url = f"{EDINET_API_URL}/documents/{doc_id}"
    params = {'type': 1} # 1: XBRL
    res = requests.get(url, params=params)
    
    if res.status_code != 200:
        return None

    # Zipを展開してXBRLファイルを探す
    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        for filename in z.namelist():
            # "PublicDoc" フォルダ内の .xbrl ファイルが財務諸表本体
            if 'PublicDoc' in filename and filename.endswith('.xbrl'):
                with z.open(filename) as f:
                    return BeautifulSoup(f, 'lxml-xml')
    return None

def find_order_backlog(soup):
    """受注残高に関連しそうなタグを探して値を返す"""
    # 企業によってタグ名が微妙に異なるため、部分一致で探索
    # 標準的なタグ: jpcrp_cor:OrderBacklog...
    
    candidates = []
    
    # 全タグから 'OrderBacklog' (受注残高) を含むものを検索
    tags = soup.find_all(lambda tag: tag.name and 'OrderBacklog' in tag.name)
    
    for tag in tags:
        # コンテキスト（CurrentYear/Instantなど）を確認
        context_id = tag.get('contextRef')
        value = tag.text.strip()
        
        # 数値が入っているものだけ抽出（空文字やテキストブロックを除く）
        if value and value.replace(',', '').replace('-', '').isdigit():
            # contextRefに 'Current' や 'Instant' が含まれるものを優先（今年度期末）
            if 'Current' in context_id or 'Instant' in context_id:
                candidates.append({
                    'tag_name': tag.name,
                    'value': float(value),
                    'context': context_id
                })
    
    return candidates

def main():
    print(f"Searching documents for {TARGET_DATE}...")
    docs = get_document_list(TARGET_DATE)
    
    # ターゲット銘柄の書類だけフィルタリング（四半期報告書 or 有価証券報告書）
    target_docs = [
        d for d in docs 
        if d['secCode'] in [t + '0' for t in TARGET_TICKERS] # APIは5桁(末尾0)で返ってくることが多い
        and d['docDescription'] is not None
        and ('四半期' in d['docDescription'] or '有価証券' in d['docDescription'])
    ]
    
    print(f"Found {len(target_docs)} relevant documents.")
    
    results = []
    
    for doc in target_docs:
        ticker = doc['secCode'][:-1]
        name = doc['filerName']
        print(f"Processing: {name} ({ticker})...")
        
        soup = get_xbrl_data(doc['docID'])
        if not soup:
            print(" -> Failed to download XBRL.")
            continue
            
        backlogs = find_order_backlog(soup)
        
        if backlogs:
            # 複数のタグが見つかった場合、最も数値が大きいもの（連結合計の可能性が高い）を採用する簡易ロジック
            # 厳密にはContextRef解析が必要だが、スクリーニング用途ならこれで十分機能する
            best_match = max(backlogs, key=lambda x: x['value'])
            results.append({
                'Ticker': ticker,
                'Name': name,
                'OrderBacklog (Million Yen?)': best_match['value'] / 1000000, # 単位調整（仮）
                'Raw Value': best_match['value'],
                'Tag Found': best_match['tag_name']
            })
            print(f" -> Found Backlog: {best_match['value']}")
        else:
            print(" -> Order Backlog tag NOT found (Look for 'Inventories' instead?).")

    # 結果表示
    if results:
        df = pd.DataFrame(results)
        print("\n=== Screening Results ===")
        print(df)
    else:
        print("\nNo backlog data found.")

if __name__ == "__main__":
    main()
