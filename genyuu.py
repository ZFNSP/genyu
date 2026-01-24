def get_document_list(date):
    """指定日の提出書類一覧を取得（ブロック回避版）"""
    params = {'date': date, 'type': 2}
    # 【追加】ブラウザのふりをするためのヘッダー
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        # verify=False はSSLエラー回避用（警告が出ますが無視してOKです）
        res = requests.get(f"{EDINET_API_URL}/documents.json", params=params, headers=headers, timeout=10, verify=False)
        
        if res.status_code != 200:
            return []
        
        data = res.json()
        return data.get('results', [])
    except Exception as e:
        st.error(f"API接続エラー: {e}")
        return []
