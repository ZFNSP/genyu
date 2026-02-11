import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="中小型AIインフラ株 スクリーナー", layout="wide")

st.title("⚡ 次世代AIインフラ：中小型バリュー株 発掘システム")
st.markdown("時価総額の壁（流動性プレミアム）により市場から放置されている、インフラ・電線・空調関連のバリュー株を抽出します。")

# --- 1. ダミーデータの生成（実際はJ-Quants APIやCSVから読み込む部分） ---
@st.cache_data
def load_data():
    np.random.seed(42)
    # 500社のダミー企業データを生成
    tickers = [f"{np.random.randint(1000, 9999)}.T" for _ in range(500)]
    sectors = np.random.choice(['電気機器', '機械', '建設業', '非鉄金属', '情報・通信業', 'その他'], 500)
    
    # AIインフラ関連キーワードをランダムに付与
    keywords_pool = ['データセンター', '空調', '冷却', '変圧器', '電線', '半導体', '光ファイバー', 'なし']
    business_desc = [np.random.choice(keywords_pool) for _ in range(500)]
    
    df = pd.DataFrame({
        'Ticker': tickers,
        'Sector': sectors,
        'Business_Keyword': business_desc,
        'Market_Cap_Billion': np.random.uniform(50, 2000, 500), # 時価総額 (億円)
        'PER': np.random.uniform(5, 50, 500),                   # PER (倍)
        'PBR': np.random.uniform(0.3, 3.0, 500),                # PBR (倍)
        'OP_Growth_YoY': np.random.uniform(-20, 50, 500),       # 営業利益成長率 (%)
        'ROIC': np.random.uniform(0, 25, 500)                   # ROIC (%)
    })
    return df

df = load_data()

# --- 2. サイドバー：金融工学的なスクリーニング条件 ---
st.sidebar.header("🔍 スクリーニング条件")

# ① 流動性プレミアムの抽出（機関投資家が買えないサイズ）
max_market_cap = st.sidebar.slider("時価総額の上限 (億円)", min_value=100, max_value=2000, value=500, step=50)

# ② バリュートラップの回避（安かろう悪かろうを弾く）
max_per = st.sidebar.slider("PERの上限 (倍)", min_value=5, max_value=30, value=15, step=1)
min_op_growth = st.sidebar.slider("営業利益成長率の下限 (%)", min_value=-10, max_value=30, value=10, step=1)
min_roic = st.sidebar.slider("ROICの下限 (%) - 資本効率", min_value=0, max_value=20, value=8, step=1)

# ③ テーマ性（ナラティブ）のフィルタ
target_keywords = st.sidebar.multiselect(
    "事業内容キーワード (AIインフラ関連)",
    ['データセンター', '空調', '冷却', '変圧器', '電線', '半導体', '光ファイバー'],
    default=['空調', '冷却', '変圧器', '電線']
)

# --- 3. データフィルタリング実行 ---
filtered_df = df[
    (df['Market_Cap_Billion'] <= max_market_cap) &
    (df['PER'] <= max_per) &
    (df['OP_Growth_YoY'] >= min_op_growth) &
    (df['ROIC'] >= min_roic) &
    (df['Business_Keyword'].isin(target_keywords))
]

# --- 4. 結果表示と可視化 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"抽出銘柄: {len(filtered_df)} 社")
    st.dataframe(filtered_df.sort_values('PER').reset_index(drop=True), height=400)

with col2:
    st.subheader("📊 リスク・リターン構造（PER vs 成長率）")
    if not filtered_df.empty:
        # 散布図：横軸PER（割安度）、縦軸利益成長率（モメンタム）、バブルの大きさROIC（クオリティ）
        fig = px.scatter(
            filtered_df, x='PER', y='OP_Growth_YoY', 
            size='ROIC', color='Business_Keyword',
            hover_name='Ticker',
            labels={'OP_Growth_YoY': '営業利益成長率 (%)', 'PER': 'PER (倍)'},
            title="左上（低PER・高成長）の象限が「未発掘のアルファ」"
        )
        # 視覚的なセーフティゾーンをハイライト
        fig.add_vrect(x0=0, x1=15, fillcolor="green", opacity=0.1, line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("条件に一致する銘柄がありません。フィルタを緩めてください。")

# --- 5. クオンツ分析のインプリケーション ---
st.markdown("---")
st.markdown("""
### 💡 金融工学的な分析のポイント
* **散布図の「左上」を狙う：** PERが低く（左側）、かつ利益成長率が高い（上側）銘柄は、市場がまだAIインフラ特需によるEPSの改善を価格に織り込んでいない「ミスプライシング」の可能性が高いです。
* **バブルの大きさ（ROIC）：** 円が大きい銘柄ほど投下資本利益率が高く、無駄な設備投資をせずに効率よく現金を稼ぎ出している「質の高いバリュー株」です。
""")
