import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

st.title("🇺🇸 US Market Rotation Analysis")

# 1. データの取得 (期間: 直近6ヶ月)
tickers = ["QQQ", "IWM", "VUG", "VTV", "RSP", "SPY"]

# 【修正点】auto_adjust=False を指定して 'Adj Close' を確実に取得しようと試みる
raw_data = yf.download(tickers, period="6mo", auto_adjust=False)

# 【修正点】データ構造の確認と安全な抽出
# マルチインデックスの場合や、カラム名が異なる場合に対応
if 'Adj Close' in raw_data.columns:
    data = raw_data['Adj Close']
elif 'Close' in raw_data.columns:
    data = raw_data['Close']
else:
    # yfinanceのバージョンによっては構造が違う場合があるため、単純化して取得
    # auto_adjust=Trueで再取得してCloseを使う
    raw_data = yf.download(tickers, period="6mo", auto_adjust=True)
    data = raw_data['Close']

# データが空でないか確認
if data.empty:
    st.error("データの取得に失敗しました。しばらく待ってから再読み込みしてください。")
else:
    # 2. レシオの計算
    # 開始日を「1.0」として正規化（リベース）することで、変化率を比較しやすくします
    df_ratios = pd.DataFrame()
    
    # ゼロ除算回避のため、データが存在することを確認
    try:
        df_ratios['Size (Tech vs Small)'] = (data['QQQ'] / data['IWM']) / (data['QQQ'].iloc[0] / data['IWM'].iloc[0])
        df_ratios['Style (Growth vs Value)'] = (data['VUG'] / data['VTV']) / (data['VUG'].iloc[0] / data['VTV'].iloc[0])
        df_ratios['Breadth (Equal vs Cap)'] = (data['RSP'] / data['SPY']) / (data['RSP'].iloc[0] / data['SPY'].iloc[0])

        # 3. 画像の描画 (3つのサブプロット)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # 色の設定
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 青、オレンジ、緑
        titles = [
            "1. Size Rotation: QQQ / IWM (Down = Small Cap Outperformance)",
            "2. Style Rotation: VUG / VTV (Down = Value Outperformance)",
            "3. Market Breadth: RSP / SPY (Up = Broadening Rally)"
        ]

        for i, col in enumerate(df_ratios.columns):
            ax = axes[i]
            ax.plot(df_ratios.index, df_ratios[col], label=col, color=colors[i], linewidth=2)
            
            # トレンドを見やすくするための25日移動平均線
            sma = df_ratios[col].rolling(window=25).mean()
            ax.plot(df_ratios.index, sma, label="25-day Trend", color='gray', linestyle='--', alpha=0.7)
            
            ax.set_title(titles[i], fontsize=14, fontweight='bold')
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            
            # 最新値の表示
            last_val = df_ratios[col].iloc[-1]
            last_date = df_ratios.index[-1]
            ax.text(last_date, last_val, f'{last_val:.3f}', 
                    fontsize=12, verticalalignment='bottom', fontweight='bold')

        plt.xlabel("Date")
        plt.tight_layout()
        
        # Streamlitで表示
        st.pyplot(fig)

    except Exception as e:
        st.error(f"計算または描画中にエラーが発生しました: {e}")
