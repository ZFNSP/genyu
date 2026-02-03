import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# 1. データの取得 (期間: 直近6ヶ月)
tickers = ["QQQ", "IWM", "VUG", "VTV", "RSP", "SPY"]
# auto_adjust=False を追加して、'Adj Close' 列を確実に取得する
data = yf.download(tickers, period="6mo", auto_adjust=False)['Adj Close']

# 2. レシオの計算
# 開始日を「1.0」として正規化（リベース）することで、変化率を比較しやすくします
df_ratios = pd.DataFrame()
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
    ax.text(df_ratios.index[-1], last_val, f'{last_val:.3f}', 
            fontsize=12, verticalalignment='bottom', fontweight='bold')

plt.xlabel("Date")
plt.tight_layout()
plt.show()
