import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

# ページ設定
st.set_page_config(page_title="原油価格分析 (Schwartz Model)", layout="centered")

class SchwartzOneFactor:
    def __init__(self, dt=1/252):
        self.dt = dt
        self.kappa = None
        self.alpha = None
        self.sigma = None
        self.stationary_std = None

    def fit(self, prices):
        log_prices = np.log(prices)
        x_t = log_prices[1:]
        x_t_minus_1 = log_prices[:-1]
        
        slope, intercept = np.polyfit(x_t_minus_1, x_t, 1)
        
        self.kappa = -np.log(slope) / self.dt
        self.alpha = intercept / (1 - slope)
        
        residuals = x_t - (slope * x_t_minus_1 + intercept)
        resid_std = np.std(residuals)
        
        self.sigma = resid_std * np.sqrt((2 * self.kappa) / (1 - slope**2))
        self.stationary_std = self.sigma / np.sqrt(2 * self.kappa)
        
        return {
            "Kappa (Speed)": self.kappa,
            "Alpha (Log Mean)": self.alpha,
            "Sigma (Vol)": self.sigma,
            "Half-Life (Days)": (np.log(2) / self.kappa) * 252
        }

    def calculate_z_score(self, current_price):
        if self.alpha is None: return 0.0
        log_price = np.log(current_price)
        return (log_price - self.alpha) / self.stationary_std

def main():
    st.title("🛢️ 原油先物 (CL=F) ミスプライス分析")
    st.markdown("Schwartzの1ファクターモデル（平均回帰）を用いた適正価格の推定")

    # サイドバー設定
    with st.sidebar:
        ticker = st.text_input("ティッカーシンボル", value="CL=F")
        lookback = st.slider("過去データ期間 (年)", 1, 5, 2)
        run_btn = st.button("分析実行")

    if run_btn:
        with st.spinner(f'{ticker} のデータを取得・計算中...'):
            try:
                # データ取得
                data = yf.download(ticker, period=f"{lookback}y", interval="1d", progress=False)
                
                if len(data) < 100:
                    st.error("エラー: データが不足しています。ティッカーを確認してください。")
                    return
                
                prices = data['Close'].values.flatten()
                last_date = data.index[-1].date()
                current_price = prices[-1]

                # モデル適用
                model = SchwartzOneFactor()
                params = model.fit(prices)
                z_score = model.calculate_z_score(current_price)
                mean_price = np.exp(params['Alpha (Log Mean)'])

                # --- 結果表示セクション ---
                
                # メイン指標の表示 (Metrics)
                col1, col2, col3 = st.columns(3)
                col1.metric("現在価格", f"${current_price:.2f}")
                col2.metric("理論適正価格 (長期)", f"${mean_price:.2f}", delta=f"{current_price - mean_price:.2f}")
                col3.metric("Zスコア (乖離度)", f"{z_score:.2f} σ")

                st.divider()

                # シグナル判定
                if z_score > 2.0:
                    st.error(f"### 📉 SELL SIGNAL (Overbought)\n現在価格は長期平均から +{z_score:.2f}σ 乖離しており、統計的に割高です。")
                elif z_score < -2.0:
                    st.success(f"### 📈 BUY SIGNAL (Oversold)\n現在価格は長期平均から {z_score:.2f}σ 乖離しており、統計的に割安です。")
                else:
                    st.info(f"### ⚖️ NEUTRAL\n現在価格は通常の変動範囲内（±2σ以内）です。")

                st.divider()

                # パラメータ詳細
                with st.expander("詳細なモデルパラメータを見る"):
                    st.write(f"**平均回帰速度 (Kappa):** {params['Kappa (Speed)']:.4f}")
                    st.write(f"**半減期 (Half-Life):** {params['Half-Life (Days)']:.1f} 日")
                    st.write(f"**ボラティリティ (Sigma):** {params['Sigma (Vol)']:.4f}")
                    st.caption("半減期が短いほど、価格が平均に戻る力が強いことを示します。")

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

    else:
        st.info("サイドバーの「分析実行」ボタンを押してください。")

if __name__ == "__main__":
    main()
