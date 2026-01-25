import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 0. ページ設定 & 物理モデル定義
# ==========================================
st.set_page_config(page_title="LPPL Bubble Monitor", layout="wide")

def lppl_func(t, tc, m, omega, A, B, C, phi):
    """
    Johansen-Ledoit-Sornette (JLS) Model
    ln p(t) = A + B(tc - t)^m + C(tc - t)^m * cos(omega * ln(tc - t) + phi)
    """
    dt = tc - t
    dt[dt <= 0] = 1e-8 # 特異点での発散を防ぐための微小項
    return A + np.power(dt, m) * (B + C * np.cos(omega * np.log(dt) + phi))

def objective(params, t, p_obs):
    tc, m, omega, A, B, C, phi = params
    p_est = lppl_func(t, tc, m, omega, A, B, C, phi)
    return np.sum((p_obs - p_est)**2)

# ==========================================
# 1. サイドバー（実験パラメータ設定）
# ==========================================
st.sidebar.title("🎛️ 観測パラメータ設定")

ticker = st.sidebar.text_input("銘柄コード (Yahoo)", value="^KS11")

# バブルの「開始点」を調整するスライダー
# デフォルトを「2025-01-01」に設定（先ほどの成功体験に基づく）
start_date = st.sidebar.date_input("解析開始日 (Start Date)", datetime(2025, 1, 1))

st.sidebar.markdown("---")
st.sidebar.subheader("物理制約条件 (Bounds)")
m_min, m_max = st.sidebar.slider("べき指数 m の範囲", 0.1, 1.0, (0.1, 0.9))
w_min, w_max = st.sidebar.slider("対数角周波数 ω の範囲", 4.0, 20.0, (6.0, 13.0))

run_btn = st.sidebar.button("解析実行 (Run LPPL)")

# ==========================================
# 2. メイン処理
# ==========================================
st.title("📈 KOSPI 臨界点モニタリングシステム")
st.markdown("対数周期べき乗則 (LPPL) を用いて、金融バブルの崩壊時刻 $t_c$ を物理的に推定します。")

if run_btn:
    with st.spinner(f"{ticker} のデータを取得し、非線形最適化を行っています..."):
        # --- データ取得 ---
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if df.empty:
                st.error("データが取得できませんでした。日付や銘柄コードを確認してください。")
                st.stop()
            
            # データ整形（1次元化）
            if isinstance(df.columns, pd.MultiIndex):
                data = df['Close'].iloc[:, 0]
            else:
                data = df['Close']
            
            data = data.dropna()
            prices = data.values.flatten()
            t_data = np.arange(len(prices))
            p_data = np.log(prices)
            
            if len(data) < 30:
                st.error("データ点数が少なすぎます。開始日をもっと過去にしてください。")
                st.stop()

        except Exception as e:
            st.error(f"データ取得エラー: {e}")
            st.stop()

        # --- 最適化計算 ---
        current_t = t_data[-1]
        current_price = p_data[-1]
        
        # 初期値推定
        initial_guess = [
            current_t + 40,   # tc
            0.5,              # m
            9.0,              # omega
            current_price,    # A
            -0.1,             # B
            0.05,             # C
            0                 # phi
        ]
        
        # 制約条件
        bounds = [
            (current_t + 1, current_t + 365), # tc
            (m_min, m_max),  # m
            (w_min, w_max),  # omega
            (None, None), (None, None), (None, None), (0, 2*np.pi)
        ]
        
        # ソルバー実行
        res = minimize(objective, initial_guess, args=(t_data, p_data), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 10000, 'ftol': 1e-9})
        
        # --- 結果表示 ---
        if res.success:
            tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est = res.x
            
            # 日付変換
            last_date = pd.to_datetime(data.index[-1])
            days_remaining = tc_est - current_t
            crash_date = last_date + timedelta(days=float(days_remaining))
            
            # 1. 重要指標のカード表示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("推定 X-Day (tc)", crash_date.strftime('%Y-%m-%d'))
            with col2:
                st.metric("残り時間", f"{days_remaining:.1f} 営業日")
            with col3:
                # 誤差を表示（小さいほど信頼度高）
                rmse = np.sqrt(res.fun / len(p_data))
                st.metric("モデル適合誤差 (RMSE)", f"{rmse:.4f}")

            # 2. 物理パラメータの健全性チェック
            st.markdown("### 🧬 物理パラメータ診断")
            c1, c2 = st.columns(2)
            
            # m の評価
            m_status = "✅ 正常 (加速中)" if 0.1 < m_est < 0.9 else "⚠️ 異常 (フィッティング失敗の可能性)"
            c1.info(f"**べき指数 m = {m_est:.4f}**\n\n{m_status}")
            
            # omega の評価
            w_status = "✅ 正常 (対数周期振動)" if 6.0 < omega_est < 13.0 else "⚠️ 異常 (ランダムウォーク)"
            c2.info(f"**対数角周波数 $\omega$ = {omega_est:.4f}**\n\n{w_status}")

            # 3. グラフ描画
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 実測値
            ax.plot(data.index, p_data, label='Actual Log Price', color='blue', alpha=0.6)
            
            # 未来予測
            t_future = np.arange(0, tc_est, 0.1)
            p_future = lppl_func(t_future, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            date_future = [data.index[0] + timedelta(days=float(x)) for x in t_future]
            
            ax.plot(date_future, p_future, label='LPPL Model Fit', color='red', linestyle='--')
            ax.axvline(crash_date, color='green', linestyle=':', label='Critical Time')
            
            ax.set_title(f"LPPL Analysis for {ticker}", fontsize=14)
            ax.set_ylabel("Log Price")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)
            
            # 4. 投資アクションの提案
            st.subheader("🛡️ 推奨アクション")
            if days_remaining < 10:
                st.error("🚨 **DANGER ZONE**: 崩壊までの時間がわずかです。ポジションの解消を強く推奨します。")
            elif days_remaining < 30:
                st.warning("⚠️ **CAUTION**: X-Dayが近づいています。ストップロスを引き上げ、警戒してください。")
            else:
                st.success("🟢 **HOLD**: バブルは継続中です。利益を伸ばすフェーズです。")

        else:
            st.error("最適化に失敗しました。開始日を変更するか、制約条件を緩めてください。")
            st.write(res.message)

else:
    st.info("サイドバーの「解析実行」ボタンを押して、最新のX-Dayを計算してください。")
