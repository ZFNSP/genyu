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
st.set_page_config(page_title="LPPL Bubble Monitor Pro", layout="wide")

def lppl_func(t, tc, m, omega, A, B, C, phi):
    """
    Johansen-Ledoit-Sornette (JLS) Model
    """
    dt = tc - t
    dt[dt <= 0] = 1e-8 
    return A + np.power(dt, m) * (B + C * np.cos(omega * np.log(dt) + phi))

def objective(params, t, p_obs):
    tc, m, omega, A, B, C, phi = params
    p_est = lppl_func(t, tc, m, omega, A, B, C, phi)
    return np.sum((p_obs - p_est)**2)

# ==========================================
# 1. サイドバー（実験パラメータ設定）
# ==========================================
st.sidebar.title("🎛️ 観測パラメータ設定")
ticker = st.sidebar.text_input("銘柄コード", value="^KS11")
start_date = st.sidebar.date_input("解析開始日 (Start Date)", datetime(2025, 1, 1))

st.sidebar.markdown("---")
st.sidebar.subheader("物理制約条件 (Bounds)")
m_min, m_max = st.sidebar.slider("べき指数 m の範囲", 0.1, 1.0, (0.1, 0.9))
w_min, w_max = st.sidebar.slider("対数角周波数 ω の範囲", 4.0, 20.0, (6.0, 13.0))

run_btn = st.sidebar.button("解析実行 (Run LPPL)")

# ==========================================
# 2. メイン処理
# ==========================================
st.title("📈 KOSPI 臨界点 & 乖離モニタリング")
st.markdown("LPPLモデルによる崩壊予測と、**「市場の減速（理論からの乖離）」**をリアルタイムで監視します。")

if run_btn:
    with st.spinner(f"{ticker} のデータを解析中..."):
        # --- データ取得 ---
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if df.empty:
                st.error("データ取得失敗。")
                st.stop()
            
            # データ整形
            if isinstance(df.columns, pd.MultiIndex):
                data = df['Close'].iloc[:, 0]
            else:
                data = df['Close']
            
            data = data.dropna()
            prices = data.values.flatten()
            t_data = np.arange(len(prices))
            p_data = np.log(prices)
            
            if len(data) < 30:
                st.error("データ不足です。開始日を過去にしてください。")
                st.stop()

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        # --- 最適化計算 ---
        current_t = t_data[-1]
        current_price = p_data[-1]
        
        # 初期値 & 制約
        initial_guess = [current_t + 40, 0.5, 9.0, current_price, -0.1, 0.05, 0]
        bounds = [
            (current_t + 1, current_t + 365),
            (m_min, m_max),
            (w_min, w_max),
            (None, None), (None, None), (None, None), (0, 2*np.pi)
        ]
        
        res = minimize(objective, initial_guess, args=(t_data, p_data), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 10000, 'ftol': 1e-9})
        
        # --- 結果表示 ---
        if res.success:
            tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est = res.x
            
            last_date = pd.to_datetime(data.index[-1])
            days_remaining = tc_est - current_t
            crash_date = last_date + timedelta(days=float(days_remaining))
            
            # === 新機能: 乖離（Residual）の計算 ===
            # 全期間の理論値を計算
            p_model_full = lppl_func(t_data, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            residuals = p_data - p_model_full # 実測 - 理論
            
            # 直近5日間の乖離傾向をチェック
            recent_resid = residuals[-5:]
            is_damped = np.all(recent_resid < 0) # 5日連続で理論値を下回っているか
            resid_trend = recent_resid[-1] - recent_resid[0] # 乖離が拡大しているか(マイナスなら悪化)

            # 1. ステータスパネル
            c1, c2, c3 = st.columns(3)
            c1.metric("推定 X-Day (tc)", crash_date.strftime('%Y-%m-%d'))
            c2.metric("残り時間", f"{days_remaining:.1f} 営業日")
            
            # 乖離アラート機能
            if is_damped and resid_trend < -0.01:
                c3.error("📉 減速検知 (Damping)")
                st.toast("警告: 実測値が理論値を連続して下回っています。バブルの勢いが削がれています。", icon="⚠️")
            elif days_remaining < 20:
                c3.warning("🔥 加速中 (Critical)")
            else:
                c3.success("🚀 上昇トレンド維持")

            # 2. メイングラフ（価格）
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # 上段: 価格チャート
            ax1.plot(data.index, p_data, label='Actual Log Price', color='blue', alpha=0.6)
            
            t_future = np.arange(0, tc_est, 0.1)
            p_future = lppl_func(t_future, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            date_future = [data.index[0] + timedelta(days=float(x)) for x in t_future]
            
            ax1.plot(date_future, p_future, label='LPPL Model', color='red', linestyle='--')
            ax1.axvline(crash_date, color='green', linestyle=':', label='X-Day')
            ax1.set_ylabel("Log Price")
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--')
            ax1.set_title(f"LPPL Analysis (m={m_est:.3f}, $\omega$={omega_est:.3f})")

            # 下段: 乖離チャート（ここが重要）
            ax2.bar(data.index, residuals, color=['red' if r < 0 else 'green' for r in residuals], alpha=0.7, width=1.0)
            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.set_ylabel("Deviation (Residual)")
            ax2.set_title("Deviation from Theory (Green=Stronger, Red=Weaker)")
            ax2.grid(True, alpha=0.5)
            
            st.pyplot(fig)
            
            st.markdown("""
            ### 🔍 乖離モニターの見方
            * **下のグラフ（棒グラフ）**を見てください。
            * **<span style="color:green">緑のバー</span>**: 実測値が理論より強く、バブルのエネルギーが充填されています。
            * **<span style="color:red">赤のバー</span>**: 実測値が理論を下回っています。これが**深く、長く続くと「市場が気づいて減速した」合図**です。
            """, unsafe_allow_html=True)

        else:
            st.error("最適化失敗。パラメータ範囲を調整してください。")
