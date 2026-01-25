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
st.set_page_config(page_title="LPPL Precision Monitor", layout="wide")

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
st.title("📈 LPPL バブル崩壊予測＆精度評価")
st.markdown("モデルの予測値だけでなく、その**「測定精度（信頼区間）」**を物理的に評価します。")

if run_btn:
    with st.spinner(f"{ticker} のハミルトニアンを解析中..."):
        # --- データ取得 ---
        try:
            df = yf.download(ticker, start=start_date, progress=False)
            if df.empty:
                st.error("データ取得失敗。")
                st.stop()
            
            if isinstance(df.columns, pd.MultiIndex):
                data = df['Close'].iloc[:, 0]
            else:
                data = df['Close']
            
            data = data.dropna()
            prices = data.values.flatten()
            t_data = np.arange(len(prices))
            p_data = np.log(prices)
            
            if len(data) < 30:
                st.error("データ不足。期間を長くしてください。")
                st.stop()

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        # --- 最適化計算 ---
        current_t = t_data[-1]
        current_price = p_data[-1]
        
        initial_guess = [current_t + 40, 0.5, 9.0, current_price, -0.1, 0.05, 0]
        bounds = [
            (current_t + 1, current_t + 365),
            (m_min, m_max),
            (w_min, w_max),
            (None, None), (None, None), (None, None), (0, 2*np.pi)
        ]
        
        # 1. 最適化実行
        res = minimize(objective, initial_guess, args=(t_data, p_data), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 10000, 'ftol': 1e-9})
        
        if res.success:
            tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est = res.x
            
            # --- 精度指標の計算 (Physics Metrics) ---
            # 1. R^2 Score (決定係数)
            p_model = lppl_func(t_data, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            ss_res = np.sum((p_data - p_model) ** 2) # 残差変動
            ss_tot = np.sum((p_data - np.mean(p_data)) ** 2) # 全変動
            r_squared = 1 - (ss_res / ss_tot)
            
            # 2. X-Dayの不確定性 (Uncertainty Estimation)
            # 目的関数周りの数値微分で曲率を推定
            delta_t = 0.1
            obj_plus = objective([tc_est + delta_t, m_est, omega_est, A_est, B_est, C_est, phi_est], t_data, p_data)
            obj_minus = objective([tc_est - delta_t, m_est, omega_est, A_est, B_est, C_est, phi_est], t_data, p_data)
            obj_center = res.fun
            # 2階微分 (Hessian approx for tc)
            d2_obj = (obj_plus - 2*obj_center + obj_minus) / (delta_t**2)
            
            # 標準誤差 (Hessianの逆数に比例)
            # sigma^2 = 2 * sigma_resid^2 / curvature
            sigma_resid_sq = ss_res / (len(p_data) - 7) # 自由度調整済み分散
            if d2_obj > 1e-8:
                sigma_tc = np.sqrt(2 * sigma_resid_sq / d2_obj)
            else:
                sigma_tc = 99.9 # 計算不能（フラット）
            
            last_date = pd.to_datetime(data.index[-1])
            days_remaining = tc_est - current_t
            crash_date = last_date + timedelta(days=float(days_remaining))
            
            # --- UI表示 ---
            
            # 1. メイン予測パネル
            st.markdown("### 🗓️ 臨界点予測と不確定性")
            c1, c2, c3 = st.columns(3)
            
            c1.metric("推定 X-Day ($t_c$)", 
                      f"{crash_date.strftime('%Y-%m-%d')}",
                      delta=f"± {sigma_tc:.1f} 日 (誤差)", delta_color="off")
            
            c2.metric("残り時間", f"{days_remaining:.1f} 営業日")
            
            # 判定ロジック
            r2_color = "normal"
            if r_squared > 0.95: r2_color = "off" # Excellent
            elif r_squared < 0.8: r2_color = "inverse" # Bad
            
            c3.metric("決定係数 ($R^2$)", f"{r_squared:.4f}", 
                      delta="1.0に近いほど高精度", delta_color=r2_color)

            # 2. 物理パラメータ診断
            st.markdown("### 🧬 物理パラメータ & 測定精度")
            with st.container():
                col_m, col_w, col_acc = st.columns(3)
                
                # m
                with col_m:
                    st.info(f"**べき指数 $m = {m_est:.4f}$**")
                    if 0.1 < m_est < 0.9: st.success("✅ 正常 (加速中)")
                    else: st.warning("⚠️ 異常")
                
                # omega
                with col_w:
                    st.info(f"**振動数 $\omega = {omega_est:.4f}$**")
                    if 6.0 < omega_est < 13.0: st.success("✅ 正常 (周期性あり)")
                    else: st.warning("⚠️ 異常 (ノイズ)")

                # 精度評価
                with col_acc:
                    rmse = np.sqrt(ss_res / len(p_data))
                    st.info(f"**RMSE (平均誤差) = {rmse:.4f}**")
                    if r_squared > 0.90:
                        st.success("✅ フィッティング良好")
                    else:
                        st.error("⚠️ 信頼性低")

            # 3. グラフ
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            ax1.plot(data.index, p_data, label='Actual Log Price', color='blue', alpha=0.5)
            t_future = np.arange(0, tc_est, 0.1)
            p_future = lppl_func(t_future, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            date_future = [data.index[0] + timedelta(days=float(x)) for x in t_future]
            
            ax1.plot(date_future, p_future, label=f'LPPL Fit ($R^2$={r_squared:.2f})', color='red', linestyle='--')
            
            # X-Dayの誤差範囲を帯で表示
            if sigma_tc < 20:
                crash_date_min = last_date + timedelta(days=float(days_remaining - sigma_tc))
                crash_date_max = last_date + timedelta(days=float(days_remaining + sigma_tc))
                ax1.axvspan(crash_date_min, crash_date_max, color='green', alpha=0.1, label='Uncertainty Range')
            
            ax1.axvline(crash_date, color='green', linestyle=':', label='X-Day')
            ax1.set_ylabel("Log Price")
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--')
            ax1.set_title(f"LPPL Analysis (Confidence: {r_squared*100:.1f}%)")

            # 乖離
            residuals = p_data - p_model
            ax2.bar(data.index, residuals, color=['red' if r < 0 else 'green' for r in residuals], alpha=0.7)
            ax2.axhline(0, color='black')
            ax2.set_ylabel("Residuals")
            ax2.grid(True, alpha=0.5)
            
            st.pyplot(fig)

        else:
            st.error("最適化失敗。")
