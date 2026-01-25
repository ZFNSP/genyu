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
st.set_page_config(page_title="LPPL Backtest Monitor", layout="wide")

def lppl_func(t, tc, m, omega, A, B, C, phi):
    """ Johansen-Ledoit-Sornette (JLS) Model """
    dt = tc - t
    dt[dt <= 0] = 1e-8 
    return A + np.power(dt, m) * (B + C * np.cos(omega * np.log(dt) + phi))

def objective(params, t, p_obs):
    tc, m, omega, A, B, C, phi = params
    p_est = lppl_func(t, tc, m, omega, A, B, C, phi)
    return np.sum((p_obs - p_est)**2)

# ==========================================
# 1. サイドバー（設定）
# ==========================================
st.sidebar.title("🎛️ バックテスト設定")

# 銘柄選択
INDEX_MAP = {
    "🇰🇷 KOSPI": "^KS11", "🇰🇷 KOSDAQ": "^KQ11",
    "🇯🇵 日経平均": "^N225", "🇺🇸 NASDAQ 100": "^NDX",
    "🇺🇸 SOX指数": "^SOX", "🇺🇸 S&P 500": "^GSPC",
    "🇺🇸 ダウ平均": "^DJI", "🪙 ビットコイン": "BTC-USD",
    "🪙 イーサリアム": "ETH-USD", "🥈 銀 (Silver)": "SI=F", "🥇 金 (Gold)": "GC=F"
}
input_mode = st.sidebar.radio("モード", ["リスト選択", "手動入力"])
if input_mode == "リスト選択":
    selected = st.sidebar.selectbox("銘柄", list(INDEX_MAP.keys()))
    ticker = INDEX_MAP[selected]
else:
    ticker = st.sidebar.text_input("銘柄コード", value="SI=F")

st.sidebar.markdown("---")
# 日付設定（ここが重要）
start_date = st.sidebar.date_input("解析開始 (Start)", datetime(2025, 1, 1))
# デフォルトを「昨日」に設定
end_date = st.sidebar.date_input("解析終了 (End / 訓練データ打切日)", datetime.now() - timedelta(days=1))

st.sidebar.info(f"""
💡 **ヒント:** 「解析終了」を**1ヶ月前**の日付にしてみてください。
そこから今日までの「予測線(赤)」と「実際(青)」が重なれば、そのモデルは信頼できます。
""")

# パラメータ設定
st.sidebar.subheader("物理制約条件")
m_min, m_max = st.sidebar.slider("べき指数 m", 0.1, 1.0, (0.1, 0.9))
w_min, w_max = st.sidebar.slider("振動数 ω", 4.0, 20.0, (6.0, 13.0))

run_btn = st.sidebar.button("検証実行 (Run Test)")

# ==========================================
# 2. メイン処理
# ==========================================
st.title("🛡️ LPPL 未来予測検証システム")

if run_btn:
    with st.spinner(f"{ticker} の検証シミュレーション中..."):
        try:
            # データは「今日」まで全部取る（答え合わせ用）
            full_df = yf.download(ticker, start=start_date, progress=False)
            if full_df.empty:
                st.error("データ取得失敗")
                st.stop()
            
            # Closeデータの抽出
            if isinstance(full_df.columns, pd.MultiIndex):
                full_data = full_df['Close'].iloc[:, 0]
            else:
                full_data = full_df['Close']
            full_data = full_data.dropna()

            # --- データの分割 (Train / Test Split) ---
            # 指定された「終了日」までのデータを訓練用とする
            train_mask = full_data.index <= pd.Timestamp(end_date)
            train_data = full_data[train_mask]
            test_data = full_data[~train_mask] # 終了日以降（答え合わせ用）

            if len(train_data) < 30:
                st.error("訓練データが少なすぎます。終了日をもっと未来にするか、開始日を過去にしてください。")
                st.stop()

            # 最適化用データ作成（訓練データのみ使用！）
            prices_train = train_data.values.flatten()
            t_train = np.arange(len(prices_train))
            p_train = np.log(prices_train)
            
            last_train_t = t_train[-1]
            last_train_price = p_train[-1]

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        # --- 最適化計算 (訓練データのみにフィット) ---
        initial_guess = [last_train_t + 40, 0.5, 9.0, last_train_price, -0.1, 0.05, 0]
        bounds = [
            (last_train_t + 1, last_train_t + 365),
            (m_min, m_max), (w_min, w_max),
            (None, None), (None, None), (None, None), (0, 2*np.pi)
        ]
        
        res = minimize(objective, initial_guess, args=(t_train, p_train), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 10000, 'ftol': 1e-9})
        
        if res.success:
            tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est = res.x
            
            # 日付計算
            last_train_date = train_data.index[-1]
            days_remaining = tc_est - last_train_t
            crash_date = last_train_date + timedelta(days=float(days_remaining))
            
            # 精度計算 (R^2 on Train)
            p_model_train = lppl_func(t_train, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            ss_res = np.sum((p_train - p_model_train) ** 2)
            ss_tot = np.sum((p_train - np.mean(p_train)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # --- 結果表示 ---
            st.markdown(f"### 🗓️ 解析基準日: {last_train_date.strftime('%Y-%m-%d')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("推定 X-Day", crash_date.strftime('%Y-%m-%d'))
            c2.metric("その時点での残り時間", f"{days_remaining:.1f} 日")
            c3.metric("訓練データ適合度 ($R^2$)", f"{r_squared:.4f}")
            
            # --- グラフ描画（ここがハイライト） ---
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # 1. 訓練データ（実線・青）
            ax.plot(train_data.index, np.log(train_data.values), label='Training Data (Used for fit)', color='blue', linewidth=1.5)
            
            # 2. テストデータ（実線・オレンジ） -> ここが「市場の反応」
            if not test_data.empty:
                ax.plot(test_data.index, np.log(test_data.values), label='Actual Market Reaction (Unseen)', color='orange', linewidth=2)
            
            # 3. LPPL予測線（点線・赤）
            # 時間軸を未来まで拡張
            t_future_len = int(days_remaining + 20) # X-Dayの少し先まで
            t_full = np.arange(0, last_train_t + t_future_len, 0.1)
            p_full = lppl_func(t_full, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
            
            # t=0 に対応する日付
            date_start = train_data.index[0]
            date_full = [date_start + timedelta(days=float(x)) for x in t_full]
            
            ax.plot(date_full, p_full, label='LPPL Projection', color='red', linestyle='--', alpha=0.8)
            
            # 境界線とX-Day
            ax.axvline(last_train_date, color='black', linestyle='-', alpha=0.5, label='Analysis End Date')
            ax.axvline(crash_date, color='green', linestyle=':', label='Predicted X-Day')
            
            ax.set_title(f"Backtest: Fit up to {last_train_date.strftime('%Y-%m-%d')} vs Actual Reality")
            ax.set_ylabel("Log Price")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            
            # --- 物理的考察 ---
            st.markdown("### 👨‍🔬 物理実験の評価")
            if not test_data.empty:
                # 予測誤差の検証
                # テスト期間の初日と最終日の誤差を見る
                p_test_log = np.log(test_data.values)
                # モデルの予測値を取得するためのインデックス換算が複雑なため、簡易的に最後の値と比較
                
                st.info("""
                **グラフの見方:**
                * **<span style="color:blue">青い線</span>**: AIに学習させた過去のデータ。
                * **<span style="color:red">赤い点線</span>**: その時点でのAIの予言。
                * **<span style="color:orange">オレンジの線</span>**: **解析日以降に市場が実際にどう動いたか（答え）。**
                
                **判定:**
                * オレンジの線が赤い点線に沿っていれば、**「市場は物理法則通りに動いている（予測成功）」**。
                * オレンジが赤から大きく外れていれば、**「予測時点以降に市場の構造が変わった（外れ）」**。
                """, unsafe_allow_html=True)
            else:
                st.warning("解析終了日が今日のため、答え合わせ（オレンジの線）は表示されません。過去の日付を設定すると検証できます。")

        else:
            st.error("最適化失敗。")
