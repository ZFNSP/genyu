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
st.set_page_config(page_title="LPPL Advanced Monitor", layout="wide")

# ==========================================
# 0. ページ設定 & 関数定義 (既存の場所に追加)
# ==========================================

# ▼▼▼▼▼ ハースト指数計算関数 (追加) ▼▼▼▼▼
def calculate_hurst(ts, max_lag=20):
    """
    R/S分析によるハースト指数の簡易計算
    ts: 価格データ (1次元配列)
    """
    try:
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0 # 簡易的なHurst推定
    except:
        return 0.5 # エラー時はランダム(0.5)を返す
# ▲▲▲▲▲ ここまで ▲▲▲▲▲

# ... (中略: メイン処理の中) ...

        # ==========================================
        # データ取得後の処理エリア
        # ==========================================
        
        # ▼▼▼▼▼ 計算と表示 (メイン処理内に追加) ▼▼▼▼▼
        # 価格データのログをとって計算
        h_val = calculate_hurst(np.log(full_data.values), max_lag=min(100, len(full_data)//2))
        
        # 表示用カラムを作成（既存のmetricエリアに追加推奨）
        st.markdown("### 🌊 モメンタム持続性診断 (Hurst Exponent)")
        c_h1, c_h2 = st.columns([1, 3])
        
        with c_h1:
            st.metric("ハースト指数 (H)", f"{h_val:.4f}")
        
        with c_h2:
            if h_val > 0.65:
                st.success(f"🚀 **強烈なトレンド持続中 (H > 0.65)**\n\n慣性の法則が強く働いています。「下がるまで持ち続ける」が正解です。")
            elif h_val > 0.5:
                st.info(f"📈 **緩やかなトレンド (0.5 < H < 0.65)**\n\n上昇傾向ですが、ノイズも混じっています。")
            elif h_val < 0.4:
                st.warning(f"📉 **平均回帰性が強い (H < 0.4)**\n\n上がれば叩かれる「ジグザグ相場」です。深追いは禁物。")
            else:
                st.warning(f"🎲 **ランダムウォーク (H ≈ 0.5)**\n\n方向感がありません。")
        # ▲▲▲▲▲ ここまで ▲▲▲▲▲

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
st.sidebar.title("🎛️ LPPL 解析ラボ")

# --- 銘柄選択 ---
INDEX_MAP = {
    "🇰🇷 KOSPI": "^KS11", "🇰🇷 KOSDAQ": "^KQ11",
    "🇯🇵 日経平均": "^N225", "🇺🇸 NASDAQ 100": "^NDX",
    "🇺🇸 SOX指数": "^SOX", "🇺🇸 S&P 500": "^GSPC",
    "🇺🇸 ダウ平均": "^DJI", "🪙 ビットコイン": "BTC-USD",
    "🪙 イーサリアム": "ETH-USD", "🥈 銀 (Silver)": "SI=F", "🥇 金 (Gold)": "GC=F"
}
input_mode = st.sidebar.radio("モード選択", ["リストから選ぶ", "コード手動入力"])

if input_mode == "リストから選ぶ":
    selected = st.sidebar.selectbox("対象銘柄", list(INDEX_MAP.keys()))
    ticker = INDEX_MAP[selected]
else:
    ticker = st.sidebar.text_input("銘柄コード (Yahoo)", value="SI=F")

st.sidebar.markdown("---")

# --- 日付設定 ---
st.sidebar.subheader("📅 解析期間")
start_date = st.sidebar.date_input("開始日 (Start)", datetime(2025, 1, 1))
end_date = st.sidebar.date_input("終了日 (End / 検証用)", datetime.now() - timedelta(days=1))

# --- パラメータ制約 ---
st.sidebar.subheader("🧬 物理パラメータ制約")
m_min, m_max = st.sidebar.slider("べき指数 m", 0.1, 1.0, (0.1, 0.9))
w_min, w_max = st.sidebar.slider("振動数 ω", 4.0, 20.0, (6.0, 13.0))

# --- 実行モード ---
analysis_type = st.sidebar.radio("実行タイプ", ["通常解析 (バックテスト)", "堅牢性検証 (Sensitivity Check)"])
run_btn = st.sidebar.button("解析開始")

# ==========================================
# 2. メイン処理
# ==========================================
st.title(f"🛡️ LPPL Physics Monitor: {ticker}")

if run_btn:
    with st.spinner(f"{ticker} のデータを取得・計算中..."):
        try:
            # データ取得 (全期間)
            full_df = yf.download(ticker, start=start_date, progress=False)
            if full_df.empty:
                st.error("データ取得失敗")
                st.stop()
            
            if isinstance(full_df.columns, pd.MultiIndex):
                full_data = full_df['Close'].iloc[:, 0]
            else:
                full_data = full_df['Close']
            full_data = full_data.dropna()

            # 訓練データの切り出し
            train_mask = full_data.index <= pd.Timestamp(end_date)
            train_data = full_data[train_mask]
            test_data = full_data[~train_mask]

            if len(train_data) < 30:
                st.error("データ不足。期間を調整してください。")
                st.stop()

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        # ==========================================
        # モードA: 通常解析 (バックテスト)
        # ==========================================
        if analysis_type == "通常解析 (バックテスト)":
            prices = train_data.values.flatten()
            t_data = np.arange(len(prices))
            p_data = np.log(prices)
            
            last_t = t_data[-1]
            last_price = p_data[-1]
            
            # 最適化
            initial_guess = [last_t + 40, 0.5, 9.0, last_price, -0.1, 0.05, 0]
            bounds = [(last_t + 1, last_t + 365), (m_min, m_max), (w_min, w_max), (None, None), (None, None), (None, None), (0, 2*np.pi)]
            
            res = minimize(objective, initial_guess, args=(t_data, p_data), method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
            
            if res.success:
                tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est = res.x
                
                # 結果計算
                days_rem = tc_est - last_t
                crash_date = train_data.index[-1] + timedelta(days=float(days_rem))
                
                # R2
                p_model = lppl_func(t_data, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
                r2 = 1 - (np.sum((p_data - p_model)**2) / np.sum((p_data - np.mean(p_data))**2))

                # 表示
                st.markdown(f"### 🗓️ 予測結果 (基準日: {train_data.index[-1].strftime('%Y-%m-%d')})")
                c1, c2, c3 = st.columns(3)
                c1.metric("推定 X-Day", crash_date.strftime('%Y-%m-%d'), delta="Singularity", delta_color="inverse")
                c2.metric("残り時間", f"{days_rem:.1f} 日")
                c3.metric("決定係数 ($R^2$)", f"{r2:.4f}")
                
                # グラフ
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(train_data.index, np.log(train_data.values), label='Training Data', color='blue')
                if not test_data.empty:
                    ax.plot(test_data.index, np.log(test_data.values), label='Actual Market (Answer)', color='orange', linewidth=2)
                
                # 予測線
                t_future = np.arange(0, last_t + days_rem + 20, 0.1)
                p_future = lppl_func(t_future, tc_est, m_est, omega_est, A_est, B_est, C_est, phi_est)
                date_future = [train_data.index[0] + timedelta(days=float(x)) for x in t_future]
                
                ax.plot(date_future, p_future, label='LPPL Model', color='red', linestyle='--')
                ax.axvline(crash_date, color='green', linestyle=':', label='X-Day')
                ax.set_title("Backtesting Visualization")
                ax.legend()
                ax.grid(True, alpha=0.5)
                st.pyplot(fig)

                # パラメータ診断
                st.markdown("#### 🧬 物理パラメータ診断")
                col_m, col_w = st.columns(2)
                with col_m:
                    st.info(f"**m = {m_est:.4f}**")
                    if 0.1 < m_est < 0.9: st.success("✅ 加速中 (Valid)")
                    else: st.warning("⚠️ 異常値")
                with col_w:
                    st.info(f"**ω = {omega_est:.4f}**")
                    if 6.0 < omega_est < 13.0: st.success("✅ 周期性あり (Valid)")
                    else: st.warning("⚠️ ノイズ可能性")

            else:
                st.error("最適化失敗")

        # ==========================================
        # モードB: 堅牢性検証 (New!)
        # ==========================================
        elif analysis_type == "堅牢性検証 (Sensitivity Check)":
            st.markdown("### 🧪 初期条件に対する感度分析 (Perturbation Analysis)")
            st.info("解析開始日(Start Date)を **1週間ずつずらしながら 5回計算** し、X-Dayの収束性を確認します。")
            
            results = []
            progress_bar = st.progress(0)
            
            # 5回の試行
            shifts = [0, 7, 14, 21, 28] 
            
            for i, shift in enumerate(shifts):
                # データをずらして取得
                shifted_start = pd.to_datetime(start_date) + timedelta(days=shift)
                
                # データスライス
                mask = (full_data.index >= shifted_start) & (full_data.index <= pd.Timestamp(end_date))
                sub_data = full_data[mask]
                
                if len(sub_data) < 30: continue
                
                # 計算
                prices = sub_data.values.flatten()
                t_sub = np.arange(len(prices))
                p_sub = np.log(prices)
                last_t = t_sub[-1]
                last_price = p_sub[-1]
                
                init = [last_t + 40, 0.5, 9.0, last_price, -0.1, 0.05, 0]
                bnds = [(last_t + 1, last_t + 365), (m_min, m_max), (w_min, w_max), (None, None), (None, None), (None, None), (0, 2*np.pi)]
                
                res = minimize(objective, init, args=(t_sub, p_sub), method='L-BFGS-B', bounds=bnds, options={'maxiter': 5000})
                
                if res.success:
                    tc_val = res.x[0]
                    days_r = tc_val - last_t
                    pred_date = sub_data.index[-1] + timedelta(days=float(days_r))
                    results.append({
                        "Start Date": shifted_start.strftime('%Y-%m-%d'),
                        "Predicted X-Day": pred_date,
                        "m": res.x[1],
                        "omega": res.x[2]
                    })
                
                progress_bar.progress((i + 1) / len(shifts))
            
            if results:
                res_df = pd.DataFrame(results)
                
                # 統計
                mean_date = res_df["Predicted X-Day"].mean()
                min_date = res_df["Predicted X-Day"].min()
                max_date = res_df["Predicted X-Day"].max()
                diff_days = (max_date - min_date).days
                
                c1, c2 = st.columns(2)
                c1.dataframe(res_df.style.format({"m": "{:.3f}", "omega": "{:.3f}"}))
                
                # 判定ロジック
                with c2:
                    st.markdown(f"**平均 X-Day:** {mean_date.strftime('%Y-%m-%d')}")
                    st.markdown(f"**ばらつき (Max-Min):** {diff_days} 日")
                    
                    if diff_days <= 10:
                        st.success("✅ **結果は堅牢です (Robust)**\n\n初期条件を変えても予測日は収束しています。信頼性は高いです。")
                    elif diff_days <= 30:
                        st.warning("⚠️ **結果はやや不安定です**\n\n予測日に1ヶ月程度の幅があります。レンジで捉えてください。")
                    else:
                        st.error("❌ **結果は信頼できません (Unstable)**\n\n開始日によって予測が大きく変わります。現在はモデルが適合しない局面です。")
                
                # 分布プロット
                fig, ax = plt.subplots(figsize=(10, 2))
                dates = [d.to_pydatetime() for d in res_df["Predicted X-Day"]]
                ax.scatter(dates, [1]*len(dates), c='red', s=100, alpha=0.6)
                ax.set_yticks([])
                ax.set_title("Distribution of Predicted X-Days")
                ax.grid(True, axis='x')
                st.pyplot(fig)
                
            else:
                st.error("有効な解析結果が得られませんでした。")
