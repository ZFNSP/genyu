import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

class SchwartzOneFactor:
    def __init__(self, dt=1/252):
        """
        dt: データの時間間隔（日次データなら1/252年）
        """
        self.dt = dt
        self.kappa = None # 平均回帰速度
        self.alpha = None # 長期平均価格（対数）
        self.sigma = None # ボラティリティ
        self.stationary_std = None # 定常状態の標準偏差

    def fit(self, prices):
        """
        過去の価格データからパラメータを推定する（AR(1)プロセスへの回帰）
        """
        # 対数価格
        log_prices = np.log(prices)
        
        # x_t と x_{t-1} の用意
        x_t = log_prices[1:]
        x_t_minus_1 = log_prices[:-1]
        
        # 線形回帰: x_t = slope * x_{t-1} + intercept
        # slope = e^{-kappa * dt}
        # intercept = alpha * (1 - e^{-kappa * dt})
        slope, intercept = np.polyfit(x_t_minus_1, x_t, 1)
        
        # 残差から条件付き分散を計算
        residuals = x_t - (slope * x_t_minus_1 + intercept)
        resid_std = np.std(residuals)
        
        # 物理パラメータへの変換
        self.kappa = -np.log(slope) / self.dt
        self.alpha = intercept / (1 - slope)
        
        # Var(epsilon) = (sigma^2 / 2kappa) * (1 - e^{-2kappa*dt}) から sigma を逆算
        self.sigma = resid_std * np.sqrt((2 * self.kappa) / (1 - slope**2))
        
        # 定常状態（t -> inf）での標準偏差: sigma / sqrt(2*kappa)
        self.stationary_std = self.sigma / np.sqrt(2 * self.kappa)
        
        return {
            "Kappa (Speed)": self.kappa,
            "Alpha (Log Mean)": self.alpha,
            "Implied Price Mean": np.exp(self.alpha + (self.stationary_std**2)/2), # 対数正規分布の平均
            "Sigma (Vol)": self.sigma,
            "Half-Life (Days)": (np.log(2) / self.kappa) * 252
        }

    def calculate_z_score(self, current_price):
        """
        現在の価格が長期平均から何シグマ離れているかを計算
        """
        if self.alpha is None:
            raise Exception("Model not fitted yet.")
            
        log_price = np.log(current_price)
        # 定常分布におけるZスコア
        z_score = (log_price - self.alpha) / self.stationary_std
        return z_score

# --- 実行セクション ---

def run_analysis(ticker="CL=F", lookback_years=2):
    # 1. データ取得（WTI原油先物）
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, period=f"{lookback_years}y", interval="1d", progress=False)
    
    if len(data) < 100:
        print("データが不足しています。")
        return

    # 終値を使用
    prices = data['Close'].values.flatten()
    dates = data.index
    
    # 2. モデルのキャリブレーション
    model = SchwartzOneFactor()
    params = model.fit(prices)
    
    print("\n--- Estimated Parameters ---")
    for k, v in params.items():
        print(f"{k}: {v:.4f}")
        
    # 3. 現在のシグナル判定
    current_price = prices[-1]
    z_score = model.calculate_z_score(current_price)
    
    print(f"\n--- Current Signal ({dates[-1].date()}) ---")
    print(f"Current Price: {current_price:.2f}")
    print(f"Long-term Mean Level: {np.exp(model.alpha):.2f}")
    print(f"Z-Score: {z_score:.2f}")
    
    if abs(z_score) > 2.0:
        action = "SELL (Overbought)" if z_score > 0 else "BUY (Oversold)"
        print(f"Signal: >> {action} << (Mean Reversion Chance)")
    else:
        print("Signal: Neutral (Within normal range)")

    # 4. 可視化
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label='Market Price', color='black', alpha=0.6)
    
    # 理論的な平均回帰バンド（長期平均 ± 2シグマ）
    mean_price = np.exp(model.alpha)
    upper_band = np.exp(model.alpha + 2 * model.stationary_std)
    lower_band = np.exp(model.alpha - 2 * model.stationary_std)
    
    plt.axhline(mean_price, color='green', linestyle='--', label='Long-term Mean (exp(alpha))')
    plt.axhline(upper_band, color='red', linestyle=':', label='+2 Sigma')
    plt.axhline(lower_band, color='blue', linestyle=':', label='-2 Sigma')
    
    plt.title(f"Schwartz 1-Factor Model: Mean Reversion Analysis ({ticker})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 実行（Google Colabやローカル環境で実行可能）
if __name__ == "__main__":
    # yfinanceがインストールされている前提
    try:
        run_analysis()
    except Exception as e:
        print(f"Error: {e}")
        print("必要なライブラリ: pip install yfinance pandas numpy matplotlib scipy")
