import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

class SchwartzOneFactor:
    def __init__(self, dt=1/252):
        self.dt = dt
        self.kappa = None
        self.alpha = None
        self.sigma = None
        self.stationary_std = None

    def fit(self, prices):
        # 対数価格
        log_prices = np.log(prices)
        
        # 回帰用データセット
        x_t = log_prices[1:]
        x_t_minus_1 = log_prices[:-1]
        
        # 線形回帰 (OU過程の離散化)
        slope, intercept = np.polyfit(x_t_minus_1, x_t, 1)
        
        # パラメータ復元
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

def run_analysis(ticker="CL=F", lookback_years=2):
    print(f"--- 原油先物 ({ticker}) 分析開始 ---")
    print(f"データ取得中 ({lookback_years}年分)...")
    
    try:
        # yfinanceでのデータ取得
        data = yf.download(ticker, period=f"{lookback_years}y", interval="1d", progress=False)
        if len(data) < 100:
            print("エラー: データが不足しています。")
            return
        
        prices = data['Close'].values.flatten()
        last_date = data.index[-1].date()
        
        # モデル適用
        model = SchwartzOneFactor()
        params = model.fit(prices)
        
        # 結果表示
        print("\n" + "="*40)
        print("   SCHWARTZ 1-FACTOR MODEL RESULTS")
        print("="*40)
        
        print(f"\n[推計パラメータ]")
        print(f"  > 平均回帰速度 (Kappa) : {params['Kappa (Speed)']:.4f}")
        print(f"  > 半減期 (Half-Life)   : {params['Half-Life (Days)']:.1f} 日")
        print(f"  > ボラティリティ (Sigma): {params['Sigma (Vol)']:.4f}")
        print(f"  > 長期均衡価格 (Mean)  : ${np.exp(params['Alpha (Log Mean)']):.2f}")
        
        # シグナル判定
        current_price = prices[-1]
        z_score = model.calculate_z_score(current_price)
        
        print(f"\n[現在価格の評価 ({last_date})]")
        print(f"  > 現在価格 : ${current_price:.2f}")
        print(f"  > Zスコア  : {z_score:.2f} σ")
        
        print("\n[判定]")
        if z_score > 2.0:
            print("  >> SELL SIGNAL (統計的に割高)")
            print("     価格が長期平均から +2σ 以上乖離しています。")
        elif z_score < -2.0:
            print("  >> BUY SIGNAL (統計的に割安)")
            print("     価格が長期平均から -2σ 以上乖離しています。")
        else:
            print("  >> NEUTRAL (レンジ内)")
            print("     統計的に有意な歪みはありません。")
            
        print("="*40)

    except Exception as e:
        print(f"\n実行エラーが発生しました: {e}")
        print("考えられる原因: インターネット接続がない、または yfinance がインストールされていません。")

if __name__ == "__main__":
    run_analysis()
