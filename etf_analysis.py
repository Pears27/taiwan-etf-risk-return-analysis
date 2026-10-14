# ===== 4-1 資料讀取與整理 =====
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang TC"]
plt.rcParams["axes.unicode_minus"] = False

etfs = {
    "0050": "0050.TW",
    "0056": "0056.TW",
    "00692": "00692.TW",
    "00713": "00713.TW",
    "00878": "00878.TW"
}

prices = pd.DataFrame()

for name, ticker in etfs.items():
    data = yf.download(
        ticker,
        start="2021-01-01",
        end="2026-01-01",
        auto_adjust=False,
        progress=False,
        multi_level_index=False
    )

    prices[name] = data["Adj Close"]

print(prices.isnull().sum())

prices = prices.dropna()

print(prices.head())
print(prices.tail())
print(prices.shape)

print(prices.groupby(prices.index.year).size())

# ===== 4-2 計算報酬率 =====

# 每日報酬率
daily_returns = prices.pct_change().dropna()

print("每日報酬率前5筆：")
print(daily_returns.head())

# 累積報酬率
cumulative_returns = prices.iloc[-1] / prices.iloc[0] - 1

# 實際研究期間
years = (prices.index[-1] - prices.index[0]).days / 365.25

# 年化報酬率
annualized_returns = (
    (prices.iloc[-1] / prices.iloc[0]) ** (1 / years)
    - 1
)

# 整理結果
return_summary = pd.DataFrame({
    "累積報酬率(%)": cumulative_returns * 100,
    "年化報酬率(%)": annualized_returns * 100
})

print("\n報酬率分析結果：")
print(return_summary.round(2))

# ===== 4-3 計算風險指標 =====

# 年化波動率
annualized_volatility = daily_returns.std() * np.sqrt(252)

# 計算每一天截至當時的歷史最高點
running_max = prices.cummax()

# 計算回撤
drawdowns = prices / running_max - 1

# 找出最大回撤
max_drawdown = drawdowns.min()

# 整理結果
risk_summary = pd.DataFrame({
    "年化波動率(%)": annualized_volatility * 100,
    "最大回撤(%)": max_drawdown * 100
})

print("\n風險分析結果：")
print(risk_summary.round(2))

# ===== 4-4 資料視覺化 =====

# 計算每天相對於研究起點的累積報酬率
cumulative_curve = (prices / prices.iloc[0] - 1) * 100

plt.figure(figsize=(10, 6))

for etf in cumulative_curve.columns:
    plt.plot(cumulative_curve.index, cumulative_curve[etf], label=etf)

plt.title("2021–2025年五檔ETF累積報酬走勢")
plt.xlabel("日期")
plt.ylabel("累積報酬率（%）")
plt.axhline(0, linewidth=0.8, linestyle="--", alpha=0.6)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("01_累積報酬走勢.png", dpi=300)
plt.close()

comparison = pd.DataFrame({
    "年化報酬率": annualized_returns * 100,
    "年化波動率": annualized_volatility * 100
})

ax = comparison.plot(
    kind="bar",
    figsize=(9, 6)
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=3)

plt.title("五檔ETF年化報酬率與波動率比較")
plt.xlabel("ETF")
plt.ylabel("百分比（%）")
plt.xticks(rotation=0)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("02_報酬與波動率比較.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))

x = annualized_volatility * 100
y = annualized_returns * 100

plt.scatter(x, y, s=100)

for etf in x.index:
    plt.annotate(
        etf,
        (x[etf], y[etf]),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.title("五檔ETF風險－報酬關係")
plt.xlabel("年化波動率（%）")
plt.ylabel("年化報酬率（%）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.margins(x=0.08, y=0.08)

plt.savefig("03_風險報酬散點圖.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))

max_drawdown_percent = max_drawdown * 100

bars = plt.bar(
    max_drawdown_percent.index,
    max_drawdown_percent.values
)

plt.title("五檔ETF最大回撤比較")
plt.xlabel("ETF")
plt.ylabel("最大回撤（%）")
plt.axhline(0, linewidth=0.8)
plt.grid(axis="y", alpha=0.3)

plt.bar_label(
    bars,
    labels=[f"{v:.2f}%" for v in max_drawdown_percent.values],
    padding=3
)

plt.tight_layout()
plt.savefig("04_最大回撤比較.png", dpi=300)
plt.close()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 儲存五檔 ETF 的共同交易日調整後收盤價
prices.to_csv(
    DATA_DIR / "adjusted_close.csv",
    encoding="utf-8-sig",
    index_label="Date"
)

summary = pd.DataFrame({
    "cumulative_return_pct": cumulative_returns * 100,
    "annualized_return_pct": annualized_returns * 100,
    "annualized_volatility_pct": annualized_volatility * 100,
    "max_drawdown_pct": max_drawdown * 100,
})

summary.index.name = "ETF"
summary = summary.round(2)

summary.to_csv(
    RESULTS_DIR / "summary.csv",
    encoding="utf-8-sig"
)

print("\n研究結果總表：")
print(summary)