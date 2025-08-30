import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import datetime as dt
import plotly.graph_objects as go

BINANCE_BASE = "https://api.binance.com"

INTERVALS = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h"
}

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def fetch_klines(symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    limit = 1000
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000)
    }
    frames = []
    while True:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume","close_time","quote_volume",
            "num_trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        for col in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        frames.append(df)
        last_open = int(df["open_time"].iloc[-1].timestamp() * 1000)
        params["startTime"] = last_open + 1
        if len(data) < limit or params["startTime"] >= params["endTime"]:
            break
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.loc[out["open_time"] < pd.to_datetime(int(end.timestamp()*1000), unit="ms", utc=True)]
    return out.reset_index(drop=True)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def backtest(df: pd.DataFrame, buy_thr: float, sell_thr: float,
             use_trend_filter: bool, use_volume_filter: bool,
             fee_per_side: float, initial_capital: float):
    df = df.copy()
    df["taker_ratio"] = (df["taker_buy_base"] / df["volume"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ema200"] = ema(df["close"], 200)
    df["vol_ema20"] = ema(df["volume"], 20)

    holding = False
    equity = [initial_capital]
    entries, exits = [], []
    trade_pnls = []
    entry_equity = None
    entry_price  = None

    for i in range(len(df)-1):
        price = df["close"].iloc[i]
        next_ret = np.log(df["close"].iloc[i+1] / df["close"].iloc[i])
        taker_ok = df["taker_ratio"].iloc[i] >= buy_thr
        exit_ok  = df["taker_ratio"].iloc[i] <= sell_thr
        trend_ok = (df["close"].iloc[i] > df["ema200"].iloc[i]) if use_trend_filter else True
        vol_ok   = (df["volume"].iloc[i] > df["vol_ema20"].iloc[i]) if use_volume_filter else True

        e = equity[-1]

        if not holding and taker_ok and trend_ok and vol_ok:
            e *= (1 - fee_per_side)
            holding = True
            entries.append((df["open_time"].iloc[i], float(price)))
            entry_equity = e
            entry_price = price
        elif holding and exit_ok:
            e *= (1 - fee_per_side)
            holding = False
            exits.append((df["open_time"].iloc[i], float(price)))
            if entry_equity is not None:
                trade_pnls.append(e - entry_equity)
                entry_equity = None
                entry_price = None

        if holding:
            e *= np.exp(next_ret)

        equity.append(e)

    result = pd.DataFrame({
        "time": df["open_time"],
        "close": df["close"],
        "equity": equity[:-1],
        "taker_ratio": df["taker_ratio"],
        "ema200": df["ema200"],
        "vol": df["volume"],
        "vol_ema20": df["vol_ema20"]
    })

    total_return = (result["equity"].iloc[-1] / result["equity"].iloc[0] - 1) * 100.0
    dd = (result["equity"].cummax() - result["equity"]) / result["equity"].cummax()
    max_dd = dd.max() * 100.0
    n_trades = len(trade_pnls)
    win_rate = (np.mean([1 if p>0 else 0 for p in trade_pnls]) * 100.0) if n_trades>0 else 0.0

    kpis = {
        "Total Return %": round(total_return, 2),
        "Max Drawdown %": round(float(max_dd), 2),
        "# Trades": int(n_trades),
        "Win Rate %": round(float(win_rate), 2)
    }

    annotations = {"entries": entries, "exits": exits}
    return result, kpis, annotations

def plot_equity(result: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["time"], y=result["equity"], mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Equity (base currency)")
    return fig

def plot_price_with_trades(df: pd.DataFrame, ann):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["open_time"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df["open_time"], y=ema(df["close"], 200), mode="lines", name="EMA200"))

    if ann["entries"]:
        xs, ys = zip(*ann["entries"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Entry", marker=dict(symbol="triangle-up", size=10)))
    if ann["exits"]:
        xs, ys = zip(*ann["exits"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Exit", marker=dict(symbol="triangle-down", size=10)))
    fig.update_layout(title="Price with Entries/Exits")
    return fig

def ui_sidebar():
    st.sidebar.header("Parameters")
    symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0)
    interval = st.sidebar.selectbox("Interval", list(INTERVALS.values()), index=1)
    end_date = st.sidebar.date_input("End Date (UTC)", dt.datetime.utcnow().date())
    start_date = st.sidebar.date_input("Start Date (UTC)", end_date - dt.timedelta(days=60))
    buy_thr = st.sidebar.slider("Buy Threshold (taker ratio ≥)", 0.50, 0.99, 0.80, 0.01)
    sell_thr = st.sidebar.slider("Sell Threshold (taker ratio ≤)", 0.40, 0.95, 0.60, 0.01)
    if sell_thr >= buy_thr:
        st.sidebar.warning("Sell threshold must be < Buy threshold.")
    use_trend = st.sidebar.checkbox("Trend filter (Close > EMA200)", True)
    use_vol   = st.sidebar.checkbox("Volume filter (Vol > EMA20)", True)
    fee = st.sidebar.number_input("Fee per side (e.g., 0.001 = 0.1%)", value=0.001, min_value=0.0, max_value=0.01, step=0.0005, format="%.4f")
    capital = st.sidebar.number_input("Initial Capital", value=10000.0, min_value=100.0, step=100.0)
    run_btn = st.sidebar.button("Run Backtest")
    return {
        "symbol": symbol, "interval": interval,
        "start": dt.datetime.combine(start_date, dt.time.min).replace(tzinfo=dt.timezone.utc),
        "end": dt.datetime.combine(end_date, dt.time.min).replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1),
        "buy_thr": buy_thr, "sell_thr": sell_thr,
        "use_trend": use_trend, "use_vol": use_vol,
        "fee": fee, "capital": capital, "run": run_btn
    }

def main():
    st.set_page_config(page_title="Taker-Flow Strategy Dashboard", layout="wide")
    st.title("📈 Taker-Flow Strategy Backtester (Binance)")
    st.markdown("Long-only system: **Enter** when taker-buy ratio ≥ BuyThreshold (+ optional filters). **Exit** when ratio ≤ SellThreshold.")

    cfg = ui_sidebar()
    if not cfg["run"]:
        st.info("Adjust parameters in the sidebar and click **Run Backtest**.")
        st.stop()

    with st.spinner("Fetching data from Binance..."):
        df = fetch_klines(cfg["symbol"], cfg["interval"], cfg["start"], cfg["end"])
    if df.empty:
        st.error("No data returned. Try a different date range/symbol/interval.")
        st.stop()

    st.success(f"Fetched {len(df)} candles for {cfg['symbol']} ({cfg['interval']}).")

    result, kpis, ann = backtest(
        df, cfg["buy_thr"], cfg["sell_thr"],
        cfg["use_trend"], cfg["use_vol"],
        cfg["fee"], cfg["capital"]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return %", kpis["Total Return %"])
    c2.metric("Max Drawdown %", kpis["Max Drawdown %"])
    c3.metric("# Trades", kpis["# Trades"])
    c4.metric("Win Rate %", kpis["Win Rate %"])

    st.plotly_chart(plot_equity(result), use_container_width=True)
    st.plotly_chart(plot_price_with_trades(df, ann), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["time"], y=result["taker_ratio"], mode="lines", name="Taker Buy Ratio"))
    fig.add_hline(y=cfg["buy_thr"], line=dict(color="green", dash="dot"))
    fig.add_hline(y=cfg["sell_thr"], line=dict(color="red", dash="dot"))
    fig.update_layout(title="Taker Buy Ratio")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("Download Result CSV", result.to_csv(index=False).encode("utf-8"),
                       file_name="backtest_result.csv", mime="text/csv")

if __name__ == "__main__":
    main()