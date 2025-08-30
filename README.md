# Taker-Flow Strategy Dashboard (Streamlit)

A web dashboard to backtest a simple taker-buy ratio strategy against Binance data.

## Features
- Fetches **Binance klines** (includes `takerBuyBaseVolume`) without API keys.
- Parameters in the sidebar:
  - Buy ≥ threshold, Sell ≤ threshold
  - Trend filter (Close > EMA200), Volume filter (Vol > EMA20)
  - Fees, initial capital
- Charts: Equity curve, Price with entries/exits, Taker-ratio panel
- CSV export of results

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push these files to a **public GitHub repo**.
2. Go to https://share.streamlit.io
3. "New app" → point to your repo → `app.py`
4. Deploy.

## Deploy (Hugging Face Spaces)
1. Create a new Space → Framework: **Streamlit**
2. Upload `app.py` and `requirements.txt`
3. The Space will build and launch automatically.