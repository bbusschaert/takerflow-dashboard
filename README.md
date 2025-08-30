# Taker-Flow Strategy Dashboard (Streamlit) — Fixed Build

This build adds:
- Robust Binance fetch: host rotation, retry/backoff, capped page pulls.
- Friendly UI errors if data is empty or zero-volume.
- Plotly `add_hline` compatibility fix.
- Defensive guards around data fields.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push these three files to a **public GitHub repo**.
2. Go to https://share.streamlit.io → **New app**.
3. Pick your repo, branch `main`, and set **Main file path** to `app.py`.
4. Deploy. First build may take a few minutes.

## Recommended first run (to verify)
- Symbol: BTCUSDT
- Interval: 1h
- Date range: last 60 days
- Buy ≥ 0.80, Sell ≤ 0.60
- Filters ON, Fee 0.001