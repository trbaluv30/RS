# Regime CRS Screener

A Streamlit app built from a notebook that:

- downloads SPY and current S&P 500 constituents
- detects skewness regimes with expanding-window thresholds
- computes TC2000-style comparative relative strength
- builds strongest / weakest tables
- adds three extra columns:
  - IV Gauge (Z)
  - IV Gauge (Robust Z)
  - Days to ER

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Deploy the repo and set `app.py` as the entrypoint.
