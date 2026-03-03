
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from scipy.stats import skew


st.set_page_config(
    page_title="Regime CRS Screener",
    page_icon="📈",
    layout="wide"
)


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
@dataclass
class Config:
    start_date: str = "2015-01-01"
    end_date: str = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    analysis_start: str = "2018-01-01"

    skewness_window: int = 63
    regime_lower_pct: float = 33
    regime_upper_pct: float = 67
    min_regime_history: int = 252

    rs_lookbacks: Tuple[int, int, int] = (21, 63, 126)

    top_decile: int = 10
    bottom_decile: int = 10
    rebalance_freq: str = "W"

    min_price: float = 5.0
    max_missing_pct: float = 0.20

    iv_window_days: int = 252
    iv_rv_lookback: int = 20
    strongest_n: int = 50
    weakest_n: int = 50


DEFAULT_CFG = Config()


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def sigma_bucket(z: float):
    if pd.isna(z):
        return np.nan
    z = float(z)
    if z < -2:
        return 1
    if z < -1:
        return 2
    if z <= 1:
        return 3
    if z <= 2:
        return 4
    return 5


def bucket_label(bucket):
    if pd.isna(bucket):
        return "NA"
    return f"Section {int(bucket)}"


def gauge_text_from_bucket(bucket):
    if pd.isna(bucket):
        return "NA"
    dots = ["○"] * 5
    dots[int(bucket) - 1] = "●"
    return "".join(dots)


def gauge_display(z: float) -> str:
    if pd.isna(z):
        return "NA"
    bucket = sigma_bucket(z)
    return f"{gauge_text_from_bucket(bucket)} | {bucket_label(bucket)} | {z:+.2f}σ"


def compute_realized_vol_from_close(close: pd.Series, lookback: int = 20) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce").dropna()
    if len(close) < lookback + 5:
        return pd.Series(dtype=float)
    log_ret = np.log(close).diff()
    rv = log_ret.rolling(lookback).std(ddof=1) * np.sqrt(252)
    return rv.dropna()


def standard_z_from_series(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 20:
        return np.nan
    mu = s.mean()
    sigma = s.std(ddof=1)
    if sigma == 0 or pd.isna(sigma):
        return 0.0
    return float((s.iloc[-1] - mu) / sigma)


def robust_z_from_series(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 20:
        return np.nan
    med = s.median()
    mad = (s - med).abs().median()
    robust_scale = 1.4826 * mad
    if robust_scale == 0 or pd.isna(robust_scale):
        std = s.std(ddof=1)
        if std == 0 or pd.isna(std):
            return 0.0
        return float((s.iloc[-1] - med) / std)
    return float((s.iloc[-1] - med) / robust_scale)


def normalize_index(df: pd.DataFrame | pd.Series):
    if getattr(df.index, "tz", None):
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    tickers = tables[0]["Symbol"].tolist()
    tickers = sorted({str(t).replace(".", "-") for t in tickers})
    bad = {"Q", "SNDK"}
    tickers = [t for t in tickers if t not in bad]
    return tickers


@st.cache_data(ttl=21600, show_spinner=True)
def download_market_data(start_date: str, end_date: str, min_price: float, max_missing_pct: float):
    tickers = get_sp500_tickers()

    spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_close = spy_data["Close"]["SPY"]
    else:
        spy_close = spy_data["Close"]
    spy_close = normalize_index(spy_close.copy())
    spy_returns = spy_close.pct_change().dropna()

    stock_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        threads=True,
        auto_adjust=False,
    )

    if isinstance(stock_data.columns, pd.MultiIndex):
        prices = stock_data["Close"].copy()
    else:
        prices = stock_data[["Close"]].copy()

    prices = normalize_index(prices.copy())
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    prices = prices.ffill(limit=5)
    prices = prices.where(prices >= min_price)

    missing_pct = prices.isnull().sum() / len(prices)
    valid_tickers = missing_pct[missing_pct < max_missing_pct].index.tolist()
    prices = prices[valid_tickers]
    stock_returns = prices.pct_change()

    return {
        "spy_close": spy_close,
        "spy_returns": spy_returns,
        "prices": prices,
        "stock_returns": stock_returns,
        "tickers": valid_tickers,
    }


def calculate_rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=window).apply(
        lambda x: skew(x, bias=False), raw=True
    )


def classify_regimes_expanding_window(
    skewness_series: pd.Series,
    lower_pct: float,
    upper_pct: float,
    min_history: int = 252,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    regime = pd.Series(index=skewness_series.index, dtype=object)
    lower_thresholds = pd.Series(index=skewness_series.index, dtype=float)
    upper_thresholds = pd.Series(index=skewness_series.index, dtype=float)

    for i in range(len(skewness_series)):
        if i < min_history:
            continue

        hist = skewness_series.iloc[: i + 1].dropna()
        if len(hist) < min_history:
            continue

        lower = np.nanpercentile(hist, lower_pct)
        upper = np.nanpercentile(hist, upper_pct)
        current = skewness_series.iloc[i]

        lower_thresholds.iloc[i] = lower
        upper_thresholds.iloc[i] = upper

        if pd.isna(current):
            regime.iloc[i] = np.nan
        elif current > upper:
            regime.iloc[i] = "positive_skew"
        elif current < lower:
            regime.iloc[i] = "negative_skew"
        else:
            regime.iloc[i] = "neutral"

    return regime, lower_thresholds, upper_thresholds


def calculate_relative_strength(prices: pd.DataFrame, lookbacks: Tuple[int, int, int]) -> pd.DataFrame:
    rs_scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for ticker in prices.columns:
        p = prices[ticker]
        p_21 = p.shift(lookbacks[0])
        p_63 = p.shift(lookbacks[1])
        p_126 = p.shift(lookbacks[2])
        rs_scores[ticker] = p * (2 / p_21 + 1 / p_63 + 1 / p_126)
    return rs_scores


def create_regime_crs_portfolio(
    rs_ranks: pd.DataFrame,
    regime: pd.Series,
    stock_returns: pd.DataFrame,
    top_decile: int,
    bottom_decile: int,
    rebalance_freq: str = "W",
) -> tuple[pd.Series, pd.DataFrame]:
    common_idx = rs_ranks.index.intersection(regime.dropna().index).intersection(stock_returns.index)
    rs_ranks = rs_ranks.loc[common_idx]
    regime = regime.loc[common_idx]
    returns = stock_returns.loc[common_idx]

    rebal_dates = rs_ranks.resample(rebalance_freq).last().index
    rebal_dates = rebal_dates[rebal_dates.isin(rs_ranks.index)]

    portfolio_returns = pd.Series(index=common_idx, dtype=float)
    holdings_record = []
    current_holdings = []
    last_regime_for_holdings = None

    for date in common_idx:
        current_regime = regime.loc[date]

        if current_regime == "negative_skew" or pd.isna(current_regime):
            portfolio_returns.loc[date] = 0.0
            if last_regime_for_holdings != "negative_skew":
                holdings_record.append({"date": date, "regime": current_regime, "n_holdings": 0, "action": "EXIT_TO_CASH"})
                current_holdings = []
                last_regime_for_holdings = "negative_skew"
            continue

        regime_changed = last_regime_for_holdings != current_regime
        is_rebal_day = date in rebal_dates

        if is_rebal_day or regime_changed or len(current_holdings) == 0:
            day_ranks = rs_ranks.loc[date].dropna()

            if current_regime == "positive_skew":
                current_holdings = day_ranks[day_ranks <= bottom_decile].index.tolist()
                action = "LONG_LOSERS"
            elif current_regime == "neutral":
                current_holdings = day_ranks[day_ranks >= (100 - top_decile)].index.tolist()
                action = "LONG_WINNERS"
            else:
                current_holdings = []
                action = "UNKNOWN"

            holdings_record.append(
                {"date": date, "regime": current_regime, "n_holdings": len(current_holdings), "action": action}
            )
            last_regime_for_holdings = current_regime

        if current_holdings:
            valid_holdings = [h for h in current_holdings if h in returns.columns]
            portfolio_returns.loc[date] = returns.loc[date, valid_holdings].mean() if valid_holdings else 0.0
        else:
            portfolio_returns.loc[date] = 0.0

    return portfolio_returns.dropna(), pd.DataFrame(holdings_record)


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {k: np.nan for k in ["cagr", "vol", "sharpe", "max_drawdown", "total_return"]}
    equity = (1 + returns).cumprod()
    total_return = equity.iloc[-1] - 1
    years = len(returns) / 252
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std(ddof=1) * np.sqrt(252)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252) if returns.std(ddof=1) > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
    }


def _normalize_ts(x):
    ts = pd.Timestamp(x)
    try:
        ts = ts.tz_localize(None)
    except Exception:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            pass
    return ts.normalize()


@st.cache_data(ttl=21600, show_spinner=False)
def get_days_to_earnings_cached(ticker: str):
    ticker = ticker.upper().strip()
    today = pd.Timestamp.today().normalize()

    try:
        tkr = yf.Ticker(ticker)
        edf = tkr.get_earnings_dates(limit=12)
        if edf is not None and len(edf) > 0:
            idx = pd.to_datetime(edf.index, errors="coerce")
            idx = [_normalize_ts(x) for x in idx if pd.notna(x)]
            future = [d for d in idx if d >= today]
            if future:
                return int((min(future) - today).days)
    except Exception:
        pass

    try:
        tkr = yf.Ticker(ticker)
        cal = tkr.calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", None)
            if raw is not None:
                if isinstance(raw, (list, tuple, pd.Series, np.ndarray)):
                    vals = [_normalize_ts(x) for x in raw if pd.notna(x)]
                    future = [d for d in vals if d >= today]
                    if future:
                        return int((min(future) - today).days)
                else:
                    next_dt = _normalize_ts(raw)
                    if next_dt >= today:
                        return int((next_dt - today).days)
    except Exception:
        pass

    return np.nan


def get_iv_stats_from_prices(prices: pd.DataFrame, ticker: str, window_days: int, rv_lookback: int) -> dict:
    out = {
        "ticker": ticker,
        "IV Proxy": np.nan,
        "Z Score": np.nan,
        "Robust Z Score": np.nan,
        "Days to ER": np.nan,
    }

    if ticker not in prices.columns:
        return out

    px = pd.to_numeric(prices[ticker], errors="coerce").dropna()
    if px.empty:
        return out

    rv = compute_realized_vol_from_close(px, lookback=rv_lookback)
    if len(rv) < window_days:
        return out

    hist = rv.tail(window_days)
    z = standard_z_from_series(hist)
    rz = robust_z_from_series(hist)

    out["IV Proxy"] = float(hist.iloc[-1])
    out["Z Score"] = z
    out["Robust Z Score"] = rz
    out["Days to ER"] = get_days_to_earnings_cached(ticker)

    return out


def enrich_rank_table_with_iv(rank_df: pd.DataFrame, prices: pd.DataFrame, window_days: int, rv_lookback: int) -> pd.DataFrame:
    rows = []
    for t in rank_df["Ticker"].astype(str).tolist():
        rows.append(get_iv_stats_from_prices(prices, t, window_days=window_days, rv_lookback=rv_lookback))
    iv_df = pd.DataFrame(rows)

    df = rank_df.merge(iv_df, left_on="Ticker", right_on="ticker", how="left").drop(columns=["ticker"])
    df["IV Gauge (Z)"] = df["Z Score"].apply(gauge_display)
    df["IV Gauge (Robust Z)"] = df["Robust Z Score"].apply(gauge_display)

    show = [
        "Ticker",
        "RS Rank",
        "IV Gauge (Z)",
        "IV Gauge (Robust Z)",
        "Days to ER",
        "IV Proxy",
        "Z Score",
        "Robust Z Score",
    ]
    return df[show]


def build_strongest_weakest_tables(
    rs_ranks: pd.DataFrame,
    prices: pd.DataFrame,
    strongest_n: int,
    weakest_n: int,
    iv_window_days: int,
    iv_rv_lookback: int,
):
    as_of_date = rs_ranks.dropna(how="all").index.max()
    day = rs_ranks.loc[as_of_date].dropna().sort_values()

    weakest = pd.DataFrame({"Ticker": day.head(weakest_n).index, "RS Rank": day.head(weakest_n).values})
    strongest_slice = day.tail(strongest_n).sort_values(ascending=False)
    strongest = pd.DataFrame({"Ticker": strongest_slice.index, "RS Rank": strongest_slice.values})

    weakest = enrich_rank_table_with_iv(weakest, prices, window_days=iv_window_days, rv_lookback=iv_rv_lookback)
    strongest = enrich_rank_table_with_iv(strongest, prices, window_days=iv_window_days, rv_lookback=iv_rv_lookback)
    return as_of_date, strongest, weakest


def build_strategy_package(cfg: Config):
    market = download_market_data(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        min_price=cfg.min_price,
        max_missing_pct=cfg.max_missing_pct,
    )

    spy_close = market["spy_close"]
    spy_returns = market["spy_returns"]
    prices = market["prices"]
    stock_returns = market["stock_returns"]

    spy_skewness = calculate_rolling_skewness(spy_returns, cfg.skewness_window)
    spy_regime, lower_thr, upper_thr = classify_regimes_expanding_window(
        spy_skewness, cfg.regime_lower_pct, cfg.regime_upper_pct, cfg.min_regime_history
    )

    rs_scores = calculate_relative_strength(prices, cfg.rs_lookbacks)
    rs_ranks = rs_scores.rank(axis=1, pct=True, na_option="keep") * 100

    crs_returns, holdings = create_regime_crs_portfolio(
        rs_ranks=rs_ranks,
        regime=spy_regime,
        stock_returns=stock_returns,
        top_decile=cfg.top_decile,
        bottom_decile=cfg.bottom_decile,
        rebalance_freq=cfg.rebalance_freq,
    )

    common = crs_returns.index.intersection(spy_returns.index)
    crs_aligned = crs_returns.loc[common]
    spy_aligned = spy_returns.loc[common]

    strongest_date, strongest_50, weakest_50 = build_strongest_weakest_tables(
        rs_ranks=rs_ranks,
        prices=prices,
        strongest_n=cfg.strongest_n,
        weakest_n=cfg.weakest_n,
        iv_window_days=cfg.iv_window_days,
        iv_rv_lookback=cfg.iv_rv_lookback,
    )

    return {
        "spy_close": spy_close,
        "spy_returns": spy_returns,
        "spy_skewness": spy_skewness,
        "spy_regime": spy_regime,
        "lower_thr": lower_thr,
        "upper_thr": upper_thr,
        "prices": prices,
        "stock_returns": stock_returns,
        "rs_ranks": rs_ranks,
        "crs_returns": crs_aligned,
        "spy_aligned": spy_aligned,
        "holdings": holdings,
        "strongest_date": strongest_date,
        "strongest_50": strongest_50,
        "weakest_50": weakest_50,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📈 Regime CRS Screener")
st.caption(
    "Built from your notebook's regime + comparative relative strength workflow. "
    "The IV columns below are RV-based volatility proxies, not true options-implied volatility."
)

with st.sidebar:
    st.header("Settings")

    start_date = st.date_input("Start date", pd.to_datetime(DEFAULT_CFG.start_date))
    end_date = st.date_input("End date", pd.to_datetime(DEFAULT_CFG.end_date))
    analysis_start = st.date_input("Analysis start", pd.to_datetime(DEFAULT_CFG.analysis_start))

    skewness_window = st.slider("Skewness window", 21, 126, DEFAULT_CFG.skewness_window, step=21)
    min_regime_history = st.slider("Min regime history", 126, 504, DEFAULT_CFG.min_regime_history, step=21)

    top_decile = st.slider("Top decile threshold", 5, 20, DEFAULT_CFG.top_decile)
    bottom_decile = st.slider("Bottom decile threshold", 5, 20, DEFAULT_CFG.bottom_decile)

    iv_window_days = st.slider("IV scoring window", 126, 504, DEFAULT_CFG.iv_window_days, step=21)
    iv_rv_lookback = st.slider("RV lookback", 10, 63, DEFAULT_CFG.iv_rv_lookback)

    strongest_n = st.slider("Strongest tickers", 10, 100, DEFAULT_CFG.strongest_n, step=10)
    weakest_n = st.slider("Weakest tickers", 10, 100, DEFAULT_CFG.weakest_n, step=10)

    run = st.button("Run analysis", type="primary")

if "run_once" not in st.session_state:
    st.session_state.run_once = True
    run = True

if run:
    cfg = Config(
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        analysis_start=pd.Timestamp(analysis_start).strftime("%Y-%m-%d"),
        skewness_window=skewness_window,
        min_regime_history=min_regime_history,
        top_decile=top_decile,
        bottom_decile=bottom_decile,
        iv_window_days=iv_window_days,
        iv_rv_lookback=iv_rv_lookback,
        strongest_n=strongest_n,
        weakest_n=weakest_n,
    )

    with st.spinner("Downloading market data and building results..."):
        pkg = build_strategy_package(cfg)

    crs = pkg["crs_returns"]
    spy = pkg["spy_aligned"]
    crs = crs[crs.index >= pd.Timestamp(cfg.analysis_start)]
    spy = spy[spy.index >= pd.Timestamp(cfg.analysis_start)]

    crs_metrics = calculate_performance_metrics(crs)
    spy_metrics = calculate_performance_metrics(spy)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy CAGR", f"{crs_metrics['cagr']*100:.2f}%")
    c2.metric("Strategy Sharpe", f"{crs_metrics['sharpe']:.2f}")
    c3.metric("Max Drawdown", f"{crs_metrics['max_drawdown']*100:.2f}%")
    c4.metric("SPY CAGR", f"{spy_metrics['cagr']*100:.2f}%")

    st.subheader("Equity Curves")
    eq = pd.DataFrame({
        "Regime CRS": (1 + crs).cumprod(),
        "SPY": (1 + spy).cumprod(),
    }).dropna()
    st.line_chart(eq)

    st.subheader("Regime Monitor")
    regime_df = pd.DataFrame({
        "Rolling Skewness": pkg["spy_skewness"],
        "Lower Threshold": pkg["lower_thr"],
        "Upper Threshold": pkg["upper_thr"],
    }).dropna()
    st.line_chart(regime_df.tail(750))

    regime_counts = pkg["spy_regime"].dropna().value_counts()
    r1, r2, r3 = st.columns(3)
    r1.metric("Positive skew days", int(regime_counts.get("positive_skew", 0)))
    r2.metric("Neutral days", int(regime_counts.get("neutral", 0)))
    r3.metric("Negative skew days", int(regime_counts.get("negative_skew", 0)))

    st.subheader(f"Strongest {cfg.strongest_n} | as of {pkg['strongest_date'].date()}")
    st.dataframe(
        pkg["strongest_50"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "RS Rank": st.column_config.NumberColumn(format="%.2f"),
            "Days to ER": st.column_config.NumberColumn(format="%d"),
            "IV Proxy": st.column_config.NumberColumn(format="%.2f"),
            "Z Score": st.column_config.NumberColumn(format="%.2f"),
            "Robust Z Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.subheader(f"Weakest {cfg.weakest_n} | as of {pkg['strongest_date'].date()}")
    st.dataframe(
        pkg["weakest_50"],
        use_container_width=True,
        hide_index=True,
        column_config={
            "RS Rank": st.column_config.NumberColumn(format="%.2f"),
            "Days to ER": st.column_config.NumberColumn(format="%d"),
            "IV Proxy": st.column_config.NumberColumn(format="%.2f"),
            "Z Score": st.column_config.NumberColumn(format="%.2f"),
            "Robust Z Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("Notes"):
        st.markdown(
            """
- **IV Gauge (Z):** standard z-score of an RV-based volatility proxy.
- **IV Gauge (Robust Z):** median/MAD version, more stable for fat tails and outliers.
- **Days to ER:** next earnings date when Yahoo returns it.
- These are **not true options IV values**. They are price-history volatility proxies that are easier to scale in a free public Streamlit app.
"""
        )
else:
    st.info("Click **Run analysis** from the sidebar.")
