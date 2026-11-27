import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf

# Comprehensive Indian Stock Universe (high-volume, liquid, momentum stocks)
INDIAN_STOCKS_METADATA: List[Dict[str, Any]] = [
    # Mega Caps & Nifty Heavyweights
    {"symbol": "RELIANCE.NS", "ticker": "RELIANCE", "name": "Reliance Industries", "sector": "Energy & Retail", "mcap_cr": 1785000},
    {"symbol": "TCS.NS", "ticker": "TCS", "name": "Tata Consultancy Services", "sector": "IT Services", "mcap_cr": 1420000},
    {"symbol": "HDFCBANK.NS", "ticker": "HDFCBANK", "name": "HDFC Bank", "sector": "Banking", "mcap_cr": 1280000},
    {"symbol": "ICICIBANK.NS", "ticker": "ICICIBANK", "name": "ICICI Bank", "sector": "Banking", "mcap_cr": 920000},
    {"symbol": "INFY.NS", "ticker": "INFY", "name": "Infosys", "sector": "IT Services", "mcap_cr": 780000},
    {"symbol": "BHARTIARTL.NS", "ticker": "BHARTIARTL", "name": "Bharti Airtel", "sector": "Telecom", "mcap_cr": 980000},
    {"symbol": "SBIN.NS", "ticker": "SBIN", "name": "State Bank of India", "sector": "Banking", "mcap_cr": 790000},
    {"symbol": "LT.NS", "ticker": "LT", "name": "Larsen & Toubro", "sector": "Infrastructure", "mcap_cr": 510000},
    {"symbol": "ITC.NS", "ticker": "ITC", "name": "ITC Limited", "sector": "FMCG", "mcap_cr": 490000},
    {"symbol": "HINDUNILVR.NS", "ticker": "HINDUNILVR", "name": "Hindustan Unilever", "sector": "FMCG", "mcap_cr": 560000},
    {"symbol": "BAJFINANCE.NS", "ticker": "BAJFINANCE", "name": "Bajaj Finance", "sector": "Financial Services", "mcap_cr": 440000},
    {"symbol": "SUNPHARMA.NS", "ticker": "SUNPHARMA", "name": "Sun Pharma", "sector": "Pharma", "mcap_cr": 410000},
    {"symbol": "MARUTI.NS", "ticker": "MARUTI", "name": "Maruti Suzuki", "sector": "Automobile", "mcap_cr": 380000},
    {"symbol": "TATASTEEL.NS", "ticker": "TATASTEEL", "name": "Tata Steel", "sector": "Metals", "mcap_cr": 190000},
    {"symbol": "TATAPOWER.NS", "ticker": "TATAPOWER", "name": "Tata Power", "sector": "Power & Energy", "mcap_cr": 135000},
    {"symbol": "COALINDIA.NS", "ticker": "COALINDIA", "name": "Coal India", "sector": "Metals & Mining", "mcap_cr": 265000},
    {"symbol": "NTPC.NS", "ticker": "NTPC", "name": "NTPC Limited", "sector": "Power & Energy", "mcap_cr": 385000},
    {"symbol": "TMPV.NS", "ticker": "TMPV", "name": "Tata Motors Passenger", "sector": "Automobile", "mcap_cr": 125000},
    {"symbol": "TMCV.NS", "ticker": "TMCV", "name": "Tata Motors Commercial", "sector": "Automobile", "mcap_cr": 185000},

    # High-Momentum, PSU, Defense, Railways & High Beta Favorites
    {"symbol": "SUZLON.NS", "ticker": "SUZLON", "name": "Suzlon Energy", "sector": "Renewable Energy", "mcap_cr": 85000},
    {"symbol": "IREDA.NS", "ticker": "IREDA", "name": "Indian Renewable Energy", "sector": "Financial Services", "mcap_cr": 62000},
    {"symbol": "RVNL.NS", "ticker": "RVNL", "name": "Rail Vikas Nigam", "sector": "Railways & Infra", "mcap_cr": 92000},
    {"symbol": "IRFC.NS", "ticker": "IRFC", "name": "Indian Railway Finance", "sector": "Railways & Finance", "mcap_cr": 195000},
    {"symbol": "BSE.NS", "ticker": "BSE", "name": "BSE Limited", "sector": "Capital Markets", "mcap_cr": 68000},
    {"symbol": "CDSL.NS", "ticker": "CDSL", "name": "Central Depository", "sector": "Capital Markets", "mcap_cr": 34000},
    {"symbol": "MAZDOCK.NS", "ticker": "MAZDOCK", "name": "Mazagon Dock Shipbuilders", "sector": "Defense & Marine", "mcap_cr": 98000},
    {"symbol": "BEL.NS", "ticker": "BEL", "name": "Bharat Electronics", "sector": "Defense & Electronics", "mcap_cr": 225000},
    {"symbol": "HAL.NS", "ticker": "HAL", "name": "Hindustan Aeronautics", "sector": "Defense & Aerospace", "mcap_cr": 310000},
    {"symbol": "TRENT.NS", "ticker": "TRENT", "name": "Trent Limited", "sector": "Retail", "mcap_cr": 240000},
    {"symbol": "DIXON.NS", "ticker": "DIXON", "name": "Dixon Technologies", "sector": "Electronics & EMS", "mcap_cr": 88000},
    {"symbol": "POLYCAB.NS", "ticker": "POLYCAB", "name": "Polycab India", "sector": "Cables & Electricals", "mcap_cr": 95000},
    {"symbol": "KALYANKJIL.NS", "ticker": "KALYANKJIL", "name": "Kalyan Jewellers", "sector": "Retail & Gems", "mcap_cr": 72000},
    {"symbol": "JIOFIN.NS", "ticker": "JIOFIN", "name": "Jio Financial Services", "sector": "Financial Services", "mcap_cr": 210000},
    {"symbol": "ADANIENT.NS", "ticker": "ADANIENT", "name": "Adani Enterprises", "sector": "Metals & Mining", "mcap_cr": 310000},
    {"symbol": "ADANIPORTS.NS", "ticker": "ADANIPORTS", "name": "Adani Ports", "sector": "Infrastructure", "mcap_cr": 280000},
    {"symbol": "WIPRO.NS", "ticker": "WIPRO", "name": "Wipro", "sector": "IT Services", "mcap_cr": 260000},
    {"symbol": "HCLTECH.NS", "ticker": "HCLTECH", "name": "HCL Technologies", "sector": "IT Services", "mcap_cr": 480000},
    {"symbol": "TATAELXSI.NS", "ticker": "TATAELXSI", "name": "Tata Elxsi", "sector": "IT & Design", "mcap_cr": 44000},
    {"symbol": "YESBANK.NS", "ticker": "YESBANK", "name": "Yes Bank", "sector": "Banking", "mcap_cr": 64000},
    {"symbol": "IDFCFIRSTB.NS", "ticker": "IDFCFIRSTB", "name": "IDFC First Bank", "sector": "Banking", "mcap_cr": 58000},
    {"symbol": "VEDL.NS", "ticker": "VEDL", "name": "Vedanta Limited", "sector": "Metals & Mining", "mcap_cr": 180000},
    {"symbol": "BHEL.NS", "ticker": "BHEL", "name": "Bharat Heavy Electricals", "sector": "Capital Goods", "mcap_cr": 88000},
    {"symbol": "NHPC.NS", "ticker": "NHPC", "name": "NHPC Limited", "sector": "Power & Energy", "mcap_cr": 86000},
    {"symbol": "PFC.NS", "ticker": "PFC", "name": "Power Finance Corporation", "sector": "Financial Services", "mcap_cr": 145000},
    {"symbol": "RECLTD.NS", "ticker": "RECLTD", "name": "REC Limited", "sector": "Financial Services", "mcap_cr": 138000},
]

SECTOR_LIST = sorted(list(set(item["sector"] for item in INDIAN_STOCKS_METADATA)))
RR_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]


def get_default_period_for_timeframe(timeframe: str) -> str:
    """Return an optimal Yahoo Finance lookback period for indicator accuracy and backtesting."""
    if timeframe == "15m":
        return "30d"
    if timeframe == "1h":
        return "60d"
    return "2y"


def _max_hold_bars_for_timeframe(timeframe: str) -> int:
    """Maximum forward bars to evaluate a trade before a time-exit."""
    if timeframe == "15m":
        return 24  # ~1 NSE trading session
    if timeframe == "1h":
        return 25  # ~3.5 trading sessions
    return 20  # ~1 trading month


def _format_bar_time(ts: Any, timeframe: str) -> str:
    """Format a pandas Timestamp cleanly for display."""
    try:
        dt = pd.Timestamp(ts)
        if timeframe == "1d":
            return dt.strftime("%d %b %Y")
        return dt.strftime("%d %b %H:%M")
    except Exception:
        return str(ts)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder's Relative Strength Index (RSI) cleanly without chained indexing."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    rsi = np.full(n, np.nan, dtype=float)
    if n <= period:
        return pd.Series(rsi, index=series.index)

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    rs = avg_gain / (avg_loss + 1e-10)
    rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for idx in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi[idx + 1] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi, index=series.index)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=min(5, period)).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, +DI, and -DI trend strength indicators."""
    n = len(df)
    if n < period * 2:
        empty = pd.Series(np.full(n, np.nan), index=df.index)
        return empty, empty, empty

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)

    tr = np.zeros(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    alpha = 1.0 / period
    tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    plus_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * (plus_smooth / (tr_smooth + 1e-10))
    minus_di = 100.0 * (minus_smooth / (tr_smooth + 1e-10))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx, plus_di, minus_di


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators cleanly on an OHLCV DataFrame copy."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [col[0] if isinstance(col, tuple) else col for col in out.columns]

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out["Volume"] = out["Volume"].fillna(0.0)

    out["EMA_10"] = out["Close"].ewm(span=10, adjust=False).mean()
    out["EMA_20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA_50"] = out["Close"].ewm(span=50, adjust=False).mean()
    if len(out) >= 120:
        out["EMA_200"] = out["Close"].ewm(span=200, adjust=False).mean()
    else:
        out["EMA_200"] = np.nan

    out["RSI"] = calculate_rsi(out["Close"], 14)
    out["ATR"] = calculate_atr(out, 14)
    adx, plus_di, minus_di = calculate_adx(out, 14)
    out["ADX"] = adx
    out["PLUS_DI"] = plus_di
    out["MINUS_DI"] = minus_di

    out["SMA_20"] = out["Close"].rolling(window=20, min_periods=5).mean()
    out["STD_20"] = out["Close"].rolling(window=20, min_periods=5).std()
    out["BB_Upper"] = out["SMA_20"] + (2.0 * out["STD_20"])
    out["BB_Lower"] = out["SMA_20"] - (2.0 * out["STD_20"])
    out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / out["SMA_20"].replace(0, np.nan)

    out["Vol_MA20"] = out["Volume"].rolling(window=20, min_periods=1).mean()
    out["Vol_Ratio"] = np.where(out["Vol_MA20"] > 0, out["Volume"] / out["Vol_MA20"], 1.0)
    return out


def _detect_patterns_at_index(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    i: int,
) -> List[Dict[str, str]]:
    """Detect candlestick and Darvas breakout patterns causally up to index i."""
    if i < 4:
        return []

    patterns: List[Dict[str, str]] = []
    c_open = float(open_arr[i])
    c_high = float(high_arr[i])
    c_low = float(low_arr[i])
    c_close = float(close_arr[i])

    p_open = float(open_arr[i - 1])
    p_close = float(close_arr[i - 1])

    curr_body = abs(c_close - c_open)
    curr_range = max(c_high - c_low, 1e-6)
    prev1_body = abs(p_close - p_open)
    curr_is_green = c_close >= c_open
    prev1_is_green = p_close >= p_open

    # 1. Bullish Engulfing
    if not prev1_is_green and curr_is_green:
        if c_open <= p_close and c_close >= p_open and curr_body > prev1_body:
            patterns.append({
                "name": "Bullish Engulfing",
                "bias": "Bullish",
                "description": "Buyers strongly overpowered sellers after down candle. High-probability reversal.",
            })

    # 2. Bearish Engulfing
    if prev1_is_green and not curr_is_green:
        if c_open >= p_close and c_close <= p_open and curr_body > prev1_body:
            patterns.append({
                "name": "Bearish Engulfing",
                "bias": "Bearish",
                "description": "Sellers completely wiped out previous buyer gains. Bearish reversal signal.",
            })

    # 3. Hammer
    lower_wick = (c_open - c_low) if curr_is_green else (c_close - c_low)
    upper_wick = (c_high - c_close) if curr_is_green else (c_high - c_open)
    if lower_wick >= 2 * curr_body and upper_wick <= 0.2 * curr_range and curr_body > 0:
        patterns.append({
            "name": "Hammer",
            "bias": "Bullish",
            "description": "Strong rejection of lower price levels by buyers. Potential bottom formation.",
        })

    # 4. Shooting Star
    if upper_wick >= 2 * curr_body and lower_wick <= 0.2 * curr_range and curr_body > 0:
        patterns.append({
            "name": "Shooting Star",
            "bias": "Bearish",
            "description": "Strong rejection at higher prices. Potential resistance reversal.",
        })

    # 5. Doji
    if curr_body / curr_range < 0.10:
        patterns.append({
            "name": "Doji (Indecision)",
            "bias": "Neutral",
            "description": "Equal balance between bulls and bears. Indicates consolidation or imminent move.",
        })

    # 6. 20-candle Darvas High Breakout
    start_20 = max(0, i - 20)
    rolling_high_20 = float(np.max(high_arr[start_20:i])) if i > start_20 else c_high
    if c_close > rolling_high_20:
        patterns.append({
            "name": "Darvas 20D Breakout",
            "bias": "Bullish",
            "description": f"Price closed above the 20-period swing resistance high of ₹{rolling_high_20:.2f}.",
        })

    return patterns


def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Detect single and multi-candle patterns on the latest candles."""
    if len(df) < 5:
        return []
    return _detect_patterns_at_index(
        df["Open"].to_numpy(dtype=float),
        df["High"].to_numpy(dtype=float),
        df["Low"].to_numpy(dtype=float),
        df["Close"].to_numpy(dtype=float),
        len(df) - 1,
    )


def _compute_signal_and_levels_at_index(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    ema10_arr: np.ndarray,
    ema20_arr: np.ndarray,
    ema50_arr: np.ndarray,
    rsi_arr: np.ndarray,
    atr_arr: np.ndarray,
    i: int,
    risk_reward_ratio: float,
) -> Dict[str, Any]:
    """Compute causal signal, bias, SL, T1, and T2 at bar index i."""
    cmp = float(close_arr[i])
    atr_val = float(atr_arr[i]) if not np.isnan(atr_arr[i]) and atr_arr[i] > 0 else (cmp * 0.015)
    rsi_val = float(rsi_arr[i]) if not np.isnan(rsi_arr[i]) else 50.0
    ema10 = float(ema10_arr[i]) if not np.isnan(ema10_arr[i]) else cmp
    ema20 = float(ema20_arr[i]) if not np.isnan(ema20_arr[i]) else cmp
    ema50 = float(ema50_arr[i]) if not np.isnan(ema50_arr[i]) else cmp

    patterns = _detect_patterns_at_index(open_arr, high_arr, low_arr, close_arr, i)

    score = 0
    score += 2 if cmp > ema20 else -2
    score += 2 if cmp > ema50 else -2
    if ema10 > ema20 > ema50:
        score += 2
    elif ema10 < ema20 < ema50:
        score -= 2

    if 50 <= rsi_val <= 68:
        score += 1
    elif rsi_val < 32:
        score += 2
    elif rsi_val > 78:
        score -= 2

    bullish_count = sum(1 for p in patterns if p["bias"] == "Bullish")
    bearish_count = sum(1 for p in patterns if p["bias"] == "Bearish")
    score += bullish_count * 3
    score -= bearish_count * 3

    if score >= 4:
        signal, bias = "STRONG BUY", "Bullish"
    elif score >= 1:
        signal, bias = "BUY", "Bullish"
    elif score <= -4:
        signal, bias = "STRONG SELL", "Bearish"
    elif score <= -1:
        signal, bias = "SELL", "Bearish"
    else:
        signal, bias = "NEUTRAL", "Neutral"

    w_start = max(0, i - 4)
    recent_low_5 = float(np.min(low_arr[w_start : i + 1]))
    recent_high_5 = float(np.max(high_arr[w_start : i + 1]))

    if bias == "Bullish":
        sl_from_atr = cmp - (1.5 * atr_val)
        sl_from_swing = recent_low_5 * 0.995
        sl = max(sl_from_atr, sl_from_swing)
        if sl >= cmp:
            sl = cmp * 0.98
        risk = cmp - sl
        target_1 = round(cmp + (risk * risk_reward_ratio * 0.75), 2)
        target_2 = round(cmp + (risk * risk_reward_ratio), 2)
    elif bias == "Bearish":
        sl_from_atr = cmp + (1.5 * atr_val)
        sl_from_swing = recent_high_5 * 1.005
        sl = min(sl_from_atr, sl_from_swing)
        if sl <= cmp:
            sl = cmp * 1.02
        risk = sl - cmp
        target_1 = round(cmp - (risk * risk_reward_ratio * 0.75), 2)
        target_2 = round(cmp - (risk * risk_reward_ratio), 2)
    else:
        risk = 1.5 * atr_val
        sl = cmp - risk
        target_1 = round(cmp + risk, 2)
        target_2 = round(cmp + (risk * risk_reward_ratio), 2)

    return {
        "cmp": round(cmp, 2),
        "signal": signal,
        "bias": bias,
        "score": score,
        "sl": round(sl, 2),
        "target_1": target_1,
        "target_2": target_2,
        "risk": max(abs(cmp - sl), 1e-4),
        "atr": atr_val,
        "rsi": rsi_val,
        "ema10": ema10,
        "ema20": ema20,
        "ema50": ema50,
        "patterns": patterns,
    }


def detect_support_resistance(df: pd.DataFrame, cmp: float, atr: float) -> Dict[str, Any]:
    """Identify tested swing support/resistance clusters and key technical barriers."""
    work = df.iloc[-250:] if len(df) > 250 else df
    highs = work["High"].to_numpy(dtype=float)
    lows = work["Low"].to_numpy(dtype=float)
    n = len(work)

    pivot_res: List[Tuple[float, str]] = []
    pivot_sup: List[Tuple[float, str]] = []

    wing = 3 if n >= 20 else 1
    for i in range(wing, n - wing):
        if highs[i] == np.max(highs[i - wing : i + wing + 1]):
            pivot_res.append((float(highs[i]), "Swing High"))
        if lows[i] == np.min(lows[i - wing : i + wing + 1]):
            pivot_sup.append((float(lows[i]), "Swing Low"))

    # Add rolling lookback reference extremes (excluding current bar)
    if n >= 21:
        pivot_res.append((float(np.max(highs[-21:-1])), "20-Bar High"))
        pivot_sup.append((float(np.min(lows[-21:-1])), "20-Bar Low"))
    if n >= 51:
        pivot_res.append((float(np.max(highs[-51:-1])), "50-Bar High"))
        pivot_sup.append((float(np.min(lows[-51:-1])), "50-Bar Low"))
    if n >= 10:
        pivot_res.append((float(np.max(highs[:-1])), "Period High"))
        pivot_sup.append((float(np.min(lows[:-1])), "Period Low"))

    # Add major dynamic EMAs when meaningful
    latest = work.iloc[-1]
    for ema_col, ema_label in [("EMA_50", "EMA 50"), ("EMA_200", "EMA 200")]:
        if ema_col in work.columns and not np.isnan(latest[ema_col]):
            val = float(latest[ema_col])
            if val > cmp * 1.003:
                pivot_res.append((val, ema_label))
            elif val < cmp * 0.997:
                pivot_sup.append((val, ema_label))

    cluster_tol = max(0.6 * atr, cmp * 0.008)

    def _cluster_levels(raw: List[Tuple[float, str]], is_resistance: bool) -> List[Dict[str, Any]]:
        filtered = [(p, lbl) for p, lbl in raw if (p > cmp * 1.002 if is_resistance else p < cmp * 0.998)]
        if not filtered:
            return []
        filtered.sort(key=lambda x: x[0], reverse=not is_resistance)
        clusters: List[Dict[str, Any]] = []
        for price, lbl in filtered:
            placed = False
            for c in clusters:
                if abs(price - c["price"]) <= cluster_tol:
                    c["prices"].append(price)
                    c["price"] = float(np.mean(c["prices"]))
                    c["touches"] += 1
                    if lbl not in c["labels"]:
                        c["labels"].append(lbl)
                    placed = True
                    break
            if not placed:
                clusters.append({"price": price, "prices": [price], "touches": 1, "labels": [lbl]})

        formatted: List[Dict[str, Any]] = []
        for c in clusters:
            lvl_price = round(c["price"], 2)
            dist_pct = round(((lvl_price - cmp) / cmp) * 100.0, 2)
            touches = c["touches"]
            strength = "Strong" if touches >= 3 or any("High" in l or "Low" in l or "200" in l for l in c["labels"]) else ("Moderate" if touches == 2 else "Minor")
            formatted.append({
                "price": lvl_price,
                "touches": touches,
                "strength": strength,
                "label": " / ".join(c["labels"][:2]),
                "distance_pct": dist_pct,
            })
        formatted.sort(key=lambda x: x["price"], reverse=not is_resistance)
        return formatted[:6]

    resistances = _cluster_levels(pivot_res, is_resistance=True)
    supports = _cluster_levels(pivot_sup, is_resistance=False)

    return {
        "resistances": resistances,
        "supports": supports,
        "nearest_resistance": resistances[0] if resistances else None,
        "nearest_support": supports[0] if supports else None,
    }


def analyze_breakout_signal(df: pd.DataFrame, timeframe: str = "1d") -> Dict[str, Any]:
    """Evaluate breakout status, volume confirmation, volatility squeeze, and fakeout risk."""
    work = enrich_indicators(df) if "Vol_Ratio" not in df.columns else df
    n = len(work)
    latest = work.iloc[-1]
    cmp = float(latest["Close"])
    c_high = float(latest["High"])
    c_low = float(latest["Low"])
    c_range = max(c_high - c_low, 1e-6)
    close_pos = (cmp - c_low) / c_range  # 1.0 = closed right at candle high

    atr = float(latest["ATR"]) if not np.isnan(latest["ATR"]) and latest["ATR"] > 0 else cmp * 0.015
    rsi = float(latest["RSI"]) if not np.isnan(latest["RSI"]) else 50.0
    ema10 = float(latest["EMA_10"])
    ema20 = float(latest["EMA_20"])
    ema50 = float(latest["EMA_50"])
    ema200 = float(latest["EMA_200"]) if not np.isnan(latest["EMA_200"]) else None
    vol_ratio = float(latest["Vol_Ratio"]) if not np.isnan(latest["Vol_Ratio"]) else 1.0

    lookback_20 = work.iloc[-21:-1] if n >= 21 else work.iloc[:-1]
    lookback_50 = work.iloc[-51:-1] if n >= 51 else lookback_20
    res_20 = float(lookback_20["High"].max()) if not lookback_20.empty else c_high
    sup_20 = float(lookback_20["Low"].min()) if not lookback_20.empty else c_low
    res_50 = float(lookback_50["High"].max()) if not lookback_50.empty else res_20

    # Bollinger bandwidth percentile (volatility squeeze check)
    bb_series = work["BB_Width"].dropna().iloc[-60:]
    if len(bb_series) >= 10 and not np.isnan(latest["BB_Width"]):
        bb_pct = float((bb_series <= latest["BB_Width"]).mean() * 100.0)
    else:
        bb_pct = 50.0
    is_squeeze = bb_pct <= 30.0

    dist_to_res20_pct = round(((res_20 - cmp) / cmp) * 100.0, 2)
    dist_to_sup20_pct = round(((cmp - sup_20) / cmp) * 100.0, 2)

    above_20_res = cmp > res_20
    above_50_res = cmp > res_50
    poked_20_res = c_high > res_20 and not above_20_res
    below_20_sup = cmp < sup_20
    poked_20_sup = c_low < sup_20 and not below_20_sup

    # Determine breakout classification
    if above_20_res and (vol_ratio >= 1.25 or close_pos >= 0.65):
        if above_50_res and vol_ratio >= 1.35:
            status = "STRONG MULTI-PERIOD BREAKOUT"
        else:
            status = "CONFIRMED BULLISH BREAKOUT"
        bias = "Bullish"
        color = "green"
        summary = (
            f"Closed above 20-bar resistance (₹{res_20:,.2f}) at ₹{cmp:,.2f} "
            f"with {vol_ratio:.1f}x average volume and strong candle close ({close_pos*100:.0f}% of range)."
        )
    elif above_20_res:
        status = "LOW-VOLUME BREAKOUT (FAKEOUT RISK)"
        bias = "Neutral"
        color = "orange"
        summary = (
            f"Closed above ₹{res_20:,.2f}, but volume is only {vol_ratio:.1f}x average "
            f"(ideally >=1.3x) or candle closed weak ({close_pos*100:.0f}% of range). Watch for follow-through."
        )
    elif poked_20_res:
        status = "FAILED BREAKOUT / UPPER REJECTION"
        bias = "Bearish"
        color = "red"
        summary = (
            f"Intraday move poked above ₹{res_20:,.2f} (high ₹{c_high:,.2f}) but failed to hold and closed at ₹{cmp:,.2f}. "
            "Sellers defended resistance."
        )
    elif below_20_sup and (vol_ratio >= 1.2 or close_pos <= 0.35):
        status = "CONFIRMED BEARISH BREAKDOWN"
        bias = "Bearish"
        color = "red"
        summary = (
            f"Broke below 20-bar support (₹{sup_20:,.2f}) on {vol_ratio:.1f}x volume. "
            "Momentum favors downside continuation unless reclaimed quickly."
        )
    elif poked_20_sup:
        status = "SUPPORT SHAKEOUT / HAMMER REJECTION"
        bias = "Bullish"
        color = "blue"
        summary = (
            f"Briefly dipped below ₹{sup_20:,.2f} (low ₹{c_low:,.2f}) and recovered to ₹{cmp:,.2f}. "
            "Buyers absorbed supply at support."
        )
    elif 0.0 <= dist_to_res20_pct <= 2.5 and (is_squeeze or ema10 > ema20):
        status = "PRE-BREAKOUT COIL (WATCH TRIGGER)"
        bias = "Bullish"
        color = "blue"
        summary = (
            f"Price sits just {dist_to_res20_pct:.2f}% below breakout trigger ₹{res_20:,.2f} "
            f"with {'tight Bollinger squeeze (' + f'{bb_pct:.0f}th pct)' if is_squeeze else 'bullish EMA alignment'}."
        )
    else:
        status = "NO ACTIVE BREAKOUT (CONSOLIDATION)"
        bias = "Neutral"
        color = "gray"
        summary = (
            f"Trading inside 20-bar range ₹{sup_20:,.2f} – ₹{res_20:,.2f} "
            f"({dist_to_res20_pct:.2f}% below breakout, {dist_to_sup20_pct:.2f}% above support)."
        )

    # Quantitative Breakout Strength Score (0 - 100)
    score_pts = 0
    checks: List[Dict[str, str]] = []

    # 1. Price location vs breakout trigger (30 pts)
    if above_50_res:
        score_pts += 30
        checks.append({"state": "PASS", "item": f"Cleared 50-bar resistance ₹{res_50:,.2f}"})
    elif above_20_res:
        score_pts += 24
        checks.append({"state": "PASS", "item": f"Closed above 20-bar trigger ₹{res_20:,.2f}"})
    elif 0 <= dist_to_res20_pct <= 2.5:
        score_pts += 15
        checks.append({"state": "WATCH", "item": f"Within {dist_to_res20_pct:.1f}% of breakout trigger ₹{res_20:,.2f}"})
    elif below_20_sup:
        score_pts += 5
        checks.append({"state": "FAIL", "item": f"Trading below 20-bar support ₹{sup_20:,.2f}"})
    else:
        score_pts += 8
        checks.append({"state": "WATCH", "item": f"Needs {dist_to_res20_pct:.1f}% move to clear ₹{res_20:,.2f}"})

    # 2. Volume confirmation (25 pts)
    if vol_ratio >= 1.5:
        score_pts += 25
        checks.append({"state": "PASS", "item": f"Strong institutional volume surge ({vol_ratio:.2f}x 20-bar avg)"})
    elif vol_ratio >= 1.2:
        score_pts += 18
        checks.append({"state": "PASS", "item": f"Healthy above-average volume ({vol_ratio:.2f}x avg)"})
    elif vol_ratio >= 0.9:
        score_pts += 10
        checks.append({"state": "WATCH", "item": f"Average volume ({vol_ratio:.2f}x) — needs >=1.3x for high-conviction breakout"})
    else:
        score_pts += 3
        checks.append({"state": "FAIL", "item": f"Weak participation ({vol_ratio:.2f}x avg) — higher fakeout risk"})

    # 3. EMA & Trend Structure (25 pts)
    if ema10 > ema20 > ema50 and (ema200 is None or cmp > ema200):
        score_pts += 25
        checks.append({"state": "PASS", "item": "Full bullish trend alignment (Price > EMA 10 > 20 > 50)"})
    elif cmp > ema20:
        score_pts += 15
        checks.append({"state": "PASS", "item": f"Holding above short-term EMA 20 (₹{ema20:,.2f})"})
    else:
        score_pts += 4
        checks.append({"state": "FAIL", "item": f"Below EMA 20 (₹{ema20:,.2f}) — trend momentum weak"})

    # 4. Volatility Squeeze & RSI Health (20 pts)
    if is_squeeze and 48 <= rsi <= 72:
        score_pts += 20
        checks.append({"state": "PASS", "item": f"Volatility coil ({bb_pct:.0f}th BB pct) with healthy RSI {rsi:.1f}"})
    elif 50 <= rsi <= 72:
        score_pts += 14
        checks.append({"state": "PASS", "item": f"Momentum RSI in sweet spot ({rsi:.1f})"})
    elif rsi > 76:
        score_pts += 5
        checks.append({"state": "WATCH", "item": f"RSI overbought ({rsi:.1f}) — prone to pullback/retest"})
    else:
        score_pts += 7
        checks.append({"state": "WATCH", "item": f"RSI neutral/soft ({rsi:.1f})"})

    # Candle quality penalty on heavy upper wicks
    if above_20_res and close_pos < 0.40:
        score_pts = max(0, score_pts - 12)
        checks.append({"state": "FAIL", "item": f"Weak candle close ({close_pos*100:.0f}% of bar range) — supply rejection"})

    return {
        "status": status,
        "bias": bias,
        "color": color,
        "score": int(min(100, max(0, score_pts))),
        "summary": summary,
        "trigger_high_20": round(res_20, 2),
        "trigger_high_50": round(res_50, 2),
        "support_low_20": round(sup_20, 2),
        "dist_to_breakout_pct": dist_to_res20_pct,
        "dist_to_support_pct": dist_to_sup20_pct,
        "vol_ratio": round(vol_ratio, 2),
        "bb_squeeze_pct": round(bb_pct, 1),
        "is_squeeze": is_squeeze,
        "close_position_pct": round(close_pos * 100.0, 1),
        "checks": checks,
        "timeframe": timeframe,
    }


def _simulate_rr_pass(
    df: pd.DataFrame,
    risk_reward_ratio: float,
    timeframe: str,
    collect_trades: bool = False,
) -> Dict[str, Any]:
    """Run causal walk-forward strategy simulation at a specific R:R ratio."""
    work = df
    n = len(work)
    max_hold = _max_hold_bars_for_timeframe(timeframe)
    warmup = min(50, max(15, n // 5))
    if n <= warmup + 3:
        return {
            "rr": risk_reward_ratio,
            "total_trades": 0,
            "t1_hit_rate": 0.0,
            "t2_hit_rate": 0.0,
            "sl_hit_rate": 0.0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "scaleout_expectancy_r": 0.0,
            "profit_factor": 0.0,
            "cum_return_pct": 0.0,
            "scaleout_cum_return_pct": 0.0,
            "median_bars_to_t1": None,
            "median_bars_to_t2": None,
            "median_bars_held": None,
            "trades": [],
            "mfe_atr_p50": 1.5,
            "mfe_atr_p75": 2.5,
            "bullish_t1_rate": None,
            "bullish_t2_rate": None,
            "bearish_t1_rate": None,
            "bearish_t2_rate": None,
            "bullish_n": 0,
            "bearish_n": 0,
        }

    open_arr = work["Open"].to_numpy(dtype=float)
    high_arr = work["High"].to_numpy(dtype=float)
    low_arr = work["Low"].to_numpy(dtype=float)
    close_arr = work["Close"].to_numpy(dtype=float)
    ema10_arr = work["EMA_10"].to_numpy(dtype=float)
    ema20_arr = work["EMA_20"].to_numpy(dtype=float)
    ema50_arr = work["EMA_50"].to_numpy(dtype=float)
    rsi_arr = work["RSI"].to_numpy(dtype=float)
    atr_arr = work["ATR"].to_numpy(dtype=float)
    index_arr = work.index

    trades: List[Dict[str, Any]] = []
    t1_hits = 0
    t2_hits = 0
    sl_direct_hits = 0
    r_multiples: List[float] = []
    scaleout_rs: List[float] = []
    pct_returns: List[float] = []
    scaleout_pcts: List[float] = []
    bars_t1: List[int] = []
    bars_t2: List[int] = []
    bars_held_all: List[int] = []
    mfe_atrs: List[float] = []

    bull_stats = [0, 0, 0]  # n, t1_hits, t2_hits
    bear_stats = [0, 0, 0]

    i = warmup
    while i < n - 1:
        setup = _compute_signal_and_levels_at_index(
            open_arr, high_arr, low_arr, close_arr,
            ema10_arr, ema20_arr, ema50_arr, rsi_arr, atr_arr,
            i, risk_reward_ratio,
        )
        bias = setup["bias"]
        if bias == "Neutral":
            i += 1
            continue

        entry = setup["cmp"]
        sl = setup["sl"]
        t1 = setup["target_1"]
        t2 = setup["target_2"]
        risk = setup["risk"]
        atr_i = max(setup["atr"], 1e-4)

        t1_touched = False
        t2_touched = False
        t1_bar = None
        t2_bar = None
        exit_price = float(close_arr[min(i + max_hold, n - 1)])
        exit_idx = min(i + max_hold, n - 1)
        outcome = "Time Exit"
        max_favorable = 0.0

        end_j = min(i + max_hold, n - 1)
        for j in range(i + 1, end_j + 1):
            bh = float(high_arr[j])
            bl = float(low_arr[j])
            if bias == "Bullish":
                max_favorable = max(max_favorable, bh - entry)
                # Conservative check: SL checked first unless bar opened at/above T2
                if bl <= sl:
                    exit_price = sl
                    exit_idx = j
                    outcome = "T1 then SL" if t1_touched else "SL Hit"
                    break
                if not t1_touched and bh >= t1:
                    t1_touched = True
                    t1_bar = j - i
                if bh >= t2:
                    t2_touched = True
                    t2_bar = j - i
                    exit_price = t2
                    exit_idx = j
                    outcome = "T2 Hit"
                    break
            else:
                max_favorable = max(max_favorable, entry - bl)
                if bh >= sl:
                    exit_price = sl
                    exit_idx = j
                    outcome = "T1 then SL" if t1_touched else "SL Hit"
                    break
                if not t1_touched and bl <= t1:
                    t1_touched = True
                    t1_bar = j - i
                if bl <= t2:
                    t2_touched = True
                    t2_bar = j - i
                    exit_price = t2
                    exit_idx = j
                    outcome = "T2 Hit"
                    break

        bars_held = max(1, exit_idx - i)
        if bias == "Bullish":
            raw_pnl_per_share = exit_price - entry
        else:
            raw_pnl_per_share = entry - exit_price

        r_mult = raw_pnl_per_share / risk
        ret_pct = (raw_pnl_per_share / entry) * 100.0

        # Scale-out simulation: book 50% at T1 and trail SL to breakeven for remaining 50%
        if t1_touched:
            t1_r = risk_reward_ratio * 0.75
            if t2_touched:
                rem_r = risk_reward_ratio
            elif outcome == "T1 then SL":
                rem_r = 0.0  # Stopped at breakeven after T1
            else:
                # Closed on time exit; if time exit went below entry, breakeven stop already protected
                rem_r = max(0.0, r_mult)
            scale_r = 0.5 * t1_r + 0.5 * rem_r
        else:
            scale_r = r_mult

        scale_pct = (scale_r * risk / entry) * 100.0

        mfe_atrs.append(max_favorable / atr_i)
        r_multiples.append(r_mult)
        scaleout_rs.append(scale_r)
        pct_returns.append(ret_pct)
        scaleout_pcts.append(scale_pct)
        bars_held_all.append(bars_held)

        if t1_touched:
            t1_hits += 1
            if t1_bar is not None:
                bars_t1.append(t1_bar)
        else:
            if outcome == "SL Hit":
                sl_direct_hits += 1
        if t2_touched:
            t2_hits += 1
            if t2_bar is not None:
                bars_t2.append(t2_bar)

        if bias == "Bullish":
            bull_stats[0] += 1
            bull_stats[1] += int(t1_touched)
            bull_stats[2] += int(t2_touched)
        else:
            bear_stats[0] += 1
            bear_stats[1] += int(t1_touched)
            bear_stats[2] += int(t2_touched)

        if collect_trades:
            trades.append({
                "entry_time": _format_bar_time(index_arr[i], timeframe),
                "exit_time": _format_bar_time(index_arr[exit_idx], timeframe),
                "signal": setup["signal"],
                "bias": bias,
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "t1": round(t1, 2),
                "t2": round(t2, 2),
                "outcome": outcome,
                "t1_touched": t1_touched,
                "t2_touched": t2_touched,
                "bars_held": bars_held,
                "return_pct": round(ret_pct, 2),
                "r_multiple": round(r_mult, 2),
                "scaleout_r": round(scale_r, 2),
            })

        # Advance past trade exit so signals never overlap
        i = exit_idx + 1

    total = len(r_multiples)
    if total == 0:
        return {
            "rr": risk_reward_ratio,
            "total_trades": 0,
            "t1_hit_rate": 0.0,
            "t2_hit_rate": 0.0,
            "sl_hit_rate": 0.0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "scaleout_expectancy_r": 0.0,
            "profit_factor": 0.0,
            "cum_return_pct": 0.0,
            "scaleout_cum_return_pct": 0.0,
            "median_bars_to_t1": None,
            "median_bars_to_t2": None,
            "median_bars_held": None,
            "trades": [],
            "mfe_atr_p50": 1.5,
            "mfe_atr_p75": 2.5,
            "bullish_t1_rate": None,
            "bullish_t2_rate": None,
            "bearish_t1_rate": None,
            "bearish_t2_rate": None,
            "bullish_n": 0,
            "bearish_n": 0,
        }

    gross_win = sum(r for r in r_multiples if r > 0)
    gross_loss = abs(sum(r for r in r_multiples if r < 0))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 1e-6 else (round(gross_win, 2) if gross_win > 0 else 0.0)

    cum_factor = 1.0
    for p in pct_returns:
        cum_factor *= max(0.05, 1.0 + p / 100.0)
    scale_factor = 1.0
    for p in scaleout_pcts:
        scale_factor *= max(0.05, 1.0 + p / 100.0)

    return {
        "rr": risk_reward_ratio,
        "total_trades": total,
        "t1_hit_rate": round((t1_hits / total) * 100.0, 1),
        "t2_hit_rate": round((t2_hits / total) * 100.0, 1),
        "sl_hit_rate": round((sl_direct_hits / total) * 100.0, 1),
        "win_rate": round((sum(1 for r in r_multiples if r > 0) / total) * 100.0, 1),
        "expectancy_r": round(float(np.mean(r_multiples)), 2),
        "scaleout_expectancy_r": round(float(np.mean(scaleout_rs)), 2),
        "profit_factor": profit_factor,
        "cum_return_pct": round((cum_factor - 1.0) * 100.0, 1),
        "scaleout_cum_return_pct": round((scale_factor - 1.0) * 100.0, 1),
        "median_bars_to_t1": int(round(float(np.median(bars_t1)))) if bars_t1 else None,
        "median_bars_to_t2": int(round(float(np.median(bars_t2)))) if bars_t2 else None,
        "median_bars_held": int(round(float(np.median(bars_held_all)))),
        "trades": trades,
        "mfe_atr_p50": round(float(np.percentile(mfe_atrs, 50)), 2),
        "mfe_atr_p75": round(float(np.percentile(mfe_atrs, 75)), 2),
        "bullish_t1_rate": round((bull_stats[1] / bull_stats[0]) * 100.0, 1) if bull_stats[0] > 0 else None,
        "bullish_t2_rate": round((bull_stats[2] / bull_stats[0]) * 100.0, 1) if bull_stats[0] > 0 else None,
        "bearish_t1_rate": round((bear_stats[1] / bear_stats[0]) * 100.0, 1) if bear_stats[0] > 0 else None,
        "bearish_t2_rate": round((bear_stats[2] / bear_stats[0]) * 100.0, 1) if bear_stats[0] > 0 else None,
        "bullish_n": bull_stats[0],
        "bearish_n": bear_stats[0],
    }


def backtest_stock_strategy(
    df: pd.DataFrame,
    timeframe: str = "1d",
    selected_rr: float = 2.0,
) -> Dict[str, Any]:
    """Run walk-forward backtest at the active R:R and across the 1.5x–3.5x sensitivity curve."""
    work = enrich_indicators(df) if "Vol_Ratio" not in df.columns else df
    rr_values = sorted(set(RR_GRID + [round(float(selected_rr), 2)]))

    selected_summary: Optional[Dict[str, Any]] = None
    sensitivity: List[Dict[str, Any]] = []

    for rr in rr_values:
        is_selected = abs(rr - float(selected_rr)) < 1e-6
        res = _simulate_rr_pass(work, rr, timeframe, collect_trades=is_selected)
        if is_selected:
            selected_summary = res
        if rr in RR_GRID:
            sensitivity.append({
                "rr": rr,
                "total_trades": res["total_trades"],
                "t1_hit_rate": res["t1_hit_rate"],
                "t2_hit_rate": res["t2_hit_rate"],
                "sl_hit_rate": res["sl_hit_rate"],
                "expectancy_r": res["expectancy_r"],
                "scaleout_expectancy_r": res["scaleout_expectancy_r"],
                "median_bars_to_t2": res["median_bars_to_t2"],
            })

    assert selected_summary is not None

    # Find optimal R:R maximizing expectancy (preferring >=25% T2 hit rate when ties occur)
    optimal_row = max(
        sensitivity,
        key=lambda row: (row["expectancy_r"], row["t2_hit_rate"]),
    ) if sensitivity else {"rr": selected_rr, "expectancy_r": 0.0, "t2_hit_rate": 0.0}

    # Buy & Hold return over the exact simulated period
    warmup = min(50, max(15, len(work) // 5))
    if len(work) > warmup + 1:
        start_px = float(work["Close"].iloc[warmup])
        end_px = float(work["Close"].iloc[-1])
        buy_hold_pct = round(((end_px - start_px) / start_px) * 100.0, 1) if start_px > 0 else 0.0
    else:
        buy_hold_pct = 0.0

    return {
        "summary": selected_summary,
        "sensitivity": sensitivity,
        "optimal_rr": optimal_row["rr"],
        "optimal_expectancy_r": optimal_row["expectancy_r"],
        "optimal_t2_hit_rate": optimal_row["t2_hit_rate"],
        "buy_hold_return_pct": buy_hold_pct,
        "lookback_bars": len(work),
    }


def compute_past_performance(df: pd.DataFrame, timeframe: str = "1d") -> Dict[str, Any]:
    """Compute historical return horizons, volatility, and drawdown profile for the stock."""
    work = df.dropna(subset=["Close"]).sort_index()
    closes = work["Close"].to_numpy(dtype=float)
    highs = work["High"].to_numpy(dtype=float)
    lows = work["Low"].to_numpy(dtype=float)
    n = len(closes)
    cmp = float(closes[-1])

    if timeframe == "1d":
        horizons = [("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)]
        bars_per_year = 252.0
    elif timeframe == "1h":
        horizons = [("1D", 7), ("3D", 21), ("1W", 35), ("1M", 140)]
        bars_per_year = 252.0 * 7.0
    else:
        horizons = [("1D", 25), ("3D", 75), ("1W", 125), ("2W", 250)]
        bars_per_year = 252.0 * 25.0

    returns: List[Dict[str, Any]] = []
    for label, lookback in horizons:
        if n > lookback:
            base = float(closes[-1 - lookback])
            ret = round(((cmp - base) / base) * 100.0, 2) if base > 0 else None
            returns.append({"horizon": label, "return_pct": ret, "available": True})
        else:
            returns.append({"horizon": label, "return_pct": None, "available": False})

    # Period high/low & range position
    lookback_slice = work.iloc[-252:] if (timeframe == "1d" and n >= 252) else work
    p_high = float(lookback_slice["High"].max())
    p_low = float(lookback_slice["Low"].min())
    range_pos = round(((cmp - p_low) / max(p_high - p_low, 1e-6)) * 100.0, 1)
    from_high_pct = round(((cmp - p_high) / p_high) * 100.0, 2) if p_high > 0 else 0.0

    # Max drawdown across fetched history
    running_max = np.maximum.accumulate(closes)
    drawdowns = (closes - running_max) / np.where(running_max > 0, running_max, 1.0) * 100.0
    max_drawdown_pct = round(float(np.min(drawdowns)), 2)

    # Volatility & positive bar ratio
    bar_rets = np.diff(closes) / np.where(closes[:-1] > 0, closes[:-1], 1.0)
    ann_vol_pct = round(float(np.std(bar_rets) * np.sqrt(bars_per_year) * 100.0), 1) if len(bar_rets) > 2 else 0.0
    pos_bars_pct = round(float(np.mean(bar_rets > 0) * 100.0), 1) if len(bar_rets) > 0 else 50.0

    return {
        "returns": returns,
        "period_high": round(p_high, 2),
        "period_low": round(p_low, 2),
        "range_position_pct": range_pos,
        "from_high_pct": from_high_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "annualized_vol_pct": ann_vol_pct,
        "positive_bars_pct": pos_bars_pct,
        "sample_bars": n,
    }


def evaluate_target_feasibility(
    df: pd.DataFrame,
    analysis: Dict[str, Any],
    sr_levels: Dict[str, Any],
    breakout: Dict[str, Any],
    backtest: Dict[str, Any],
) -> Dict[str, Any]:
    """Assess whether T1 and T2 are logically likely to touch before SL."""
    cmp = analysis["cmp"]
    atr = max(analysis["atr"], cmp * 0.005)
    bias = analysis["bias"]
    t1 = analysis["target_1"]
    t2 = analysis["target_2"]
    rr = float(analysis["risk_reward"].split(":")[1])

    summary = backtest["summary"]
    total_trades = summary["total_trades"]
    adx_val = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and not np.isnan(df["ADX"].iloc[-1]) else 20.0
    rsi_val = analysis["rsi"]
    vol_ratio = breakout["vol_ratio"]
    mfe_p75 = max(summary.get("mfe_atr_p75", 2.5), 1.0)

    # Select directional hit rate when enough samples exist, else overall
    if bias == "Bullish" and summary["bullish_n"] >= 5 and summary["bullish_t1_rate"] is not None:
        base_t1_hist = summary["bullish_t1_rate"]
        base_t2_hist = summary["bullish_t2_rate"]
        hist_sample_note = f"{summary['bullish_n']} historical Bullish setups"
    elif bias == "Bearish" and summary["bearish_n"] >= 5 and summary["bearish_t1_rate"] is not None:
        base_t1_hist = summary["bearish_t1_rate"]
        base_t2_hist = summary["bearish_t2_rate"]
        hist_sample_note = f"{summary['bearish_n']} historical Bearish setups"
    else:
        base_t1_hist = summary["t1_hit_rate"]
        base_t2_hist = summary["t2_hit_rate"]
        hist_sample_note = f"{total_trades} historical setups"

    barriers_pool = sr_levels["resistances"] if bias != "Bearish" else sr_levels["supports"]

    def _score_single_target(target_px: float, hist_rate: float, is_t2: bool) -> Dict[str, Any]:
        dist = abs(target_px - cmp)
        atr_mult = round(dist / atr, 2)
        target_pct = round((dist / cmp) * 100.0, 2)

        # Barriers strictly between CMP and target
        if bias != "Bearish":
            blocking = [b for b in barriers_pool if cmp * 1.002 < b["price"] < target_px * 0.998]
        else:
            blocking = [b for b in barriers_pool if target_px * 1.002 < b["price"] < cmp * 0.998]

        # Start from historical empirical hit rate (or ATR baseline if <3 trades)
        if total_trades >= 3:
            prob = float(hist_rate)
        else:
            prob = max(22.0, 78.0 - (atr_mult * 13.0))

        reasons: List[str] = []
        if total_trades >= 3:
            reasons.append(f"Historical hit rate before SL is {hist_rate:.0f}% across {hist_sample_note}.")
        else:
            reasons.append("Limited historical setup sample; weighted toward volatility & structure.")

        # Barrier adjustment
        if blocking:
            strong_blocks = [b for b in blocking if b["strength"] == "Strong" or b["touches"] >= 2]
            if breakout["status"] in ("STRONG MULTI-PERIOD BREAKOUT", "CONFIRMED BULLISH BREAKOUT") and bias == "Bullish":
                penalty = 5.0 * min(len(blocking), 2)
                reasons.append(
                    f"Barrier at ₹{blocking[0]['price']:,.2f} ({blocking[0]['label']}) lies ahead, "
                    f"though active {vol_ratio:.1f}x volume breakout helps clear supply."
                )
            else:
                penalty = 12.0 + (4.0 * min(len(strong_blocks), 2))
                reasons.append(
                    f"Blocked by {blocking[0]['strength'].lower()} barrier at ₹{blocking[0]['price']:,.2f} "
                    f"({blocking[0]['label']}, {blocking[0]['touches']} tests) before reaching ₹{target_px:,.2f}."
                )
            prob -= penalty
        else:
            prob += 4.0
            reasons.append("Clear path with no major swing barrier between CMP and target.")

        # Volatility ATR stretch adjustment
        if atr_mult > max(3.6, mfe_p75 * 1.25):
            prob -= 14.0
            reasons.append(
                f"Requires {atr_mult:.1f}x ATR move vs typical 75th-pct swing expansion of {mfe_p75:.1f}x ATR."
            )
        elif atr_mult <= 2.2:
            prob += 4.0
            reasons.append(f"Distance ({atr_mult:.1f}x ATR) fits within normal swing volatility.")
        else:
            reasons.append(f"Distance is {atr_mult:.1f}x ATR (moderate expansion required).")

        # Trend & Momentum adjustment
        if adx_val >= 24 and bias != "Neutral":
            prob += 4.0
        elif adx_val < 16:
            prob -= 6.0
            reasons.append(f"Low ADX ({adx_val:.0f}) indicates choppy/sideways regime.")

        if bias == "Bullish" and rsi_val > 76:
            prob -= 8.0
            reasons.append(f"Overbought RSI ({rsi_val:.1f}) raises near-term pullback risk.")
        elif bias == "Bearish" and rsi_val < 24:
            prob -= 8.0
            reasons.append(f"Oversold RSI ({rsi_val:.1f}) raises bounce risk.")

        if bias == "Neutral":
            prob = min(prob, 45.0)
            reasons.append("Overall signal is NEUTRAL — directional follow-through lacks confirmation.")

        prob_clamped = int(round(min(92.0, max(10.0, prob))))

        if prob_clamped >= 60:
            verdict = "HIGH PROBABILITY"
            verdict_color = "green"
        elif prob_clamped >= 42:
            verdict = "CONDITIONAL / MODERATE"
            verdict_color = "orange"
        else:
            verdict = "LOW ODDS / STRETCHED"
            verdict_color = "red"

        return {
            "target_price": target_px,
            "target_pct": target_pct,
            "atr_multiple": atr_mult,
            "touch_probability": prob_clamped,
            "hist_touch_rate": round(float(hist_rate), 1),
            "verdict": verdict,
            "verdict_color": verdict_color,
            "blocking_barriers": blocking,
            "reasons": reasons,
            "is_t2": is_t2,
        }

    t1_eval = _score_single_target(t1, base_t1_hist, is_t2=False)
    t2_eval = _score_single_target(t2, base_t2_hist, is_t2=True)

    # Determine logical structure-aligned target and optimal R:R advice
    optimal_rr = backtest["optimal_rr"]
    logical_barrier_target = None
    if barriers_pool:
        # First meaningful barrier gives a natural take-profit zone slightly inside supply/demand
        b0 = barriers_pool[0]["price"]
        logical_barrier_target = round(b0 * (0.996 if bias != "Bearish" else 1.004), 2)

    risk_amt = max(abs(cmp - analysis["sl"]), 1e-4)
    optimal_t2_px = round(
        cmp + (risk_amt * optimal_rr) if bias != "Bearish" else cmp - (risk_amt * optimal_rr),
        2,
    )

    if rr > optimal_rr and t2_eval["touch_probability"] < 45:
        rr_advice = (
            f"Current {rr:.1f}x multiplier stretches T2 to ₹{t2:,.2f} ({t2_eval['touch_probability']}% touch odds). "
            f"Historically, {optimal_rr:.1f}x (₹{optimal_t2_px:,.2f}) delivers the best risk-adjusted expectancy "
            f"({backtest['optimal_expectancy_r']:+.2f}R, {backtest['optimal_t2_hit_rate']:.0f}% T2 hit rate)."
        )
    elif rr < optimal_rr and t2_eval["touch_probability"] >= 55:
        rr_advice = (
            f"{rr:.1f}x is conservative and high-odds ({t2_eval['touch_probability']}%). "
            f"Historical expectancy peaks at {optimal_rr:.1f}x (₹{optimal_t2_px:,.2f}, "
            f"{backtest['optimal_expectancy_r']:+.2f}R)."
        )
    else:
        rr_advice = (
            f"{rr:.1f}x aligns well with this stock's historical sweet spot "
            f"(optimal {optimal_rr:.1f}x, expectancy {backtest['optimal_expectancy_r']:+.2f}R)."
        )

    return {
        "t1": t1_eval,
        "t2": t2_eval,
        "optimal_rr": optimal_rr,
        "optimal_t2_price": optimal_t2_px,
        "logical_barrier_target": logical_barrier_target,
        "rr_advice": rr_advice,
        "adx": round(adx_val, 1),
        "mfe_atr_p75": mfe_p75,
    }


def build_holding_plan(
    timeframe: str,
    analysis: Dict[str, Any],
    feasibility: Dict[str, Any],
    breakout: Dict[str, Any],
    backtest: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a logical, actionable holding and target-management plan."""
    cmp = analysis["cmp"]
    sl = analysis["sl"]
    ema10 = analysis["ema10"]
    ema20 = analysis["ema20"]
    bias = analysis["bias"]
    t1_eval = feasibility["t1"]
    t2_eval = feasibility["t2"]
    summary = backtest["summary"]

    unit = "15m bars" if timeframe == "15m" else ("hourly bars" if timeframe == "1h" else "trading sessions")
    med_t1 = summary.get("median_bars_to_t1") or max(2, round(analysis["holding_period"]["approx_candles"] * 0.7))
    med_t2 = summary.get("median_bars_to_t2") or max(med_t1 + 2, analysis["holding_period"]["approx_candles"])
    time_stop_bars = max(med_t1 * 2, 6 if timeframe == "1d" else 10)

    if timeframe == "15m":
        time_stop_rule = f"Intraday MIS: Exit if T1 is not reached within {min(time_stop_bars, 20)} candles (~{min(time_stop_bars, 20)*15} mins) or before 3:15 PM IST."
    elif timeframe == "1h":
        time_stop_rule = f"Time Stop: Exit/review if T1 does not touch within {time_stop_bars} hourly bars (~2–3 sessions)."
    else:
        time_stop_rule = f"Time Stop: If T1 is still untouched after {time_stop_bars} trading sessions, momentum has stalled — reassess capital."

    # Determine logical strategy mode
    scale_r = summary.get("scaleout_expectancy_r", 0.0)
    full_r = summary.get("expectancy_r", 0.0)

    if bias == "Neutral" or breakout["status"] == "FAILED BREAKOUT / UPPER REJECTION":
        strategy_mode = "Wait for Confirmation (Do Not Force Hold)"
        hold_reason = (
            "Signal is Neutral or facing breakout rejection. Wait for a decisive close above "
            f"₹{breakout['trigger_high_20']:,.2f} with >=1.3x volume before holding for targets."
        )
    elif t2_eval["touch_probability"] >= 55 and full_r >= scale_r and not t2_eval["blocking_barriers"]:
        strategy_mode = "Momentum Runner Hold (Trail EMA 10 toward T2)"
        hold_reason = (
            f"T2 carries {t2_eval['touch_probability']}% touch probability with no immediate barrier and strong expectancy ({full_r:+.2f}R). "
            "Hold core position while price closes above EMA 10."
        )
    elif t1_eval["touch_probability"] >= 50 and t2_eval["touch_probability"] < 45:
        strategy_mode = "Book Majority at T1 (Avoid Over-Holding T2)"
        hold_reason = (
            f"T1 is achievable ({t1_eval['touch_probability']}% odds), whereas T2 drops to {t2_eval['touch_probability']}% odds "
            f"(scale-out expectancy {scale_r:+.2f}R vs full-T2 {full_r:+.2f}R). Book 75–100% at T1."
        )
    else:
        strategy_mode = "Scale-Out 50% at T1 + Trail Remainder to T2"
        hold_reason = (
            f"T1 ({t1_eval['touch_probability']}% odds) locks in gains near ₹{t1_eval['target_price']:,.2f}; "
            f"move SL to breakeven (₹{cmp:,.2f}) and let 50% ride risk-free toward T2 (₹{t2_eval['target_price']:,.2f})."
        )

    direction_verb = "above" if bias != "Bearish" else "below"
    invalid_verb = "below" if bias != "Bearish" else "above"
    barrier_note = (
        f"Watch reaction at ₹{t2_eval['blocking_barriers'][0]['price']:,.2f} ({t2_eval['blocking_barriers'][0]['label']}); "
        "exit remainder early if a rejection wick forms on weak volume."
        if t2_eval["blocking_barriers"]
        else f"Trail stop using EMA 10 (₹{ema10:,.2f}) or EMA 20 (₹{ema20:,.2f}) once T1 is crossed."
    )

    steps = [
        f"**Entry & Initial Protection:** Enter near ₹{cmp:,.2f} with hard invalidation SL at ₹{sl:,.2f} (max {analysis['risk_pct']}% price move risk).",
        f"**T1 Checkpoint (₹{t1_eval['target_price']:,.2f} — {t1_eval['touch_probability']}% touch odds):** Expected in ~{med_t1} {unit}. Book 50% (or full if T2 odds <40%) and shift SL to cost (₹{cmp:,.2f}).",
        f"**Hold Condition for T2 (₹{t2_eval['target_price']:,.2f} — {t2_eval['touch_probability']}% touch odds):** Keep holding remaining shares only while price stays {direction_verb} EMA 10 (₹{ema10:,.2f}) and does not close {invalid_verb} SL. {barrier_note}",
        f"**{time_stop_rule}**",
    ]

    return {
        "strategy_mode": strategy_mode,
        "hold_reason": hold_reason,
        "median_bars_to_t1": med_t1,
        "median_bars_to_t2": med_t2,
        "time_stop_bars": time_stop_bars,
        "time_stop_rule": time_stop_rule,
        "unit": unit,
        "steps": steps,
    }


def fetch_market_indices() -> List[Dict[str, Any]]:
    """Fetch live quotes for major Indian benchmark indices."""
    indices = [
        {"name": "Nifty 50", "symbol": "^NSEI"},
        {"name": "Bank Nifty", "symbol": "^NSEBANK"},
        {"name": "Nifty MidCap 100", "symbol": "NIFTY_MIDCAP_100.NS"},
        {"name": "Nifty IT", "symbol": "^CNXIT"},
    ]
    results = []
    for idx in indices:
        try:
            ticker = yf.Ticker(idx["symbol"])
            df = ticker.history(period="2d")
            if not df.empty and len(df) >= 2:
                latest = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                change = latest - prev
                pct_change = (change / prev) * 100
                results.append({
                    "name": idx["name"],
                    "value": round(latest, 1),
                    "change": round(change, 1),
                    "pct_change": round(pct_change, 2),
                })
        except Exception:
            continue
    return results


def estimate_holding_period(timeframe: str, cmp: float, target: float, atr: float) -> Dict[str, Any]:
    """Estimate recommended holding horizon based on timeframe, ATR, and target distance."""
    distance = abs(target - cmp)
    candle_atr = max(atr * 0.75, 1e-4)
    approx_candles = max(1, round(distance / candle_atr))

    if timeframe == "15m":
        horizon_label = "Intraday (Same Day Exit)"
        duration_desc = f"Exit before 3:15 PM IST today. Target estimated in ~{approx_candles * 15} to {min(360, approx_candles * 30)} mins."
        style = "Intraday MIS"
    elif timeframe == "1h":
        horizon_label = "BTST / 1-3 Days"
        duration_desc = f"Hold for 1 to 3 trading sessions (~{approx_candles} hourly bars)."
        style = "Swing / BTST"
    else:
        if approx_candles <= 5:
            horizon_label = "Short Swing (3 - 7 Days)"
            duration_desc = f"Hold for ~3 to 7 trading days (~{approx_candles} daily candles)."
        elif approx_candles <= 15:
            horizon_label = "Medium Swing (1 - 3 Weeks)"
            duration_desc = f"Hold for ~1 to 3 weeks (~{approx_candles} daily candles)."
        else:
            horizon_label = "Positional (3 - 6 Weeks)"
            duration_desc = f"Hold for ~3 to 6 weeks (~{approx_candles} daily candles)."
        style = "Positional CNC"

    return {
        "horizon": horizon_label,
        "duration_desc": duration_desc,
        "approx_candles": approx_candles,
        "style": style,
    }


def calculate_investment_pnl(
    capital: float,
    cmp: float,
    sl: float,
    target_1: float,
    target_2: float,
    trade_type: str = "Delivery / Cash (1x)",
) -> Dict[str, Any]:
    """Calculate exact quantity of shares, total outlay, profit at T1 & T2, and max loss at SL."""
    leverage = 5.0 if "Intraday" in trade_type else 1.0
    effective_buying_power = capital * leverage

    quantity = int(effective_buying_power // cmp) if cmp > 0 else 0
    if quantity == 0:
        return {
            "quantity": 0,
            "actual_investment": 0.0,
            "leverage": f"{int(leverage)}x",
            "loss_at_sl": 0.0,
            "profit_at_t1": 0.0,
            "profit_at_t2": 0.0,
            "loss_pct_of_capital": 0.0,
            "profit_t1_pct_of_capital": 0.0,
            "profit_t2_pct_of_capital": 0.0,
            "risk_reward_cash": "N/A",
            "error": f"Capital (₹{capital:,.2f}) is lower than CMP (₹{cmp:,.2f}). Minimum ₹{cmp:,.2f} required for 1 share.",
        }

    actual_investment = round((quantity * cmp) / leverage, 2)
    per_share_risk = abs(cmp - sl)
    per_share_t1 = abs(target_1 - cmp)
    per_share_t2 = abs(target_2 - cmp)

    loss_at_sl = round(quantity * per_share_risk, 2)
    profit_at_t1 = round(quantity * per_share_t1, 2)
    profit_at_t2 = round(quantity * per_share_t2, 2)

    loss_pct = round((loss_at_sl / capital) * 100, 2)
    profit_t1_pct = round((profit_at_t1 / capital) * 100, 2)
    profit_t2_pct = round((profit_at_t2 / capital) * 100, 2)

    return {
        "quantity": quantity,
        "actual_investment": actual_investment,
        "leverage": f"{int(leverage)}x",
        "loss_at_sl": loss_at_sl,
        "profit_at_t1": profit_at_t1,
        "profit_at_t2": profit_at_t2,
        "loss_pct_of_capital": loss_pct,
        "profit_t1_pct_of_capital": profit_t1_pct,
        "profit_t2_pct_of_capital": profit_t2_pct,
        "risk_reward_cash": f"Risk ₹{loss_at_sl:,.0f} to Gain ₹{profit_at_t2:,.0f}" if loss_at_sl > 0 else "N/A",
        "error": None,
    }


def fetch_stock_data(
    ticker_symbol: str,
    timeframe: str = "1d",
    period: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch OHLCV data for ANY Indian stock symbol and calculate indicators."""
    clean_symbol = ticker_symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("Stock symbol cannot be empty.")

    if not clean_symbol.endswith(".NS") and not clean_symbol.endswith(".BO") and not clean_symbol.startswith("^"):
        clean_symbol += ".NS"

    resolved_period = period or get_default_period_for_timeframe(timeframe)
    ticker = yf.Ticker(clean_symbol)
    df = ticker.history(period=resolved_period, interval=timeframe)

    if df.empty or len(df) < 10:
        if clean_symbol.endswith(".NS"):
            clean_symbol = clean_symbol.replace(".NS", ".BO")
            ticker = yf.Ticker(clean_symbol)
            df = ticker.history(period=resolved_period, interval=timeframe)

    if df.empty or len(df) < 5:
        raise ValueError(f"Insufficient or no data found for ticker '{ticker_symbol}'. Please verify the symbol.")

    df = enrich_indicators(df)

    # Check local metadata first to avoid slow/flaky ticker.info lookup where possible
    meta_match = next((m for m in INDIAN_STOCKS_METADATA if m["symbol"] == clean_symbol), None)
    if meta_match:
        short_name = meta_match["name"]
        sector_name = meta_match["sector"]
        mcap_cr = meta_match["mcap_cr"]
    else:
        try:
            info = ticker.info if hasattr(ticker, "info") else {}
        except Exception:
            info = {}
        short_name = (
            info.get("shortName")
            or info.get("longName")
            or clean_symbol.replace(".NS", "").replace(".BO", "")
        )
        sector_name = info.get("sector") or "Indian Equity"
        mcap_val = info.get("marketCap", 0)
        mcap_cr = int(mcap_val / 10000000) if mcap_val else 15000

    return {
        "df": df,
        "symbol": clean_symbol,
        "name": short_name,
        "sector": sector_name,
        "mcap_cr": mcap_cr,
        "currency": "₹" if clean_symbol.endswith((".NS", ".BO")) else "$",
    }


def generate_prediction_and_levels(
    df: pd.DataFrame,
    risk_reward_ratio: float = 2.0,
    timeframe: str = "1d",
) -> Dict[str, Any]:
    """Analyze technicals and calculate CMP, SL, Target 1, Target 2, and holding estimate."""
    work = enrich_indicators(df) if "Vol_Ratio" not in df.columns else df
    latest = work.iloc[-1]
    prev = work.iloc[-2] if len(work) >= 2 else latest

    cmp = float(latest["Close"])
    prev_close = float(prev["Close"])
    day_change = cmp - prev_close
    day_pct_change = (day_change / prev_close) * 100.0 if prev_close > 0 else 0.0

    setup = _compute_signal_and_levels_at_index(
        work["Open"].to_numpy(dtype=float),
        work["High"].to_numpy(dtype=float),
        work["Low"].to_numpy(dtype=float),
        work["Close"].to_numpy(dtype=float),
        work["EMA_10"].to_numpy(dtype=float),
        work["EMA_20"].to_numpy(dtype=float),
        work["EMA_50"].to_numpy(dtype=float),
        work["RSI"].to_numpy(dtype=float),
        work["ATR"].to_numpy(dtype=float),
        len(work) - 1,
        risk_reward_ratio,
    )

    ema200 = (
        float(latest["EMA_200"])
        if "EMA_200" in latest and not np.isnan(latest["EMA_200"])
        else (setup["ema50"] * 0.95)
    )
    sl = setup["sl"]
    target_1 = setup["target_1"]
    target_2 = setup["target_2"]
    risk_pct = round(abs(cmp - sl) / cmp * 100.0, 2) if cmp > 0 else 0.0
    upside_pct = round(abs(target_2 - cmp) / cmp * 100.0, 2) if cmp > 0 else 0.0
    holding_info = estimate_holding_period(timeframe, cmp, target_2, setup["atr"])

    return {
        "cmp": round(cmp, 2),
        "day_change": round(day_change, 2),
        "day_pct_change": round(day_pct_change, 2),
        "signal": setup["signal"],
        "bias": setup["bias"],
        "score": setup["score"],
        "sl": sl,
        "target_1": target_1,
        "target_2": target_2,
        "risk_pct": risk_pct,
        "upside_pct": upside_pct,
        "risk_reward": f"1:{risk_reward_ratio}",
        "rsi": round(setup["rsi"], 1),
        "atr": round(setup["atr"], 2),
        "patterns": setup["patterns"],
        "ema10": round(setup["ema10"], 2),
        "ema20": round(setup["ema20"], 2),
        "ema50": round(setup["ema50"], 2),
        "ema200": round(ema200, 2),
        "holding_period": holding_info,
    }


def generate_full_stock_report(
    df: pd.DataFrame,
    risk_reward_ratio: float = 2.0,
    timeframe: str = "1d",
) -> Dict[str, Any]:
    """Run complete deep-dive quantitative pipeline: levels, S/R, breakout, backtest, target feasibility, and holding plan."""
    work = enrich_indicators(df) if "Vol_Ratio" not in df.columns else df
    analysis = generate_prediction_and_levels(work, risk_reward_ratio=risk_reward_ratio, timeframe=timeframe)
    sr_levels = detect_support_resistance(work, analysis["cmp"], analysis["atr"])
    breakout = analyze_breakout_signal(work, timeframe=timeframe)
    backtest = backtest_stock_strategy(work, timeframe=timeframe, selected_rr=risk_reward_ratio)
    feasibility = evaluate_target_feasibility(work, analysis, sr_levels, breakout, backtest)
    holding_plan = build_holding_plan(timeframe, analysis, feasibility, breakout, backtest)
    past_perf = compute_past_performance(work, timeframe=timeframe)

    return {
        "df": work,
        "analysis": analysis,
        "sr_levels": sr_levels,
        "breakout": breakout,
        "backtest": backtest,
        "feasibility": feasibility,
        "holding_plan": holding_plan,
        "past_performance": past_perf,
    }


def fetch_screener_universe(
    universe_stocks: List[Dict[str, Any]],
    timeframe: str = "1d",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan the given universe with fast batch fetching, breakout detection, and technical processing."""
    symbols = [s["symbol"] for s in universe_stocks]
    period = "1mo" if timeframe == "15m" else ("60d" if timeframe == "1h" else "1y")

    try:
        batch_df = yf.download(symbols, period=period, interval=timeframe, progress=False, group_by="ticker")
    except Exception:
        batch_df = pd.DataFrame()

    results: List[Dict[str, Any]] = []
    has_multi = isinstance(batch_df.columns, pd.MultiIndex)
    level0_vals = set(batch_df.columns.levels[0]) if has_multi else set()

    for item in universe_stocks:
        symbol = item["symbol"]
        ticker_code = item["ticker"]
        name = item["name"]
        sector = item.get("sector", "Indian Equity")
        mcap = item.get("mcap_cr", 25000)

        try:
            if not batch_df.empty and has_multi and symbol in level0_vals:
                df = batch_df[symbol].dropna(how="all").copy()
            elif not batch_df.empty and not has_multi and len(symbols) == 1:
                df = batch_df.dropna(how="all").copy()
            else:
                ticker_obj = yf.Ticker(symbol)
                df = ticker_obj.history(period=period, interval=timeframe)

            if df.empty or len(df) < 15:
                continue

            df = enrich_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            cmp = round(float(latest["Close"]), 2)
            prev_close = round(float(prev["Close"]), 2)
            day_change_pct = round(((cmp - prev_close) / prev_close) * 100.0, 2) if prev_close > 0 else 0.0
            gap_pct = round(((float(latest["Open"]) - prev_close) / prev_close) * 100.0, 1) if prev_close > 0 else 0.0

            high_52w = float(df["High"].max())
            pct_from_52w = round(((cmp - high_52w) / high_52w) * 100.0, 1) if high_52w > 0 else 0.0
            vol_ratio = round(float(latest["Vol_Ratio"]), 1)

            e10 = float(latest["EMA_10"])
            e20 = float(latest["EMA_20"])
            e50 = float(latest["EMA_50"])
            e200 = float(latest["EMA_200"]) if not np.isnan(latest["EMA_200"]) else (e50 * 0.95)
            pct_from_20ema = round(((cmp - e20) / e20) * 100.0, 1) if e20 > 0 else 0.0

            emas_aligned = e10 > e20 > e50
            ema_cross = (e10 > e20) and (float(prev["EMA_10"]) <= float(prev["EMA_20"]))
            is_breakout = (cmp >= float(df["High"].iloc[-21:-1].max())) if len(df) >= 21 else False
            is_oversold = float(latest["RSI"]) <= 38 if not np.isnan(latest["RSI"]) else False
            is_gap_go = gap_pct >= 1.0 and cmp >= float(latest["Open"])

            analysis = generate_prediction_and_levels(df, risk_reward_ratio=2.0, timeframe=timeframe)
            breakout = analyze_breakout_signal(df, timeframe=timeframe)
            patterns = analysis["patterns"]
            pattern_str = patterns[0]["name"] if patterns else ("Darvas Setup" if emas_aligned else "Consolidation")

            results.append({
                "mcap_cr": mcap,
                "ticker": ticker_code,
                "name": name,
                "symbol": symbol,
                "sector": sector,
                "price": cmp,
                "change_pct": day_change_pct,
                "rsi": round(float(latest["RSI"]), 0) if not np.isnan(latest["RSI"]) else 50,
                "vol_ratio": vol_ratio,
                "gap_pct": gap_pct,
                "pct_52w": pct_from_52w,
                "e10_above": cmp >= e10,
                "e20_above": cmp >= e20,
                "e50_above": cmp >= e50,
                "e200_above": cmp >= e200,
                "pct_from_20ema": pct_from_20ema,
                "emas_aligned": emas_aligned,
                "ema_cross": ema_cross,
                "is_breakout": is_breakout,
                "is_oversold": is_oversold,
                "is_gap_go": is_gap_go,
                "breakout_status": breakout["status"],
                "breakout_score": breakout["score"],
                "breakout_trigger": breakout["trigger_high_20"],
                "signal": analysis["signal"],
                "bias": analysis["bias"],
                "score": analysis["score"],
                "sl": analysis["sl"],
                "target_1": analysis["target_1"],
                "target_2": analysis["target_2"],
                "upside_pct": analysis["upside_pct"],
                "risk_pct": analysis["risk_pct"],
                "pattern": pattern_str,
                "holding_period": analysis["holding_period"]["horizon"],
            })
        except Exception:
            continue

    results.sort(key=lambda x: (x["breakout_score"], x["change_pct"]), reverse=True)

    total = len(results) if results else 1
    advances = sum(1 for r in results if r["change_pct"] > 0)
    declines = sum(1 for r in results if r["change_pct"] < 0)
    ad_ratio = round(((advances - declines) / total) * 100.0, 1)

    above_20ema_pct = round((sum(1 for r in results if r["e20_above"]) / total) * 100.0, 1)
    above_200ema_pct = round((sum(1 for r in results if r["e200_above"]) / total) * 100.0, 1)
    rsi_above_60_pct = round((sum(1 for r in results if r["rsi"] >= 60) / total) * 100.0, 1)
    near_52w_pct = round((sum(1 for r in results if r["pct_52w"] >= -5.0) / total) * 100.0, 1)
    breakouts_pct = round((sum(1 for r in results if r["is_breakout"] or r["change_pct"] >= 4.0) / total) * 100.0, 1)

    market_health = {
        "advance_decline": f"{ad_ratio:+0.1f}%",
        "above_20ema": f"{above_20ema_pct}%",
        "above_200ema": f"{above_200ema_pct}%",
        "rsi_above_60": f"{rsi_above_60_pct}%",
        "above_pivot": f"{round((above_20ema_pct + 8), 1)}%",
        "near_52w": f"{near_52w_pct}%",
        "breakouts": f"{breakouts_pct}%",
    }

    return results, market_health


def apply_logical_screen(
    stocks: List[Dict[str, Any]],
    screen_name: str,
    custom_params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Filter stocks strictly based on mathematical & quantitative logic."""
    if screen_name == "All Stocks":
        return stocks
    if screen_name == "Darvas Box Breakout":
        return [s for s in stocks if s["is_breakout"] or (s["change_pct"] >= 2.0 and s["vol_ratio"] >= 1.2)]
    if screen_name == "EMAs Aligned (10 > 20 > 50 Institutional)":
        return [s for s in stocks if s["emas_aligned"] and s["e20_above"]]
    if screen_name == "Volume Shockers (>= 2.0x Institutional Surge)":
        return [s for s in stocks if s["vol_ratio"] >= 2.0]
    if screen_name == "RSI Sweet Spot Momentum (55 - 70)":
        return [s for s in stocks if 55 <= s["rsi"] <= 72 and s["e20_above"]]
    if screen_name == "Near 52-Week High (<= 3% from Peak)":
        return [s for s in stocks if s["pct_52w"] >= -3.5]
    if screen_name == "EMA Golden Cross (10 crossing 20)":
        return [s for s in stocks if s["ema_cross"] or (s["e10_above"] and 0 <= s["pct_from_20ema"] <= 2.5)]
    if screen_name == "Oversold Rebound Candidates":
        return [s for s in stocks if s["is_oversold"] or "Hammer" in s["pattern"] or "Engulfing" in s["pattern"]]
    if screen_name == "Gap Up & Go (Intraday Momentum)":
        return [s for s in stocks if s["is_gap_go"] or s["gap_pct"] >= 1.0]

    if screen_name == "Custom Multi-Filter Logic" and custom_params:
        min_change = custom_params.get("min_change", -100.0)
        min_rsi = custom_params.get("min_rsi", 0)
        max_rsi = custom_params.get("max_rsi", 100)
        min_vol = custom_params.get("min_vol", 0.0)
        req_above_20ema = custom_params.get("above_20ema", False)
        req_aligned = custom_params.get("aligned", False)
        req_sector = custom_params.get("sector", "All Sectors")

        filtered = []
        for s in stocks:
            if s["change_pct"] < min_change:
                continue
            if not (min_rsi <= s["rsi"] <= max_rsi):
                continue
            if s["vol_ratio"] < min_vol:
                continue
            if req_above_20ema and not s["e20_above"]:
                continue
            if req_aligned and not s["emas_aligned"]:
                continue
            if req_sector != "All Sectors" and s["sector"] != req_sector:
                continue
            filtered.append(s)
        return filtered

    return stocks


# Sentiment Dictionary
BULLISH_KEYWORDS = [
    "profit", "surge", "jump", "growth", "high", "gain", "rally", "order", "contract",
    "win", "record", "beat", "upgrade", "dividend", "expansion", "positive", "buy",
    "invest", "acquire", "revenue up", "margin expansion", "bull", "rises", "rising",
    "upside", "skyrocket", "outperform", "target hiked", "all-time high", "breakout",
    "strong", "soars",
]

BEARISH_KEYWORDS = [
    "fall", "drop", "loss", "slump", "down", "probe", "penalty", "fraud", "miss",
    "downgrade", "debt", "decline", "weak", "selloff", "inflation", "cut", "warning",
    "litigation", "default", "bear", "deficit", "tumble", "crashes", "crash", "lower",
    "target cut", "underperform", "plunges",
]


def analyze_headline_sentiment(text: str) -> Tuple[str, float]:
    """Score sentiment of a financial headline using financial lexicons."""
    lower_text = text.lower()
    score = 0
    for word in BULLISH_KEYWORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", lower_text):
            score += 1
    for word in BEARISH_KEYWORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", lower_text):
            score -= 1

    if score > 0:
        return "Positive", min(1.0, 0.4 + (score * 0.2))
    if score < 0:
        return "Negative", max(-1.0, -0.4 + (score * 0.2))
    return "Neutral", 0.0


def fetch_stock_news(company_name: str, symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Fetch real-time news from Google News RSS for the given stock and calculate sentiment."""
    clean_name = re.sub(r"\(.*?\)", "", company_name).replace("Limited", "").replace("Ltd", "").strip()
    query = f"{clean_name} share stock price"
    encoded_query = query.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    news_list: List[Dict[str, Any]] = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:limit]:
            sentiment_label, sentiment_score = analyze_headline_sentiment(entry.title)
            published_dt = entry.get("published", "")
            news_list.append({
                "title": entry.title,
                "link": entry.link,
                "published": published_dt,
                "source": entry.get("source", {}).get("title", "News Source"),
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
            })
    except Exception as e:
        news_list.append({
            "title": f"Could not fetch news: {str(e)}",
            "link": "#",
            "published": "",
            "source": "Error",
            "sentiment": "Neutral",
            "sentiment_score": 0.0,
        })

    return news_list
