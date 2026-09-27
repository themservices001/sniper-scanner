"""
Sniper Core Scanner — replicates Pine Script indicator logic
Runs on GitHub Actions, fetches Binance 1D data, sends Telegram alerts.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── CONFIG ──
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
STATE_FILE = "state.json"
TIMEFRAME = "1d"
CANDLE_LIMIT = 300          # enough for EMA200 + warmup
HTF_EMA_FAST = 50
HTF_EMA_SLOW = 100
MIN_CONFLUENCE = 7
VOL_SPIKE_MULT = 1.5
FLAT_THRESHOLD_PCT = 3.0

# ── BINANCE PUBLIC API ──
BASE_URL = "https://api.binance.com"

def fetch_klines(symbol: str, interval: str = "1d", limit: int = 300):
    """Fetch OHLCV candles from Binance public API (no API key)."""
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ── INDICATOR CALCULATIONS (must match Pine Script) ──

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def adx(df: pd.DataFrame, length: int = 14):
    """Return (adx, plus_di, minus_di) series."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.rolling(length).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(length).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(length).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_s = dx.rolling(length).mean()
    return adx_s, plus_di, minus_di

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ── CONFLUENCE SCORE (mirrors Pine Script) ──

def compute_confluence(df: pd.DataFrame, direction: str) -> int:
    """
    Replicates the 11-factor confluence score from Sniper Core.
    direction: 'LONG' or 'SHORT'
    """
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    ema9 = ema(c, 9)
    ema21 = ema(c, 21)
    ema50 = ema(c, 50)
    ema100 = ema(c, 100)
    rsi7 = rsi(c, 7)
    atr14 = atr(df, 14)
    adx14, _, _ = adx(df, 14)
    macd_line, signal_line, hist = macd(c)
    vol_sma20 = v.rolling(20).mean()

    last = -1
    score = 0

    if direction == "LONG":
        score += 1 if ema9.iloc[last] > ema21.iloc[last] else 0
        score += 1 if c.iloc[last] > ema100.iloc[last] else 0
        score += 1 if c.iloc[last] > ema100.iloc[last] else 0  # HTF proxy (chart EMA100)
        score += 1 if 50 < rsi7.iloc[last] < 85 else 0
        score += 1 if adx14.iloc[last] > 20 else 0
        score += 1 if v.iloc[last] > vol_sma20.iloc[last] * VOL_SPIKE_MULT else 0
        score += 1 if macd_line.iloc[last] > signal_line.iloc[last] else 0
        # Micro-breakout (3-bar)
        micro_high = h.rolling(3).max()
        score += 1 if c.iloc[last] > micro_high.iloc[last - 1] and c.iloc[last] > o.iloc[last] else 0
        # Velocity (ROC2 acceleration)
        roc2 = c.pct_change(2)
        vel = ema(roc2, 3)
        score += 1 if vel.iloc[last] > 0 and vel.iloc[last] > vel.iloc[last - 1] else 0
        # Volume pressure (simplified CLV * vol ratio)
        clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        press = ema(clv * (v / vol_sma20), 3)
        score += 1 if press.iloc[last] > 0.15 else 0
        # Wick rejection (lower wick)
        body = (c - o).abs()
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        score += 1 if lower_wick.iloc[last] > body.iloc[last] * 1.2 and c.iloc[last] > o.iloc[last] else 0

    else:  # SHORT
        score += 1 if ema9.iloc[last] < ema21.iloc[last] else 0
        score += 1 if c.iloc[last] < ema100.iloc[last] else 0
        score += 1 if c.iloc[last] < ema100.iloc[last] else 0
        score += 1 if 15 < rsi7.iloc[last] < 50 else 0
        score += 1 if adx14.iloc[last] > 20 else 0
        score += 1 if v.iloc[last] > vol_sma20.iloc[last] * VOL_SPIKE_MULT else 0
        score += 1 if macd_line.iloc[last] < signal_line.iloc[last] else 0
        micro_low = l.rolling(3).min()
        score += 1 if c.iloc[last] < micro_low.iloc[last - 1] and c.iloc[last] < o.iloc[last] else 0
        roc2 = c.pct_change(2)
        vel = ema(roc2, 3)
        score += 1 if vel.iloc[last] < 0 and vel.iloc[last] < vel.iloc[last - 1] else 0
        clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        press = ema(clv * (v / vol_sma20), 3)
        score += 1 if press.iloc[last] < -0.15 else 0
        body = (c - o).abs()
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        score += 1 if upper_wick.iloc[last] > body.iloc[last] * 1.2 and c.iloc[last] < o.iloc[last] else 0

    return score


# ── SIGNAL LOGIC (mirrors Pine Script) ──

def check_signal(df: pd.DataFrame, symbol: str):
    """Returns dict with signal info, or None if no signal."""
    if len(df) < 210:
        return None

    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    ema9 = ema(c, 9)
    ema21 = ema(c, 21)
    ema50 = ema(c, 50)
    ema100 = ema(c, 100)
    rsi7 = rsi(c, 7)
    atr14 = atr(df, 14)
    adx14, _, _ = adx(df, 14)
    vol_sma20 = v.rolling(20).mean()

    last = -1
    prev = -2

    # ── HTF proxy: use 4H data would require separate fetch.
    # For simplicity, we use EMA100 on the 1D chart as the HTF proxy.
    # The Pine Script's HTF filter uses 4H and D EMAs; here we check
    # price above chart EMA100 as a conservative substitute.
    htf_ok_long = c.iloc[last] > ema100.iloc[last]
    htf_ok_short = c.iloc[last] < ema100.iloc[last]

    # ── Base conditions ──
    base_long = (
        c.iloc[last] > ema100.iloc[last]
        and htf_ok_long
        and 45 < rsi7.iloc[last] < 85
        and v.iloc[last] > vol_sma20.iloc[last] * VOL_SPIKE_MULT
        and ema9.iloc[last] > ema9.iloc[prev]
    )
    base_short = (
        c.iloc[last] < ema100.iloc[last]
        and htf_ok_short
        and 15 < rsi7.iloc[last] < 55
        and v.iloc[last] > vol_sma20.iloc[last] * VOL_SPIKE_MULT
        and ema9.iloc[last] < ema9.iloc[prev]
    )

    # ── Confluence ──
    score_long = compute_confluence(df, "LONG")
    score_short = compute_confluence(df, "SHORT")
    conf_gate_long = score_long >= MIN_CONFLUENCE
    conf_gate_short = score_short >= MIN_CONFLUENCE

    # ── Signal ──
    long_signal = base_long and conf_gate_long
    short_signal = base_short and conf_gate_short

    if not long_signal and not short_signal:
        return None

    direction = "LONG" if long_signal else "SHORT"
    score = score_long if long_signal else score_short
    is_elite = score >= 8

    # ── Adaptive floor ──
    vol_floor = vol_sma20.iloc[last]
    adx_floor = adx14.rolling(20).mean().iloc[last]
    vol_pass = v.iloc[last] >= vol_floor
    adx_pass = adx14.iloc[last] >= adx_floor

    # ── 3-bar trend ──
    flat = FLAT_THRESHOLD_PCT / 100.0
    vol_dir = 1 if v.iloc[last] > v.iloc[last - 3] * (1 + flat) else (-1 if v.iloc[last] < v.iloc[last - 3] * (1 - flat) else 0)
    adx_dir = 1 if adx14.iloc[last] > adx14.iloc[last - 3] * (1 + flat) else (-1 if adx14.iloc[last] < adx14.iloc[last - 3] * (1 - flat) else 0)
    arrow = lambda d: "▲" if d == 1 else ("▼" if d == -1 else "▬")

    return {
        "symbol": symbol,
        "direction": direction,
        "elite": is_elite,
        "score": score,
        "price": float(c.iloc[last]),
        "rsi": round(float(rsi7.iloc[last]), 1),
        "adx": round(float(adx14.iloc[last]), 1),
        "vol_ratio": round(float(v.iloc[last] / vol_sma20.iloc[last]), 2),
        "vol_floor_ok": vol_pass,
        "adx_floor_ok": adx_pass,
        "trend3": f"V{arrow(vol_dir)}│A{arrow(adx_dir)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── TELEGRAM ──

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  Telegram not configured — printing to console.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"  Telegram error: {r.text}")
    except Exception as e:
        print(f"  Telegram exception: {e}")


def format_signal(sig: dict) -> str:
    emoji = "🟢" if sig["direction"] == "LONG" else "🔴"
    elite = " ⚡ ELITE" if sig["elite"] else ""
    floor = "✓" if sig["vol_floor_ok"] and sig["adx_floor_ok"] else "✗"
    return (
        f"{emoji} *{sig['direction']}{elite}* — {sig['symbol']}\n"
        f"Price: `{sig['price']}`\n"
        f"Confluence: `{sig['score']}/11`\n"
        f"RSI: `{sig['rsi']}`  │  ADX: `{sig['adx']}`\n"
        f"Vol Ratio: `{sig['vol_ratio']}x`\n"
        f"Floor: `{floor}`  │  3-Bar: `{sig['trend3']}`"
    )


# ── STATE (avoid duplicate alerts) ──

def load_state():
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── MAIN ──

def main():
    watchlist = []
    try:
        with open("watchlist.txt", "r") as f:
            watchlist = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print("No watchlist.txt found. Exiting.")
        return

    print(f"Scanning {len(watchlist)} symbols on {TIMEFRAME}...")
    state = load_state()
    new_signals = []

    for symbol in watchlist:
        print(f"  → {symbol}")
        df = fetch_klines(symbol, TIMEFRAME, CANDLE_LIMIT)
        if df is None or len(df) < 210:
            continue

        sig = check_signal(df, symbol)
        if sig is None:
            continue

        # Deduplicate: only alert if this is a new bar's signal
        key = f"{symbol}_{sig['direction']}"
        last_bar_time = str(df.index[-1])
        if state.get(key) == last_bar_time:
            continue  # already alerted this bar

        state[key] = last_bar_time
        new_signals.append(sig)

    if new_signals:
        print(f"\n{len(new_signals)} new signal(s):")
        for sig in new_signals:
            msg = format_signal(sig)
            print(msg)
            send_telegram(msg)
    else:
        print("\nNo new signals.")

    save_state(state)


if __name__ == "__main__":
    main()
