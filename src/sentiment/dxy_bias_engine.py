import pandas as pd


def calculate_dxy_bias(
    dxy_df: pd.DataFrame,
    fast_ma: int = 10,
    slow_ma: int = 30,
    momentum_lookback: int = 5,
    min_separation_pct: float = 0.0005
) -> dict:
    """
    Calcula sesgo del DXY y lo traduce a EUR/USD.

    Retorna un dict con:
    - dxy_bias: bullish / bearish / neutral
    - eurusd_bias: bullish / bearish / neutral
    - momentum
    - ma_fast
    - ma_slow
    - strength
    """

    if dxy_df is None or dxy_df.empty:
        return {
            "dxy_bias": "neutral",
            "eurusd_bias": "neutral",
            "momentum": 0.0,
            "ma_fast": None,
            "ma_slow": None,
            "strength": "weak"
        }

    df = dxy_df.copy()

    if "close" not in df.columns:
        raise ValueError("El DataFrame de DXY debe contener columna 'close'")

    if len(df) < max(slow_ma, momentum_lookback) + 2:
        return {
            "dxy_bias": "neutral",
            "eurusd_bias": "neutral",
            "momentum": 0.0,
            "ma_fast": None,
            "ma_slow": None,
            "strength": "weak"
        }

    df["ma_fast"] = df["close"].rolling(fast_ma).mean()
    df["ma_slow"] = df["close"].rolling(slow_ma).mean()
    df["momentum"] = df["close"] - df["close"].shift(momentum_lookback)

    last = df.iloc[-1]

    ma_fast_val = float(last["ma_fast"]) if pd.notna(last["ma_fast"]) else None
    ma_slow_val = float(last["ma_slow"]) if pd.notna(last["ma_slow"]) else None
    momentum_val = float(last["momentum"]) if pd.notna(last["momentum"]) else 0.0
    close_val = float(last["close"])

    if ma_fast_val is None or ma_slow_val is None or close_val == 0:
        return {
            "dxy_bias": "neutral",
            "eurusd_bias": "neutral",
            "momentum": momentum_val,
            "ma_fast": ma_fast_val,
            "ma_slow": ma_slow_val,
            "strength": "weak"
        }

    separation_pct = abs(ma_fast_val - ma_slow_val) / close_val

    if ma_fast_val > ma_slow_val and momentum_val > 0:
        dxy_bias = "bullish"
    elif ma_fast_val < ma_slow_val and momentum_val < 0:
        dxy_bias = "bearish"
    else:
        dxy_bias = "neutral"

    if separation_pct >= min_separation_pct:
        strength = "strong"
    elif separation_pct >= (min_separation_pct / 2):
        strength = "medium"
    else:
        strength = "weak"

    if dxy_bias == "bullish":
        eurusd_bias = "bearish"
    elif dxy_bias == "bearish":
        eurusd_bias = "bullish"
    else:
        eurusd_bias = "neutral"

    return {
        "dxy_bias": dxy_bias,
        "eurusd_bias": eurusd_bias,
        "momentum": round(momentum_val, 5),
        "ma_fast": round(ma_fast_val, 5),
        "ma_slow": round(ma_slow_val, 5),
        "strength": strength,
        "separation_pct": round(separation_pct, 6)
    }