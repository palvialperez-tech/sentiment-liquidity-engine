import pandas as pd


def detect_sweep(df: pd.DataFrame, lookback: int = 10) -> dict:
    """
    Detecta un sweep básico usando la última vela contra el rango previo.

    Retorna:
    - side: buy / sell / neutral
    - has_sweep: bool
    - swept_level: nivel barrido
    """
    if df is None or df.empty or len(df) < lookback + 2:
        return {
            "side": "neutral",
            "has_sweep": False,
            "swept_level": None
        }

    recent = df.iloc[-(lookback + 1):-1].copy()
    last = df.iloc[-1]

    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())

    # Barrido de highs => contexto SELL
    if float(last["high"]) > recent_high and float(last["close"]) < recent_high:
        return {
            "side": "sell",
            "has_sweep": True,
            "swept_level": recent_high
        }

    # Barrido de lows => contexto BUY
    if float(last["low"]) < recent_low and float(last["close"]) > recent_low:
        return {
            "side": "buy",
            "has_sweep": True,
            "swept_level": recent_low
        }

    # versión más permisiva si solo rompe, aunque no cierre de vuelta
    if float(last["high"]) > recent_high:
        return {
            "side": "sell",
            "has_sweep": True,
            "swept_level": recent_high
        }

    if float(last["low"]) < recent_low:
        return {
            "side": "buy",
            "has_sweep": True,
            "swept_level": recent_low
        }

    return {
        "side": "neutral",
        "has_sweep": False,
        "swept_level": None
    }