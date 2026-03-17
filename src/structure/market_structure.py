import pandas as pd


def detect_structure_shift(df: pd.DataFrame, lookback: int = 8) -> dict:
    """
    Detecta un cambio estructural simple usando la última vela.

    BUY:
    - cierre actual rompe el máximo reciente

    SELL:
    - cierre actual rompe el mínimo reciente
    """
    if df is None or df.empty or len(df) < lookback + 2:
        return {
            "side": "neutral",
            "structure_shift": False,
            "broken_level": None
        }

    recent = df.iloc[-(lookback + 1):-1].copy()
    last = df.iloc[-1]

    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())
    last_close = float(last["close"])

    if last_close > recent_high:
        return {
            "side": "buy",
            "structure_shift": True,
            "broken_level": recent_high
        }

    if last_close < recent_low:
        return {
            "side": "sell",
            "structure_shift": True,
            "broken_level": recent_low
        }

    return {
        "side": "neutral",
        "structure_shift": False,
        "broken_level": None
    }