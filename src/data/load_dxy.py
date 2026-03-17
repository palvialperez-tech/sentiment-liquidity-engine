import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplana columnas MultiIndex de yfinance a nombres simples.
    """
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []

        for col in df.columns:
            # col puede venir como ("Close", "DX-Y.NYB")
            parts = [str(x) for x in col if x is not None and str(x).strip() != ""]
            flat_name = "_".join(parts).lower()
            flat_cols.append(flat_name)

        df.columns = flat_cols
    else:
        df.columns = [str(col).lower() for col in df.columns]

    return df


def _find_column(columns, candidates):
    """
    Busca la primera columna que coincida exacta o parcialmente.
    """
    for candidate in candidates:
        if candidate in columns:
            return candidate

    for col in columns:
        for candidate in candidates:
            if candidate in col:
                return col

    return None


def load_dxy(
    period: str = "10d",
    interval: str = "5m",
    symbol: str = "DX-Y.NYB"
) -> pd.DataFrame:
    """
    Carga datos de DXY desde Yahoo Finance y devuelve columnas estándar:
    datetime, open, high, low, close, volume
    """
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column"
    )

    if df is None or df.empty:
        raise ValueError(f"No se pudieron cargar datos para {symbol}")

    df = df.reset_index()
    df = _flatten_columns(df)

    cols = list(df.columns)

    datetime_col = _find_column(cols, ["datetime", "date"])
    open_col = _find_column(cols, ["open"])
    high_col = _find_column(cols, ["high"])
    low_col = _find_column(cols, ["low"])
    close_col = _find_column(cols, ["close"])
    volume_col = _find_column(cols, ["volume"])

    missing = []
    if datetime_col is None:
        missing.append("datetime")
    if open_col is None:
        missing.append("open")
    if high_col is None:
        missing.append("high")
    if low_col is None:
        missing.append("low")
    if close_col is None:
        missing.append("close")

    if missing:
        raise ValueError(
            f"Faltan columnas esperadas en DXY: {missing}. "
            f"Columnas recibidas: {cols}"
        )

    result = pd.DataFrame({
        "datetime": pd.to_datetime(df[datetime_col], errors="coerce"),
        "open": pd.to_numeric(df[open_col], errors="coerce"),
        "high": pd.to_numeric(df[high_col], errors="coerce"),
        "low": pd.to_numeric(df[low_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
        "volume": pd.to_numeric(df[volume_col], errors="coerce") if volume_col else 0
    })

    result = result.dropna(subset=["datetime", "open", "high", "low", "close"])
    result = result.sort_values("datetime").reset_index(drop=True)

    return result