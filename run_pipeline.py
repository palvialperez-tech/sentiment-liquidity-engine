import os
from datetime import datetime

import pandas as pd
import yaml

from src.data.load_dxy import load_dxy
from src.data.load_eurusd import load_eurusd
from src.sentiment.dxy_bias_engine import calculate_dxy_bias
from src.sentiment.confluence_engine import calculate_confluence_score
from src.signals.signal_engine import generate_signal
from src.signals.signal_ranker import rank_signal
from src.liquidity.sweep_detector import detect_sweep
from src.structure.market_structure import detect_structure_shift
from src.liquidity.entry_context import build_entry_context


def load_config(config_path: str = "config/settings.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No existe el archivo de configuración: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_technical_context(eurusd_df: pd.DataFrame) -> dict:
    sweep_result = detect_sweep(eurusd_df, lookback=10)
    structure_result = detect_structure_shift(eurusd_df, lookback=8)

    side = "neutral"

    if sweep_result["side"] == structure_result["side"] and sweep_result["side"] != "neutral":
        side = sweep_result["side"]
    elif sweep_result["side"] != "neutral":
        side = sweep_result["side"]
    elif structure_result["side"] != "neutral":
        side = structure_result["side"]

    return {
        "side": side,
        "has_sweep": sweep_result["has_sweep"],
        "structure_shift": structure_result["structure_shift"],
        "sweep_side": sweep_result["side"],
        "structure_side": structure_result["side"],
        "swept_level": sweep_result["swept_level"],
        "broken_level": structure_result["broken_level"]
    }


def ensure_output_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_signal(signal_result: dict, rank_result: dict, technical_context: dict, entry_context: dict, dxy_context: dict, output_path: str) -> None:
    ensure_output_dir(output_path)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal": signal_result.get("signal"),
        "confidence": signal_result.get("confidence"),
        "rank": rank_result.get("rank"),
        "quality": rank_result.get("quality"),
        "score": signal_result.get("score"),
        "buy_score": signal_result.get("buy_score"),
        "sell_score": signal_result.get("sell_score"),
        "session_active": signal_result.get("session_active"),
        "session_name": signal_result.get("session_name"),
        "timezone": signal_result.get("timezone"),
        "technical_side": technical_context.get("side"),
        "has_sweep": technical_context.get("has_sweep"),
        "structure_shift": technical_context.get("structure_shift"),
        "sweep_side": technical_context.get("sweep_side"),
        "structure_side": technical_context.get("structure_side"),
        "entry_side": entry_context.get("side"),
        "entry_valid": entry_context.get("is_valid"),
        "alignment_score": entry_context.get("alignment_score"),
        "setup_type": entry_context.get("setup_type"),
        "dxy_bias": dxy_context.get("dxy_bias"),
        "eurusd_bias": dxy_context.get("eurusd_bias"),
        "dxy_strength": dxy_context.get("strength"),
        "reasons": " | ".join(signal_result.get("reasons", [])),
        "entry_reasons": " | ".join(entry_context.get("reasons", []))
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(output_path):
        df_old = pd.read_csv(output_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    print("\n==============================")
    print(" SENTIMENT LIQUIDITY ENGINE")
    print("==============================\n")

    config = load_config()

    print("[1/7] Cargando EUR/USD...")
    eurusd_df = load_eurusd(
        period="10d",
        interval=config["timeframes"]["execution"]
    )

    print("[2/7] Cargando DXY...")
    dxy_df = load_dxy(
        period="10d",
        interval=config["timeframes"]["execution"],
        symbol=config["dxy_symbol"]
    )

    print("[3/7] Calculando bias DXY...")
    dxy_context = calculate_dxy_bias(
        dxy_df=dxy_df,
        fast_ma=config["dxy_bias"]["fast_ma"],
        slow_ma=config["dxy_bias"]["slow_ma"],
        momentum_lookback=config["dxy_bias"]["momentum_lookback"],
        min_separation_pct=config["dxy_bias"]["min_separation_pct"]
    )

    print("[4/7] Detectando contexto técnico...")
    technical_context = build_technical_context(eurusd_df)

    print("[5/7] Validando entry context...")
    entry_context = build_entry_context(
        technical_context=technical_context,
        dxy_context=dxy_context
    )

    print("[6/7] Calculando confluencia...")
    confluence_result = calculate_confluence_score(
        technical_context=technical_context,
        entry_context=entry_context,
        dxy_context=dxy_context,
        weights=config["weights"],
        sessions_config=config["sessions"],
        timezone_name=config.get("timezone", "America/Santiago")
    )

    print("[7/7] Generando señal final...")
    signal_result = generate_signal(
        confluence_result=confluence_result,
        scoring_config=config["scoring"]
    )

    rank_result = rank_signal(signal_result)

    print("\n========= RESULTADO =========")
    print(f"Señal:          {signal_result['signal']}")
    print(f"Confianza:      {signal_result['confidence']}")
    print(f"Rank:           {rank_result['rank']}")
    print(f"Quality:        {rank_result['quality']}")
    print(f"Score final:    {signal_result['score']}")
    print(f"BUY score:      {signal_result['buy_score']}")
    print(f"SELL score:     {signal_result['sell_score']}")
    print(f"Sesión activa:  {signal_result['session_active']}")
    print(f"Sesión:         {signal_result['session_name']}")
    print(f"Timezone:       {signal_result['timezone']}")
    print(f"Hora local:     {signal_result['current_hour']}")

    print("\nContexto técnico:")
    print(f"- Side:             {technical_context['side']}")
    print(f"- Has sweep:        {technical_context['has_sweep']}")
    print(f"- Structure shift:  {technical_context['structure_shift']}")
    print(f"- Sweep side:       {technical_context['sweep_side']}")
    print(f"- Structure side:   {technical_context['structure_side']}")
    print(f"- Swept level:      {technical_context['swept_level']}")
    print(f"- Broken level:     {technical_context['broken_level']}")

    print("\nEntry context:")
    print(f"- Entry side:       {entry_context['side']}")
    print(f"- Entry valid:      {entry_context['is_valid']}")
    print(f"- Alignment score:  {entry_context['alignment_score']}")
    print(f"- Setup type:       {entry_context['setup_type']}")

    print("\nBias DXY:")
    print(f"- DXY bias:         {dxy_context['dxy_bias']}")
    print(f"- EURUSD bias:      {dxy_context['eurusd_bias']}")
    print(f"- Momentum:         {dxy_context['momentum']}")
    print(f"- MA fast:          {dxy_context['ma_fast']}")
    print(f"- MA slow:          {dxy_context['ma_slow']}")
    print(f"- Strength:         {dxy_context['strength']}")

    print("\nRazones entry:")
    for reason in entry_context["reasons"]:
        print(f"  - {reason}")

    print("\nRazones señal:")
    for reason in signal_result["reasons"]:
        print(f"  - {reason}")

    if config["output"]["save_signals_csv"]:
        output_path = config["output"]["signals_path"]
        save_signal(signal_result, rank_result, technical_context, entry_context, dxy_context, output_path)
        print(f"\nSeñal guardada en: {output_path}")

    print("\nProceso finalizado.\n")


if __name__ == "__main__":
    main()