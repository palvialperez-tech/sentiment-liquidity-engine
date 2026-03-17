from datetime import datetime
from zoneinfo import ZoneInfo


def is_hour_in_session(current_hour: int, start_hour: int, end_hour: int) -> bool:
    if start_hour <= end_hour:
        return start_hour <= current_hour <= end_hour
    return current_hour >= start_hour or current_hour <= end_hour


def get_active_session(sessions_config: dict, timezone_name: str = "America/Santiago") -> dict:
    now_dt = datetime.now(ZoneInfo(timezone_name))
    current_hour = now_dt.hour

    for session_name, session_data in sessions_config.items():
        if not session_data.get("enabled", False):
            continue

        start_hour = session_data.get("start_hour")
        end_hour = session_data.get("end_hour")

        if start_hour is None or end_hour is None:
            continue

        if is_hour_in_session(current_hour, start_hour, end_hour):
            return {
                "session_active": True,
                "session_name": session_name,
                "current_hour": current_hour,
                "timezone": timezone_name
            }

    return {
        "session_active": False,
        "session_name": None,
        "current_hour": current_hour,
        "timezone": timezone_name
    }


def calculate_confluence_score(
    technical_context: dict,
    entry_context: dict,
    dxy_context: dict,
    weights: dict,
    sessions_config: dict,
    timezone_name: str = "America/Santiago"
) -> dict:
    session_info = get_active_session(sessions_config, timezone_name=timezone_name)

    buy_score = 0
    sell_score = 0
    reasons_buy = []
    reasons_sell = []

    sweep_weight = weights.get("sweep", 2)
    structure_weight = weights.get("structure_shift", 2)
    dxy_weight = weights.get("dxy_bias", 3)
    session_weight = weights.get("active_session", 1)

    technical_side = technical_context.get("side", "neutral")
    has_sweep = technical_context.get("has_sweep", False)
    structure_shift = technical_context.get("structure_shift", False)

    eurusd_bias = dxy_context.get("eurusd_bias", "neutral")
    dxy_strength = dxy_context.get("strength", "weak")

    entry_side = entry_context.get("side", "neutral")
    entry_valid = entry_context.get("is_valid", False)
    alignment_score = entry_context.get("alignment_score", 0)

    # Si el setup no es válido, devolvemos score muy bajo
    if not entry_valid or entry_side == "neutral":
        return {
            "buy_score": 0,
            "sell_score": 0,
            "session_active": session_info["session_active"],
            "session_name": session_info["session_name"],
            "current_hour": session_info["current_hour"],
            "timezone": session_info["timezone"],
            "reasons_buy": ["setup inválido por falta de confluencia técnica"],
            "reasons_sell": ["setup inválido por falta de confluencia técnica"]
        }

    if technical_side == "buy" and has_sweep:
        buy_score += sweep_weight
        reasons_buy.append(f"sweep técnico detectado (+{sweep_weight})")

    if technical_side == "sell" and has_sweep:
        sell_score += sweep_weight
        reasons_sell.append(f"sweep técnico detectado (+{sweep_weight})")

    if technical_side == "buy" and structure_shift:
        buy_score += structure_weight
        reasons_buy.append(f"cambio estructural alcista (+{structure_weight})")

    if technical_side == "sell" and structure_shift:
        sell_score += structure_weight
        reasons_sell.append(f"cambio estructural bajista (+{structure_weight})")

    if eurusd_bias == "bullish":
        buy_score += dxy_weight
        reasons_buy.append(f"DXY favorece compras EUR/USD (+{dxy_weight})")
        if dxy_strength == "strong":
            buy_score += 1
            reasons_buy.append("fuerza inversa DXY fuerte (+1)")

    elif eurusd_bias == "bearish":
        sell_score += dxy_weight
        reasons_sell.append(f"DXY favorece ventas EUR/USD (+{dxy_weight})")
        if dxy_strength == "strong":
            sell_score += 1
            reasons_sell.append("fuerza inversa DXY fuerte (+1)")

    # Bonus por alineación validada
    if entry_side == "buy":
        buy_score += max(alignment_score, 0)
        reasons_buy.append(f"alineación técnica validada (+{max(alignment_score, 0)})")

    elif entry_side == "sell":
        sell_score += max(alignment_score, 0)
        reasons_sell.append(f"alineación técnica validada (+{max(alignment_score, 0)})")

    if session_info["session_active"]:
        buy_score += session_weight
        sell_score += session_weight
        reasons_buy.append(f"sesión activa: {session_info['session_name']} (+{session_weight})")
        reasons_sell.append(f"sesión activa: {session_info['session_name']} (+{session_weight})")

    return {
        "buy_score": buy_score,
        "sell_score": sell_score,
        "session_active": session_info["session_active"],
        "session_name": session_info["session_name"],
        "current_hour": session_info["current_hour"],
        "timezone": session_info["timezone"],
        "reasons_buy": reasons_buy,
        "reasons_sell": reasons_sell
    }