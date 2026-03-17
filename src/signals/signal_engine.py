def generate_signal(
    confluence_result: dict,
    scoring_config: dict
) -> dict:
    """
    Genera la señal final según score BUY/SELL.
    """

    buy_score = confluence_result.get("buy_score", 0)
    sell_score = confluence_result.get("sell_score", 0)

    min_signal_score = scoring_config.get("min_signal_score", 5)
    strong_signal_score = scoring_config.get("strong_signal_score", 7)

    signal = "NO_SIGNAL"
    confidence = "low"
    score = 0
    reasons = []

    if buy_score >= min_signal_score and buy_score > sell_score:
        signal = "BUY"
        score = buy_score
        reasons = confluence_result.get("reasons_buy", [])

    elif sell_score >= min_signal_score and sell_score > buy_score:
        signal = "SELL"
        score = sell_score
        reasons = confluence_result.get("reasons_sell", [])

    if signal in ["BUY", "SELL"]:
        if score >= strong_signal_score:
            confidence = "high"
        else:
            confidence = "medium"

    return {
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "session_active": confluence_result.get("session_active", False),
        "session_name": confluence_result.get("session_name"),
        "current_hour": confluence_result.get("current_hour"),
        "timezone": confluence_result.get("timezone"),
        "reasons": reasons
    }