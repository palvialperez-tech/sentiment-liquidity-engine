def rank_signal(signal_result: dict) -> dict:
    """
    Clasifica la señal en niveles de calidad.
    """
    signal = signal_result.get("signal", "NO_SIGNAL")
    score = signal_result.get("score", 0)
    confidence = signal_result.get("confidence", "low")

    if signal == "NO_SIGNAL":
        rank = "NO_SETUP"
        quality = 0

    elif confidence == "high" and score >= 8:
        rank = "A+"
        quality = 5

    elif confidence == "high":
        rank = "A"
        quality = 4

    elif confidence == "medium" and score >= 6:
        rank = "B"
        quality = 3

    elif confidence == "medium":
        rank = "C"
        quality = 2

    else:
        rank = "D"
        quality = 1

    return {
        "rank": rank,
        "quality": quality
    }