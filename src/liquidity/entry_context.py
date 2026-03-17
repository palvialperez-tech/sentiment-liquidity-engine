def build_entry_context(technical_context: dict, dxy_context: dict) -> dict:
    """
    Valida la confluencia entre:
    - sweep
    - estructura
    - bias DXY

    Retorna un contexto depurado para decidir si hay setup real.
    """

    sweep_side = technical_context.get("sweep_side", "neutral")
    structure_side = technical_context.get("structure_side", "neutral")
    has_sweep = technical_context.get("has_sweep", False)
    structure_shift = technical_context.get("structure_shift", False)
    eurusd_bias = dxy_context.get("eurusd_bias", "neutral")

    reasons = []
    side = "neutral"
    alignment_score = 0
    is_valid = False
    setup_type = "none"

    # Caso ideal: sweep + estructura alineados
    if has_sweep and structure_shift and sweep_side == structure_side and sweep_side != "neutral":
        side = sweep_side
        alignment_score += 3
        is_valid = True
        setup_type = "aligned_technical"
        reasons.append("sweep y estructura alineados (+3)")

    # Caso medio: solo sweep válido
    elif has_sweep and sweep_side != "neutral":
        side = sweep_side
        alignment_score += 1
        reasons.append("solo sweep detectado (+1)")

    # Caso medio: solo estructura válida
    elif structure_shift and structure_side != "neutral":
        side = structure_side
        alignment_score += 1
        reasons.append("solo estructura detectada (+1)")

    # Penalización si sweep y estructura chocan
    if (
        has_sweep and structure_shift
        and sweep_side != "neutral"
        and structure_side != "neutral"
        and sweep_side != structure_side
    ):
        alignment_score -= 2
        reasons.append("contradicción entre sweep y estructura (-2)")

    # Confirmación con DXY
    if side != "neutral" and eurusd_bias == side:
        alignment_score += 2
        reasons.append("DXY alineado con el setup (+2)")
    elif side != "neutral" and eurusd_bias != "neutral" and eurusd_bias != side:
        alignment_score -= 2
        reasons.append("DXY contradice el setup (-2)")

    # Si queda muy castigado, se invalida
    if alignment_score >= 2 and side != "neutral":
        is_valid = True

    if alignment_score <= 0:
        is_valid = False
        if setup_type == "aligned_technical":
            setup_type = "weak_alignment"
        elif setup_type == "none":
            setup_type = "invalid"

    return {
        "side": side,
        "is_valid": is_valid,
        "alignment_score": alignment_score,
        "setup_type": setup_type,
        "reasons": reasons
    }