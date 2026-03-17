import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
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


st.set_page_config(
    page_title="Sentiment Liquidity Engine",
    layout="wide"
)


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


def run_engine() -> dict:
    config = load_config()

    eurusd_df = load_eurusd(
        period="10d",
        interval=config["timeframes"]["execution"]
    )

    dxy_df = load_dxy(
        period="10d",
        interval=config["timeframes"]["execution"],
        symbol=config["dxy_symbol"]
    )

    dxy_context = calculate_dxy_bias(
        dxy_df=dxy_df,
        fast_ma=config["dxy_bias"]["fast_ma"],
        slow_ma=config["dxy_bias"]["slow_ma"],
        momentum_lookback=config["dxy_bias"]["momentum_lookback"],
        min_separation_pct=config["dxy_bias"]["min_separation_pct"]
    )

    technical_context = build_technical_context(eurusd_df)

    entry_context = build_entry_context(
        technical_context=technical_context,
        dxy_context=dxy_context
    )

    confluence_result = calculate_confluence_score(
        technical_context=technical_context,
        entry_context=entry_context,
        dxy_context=dxy_context,
        weights=config["weights"],
        sessions_config=config["sessions"],
        timezone_name=config.get("timezone", "America/Santiago")
    )

    signal_result = generate_signal(
        confluence_result=confluence_result,
        scoring_config=config["scoring"]
    )

    rank_result = rank_signal(signal_result)

    latest_price = float(eurusd_df.iloc[-1]["close"]) if not eurusd_df.empty else None
    previous_price = float(eurusd_df.iloc[-2]["close"]) if len(eurusd_df) > 1 else None
    price_change = None if latest_price is None or previous_price is None else latest_price - previous_price

    return {
        "config": config,
        "eurusd_df": eurusd_df,
        "dxy_df": dxy_df,
        "dxy_context": dxy_context,
        "technical_context": technical_context,
        "entry_context": entry_context,
        "confluence_result": confluence_result,
        "signal_result": signal_result,
        "rank_result": rank_result,
        "latest_price": latest_price,
        "price_change": price_change
    }


def ensure_output_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_signals_history(signals_path: str) -> pd.DataFrame:
    if os.path.exists(signals_path):
        try:
            return pd.read_csv(signals_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def build_signal_row(signal_result: dict, rank_result: dict, technical_context: dict, entry_context: dict, dxy_context: dict) -> dict:
    return {
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


def is_duplicate_signal(history_df: pd.DataFrame, new_row: dict) -> bool:
    if history_df.empty:
        return False

    last_row = history_df.iloc[-1].to_dict()

    keys_to_compare = [
        "signal",
        "rank",
        "score",
        "session_name",
        "technical_side",
        "entry_side",
        "entry_valid",
        "alignment_score",
        "setup_type",
        "dxy_bias",
        "eurusd_bias",
        "dxy_strength"
    ]

    for key in keys_to_compare:
        if str(last_row.get(key)) != str(new_row.get(key)):
            return False

    return True


def auto_save_signal_if_needed(signal_result: dict, rank_result: dict, technical_context: dict, entry_context: dict, dxy_context: dict, output_path: str) -> tuple[str, dict | None]:
    signal = signal_result.get("signal")
    is_valid = entry_context.get("is_valid", False)

    if signal not in ["BUY", "SELL"]:
        return "not_saved_no_signal", None

    if not is_valid:
        return "not_saved_invalid", None

    ensure_output_dir(output_path)

    history_df = load_signals_history(output_path)
    new_row = build_signal_row(signal_result, rank_result, technical_context, entry_context, dxy_context)

    if is_duplicate_signal(history_df, new_row):
        return "not_saved_duplicate", new_row

    df_new = pd.DataFrame([new_row])

    if history_df.empty:
        df_all = df_new
    else:
        df_all = pd.concat([history_df, df_new], ignore_index=True)

    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
    return "saved", new_row


def signal_html(signal: str) -> str:
    if signal == "BUY":
        bg = "#123524"
        border = "#22c55e"
        text = "🟢 BUY"
    elif signal == "SELL":
        bg = "#3a1616"
        border = "#ef4444"
        text = "🔴 SELL"
    else:
        bg = "#2b2b2b"
        border = "#9ca3af"
        text = "⚪ NO SIGNAL"

    return f"""
    <div style="
        background-color:{bg};
        border:1px solid {border};
        border-radius:14px;
        padding:18px;
        text-align:center;
        font-size:30px;
        font-weight:700;">
        {text}
    </div>
    """


def setup_html(is_valid: bool, setup_type: str) -> str:
    if is_valid:
        bg = "#102a43"
        border = "#38bdf8"
        text = f"✅ SETUP VÁLIDO · {setup_type}"
    else:
        bg = "#3a1616"
        border = "#ef4444"
        text = f"❌ SETUP INVÁLIDO · {setup_type}"

    return f"""
    <div style="
        background-color:{bg};
        border:1px solid {border};
        border-radius:14px;
        padding:16px;
        text-align:center;
        font-size:20px;
        font-weight:600;">
        {text}
    </div>
    """


def alert_html(signal: str, confidence: str, rank: str, is_valid: bool) -> str:
    if signal == "BUY" and is_valid:
        bg = "#052e16"
        border = "#22c55e"
        text = f"🚨 ALERTA ACTIVA: BUY · {confidence.upper()} · Rank {rank}"
    elif signal == "SELL" and is_valid:
        bg = "#450a0a"
        border = "#ef4444"
        text = f"🚨 ALERTA ACTIVA: SELL · {confidence.upper()} · Rank {rank}"
    else:
        bg = "#1f2937"
        border = "#6b7280"
        text = "Sin alerta activa de alta prioridad"

    return f"""
    <div style="
        background-color:{bg};
        border:2px solid {border};
        border-radius:14px;
        padding:18px;
        text-align:center;
        font-size:22px;
        font-weight:700;">
        {text}
    </div>
    """


def bias_badge_html(label: str, value: str) -> str:
    value_upper = str(value).upper()

    if "BULL" in value_upper or value == "buy":
        bg = "#123524"
        border = "#22c55e"
    elif "BEAR" in value_upper or value == "sell":
        bg = "#3a1616"
        border = "#ef4444"
    else:
        bg = "#2b2b2b"
        border = "#9ca3af"

    return f"""
    <div style="
        background-color:{bg};
        border:1px solid {border};
        border-radius:10px;
        padding:10px 12px;
        margin-bottom:8px;">
        <div style="font-size:12px; opacity:0.8;">{label}</div>
        <div style="font-size:18px; font-weight:700;">{value}</div>
    </div>
    """


def format_reasons(reasons: list) -> str:
    if not reasons:
        return "- Sin razones disponibles"
    return "\n".join([f"- {r}" for r in reasons])


def get_last_valid_signal(history_df: pd.DataFrame) -> dict:
    if history_df.empty:
        return {}

    df = history_df.copy()

    if "entry_valid" in df.columns:
        df = df[df["entry_valid"] == True]

    if "signal" in df.columns:
        df = df[df["signal"].isin(["BUY", "SELL"])]

    if df.empty:
        return {}

    return df.iloc[-1].to_dict()


def render_new_signal_popup(saved_row: dict):
    signal = saved_row.get("signal", "NO_SIGNAL")
    rank = saved_row.get("rank", "-")
    score = saved_row.get("score", "-")
    session_name = saved_row.get("session_name", "-")

    if signal == "BUY":
        bg = "#052e16"
        border = "#22c55e"
        icon = "🟢"
    else:
        bg = "#450a0a"
        border = "#ef4444"
        icon = "🔴"

    st.markdown(
        f"""
        <div style="
            background-color:{bg};
            border:2px solid {border};
            border-radius:16px;
            padding:20px;
            text-align:center;
            font-size:24px;
            font-weight:700;
            margin-bottom:16px;">
            {icon} NUEVA SEÑAL GUARDADA: {signal} · Rank {rank} · Score {score} · Sesión {session_name}
        </div>
        """,
        unsafe_allow_html=True
    )


def play_alert_sound():
    st.components.v1.html(
        """
        <script>
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioContext();

        function beep(freq, duration, delay) {
            setTimeout(() => {
                const oscillator = ctx.createOscillator();
                const gainNode = ctx.createGain();

                oscillator.type = "sine";
                oscillator.frequency.value = freq;
                oscillator.connect(gainNode);
                gainNode.connect(ctx.destination);

                gainNode.gain.setValueAtTime(0.001, ctx.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.01);
                gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration / 1000);

                oscillator.start();
                oscillator.stop(ctx.currentTime + duration / 1000);
            }, delay);
        }

        beep(880, 180, 0);
        beep(1174, 180, 220);
        beep(1567, 250, 460);
        </script>
        """,
        height=0,
    )


st.title("Sentiment Liquidity Engine")
st.caption("Dashboard V5 · alerta sonora + popup visual")

top_a, top_b, top_c, top_d = st.columns([1, 1, 1, 3])

with top_a:
    if st.button("Actualizar ahora"):
        st.rerun()

with top_b:
    show_history = st.toggle("Ver historial", value=True)

with top_c:
    auto_refresh = st.toggle("Auto refresh", value=False)

refresh_seconds = 30
if auto_refresh:
    refresh_seconds = st.selectbox(
        "Segundos",
        options=[15, 30, 60, 120],
        index=1
    )

try:
    result = run_engine()

    config = result["config"]
    eurusd_df = result["eurusd_df"]
    dxy_df = result["dxy_df"]
    dxy_context = result["dxy_context"]
    technical_context = result["technical_context"]
    entry_context = result["entry_context"]
    signal_result = result["signal_result"]
    rank_result = result["rank_result"]
    latest_price = result["latest_price"]
    price_change = result["price_change"]

    signals_path = config["output"]["signals_path"]
    save_status, saved_row = auto_save_signal_if_needed(
        signal_result=signal_result,
        rank_result=rank_result,
        technical_context=technical_context,
        entry_context=entry_context,
        dxy_context=dxy_context,
        output_path=signals_path
    )

    history_df = load_signals_history(signals_path)
    last_valid_signal = get_last_valid_signal(history_df)

    st.markdown("---")

    if save_status == "saved" and saved_row:
        render_new_signal_popup(saved_row)
        play_alert_sound()
        st.success("Señal válida nueva guardada automáticamente en el historial.")
    elif save_status == "not_saved_duplicate":
        st.info("La señal actual ya estaba guardada. No se duplicó.")
    elif save_status == "not_saved_invalid":
        st.warning("No se guardó porque el setup actual es inválido.")
    elif save_status == "not_saved_no_signal":
        st.info("No se guardó porque no hay señal BUY/SELL activa.")

    st.markdown(
        alert_html(
            signal_result["signal"],
            signal_result["confidence"],
            rank_result["rank"],
            entry_context["is_valid"]
        ),
        unsafe_allow_html=True
    )

    st.markdown("")

    hero_left, hero_mid, hero_right = st.columns([2, 2, 2])

    with hero_left:
        st.markdown(signal_html(signal_result["signal"]), unsafe_allow_html=True)

    with hero_mid:
        st.markdown(
            setup_html(entry_context["is_valid"], entry_context["setup_type"]),
            unsafe_allow_html=True
        )

    with hero_right:
        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                border:1px solid #4b5563;
                border-radius:12px;
                padding:16px;
                text-align:center;">
                <div style="font-size:13px; opacity:0.85;">RANK</div>
                <div style="font-size:34px; font-weight:700;">{rank_result['rank']}</div>
                <div style="font-size:14px;">Quality: {rank_result['quality']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("EUR/USD", f"{latest_price:.5f}" if latest_price is not None else "-",
              delta=f"{price_change:.5f}" if price_change is not None else None)
    m2.metric("Score", signal_result["score"])
    m3.metric("BUY Score", signal_result["buy_score"])
    m4.metric("SELL Score", signal_result["sell_score"])
    m5.metric("Sesión", signal_result["session_name"] or "Sin sesión")
    m6.metric("Confianza", signal_result["confidence"])
    m7.metric("Hora local", signal_result["current_hour"])

    st.markdown("---")

    status_left, status_right = st.columns(2)

    with status_left:
        st.subheader("Estado del motor")
        st.success("Motor ejecutado correctamente")
        st.write(f"- Timezone: {signal_result['timezone']}")
        st.write(f"- Auto refresh: {'Activo' if auto_refresh else 'Desactivado'}")
        st.write(f"- Intervalo ejecución: {config['timeframes']['execution']}")
        st.write(f"- Símbolo principal: {config['symbol']}")
        st.write(f"- Símbolo DXY: {config['dxy_symbol']}")

    with status_right:
        st.subheader("Última señal válida guardada")
        if last_valid_signal:
            st.write(f"- Timestamp: {last_valid_signal.get('timestamp', '-')}")
            st.write(f"- Signal: {last_valid_signal.get('signal', '-')}")
            st.write(f"- Rank: {last_valid_signal.get('rank', '-')}")
            st.write(f"- Score: {last_valid_signal.get('score', '-')}")
            st.write(f"- Session: {last_valid_signal.get('session_name', '-')}")
            st.write(f"- DXY bias: {last_valid_signal.get('dxy_bias', '-')}")
        else:
            st.info("No hay señales válidas guardadas todavía.")

    st.markdown("---")

    left, center, right = st.columns([1.3, 1.3, 1.4])

    with left:
        st.subheader("Bias y contexto")
        st.markdown(bias_badge_html("DXY bias", dxy_context["dxy_bias"]), unsafe_allow_html=True)
        st.markdown(bias_badge_html("EUR/USD bias", dxy_context["eurusd_bias"]), unsafe_allow_html=True)
        st.markdown(bias_badge_html("Technical side", technical_context["side"]), unsafe_allow_html=True)
        st.markdown(bias_badge_html("Entry side", entry_context["side"]), unsafe_allow_html=True)

        st.write("**Datos DXY**")
        st.write(f"- Momentum: {dxy_context['momentum']}")
        st.write(f"- MA fast: {dxy_context['ma_fast']}")
        st.write(f"- MA slow: {dxy_context['ma_slow']}")
        st.write(f"- Strength: {dxy_context['strength']}")
        st.write(f"- Separation %: {dxy_context.get('separation_pct')}")

    with center:
        st.subheader("Contexto técnico")
        tech_df = pd.DataFrame([
            ["Side", technical_context["side"]],
            ["Has sweep", technical_context["has_sweep"]],
            ["Structure shift", technical_context["structure_shift"]],
            ["Sweep side", technical_context["sweep_side"]],
            ["Structure side", technical_context["structure_side"]],
            ["Swept level", technical_context["swept_level"]],
            ["Broken level", technical_context["broken_level"]],
        ], columns=["Campo", "Valor"])
        st.dataframe(tech_df, use_container_width=True, hide_index=True)

        st.subheader("Entry context")
        entry_df = pd.DataFrame([
            ["Entry valid", entry_context["is_valid"]],
            ["Alignment score", entry_context["alignment_score"]],
            ["Setup type", entry_context["setup_type"]],
        ], columns=["Campo", "Valor"])
        st.dataframe(entry_df, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Razones")
        st.write("**Entry reasons**")
        st.code(format_reasons(entry_context["reasons"]), language="text")

        st.write("**Signal reasons**")
        st.code(format_reasons(signal_result["reasons"]), language="text")

    st.markdown("---")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("EUR/USD")
        eurusd_chart = eurusd_df[["datetime", "close"]].tail(120).copy()
        eurusd_chart = eurusd_chart.set_index("datetime")
        st.line_chart(eurusd_chart)

    with chart_right:
        st.subheader("DXY")
        dxy_chart = dxy_df[["datetime", "close"]].tail(120).copy()
        dxy_chart = dxy_chart.set_index("datetime")
        st.line_chart(dxy_chart)

    if show_history:
        st.markdown("---")
        st.subheader("Historial y filtros")

        filter_left, filter_mid, filter_right = st.columns(3)

        with filter_left:
            filter_signal = st.selectbox(
                "Filtrar por señal",
                options=["ALL", "BUY", "SELL", "NO_SIGNAL"],
                index=0
            )

        with filter_mid:
            filter_valid = st.selectbox(
                "Filtrar por validez",
                options=["ALL", "VALID_ONLY", "INVALID_ONLY"],
                index=0
            )

        with filter_right:
            rows_limit = st.selectbox(
                "Filas a mostrar",
                options=[10, 20, 30, 50, 100],
                index=2
            )

        if not history_df.empty:
            filtered_df = history_df.copy()

            if filter_signal != "ALL" and "signal" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["signal"] == filter_signal]

            if filter_valid == "VALID_ONLY" and "entry_valid" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["entry_valid"] == True]
            elif filter_valid == "INVALID_ONLY" and "entry_valid" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["entry_valid"] == False]

            score_chart_cols = [c for c in ["timestamp", "buy_score", "sell_score", "score"] if c in filtered_df.columns]
            if len(score_chart_cols) >= 2:
                chart_df = filtered_df[score_chart_cols].tail(40).copy()
                if "timestamp" in chart_df.columns:
                    chart_df = chart_df.set_index("timestamp")
                st.subheader("Evolución de scores")
                st.line_chart(chart_df)

            preferred_cols = [
                "timestamp", "signal", "confidence", "rank", "quality", "score",
                "buy_score", "sell_score", "session_name", "technical_side",
                "entry_side", "entry_valid", "alignment_score", "setup_type",
                "dxy_bias", "eurusd_bias", "dxy_strength"
            ]
            cols = [c for c in preferred_cols if c in filtered_df.columns]

            st.subheader("Tabla historial")
            st.dataframe(filtered_df[cols].tail(rows_limit), use_container_width=True)
        else:
            st.info("Aún no hay historial de señales guardado.")

    st.markdown("---")

    tape_left, tape_right = st.columns(2)

    with tape_left:
        st.subheader("Últimas velas EUR/USD")
        eurusd_tail = eurusd_df[["datetime", "open", "high", "low", "close"]].tail(20).copy()
        st.dataframe(eurusd_tail, use_container_width=True)

    with tape_right:
        st.subheader("Últimas velas DXY")
        dxy_tail = dxy_df[["datetime", "open", "high", "low", "close"]].tail(20).copy()
        st.dataframe(dxy_tail, use_container_width=True)

    if auto_refresh:
        st.caption(f"Auto refresh activo: recargando en {refresh_seconds} segundos...")
        time.sleep(refresh_seconds)
        st.rerun()

except Exception as e:
    st.error(f"Error al ejecutar el motor: {e}")