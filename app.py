from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_engine import (
    INDIAN_STOCKS_METADATA,
    calculate_investment_pnl,
    fetch_screener_universe,
    fetch_stock_data,
    fetch_stock_news,
    generate_full_stock_report,
)

# Page configuration
st.set_page_config(
    page_title="Deep Dive Stock Predictor & P&L Simulator",
    page_icon=":material/candlestick_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Quick lookup maps for popular Indian stocks
STOCK_DISPLAY_MAP: Dict[str, str] = {
    item["symbol"]: f"{item['ticker']} — {item['name']} ({item['sector']})"
    for item in INDIAN_STOCKS_METADATA
}
POPULAR_SYMBOLS: List[str] = [item["symbol"] for item in INDIAN_STOCKS_METADATA]
QUICK_PICK_SYMBOLS: List[str] = [
    "TATAPOWER.NS",
    "HAL.NS",
    "BEL.NS",
    "SUZLON.NS",
    "BSE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "TCS.NS",
    "SBIN.NS",
    "RELIANCE.NS",
]


@st.cache_data(ttl="5m", max_entries=100)
def cached_fetch_stock_data(symbol: str, timeframe: str, refresh_token: int = 0) -> Dict[str, Any]:
    """Fetch and cache OHLCV data with technical indicators."""
    _ = refresh_token
    return fetch_stock_data(symbol, timeframe=timeframe)


@st.cache_data(ttl="10m", max_entries=100)
def cached_fetch_stock_news(company_name: str, symbol: str, limit: int = 4, refresh_token: int = 0) -> List[Dict[str, Any]]:
    """Fetch and cache Google News RSS articles for the active stock."""
    _ = refresh_token
    return fetch_stock_news(company_name, symbol, limit=limit)


@st.cache_data(ttl="5m", max_entries=6)
def cached_fetch_screener(timeframe: str, refresh_token: int = 0):
    """Fetch and cache NSE breakout universe scan."""
    _ = refresh_token
    return fetch_screener_universe(INDIAN_STOCKS_METADATA, timeframe=timeframe)


def format_stock_option(symbol: str) -> str:
    """Format stock symbol cleanly in the selectbox dropdown."""
    if symbol in STOCK_DISPLAY_MAP:
        meta = next((m for m in INDIAN_STOCKS_METADATA if m["symbol"] == symbol), None)
        if meta:
            return f"{symbol} — {meta['name']} ({meta['sector']})"
        return STOCK_DISPLAY_MAP[symbol]
    clean = str(symbol).strip().upper()
    return f"{clean} (Custom NSE/BSE Ticker)"


FORMATTED_POPULAR_OPTIONS = {format_stock_option(sym) for sym in POPULAR_SYMBOLS}


def normalize_symbol(raw_symbol: Optional[str]) -> Optional[str]:
    """Normalize user-selected or typed ticker into Yahoo Finance NSE/BSE format."""
    if raw_symbol is None:
        return None
    cleaned = str(raw_symbol).split(" — ")[0].strip().upper()
    if not cleaned:
        return None
    if cleaned in STOCK_DISPLAY_MAP:
        return cleaned
    if cleaned.endswith((".NS", ".BO")) or cleaned.startswith("^"):
        return cleaned
    return f"{cleaned}.NS"


def select_stock_symbol(symbol: str) -> None:
    """Callback helper to switch the active stock cleanly before rerun."""
    st.session_state["stock"] = symbol


def clear_stock_selection() -> None:
    """Callback helper to return to the stock picker & breakout radar."""
    st.session_state["stock"] = None


def trigger_refresh() -> None:
    """Increment refresh token so cached data reloads without resetting widgets."""
    st.session_state["refresh_token"] = st.session_state.get("refresh_token", 0) + 1


# Initialize non-widget session state
st.session_state.setdefault("refresh_token", 0)
st.session_state.setdefault("scan_universe_requested", False)

# ----------------- HEADER -----------------
st.title("Deep Dive Stock Predictor & P&L Simulator", anchor=False)
st.caption(
    "Analyze any NSE/BSE stock with logical stop-loss & target touch probabilities, "
    "step-by-step holding rules, walk-forward backtests, and breakout confirmation signals."
)

# ----------------- STOCK SEARCH & INPUT CONTROLS -----------------
# Keep any stock from URL query params (including custom tickers) in the options list
# so bind="query-params" restores custom symbols across browser refreshes cleanly.
url_stock = st.query_params.get("stock")
select_options = list(POPULAR_SYMBOLS)
if url_stock and url_stock not in select_options and url_stock not in FORMATTED_POPULAR_OPTIONS:
    select_options.insert(0, url_stock)

with st.container(border=True):
    ctrl_stock, ctrl_tf, ctrl_rr, ctrl_actions = st.columns([2.5, 1.2, 1.5, 1.3], vertical_alignment="bottom")

    with ctrl_stock:
        raw_selected_stock = st.selectbox(
            "Stock symbol (pick popular NSE or type ANY NSE/BSE ticker)",
            options=select_options,
            index=None,
            placeholder="Select NSE stock or type e.g. SUZLON, IREDA, 500325.BO...",
            format_func=format_stock_option,
            accept_new_options=True,
            key="stock",
            bind="query-params",
            help="Your selected stock stays synced in the URL so refreshing never resets back to Reliance.",
        )

    with ctrl_tf:
        selected_timeframe = (
            st.segmented_control(
                "Timeframe",
                options=["1d", "1h", "15m"],
                default="1d",
                key="tf",
                bind="query-params",
                help="1d = Daily Swing (2Y) · 1h = BTST / Positional (60D) · 15m = Intraday MIS (30D)",
                width="stretch",
            )
            or "1d"
        )

    with ctrl_rr:
        rr_slider = st.slider(
            "Target risk-reward multiplier",
            min_value=1.5,
            max_value=3.5,
            value=2.0,
            step=0.5,
            key="rr",
            bind="query-params",
            help="Adjusts T1 (0.75x R:R) and T2 (1.0x R:R). Check Target Touch Logic below to verify odds.",
        )

    with ctrl_actions:
        with st.container(horizontal=True):
            st.button(
                "Refresh",
                key="btn_refresh",
                icon=":material/refresh:",
                on_click=trigger_refresh,
                help="Fetch fresh market prices & news while keeping this stock active",
                width="stretch",
            )
            if raw_selected_stock:
                st.button(
                    "Radar",
                    key="btn_radar",
                    icon=":material/radar:",
                    on_click=clear_stock_selection,
                    help="Clear selection and return to Breakout Radar",
                    width="stretch",
                )

active_symbol = normalize_symbol(raw_selected_stock)


def render_breakout_radar_table(timeframe: str, key_prefix: str = "radar") -> None:
    """Render filterable NSE breakout scanner with one-click Analyze action."""
    with st.spinner("Scanning NSE universe for breakout signals and volume surges..."):
        stocks, health = cached_fetch_screener(timeframe, st.session_state["refresh_token"])

    if not stocks:
        st.warning("Could not load NSE screener universe right now. Pick or type a stock symbol above.")
        return

    # Market Breadth Strip
    with st.container(horizontal=True):
        st.metric("Advance / Decline", health["advance_decline"], border=True)
        st.metric("Above 20 EMA", health["above_20ema"], border=True)
        st.metric("Above 200 EMA", health["above_200ema"], border=True)
        st.metric("RSI >= 60", health["rsi_above_60"], border=True)
        st.metric("Active Breakouts", health["breakouts"], border=True)

    radar_df = pd.DataFrame(stocks)
    filter_choice = st.segmented_control(
        "Filter breakout radar",
        options=["All Signals", "Breakouts & Coils", "Strong Buy", "High Volume (>=1.5x)"],
        default="All Signals",
        key=f"{key_prefix}_filter",
    )

    if filter_choice == "Breakouts & Coils":
        radar_df = radar_df[
            radar_df["breakout_status"].str.contains("BREAKOUT|COIL", case=False, na=False)
        ]
    elif filter_choice == "Strong Buy":
        radar_df = radar_df[radar_df["signal"].isin(["STRONG BUY", "BUY"])]
    elif filter_choice == "High Volume (>=1.5x)":
        radar_df = radar_df[radar_df["vol_ratio"] >= 1.5]

    if radar_df.empty:
        st.info("No stocks match this filter on the selected timeframe.")
        return

    display_df = radar_df[
        [
            "ticker",
            "name",
            "price",
            "change_pct",
            "breakout_status",
            "breakout_score",
            "vol_ratio",
            "rsi",
            "signal",
            "sl",
            "target_2",
            "symbol",
        ]
    ].copy()
    display_df["action"] = ":material/analytics: Analyze"

    btn_key = f"{key_prefix}_analyze_btn"

    def _on_analyze_click() -> None:
        click_state = st.session_state.get(btn_key)
        if click_state and click_state.get("row") is not None:
            row_idx = int(click_state["row"])
            if 0 <= row_idx < len(display_df):
                st.session_state["stock"] = str(display_df.iloc[row_idx]["symbol"])

    st.dataframe(
        display_df.drop(columns=["symbol"]),
        column_config={
            "action": st.column_config.ButtonColumn(
                "Open",
                type="primary",
                on_click=_on_analyze_click,
                key=btn_key,
                pinned=True,
            ),
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "name": st.column_config.TextColumn("Company"),
            "price": st.column_config.NumberColumn("CMP (₹)", format="₹%.2f"),
            "change_pct": st.column_config.NumberColumn("Chg %", format="%.2f%%"),
            "breakout_status": st.column_config.TextColumn("Breakout signal"),
            "breakout_score": st.column_config.ProgressColumn("BO score", min_value=0, max_value=100),
            "vol_ratio": st.column_config.NumberColumn("Vol ratio", format="%.1fx"),
            "rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
            "signal": st.column_config.TextColumn("AI signal"),
            "sl": st.column_config.NumberColumn("SL (₹)", format="₹%.2f"),
            "target_2": st.column_config.NumberColumn("T2 (1:2)", format="₹%.2f"),
        },
        hide_index=True,
        width="stretch",
        alt="NSE Breakout Radar showing breakout status, volume surge, RSI, and AI signals",
    )


# ----------------- EMPTY STATE: NO FORCED RELIANCE DEFAULT -----------------
if not active_symbol:
    st.info(
        "No default stock is forced on startup — pick any stock below, type a custom NSE/BSE symbol above, "
        "or scan today's live breakout signals. Your choice stays pinned across browser refreshes.",
        title="Select a stock to begin analysis",
        icon=":material/search:",
    )

    st.markdown("##### Quick-pick popular & momentum stocks")
    pill_cols = st.columns(5)
    for idx, sym in enumerate(QUICK_PICK_SYMBOLS):
        ticker_short = sym.replace(".NS", "")
        with pill_cols[idx % 5]:
            st.button(
                ticker_short,
                key=f"quick_pick_{ticker_short}",
                on_click=select_stock_symbol,
                args=(sym,),
                icon=":material/trending_up:",
                width="stretch",
            )

    st.space("small")
    with st.container(border=True):
        radar_head, radar_btn_col = st.columns([3, 1], vertical_alignment="center")
        with radar_head:
            st.subheader("Live NSE breakout radar", icon=":material/radar:", anchor=False)
            st.caption("Scan 46 liquid Indian stocks to spot active breakouts, pre-breakout squeezes, and volume surges.")
        with radar_btn_col:
            if st.button(
                "Scan breakout signals",
                type="primary",
                icon=":material/bolt:",
                key="btn_launch_radar",
                width="stretch",
            ):
                st.session_state["scan_universe_requested"] = True

        if st.session_state["scan_universe_requested"]:
            render_breakout_radar_table(selected_timeframe, key_prefix="landing_radar")
    st.stop()


# ----------------- FETCH STOCK DATA & QUANTITATIVE REPORT -----------------
try:
    with st.spinner(f"Loading live price history, backtest, and breakout signals for {active_symbol}..."):
        stock_data = cached_fetch_stock_data(
            active_symbol,
            selected_timeframe,
            st.session_state["refresh_token"],
        )
        df = stock_data["df"]
        company_name = stock_data["name"]
        sector_name = stock_data["sector"]
        resolved_symbol = stock_data["symbol"]
        currency = stock_data["currency"]

        report = generate_full_stock_report(
            df,
            risk_reward_ratio=rr_slider,
            timeframe=selected_timeframe,
        )
        stock_news = cached_fetch_stock_news(
            company_name,
            resolved_symbol,
            limit=4,
            refresh_token=st.session_state["refresh_token"],
        )
except Exception as exc:
    st.error(f"Could not load stock '{active_symbol}': {exc}", icon=":material/error:")
    st.info("Tip: Enter a valid NSE symbol (e.g. TATAPOWER, SUZLON, IREDA, TCS, HDFCBANK) or append '.BO' for BSE.")
    st.stop()

analysis = report["analysis"]
sr_levels = report["sr_levels"]
breakout = report["breakout"]
backtest = report["backtest"]
feasibility = report["feasibility"]
holding_plan = report["holding_plan"]
past_perf = report["past_performance"]

cmp = analysis["cmp"]
sl = analysis["sl"]
target_1 = analysis["target_1"]
target_2 = analysis["target_2"]
t1_eval = feasibility["t1"]
t2_eval = feasibility["t2"]
bias = analysis["bias"]

# ----------------- STOCK HEADER STRIP -----------------
with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader(f"{company_name} ({resolved_symbol})", anchor=False)
    sig_color = "green" if "BUY" in analysis["signal"] else ("red" if "SELL" in analysis["signal"] else "gray")
    st.badge(f"AI Signal: {analysis['signal']}", color=sig_color, icon=":material/auto_awesome:")
    st.badge(breakout["status"], color=breakout["color"], icon=":material/bolt:")
    st.badge(f"{sector_name}", color="blue")

# ----------------- SECTION 1: LOGICAL TARGETS & WILL-IT-TOUCH VERDICT -----------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(
        label="Current market price (CMP)",
        value=f"{currency}{cmp:,.2f}",
        delta=f"{analysis['day_change']:+.2f} ({analysis['day_pct_change']:+.2f}%)",
        border=True,
        height="stretch",
    )

with kpi_col2:
    st.metric(
        label="Logical stop loss (ATR + Swing)",
        value=f"{currency}{sl:,.2f}",
        delta=f"-{analysis['risk_pct']:.2f}% risk",
        delta_color="inverse",
        border=True,
        height="stretch",
    )

with kpi_col3:
    st.metric(
        label=f"Target 1 ({rr_slider * 0.75:.2f}R)",
        value=f"{currency}{target_1:,.2f}",
        delta=f"{t1_eval['target_pct']:.2f}% move · {t1_eval['touch_probability']}% touch odds",
        delta_color="normal" if t1_eval["touch_probability"] >= 45 else "off",
        border=True,
        height="stretch",
    )

with kpi_col4:
    st.metric(
        label=f"Target 2 ({rr_slider:.1f}R)",
        value=f"{currency}{target_2:,.2f}",
        delta=f"{t2_eval['target_pct']:.2f}% move · {t2_eval['touch_probability']}% touch odds",
        delta_color="normal" if t2_eval["touch_probability"] >= 45 else "off",
        border=True,
        height="stretch",
    )

with st.container(border=True):
    st.markdown("#### :material/target: Target touch logic — will T1 & T2 actually touch?")
    tc1, tc2 = st.columns(2)

    for col, label, ev in [
        (tc1, f"Target 1 ({currency}{t1_eval['target_price']:,.2f})", t1_eval),
        (tc2, f"Target 2 ({currency}{t2_eval['target_price']:,.2f})", t2_eval),
    ]:
        with col:
            with st.container(border=True, height="stretch"):
                h_left, h_right = st.columns([2, 1], vertical_alignment="center")
                with h_left:
                    st.markdown(f"**{label}**")
                with h_right:
                    st.badge(f"{ev['touch_probability']}% — {ev['verdict']}", color=ev["verdict_color"])

                st.progress(ev["touch_probability"] / 100.0)
                st.caption(
                    f"Stretch: **{ev['atr_multiple']:.2f}x ATR** | "
                    f"Historical Hit Rate (before SL): **{ev['hist_touch_rate']:.0f}%** | "
                    f"ADX Trend Strength: **{feasibility['adx']}**"
                )
                for reason in ev["reasons"]:
                    st.markdown(f"- {reason}")

    advice_icon = ":material/verified:" if t2_eval["touch_probability"] >= 50 else ":material/warning:"
    advice_msg = feasibility["rr_advice"]
    if feasibility["logical_barrier_target"] is not None:
        advice_msg += (
            f" Nearest structure-aligned zone before major supply/demand: "
            f"**{currency}{feasibility['logical_barrier_target']:,.2f}**."
        )
    if t2_eval["touch_probability"] >= 50:
        st.info(advice_msg, title="Risk-Reward Multiplier Check", icon=advice_icon)
    else:
        st.warning(advice_msg, title="Risk-Reward Multiplier Check", icon=advice_icon)

# ----------------- SECTION 2: CAPITAL ALLOCATION & P&L SIMULATOR -----------------
with st.container(border=True):
    st.markdown("#### :material/account_balance_wallet: Capital allocation & profit/loss simulator")

    pnl_col_input, pnl_col_shares, pnl_col_loss, pnl_col_t1, pnl_col_t2 = st.columns([1.8, 1.1, 1.2, 1.2, 1.2])

    with pnl_col_input:
        invest_capital = st.number_input(
            "Your investment capital (₹)",
            min_value=500.0,
            max_value=10000000.0,
            value=50000.0,
            step=5000.0,
            key="capital",
            bind="query-params",
        )
        margin_type = st.segmented_control(
            "Margin type",
            options=["Delivery (1x Cash)", "Intraday MIS (5x Leverage)"],
            default="Intraday MIS (5x Leverage)" if selected_timeframe == "15m" else "Delivery (1x Cash)",
            key="margin",
            bind="query-params",
        ) or "Delivery (1x Cash)"

    pnl_result = calculate_investment_pnl(
        capital=float(invest_capital),
        cmp=cmp,
        sl=sl,
        target_1=target_1,
        target_2=target_2,
        trade_type=margin_type,
    )

    qty_label = "Shares to Short" if bias == "Bearish" else "Shares to Buy"
    with pnl_col_shares:
        st.metric(
            label=qty_label,
            value=f"{pnl_result['quantity']} Qty",
            delta=f"{currency}{pnl_result['actual_investment']:,.0f} Margin ({pnl_result['leverage']})",
            border=True,
            height="stretch",
        )

    with pnl_col_loss:
        st.metric(
            label="Max loss (SL hit)",
            value=f"-{currency}{pnl_result['loss_at_sl']:,.2f}",
            delta=f"-{pnl_result['loss_pct_of_capital']}% Capital",
            delta_color="inverse",
            border=True,
            height="stretch",
        )

    with pnl_col_t1:
        st.metric(
            label="Target 1 profit",
            value=f"+{currency}{pnl_result['profit_at_t1']:,.2f}",
            delta=f"+{pnl_result['profit_t1_pct_of_capital']}% Capital",
            border=True,
            height="stretch",
        )

    with pnl_col_t2:
        st.metric(
            label="Target 2 profit",
            value=f"+{currency}{pnl_result['profit_at_t2']:,.2f}",
            delta=f"+{pnl_result['profit_t2_pct_of_capital']}% Capital",
            border=True,
            height="stretch",
        )

    if pnl_result.get("error"):
        st.warning(pnl_result["error"], icon=":material/warning:")
    elif bias == "Bearish" and "Delivery" in margin_type:
        st.warning(
            "Signal is Bearish (Short setup), while Delivery (1x Cash) in Indian equities is Long-only. "
            "Switch to Intraday MIS (5x) or F&O, or wait for a Bullish reversal.",
            icon=":material/info:",
        )

    pattern_tag = f" | Pattern: **{analysis['patterns'][0]['name']}**" if analysis["patterns"] else ""
    st.caption(
        f"**Expected Cash P&L:** {pnl_result['risk_reward_cash']} | "
        f"**Recommended Hold Mode:** {holding_plan['strategy_mode']} | "
        f"**Typical Time to T1 / T2:** ~{holding_plan['median_bars_to_t1']} / ~{holding_plan['median_bars_to_t2']} {holding_plan['unit']}"
        f"{pattern_tag}"
    )

# ----------------- SECTION 3: INTERACTIVE CANDLESTICK CHART & LEVELS -----------------
chart_hdr_col, chart_zoom_col, chart_sr_col = st.columns([2.2, 1.5, 1.3], vertical_alignment="center")
with chart_hdr_col:
    st.markdown(f"#### :material/candlestick_chart: {company_name} ({resolved_symbol}) — {selected_timeframe.upper()} chart & barriers")
with chart_zoom_col:
    if selected_timeframe == "1d":
        zoom_opt = st.segmented_control(
            "Chart window",
            options=["3M", "6M", "1Y", "2Y"],
            default="6M",
            key="chart_zoom_1d",
            label_visibility="collapsed",
        ) or "6M"
        zoom_bars_map = {"3M": 65, "6M": 130, "1Y": 252, "2Y": len(df)}
        chart_df = df.iloc[-zoom_bars_map[zoom_opt]:]
    else:
        zoom_opt = st.segmented_control(
            "Chart window",
            options=["Recent", "Full Window"],
            default="Recent",
            key="chart_zoom_intra",
            label_visibility="collapsed",
        ) or "Recent"
        recent_count = 120 if selected_timeframe == "15m" else 90
        chart_df = df.iloc[-recent_count:] if zoom_opt == "Recent" else df
with chart_sr_col:
    show_sr_lines = st.toggle("Show S/R & Breakout levels", value=True, key="show_sr")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.78, 0.22],
)

fig.add_trace(
    go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="Price",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ),
    row=1,
    col=1,
)

fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_10"], line=dict(color="#06b6d4", width=1.2), name="EMA 10"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_20"], line=dict(color="#3b82f6", width=1.5), name="EMA 20"), row=1, col=1)
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_50"], line=dict(color="#f59e0b", width=1.5), name="EMA 50"), row=1, col=1)
if "EMA_200" in chart_df.columns and chart_df["EMA_200"].notna().any():
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_200"], line=dict(color="#a855f7", width=1.3, dash="dot"), name="EMA 200"), row=1, col=1)

# Key execution lines with Touch % annotations
fig.add_hline(y=cmp, line_dash="dash", line_color="#eab308", annotation_text=f"CMP: {currency}{cmp:.2f}", annotation_position="top left", row=1, col=1)
fig.add_hline(y=sl, line_dash="solid", line_color="#ef4444", annotation_text=f"SL: {currency}{sl:.2f}", annotation_position="bottom left", row=1, col=1)
fig.add_hline(
    y=target_1,
    line_dash="dot",
    line_color="#10b981",
    annotation_text=f"T1: {currency}{target_1:.2f} ({t1_eval['touch_probability']}% odds)",
    annotation_position="top left",
    row=1,
    col=1,
)
fig.add_hline(
    y=target_2,
    line_dash="solid",
    line_color="#22c55e",
    annotation_text=f"T2: {currency}{target_2:.2f} ({t2_eval['touch_probability']}% odds)",
    annotation_position="top left",
    row=1,
    col=1,
)

if show_sr_lines:
    fig.add_hline(
        y=breakout["trigger_high_20"],
        line_dash="dashdot",
        line_color="#c084fc",
        annotation_text=f"20-Bar Breakout: {currency}{breakout['trigger_high_20']:.2f}",
        annotation_position="top right",
        row=1,
        col=1,
    )
    if sr_levels["nearest_resistance"]:
        nr = sr_levels["nearest_resistance"]
        if abs(nr["price"] - breakout["trigger_high_20"]) / cmp > 0.005:
            fig.add_hline(
                y=nr["price"],
                line_dash="dot",
                line_color="#f97316",
                annotation_text=f"Res ({nr['touches']}x): {currency}{nr['price']:.2f}",
                annotation_position="bottom right",
                row=1,
                col=1,
            )
    if sr_levels["nearest_support"]:
        ns = sr_levels["nearest_support"]
        fig.add_hline(
            y=ns["price"],
            line_dash="dot",
            line_color="#38bdf8",
            annotation_text=f"Sup ({ns['touches']}x): {currency}{ns['price']:.2f}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

vol_colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(chart_df["Close"], chart_df["Open"])]
fig.add_trace(
    go.Bar(x=chart_df.index, y=chart_df["Volume"], marker_color=vol_colors, name="Volume", showlegend=False),
    row=2,
    col=1,
)

fig.update_layout(
    height=560,
    margin=dict(l=12, r=12, t=28, b=12),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(
    fig,
    width="stretch",
    alt=f"{resolved_symbol} candlestick chart with EMA overlays, stop-loss, T1, T2, and breakout resistance levels",
)

# ----------------- SECTION 4: LOGICAL HOLDING PLAN & BREAKOUT SIGNAL -----------------
hold_col, bo_col = st.columns(2)

with hold_col:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/schedule: Logical holding plan — what to hold & when to exit")
        st.badge(holding_plan["strategy_mode"], color="blue", icon=":material/route:")
        st.write(holding_plan["hold_reason"])

        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("Median time to T1", f"~{holding_plan['median_bars_to_t1']} {holding_plan['unit']}")
        hm2.metric("Median time to T2", f"~{holding_plan['median_bars_to_t2']} {holding_plan['unit']}")
        hm3.metric("Max hold time-stop", f"{holding_plan['time_stop_bars']} {holding_plan['unit']}")

        st.markdown("**Step-by-step execution & trailing rules:**")
        for idx, step_text in enumerate(holding_plan["steps"], start=1):
            st.markdown(f"{idx}. {step_text}")

with bo_col:
    with st.container(border=True, height="stretch"):
        st.markdown("#### :material/bolt: Breakout signal & confirmation audit")
        st.badge(f"{breakout['status']} ({breakout['score']}/100)", color=breakout["color"])
        st.write(breakout["summary"])

        bm1, bm2, bm3 = st.columns(3)
        bm1.metric("20-Bar trigger", f"{currency}{breakout['trigger_high_20']:,.2f}", f"{breakout['dist_to_breakout_pct']:+.2f}% away")
        bm2.metric("Volume surge", f"{breakout['vol_ratio']:.2f}x", "vs 20-bar avg")
        bm3.metric("Volatility coil", f"{breakout['bb_squeeze_pct']:.0f}th pct", "Squeeze" if breakout["is_squeeze"] else "Normal")

        st.markdown("**Breakout checklist:**")
        for chk in breakout["checks"]:
            tag = ":green-badge[PASS]" if chk["state"] == "PASS" else (":orange-badge[WATCH]" if chk["state"] == "WATCH" else ":red-badge[FAIL]")
            st.markdown(f"- {tag} {chk['item']}")

        if sr_levels["resistances"] or sr_levels["supports"]:
            with st.expander("Nearest structural resistance & support barriers", icon=":material/layers:"):
                sr_rows = [
                    {
                        "Zone": "Resistance",
                        "Price": r["price"],
                        "Dist %": r["distance_pct"],
                        "Tests": r["touches"],
                        "Strength": r["strength"],
                        "Source": r["label"],
                    }
                    for r in sr_levels["resistances"][:4]
                ] + [
                    {
                        "Zone": "Support",
                        "Price": s["price"],
                        "Dist %": s["distance_pct"],
                        "Tests": s["touches"],
                        "Strength": s["strength"],
                        "Source": s["label"],
                    }
                    for s in sr_levels["supports"][:4]
                ]
                st.dataframe(
                    pd.DataFrame(sr_rows),
                    column_config={
                        "Price": st.column_config.NumberColumn(f"Price ({currency})", format=f"{currency}%.2f"),
                        "Dist %": st.column_config.NumberColumn("From CMP", format="%.2f%%"),
                    },
                    hide_index=True,
                    width="stretch",
                    alt="Detected structural resistance and support zones",
                )

# ----------------- SECTION 5: PAST PERFORMANCES & STRATEGY BACKTEST -----------------
with st.container(border=True):
    st.markdown("#### :material/history: Past performances & walk-forward target backtest")

    # 5A. Historical Stock Returns
    ret_items = [r for r in past_perf["returns"] if r["available"]]
    if ret_items:
        ret_cols = st.columns(len(ret_items) + 3)
        for idx, item in enumerate(ret_items):
            with ret_cols[idx]:
                val = item["return_pct"]
                st.metric(f"{item['horizon']} return", f"{val:+.2f}%", border=True, height="stretch")
        with ret_cols[-3]:
            st.metric("From period high", f"{past_perf['from_high_pct']:+.2f}%", border=True, height="stretch")
        with ret_cols[-2]:
            st.metric("Max drawdown", f"{past_perf['max_drawdown_pct']:.2f}%", border=True, height="stretch")
        with ret_cols[-1]:
            st.metric("Ann. volatility", f"{past_perf['annualized_vol_pct']:.1f}%", border=True, height="stretch")

    st.space("small")

    # 5B. Walk-Forward Strategy Backtest Summary at Active R:R
    bt_sum = backtest["summary"]
    bt_c1, bt_c2, bt_c3, bt_c4, bt_c5 = st.columns(5)
    bt_c1.metric("Simulated setups", f"{bt_sum['total_trades']}", f"{backtest['lookback_bars']} bars tested", border=True, height="stretch")
    bt_c2.metric(f"T1 hit rate ({rr_slider*0.75:.2f}R)", f"{bt_sum['t1_hit_rate']:.1f}%", "before SL", border=True, height="stretch")
    bt_c3.metric(f"T2 hit rate ({rr_slider:.1f}R)", f"{bt_sum['t2_hit_rate']:.1f}%", "before SL", border=True, height="stretch")
    bt_c4.metric(
        "Expectancy / trade",
        f"{bt_sum['expectancy_r']:+.2f}R",
        f"Scale-out: {bt_sum['scaleout_expectancy_r']:+.2f}R",
        border=True,
        height="stretch",
    )
    bt_c5.metric(
        "Strategy vs Buy & Hold",
        f"{bt_sum['scaleout_cum_return_pct']:+.1f}%",
        f"Buy & Hold: {backtest['buy_hold_return_pct']:+.1f}%",
        border=True,
        height="stretch",
    )

    # 5C. R:R Multiplier Comparison Table (1.5x to 3.5x) & Recent Trades
    sens_col, trades_col = st.columns([1.05, 1.35])

    with sens_col:
        st.markdown("**Does increasing R:R actually touch? (1.5x–3.5x backtest curve)**")
        sens_rows = []
        risk_dist = max(abs(cmp - sl), 1e-4)
        for row in backtest["sensitivity"]:
            rr_val = row["rr"]
            proj_t2 = round(cmp + risk_dist * rr_val if bias != "Bearish" else cmp - risk_dist * rr_val, 2)
            tag = "Active + Optimal" if (abs(rr_val - rr_slider) < 1e-6 and abs(rr_val - backtest["optimal_rr"]) < 1e-6) else (
                "Optimal" if abs(rr_val - backtest["optimal_rr"]) < 1e-6 else (
                    "Active" if abs(rr_val - rr_slider) < 1e-6 else ""
                )
            )
            sens_rows.append({
                "R:R": f"1:{rr_val:.1f}",
                "T2 Today": proj_t2,
                "T1 Hit %": row["t1_hit_rate"],
                "T2 Hit %": row["t2_hit_rate"],
                "Exp (R)": row["expectancy_r"],
                "Scale Exp": row["scaleout_expectancy_r"],
                "Note": tag,
            })

        st.dataframe(
            pd.DataFrame(sens_rows),
            column_config={
                "T2 Today": st.column_config.NumberColumn(f"T2 ({currency})", format=f"{currency}%.2f"),
                "T1 Hit %": st.column_config.NumberColumn("T1 Hit %", format="%.1f%%"),
                "T2 Hit %": st.column_config.ProgressColumn("T2 Hit %", min_value=0, max_value=100, format="%.1f%%"),
                "Exp (R)": st.column_config.NumberColumn("Full Exp (R)", format="%+.2fR"),
                "Scale Exp": st.column_config.NumberColumn("50/50 Exp (R)", format="%+.2fR"),
            },
            hide_index=True,
            width="stretch",
            alt="Sensitivity table comparing T1 and T2 hit rates and expectancy across 1.5x to 3.5x R:R multipliers",
        )

    with trades_col:
        st.markdown(f"**Recent historical setups at 1:{rr_slider:.1f} R:R (non-overlapping)**")
        if bt_sum["trades"]:
            recent_trades_df = pd.DataFrame(list(reversed(bt_sum["trades"][-12:])))[
                ["entry_time", "signal", "entry", "sl", "t2", "outcome", "bars_held", "return_pct", "r_multiple"]
            ]
            st.dataframe(
                recent_trades_df,
                column_config={
                    "entry_time": st.column_config.TextColumn("Entry"),
                    "signal": st.column_config.TextColumn("Signal"),
                    "entry": st.column_config.NumberColumn(f"Entry", format=f"{currency}%.2f"),
                    "sl": st.column_config.NumberColumn("SL", format=f"{currency}%.2f"),
                    "t2": st.column_config.NumberColumn("T2", format=f"{currency}%.2f"),
                    "outcome": st.column_config.TextColumn("Outcome"),
                    "bars_held": st.column_config.NumberColumn("Bars"),
                    "return_pct": st.column_config.NumberColumn("Ret %", format="%+.2f%%"),
                    "r_multiple": st.column_config.NumberColumn("R", format="%+.2fR"),
                },
                hide_index=True,
                width="stretch",
                alt="Recent historical walk-forward trade setups and outcomes for this stock",
            )
        else:
            st.info("Not enough historical bars on this timeframe to simulate closed signal setups.")

# ----------------- SECTION 6: NSE UNIVERSE BREAKOUT RADAR (ON-DEMAND) -----------------
radar_exp = st.expander(
    "Scan NSE universe breakout radar (46 liquid stocks)",
    on_change="rerun",
    icon=":material/radar:",
)
if radar_exp.open:
    with radar_exp:
        render_breakout_radar_table(selected_timeframe, key_prefix="deep_radar")

# ----------------- SECTION 7: LIVE CATALYST NEWS & SENTIMENT -----------------
st.markdown("#### :material/newspaper: Live catalyst news & sentiment")
if stock_news:
    news_cols = st.columns(len(stock_news))
    for i, item in enumerate(stock_news):
        with news_cols[i]:
            with st.container(border=True, height="stretch"):
                badge_color = (
                    "green"
                    if item["sentiment"] == "Positive"
                    else ("red" if item["sentiment"] == "Negative" else "gray")
                )
                st.badge(item["sentiment"], color=badge_color)
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"Source: {item['source']}")
else:
    st.info("No recent news articles found for this stock.")

# ----------------- FOOTER -----------------
st.caption(
    "**Educational & Risk Disclaimer:** This tool is strictly for technical analysis research, "
    "historical walk-forward simulation, and quantitative risk-reward evaluation. It does not provide "
    "SEBI-registered financial advisory or guaranteed returns. Always manage position sizing and stop-losses."
)
