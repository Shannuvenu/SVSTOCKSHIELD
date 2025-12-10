import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="S V STOCKSHIELD",
    page_icon="📈",
    layout="wide",
)

# ---------------- GLOBAL STYLE (UI CLEANUP) ----------------

st.markdown(
    """
    <style>
    :root {
        --bg-main: #05060b;
        --bg-card: #10121b;
        --bg-card-soft: #141826;
        --border-subtle: #292d3a;
        --accent: #ffb347;
        --accent-soft: rgba(255,179,71,0.1);
        --accent-red: #ff4d4d;
        --accent-green: #27d98a;
        --text-main: #f5f5f5;
        --text-muted: #b3b3c3;
    }

    /* App background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top, #151729 0, #05060b 45%);
        color: var(--text-main);
    }
    [data-testid="stSidebar"] {
        background-color: #0b0d15;
        border-right: 1px solid #222433;
    }

    /* Heading vs body font */
    h1, h2, h3, h4, h5, h6, .big-title {
        font-family: "Georgia", "Times New Roman", serif !important;
        letter-spacing: 0.08rem;
    }
    body, p, span, div, label, .subtitle {
        font-family: "Times New Roman", serif !important;
    }

    .big-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--text-main);
    }

    .subtitle {
        font-size: 1rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    .metric-card {
        padding: 1rem 1.3rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, #10121b, #121624);
        border: 1px solid var(--border-subtle);
        box-shadow: 0 14px 35px rgba(0,0,0,0.65);
    }

    .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12rem;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text-main);
    }

    .metric-sub {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin-top: 0.1rem;
    }

    .risk-high {
        color: var(--accent-red) !important;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-medium {
        color: #ffcc33 !important;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .risk-low {
        color: var(--accent-green) !important;
        font-weight: 700;
        font-size: 1.3rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 999px;
        border: 1px solid transparent;
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
        padding-left: 1rem;
        padding-right: 1rem;
        font-size: 0.92rem;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: radial-gradient(circle at top left, var(--accent-soft), transparent);
        border-color: var(--accent);
        color: var(--text-main) !important;
        font-weight: 600;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] .markdown-text-container p {
        font-size: 0.92rem;
        line-height: 1.45;
        color: var(--text-muted);
    }
    [data-testid="stSidebar"] .markdown-text-container strong {
        color: var(--text-main);
    }

    /* File uploader label smaller so it doesn’t look messy */
    .uploadedFileName {
        font-size: 0.8rem !important;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 0.75rem 0 0.2rem 0;
        font-size: 0.9rem;
        color: var(--text-muted);
        border-top: 1px solid #202233;
        margin-top: 1.5rem;
    }
    .app-footer span {
        background: linear-gradient(90deg, #ffb347, #ff4d4d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- HELPERS -----------------


def fetch_index_snapshot(ticker: str):
    """Fetch last 5 days of index and return (close, change, %, emoji)."""
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if len(data) < 2:
            return None
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])
        change = last - prev
        pct = (change / prev) * 100
        emoji = "🟢" if change >= 0 else "🔴"
        return last, change, pct, emoji
    except Exception:
        return None


def classify_risk(score: int) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


def compute_integrity_score(df: pd.DataFrame) -> int:
    """Higher integrity = less manipulation-like behaviour."""
    if df.empty:
        return 0
    vol = df["PctChange"].abs().mean()
    high_days = (df["RiskScore"] >= 70).mean()
    med_days = (df["RiskScore"] >= 40).mean()

    raw = 100 - (vol * 2 + high_days * 100 + med_days * 40)
    integrity = max(0, min(100, int(round(raw))))
    return integrity


def detect_fake_breakout(df: pd.DataFrame) -> bool:
    """Very rough fake breakout detection on recent candles."""
    if len(df) < 15:
        return False
    recent_window = min(20, len(df))
    recent = df.tail(recent_window)
    prior = df.iloc[:-recent_window]

    if prior.empty:
        return False

    prev_high = prior["Close"].max()
    recent_high = recent["Close"].max()
    last_close = df["Close"].iloc[-1]

    breakout = recent_high > prev_high * 1.02
    failure = breakout and last_close < prev_high * 0.99
    return failure


def detect_retail_trap(df: pd.DataFrame) -> bool:
    """Look for long upper wick bearish candle after a rally."""
    if len(df) < 5:
        return False
    last = df.iloc[-1]
    body = abs(last["Close"] - last["Open"])
    upper_wick = last["High"] - max(last["Close"], last["Open"])
    lower_wick = min(last["Close"], last["Open"]) - last["Low"]

    recent_run = df["Close"].iloc[-2] > df["Close"].iloc[-5]
    trap = (
        (last["Close"] < last["Open"])
        and (upper_wick > body * 1.5)
        and (upper_wick > abs(lower_wick) * 1.2)
        and bool(recent_run)
    )
    return trap


def operator_fingerprint_scores(df: pd.DataFrame) -> dict:
    """
    Return rough probabilities (0-100) for different operator patterns
    using only OHLCV and risk scores.
    """
    scores = {
        "Pump-Dump": 10,
        "Accumulation → Spike": 10,
        "Volume Crash Sell-off": 10,
        "Laddering Run-up": 10,
    }
    if len(df) < 20:
        return scores

    window = min(60, len(df))
    seg = df.tail(window).reset_index(drop=True)

    # Pump-Dump: strong rise then fall from peak
    start_price = seg["Close"].iloc[0]
    max_price = seg["Close"].max()
    end_price = seg["Close"].iloc[-1]
    peak_idx = seg["Close"].idxmax()

    rise = (max_price / start_price) - 1
    fall_from_peak = (max_price - end_price) / max_price

    if rise > 0.3 and fall_from_peak > 0.2 and peak_idx > window // 3:
        scores["Pump-Dump"] = 80
    elif rise > 0.2 and fall_from_peak > 0.1:
        scores["Pump-Dump"] = 50

    # Accumulation → Spike
    half = window // 2
    early = seg.iloc[:half]
    late = seg.iloc[half:]

    vol_early = early["Volume"].mean()
    vol_late = late["Volume"].mean()
    risk_early = early["RiskScore"].mean()
    risk_late = late["RiskScore"].mean()
    price_spike = (late["Close"].max() / early["Close"].mean()) - 1

    if vol_late > vol_early * 1.5 and price_spike > 0.15 and risk_late > risk_early * 1.5:
        scores["Accumulation → Spike"] = 80
    elif vol_late > vol_early * 1.3 and price_spike > 0.1:
        scores["Accumulation → Spike"] = 50

    # Volume Crash Sell-off
    big_down = seg[seg["PctChange"] < -4]
    if not big_down.empty and (big_down["Volume"] > seg["Volume"].mean() * 1.5).any():
        scores["Volume Crash Sell-off"] = 75

    # Laddering Run-up
    green = seg["Close"] > seg["PrevClose"]
    streaks = green.groupby((green != green.shift()).cumsum()).cumsum()
    long_streak = streaks.max()
    if long_streak >= 5 and rise > 0.15:
        scores["Laddering Run-up"] = 70

    return scores


# -------------------------- SIDEBAR ------------------------

with st.sidebar:
    st.markdown("#### 📊 S V STOCKSHIELD")
    st.markdown(
        "Real-time **candlestick charts**, **operator risk detection** and "
        "**watchlist scanning** for Indian stocks.\n\n"
        "Use NSE symbols like: `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `ADANIENT.NS`."
    )

    symbol = st.text_input("Single Stock Symbol", value="RELIANCE.NS")
    period = st.selectbox("History Period", ["5d", "1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.selectbox("Candle Timeframe", ["5m", "15m", "30m", "60m", "1d"], index=4)

    st.markdown("---")
    watchlist_input = st.text_input(
        "📋 Watchlist Symbols (comma separated)",
        value="RELIANCE.NS, TCS.NS, HDFCBANK.NS, ADANIENT.NS",
        help="Used in Multi-Stock Scanner tab",
    )

    st.markdown("---")
    auto_refresh = st.checkbox("⏱ Auto-refresh every 60 seconds", value=False)

    st.markdown("---")
    fii_file = st.file_uploader(
        "💰 Upload FII/DII CSV (optional)",
        type=["csv"],
        help="Columns: Date, FII, DII (Net optional)",
    )

    st.markdown("---")
    hype_file = st.file_uploader(
        "🔥 Upload Social/Hype CSV (optional)",
        type=["csv"],
        help="Columns example: Date, HypeScore (0-100)",
        key="hype_uploader",
    )
    option_file = st.file_uploader(
        "📉 Upload Options OI CSV (optional)",
        type=["csv"],
        help="Columns example: Strike, CE_OI, PE_OI, Change_CE_OI, Change_PE_OI",
        key="opt_uploader",
    )

    st.markdown("---")
    st.markdown("Data source: **Yahoo Finance (yfinance API)**")

# ------------------------ HEADER --------------------------

st.markdown('<div class="big-title">📈 S V STOCKSHIELD</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Live Candlesticks • Operator Manipulation Scanner • Watchlist & Forensics</div>',
    unsafe_allow_html=True,
)
st.write("")

# ------------------------ INDICES --------------------------

c1, c2, c3 = st.columns(3)
indices = {
    "📉 NIFTY 50": "^NSEI",
    "📊 SENSEX": "^BSESN",
    "🏦 BANKNIFTY": "^NSEBANK",
}

for (label, ticker), col in zip(indices.items(), [c1, c2, c3]):
    snap = fetch_index_snapshot(ticker)
    with col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>{label}</div>", unsafe_allow_html=True)
        if snap is None:
            st.markdown("<div class='metric-value'>—</div>", unsafe_allow_html=True)
            st.markdown("<div class='metric-sub'>No data</div>", unsafe_allow_html=True)
        else:
            last, change, pct, emoji = snap
            st.markdown(
                f"<div class='metric-value'>{emoji} {last:,.0f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='metric-sub'>{change:+.0f} ({pct:+.2f}%)</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -------------------- FETCH SINGLE STOCK DATA --------------------

if not symbol.strip():
    st.warning("Enter a valid single stock symbol like `RELIANCE.NS`.")
    st.stop()

try:
    data = yf.download(symbol.strip(), period=period, interval=interval, progress=False)
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

if data.empty:
    st.error("No data returned. Try another symbol / period / timeframe.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [c[0] for c in data.columns]

data = data.reset_index()
time_col = data.columns[0]

# ----------------------- TABS -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📉 Candlestick Chart",
        "🚨 Operator Risk Scanner",
        "📋 Multi-Stock Scanner",
        "💰 FII/DII Flows",
        "🧬 Advanced Forensics",
    ]
)

# ==========================================================
#                     TAB 1 – CANDLESTICK
# ==========================================================

with tab1:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data[time_col],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color="lime",
                decreasing_line_color="red",
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        xaxis_title="Time",
        yaxis_title="Price",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Show OHLCV Data"):
        st.dataframe(data)

# ==========================================================
#                     TAB 2 – RISK SCANNER
# ==========================================================

with tab2:
    df = data.copy()

    df["PrevClose"] = df["Close"].shift(1)
    df["PctChange"] = ((df["Close"] - df["PrevClose"]) / df["PrevClose"].abs()) * 100
    df["VolAvg5"] = df["Volume"].rolling(5).mean()

    df["VolumeSpike"] = df["Volume"] > 1.5 * df["VolAvg5"]
    df["PriceSpike"] = df["PctChange"].abs() > 2.0

    df["Up"] = df["Close"] > df["PrevClose"]
    df["Streak"] = df["Up"].groupby((df["Up"] != df["Up"].shift()).cumsum()).cumsum()
    df["TrendRun"] = df["Streak"] >= 2

    df["RiskScore"] = (
        df["VolumeSpike"].astype(int) * 30
        + df["PriceSpike"].astype(int) * 40
        + df["TrendRun"].astype(int) * 30
    )

    df["RiskLevel"] = df["RiskScore"].apply(classify_risk)

    latest = df.iloc[-1]
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("### Latest Risk")
        css_class = (
            "risk-high"
            if latest["RiskLevel"] == "High"
            else "risk-medium"
            if latest["RiskLevel"] == "Medium"
            else "risk-low"
        )
        st.markdown(
            f"<div class='{css_class}'>{int(latest['RiskScore'])} ({latest['RiskLevel']})</div>",
            unsafe_allow_html=True,
        )

    with colB:
        max_idx = df["RiskScore"].idxmax()
        max_row = df.loc[max_idx]
        st.markdown("### Peak Risk (Selected Period)")
        st.write(f"**{int(max_row['RiskScore'])}** on **{max_row[time_col].date()}**")

    with colC:
        st.markdown("### Interpretation")
        if latest["RiskLevel"] == "High":
            st.write("⚠️ Strong abnormal behaviour — possible operator move.")
        elif latest["RiskLevel"] == "Medium":
            st.write("🔶 Some abnormal patterns — monitor closely.")
        else:
            st.write("🟢 No major manipulation signals in the latest candle.")

    st.markdown("---")
    st.markdown("### 🔍 Advanced Operator Insight")

    integrity = compute_integrity_score(df)
    fake_bo = detect_fake_breakout(df)
    trap = detect_retail_trap(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Integrity Score (0-100)**")
        iclass = (
            "risk-low"
            if integrity >= 70
            else "risk-medium"
            if integrity >= 40
            else "risk-high"
        )
        st.markdown(f"<div class='{iclass}'>{integrity}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Fake Breakout Detector**")
        if fake_bo:
            st.write("🚨 Possible **fake breakout** trap in recent candles.")
        else:
            st.write("✅ No clear fake breakout pattern found.")

    with col3:
        st.markdown("**Retail Trap Warning**")
        if trap:
            st.write("⚠️ Long-wick bearish candle after rally — **retail trap risk**.")
        else:
            st.write("🟢 No strong retail trap signal in last candle.")

    st.markdown("### 📈 Suspicion Timeline (Risk Score)")
    fig_risk = go.Figure(
        data=[
            go.Scatter(
                x=df[time_col],
                y=df["RiskScore"],
                mode="lines+markers",
                name="RiskScore",
            )
        ]
    )
    fig_risk.update_layout(
        template="plotly_dark",
        height=300,
        xaxis_title="Time",
        yaxis_title="Risk Score",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("### 🧾 Candle-wise Risk Table")

    df_show = df[
        [
            time_col,
            "Close",
            "Volume",
            "PctChange",
            "VolumeSpike",
            "PriceSpike",
            "TrendRun",
            "RiskScore",
            "RiskLevel",
        ]
    ].copy()

    df_show["PctChange"] = df_show["PctChange"].round(2)
    df_show["VolumeSpike"] = df_show["VolumeSpike"].map({True: "Yes", False: "No"})
    df_show["PriceSpike"] = df_show["PriceSpike"].map({True: "Yes", False: "No"})
    df_show["TrendRun"] = df_show["TrendRun"].map({True: "Yes", False: "No"})

    def highlight(row):
        if row["RiskLevel"] == "High":
            return ["background-color: rgba(255,0,0,0.25)"] * len(row)
        if row["RiskLevel"] == "Medium":
            return ["background-color: rgba(255,200,0,0.20)"] * len(row)
        return [""] * len(row)

    st.dataframe(df_show.style.apply(highlight, axis=1), use_container_width=True)

# ==========================================================
#               TAB 3 – MULTI-STOCK WATCHLIST SCANNER
# ==========================================================

with tab3:
    st.markdown("### 📋 Latest Risk for Watchlist Stocks")

    symbols_list = [s.strip() for s in watchlist_input.split(",") if s.strip()]
    rows = []

    for sym in symbols_list:
        try:
            dfw = yf.download(sym, period="1mo", interval="1d", progress=False)
        except Exception:
            continue

        if dfw.empty:
            continue

        if isinstance(dfw.columns, pd.MultiIndex):
            dfw.columns = [c[0] for c in dfw.columns]

        dfw = dfw.reset_index()
        tcol = dfw.columns[0]

        dfw["PrevClose"] = dfw["Close"].shift(1)
        dfw["PctChange"] = ((dfw["Close"] - dfw["PrevClose"]) / dfw["PrevClose"].abs()) * 100
        dfw["VolAvg5"] = dfw["Volume"].rolling(5).mean()

        dfw["VolumeSpike"] = dfw["Volume"] > 1.5 * dfw["VolAvg5"]
        dfw["PriceSpike"] = dfw["PctChange"].abs() > 2.0

        dfw["Up"] = dfw["Close"] > dfw["PrevClose"]
        dfw["Streak"] = dfw["Up"].groupby((dfw["Up"] != dfw["Up"].shift()).cumsum()).cumsum()
        dfw["TrendRun"] = dfw["Streak"] >= 2

        dfw["RiskScore"] = (
            dfw["VolumeSpike"].astype(int) * 30
            + dfw["PriceSpike"].astype(int) * 40
            + dfw["TrendRun"].astype(int) * 30
        )

        last = dfw.iloc[-1]
        score = int(last["RiskScore"])
        rows.append(
            {
                "Symbol": sym,
                "Date": last[tcol],
                "Close": last["Close"],
                "Volume": last["Volume"],
                "PctChange": round(last["PctChange"], 2),
                "RiskScore": score,
                "RiskLevel": classify_risk(score),
            }
        )

    if not rows:
        st.info("No data loaded for watchlist. Check symbols.")
    else:
        watch_df = pd.DataFrame(rows)
        watch_df = watch_df.sort_values("RiskScore", ascending=False).reset_index(drop=True)

        def highlight_watch(row):
            if row["RiskLevel"] == "High":
                return ["background-color: rgba(255,0,0,0.25)"] * len(row)
            if row["RiskLevel"] == "Medium":
                return ["background-color: rgba(255,200,0,0.20)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            watch_df.style.apply(highlight_watch, axis=1),
            use_container_width=True,
        )

# ==========================================================
#               TAB 4 – FII / DII FLOWS (CSV)
# ==========================================================

with tab4:
    st.markdown("### 💰 FII / DII Flow Visualisation")

    if fii_file is None:
        st.info(
            "Upload a CSV with columns like: **Date, FII, DII, Net**.\n\n"
            "You can download this data from NSE or other sites and drop it here "
            "for analysis."
        )
    else:
        try:
            fii_df = pd.read_csv(fii_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
        else:
            cols_lower = {c.lower(): c for c in fii_df.columns}
            if "date" not in cols_lower or "fii" not in cols_lower or "dii" not in cols_lower:
                st.error("CSV must contain at least 'Date', 'FII', 'DII' columns.")
            else:
                date_col = cols_lower["date"]
                fii_col = cols_lower["fii"]
                dii_col = cols_lower["dii"]

                fii_df[date_col] = pd.to_datetime(fii_df[date_col])
                fii_df = fii_df.sort_values(date_col)

                net_candidates = [c for c in fii_df.columns if c.lower() == "net"]
                if net_candidates:
                    net_col = net_candidates[0]
                else:
                    fii_df["Net"] = fii_df[fii_col] + fii_df[dii_col]
                    net_col = "Net"

                st.markdown("#### Recent FII/DII Activity")
                st.dataframe(fii_df[[date_col, fii_col, dii_col, net_col]].tail(20))

                fig_flows = go.Figure()
                fig_flows.add_trace(
                    go.Bar(
                        x=fii_df[date_col],
                        y=fii_df[net_col],
                        name="Net Flow",
                    )
                )
                fig_flows.update_layout(
                    template="plotly_dark",
                    height=420,
                    xaxis_title="Date",
                    yaxis_title="Net (FII + DII)",
                )
                st.plotly_chart(fig_flows, use_container_width=True)

# ==========================================================
#               TAB 5 – ADVANCED FORENSICS
# ==========================================================

with tab5:
    st.markdown("### 🧬 Operator Fingerprint & Forensic Analytics")

    df_for = df.copy()
    scores = operator_fingerprint_scores(df_for)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧬 Operator Fingerprint Patterns")
        for name, val in scores.items():
            st.write(f"- **{name}** → Probability ~ **{val}%**")

    with col2:
        st.markdown("#### 🌡 Integrity vs Suspicion")
        integrity2 = compute_integrity_score(df_for)
        high_days = int((df_for["RiskScore"] >= 70).sum())
        med_days = int((df_for["RiskScore"].between(40, 69)).sum())
        st.write(f"- Integrity Score: **{integrity2}/100**")
        st.write(f"- High-risk candles: **{high_days}**")
        st.write(f"- Medium-risk candles: **{med_days}**")

    st.markdown("---")
    st.markdown("### 🔥 Volume / Activity Heatmap (Delivery Proxy)")

    heat_df = df_for[[time_col, "Volume", "RiskScore"]].copy()
    heat_df["Date"] = pd.to_datetime(heat_df[time_col]).dt.date
    heat_df["Day"] = pd.to_datetime(heat_df[time_col]).dt.day
    heat_df["Month"] = pd.to_datetime(heat_df[time_col]).dt.month

    heat_pivot = heat_df.pivot_table(
        index="Month", columns="Day", values="Volume", aggfunc="mean"
    )

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns,
            y=heat_pivot.index,
            colorscale="Viridis",
        )
    )
    fig_heat.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Day of Month",
        yaxis_title="Month",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📣 Social Hype vs Price (optional CSV)")

    if hype_file is None:
        st.info(
            "Upload a CSV with columns: **Date, HypeScore** "
            "(0-100 from Google Trends / Twitter etc.) "
            "to see correlation with price moves."
        )
    else:
        try:
            hype_df = pd.read_csv(hype_file)
            hype_df.columns = [c.strip() for c in hype_df.columns]
        except Exception as e:
            st.error(f"Error reading hype CSV: {e}")
        else:
            if not {"Date", "HypeScore"}.issubset(set(hype_df.columns)):
                st.error("Hype CSV must have 'Date' and 'HypeScore' columns.")
            else:
                hype_df["Date"] = pd.to_datetime(hype_df["Date"]).dt.date
                price_df = df_for.copy()
                price_df["Date"] = pd.to_datetime(price_df[time_col]).dt.date
                merged = pd.merge(price_df, hype_df, on="Date", how="inner")
                if merged.empty:
                    st.warning("No overlapping dates between price and hype data.")
                else:
                    corr = merged["PctChange"].corr(merged["HypeScore"])
                    st.write(
                        f"Correlation between **hype** and **price change**: **{corr:.2f}**"
                    )
                    fig_hype = go.Figure()
                    fig_hype.add_trace(
                        go.Bar(x=merged["Date"], y=merged["HypeScore"], name="HypeScore")
                    )
                    fig_hype.add_trace(
                        go.Scatter(
                            x=merged["Date"],
                            y=merged["PctChange"],
                            mode="lines+markers",
                            name="PctChange",
                            yaxis="y2",
                        )
                    )
                    fig_hype.update_layout(
                        template="plotly_dark",
                        height=380,
                        yaxis=dict(title="HypeScore"),
                        yaxis2=dict(
                            title="PctChange",
                            overlaying="y",
                            side="right",
                        ),
                    )
                    st.plotly_chart(fig_hype, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧾 Options Chain Manipulation (optional CSV)")

    if option_file is None:
        st.info(
            "Upload an options OI CSV with columns like "
            "`Strike, CE_OI, PE_OI, Change_CE_OI, Change_PE_OI` "
            "to highlight abnormal OI spikes."
        )
    else:
        try:
            opt_df = pd.read_csv(option_file)
        except Exception as e:
            st.error(f"Error reading options CSV: {e}")
        else:
            required = {"Strike", "CE_OI", "PE_OI"}
            if not required.issubset(set(opt_df.columns)):
                st.error("Options CSV must have at least Strike, CE_OI, PE_OI columns.")
            else:
                opt_df["Total_OI"] = opt_df["CE_OI"] + opt_df["PE_OI"]
                opt_df["PCR"] = opt_df["PE_OI"] / opt_df["CE_OI"].replace(0, pd.NA)
                top_oi = opt_df.sort_values("Total_OI", ascending=False).head(10)

                st.markdown("#### Highest OI Strikes")
                st.dataframe(top_oi[["Strike", "Total_OI", "PCR"]])

                if {"Change_CE_OI", "Change_PE_OI"}.issubset(set(opt_df.columns)):
                    opt_df["Change_Total"] = opt_df["Change_CE_OI"] + opt_df["Change_PE_OI"]
                    spike = opt_df.sort_values("Change_Total", ascending=False).head(10)
                    st.markdown("#### Biggest OI Change (Possible expiry manipulation)")
                    st.dataframe(spike[["Strike", "Change_Total"]])

    st.markdown("---")
    st.markdown("### ⚖ SEBI Case Similarity (static knowledge)")

    sebi_cases = {
        "PCJEWELLER.NS": "Price / volume manipulation case (2018)",
        "RPOWER.NS": "Irregularities and debt issues (historic)",
        "SUZLON.NS": "High volatility & restructuring history",
        "YESBANK.NS": "Fraud / NPA related crisis (2020)",
    }

    if symbol.strip().upper() in sebi_cases:
        st.write(
            f"⚠️ **{symbol.upper()}** itself has been part of a SEBI / regulatory "
            f"concern in the past: {sebi_cases[symbol.strip().upper()]}."
        )
    else:
        integrity2b = compute_integrity_score(df_for)
        high_days_ratio = (df_for["RiskScore"] >= 70).mean()
        if integrity2b < 40 and high_days_ratio > 0.1:
            st.write(
                "⚠️ Behaviour pattern looks **similar to historic SEBI-flagged stocks** "
                "(frequent high-risk spikes + low integrity)."
            )
        else:
            st.write(
                "✅ No strong similarity with known SEBI-flagged price manipulation cases "
                "based on this limited pattern analysis."
            )

# ------------------- AUTO-REFRESH & FOOTER -------------------

if auto_refresh:
    time.sleep(60)
    st.experimental_rerun()

st.markdown(
    """
    <div class="app-footer">
        Built by <span>venugAAdu</span> ⚡
    </div>
    """,
    unsafe_allow_html=True,
)
