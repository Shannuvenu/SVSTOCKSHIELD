# app.py - SV STOCKSHIELD (updated formatting + nicer fundamentals UI)
# Requirements: streamlit, yfinance, pandas, plotly
# pip install streamlit yfinance pandas plotly

import time
from io import BytesIO
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="S V STOCKSHIELD", page_icon="📈", layout="wide")

# ---------------- GLOBAL STYLE ----------------
st.markdown(
    """
    <style>
    :root {
        --heading-font: "Georgia", "Times New Roman", serif;
        --body-font: "Times New Roman", serif;
    }
    html, body, [class*="css"]  {
        font-family: var(--body-font);
        background-color: #050608 !important;
        color: #f5f5f5;
    }
    h1,h2,h3,h4 { font-family: var(--heading-font); letter-spacing: 0.05rem; }
    .big-title { font-size: 2.6rem; font-weight:700; color:#ffffff; letter-spacing:0.12rem; }
    .subtitle { font-size:1rem; color:#bbbbbb; margin-bottom:10px; }
    .metric-card { padding:0.9rem 1.1rem; border-radius:0.8rem; background:#101118; border:1px solid #23252f; box-shadow: 0 0 18px rgba(0,0,0,0.6); }
    .metric-label { font-size:0.85rem; color:#9a9a9a; text-transform:uppercase; }
    .metric-value { font-size:1.6rem; font-weight:bold; color:#ffffff; }
    .metric-sub { font-size:0.85rem; color:#c7c7c7; }
    .risk-high { color:#ff4d4d !important; font-weight:700; font-size:1.4rem; }
    .risk-medium { color:#ffcc33 !important; font-weight:700; font-size:1.4rem; }
    .risk-low { color:#33dd77 !important; font-weight:700; font-size:1.4rem; }
    .manip-badge { padding:0.2rem 0.7rem; border-radius:999px; font-size:0.85rem; display:inline-block; }
    .manip-low { background: rgba(51,221,119,0.18); border:1px solid #33dd77; color:#c7ffe0; }
    .manip-medium { background: rgba(255,204,51,0.18); border:1px solid #ffcc33; color:#fff1c4; }
    .manip-high { background: rgba(255,77,77,0.2); border:1px solid #ff4d4d; color:#ffd6d6; }
    .fund-key { font-weight:700; color:#dcdcdc; }
    .fund-val { color:#ffffff; font-size:1rem; }
    .fund-card { padding:0.6rem 0.9rem; border-radius:8px; background:#0f1113; border:1px solid #222327; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Utility helpers ----------------


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def make_template_bytes(kind: str) -> bytes:
    if kind == "fii":
        df = pd.DataFrame({"Date": ["2025-12-01", "2025-12-02"], "FII": [1000, -500], "DII": [-200, 300], "Net": [800, -200]})
        return df_to_csv_bytes(df)
    if kind == "hype":
        df = pd.DataFrame({"Date": ["2025-12-01", "2025-12-02"], "HypeScore": [45, 60]})
        return df_to_csv_bytes(df)
    if kind == "options":
        df = pd.DataFrame({"Strike": [1500, 1600], "CE_OI": [10000, 5000], "PE_OI": [8000, 12000], "Change_CE_OI": [200, -100], "Change_PE_OI": [-50, 300]})
        return df_to_csv_bytes(df)
    return b""


def format_indian_number(n):
    """Return number formatted using Indian units:
       - >= 1e7 -> Crore (Cr) with 2 decimals
       - >= 1e5 -> Lakh (L) with 2 decimals
       - else use normal with commas.
    """
    try:
        if n is None:
            return "-"
        if isinstance(n, (str, bool)):
            return str(n)
        n = float(n)
        absn = abs(n)
        sign = "-" if n < 0 else ""
        if absn >= 1e7:
            return f"{sign}{absn/1e7:,.2f} Cr"
        if absn >= 1e5:
            return f"{sign}{absn/1e5:,.2f} L"
        # fallback with commas
        return f"{sign}{int(round(absn)):,}"
    except Exception:
        return str(n)


def human_readable_ratio(x):
    try:
        if x is None:
            return "-"
        return f"{x:.2f}" if isinstance(x, (float, int)) else str(x)
    except Exception:
        return str(x)


# ----------------- FETCH INDEX SNAPSHOT -----------------


def fetch_index_snapshot(ticker: str):
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


# ---------------- Risk helpers (unchanged core) ----------------


def classify_risk(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def calc_risk_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["PrevClose"] = df["Close"].shift(1)
    df["PctChange"] = ((df["Close"] - df["PrevClose"]) / df["PrevClose"].abs()) * 100
    df["VolAvg5"] = df["Volume"].rolling(5).mean()
    df["VolumeSpike"] = df["Volume"] > 1.5 * df["VolAvg5"]
    df["PriceSpike"] = df["PctChange"].abs() > 2.0
    df["Up"] = df["Close"] > df["PrevClose"]
    df["Streak"] = df["Up"].groupby((df["Up"] != df["Up"].shift()).cumsum()).cumsum()
    df["TrendRun"] = df["Streak"] >= 2
    df["RiskScore"] = df["VolumeSpike"].astype(int) * 30 + df["PriceSpike"].astype(int) * 40 + df["TrendRun"].astype(int) * 30
    df["RiskLevel"] = df["RiskScore"].apply(classify_risk)
    return df


def compute_integrity_score(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    vol = df["PctChange"].abs().mean()
    high_days = (df["RiskScore"] >= 70).mean()
    med_days = (df["RiskScore"] >= 40).mean()
    raw = 100 - (vol * 2 + high_days * 100 + med_days * 40)
    return max(0, min(100, int(round(raw))))


def detect_fake_breakout(df: pd.DataFrame) -> bool:
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
    return bool(failure)


def detect_retail_trap(df: pd.DataFrame) -> bool:
    if len(df) < 5:
        return False
    last = df.iloc[-1]
    body = abs(last["Close"] - last["Open"])
    upper_wick = last["High"] - max(last["Close"], last["Open"])
    lower_wick = min(last["Close"], last["Open"]) - last["Low"]
    recent_run = bool(df["Close"].iloc[-2] > df["Close"].iloc[-5]) if len(df) >= 5 else False
    trap = (last["Close"] < last["Open"]) and (upper_wick > body * 1.5) and (upper_wick > abs(lower_wick) * 1.2) and recent_run
    return bool(trap)


def whale_footprint(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    last = df.iloc[-1]
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else df["Volume"].mean()
    if pd.isna(avg_vol) or avg_vol == 0:
        return 0
    volume_factor = last["Volume"] / avg_vol
    body = abs(last["Close"] - last["Open"])
    range_ = last["High"] - last["Low"]
    small_body = range_ > 0 and body < range_ * 0.35
    if volume_factor > 3 and small_body:
        return 85
    if volume_factor > 2 and small_body:
        return 65
    if volume_factor > 1.5:
        return 45
    return 15


def price_volume_divergence(df: pd.DataFrame) -> tuple:
    if df.empty:
        return 0, "No signal"
    last = df.iloc[-1]
    vol_avg = df["Volume"].rolling(10).mean().iloc[-1] if len(df) >= 10 else df["Volume"].mean()
    if pd.isna(vol_avg) or vol_avg == 0:
        return 0, "No signal"
    msg = "No major divergence"
    score = 20
    if last["PctChange"] > 1.0 and last["Volume"] < vol_avg * 0.8:
        msg = "Price up but volume down → weak rally / possible operator push"
        score = 70
    elif last["PctChange"] < -1.0 and last["Volume"] > vol_avg * 1.2:
        msg = "Price down with heavy volume → panic selling / strong distribution"
        score = 75
    elif last["PctChange"] > 1.0 and last["Volume"] > vol_avg * 1.2:
        msg = "Price and volume both strong → genuine momentum"
        score = 40
    return score, msg


def operator_fingerprint_scores(df: pd.DataFrame) -> dict:
    scores = {"Pump-Dump": 10, "Accumulation → Spike": 10, "Volume Crash Sell-off": 10, "Laddering Run-up": 10}
    if len(df) < 20:
        return scores
    window = min(60, len(df))
    seg = df.tail(window).reset_index(drop=True)
    start_price = seg["Close"].iloc[0]
    max_price = seg["Close"].max()
    end_price = seg["Close"].iloc[-1]
    peak_idx = seg["Close"].idxmax()
    rise = (max_price / start_price) - 1 if start_price != 0 else 0
    fall_from_peak = (max_price - end_price) / max_price if max_price != 0 else 0
    if rise > 0.3 and fall_from_peak > 0.2 and peak_idx > window // 3:
        scores["Pump-Dump"] = 80
    elif rise > 0.2 and fall_from_peak > 0.1:
        scores["Pump-Dump"] = 50
    half = window // 2
    early = seg.iloc[:half]
    late = seg.iloc[half:]
    vol_early = early["Volume"].mean() if not early.empty else 0
    vol_late = late["Volume"].mean() if not late.empty else 0
    risk_early = early["RiskScore"].mean() if not early.empty else 0
    risk_late = late["RiskScore"].mean() if not late.empty else 0
    price_spike = (late["Close"].max() / early["Close"].mean()) - 1 if early["Close"].mean() != 0 else 0
    if vol_late > vol_early * 1.5 and price_spike > 0.15 and risk_late > risk_early * 1.5:
        scores["Accumulation → Spike"] = 80
    elif vol_late > vol_early * 1.3 and price_spike > 0.1:
        scores["Accumulation → Spike"] = 50
    big_down = seg[seg["PctChange"] < -4]
    if not big_down.empty and (big_down["Volume"] > seg["Volume"].mean() * 1.5).any():
        scores["Volume Crash Sell-off"] = 75
    green = seg["Close"] > seg["PrevClose"]
    streaks = green.groupby((green != green.shift()).cumsum()).cumsum()
    long_streak = streaks.max() if not streaks.empty else 0
    if long_streak >= 5 and rise > 0.15:
        scores["Laddering Run-up"] = 70
    return scores


def manipulation_index(latest_risk, integrity, whale_score, div_score, fake_bo, trap):
    comps = [min(int(latest_risk), 100), 100 - int(integrity), int(whale_score), int(div_score)]
    if fake_bo:
        comps.append(80)
    if trap:
        comps.append(70)
    return int(round(sum(comps) / len(comps))) if comps else 0


def manipulation_label(idx: int):
    if idx >= 70:
        return "High", "manip-high"
    if idx >= 40:
        return "Medium", "manip-medium"
    return "Low", "manip-low"


def smart_explanation(symbol, latest_row, integrity, whale_score, div_msg, fake_bo, trap, manip_idx):
    lines = []
    lines.append(f"{symbol.upper()} closed at **{latest_row['Close']:.2f}**, with a daily move of **{latest_row['PctChange']:.2f}%**.")
    lines.append(f"Overall **integrity score** is **{integrity}/100** (higher = cleaner price action).")
    if whale_score >= 70:
        lines.append("🔍 **Whale footprint**: very strong — big player activity detected.")
    elif whale_score >= 45:
        lines.append("🔍 **Whale footprint**: moderate — some large orders visible.")
    else:
        lines.append("🔍 **Whale footprint**: weak — mostly normal participation.")
    lines.append(f"📊 **Price–volume view**: {div_msg}.")
    if fake_bo:
        lines.append("⚠️ Pattern resembles a **failed breakout** – breakout traders may be trapped.")
    if trap:
        lines.append("⚠️ Latest candle looks like a **retail trap** (long upper wick after a rally).")
    if manip_idx >= 70:
        lines.append("🧨 Combined indicators point to **high probability of operator-driven behaviour**. Fresh entries should be very cautious.")
    elif manip_idx >= 40:
        lines.append("🟠 Signals indicate **moderate manipulation risk** – good for observation, not blind entries.")
    else:
        lines.append("🟢 No strong manipulation cluster right now – behaviour is closer to organic trading.")
    return "\n\n".join(lines)


# ------------------------ HEADER --------------------------
st.markdown('<div class="big-title">📈 S V STOCKSHIELD</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Live Candlesticks • Operator Manipulation Scanner • Watchlist & Forensics</div>', unsafe_allow_html=True)
st.write("")

# ------------------------ INDEX CARDS ---------------------
c1, c2, c3, c4 = st.columns([1, 1, 1, 0.7])
indices = {"📉 NIFTY 50": "^NSEI", "📊 SENSEX": "^BSESN", "🏦 BANKNIFTY": "^NSEBANK"}
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
            st.markdown(f"<div class='metric-value'>{emoji} {last:,.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-sub'>{change:+.0f} ({pct:+.2f}%)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with c4:
    auto_refresh = st.checkbox("⏱ Auto-refresh (60s)", value=False)

st.write("")

# ---------------------- TABS LAYOUT -----------------------
(tab_setup, tab_risk, tab_watch, tab_fii, tab_adv, tab_fundamentals) = st.tabs(
    ["📉 Candlesticks & Setup", "🚨 Operator Risk Scanner", "📋 Multi-Stock & Sector Risk", "💰 FII / DII Flows", "🧬 Advanced Forensics", "📚 Fundamentals"]
)

# ---------------- TAB 1 - Setup & Candles ----------------
with tab_setup:
    st.subheader("Chart Setup")
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        symbol = st.text_input("Stock symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")
        period = st.selectbox("History period", ["5d", "1mo", "3mo", "6mo", "1y"], index=2)
        interval = st.selectbox("Candle timeframe", ["5m", "15m", "30m", "60m", "1d"], index=4)
    with col_r:
        st.markdown("**Notes**")
        st.markdown("- Use `.NS` for NSE, `.BO` for BSE.\n- Intraday intervals (5m..60m) work best with short periods.\n- Daily candles work with 3mo+.")

    st.markdown("---")
    if not symbol.strip():
        st.warning("Enter a valid stock symbol to load the chart.")
    else:
        try:
            data_raw = yf.download(symbol.strip(), period=period, interval=interval, progress=False)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

        if data_raw.empty:
            st.error("No data returned. Try another symbol / period / timeframe.")
            st.stop()

        if isinstance(data_raw.columns, pd.MultiIndex):
            data_raw.columns = [c[0] for c in data_raw.columns]

        data = data_raw.reset_index()
        time_col = data.columns[0]

        st.session_state["symbol"] = symbol.strip()
        st.session_state["data_raw"] = data_raw
        st.session_state["data"] = data
        st.session_state["time_col"] = time_col

        st.subheader("📊 Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=data[time_col], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], increasing_line_color="lime", decreasing_line_color="red")])
        fig.update_layout(template="plotly_dark", height=520, xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📄 Show OHLCV data"):
            st.dataframe(data, use_container_width=True)

if "data_raw" not in st.session_state:
    st.stop()

symbol = st.session_state["symbol"]
data_raw = st.session_state["data_raw"]
data = st.session_state["data"]
time_col = st.session_state["time_col"]
df = calc_risk_from_raw(data_raw).reset_index()

# ---------------- TAB 2 - Risk ----------------
with tab_risk:
    st.subheader("Risk Snapshot & Manipulation Meter")
    latest = df.iloc[-1]
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown("**Latest Risk**")
        css_class = ("risk-high" if latest["RiskLevel"] == "High" else "risk-medium" if latest["RiskLevel"] == "Medium" else "risk-low")
        st.markdown(f"<div class='{css_class}'>{int(latest['RiskScore'])} ({latest['RiskLevel']})</div>", unsafe_allow_html=True)
    with colB:
        max_idx = df["RiskScore"].idxmax()
        max_row = df.loc[max_idx]
        st.markdown("**Peak Risk in Selected Period**")
        st.write(f"Score **{int(max_row['RiskScore'])}** on **{max_row[time_col].date()}**")
    integrity = compute_integrity_score(df)
    fake_bo = detect_fake_breakout(df)
    trap = detect_retail_trap(df)
    whale_score = whale_footprint(df)
    div_score, div_msg = price_volume_divergence(df)
    manip_idx = manipulation_index(int(latest["RiskScore"]), integrity, whale_score, div_score, fake_bo, trap)
    manip_label, manip_css = manipulation_label(manip_idx)
    with colC:
        st.markdown("**Integrity Score (0–100)**")
        iclass = ("risk-low" if integrity >= 70 else "risk-medium" if integrity >= 40 else "risk-high")
        st.markdown(f"<div class='{iclass}'>{integrity}</div>", unsafe_allow_html=True)
    with colD:
        st.markdown("**Manipulation Index (0–100)**")
        st.markdown(f"<span class='manip-badge {manip_css}'>{manip_idx} – {manip_label}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Smart Operator Summary")
    summary_text = smart_explanation(symbol, latest, integrity, whale_score, div_msg, fake_bo, trap, manip_idx)
    st.markdown(summary_text)

    st.markdown("### 📈 Suspicion Timeline (Risk Score)")
    fig_risk = go.Figure(data=[go.Scatter(x=df[time_col], y=df["RiskScore"], mode="lines+markers", name="RiskScore")])
    fig_risk.update_layout(template="plotly_dark", height=320, xaxis_title="Time", yaxis_title="Risk Score")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("### 🧾 Candle-wise Risk Table")
    df_show = df[[time_col, "Close", "Volume", "PctChange", "VolumeSpike", "PriceSpike", "TrendRun", "RiskScore", "RiskLevel"]].copy()
    df_show["PctChange"] = df_show["PctChange"].round(2)
    for col in ["VolumeSpike", "PriceSpike", "TrendRun"]:
        df_show[col] = df_show[col].map({True: "Yes", False: "No"})
    def highlight(row):
        if row["RiskLevel"] == "High":
            return ["background-color: rgba(255,0,0,0.25)"] * len(row)
        if row["RiskLevel"] == "Medium":
            return ["background-color: rgba(255,200,0,0.20)"] * len(row)
        return [""] * len(row)
    st.dataframe(df_show.style.apply(highlight, axis=1), use_container_width=True)
    st.markdown("---")
    st.markdown("### 📄 Download CSV")
    st.download_button("Download risk table CSV", df_show.to_csv(index=False).encode("utf-8"), "risk_table.csv", "text/csv")

# ---------------- TAB 3 - Watchlist ----------------
with tab_watch:
    st.subheader("Watchlist Risk & Sector View")
    watchlist_input = st.text_input("Watchlist symbols (comma separated)", value="RELIANCE.NS, TCS.NS, HDFCBANK.NS, ADANIENT.NS")
    symbols_list = [s.strip() for s in watchlist_input.split(",") if s.strip()]
    rows = []
    sector_map = {"RELIANCE.NS": "Energy & Conglomerate", "TCS.NS": "IT Services", "HDFCBANK.NS": "Banking", "ADANIENT.NS": "Infra & Diversified", "ICICIBANK.NS": "Banking", "INFY.NS": "IT Services", "SBIN.NS": "Banking"}
    for sym in symbols_list:
        try:
            dfw_raw = yf.download(sym, period="1mo", interval="1d", progress=False)
        except Exception:
            continue
        if dfw_raw.empty:
            continue
        if isinstance(dfw_raw.columns, pd.MultiIndex):
            dfw_raw.columns = [c[0] for c in dfw_raw.columns]
        dfw = calc_risk_from_raw(dfw_raw).reset_index()
        last = dfw.iloc[-1]
        score = int(last["RiskScore"])
        rows.append({"Symbol": sym, "Sector": sector_map.get(sym.upper(), "Unknown"), "Date": last[dfw.columns[0]], "Close": last["Close"], "Volume": last["Volume"], "PctChange": round(last["PctChange"], 2), "RiskScore": score, "RiskLevel": classify_risk(score)})
    if not rows:
        st.info("No data loaded for watchlist. Check symbols.")
    else:
        watch_df = pd.DataFrame(rows).sort_values("RiskScore", ascending=False).reset_index(drop=True)
        def highlight_watch(row):
            if row["RiskLevel"] == "High":
                return ["background-color: rgba(255,0,0,0.25)"] * len(row)
            if row["RiskLevel"] == "Medium":
                return ["background-color: rgba(255,200,0,0.20)"] * len(row)
            return [""] * len(row)
        st.markdown("### 📋 Latest Risk by Stock")
        st.dataframe(watch_df.style.apply(highlight_watch, axis=1), use_container_width=True)
        st.markdown("### 🧊 Sector-wise Average Risk")
        sector_risk = watch_df.groupby("Sector")["RiskScore"].mean().reset_index()
        if not sector_risk.empty:
            fig_sec = go.Figure(data=[go.Bar(x=sector_risk["Sector"], y=sector_risk["RiskScore"], marker=dict(color=["#ff4d4d" if v >= 70 else "#ffcc33" if v >= 40 else "#33dd77" for v in sector_risk["RiskScore"]]))])
            fig_sec.update_layout(template="plotly_dark", height=380, xaxis_title="Sector", yaxis_title="Average Risk Score")
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("No sector mapping available for these symbols.")

# ---------------- TAB 4 - FII / DII ----------------
with tab_fii:
    st.subheader("FII / DII Flow Visualisation")
    col_demo, col_upload = st.columns([1, 1])
    with col_demo:
        st.markdown("**Download FII/DII CSV template**")
        st.download_button("Download FII/DII template", make_template_bytes("fii"), file_name="fii_template.csv", mime="text/csv")
    with col_upload:
        fii_file = st.file_uploader("Upload FII/DII CSV (Date, FII, DII, Net optional)", type=["csv"], key="fii_uploader")
    if fii_file is None:
        st.info("Upload a CSV with columns like: **Date, FII, DII, Net**.\nYou can edit the template after download and re-upload.")
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
                st.markdown("### Recent FII/DII Activity")
                st.dataframe(fii_df[[date_col, fii_col, dii_col, net_col]].tail(20))
                fig_flows = go.Figure()
                fig_flows.add_trace(go.Bar(x=fii_df[date_col], y=fii_df[net_col], name="Net Flow"))
                fig_flows.update_layout(template="plotly_dark", height=420, xaxis_title="Date", yaxis_title="Net (FII + DII)")
                st.plotly_chart(fig_flows, use_container_width=True)

# ---------------- TAB 5 - Advanced Forensics ----------------
with tab_adv:
    st.subheader("Operator Fingerprint & Forensic Analytics")
    df_for = df.copy()
    scores = operator_fingerprint_scores(df_for)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧬 Operator Fingerprint Patterns")
        for name, val in scores.items():
            st.write(f"- **{name}** → Probability ~ **{val}%**")
    with col2:
        st.markdown("#### 🌡 Integrity vs Suspicion")
        integrity = compute_integrity_score(df_for)
        high_days = int((df_for["RiskScore"] >= 70).sum())
        med_days = int((df_for["RiskScore"].between(40, 69)).sum())
        st.write(f"- Integrity Score: **{integrity}/100**")
        st.write(f"- High-risk candles: **{high_days}**")
        st.write(f"- Medium-risk candles: **{med_days}**")
    st.markdown("---")
    st.markdown("### ⏱ Multi-Timeframe Risk Alignment")
    tf_data = []
    for label, intr, per in [("5m", "5m", "5d"), ("15m", "15m", "5d"), ("60m", "60m", "1mo"), ("1d", "1d", "3mo")]:
        try:
            d_r, d_l = 0, "No data"
            d_r, d_l = (lambda s, p, i: (0, "No data"))(symbol, per, intr)  # placeholder to avoid blocking
        except Exception:
            d_r, d_l = 0, "No data"
        # use simple_risk_for_interval only if you want to call yf here (skipped to save time)
        tf_data.append({"Timeframe": label, "RiskScore": d_r, "RiskLevel": d_l})
    tf_df = pd.DataFrame(tf_data)
    st.dataframe(tf_df, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🔥 Volume / Activity Heatmap")
    heat_df = df_for[[time_col, "Volume", "RiskScore"]].copy()
    heat_df["Date"] = pd.to_datetime(heat_df[time_col]).dt.date
    heat_df["Day"] = pd.to_datetime(heat_df[time_col]).dt.day
    heat_df["Month"] = pd.to_datetime(heat_df[time_col]).dt.month
    heat_pivot = heat_df.pivot_table(index="Month", columns="Day", values="Volume", aggfunc="mean")
    if not heat_pivot.empty:
        fig_heat = go.Figure(data=go.Heatmap(z=heat_pivot.values, x=heat_pivot.columns, y=heat_pivot.index, colorscale="Viridis"))
        fig_heat.update_layout(template="plotly_dark", height=350, xaxis_title="Day of Month", yaxis_title="Month")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data to build heatmap.")
    st.markdown("---")
    st.markdown("### 📣 Social Hype & Options templates")
    st.download_button("Download Hype template", make_template_bytes("hype"), "hype_template.csv", "text/csv")
    st.download_button("Download Options template", make_template_bytes("options"), "options_template.csv", "text/csv")

# ---------------- TAB 6 - Fundamentals (nicer UI) ----------------
with tab_fundamentals:
    st.subheader("Fundamentals — company financials & filings")
    st.markdown("Enter a stock ticker (e.g., `INFY.NS`) and click Fetch.")

    fcol_l, fcol_r = st.columns([1.4, 1])
    with fcol_l:
        f_symbol = st.text_input("Ticker for fundamentals", value=symbol)
    with fcol_r:
        st.markdown("Notes:")
        st.markdown("- Data pulled best-effort from `yfinance`.")
        st.markdown("- For full filings, upload or paste the annual report links / PDFs below.")

    if st.button("Fetch fundamentals", key="fetch_fund"):
        if not f_symbol.strip():
            st.error("Enter a ticker symbol first.")
        else:
            with st.spinner("Fetching fundamentals..."):
                try:
                    t = yf.Ticker(f_symbol.strip())
                    info = {}
                    try:
                        info = t.info or {}
                    except Exception:
                        info = {}
                    # basic snapshot
                    snapshot = {
                        "Symbol": f_symbol.strip().upper(),
                        "Name": info.get("shortName", "-"),
                        "Market Cap": format_indian_number(info.get("marketCap")),
                        "P/E (TTM)": human_readable_ratio(info.get("trailingPE")),
                        "Forward P/E": human_readable_ratio(info.get("forwardPE")),
                        "Dividend Yield (%)": (f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "-") if isinstance(info.get("dividendYield"), (int, float)) else (human_readable_ratio(info.get("dividendYield")) if info.get("dividendYield") else "-"),
                        "Book Value": format_indian_number(info.get("bookValue")),
                        "ROE": (f"{info.get('returnOnEquity')*100:.2f}%" if info.get("returnOnEquity") else "-") if isinstance(info.get("returnOnEquity"), (int, float)) else human_readable_ratio(info.get("returnOnEquity")),
                        "Price/Book": human_readable_ratio(info.get("priceToBook")),
                        "52wk High / Low": f"{info.get('fiftyTwoWeekHigh', '-')}/{info.get('fiftyTwoWeekLow', '-')}",
                    }
                    # quarterly statements (best-effort)
                    try:
                        q_pl = t.quarterly_financials.fillna(0)
                    except Exception:
                        q_pl = pd.DataFrame()
                    try:
                        q_bs = t.quarterly_balance_sheet.fillna(0)
                    except Exception:
                        q_bs = pd.DataFrame()
                    try:
                        q_cf = t.quarterly_cashflow.fillna(0)
                    except Exception:
                        q_cf = pd.DataFrame()
                except Exception as e:
                    st.error(f"Failed to fetch fundamentals: {e}")
                    q_pl, q_bs, q_cf, snapshot = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
                # display snapshot as cards
                st.markdown("### Snapshot")
                kcols = st.columns(3)
                keys = list(snapshot.items())
                for i, (k, v) in enumerate(keys):
                    col = kcols[i % 3]
                    with col:
                        st.markdown(f"<div class='fund-card'><div class='fund-key'>{k}</div><div class='fund-val'>{v}</div></div>", unsafe_allow_html=True)

                st.markdown("---")
                # neat table of snapshot for download
                snap_df = pd.DataFrame({"Key": list(snapshot.keys()), "Value": list(snapshot.values())})
                st.dataframe(snap_df, use_container_width=True)
                st.download_button("Download snapshot CSV", df_to_csv_bytes(snap_df), "fund_snapshot.csv", "text/csv")

                # quarterly P&L
                st.markdown("### Quarterly Profit & Loss")
                if not q_pl.empty:
                    # format large numbers in table view for readability
                    qpl_display = q_pl.copy()
                    for c in qpl_display.columns:
                        qpl_display[c] = qpl_display[c].apply(format_indian_number)
                    st.dataframe(qpl_display.astype(str), use_container_width=True)
                    st.download_button("Download quarterly P&L CSV", df_to_csv_bytes(q_pl.reset_index()), "quarterly_pl.csv", "text/csv")
                else:
                    st.info("Quarterly P&L not available via yfinance for this symbol.")

                st.markdown("### Quarterly Balance Sheet")
                if not q_bs.empty:
                    qbs_display = q_bs.copy()
                    for c in qbs_display.columns:
                        qbs_display[c] = qbs_display[c].apply(format_indian_number)
                    st.dataframe(qbs_display.astype(str), use_container_width=True)
                    st.download_button("Download quarterly BS CSV", df_to_csv_bytes(q_bs.reset_index()), "quarterly_bs.csv", "text/csv")
                else:
                    st.info("Quarterly balance sheet not available via yfinance for this symbol.")

                st.markdown("### Quarterly Cash Flows")
                if not q_cf.empty:
                    qcf_display = q_cf.copy()
                    for c in qcf_display.columns:
                        qcf_display[c] = qcf_display[c].apply(format_indian_number)
                    st.dataframe(qcf_display.astype(str), use_container_width=True)
                    st.download_button("Download quarterly CF CSV", df_to_csv_bytes(q_cf.reset_index()), "quarterly_cf.csv", "text/csv")
                else:
                    st.info("Quarterly cash flows not available via yfinance for this symbol.")

                st.markdown("---")
                st.markdown("### Shareholding pattern (upload or download template)")
                tmpl = pd.DataFrame({"Quarter": ["Mar-2025", "Dec-2024"], "Promoters": [14.3, 14.6], "FIIs": [30.0, 31.0], "DIIs": [40.0, 39.0], "Public": [15.7, 15.4]})
                st.download_button("Download shareholding template", df_to_csv_bytes(tmpl), "shareholding_template.csv", "text/csv")
                shp_file = st.file_uploader("Upload shareholding CSV (optional)", type=["csv"], key="sharehold_uploader")
                if shp_file:
                    try:
                        shp_df = pd.read_csv(shp_file)
                        st.dataframe(shp_df, use_container_width=True)
                        st.download_button("Download shareholding CSV", df_to_csv_bytes(shp_df), "shareholding.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error reading shareholding CSV: {e}")

                st.markdown("---")
                st.markdown("### Announcements & Annual Reports (paste links or upload files)")
                ann_text = st.text_area("Paste announcement URLs / notes (one per line)", height=120)
                st.markdown("You can paste annual report links here or upload PDFs in future releases.")

# ------------------- AUTO REFRESH -------------------------
if auto_refresh:
    time.sleep(60)
    st.experimental_rerun()

# ------------------- FOOTER ------------------------------
st.markdown("<br><center>🛡️ Built with discipline and obsession by <b>venugAAdu</b>.</center>", unsafe_allow_html=True)