# app.py — S V STOCKSHIELD (stable, fixed duplicate IDs, robust charts, black+red UI, logo slot)
# Requirements:
# pip install streamlit yfinance pandas plotly requests numpy python-dotenv
# Optional for social hype: pip install snscrape vaderSentiment

from http import cookies
import os
import time
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, List
from pytrends.request import TrendReq
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import openpyxl
from supabase import create_client
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

cookies = EncryptedCookieManager(
    prefix="svstockshield_",
    password="supersecretkey"
)

if not cookies.ready():
    st.stop()
def portfolio_correlation_analysis(symbols):
    price_data = {}

    for sym in symbols:
        df = yf.download(sym, period="1y", interval="1d", progress=False)
        if not df.empty:
            price_data[sym] = df["Close"]

    prices = pd.DataFrame(price_data).dropna()
    returns = prices.pct_change().dropna()

    corr_matrix = returns.corr()

    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

    volatility = returns.std().mean()

    return corr_matrix, avg_corr, volatility
# ================= EXPLAINABLE AI RISK ENGINE =================

def explain_index_score(rsi, macd, volatility, score):
    reasons = []

    if rsi > 70:
        reasons.append("RSI is in overbought zone indicating buying exhaustion.")
    elif rsi < 30:
        reasons.append("RSI is in oversold zone indicating panic selling.")
    else:
        reasons.append("RSI is neutral showing market indecision.")

    if macd < 0:
        reasons.append("MACD shows bearish momentum building.")
    else:
        reasons.append("MACD still positive but weakening.")

    if volatility > 2:
        reasons.append("Volatility expansion suggests unstable market conditions.")
    else:
        reasons.append("Low volatility shows hidden risk building silently.")

    reasons.append(f"These combined signals produced the Index Risk Score of {score}.")

    return reasons[:2]
# ================= SUPABASE AUTHENTICATION ================= 

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

def auth_ui():
    st.title("🔐 Login to SV STOCKSHIELD")

    mode = st.radio("Choose", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button(mode):
        try:
            if mode == "Sign Up":
                res = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })

                # ✅ THIS IS THE IMPORTANT PART
                if res.user:
                    st.success("Signup successful. Please login.")
                else:
                    st.error("Signup failed")

            else:  # Login
                res = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                if res.user:
                    cookies["access_token"] = res.session.access_token
                    cookies["refresh_token"] = res.session.refresh_token
                    cookies.save()
                    st.rerun()
                else:
                    st.error("Login failed")

        except Exception as e:
            st.error(str(e))
if "access_token" in cookies:
    supabase.auth.set_session(
        cookies["access_token"],
        cookies["refresh_token"]
    )
else:
    auth_ui()
    st.stop()
with st.expander("📘 How to Use SV STOCKSHIELD (Read Once)"):
    st.markdown("""
### 🛡️ What is SV STOCKSHIELD?
SV STOCKSHIELD is a **market intelligence & risk-awareness tool** built for retail traders.
It does **NOT give buy/sell calls**.  
It helps you understand **who controls the price**, **where risk is hiding**, and **when NOT to trade**.

---

### 🔍 Core Philosophy
> **Risk management is real profit.**  
If you can avoid bad trades, profits take care of themselves.

---

## 📊 MODULES EXPLAINED

### 1️⃣ Candlesticks & Setup
- Shows **price + volume behavior**
- Helps spot:
  - Fake breakouts  
  - Weak rallies  
  - Distribution candles  
- Use this to judge **price honesty**, not direction.

---

### 2️⃣ Operator Risk Scanner
- Detects **abnormal volume, candle traps & sudden spikes**
- Answers one question:
  > *Is smart money active here or is retail being trapped?*
- If risk is HIGH → **stay away**

---

### 3️⃣ Market Hype (Google Trends)
- Measures **public attention**, not fundamentals
- High hype = emotional crowd
- Low hype = quiet accumulation zone
- Use hype to **avoid FOMO**, not chase it

---

### 4️⃣ Fundamentals
- Pulls **company financials, ratios & quarterly data**
- Use this to check:
  - Valuation sanity  
  - Financial stability  
- Fundamentals tell **WHAT to hold**, not WHEN to enter

---

### 5️⃣ Alerts Engine
- Combines multiple signals
- Green = no immediate danger  
- Red = something is off → slow down

---

## 🔗 RELATED TOOLS (ECOSYSTEM)

### 📊 StoxEye
- Fast visual scanner
- Designed for **quick market overview**
- Use it when you want speed, not depth

### 🛡️ Secure First Calculator
- Position sizing & capital protection tool
- Tells you:
  - How much to risk  
  - How much you can lose safely  
- Use **before every trade**
> Capital protection comes first. Always.

---

## ⚠️ IMPORTANT DISCLAIMERS
- This is **not financial advice**
- No buy/sell recommendations
- This tool helps you **think**, not follow

---

## 🧠 How to Use Like a Pro
1. Check **risk first**
2. Then check **price + volume**
3. Then confirm with **fundamentals**
4. Use **Secure First** before entering
5. If confused → **do nothing**

> Doing nothing is also a position.
""")                    
quotes = [
    "Trade the plan, not emotions.",
    "Capital protection comes first.",
    "Risk management is real profit.",
    "Smart money leaves clues.",
    "Patience pays more than prediction."
]

if "quote_time" not in st.session_state:
    st.session_state.quote_time = time.time()

if time.time() - st.session_state.quote_time > 30:
    st.session_state.quote_time = time.time()
    st.session_state.current_quote = np.random.choice(quotes)

st.markdown(
    f"<div style='font-family:cursive;color:#ff4d4d;font-size:18px'>"
    f"“{st.session_state.get('current_quote', quotes[0])}”</div>",
    unsafe_allow_html=True
)



# ---------- SESSION INIT ----------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ---------------- CONFIG & KEYS ----------------
st.set_page_config(page_title="S V STOCKSHIELD", page_icon="📈", layout="wide")
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
body {
    background-color: #0e1117;
}

/* ---------- MAIN TITLE GLOW ---------- */
.title-glow {
    color: #ff2b2b;
    text-shadow: 0 0 12px #ff2b2b;
    font-weight: 700;
}

/* ---------- CARD BOX (no red border) ---------- */
.ui-card {
    background: #161b22;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
    margin-bottom: 16px;
}

/* ---------- INDEX CARDS ---------- */
.index-card {
    background: #161b22;
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 0 12px rgba(0,255,100,0.08);
}

/* ---------- TAB BUTTON STYLE ---------- */
.stTabs [data-baseweb="tab"] {
    background: #161b22;
    border-radius: 10px;
    padding: 8px 14px;
    margin-right: 6px;
}

/* ---------- BUTTON STYLE ---------- */
.stButton>button {
    background: #1f2630;
    border-radius: 10px;
    border: 1px solid #2c3440;
}

/* ---------- THEORY / CONTENT BOX ---------- */
.content-box {
    background: #141922;
    padding: 20px;
    border-radius: 12px;
    line-height: 1.6;
    box-shadow: 0 0 8px rgba(255,255,255,0.04);
    margin-bottom: 14px;
}

</style>
""", unsafe_allow_html=True)

# load FMP API key from streamlit secrets or environment
FMP_API_KEY = None
try:
    if "FMP" in st.secrets and "KEY" in st.secrets["FMP"]:
        FMP_API_KEY = st.secrets["FMP"]["KEY"]
except Exception:
    pass
if not FMP_API_KEY:
    FMP_API_KEY = os.getenv("FMP_API_KEY", None)

BASE_FMP = "https://financialmodelingprep.com/api/v3"


def call_fmp(path: str, params: dict = None):
    if not FMP_API_KEY:
        return None
    try:
        params = params or {}
        params["apikey"] = FMP_API_KEY
        url = f"{BASE_FMP}/{path}"
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def market_problem_mapper(df, latest_row):
    problems = []

    if latest_row["VolumeSpike"] and not latest_row["PriceSpike"]:
        problems.append("High volume without price move → accumulation/distribution.")

    if latest_row["PriceSpike"] and not latest_row["VolumeSpike"]:
        problems.append("Price spike without volume support → weak/fake move.")

    if latest_row["TrendRun"]:
        problems.append("Extended one-side trend — retail may enter late.")

    if latest_row["RiskScore"] >= 70:
        problems.append("Current structure shows manipulation characteristics.")

    if not problems:
        problems.append("No abnormal market behaviour detected. Price action looks organic.")

    return problems


# ---------------- STYLE / THEME ----------------
st.markdown("""
<style>

/* ===== GLOBAL DARK THEME ===== */
html, body, [class*="css"] {
    background-color: #000000 !important;
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== RED GLOW CARDS ===== */
.metric-card, .fund-card {
    background: #0a0a0a;
    border: 1px solid #ff1a1a;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(255,0,0,0.4);
    transition: 0.3s;
}

.metric-card:hover, .fund-card:hover {
    box-shadow: 0 0 25px rgba(255,0,0,0.8);
    transform: scale(1.02);
}

/* ===== BUTTON STYLE ===== */
.stButton>button {
    background-color: #0a0a0a;
    color: white;
    border: 1px solid #ff1a1a;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
    transition: 0.3s;
}
            
.index-card {
    background: #111111;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 0 12px rgba(0,0,0,0.6);
    border: 1px solid #222;
    text-align: center;
}

.index-card h4 {
    color: #bbbbbb;
    margin-bottom: 6px;
    font-weight: 500;
}

.index-card h2 {
    color: white;
    margin: 4px 0;
    font-size: 26px;
}

.index-card p {
    margin: 0;
    font-size: 14px;
    color: #4caf50;  /* green for change */
}            

.stButton>button:hover {
    background-color: #ff1a1a;
    color: black;
    box-shadow: 0 0 20px #ff1a1a;
}

.content-box {
    background: #111111;
    border-radius: 12px;
    padding: 18px;
    border: 1px solid #222;
    box-shadow: 0 0 10px rgba(0,0,0,0.6);
    margin-bottom: 14px;
}

.content-box h3 {
    margin-top: 0;
    color: #ffffff;
}

.content-box p {
    color: #cccccc;
    line-height: 1.6;
}

/* ===== TABS ===== */
button[data-baseweb="tab"] {
    background-color: #0a0a0a !important;
    border: 1px solid #ff1a1a !important;
    color: white !important;
    border-radius: 6px 6px 0 0 !important;
}

button[data-baseweb="tab"]:hover {
    box-shadow: 0 0 10px #ff1a1a;
}

/* ===== EXPANDERS ===== */
details {
    border: 1px solid #ff1a1a;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(255,0,0,0.4);
}

/* ===== INPUT BOXES ===== */
input, textarea {
    background-color: #0a0a0a !important;
    border: 1px solid #ff1a1a !important;
    color: white !important;
}

/* ===== TITLE GLOW ===== */
.big-title {
    font-size: 30px;
    font-weight: 800;
    color: #ff1a1a;
    text-shadow: 0 0 15px #ff1a1a;
}

/* ===== SUBTITLE ===== */
.subtitle {
    color: #bbbbbb;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def fetch_google_trends_score(keyword):
    try:
        pytrends = TrendReq(hl="en-IN", tz=330)
        pytrends.build_payload([keyword], timeframe="now 7-d", geo="IN")
        data = pytrends.interest_over_time()

        if data.empty:
            return 0, "No trend data"

        score = int(data[keyword].mean())
        return score, "Trend data fetched"

    except Exception as e:
        return 0, f"Trend fetch failed"
def format_indian_number(n):
    try:
        if n is None:
            return "-"
        if isinstance(n, str):
            return n
        n = float(n)
        absn = abs(n)
        sign = "-" if n < 0 else ""
        if absn >= 1e7:
            return f"{sign}{absn/1e7:,.2f} Cr"
        if absn >= 1e5:
            return f"{sign}{absn/1e5:,.2f} L"
        return f"{sign}{int(round(absn)):,}"
    except Exception:
        return str(n)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_csv(buf, index=True)
    return buf.getvalue()


def period_mapping_for_yf(h: str) -> str:
    m = {"1m": "1mo", "6m": "6mo", "1y": "1y", "3y": "3y", "5y": "5y", "10y": "10y", "max": "max"}
    return m.get(h, "1y")


def compute_cagr_from_series(idx: pd.Index, vals: pd.Series) -> Optional[float]:
    try:
        if len(vals) < 2:
            return None
        start = float(vals.iloc[0])
        end = float(vals.iloc[-1])
        days = (pd.to_datetime(idx[-1]) - pd.to_datetime(idx[0])).days
        years = days / 365.25 if days > 0 else None
        if start <= 0 or (years is None) or years <= 0:
            return None
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None


# ---------------- HEADER (logo slot) ----------------
logo_path = "assets/logo-red.png"
col_logo, col_title = st.columns([0.12, 1])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=72)
with col_title:
    st.markdown('<div class="big-title">🛡️ S V STOCKSHIELD</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Market Manipulation & Risk Intelligence Terminal</div>', unsafe_allow_html=True)

st.write("")

# ---------------- Top index cards ----------------
def fetch_index_snapshot(ticker: str) -> Optional[Tuple[float, float, float, str]]:
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data.shape[0] < 2:
            return None
        last = data["Close"].iloc[-1].item()
        prev = data["Close"].iloc[-2].item()
        change = last - prev
        pct = (change / prev) * 100 if prev != 0 else 0
        emoji = "🟢" if change >= 0 else "🔴"
        return last, change, pct, emoji
    except Exception:
        return None


c1, c2, c3, c4 = st.columns([1, 1, 1, 0.7])
indices = {"📉 NIFTY 50": "^NSEI", "📊 SENSEX": "^BSESN", "🏦 BANKNIFTY": "^NSEBANK"}
for (label, ticker), col in zip(indices.items(), [c1, c2, c3]):
    snap = fetch_index_snapshot(ticker)
    with col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#bdbdbd'>{label}</div>", unsafe_allow_html=True)
        if snap is None:
            st.markdown("<div style='font-size:16px'>—</div>", unsafe_allow_html=True)
            st.markdown("<div class='muted'>No data</div>", unsafe_allow_html=True)
        else:
            last, change, pct, emoji = snap
            st.markdown(f"<div style='font-size:20px'>{emoji} {last:,.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>{change:+.0f} ({pct:+.2f}%)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with c4:
    auto_refresh = st.checkbox("⏱ Auto-refresh (60s)", value=False)

st.write("")
# ================= ML RISK PREDICTION MODEL =================
from sklearn.ensemble import RandomForestClassifier

def train_risk_model(df):
    df_ml = df.copy()

    # Target: Did price fall in next 3 candles?
    df_ml["FutureDrop"] = df_ml["Close"].shift(-3) < df_ml["Close"]

    df_ml = df_ml.dropna()

    features = ["Volume", "PctChange", "RiskScore"]
    X = df_ml[features]
    y = df_ml["FutureDrop"].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model


def predict_next_risk(model, latest_row):
    # MASTER CONTROL
    if latest_row["RiskScore"] >= 70:
        return "High Chance of Fall 📉 (Manipulation Detected)"

    features = ["Volume", "PctChange", "RiskScore"]
    X_pred = latest_row[features].values.reshape(1, -1)
    pred = model.predict(X_pred)[0]

    return "High Chance of Fall 📉" if pred == 1 else "Low Risk of Fall 📈"

# ---------- SESSION INIT ----------
# ---------------- RISK HELPERS ----------------
def explain_risk_with_system_features(df):
    latest = df.iloc[-1]
    reasons = []

    if latest["VolumeSpike"]:
        reasons.append("Unusual volume activity compared to recent average.")

    if latest["PriceSpike"]:
        reasons.append("Sharp price movement detected in current candle.")

    if latest["TrendRun"]:
        reasons.append("Continuous directional candles showing momentum build-up.")

    if latest["RiskScore"] >= 70:
        reasons.append("Multiple risk signals combined — possible operator activity.")

    if not reasons:
        reasons.append("Price and volume behaviour look normal and organic.")

    return reasons
def calc_risk_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy().reset_index()
    df["PrevClose"] = df["Close"].shift(1)
    df["PctChange"] = ((df["Close"] - df["PrevClose"]) / df["PrevClose"].abs()) * 100
    df["VolAvg5"] = df["Volume"].rolling(5).mean()
    df["VolAvg10"] = df["Volume"].rolling(10).mean()
    df["VolumeSpike"] = df["Volume"] > 1.2 * df["VolAvg10"]
    df["PriceSpike"] = df["PctChange"].abs() > 1.2
    df["Up"] = df["Close"] > df["PrevClose"]
    df["Streak"] = df["Up"].groupby((df["Up"] != df["Up"].shift()).cumsum()).cumsum()
    df["TrendRun"] = df["Streak"] >= 3
    df["RiskScore"] = df["VolumeSpike"].astype(int) * 30 + df["PriceSpike"].astype(int) * 40 + df["TrendRun"].astype(int) * 30
    def classify(score):
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"
    df["RiskLevel"] = df["RiskScore"].apply(classify)
    return df
def tag_candle(row):
    """Classify candlestick pattern based on OHLC and risk metrics."""
    try:
        body = abs(row["Close"] - row["Open"]) if "Open" in row and "Close" in row else 0
        upper_wick = row["High"] - max(row["Close"], row["Open"]) if "High" in row and "Close" in row and "Open" in row else 0
        lower_wick = min(row["Close"], row["Open"]) - row["Low"] if "Low" in row and "Close" in row and "Open" in row else 0
        range_ = row["High"] - row["Low"] if "High" in row and "Low" in row else 0
        
        if row["RiskScore"] >= 70:
            if upper_wick > body * 1.5:
                return "Rejection (High Risk)"
            return "Suspicious (High Risk)"
        elif row["RiskScore"] >= 40:
            if lower_wick > body * 1.5:
                return "Doji/Reversal Signal"
            return "Caution (Medium Risk)"
        else:
            if row["Close"] > row["Open"]:
                return "Bullish"
            else:
                return "Bearish"
    except Exception:
        return "Unknown"


def backtest_risk_accuracy(df):
    high_risk = df[df["RiskScore"] >= 70]
    success = 0

    for idx in high_risk.index:
        future = df.loc[idx+1:idx+5]
        if not future.empty and future["Close"].min() < df.loc[idx]["Close"]:
            success += 1

    total = len(high_risk)
    accuracy = (success / total * 100) if total > 0 else 0
    return total, accuracy


def backtest_trade_performance(df):
    trades = []

    for i in range(len(df) - 12):
        row = df.iloc[i]

        if row["RiskScore"] >= 70:
            entry = row["Close"]
            sl = row["Low"]
            risk = entry - sl

            if risk <= 0:
                continue

            target = entry + (2 * risk)

            future = df.iloc[i+1:i+11]

            result = None
            r_multiple = 0

            for _, frow in future.iterrows():
                if frow["Low"] <= sl:
                    result = "SL Hit"
                    r_multiple = -1
                    break
                if frow["High"] >= target:
                    result = "Target Hit"
                    r_multiple = 2
                    break

            if result:
                trades.append(r_multiple)

    if not trades:
        return 0, 0, 0

    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / len(trades)) * 100
    avg_r = sum(trades) / len(trades)

    return len(trades), round(win_rate, 2), round(avg_r, 2)


def backtest_trade_performance(df):
    trades = []

    for i in range(len(df) - 12):
        row = df.iloc[i]

        if row["RiskScore"] >= 70:
            entry = row["Close"]
            sl = row["Low"]
            risk = entry - sl

            if risk <= 0:
                continue

            target = entry + (2 * risk)

            future = df.iloc[i+1:i+11]

            result = None
            r_multiple = 0

            for _, frow in future.iterrows():
                if frow["Low"] <= sl:
                    result = "SL Hit"
                    r_multiple = -1
                    break
                if frow["High"] >= target:
                    result = "Target Hit"
                    r_multiple = 2
                    break

            if result:
                trades.append(r_multiple)

    if not trades:
        return 0, 0, 0

    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / len(trades)) * 100
    avg_r = sum(trades) / len(trades)

    return len(trades), round(win_rate, 2), round(avg_r, 2)



# ================= ML RISK PREDICTION MODEL =================
from sklearn.ensemble import RandomForestClassifier

def train_risk_model(df):
    df_ml = df.copy()

    # Target: Did price fall in next 3 candles?
    df_ml["FutureDrop"] = df_ml["Close"].shift(-3) < df_ml["Close"]

    df_ml = df_ml.dropna()

    features = ["Volume", "PctChange", "RiskScore"]
    X = df_ml[features]
    y = df_ml["FutureDrop"].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model


def predict_next_risk(model, latest_row):
    # MASTER CONTROL
    if latest_row["RiskScore"] >= 70:
        return "High Chance of Fall 📉 (Manipulation Detected)"

    features = ["Volume", "PctChange", "RiskScore"]
    X_pred = latest_row[features].values.reshape(1, -1)
    pred = model.predict(X_pred)[0]

    return "High Chance of Fall 📉" if pred == 1 else "Low Risk of Fall 📈"

# ---------------- TABS ----------------
(tab_setup, tab_risk, tab_watch, tab_fii, tab_adv, tab_hype, tab_fundamentals, tab_portfolio) = st.tabs(
    [
        "📉 Candlesticks & Setup",
        "🚨 Operator Risk Scanner",
        "📋 Multi-Stock & Sector Risk",
        "💰 FII / DII Flows",
        "🧬 Advanced Forensics",
        "🔥 Market Hype",
        "📚 Fundamentals",
        "🧠 Portfolio AI"
    ]
)

# ---------------- TAB 1: Candles & Setup ----------------
with tab_setup:
    st.subheader("Chart Setup")
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        symbol = st.text_input("Stock symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")
        period = st.selectbox("History period", ["5d", "1mo", "3mo", "6mo", "1y"], index=2)
        interval = st.selectbox("Candle timeframe", ["5m", "15m", "30m", "60m", "1d"], index=4)
    with col_r:
        st.markdown("**Notes**")
        st.markdown("- Use `.NS` for NSE, `.BO` for BSE.")
        st.markdown("- Intraday intervals (5m..60m) work best with short periods.")
        st.markdown("- Daily candles work with 3mo+.")
    

    if not symbol.strip():
        st.warning("Enter a valid stock symbol to load the chart.")
    else:
        try:
            data_raw = yf.download(symbol.strip(), period=period, interval=interval, progress=False)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()
        if data_raw is None or data_raw.empty:
            st.error("No data returned. Try another symbol / period / timeframe.")
            st.stop()
        # flatten multiindex columns if present
        if isinstance(data_raw.columns, pd.MultiIndex):
            data_raw.columns = [c[0] for c in data_raw.columns]
        data = data_raw.reset_index()
        time_col = data.columns[0]
        # store in session
        st.session_state["symbol"] = symbol.strip()
        st.session_state["data_raw"] = data_raw
        st.session_state["data"] = data
        st.session_state["time_col"] = time_col
        st.markdown("""
### 🕯️ How to Read This Chart

This chart is not for predicting price.

It is for verifying whether the price movement is **natural** or **manipulated**.

🔹 Sudden big candles with high volume → suspicious activity  
🔹 Smooth movement with steady volume → organic price action  

Use this to judge **price honesty**, not direction.
""")

        st.subheader("📊 Candlestick Chart")
        # robust candlestick plotting
        df_plot = data.copy()
        if set(["Open", "High", "Low", "Close"]).issubset(df_plot.columns):
            fig = go.Figure(data=[go.Candlestick(
                x=df_plot[time_col],
                open=df_plot["Open"],
                high=df_plot["High"],
                low=df_plot["Low"],
                close=df_plot["Close"],
                increasing_line_color="#33aa44",
                decreasing_line_color="#d32f2f"
            )])
            fig.update_layout(template="plotly_dark", height=520, xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True, key=f"top_candle_{symbol.strip()}")
            with st.expander("📄 Show OHLCV data"):
                st.dataframe(df_plot, use_container_width=True)
        else:
            st.info("Insufficient OHLC data for candlestick chart.")
    # ================= tab one ends here  =================
with tab_risk:
    if "data_raw" not in st.session_state:
        st.warning("Load a chart in Tab 1 first.")
        st.stop()

    symbol = st.session_state["symbol"]
    data_raw = st.session_state["data_raw"]
    data = st.session_state["data"]
    time_col = st.session_state["time_col"]

    df = calc_risk_from_raw(data_raw).reset_index()
    df["CandleTag"] = df.apply(tag_candle, axis=1)

    reasons = explain_risk_with_system_features(df)
    latest = df.iloc[-1]
    problems = market_problem_mapper(df, latest)

    ml_model = train_risk_model(df)
    ml_prediction = predict_next_risk(ml_model, latest)
    
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown("**Latest Risk**")
        st.markdown(f"<div style='font-size:20px'>{int(latest['RiskScore'])} ({latest['RiskLevel']})</div>", unsafe_allow_html=True)
    with colB:
        max_idx = df["RiskScore"].idxmax()
        max_row = df.loc[max_idx]
        st.markdown("**Peak Risk in Selected Period**")
        st.write(f"Score **{int(max_row['RiskScore'])}** on **{pd.to_datetime(max_row[time_col]).date()}**")
        

    st.markdown("### 🔍 Why this Risk Score?")
    for r in reasons:
        st.write("•", r)

    st.markdown("### 🌍 Market Problems Detected")
    for p in problems:
        st.write("•", p)

    st.markdown("### 🤖 ML Prediction")
    st.info(ml_prediction)
    # ---------------- TAB 2: RISK ----------------
    st.markdown("### 🚨 Alerts Engine")

    # Define a simple alert_engine function to generate alerts based on risk levels
    def alert_engine(df):
        alerts = []
        latest = df.iloc[-1]
        if latest["RiskScore"] >= 70:
            alerts.append(("CRITICAL", "High risk detected! Operator activity likely."))
        elif latest["RiskScore"] >= 40:
            alerts.append(("WARNING", "Medium risk: Unstable price structure."))
        # Add more alert rules as needed
        return alerts

    alerts = alert_engine(df)

    if not alerts:
        st.success("🟢 No critical alerts")
    else:
        for level, msg in alerts:
            st.error(f"{level}: {msg}")
    st.markdown("### 🕯 Smart Candle Tags")
    st.dataframe(df[[time_col, "Close", "Volume", "RiskScore", "CandleTag"]])
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🚨 Risk Snapshot & Manipulation Meter")
    st.markdown("""
### 🚨 What is this Risk Meter?

This section tells you **whether operators / smart money are influencing the price**.

It does NOT tell buy or sell.

It tells:
- Is the move natural?
- Is retail getting trapped?
- Is volume supporting the move?

👉 If Risk is HIGH → Avoid the stock  
👉 If Risk is LOW → Price action looks organic
""")

    def compute_integrity_score(df_local: pd.DataFrame) -> int:
        if df_local.empty:
            return 0
        vol = df_local["PctChange"].abs().mean()
        high_days = (df_local["RiskScore"] >= 70).mean()
        med_days = (df_local["RiskScore"] >= 40).mean()
        raw = 100 - (vol * 2 + high_days * 100 + med_days * 40)
        return max(0, min(100, int(round(raw))))

    def detect_fake_breakout(df_local: pd.DataFrame) -> bool:
        if len(df_local) < 15:
            return False
        recent_window = min(20, len(df_local))
        recent = df_local.tail(recent_window)
        prior = df_local.iloc[:-recent_window]
        if prior.empty:
            return False
        prev_high = prior["Close"].max()
        recent_high = recent["Close"].max()
        last_close = df_local["Close"].iloc[-1]
        breakout = recent_high > prev_high * 1.02
        failure = breakout and last_close < prev_high * 0.99
        return bool(failure)

    def detect_retail_trap(df_local: pd.DataFrame) -> bool:
        if len(df_local) < 5:
            return False
        last = df_local.iloc[-1]
        body = abs(last["Close"] - last["Open"])
        upper_wick = last["High"] - max(last["Close"], last["Open"])
        lower_wick = min(last["Close"], last["Open"]) - last["Low"]
        recent_run = bool(df_local["Close"].iloc[-2] > df_local["Close"].iloc[-5]) if len(df_local) >= 5 else False
        trap = (last["Close"] < last["Open"]) and (upper_wick > body * 1.5) and (upper_wick > abs(lower_wick) * 1.2) and recent_run
        return bool(trap)

    def whale_footprint(df_local: pd.DataFrame) -> int:
        if df_local.empty:
            return 0
        last = df_local.iloc[-1]
        avg_vol = df_local["Volume"].rolling(20).mean().iloc[-1] if len(df_local) >= 20 else df_local["Volume"].mean()
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

    def price_volume_divergence(df_local: pd.DataFrame) -> Tuple[int, str]:
        if df_local.empty:
            return 0, "No signal"
        last = df_local.iloc[-1]
        vol_avg = df_local["Volume"].rolling(10).mean().iloc[-1] if len(df_local) >= 10 else df_local["Volume"].mean()
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

    integrity = compute_integrity_score(df)
    fake_bo = detect_fake_breakout(df)
    trap = detect_retail_trap(df)
    whale_score = whale_footprint(df)
    div_score, div_msg = price_volume_divergence(df)

    latest_risk = int(latest["RiskScore"])
    comps = [min(latest_risk, 100), 100 - int(integrity), int(whale_score), int(div_score)]
    if fake_bo:
        comps.append(80)
    if trap:
        comps.append(70)
    manip_idx = int(round(sum(comps) / len(comps))) if comps else 0
    manip_label = "High" if manip_idx >= 70 else "Medium" if manip_idx >= 40 else "Low"

    with colC:
        st.markdown("**Integrity Score (0–100)**")
        st.write(f"**{integrity}**")
    with colD:
        st.markdown("**Manipulation Index (0–100)**")
        color_emoji = "🟢" if manip_idx < 40 else "🟠" if manip_idx < 70 else "🔴"
        st.markdown(f"{color_emoji} **{manip_idx} — {manip_label}**")
        st.markdown("""
### 🧠 How to read this?

**Integrity Score**
- 80–100 → Price movement is clean and natural
- 60–80 → Slight operator footprints
- Below 60 → Price action is suspicious

**Manipulation Index**
- 🟢 Low → Normal trading behaviour
- 🟠 Medium → Be cautious, something unusual
- 🔴 High → Strong operator activity, avoid entry
""")

    st.markdown("### 🔍 Smart Operator Summary")
    summary_lines = [
        f"{symbol.upper()} closed at **{latest['Close']:.2f}**, daily move **{latest['PctChange']:.2f}%**.",
        f"Integrity: **{integrity}/100**.",
    ]
    if whale_score >= 70:
        summary_lines.append("🔍 Whale footprint: very strong — big player activity.")
    elif whale_score >= 45:
        summary_lines.append("🔍 Whale footprint: moderate.")
    else:
        summary_lines.append("🔍 Whale footprint: weak.")
    summary_lines.append(f"📊 Price–volume: {div_msg}.")
    if fake_bo:
        summary_lines.append("⚠️ Pattern: looks like a failed breakout.")
    if trap:
        summary_lines.append("⚠️ Pattern: possible retail trap candle.")
    if manip_idx >= 70:
        summary_lines.append("🧨 Combined indicators → high operator-driven risk.")
    elif manip_idx >= 40:
        summary_lines.append("🟠 Moderate manipulation risk — observe.")
    else:
        summary_lines.append("🟢 No strong manipulation cluster.")
    st.markdown("\n\n".join(summary_lines))
    
    #---------------------tab 2 ends here---------------------
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
        if dfw_raw is None or dfw_raw.empty:
            continue
        if isinstance(dfw_raw.columns, pd.MultiIndex):
            dfw_raw.columns = [c[0] for c in dfw_raw.columns]
        dfw = calc_risk_from_raw(dfw_raw).reset_index()
        last = dfw.iloc[-1]
        score = int(last["RiskScore"])
        rows.append({"Symbol": sym, "Sector": sector_map.get(sym.upper(), "Unknown"), "Date": last[dfw.columns[0]], "Close": last["Close"], "Volume": int(last["Volume"]), "PctChange": round(last["PctChange"], 2), "RiskScore": score, "RiskLevel": ("High" if score >= 70 else "Medium" if score >= 40 else "Low")})
    if not rows:
        st.info("No data loaded for watchlist. Check symbols.")
    else:
        watch_df = pd.DataFrame(rows).sort_values("RiskScore", ascending=False).reset_index(drop=True)
        st.dataframe(watch_df, use_container_width=True)
        st.download_button(
    "Download snapshot CSV",
    df_to_csv_bytes(watch_df),
    "snapshot.csv",
    "text/csv",
    key="btn_snapshot_csv_watchlist"
)
        sector_risk = watch_df.groupby("Sector")["RiskScore"].mean().reset_index()
        if not sector_risk.empty:
            fig_sec = go.Figure([go.Bar(x=sector_risk["Sector"], y=sector_risk["RiskScore"], marker=dict(color=["#d32f2f" if v >= 70 else "#ffcc33" if v >= 40 else "#33dd77" for v in sector_risk["RiskScore"]]))])
            fig_sec.update_layout(template="plotly_dark", height=380, xaxis_title="Sector", yaxis_title="Average Risk Score")
            st.plotly_chart(fig_sec, use_container_width=True, key=f"sector_risk_{symbol}")
        st.markdown("### ⭐ Persistent Watchlist")

    add_col, remove_col = st.columns(2)

    with add_col:
        new_stock = st.text_input("Add stock to watchlist", key="add_watch")
        if st.button("➕ Add"):
            if new_stock and new_stock.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_stock.upper())

    with remove_col:
        if st.session_state.watchlist:
            remove_stock = st.selectbox("Remove stock", st.session_state.watchlist)
            if st.button("➖ Remove"):
                st.session_state.watchlist.remove(remove_stock)

    st.markdown("**Your Watchlist:**")
    st.write(st.session_state.watchlist)    

# ---------------- TAB 4: FII / DII ----------------
with tab_fii:
    st.subheader("FII / DII Flow Visualisation")
    st.markdown("""
### 🏦 What This Means

This shows whether big institutions (FII/DII) are buying or selling.

- If institutions are selling and risk score is high → Stay away.
- If institutions are buying and risk is low → Strong zone.
""")
    col_demo, col_upload = st.columns([1, 1])
    with col_demo:
        st.markdown("**Download FII/DII CSV template**")
        template = pd.DataFrame({"Date": ["2025-12-01", "2025-12-02"], "FII": [1000, -500], "DII": [-200, 300]})
        st.download_button("Download FII/DII template", template.to_csv(index=False).encode("utf-8"), f"fii_template_{symbol}.csv", "text/csv", key=f"download_fii_template_{symbol}")
    with col_upload:
        fii_file = st.file_uploader(
    "Upload FII/DII CSV",
    type=["csv"],
    key="uploader_fii_csv"
)
    if fii_file is None:
        st.info("Upload CSV with columns Date, FII, DII (optional Net).")
    else:
        try:
            fii_df = pd.read_csv(fii_file, parse_dates=["Date"])
            if "Date" not in fii_df.columns:
                fii_df["Date"] = pd.to_datetime(fii_df.iloc[:, 0])
            if "Net" not in fii_df.columns and fii_df.shape[1] >= 3:
                fii_df["Net"] = fii_df.iloc[:, 1] + fii_df.iloc[:, 2]
            st.dataframe(fii_df.tail(50), use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fii_df["Date"], y=fii_df["Net"], name="Net Flow", marker_color="#d32f2f"))
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True, key=f"fii_flow_{symbol}")
            st.download_button("Download uploaded FII/DII", df_to_csv_bytes(fii_df), f"fii_uploaded_{symbol}.csv", "text/csv", key=f"download_fii_uploaded_{symbol}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ---------------- TAB 5: ADVANCED FORENSICS ----------------
with tab_adv:
    st.subheader("Operator Fingerprint & Forensic Analytics")
    st.markdown("""
### 🕵️ What is Operator Fingerprint?

This section detects **hidden operator patterns** from price and volume history.

It identifies if the stock has history of:

- **Pump & Dump** → Sudden rise then sharp fall (retail trap)
- **Accumulation → Spike** → Silent buying before breakout
- **Volume Crash Selloff** → Heavy dumping with volume
- **Laddering** → Continuous one-sided move to attract retail

Higher % means this pattern was strongly seen in recent history.

👉 This tells the *character* of the stock, not today's signal.
""")
    df_for = df.copy()

    def operator_fingerprint_scores(df_local: pd.DataFrame) -> Dict[str, int]:
        scores = {"Pump-Dump": 10, "Accumulation→Spike": 10, "VolumeCrashSelloff": 10, "Laddering": 10}
        if len(df_local) < 20:
            return scores
        seg = df_local.tail(min(60, len(df_local))).reset_index(drop=True)
        start_price = seg["Close"].iloc[0]
        max_price = seg["Close"].max()
        end_price = seg["Close"].iloc[-1]
        peak_idx = seg["Close"].idxmax()
        rise = (max_price / start_price) - 1 if start_price != 0 else 0
        fall_from_peak = (max_price - end_price) / max_price if max_price != 0 else 0
        if rise > 0.3 and fall_from_peak > 0.2 and peak_idx > len(seg) // 3:
            scores["Pump-Dump"] = 80
        elif rise > 0.2 and fall_from_peak > 0.1:
            scores["Pump-Dump"] = 50
        green = seg["Close"] > seg["PrevClose"]
        streaks = green.groupby((green != green.shift()).cumsum()).cumsum()
        long_streak = streaks.max() if not streaks.empty else 0
        if long_streak >= 5 and rise > 0.15:
            scores["Laddering"] = 70
        big_down = seg[seg["PctChange"] < -4]
        if not big_down.empty and (big_down["Volume"] > seg["Volume"].mean() * 1.5).any():
            scores["VolumeCrashSelloff"] = 75
        half = len(seg) // 2
        early = seg.iloc[:half]
        late = seg.iloc[half:]
        if not early.empty and not late.empty:
            if late["Volume"].mean() > early["Volume"].mean() * 1.4 and (late["Close"].max() / early["Close"].mean() - 1) > 0.12:
                scores["Accumulation→Spike"] = 70
        return scores
    

    scores = operator_fingerprint_scores(df_for)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🧬 Operator Patterns (probabilities)")
        for name, val in scores.items():
            st.write(f"- **{name}** → {val}%")
    with col2:
        st.markdown("#### 🌡 Integrity vs Suspicion")
        integrity_val = compute_integrity_score(df_for)
        high_days = int((df_for["RiskScore"] >= 70).sum())
        med_days = int((df_for["RiskScore"].between(40, 69)).sum())
        st.write(f"- Integrity Score: **{integrity_val}**")
        st.write(f"- High-risk candles: **{high_days}**, Medium-risk: **{med_days}**")

    
    st.markdown("### 🔥 Volume / Activity Heatmap")
    try:
        heat_df = df_for[[time_col, "Volume", "RiskScore"]].copy()
        heat_df["DateOnly"] = pd.to_datetime(heat_df[time_col]).dt.date
        heat_df["Day"] = pd.to_datetime(heat_df[time_col]).dt.day
        heat_df["Month"] = pd.to_datetime(heat_df[time_col]).dt.month
        pivot = heat_df.pivot_table(index="Month", columns="Day", values="Volume", aggfunc="mean")
        if not pivot.empty:
            fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis"))
            fig_heat.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_heat, use_container_width=True, key=f"heatmap_{symbol}")
        else:
            st.info("Not enough data to build heatmap.")
    except Exception:
        st.info("Heatmap generation failed for this dataset.")

    

# ---------------- TAB 6: FUNDAMENTALS ----------------
with tab_fundamentals:
    st.subheader("Fundamentals — company financials & filings")
    st.markdown("Enter a stock ticker (e.g., `INFY.NS`) and click Fetch. (FMP key optional for deeper data.)")

    left, right = st.columns([1.4, 1])
    with left:
        f_symbol = st.text_input("Ticker for fundamentals", value=symbol, key=f"f_symbol_{symbol}")
    with right:
        st.markdown("**Notes**")
        st.markdown("- Data pulled best-effort from yfinance; FMP used if API key available.")
        st.markdown("- For full filings, upload / paste annual report links or PDFs (future).")

    if st.button("Fetch fundamentals", key=f"fetch_fund_{symbol}"):
        if not f_symbol.strip():
            st.error("Enter a ticker symbol first.")
        else:
            with st.spinner("Fetching fundamentals..."):
                yf_t = yf.Ticker(f_symbol.strip())
                info = {}
                try:
                    info = yf_t.info or {}
                except Exception:
                    info = {}

                snapshot = {
                    "Symbol": f_symbol.strip().upper(),
                    "Name": info.get("shortName") or info.get("longName") or "-",
                    "Market Cap": format_indian_number(info.get("marketCap")),
                    "Price": info.get("currentPrice") or info.get("regularMarketPrice") or "-",
                    "P/E (TTM)": info.get("trailingPE") or "-",
                    "Forward P/E": info.get("forwardPE") or "-",
                    "Dividend Yield": (f"{info.get('dividendYield')*100:.2f}%" if isinstance(info.get('dividendYield'), (int, float)) else info.get("dividendYield") or "-"),
                    "Book Value": format_indian_number(info.get("bookValue")),
                    "ROE": (f"{info.get('returnOnEquity')*100:.2f}%" if isinstance(info.get('returnOnEquity'), (int, float)) else info.get("returnOnEquity") or "-"),
                    "Price/Book": info.get("priceToBook") or "-",
                    "52wk High/Low": f"{info.get('fiftyTwoWeekHigh', '-')}/{info.get('fiftyTwoWeekLow', '-')}",
                }

                # try FMP for richer data
                fmp_profile = None
                fmp_income = None
                fmp_bs = None
                fmp_cf = None
                fmp_share = None
                if FMP_API_KEY:
                    s_clean = f_symbol.strip().upper().replace(".NS", "")
                    try:
                        prof = call_fmp(f"profile/{s_clean}")
                        if prof:
                            fmp_profile = prof[0] if isinstance(prof, list) and len(prof) > 0 else prof
                        inc = call_fmp(f"income-statement/{s_clean}?limit=50")
                        bs = call_fmp(f"balance-sheet-statement/{s_clean}?limit=50")
                        cf = call_fmp(f"cash-flow-statement/{s_clean}?limit=50")
                        shp = call_fmp(f"ownership/{s_clean}")
                        fmp_income = pd.DataFrame(inc) if inc else None
                        fmp_bs = pd.DataFrame(bs) if bs else None
                        fmp_cf = pd.DataFrame(cf) if cf else None
                        fmp_share = shp
                    except Exception:
                        pass

                # quarterly via yfinance (best-effort)
                try:
                    q_pl = yf_t.quarterly_financials.T if getattr(yf_t, "quarterly_financials", None) is not None else pd.DataFrame()
                except Exception:
                    q_pl = pd.DataFrame()
                try:
                    q_bs = yf_t.quarterly_balance_sheet.T if getattr(yf_t, "quarterly_balance_sheet", None) is not None else pd.DataFrame()
                except Exception:
                    q_bs = pd.DataFrame()
                try:
                    q_cf = yf_t.quarterly_cashflow.T if getattr(yf_t, "quarterly_cashflow", None) is not None else pd.DataFrame()
                except Exception:
                    q_cf = pd.DataFrame()

                # --- Snapshot display ---
                st.markdown("### Snapshot")
                kcols = st.columns(3)
                items = list(snapshot.items())
                for i, (k, v) in enumerate(items):
                    coln = kcols[i % 3]
                    with coln:
                        st.markdown(f"<div class='fund-card'><div style='font-size:12px;color:#bdbdbd'>{k}</div><div style='font-size:16px'>{v}</div></div>", unsafe_allow_html=True)
                
                snap_df = pd.DataFrame({"Key": list(snapshot.keys()), "Value": list(snapshot.values())})
                snap_df = pd.DataFrame({"Key": list(snapshot.keys()), "Value": list(snapshot.values())})
                st.dataframe(snap_df, use_container_width=True)
                st.download_button(
    "Download snapshot CSV",
    df_to_csv_bytes(snap_df),
    "snapshot.csv",
    "text/csv",
    key="btn_snapshot_csv_fundamentals"
)

                # --- Price & multi-horizon charts (robust) ---
                st.markdown("### Price charts (multi-horizon & candlestick)")
                horizon = st.selectbox("Select horizon for analysis", ["1m", "6m", "1y", "3y", "5y", "10y", "max"], index=2, key=f"horizon_select_{f_symbol}")
                yf_period = period_mapping_for_yf(horizon)

                price_raw = None
                try:
                    price_raw = yf.download(f_symbol.strip(), period=yf_period, interval="1d", progress=False)
                except Exception:
                    price_raw = None

                price_df = None
                if price_raw is None or (isinstance(price_raw, pd.DataFrame) and price_raw.empty):
                    st.info("Price history not available for selected horizon.")
                else:
                    # flatten columns if multiindex
                    if isinstance(price_raw.columns, pd.MultiIndex):
                        price_raw.columns = [c[0] for c in price_raw.columns]
                    # safe numeric coercion
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        try:
                            if col in price_raw.columns:
                                price_raw[col] = pd.to_numeric(price_raw[col], errors="coerce")
                        except Exception:
                            price_raw[col] = np.nan
                    if not set(["Open", "High", "Low", "Close"]).issubset(price_raw.columns) or price_raw[["Open", "High", "Low", "Close"]].dropna(how="all").empty:
                        st.info("Not enough OHLC data to draw candlestick.")
                    else:
                        price_df = price_raw.dropna(subset=["Open", "High", "Low", "Close"]).copy()
                        price_df.index = pd.to_datetime(price_df.index)
                        price_df = price_df.sort_index()

                        candlestick = go.Figure()
                        candlestick.add_trace(go.Candlestick(
                            x=price_df.index,
                            open=price_df["Open"],
                            high=price_df["High"],
                            low=price_df["Low"],
                            close=price_df["Close"],
                            increasing_line_color="#33aa44",
                            decreasing_line_color="#d32f2f",
                            name="Price"
                        ))

                        try:
                            up_mask = (price_df["Close"] >= price_df["Open"])
                            vol_colors = ["#33aa44" if up else "#d32f2f" for up in up_mask]
                        except Exception:
                            vol_colors = ["#888888"] * len(price_df)

                        candlestick.add_trace(go.Bar(
                            x=price_df.index,
                            y=price_df["Volume"],
                            name="Volume",
                            marker_color=vol_colors,
                            yaxis="y2",
                            opacity=0.6,
                            hovertemplate="Vol: %{y}<extra></extra>"
                        ))

                        candlestick.update_layout(
                            template="plotly_dark",
                            height=520,
                            xaxis_rangeslider_visible=False,
                            yaxis_title="Price",
                            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume", rangemode="tozero")
                        )

                        st.plotly_chart(candlestick, use_container_width=True, key=f"candlestick_plot_{f_symbol}_{horizon}")

                        fig_close = go.Figure()
                        fig_close.add_trace(go.Bar(
                            x=price_df.index,
                            y=price_df["Close"],
                            marker_color=["#33aa44" if (price_df["Close"].iloc[i] >= price_df["Open"].iloc[i]) else "#d32f2f" for i in range(len(price_df))]
                        ))
                        fig_close.update_layout(template="plotly_dark", height=200)
                        st.plotly_chart(fig_close, use_container_width=True, key=f"close_bar_plot_{f_symbol}_{horizon}")

                        try:
                            idx = price_df.index
                            vals = price_df["Close"]
                            cagr_sel = compute_cagr_from_series(idx, vals)
                            if cagr_sel is not None:
                                st.markdown(f"- Price CAGR (selected horizon): *{(cagr_sel * 100):.2f}%*")
                            else:
                                st.markdown("- Price CAGR (selected horizon): —")
                        except Exception:
                            st.markdown("- Price CAGR (selected horizon): —")

                # --- Quarterly P&L ---
                st.markdown("### Quarterly Profit & Loss")
                if not q_pl.empty:
                    display_pl = q_pl.copy()
                    for c in display_pl.columns:
                        display_pl[c] = display_pl[c].apply(lambda x: format_indian_number(x) if (isinstance(x, (int, float)) and not pd.isna(x)) else x)
                    st.dataframe(display_pl, use_container_width=True)
                    st.download_button("Download quarterly P&L CSV", df_to_csv_bytes(q_pl.reset_index()), f"quarterly_pl_{f_symbol}.csv", "text/csv", key=f"download_quarterly_pl_{f_symbol}")
                elif fmp_income is not None:
                    st.dataframe(fmp_income.head(10).fillna("-"), use_container_width=True)
                    st.download_button("Download FMP income CSV", df_to_csv_bytes(fmp_income), f"fmp_income_{f_symbol}.csv", "text/csv", key=f"download_fmp_income_{f_symbol}")
                else:
                    st.info("Quarterly P&L not available via current sources.")

                # --- Quarterly Balance Sheet ---
                st.markdown("### Quarterly Balance Sheet")
                if not q_bs.empty:
                    display_bs = q_bs.copy()
                    for c in display_bs.columns:
                        display_bs[c] = display_bs[c].apply(lambda x: format_indian_number(x) if (isinstance(x, (int, float)) and not pd.isna(x)) else x)
                    st.dataframe(display_bs, use_container_width=True)
                    st.download_button("Download quarterly BS CSV", df_to_csv_bytes(q_bs.reset_index()), f"quarterly_bs_{f_symbol}.csv", "text/csv", key=f"download_quarterly_bs_{f_symbol}")
                elif fmp_bs is not None:
                    st.dataframe(fmp_bs.head(10).fillna("-"), use_container_width=True)
                    st.download_button("Download FMP BS CSV", df_to_csv_bytes(fmp_bs), f"fmp_bs_{f_symbol}.csv", "text/csv", key=f"download_fmp_bs_{f_symbol}")
                else:
                    st.info("Quarterly balance sheet not available.")

                # --- Quarterly Cash Flows ---
                st.markdown("### Quarterly Cash Flows")
                if not q_cf.empty:
                    display_cf = q_cf.copy()
                    for c in display_cf.columns:
                        display_cf[c] = display_cf[c].apply(lambda x: format_indian_number(x) if (isinstance(x, (int, float)) and not pd.isna(x)) else x)
                    st.dataframe(display_cf, use_container_width=True)
                    st.download_button("Download quarterly CF CSV", df_to_csv_bytes(q_cf.reset_index()), f"quarterly_cf_{f_symbol}.csv", "text/csv", key=f"download_quarterly_cf_{f_symbol}")
                elif fmp_cf is not None:
                    st.dataframe(fmp_cf.head(10).fillna("-"), use_container_width=True)
                    st.download_button("Download FMP CF CSV", df_to_csv_bytes(fmp_cf), f"fmp_cf_{f_symbol}.csv", "text/csv", key=f"download_fmp_cf_{f_symbol}")
                else:
                    st.info("Quarterly cash flows not available.")

                # --- Key ratios & derived working-cap metrics ---
                
                st.markdown("### Key ratios & derived working-cap metrics")
                derived = {}
                if fmp_income is not None and fmp_bs is not None:
                    try:
                        inc = fmp_income.copy()
                        bsdf = fmp_bs.copy()
                        rev_col = None
                        for cand in ["revenue", "totalRevenue", "revenueTotal", "sales"]:
                            for c in inc.columns:
                                if cand.lower() in c.lower():
                                    rev_col = c
                                    break
                            if rev_col:
                                break
                        rec_col = None; inv_col = None; pay_col = None
                        for cand in ["receivables", "accountsReceivable", "tradeReceivables", "netReceivables"]:
                            for c in bsdf.columns:
                                if cand.lower() in c.lower():
                                    rec_col = c
                                    break
                            if rec_col:
                                break
                        for cand in ["inventory", "inventories"]:
                            for c in bsdf.columns:
                                if cand.lower() in c.lower():
                                    inv_col = c
                                    break
                            if inv_col:
                                break
                        for cand in ["accountsPayable", "tradePayables", "payables"]:
                            for c in bsdf.columns:
                                if cand.lower() in c.lower():
                                    pay_col = c
                                    break
                            if pay_col:
                                break
                        if rec_col and rev_col:
                            latest_recs = bsdf[rec_col].iloc[0]
                            latest_rev = inc[rev_col].iloc[0]
                            derived["ReceivableDays"] = (latest_recs / latest_rev) * 365 if latest_rev and latest_recs else None
                        if inv_col and rev_col:
                            latest_inv = bsdf[inv_col].iloc[0]
                            derived["InventoryDays"] = (latest_inv / latest_rev) * 365 if latest_rev and latest_inv else None
                        if pay_col and rev_col:
                            latest_pay = bsdf[pay_col].iloc[0]
                            derived["PayableDays"] = (latest_pay / latest_rev) * 365 if latest_rev and latest_pay else None
                        if "ReceivableDays" in derived and "InventoryDays" in derived and "PayableDays" in derived:
                            derived["CCC"] = derived["ReceivableDays"] + derived["InventoryDays"] - derived["PayableDays"]
                    except Exception:
                        pass

                colset = st.columns(6)
                colset[0].metric("P/E", info.get("trailingPE") or "-")
                colset[1].metric("P/B", info.get("priceToBook") or "-")
                colset[2].metric("ROE", f"{info.get('returnOnEquity')*100:.2f}%" if isinstance(info.get('returnOnEquity'), (int, float)) else "-")
                colset[3].metric("Debt/Equity", info.get("debtToEquity") or "-")
                colset[4].metric("EV/EBITDA", "-")
                colset[5].metric("EPS (TTM)", info.get("trailingEps") or "-")

                st.markdown("**Derived working-cap**")
                dd1, dd2, dd3, dd4 = st.columns(4)
                dd1.metric("Receivable days", f"{derived.get('ReceivableDays','—'):.1f}" if derived.get('ReceivableDays') else "—")
                dd2.metric("Inventory days", f"{derived.get('InventoryDays','—'):.1f}" if derived.get('InventoryDays') else "—")
                dd3.metric("Payable days", f"{derived.get('PayableDays','—'):.1f}" if derived.get('PayableDays') else "—")
                dd4.metric("Cash Conversion Cycle", f"{derived.get('CCC','—'):.1f}" if derived.get('CCC') else "—")

                # --- Pros & Cons ---
                
                st.markdown("### Pros & Cons (auto summary)")
                pros = []
                cons = []
                try:
                    roe_val = info.get("returnOnEquity")
                    if isinstance(roe_val, (int, float)) and roe_val > 0.15:
                        pros.append(f"ROE strong ~ {roe_val*100:.1f}%")
                    de = info.get("debtToEquity")
                    if isinstance(de, (int, float)) and de < 0.6:
                        pros.append("Low Debt-to-Equity")
                    if info.get("dividendYield") and isinstance(info.get("dividendYield"), (int, float)) and info.get("dividendYield") > 0.02:
                        pros.append(f"Pays dividend ~ {info.get('dividendYield')*100:.2f}%")
                    if info.get("trailingPE") and isinstance(info.get("trailingPE"), (int, float)) and info.get("trailingPE") > 40:
                        cons.append(f"High P/E ({info.get('trailingPE'):.1f}) — valuation expensive")
                    if fmp_income is not None:
                        try:
                            inc = fmp_income
                            revc = None
                            for cand in ["revenue", "totalRevenue", "revenueTotal", "sales"]:
                                for c in inc.columns:
                                    if cand.lower() in c.lower():
                                        revc = c
                                        break
                                if revc:
                                    break
                            if revc:
                                rev_series = pd.to_numeric(inc[revc], errors="coerce").dropna()
                                if len(rev_series) >= 2:
                                    rev_series = rev_series[::-1].reset_index(drop=True)
                                    years = len(rev_series) - 1
                                    if years > 0 and rev_series.iloc[0] > 0:
                                        cagr_rev = (rev_series.iloc[-1] / rev_series.iloc[0]) ** (1 / years) - 1
                                        if cagr_rev > 0.12:
                                            pros.append(f"5yr Sales CAGR ~ {cagr_rev*100:.1f}%")
                                        else:
                                            cons.append(f"Sales growth muted (~{cagr_rev*100:.1f}% over period)")
                        except Exception:
                            pass
                except Exception:
                    pass
                if not pros:
                    pros.append("No strong pros flagged")
                if not cons:
                    cons.append("No strong cons flagged")
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown("#### PROS")
                    for p in pros:
                        st.write("- " + p)
                with pc2:
                    st.markdown("#### CONS")
                    for c in cons:
                        st.write("- " + c)

                # --- Peer comparison ---
                
                st.markdown("### Peer comparison")
                peer_input = st.text_input("Enter peer symbols separated by comma (e.g., TCS.NS,HCLTECH.NS)", value=f"{f_symbol.strip().upper()}", key=f"peer_input_{f_symbol}")
                if st.button("Fetch peers", key=f"peers_fetch_{f_symbol}"):
                    rows_peer = []
                    peer_list = [p.strip() for p in peer_input.split(",") if p.strip()]
                    for p in peer_list:
                        try:
                            tx = yf.Ticker(p)
                            pinf = tx.info or {}
                            rows_peer.append({
                                "Symbol": p,
                                "Name": pinf.get("shortName") or pinf.get("longName") or "-",
                                "CMP": pinf.get("regularMarketPrice") or pinf.get("previousClose") or "-",
                                "P/E": pinf.get("trailingPE") or "-",
                                "MarketCap": format_indian_number(pinf.get("marketCap")),
                                "ROE": (f"{pinf.get('returnOnEquity')*100:.1f}%" if isinstance(pinf.get('returnOnEquity'), (int, float)) else "-")
                            })
                        except Exception:
                            rows_peer.append({"Symbol": p, "Name": "Fetch failed"})
                    if rows_peer:
                        peer_df = pd.DataFrame(rows_peer)
                        st.dataframe(peer_df, use_container_width=True)
                        st.download_button(
    "Download peer CSV",
    df_to_csv_bytes(peer_df),
    "peers.csv",
    "text/csv",
    key="btn_peer_csv"
)

                # --- Growth metrics (CAGR) ---
                
                st.markdown("### Growth metrics (Price CAGR)")
                if price_df is not None and not price_df.empty:
                    pf = price_df.copy()
                    pf.index = pd.to_datetime(pf.index)
                    def price_cagr_for_years(years):
                        try:
                            end = pf.index.max()
                            start = end - pd.DateOffset(years=years)
                            sliced = pf[pf.index >= start]["Close"]
                            if len(sliced) < 10:
                                return None
                            return compute_cagr_from_series(sliced.index, sliced)
                        except Exception:
                            return None
                    p1 = price_cagr_for_years(1)
                    p3 = price_cagr_for_years(3)
                    p5 = price_cagr_for_years(5)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price CAGR (1y)", f"{p1*100:.2f}%" if p1 else "—")
                    c2.metric("Price CAGR (3y)", f"{p3*100:.2f}%" if p3 else "—")
                    c3.metric("Price CAGR (5y)", f"{p5*100:.2f}%" if p5 else "—")
                else:
                    st.info("Not enough price history to compute CAGR.")

                # --- Cash flows summary ---
                
                st.markdown("### Cash flow summary (annual / reported)")
                if fmp_cf is not None and not fmp_cf.empty:
                    try:
                        cf_view = fmp_cf[["date", "netCashProvidedByOperatingActivities", "netCashUsedForInvestingActivities", "netCashUsedProvidedByFinancingActivities", "netChangeInCash"]]
                        st.dataframe(cf_view.head(8).fillna("-"), use_container_width=True)
                        st.download_button("Download FMP cashflow CSV", df_to_csv_bytes(cf_view), f"fmp_cashflow_{f_symbol}.csv", "text/csv", key=f"download_fmp_cashflow_{f_symbol}")
                    except Exception:
                        st.info("Cashflow formatting issue.")
                elif not q_cf.empty:
                    disp = q_cf.copy()
                    for c in disp.columns:
                        disp[c] = disp[c].apply(lambda x: format_indian_number(x) if (isinstance(x, (int, float)) and not pd.isna(x)) else x)
                    st.dataframe(disp, use_container_width=True)
                    st.download_button("Download quarterly CF CSV (yfinance)", df_to_csv_bytes(q_cf.reset_index()), f"quarterly_cf_yf_{f_symbol}.csv", "text/csv", key=f"download_quarterly_cf_yf_{f_symbol}")
                else:
                    st.info("Cashflow not available from current sources. Use FMP key or upload reports.")

                # --- Shareholding pattern (Promoters) ---
                
                st.markdown("### Shareholding pattern (Promoters etc.)")
                if fmp_share:
                    st.json(fmp_share)
                else:
                    st.info("Shareholding not available via FMP (or no FMP key). You can upload a CSV below.")
                    share_template = pd.DataFrame({"Quarter": ["Mar-2025", "Dec-2024"], "Promoters": [14.3, 14.6], "FIIs": [30.0, 31.0], "DIIs": [39.0, 38.5], "Public": [16.7, 15.9]})
                    st.download_button("Download shareholding template", share_template.to_csv(index=False).encode("utf-8"), f"share_template_{f_symbol}.csv", "text/csv", key=f"download_share_template_{f_symbol}")
                    up = st.file_uploader(
    "Upload shareholding CSV",
    type=["csv"],
    key="uploader_shareholding"
)
                    if up:
                        try:
                            shdf = pd.read_csv(up)
                            st.dataframe(shdf, use_container_width=True)
                            st.download_button("Download uploaded shareholding", df_to_csv_bytes(shdf), f"uploaded_shareholding_{f_symbol}.csv", "text/csv", key=f"download_uploaded_share_{f_symbol}")
                        except Exception as e:
                            st.error("Failed to read shareholding CSV: " + str(e))

                # --- Recommendation / safety flag (quick) ---
                
                st.markdown("### Recommendation / Safety flag (quick)")
                score = 50
                try:
                    roe_val = info.get('returnOnEquity') or (fmp_profile.get('returnOnEquity') if fmp_profile else None)
                    pe_val = info.get('trailingPE') or (fmp_profile.get('pe') if fmp_profile else None)
                    debt_to_equity = info.get('debtToEquity') or (fmp_profile.get('debtToEquity') if fmp_profile else None)
                    if isinstance(roe_val, (int, float)) and roe_val > 0.15:
                        score += 15
                    if isinstance(debt_to_equity, (int, float)) and debt_to_equity < 0.6:
                        score += 10
                    if isinstance(pe_val, (int, float)) and pe_val > 30:
                        score -= 15
                except Exception:
                    pass
                if score >= 70:
                    rec = "BEST (Low risk / Good fundamentals)"
                elif score >= 50:
                    rec = "OK (Consider with caution)"
                else:
                    rec = "RISKY (High valuation / weak fundamentals)"
                st.markdown(f"**Quick score:** {score} → **{rec}**")

                st.success("Fundamentals fetched. Review above and download CSVs if needed.")

# 🔥 Market Hype (Google Trends)
with tab_hype:
    st.subheader("🔥 Market Attention Tracker")

    keyword = st.text_input(
        "Search keyword (company / stock name)",
        value=symbol.replace(".NS", "")
    )

    if "trend_score" not in st.session_state:
        st.session_state.trend_score = None
        st.session_state.trend_status = "Not checked"

    if st.button("Check Market Attention"):
        with st.spinner("Analyzing market attention..."):
            score, status = fetch_google_trends_score(keyword)

            # ---------- FALLBACK LOGIC ----------
            if score == 0:
                avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
                vol_ratio = latest["Volume"] / avg_vol if avg_vol > 0 else 1

                price_move = abs(latest["PctChange"])  # % move
                risk_boost = latest["RiskScore"] * 0.3  # use your own logic

                score = (
                    min(40, vol_ratio * 25) +
                    min(30, price_move * 6) +
                    min(30, risk_boost)
                )

                score = int(min(100, max(5, score)))
                status = "Google blocked → Volume + Price + Risk model"

            st.session_state.trend_score = score
            st.session_state.trend_status = status

    score = st.session_state.trend_score
    status = st.session_state.trend_status

    if score is None:
        st.info("Click the button to analyze attention.")
    else:
        st.metric("Hype Score (0–100)", score)
        st.write(f"Status: {status}")

        if score >= 60:
            st.success("🔥 High market attention")
        elif score >= 30:
            st.warning("🟠 Moderate attention")
        else:
            st.info("🟢 Low attention → Quiet zone")
# ================= PORTFOLIO HIDDEN RISK ANALYZER =================
with tab_portfolio:
    st.subheader("🧠 Portfolio Hidden Risk Analyzer (Upload Your Portfolio)")

    st.markdown("""
    Upload your portfolio Excel sheet.  
    This AI detects hidden **correlation**, **sector concentration**, and tells you **which stock to reduce/exit**.
    """)

    # ---------- TEMPLATE DOWNLOAD ----------
    template_df = pd.DataFrame({
        "StockSymbol": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
        "Quantity": [10, 15, 12],
        "BuyPrice": [1500, 900, 1000]
    })

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False)
    buf.seek(0)

    st.download_button(
        "📥 Download Portfolio Template",
        buf,
        file_name="portfolio_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="portfolio_template_download"
    )

    uploaded_file = st.file_uploader(
        "📤 Upload your portfolio Excel",
        type=["xlsx"],
        key="portfolio_file_upload"
    )

    # ---------- ANALYSIS FUNCTIONS ----------

    def analyze_portfolio(df_portfolio):
        symbols = df_portfolio["StockSymbol"].dropna().unique()

        price_frames = []
        sector_map = {}

        for s in symbols:
            dfp = yf.download(s, period="1y", interval="1d", progress=False)
            if dfp.empty or "Close" not in dfp.columns:
                continue

            temp = dfp[["Close"]].rename(columns={"Close": s})
            price_frames.append(temp)

            try:
                info = yf.Ticker(s).info
                sector_map[s] = info.get("sector", "Unknown")
            except:
                sector_map[s] = "Unknown"

        if len(price_frames) < 2:
            return None, None, None, 0.0, None

        df_prices = pd.concat(price_frames, axis=1).dropna()
        returns = df_prices.pct_change().dropna()
        corr_matrix = returns.corr()

        # Average correlation
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        vals = upper.stack().values
        avg_corr = float(vals.mean()) if len(vals) > 0 else 0.0

        sectors = pd.Series(sector_map)
        sector_counts = sectors.value_counts(normalize=True) * 100

        return df_prices, returns, corr_matrix, avg_corr, sector_counts


    def portfolio_exit_advice(returns_df, corr_matrix):
        volatility = returns_df.std()
        scores = {}

        for s in corr_matrix.columns:
            others = corr_matrix[s].drop(labels=[s])
            corr_mean = float(others.mean())
            vol = float(volatility.get(s, 0))
            scores[s] = (corr_mean * 0.7) + (vol * 0.3)

        risk_df = (
            pd.DataFrame(scores.items(), columns=["Stock", "RiskScore"])
            .sort_values("RiskScore", ascending=False)
            .reset_index(drop=True)
        )

        worst_stock = risk_df.iloc[0]["Stock"]
        return risk_df, worst_stock


    # ---------- PROCESS FILE ----------
    if uploaded_file:
        port_df = pd.read_excel(uploaded_file)

        if not {"StockSymbol", "Quantity", "BuyPrice"}.issubset(port_df.columns):
            st.error("❌ Invalid format. Use the template.")
            st.stop()

        st.success("✅ Portfolio loaded successfully")

        df_prices, returns, corr_matrix, avg_corr, sector_counts = analyze_portfolio(port_df)

        if corr_matrix is None:
            st.error("Need at least 2 valid stocks.")
            st.stop()

        # ---------- CORRELATION TABLE ----------
        st.markdown("### 🔗 Stock-to-Stock Correlation (Easy View)")

        corr_pairs = []
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                if i != j:
                    corr_pairs.append((i, j, round(float(corr_matrix.loc[i, j]), 2)))

        corr_df = (
            pd.DataFrame(corr_pairs, columns=["Stock A", "Stock B", "Correlation"])
            .sort_values("Correlation", ascending=False)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        st.dataframe(corr_df, use_container_width=True)

        # ---------- CORRELATION INSIGHT ----------
        st.markdown("### 🤖 Hidden Correlation Insight")

        if avg_corr > 0.75:
            st.error(f"🔴 Very High Correlation ({avg_corr:.2f}) — All stocks may fall together.")
        elif avg_corr > 0.5:
            st.warning(f"🟠 Moderate Correlation ({avg_corr:.2f}) — Partial diversification.")
        else:
            st.success(f"🟢 Low Correlation ({avg_corr:.2f}) — Good diversification.")

        # ---------- SECTOR CONCENTRATION ----------
        st.markdown("### 🏭 Sector Concentration")
        st.dataframe(sector_counts)

        top_sector = sector_counts.idxmax()
        top_pct = sector_counts.max()

        if top_pct > 60:
            st.error(f"🔴 {top_pct:.1f}% in {top_sector} — Sector crash = portfolio crash.")
        elif top_pct > 40:
            st.warning(f"🟠 {top_pct:.1f}% in {top_sector} — High sector dependency.")
        else:
            st.success("🟢 Well diversified across sectors.")

        # ---------- EXIT ADVICE ----------
        risk_df, worst_stock = portfolio_exit_advice(returns, corr_matrix)

        st.markdown("### 📉 Individual Stock Risk Contribution")
        st.dataframe(risk_df, use_container_width=True)

        st.markdown("### 🚨 AI Exit / Reduce Suggestion")
        st.error(f"""
    ✂️ **Reduce / Exit: {worst_stock}**

    This stock is highly correlated and volatile.  
    If market falls, this stock will drag your portfolio first.
    """)            

# ---------------- AUTO REFRESH ----------------
if auto_refresh:
    time.sleep(60)
    st.experimental_rerun()

st.markdown("### 🔗 Related Tools")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Open StoxEye"):
        st.markdown(
            "<meta http-equiv='refresh' content='0; url=https://stoxeyeeee-zjlonjhaemcc8gdcemdfo8.streamlit.app/'>",
            unsafe_allow_html=True
        )

with col2:
    if st.button("🛡 Open Secure First"):
        st.markdown(
            "<meta http-equiv='refresh' content='0; url=https://secure-first-calculator-fthrymhqbqmtnhecft4rgc.streamlit.app/'>",
            unsafe_allow_html=True
        )    

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">🛡️ Built with discipline and obsession by <b>venugAAdu</b>.</div>', unsafe_allow_html=True)
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()