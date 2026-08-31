# =============================================================================
# The Mountain Path Academy — Binomial Option Pricing Model
# Educational Streamlit App  |  Prof. V. Ravichandran
# https://themountainpathacademy.com
# -----------------------------------------------------------------------------
# Design system matched to the India–US Bond Yield Spread app
# (navy + gold "World of Finance" theme), with a Black–Scholes benchmark added.
# =============================================================================
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Binomial Option Pricing Model | The Mountain Path Academy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# BRAND PALETTE  (identical to the Bond Yield Spread app)
# -----------------------------------------------------------------------------
GOLD  = "#FFD700"
BLUE  = "#003366"
MID   = "#004d80"
CARD  = "#112240"
TXT   = "#e6f1ff"
MUTED = "#8892b0"
GRN   = "#28a745"
RED   = "#dc3545"
LB    = "#ADD8E6"
AMBER = "#f0ad4e"

LINK_ACADEMY = "https://themountainpathacademy.com"
LINK_LI      = "https://www.linkedin.com/in/trichyravis"
LINK_GH      = "https://github.com/trichyravis"

# -----------------------------------------------------------------------------
# GLOBAL STYLE
# -----------------------------------------------------------------------------
st.html(f"""
<style>
  .stApp {{
    background: linear-gradient(135deg,#1a2332,#243447,#2a3f5f) fixed;
  }}
  #MainMenu, header[data-testid="stHeader"], footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }}

  h1,h2,h3,h4 {{ color:{TXT}; letter-spacing:.2px; }}

  /* -------- Sidebar (kept, restyled navy + gold) -------- */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg,#0d1b30,#112240 70%,#0d1b30);
    border-right: 1px solid rgba(255,215,0,.22);
  }}
  [data-testid="stSidebar"] * {{ color:{TXT}; }}
  [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 {{
    color:{GOLD} !important; -webkit-text-fill-color:{GOLD} !important;
    font-size:1rem; text-transform:uppercase; letter-spacing:1.2px;
  }}
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
    color:{TXT} !important; -webkit-text-fill-color:{TXT} !important; font-weight:600;
  }}
  [data-testid="stSidebar"] hr {{ border-color: rgba(255,215,0,.18); }}

  /* -------- Tabs (gold pills) -------- */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px; background: rgba(17,34,64,.55); padding: 6px; border-radius: 12px;
    border: 1px solid rgba(255,215,0,.18); flex-wrap: wrap;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent; border-radius: 8px;
    padding: 8px 16px; font-weight: 600; font-size: 14px;
    color: #c7d3e8 !important; -webkit-text-fill-color: #c7d3e8 !important;
  }}
  .stTabs [data-baseweb="tab"] * {{
    color: #c7d3e8 !important; -webkit-text-fill-color: #c7d3e8 !important;
  }}
  .stTabs [aria-selected="true"] {{ background: {GOLD} !important; }}
  .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * {{
    color: {BLUE} !important; -webkit-text-fill-color: {BLUE} !important;
  }}
  .stTabs [data-baseweb="tab"] p {{ font-size: 14px; font-weight: 600; }}

  /* Force native widget text readable */
  [data-testid="stWidgetLabel"] *, .stRadio *, [data-baseweb="radio"] label * {{
    color: {TXT} !important; -webkit-text-fill-color: {TXT} !important;
  }}
  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {GOLD}; }}

  /* Buttons / download button */
  .stButton button, .stDownloadButton button {{
    background: {GOLD} !important; border: 1px solid {GOLD} !important;
    border-radius: 10px !important; font-weight: 800 !important;
  }}
  .stButton button p, .stButton button span, .stButton button div,
  .stDownloadButton button p, .stDownloadButton button span, .stDownloadButton button div {{
    color: {BLUE} !important; -webkit-text-fill-color: {BLUE} !important;
  }}

  /* Dataframes */
  div[data-testid="stDataFrame"] {{
    border:1px solid rgba(255,215,0,.20); border-radius:12px; overflow:hidden;
  }}

  /* Generic card look */
  .mp-card {{
    background: {CARD}; border: 1px solid rgba(255,215,0,.16);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,.28); user-select: none;
  }}
  .mp-card:hover {{ border-color: rgba(255,215,0,.42); }}

  .teach-card {{
    background: linear-gradient(135deg,{CARD},#16203c);
    border-left: 4px solid {GOLD}; border-radius: 12px;
    padding: 15px 18px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,.22);
    color: {TXT};
  }}
  .formula-box {{
    background: {CARD}; border: 1px solid rgba(255,215,0,.35);
    border-radius: 14px; padding: 16px 20px; color: {TXT}; line-height:1.7;
  }}
  .gold {{ color:{GOLD}; -webkit-text-fill-color:{GOLD}; font-weight:700; }}
  .lb   {{ color:{LB};  -webkit-text-fill-color:{LB}; }}
  .small-muted {{ color:{MUTED}; -webkit-text-fill-color:{MUTED}; font-size:.86rem; }}
</style>
""")


# Small helpers -----------------------------------------------------------------
def html(s: str):
    st.html(s)


def plotly_theme(fig, height=440, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT, family="Inter, Segoe UI, sans-serif", size=13),
        margin=dict(l=20, r=20, t=56, b=40),
        hoverlabel=dict(bgcolor=CARD, font_color=TXT, bordercolor=GOLD),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,215,0,.2)",
                    borderwidth=1) if legend else dict(),
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False,
                     linecolor="rgba(255,255,255,.2)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False,
                     linecolor="rgba(255,255,255,.2)")
    return fig


# =============================================================================
# MODEL
# =============================================================================
def calc_ud(mode, sigma, dt, u_input, d_input):
    if mode.startswith("3/"):
        u = math.exp(sigma * math.sqrt(dt))
        d = math.exp(-sigma * math.sqrt(dt))
    else:
        u, d = u_input, d_input
    return u, d


def risk_neutral_prob(r, q, dt, u, d):
    growth = math.exp((r - q) * dt)
    if abs(u - d) < 1e-14:
        return np.nan
    return (growth - d) / (u - d)


def payoff(s, k, option_type):
    return max(s - k, 0.0) if option_type == "Call" else max(k - s, 0.0)


def build_binomial(S0, K, r, q, T, n, option_type, exercise, u, d):
    dt = T / n
    p = risk_neutral_prob(r, q, dt, u, d)
    disc = math.exp(-r * dt)

    stock = np.full((n + 1, n + 1), np.nan)
    opt = np.full((n + 1, n + 1), np.nan)
    intrinsic = np.full((n + 1, n + 1), np.nan)
    continuation = np.full((n + 1, n + 1), np.nan)
    early = np.full((n + 1, n + 1), False, dtype=bool)

    for i in range(n + 1):
        for j in range(i + 1):
            # j = number of up moves; i-j = down moves
            stock[i, j] = S0 * (u ** j) * (d ** (i - j))
            intrinsic[i, j] = payoff(stock[i, j], K, option_type)

    opt[n, : n + 1] = intrinsic[n, : n + 1]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * opt[i + 1, j + 1] + (1 - p) * opt[i + 1, j])
            continuation[i, j] = cont
            if exercise == "American":
                opt[i, j] = max(intrinsic[i, j], cont)
                early[i, j] = intrinsic[i, j] > cont + 1e-10
            else:
                opt[i, j] = cont

    return {
        "dt": dt,
        "p": p,
        "disc": disc,
        "stock": stock,
        "option": opt,
        "intrinsic": intrinsic,
        "continuation": continuation,
        "early": early,
        "value": float(opt[0, 0]),
    }


# ---- Black–Scholes closed-form benchmark ------------------------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, r, q, T, sigma, option_type):
    """European Black–Scholes–Merton price with continuous dividend yield q."""
    if T <= 0 or sigma <= 0:
        fwd = S * math.exp(-q * T)
        strike_pv = K * math.exp(-r * T)
        return max(fwd - strike_pv, 0.0) if option_type == "Call" else max(strike_pv - fwd, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "Call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def effective_sigma(mode, sigma, u, dt):
    """Volatility fed to Black–Scholes.
    Volatility mode: use sigma directly.
    Manual u/d mode: back out the CRR-implied sigma from u = exp(sigma*sqrt(dt))."""
    if mode.startswith("3/"):
        return sigma
    if u > 0 and dt > 0:
        return math.log(u) / math.sqrt(dt)
    return 0.0


def node_dataframe(res, n):
    rows = []
    for i in range(n + 1):
        for j in range(i + 1):
            cont = res["continuation"][i, j]
            rows.append({
                "Step": i,
                "Up Moves": j,
                "Down Moves": i - j,
                "Stock Price": res["stock"][i, j],
                "Intrinsic Payoff": res["intrinsic"][i, j],
                "Continuation Value": np.nan if np.isnan(cont) else cont,
                "Option Value": res["option"][i, j],
                "Early Exercise": bool(res["early"][i, j]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# FIGURES  (routed through plotly_theme + brand colors)
# =============================================================================
TREE_SCALE = [[0.0, MID], [0.5, LB], [1.0, GOLD]]


def tree_figure(matrix, n, title, value_prefix="₹", early=None):
    fig = go.Figure()
    xs, ys, vals, texts, hover = [], [], [], [], []
    for i in range(n + 1):
        for j in range(i + 1):
            x = i
            y = 2 * j - i
            val = matrix[i, j]
            xs.append(x); ys.append(y); vals.append(val)
            marker = " ★" if early is not None and early[i, j] else ""
            texts.append(f"{value_prefix}{val:,.2f}{marker}")
            hover.append(f"Step {i}<br>Up moves {j}<br>Value {val:,.6f}")
    # edges
    for i in range(n):
        for j in range(i + 1):
            x0, y0 = i, 2 * j - i
            for j2 in (j, j + 1):
                x1, y1 = i + 1, 2 * j2 - (i + 1)
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                         line=dict(color="rgba(255,215,0,.28)", width=1.4),
                                         hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
        marker=dict(size=18, color=vals, colorscale=TREE_SCALE,
                    line=dict(color=GOLD, width=1.1),
                    showscale=n >= 8, colorbar=dict(title="Value")),
        textfont=dict(color=TXT, size=11), hovertext=hover, hoverinfo="text",
        showlegend=False))
    fig = plotly_theme(fig, height=max(430, min(760, 42 * n + 390)), legend=False)
    fig.update_layout(title=title)
    fig.update_xaxes(title="Time Step")
    fig.update_yaxes(title="Tree Position", showticklabels=False)
    return fig


def payoff_figure(S0, K, price, option_type):
    smax = max(S0, K) * 1.75
    x = np.linspace(0.25 * min(S0, K), smax, 220)
    intrinsic = np.maximum(x - K, 0) if option_type == "Call" else np.maximum(K - x, 0)
    pnl = intrinsic - price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=intrinsic, name="Expiry payoff",
                             line=dict(color=GOLD, width=3)))
    fig.add_trace(go.Scatter(x=x, y=pnl, name="Buyer P/L after premium",
                             line=dict(color=LB, width=3, dash="dash")))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color=MUTED)
    fig.add_vline(x=K, line_width=1, line_dash="dot", line_color=AMBER,
                  annotation_text="Strike", annotation_font_color=AMBER)
    fig = plotly_theme(fig, height=440)
    fig.update_layout(title=f"{option_type} payoff and buyer P/L",
                      legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(title="Underlying price at expiry")
    fig.update_yaxes(title="Value")
    return fig


def convergence_figure(S0, K, r, q, T, option_type, exercise, mode, sigma,
                       u_input, d_input, bs_val=None, max_n=50):
    points = sorted(set([1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, max_n]))
    vals, valid = [], []
    for n0 in points:
        dt = T / n0
        u, d = calc_ud(mode, sigma, dt, u_input, d_input)
        p = risk_neutral_prob(r, q, dt, u, d)
        if 0 <= p <= 1:
            vals.append(build_binomial(S0, K, r, q, T, n0, option_type, exercise, u, d)["value"])
            valid.append(n0)
    fig = go.Figure(go.Scatter(x=valid, y=vals, mode="lines+markers",
                               name="Binomial value", line=dict(color=GOLD, width=3),
                               marker=dict(color=GOLD, size=7)))
    if bs_val is not None and exercise == "European":
        fig.add_hline(y=bs_val, line_dash="dash", line_color=LB,
                      annotation_text=f"Black–Scholes ₹{bs_val:,.2f}",
                      annotation_font_color=LB)
    fig = plotly_theme(fig, height=440)
    fig.update_layout(title="Price convergence as the number of steps increases")
    fig.update_xaxes(title="Steps")
    fig.update_yaxes(title="Option value")
    return fig


# =============================================================================
# EXCEL EXPORT
# =============================================================================
def make_excel(inputs, res, n, mode, bs_val, bs_sigma):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        navy_fmt = wb.add_format({"bg_color": "#003366", "font_color": "#FFFFFF", "bold": True, "border": 1, "border_color": "#FFD700"})
        title_fmt = wb.add_format({"bg_color": "#112240", "font_color": "#FFD700", "bold": True, "font_size": 18, "align": "center", "valign": "vcenter"})
        section_fmt = wb.add_format({"bg_color": "#0B2747", "font_color": "#FFD700", "bold": True, "font_size": 12, "border": 1, "border_color": "#FFD700"})
        input_fmt = wb.add_format({"bg_color": "#FFF9E8", "font_color": "#071A2F", "border": 1, "border_color": "#D9D9D9"})
        num_fmt = wb.add_format({"num_format": "0.0000", "border": 1, "border_color": "#D9D9D9"})
        money_fmt = wb.add_format({"num_format": "#,##0.00", "border": 1, "border_color": "#D9D9D9"})
        pct_fmt = wb.add_format({"num_format": "0.0000%", "border": 1, "border_color": "#D9D9D9"})
        note_fmt = wb.add_format({"font_color": "#666666", "italic": True, "text_wrap": True})
        early_fmt = wb.add_format({"bg_color": "#FFE5E5", "font_color": "#9C0006", "bold": True, "border": 1, "border_color": "#D9D9D9"})

        # Summary & inputs
        ws = wb.add_worksheet("Summary & Inputs")
        writer.sheets["Summary & Inputs"] = ws
        ws.set_column("A:A", 30); ws.set_column("B:B", 22); ws.set_column("C:C", 46)
        ws.set_row(0, 30)
        ws.merge_range("A1:C1", "THE MOUNTAIN PATH ACADEMY  |  BINOMIAL OPTION PRICING MODEL", title_fmt)
        ws.write("A3", "Model setup", section_fmt); ws.write("B3", "Value", section_fmt); ws.write("C3", "Teaching note", section_fmt)
        rows = [
            ("Model mode", mode, "Manual u/d or CRR volatility-based tree"),
            ("Option type", inputs["Option Type"], "Call or Put"),
            ("Exercise style", inputs["Exercise"], "European: expiry only; American: early exercise allowed"),
            ("Spot price S0", inputs["S0"], "Current underlying price"),
            ("Strike K", inputs["K"], "Contract strike price"),
            ("Risk-free rate r", inputs["r"], "Continuously compounded annual rate"),
            ("Dividend yield q", inputs["q"], "Continuously compounded annual yield"),
            ("Maturity T", inputs["T"], "Years to expiry"),
            ("Steps n", n, "2, 3 or multi-period"),
            ("Volatility sigma", inputs["sigma"], "Used only in volatility-based mode"),
            ("Up factor u", inputs["u"], "Per-step up multiplier"),
            ("Down factor d", inputs["d"], "Per-step down multiplier"),
            ("Delta t", res["dt"], "T / n"),
            ("Risk-neutral p", res["p"], "[exp((r-q)dt)-d] / (u-d)"),
            ("Discount factor", res["disc"], "exp(-r*dt)"),
            ("Binomial option value", res["value"], "Root node value"),
            ("Black-Scholes (European)", bs_val, "Closed-form benchmark"),
            ("BS volatility used", bs_sigma, "sigma (vol mode) or ln(u)/sqrt(dt) (manual)"),
            ("Binomial - BS", res["value"] - bs_val, "Convergence gap vs closed form"),
        ]
        for rr, (k, v, note) in enumerate(rows, 3):
            ws.write(rr, 0, k, input_fmt)
            if isinstance(v, (int, float, np.floating)):
                fmt = pct_fmt if k in {"Risk-free rate r", "Dividend yield q", "Volatility sigma", "Risk-neutral p", "BS volatility used"} else money_fmt if k in {"Spot price S0", "Strike K", "Binomial option value", "Black-Scholes (European)", "Binomial - BS"} else num_fmt
                ws.write(rr, 1, float(v), fmt)
            else:
                ws.write(rr, 1, v, input_fmt)
            ws.write(rr, 2, note, note_fmt)

        # Node detail table
        df = node_dataframe(res, n)
        df.to_excel(writer, sheet_name="Node Details", index=False, startrow=2)
        wnd = writer.sheets["Node Details"]
        wnd.merge_range("A1:H1", "NODE-BY-NODE CALCULATION", title_fmt)
        wnd.set_column("A:C", 12); wnd.set_column("D:G", 20); wnd.set_column("H:H", 16)
        for c, col in enumerate(df.columns):
            wnd.write(2, c, col, navy_fmt)
        wnd.freeze_panes(3, 0)
        for rownum in range(3, 3 + len(df)):
            if bool(df.iloc[rownum - 3]["Early Exercise"]):
                wnd.set_row(rownum, None, early_fmt)

        # Trees in grid form
        for sheet_name, matrix in [("Stock Tree", res["stock"]), ("Option Tree", res["option"]),
                                   ("Intrinsic Tree", res["intrinsic"]), ("Continuation Tree", res["continuation"])]:
            w = wb.add_worksheet(sheet_name); writer.sheets[sheet_name] = w
            w.merge_range(0, 0, 0, n + 1, sheet_name.upper(), title_fmt)
            w.write(2, 0, "Step / Up moves", navy_fmt)
            for j in range(n + 1):
                w.write(2, j + 1, j, navy_fmt)
            for i in range(n + 1):
                w.write(i + 3, 0, i, navy_fmt)
                for j in range(i + 1):
                    val = matrix[i, j]
                    if np.isnan(val):
                        w.write_blank(i + 3, j + 1, None, num_fmt)
                    else:
                        w.write(i + 3, j + 1, float(val), money_fmt)
            w.set_column(0, 0, 17); w.set_column(1, n + 1, 14); w.freeze_panes(3, 1)

        # Formula sheet
        wf = wb.add_worksheet("Formula Sheet"); writer.sheets["Formula Sheet"] = wf
        wf.set_column("A:A", 28); wf.set_column("B:B", 60); wf.set_column("C:C", 62)
        wf.merge_range("A1:C1", "BINOMIAL OPTION PRICING — FORMULA SHEET", title_fmt)
        wf.write_row("A3", ["Concept", "Formula", "Interpretation"], navy_fmt)
        formulas = [
            ("Step size", "dt = T / n", "Length of each binomial interval"),
            ("Manual up/down", "u and d are supplied", "Use when up-factor/down-factor are directly given"),
            ("CRR up factor", "u = exp(sigma*sqrt(dt))", "Volatility determines the up move"),
            ("CRR down factor", "d = exp(-sigma*sqrt(dt)) = 1/u", "Reciprocal down move"),
            ("Risk-neutral probability", "p = [exp((r-q)dt) - d] / (u-d)", "Uses dividend yield q when present"),
            ("Stock node", "S(i,j)=S0 * u^j * d^(i-j)", "j is number of up moves"),
            ("Call payoff", "max(S-K,0)", "Terminal intrinsic value"),
            ("Put payoff", "max(K-S,0)", "Terminal intrinsic value"),
            ("European backward induction", "V = exp(-r*dt)[p*Vu+(1-p)*Vd]", "No early exercise"),
            ("American value", "V = max(Intrinsic, Continuation)", "Exercise whenever intrinsic exceeds continuation"),
            ("Black-Scholes (call)", "C = S e^-qT N(d1) - K e^-rT N(d2)", "European closed-form benchmark"),
            ("d1 / d2", "d1=[ln(S/K)+(r-q+0.5s^2)T]/(s*sqrt(T)); d2=d1-s*sqrt(T)", "As steps rise the binomial price converges to this"),
        ]
        for rr, row in enumerate(formulas, 3):
            for cc, val in enumerate(row):
                wf.write(rr, cc, val, input_fmt if cc == 0 else note_fmt)

    output.seek(0)
    return output.getvalue()


# =============================================================================
# HEADER
# =============================================================================
html(f"""
<div style="background:linear-gradient(90deg,{BLUE},{MID});border-radius:16px;
     padding:22px 26px;border:1px solid rgba(255,215,0,.3);user-select:none;
     box-shadow:0 6px 24px rgba(0,0,0,.35);margin-bottom:10px;">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
    <div style="font-size:34px;-webkit-text-fill-color:initial;">📈</div>
    <div style="flex:1;min-width:260px;">
      <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:13px;
           font-weight:700;letter-spacing:2px;">THE MOUNTAIN PATH ACADEMY · WORLD OF FINANCE</div>
      <div style="color:#ffffff;-webkit-text-fill-color:#ffffff;font-size:26px;
           font-weight:800;line-height:1.15;margin-top:2px;">
           Binomial Option Pricing Model</div>
      <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:14px;margin-top:3px;">
           Build the tree · risk-neutral valuation · European vs American exercise ·
           inspect every node · benchmark against Black–Scholes · export the full workbook</div>
    </div>
    <div style="text-align:right;min-width:150px;">
      <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:12px;">Educational Series by</div>
      <div style="color:#ffffff;-webkit-text-fill-color:#ffffff;font-size:15px;font-weight:700;">Prof. V. Ravichandran</div>
      <a href="{LINK_ACADEMY}" target="_blank"
         style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:12px;text-decoration:none;">
         themountainpathacademy.com ↗</a>
    </div>
  </div>
</div>
""")

# =============================================================================
# SIDEBAR — MODEL CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### Model Controls")
    mode = st.selectbox("Pricing setup", [
        "1/ Without volatility · Without yield",
        "2/ Without volatility · With yield",
        "3/ With volatility · With yield",
    ])
    option_type = st.radio("Option type", ["Call", "Put"], horizontal=True)
    exercise = st.radio("Exercise style", ["European", "American"], horizontal=True)
    period_choice = st.selectbox("Tree horizon", ["2-period", "3-period", "Multi-period"])
    n = 2 if period_choice == "2-period" else 3 if period_choice == "3-period" else st.slider("Number of steps", 4, 60, 10)

    st.markdown("---")
    st.markdown("### Contract Inputs")
    S0 = st.number_input("Spot price S₀", min_value=0.01, value=100.0, step=1.0)
    K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Time to expiry T (years)", min_value=0.01, value=1.0, step=0.25)
    r_pct = st.number_input("Risk-free rate r (%)", value=5.0, step=0.25)
    r = r_pct / 100

    q = 0.0
    if mode != "1/ Without volatility · Without yield":
        q_pct = st.number_input("Dividend / yield q (%)", value=2.0, step=0.25)
        q = q_pct / 100

    sigma = 0.20
    u_input, d_input = 1.20, 0.85
    if mode.startswith("3/"):
        sigma_pct = st.number_input("Annual volatility σ (%)", min_value=0.01, value=20.0, step=1.0)
        sigma = sigma_pct / 100
    else:
        st.markdown("### Manual Tree Factors")
        u_input = st.number_input("Up factor u", min_value=0.0001, value=1.20, step=0.01, format="%.4f")
        d_input = st.number_input("Down factor d", min_value=0.0001, value=0.85, step=0.01, format="%.4f")

    st.markdown("---")
    st.caption("Rates and yield are continuously compounded annual rates. "
               "In volatility mode the app uses the Cox–Ross–Rubinstein (CRR) specification.")

# =============================================================================
# CALC
# =============================================================================
dt = T / n
u, d = calc_ud(mode, sigma, dt, u_input, d_input)
p = risk_neutral_prob(r, q, dt, u, d)
valid = np.isfinite(p) and 0 <= p <= 1 and u > d

if not valid:
    st.error(f"No-arbitrage condition is violated for these inputs. The risk-neutral probability is "
             f"p = {p:.4f}. Adjust u, d, r, q, T or the number of steps so that 0 ≤ p ≤ 1 and u > d.")
    st.stop()

res = build_binomial(S0, K, r, q, T, n, option_type, exercise, u, d)
df_nodes = node_dataframe(res, n)

# Black–Scholes benchmark
bs_sigma = effective_sigma(mode, sigma, u, dt)
bs_val = bs_price(S0, K, r, q, T, bs_sigma, option_type)
bs_gap = res["value"] - bs_val

# =============================================================================
# SUMMARY METRICS
# =============================================================================
metrics = [
    ("Binomial Value", f"₹{res['value']:,.2f}", f"{exercise} {option_type}", "Root node price", GOLD),
    ("Black–Scholes", f"₹{bs_val:,.2f}", "European closed form", "Convergence benchmark", LB),
    ("Risk-Neutral p", f"{p:.2%}", "up probability", "No-arbitrage weight", GOLD),
    ("Up / Down u·d", f"{u:.3f} / {d:.3f}", "per step", ("CRR from σ" if mode.startswith("3/") else "manual factors"), LB),
    ("Δt", f"{res['dt']:.4f}", "years / step", f"T / n = {T:.2f} / {n}", TXT),
    ("Early-Exercise", f"{int(res['early'].sum())}", "nodes", ("American only" if exercise == "American" else "European: none"), AMBER),
]
cols = st.columns(6)
for c, (label, val, sub, note, col) in zip(cols, metrics):
    with c:
        html(f"""
        <div class="mp-card" style="text-align:left;padding:15px 16px;min-height:118px;">
          <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:11px;
               font-weight:600;text-transform:uppercase;letter-spacing:1px;">{label}</div>
          <div style="color:{col};-webkit-text-fill-color:{col};font-size:23px;
               font-weight:800;margin:4px 0;">{val}</div>
          <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:11.5px;">{sub}</div>
          <div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:11.5px;
               margin-top:5px;opacity:.85;">{note}</div>
        </div>""")

# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs([
    "🎓 Learn",
    "🌳 Price the Option",
    "🔎 Node Analysis",
    "📊 Analytics",
    "🧮 Worked Example",
    "⬇️ Excel Export",
])

# -----------------------------------------------------------------------------
# TAB 1 — LEARN
# -----------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([1.15, .85])
    with c1:
        html(f"""
        <div class="mp-card">
          <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:17px;
               font-weight:700;margin-bottom:8px;">How the binomial model works</div>
          <div class="teach-card"><b class="gold">1 · Build the stock-price tree.</b><br>
            At every interval the stock moves up by <span class="gold">u</span> or down by
            <span class="gold">d</span>. A node after <i>j</i> up moves at step <i>i</i> is
            <b>S(i,j)=S₀·uʲ·d⁽ⁱ⁻ʲ⁾</b>.</div>
          <div class="teach-card"><b class="gold">2 · Move to the risk-neutral world.</b><br>
            The expected growth of the underlying after yield is matched using
            <b>p = [e<sup>(r−q)Δt</sup> − d] / (u − d)</b>.</div>
          <div class="teach-card"><b class="gold">3 · Calculate terminal payoff.</b><br>
            Call: <b>max(S−K, 0)</b> &nbsp;·&nbsp; Put: <b>max(K−S, 0)</b>.</div>
          <div class="teach-card"><b class="gold">4 · Work backwards.</b><br>
            European value = discounted risk-neutral expected next-node value. American value =
            <b>max(intrinsic, continuation)</b> at every node.</div>
        </div>
        <div class="mp-card" style="border-color:rgba(255,215,0,.42);">
          <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:15px;font-weight:700;margin-bottom:6px;">
            The three setups in this lab</div>
          <div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:14px;line-height:1.65;">
            <b>Setup 1 — no volatility, no yield:</b> you directly specify <span class="gold">u</span> and
            <span class="gold">d</span>; q = 0.<br>
            <b>Setup 2 — no volatility, with yield:</b> you specify u, d and q.<br>
            <b>Setup 3 — volatility with yield:</b> the app computes u = e<sup>σ√Δt</sup> and
            d = e<sup>−σ√Δt</sup> using CRR, then folds q into the risk-neutral probability.</div>
        </div>
        <div class="mp-card" style="border-color:rgba(173,216,230,.42);background:linear-gradient(135deg,{CARD},#16203c);">
          <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:14px;font-weight:700;margin-bottom:4px;">
            Teaching point — what p really is</div>
          <div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:13.5px;line-height:1.6;">
            <b>p</b> is <i>not</i> a forecast of the real-world chance the stock rises. It is the probability
            that makes discounted expected values consistent with <b>no-arbitrage pricing</b>. As the number
            of steps grows, the CRR binomial price converges to the <span class="lb">Black–Scholes</span>
            closed form — shown live in the Analytics tab.</div>
        </div>
        """)
    with c2:
        html(f"""<div class="mp-card"><div style="color:{GOLD};-webkit-text-fill-color:{GOLD};
             font-size:16px;font-weight:700;margin-bottom:6px;">Formula reference</div></div>""")
        ref = ASSETS / ("binomial_key_formulae_with_yield.png" if q > 0 else "binomial_key_formulae_no_yield.png")
        if ref.exists():
            st.image(str(ref), width="stretch")
        st.caption("Formula reference adapted directly from the attached teaching deck.")

# -----------------------------------------------------------------------------
# TAB 2 — PRICE THE OPTION
# -----------------------------------------------------------------------------
with tabs[1]:
    # Binomial vs Black–Scholes comparison card
    tol = max(0.01, 0.005 * max(bs_val, 1.0))
    if exercise == "European":
        gap_col = GRN if abs(bs_gap) < tol else AMBER
        verdict = ("Converged — matches Black–Scholes closely" if abs(bs_gap) < tol
                   else "Discretisation gap — raise the number of steps to converge")
        note_line = (f"Both price the same European {option_type.lower()}. The binomial value approaches "
                     f"Black–Scholes as steps → ∞ (currently n = {n}).")
    else:
        gap_col = LB
        verdict = "American vs European benchmark — early exercise can add value"
        note_line = (f"Black–Scholes prices the <b>European</b> option. For an American {option_type.lower()} the "
                     f"binomial price is ≥ this benchmark; the difference is the early-exercise premium.")
    html(f"""
    <div class="mp-card" style="border-color:{gap_col};">
      <div style="display:flex;flex-wrap:wrap;gap:26px;align-items:flex-end;justify-content:space-between;">
        <div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:12px;">Binomial ({exercise})</div>
          <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:30px;font-weight:800;">₹{res['value']:,.4f}</div></div>
        <div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:12px;">Black–Scholes (European)</div>
          <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:30px;font-weight:800;">₹{bs_val:,.4f}</div></div>
        <div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:12px;">Binomial − BS</div>
          <div style="color:{gap_col};-webkit-text-fill-color:{gap_col};font-size:30px;font-weight:800;">{bs_gap:+.4f}</div></div>
        <div><div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:12px;">σ used for BS</div>
          <div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:30px;font-weight:800;">{bs_sigma:.2%}</div></div>
      </div>
      <div style="margin-top:10px;color:{gap_col};-webkit-text-fill-color:{gap_col};font-weight:700;font-size:14px;">{verdict}</div>
      <div style="margin-top:4px;color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:12.5px;line-height:1.5;">{note_line}</div>
    </div>
    """)

    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:16px;margin:6px 0;'>Stock-price tree</div>",
                unsafe_allow_html=True)
    if n <= 15:
        st.plotly_chart(tree_figure(res["stock"], n, "Underlying Stock-Price Tree", early=None),
                        width="stretch")
    else:
        st.info("For clarity, the graphical tree is displayed for up to 15 steps. The full multi-period "
                "values remain available in Node Analysis and Excel.")
        st.plotly_chart(tree_figure(res["stock"], 15, "First 15 Steps — Stock-Price Tree", early=None),
                        width="stretch")

    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:16px;margin:6px 0;'>Option-value tree</div>",
                unsafe_allow_html=True)
    if n <= 15:
        st.plotly_chart(tree_figure(res["option"], n, f"{exercise} {option_type} — Option Value Tree",
                                    early=res["early"] if exercise == "American" else None),
                        width="stretch")
    else:
        st.dataframe(df_nodes[df_nodes["Step"] <= 15].style.format(
            {"Stock Price": "{:,.2f}", "Intrinsic Payoff": "{:,.2f}",
             "Continuation Value": "{:,.2f}", "Option Value": "{:,.2f}"}),
            width="stretch", height=520)
    if exercise == "American":
        st.caption("★ marks a node where immediate exercise is optimal.")

# -----------------------------------------------------------------------------
# TAB 3 — NODE ANALYSIS
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:16px;'>Every node, fully explained</div>",
                unsafe_allow_html=True)
    step_filter = st.slider("Inspect step", 0, n, n)
    show = df_nodes[df_nodes["Step"] == step_filter].copy()
    st.dataframe(show.style.format(
        {"Stock Price": "{:,.4f}", "Intrinsic Payoff": "{:,.4f}",
         "Continuation Value": "{:,.4f}", "Option Value": "{:,.4f}"}),
        width="stretch")
    if step_filter < n:
        selected_up = st.selectbox("Choose node by number of up moves", show["Up Moves"].tolist())
        row = show[show["Up Moves"] == selected_up].iloc[0]
        html(f"""
        <div class="formula-box">
        <b class="gold">Node interpretation — step {int(row['Step'])}, up moves {int(row['Up Moves'])}</b><br><br>
        Stock price = <b>₹{row['Stock Price']:,.4f}</b><br>
        Intrinsic payoff = <b>₹{row['Intrinsic Payoff']:,.4f}</b><br>
        Continuation value = <b>₹{row['Continuation Value']:,.4f}</b><br>
        Final node value = <b class="gold">₹{row['Option Value']:,.4f}</b><br>
        Early exercise? <b>{'Yes' if row['Early Exercise'] else 'No'}</b>
        </div>
        """)
    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:15px;margin-top:10px;'>Full calculation table</div>",
                unsafe_allow_html=True)
    st.dataframe(df_nodes.style.format(
        {"Stock Price": "{:,.4f}", "Intrinsic Payoff": "{:,.4f}",
         "Continuation Value": "{:,.4f}", "Option Value": "{:,.4f}"}),
        width="stretch", height=520)

# -----------------------------------------------------------------------------
# TAB 4 — ANALYTICS
# -----------------------------------------------------------------------------
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(payoff_figure(S0, K, res["value"], option_type), width="stretch")
    with c2:
        max_conv = min(80, max(30, n))
        st.plotly_chart(convergence_figure(S0, K, r, q, T, option_type, exercise, mode, sigma,
                                            u_input, d_input, bs_val=bs_val, max_n=max_conv),
                        width="stretch")
    if exercise == "European":
        html(f"""<div class="mp-card" style="border-color:rgba(173,216,230,.4);">
          <div style="color:{LB};-webkit-text-fill-color:{LB};font-weight:700;font-size:14px;">
            Convergence to Black–Scholes</div>
          <div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:13.5px;line-height:1.6;margin-top:4px;">
            The dashed line is the closed-form Black–Scholes price
            <b class="lb">₹{bs_val:,.4f}</b>. At the current <b>n = {n}</b> steps the binomial value is
            <b class="gold">₹{res['value']:,.4f}</b> — a gap of <b>{bs_gap:+.4f}</b>. Increase the step
            count in the sidebar and watch the binomial estimate settle onto the benchmark.</div>
        </div>""")

    if exercise == "American":
        early_df = df_nodes[df_nodes["Early Exercise"]].copy()
        st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:15px;'>Early-exercise map</div>",
                    unsafe_allow_html=True)
        if len(early_df):
            st.dataframe(early_df[["Step", "Up Moves", "Stock Price", "Intrinsic Payoff",
                                   "Continuation Value", "Option Value"]].style.format(
                {"Stock Price": "{:,.2f}", "Intrinsic Payoff": "{:,.2f}",
                 "Continuation Value": "{:,.2f}", "Option Value": "{:,.2f}"}),
                width="stretch")
        else:
            st.success("No early-exercise node is optimal for this parameter set.")

    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:15px;margin-top:6px;'>Sensitivity: option value vs spot price</div>",
                unsafe_allow_html=True)
    spots = np.linspace(max(1, S0 * .55), S0 * 1.45, 31)
    vals = [build_binomial(s, K, r, q, T, n, option_type, exercise, u, d)["value"] for s in spots]
    fig = go.Figure(go.Scatter(x=spots, y=vals, mode="lines", line=dict(color=GOLD, width=3),
                               name="Option value"))
    fig.add_vline(x=S0, line_dash="dot", line_color=LB, annotation_text="Current spot",
                  annotation_font_color=LB)
    fig = plotly_theme(fig, height=420, legend=False)
    fig.update_xaxes(title="Spot price")
    fig.update_yaxes(title="Option value")
    st.plotly_chart(fig, width="stretch")

# -----------------------------------------------------------------------------
# TAB 5 — WORKED EXAMPLE
# -----------------------------------------------------------------------------
with tabs[4]:
    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:16px;'>Follow the current example from inputs to price</div>",
                unsafe_allow_html=True)
    html(f"""
    <div class="teach-card"><b class="gold">Step 1 — Divide time:</b> Δt = T/n = {T:.4f}/{n} = <b>{res['dt']:.6f}</b>.</div>
    <div class="teach-card"><b class="gold">Step 2 — Tree factors:</b> u = <b>{u:.6f}</b>, d = <b>{d:.6f}</b>.
        {'These are computed from volatility using CRR.' if mode.startswith('3/') else 'These are directly supplied by the user.'}</div>
    <div class="teach-card"><b class="gold">Step 3 — Risk-neutral probability:</b>
        p = [e<sup>(r−q)Δt</sup>−d]/(u−d) = <b>{p:.6f}</b>; 1−p = <b>{1-p:.6f}</b>.</div>
    <div class="teach-card"><b class="gold">Step 4 — Terminal payoff:</b>
        compute {'max(S−K,0)' if option_type == 'Call' else 'max(K−S,0)'} at each expiry node.</div>
    <div class="teach-card"><b class="gold">Step 5 — Backward induction:</b>
        discount the risk-neutral expected next-node values at e<sup>−rΔt</sup> = <b>{res['disc']:.6f}</b>{'; compare with intrinsic value at every node for the American option.' if exercise == 'American' else '.'}</div>
    <div class="teach-card" style="border-left-color:{LB};"><b class="lb">Result:</b>
        {exercise} {option_type} value at the root = <span class="gold"><b>₹{res['value']:,.4f}</b></span>
        &nbsp;·&nbsp; Black–Scholes European benchmark = <span class="lb"><b>₹{bs_val:,.4f}</b></span>
        &nbsp;·&nbsp; gap <b>{bs_gap:+.4f}</b>.</div>
    """)
    if n <= 3:
        st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:15px;'>Classroom-ready small tree</div>",
                    unsafe_allow_html=True)
        display = df_nodes.copy()
        display["Node"] = display.apply(lambda x: f"t={int(x['Step'])}, U={int(x['Up Moves'])}", axis=1)
        st.dataframe(display[["Node", "Stock Price", "Intrinsic Payoff", "Continuation Value",
                              "Option Value", "Early Exercise"]].style.format(
            {"Stock Price": "{:,.2f}", "Intrinsic Payoff": "{:,.2f}",
             "Continuation Value": "{:,.2f}", "Option Value": "{:,.2f}"}),
            width="stretch")

# -----------------------------------------------------------------------------
# TAB 6 — EXCEL EXPORT
# -----------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:16px;'>Download the complete teaching workbook</div>",
                unsafe_allow_html=True)
    html(f"""<div class="mp-card"><div style="color:{TXT};-webkit-text-fill-color:{TXT};font-size:14px;line-height:1.6;">
         The workbook contains <b class="gold">Summary &amp; Inputs</b> (now including the Black–Scholes
         benchmark and convergence gap), <b class="gold">Node Details</b>, <b class="gold">Stock / Option /
         Intrinsic / Continuation Trees</b>, and a <b class="gold">Formula Sheet</b>. American-option
         early-exercise rows are highlighted.</div></div>""")
    inputs = {"Option Type": option_type, "Exercise": exercise, "S0": S0, "K": K, "r": r, "q": q,
              "T": T, "sigma": sigma, "u": u, "d": d}
    excel_bytes = make_excel(inputs, res, n, mode, bs_val, bs_sigma)
    st.download_button("⬇ Download formatted Excel workbook", data=excel_bytes,
                       file_name=f"MPA_Binomial_{exercise}_{option_type}_{n}Step.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")
    st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:15px;margin-top:8px;'>Export preview</div>",
                unsafe_allow_html=True)
    st.dataframe(df_nodes.head(min(30, len(df_nodes))).style.format(
        {"Stock Price": "{:,.3f}", "Intrinsic Payoff": "{:,.3f}",
         "Continuation Value": "{:,.3f}", "Option Value": "{:,.3f}"}),
        width="stretch")

# =============================================================================
# FOOTER
# =============================================================================
html(f"""
<div style="margin-top:22px;background:linear-gradient(90deg,{BLUE},{MID});border-radius:16px;
     padding:20px 26px;border:1px solid rgba(255,215,0,.3);user-select:none;">
  <div style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center;">
    <div>
      <div style="color:{GOLD};-webkit-text-fill-color:{GOLD};font-size:15px;font-weight:800;">
        The Mountain Path — World of Finance</div>
      <div style="color:{LB};-webkit-text-fill-color:{LB};font-size:12.5px;margin-top:2px;">
        Bridging Theory with Practice · Excellence in Financial Education</div>
      <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:11.5px;margin-top:6px;">
        Prof. V. Ravichandran · Visiting Professor &amp; Professor of Practice at Leading Business Schools ·
        28+ Years Corporate Finance &amp; Banking</div>
    </div>
    <div style="text-align:right;display:flex;flex-direction:column;gap:6px;">
      <a href="{LINK_ACADEMY}" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};
         font-weight:700;font-size:13px;text-decoration:none;">🌐 themountainpathacademy.com ↗</a>
      <a href="{LINK_LI}" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};
         font-weight:700;font-size:13px;text-decoration:none;">in · LinkedIn ↗</a>
      <a href="{LINK_GH}" target="_blank" style="color:{GOLD};-webkit-text-fill-color:{GOLD};
         font-weight:700;font-size:13px;text-decoration:none;">⌥ GitHub ↗</a>
    </div>
  </div>
  <div style="color:{MUTED};-webkit-text-fill-color:{MUTED};font-size:11px;margin-top:12px;
       border-top:1px solid rgba(255,255,255,.1);padding-top:8px;">
    Educational content only — not investment advice. Figures are illustrative and intended for
    classroom and self-study use.</div>
</div>
""")
