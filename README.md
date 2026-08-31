# The Mountain Path Academy — Binomial Option Pricing Model

An educational Streamlit application for teaching and demonstrating the Binomial Option Pricing Model.
Styled in the Mountain Path Academy "World of Finance" navy + gold design system (matched to the
India–US Bond Yield Spread app).

## Features
- Manual up/down factor model without yield
- Manual up/down factor model with yield
- CRR volatility-based model with yield
- European and American calls/puts
- 2-period, 3-period and multi-period trees
- Interactive stock and option trees
- Node-level intrinsic vs continuation analysis
- Early exercise detection for American options
- Payoff, convergence and spot-sensitivity charts
- **Black–Scholes closed-form benchmark** with a live binomial-vs-BS comparison card
  and a convergence reference line (in volatility mode; manual modes use the CRR-implied σ)
- Downloadable formatted Excel workbook (now including the Black–Scholes benchmark and gap)
- Formula reference images derived from the supplied MPA teaching deck

## Theme
The app ships with `.streamlit/config.toml` (dark base, gold primary `#FFD700`, navy background).
Keep that file when deploying so the widgets render in the intended palette.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
Push this folder to GitHub and choose `app.py` as the main file when deploying in Streamlit Community Cloud.
