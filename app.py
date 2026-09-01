

"""
================================================================================
INDIA GDP PROJECTION FY2026-27 - STREAMLIT INTERACTIVE APP
================================================================================

Interactive financial modeling dashboard for India's FY2026-27 GDP forecast
Built with Streamlit for easy deployment and customization

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance

Features:
- Interactive scenario builder
- Real-time sensitivity analysis
- Monte Carlo simulations
- Professional visualizations
- Downloadable results
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import io
from datetime import datetime

# Data vintage: latest releases available on 1 September 2026.
# All growth rates in this app refer to India's fiscal year (April-March),
# avoiding comparisons between fiscal-year and calendar-year forecasts.
FORECAST_PERIOD = "FY2026-27"
DATA_AS_OF = "1 September 2026"
BASE_GDP_GROWTH = 6.60
BASE_OIL_PRICE = 89.27  # IMF July 2026 WEO assumption, 2026 average, USD/bbl
BASE_CPI_INFLATION = 5.00  # RBI August 2026 projection for FY2026-27
BASE_CAPEX_GROWTH = 11.50  # Union Budget central-government capex growth

SOURCES = {
    "MoSPI": "https://mospi.gov.in/uploads/latestReleases/latest_release_1772189865181_f040336d-bc57-4aed-b80f-586d9ccb279e_Press_Note_on_New_Series_of_GDP_Estimates_with_Base_Year_2022-23_27022026.pdf",
    "RBI": "https://www.rbi.org.in/",
    "IMF": "https://www.imf.org/en/publications/weo/issues/2026/07/08/world-economic-outlook-update-july-2026",
    "ADB": "https://www.adb.org/outlook/editions/july-2026",
    "World Bank": "https://www.worldbank.org/en/news/press-release/2026/04/09/india-remains-among-the-fastest-growing-economies",
    "OECD": "https://www.oecd.org/en/publications/oecd-economic-outlook-volume-2026-issue-1_8be0dba6-en.html",
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="India GDP FY2026-27 Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand palette — matched to the attached Binomial Option Pricing app
GOLD = "#FFD700"
BLUE = "#003366"
MID = "#004d80"
CARD = "#112240"
TXT = "#e6f1ff"
MUTED = "#8892b0"
LB = "#ADD8E6"
LINK_ACADEMY = "https://themountainpathacademy.com"
LINK_LI = "https://www.linkedin.com/in/trichyravis"
LINK_GH = "https://github.com/trichyravis"

# Global design system — verbatim visual language from the reference app
st.markdown(f"""
<style>
  .stApp {{ background:linear-gradient(135deg,#1a2332,#243447,#2a3f5f) fixed; }}
  #MainMenu, header[data-testid="stHeader"], footer {{ visibility:hidden; }}
  .block-container {{ padding-top:1.2rem;padding-bottom:2rem;max-width:1280px; }}
  h1,h2,h3,h4,p,li {{ color:{TXT};letter-spacing:.2px; }}
  [data-testid="stCaptionContainer"] p {{ color:{MUTED}!important; }}
  [data-testid="stSidebar"] {{ background:linear-gradient(180deg,#0d1b30,#112240 70%,#0d1b30);border-right:1px solid rgba(255,215,0,.22); }}
  [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 {{ color:{GOLD}!important;-webkit-text-fill-color:{GOLD}!important;font-size:1rem;text-transform:uppercase;letter-spacing:1.2px; }}
  [data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{ color:{TXT}!important;-webkit-text-fill-color:{TXT}!important;font-weight:600; }}
  [data-testid="stSidebar"] hr {{ border-color:rgba(255,215,0,.18); }}
  [data-testid="stSidebar"] [data-baseweb="select"] > div,[data-testid="stSidebar"] [data-baseweb="input"] > div {{ background:#f7f9fc!important;border-color:{GOLD}!important;border-radius:8px!important; }}
  [data-testid="stSidebar"] [data-baseweb="select"] input,[data-testid="stSidebar"] [data-baseweb="input"] input {{ color:#10213d!important;-webkit-text-fill-color:#10213d!important;font-weight:700!important; }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label {{ background:#10213d;border:1px solid rgba(173,216,230,.35);border-radius:8px;padding:7px 9px;margin:2px 0; }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ border-color:{GOLD};background:{MID}; }}
  .stTabs [data-baseweb="tab-list"] {{ gap:6px;background:rgba(17,34,64,.55);padding:6px;border-radius:12px;border:1px solid rgba(255,215,0,.18);flex-wrap:wrap; }}
  .stTabs [data-baseweb="tab"] {{ background:#10213d!important;border:1px solid rgba(173,216,230,.52)!important;border-radius:8px;padding:8px 16px;font-weight:700;color:#fff!important; }}
  .stTabs [data-baseweb="tab"] * {{ color:#fff!important;-webkit-text-fill-color:#fff!important; }}
  .stTabs [aria-selected="true"] {{ background:{GOLD}!important;border-color:{GOLD}!important; }}
  .stTabs [aria-selected="true"] * {{ color:{BLUE}!important;-webkit-text-fill-color:{BLUE}!important; }}
  [data-testid="stWidgetLabel"] *,.stRadio * {{ color:{TXT}!important;-webkit-text-fill-color:{TXT}!important; }}
  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{GOLD}; }}
  .stButton button,.stDownloadButton button {{ background:{GOLD}!important;border:1px solid {GOLD}!important;border-radius:10px!important;font-weight:800!important; }}
  .stButton button *,.stDownloadButton button * {{ color:{BLUE}!important;-webkit-text-fill-color:{BLUE}!important; }}
  div[data-testid="stDataFrame"] {{ border:1px solid rgba(255,215,0,.20);border-radius:12px;overflow:hidden; }}
  [data-testid="stMetric"] {{ background:{CARD};border:1px solid rgba(255,215,0,.16);border-radius:14px;padding:15px 16px;box-shadow:0 4px 18px rgba(0,0,0,.28); }}
  [data-testid="stMetricLabel"] * {{ color:{MUTED}!important;-webkit-text-fill-color:{MUTED}!important;text-transform:uppercase;letter-spacing:1px;font-size:11px!important; }}
  [data-testid="stMetricValue"] * {{ color:{GOLD}!important;-webkit-text-fill-color:{GOLD}!important;font-weight:800; }}
  .metric-box,.scenario-box {{ background:{CARD};border:1px solid rgba(255,215,0,.16);border-radius:14px;padding:18px 20px;margin:10px 0;box-shadow:0 4px 18px rgba(0,0,0,.28);color:{TXT};min-height:118px; }}
  .metric-box strong,.scenario-box strong {{ color:{MUTED};text-transform:uppercase;letter-spacing:1px;font-size:11px; }}
  .metric-box span,.scenario-box span {{ color:{GOLD}!important;-webkit-text-fill-color:{GOLD}!important;font-weight:800; }}
  .scenario-box small {{ color:{LB}; }}
  .upside {{ border-left:4px solid #28a745; }} .base {{ border-left:4px solid {GOLD}; }} .downside {{ border-left:4px solid #dc3545; }}
  [data-testid="stAlert"] {{ background:{CARD};border:1px solid rgba(255,215,0,.25);color:{TXT};border-radius:12px; }}
  hr {{ border-color:rgba(255,215,0,.18)!important; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# COLOR SCHEME
# ============================================================================

DARKBLUE = '#003366'
LIGHTBLUE = '#ADD8E6'
GOLDCOLOR = '#FFD700'

# Matplotlib charts use the same dark navy/gold presentation.
plt.rcParams.update({
    "figure.facecolor": "#112240", "axes.facecolor": "#112240",
    "axes.edgecolor": "#8892b0", "axes.labelcolor": "#e6f1ff",
    "axes.titlecolor": "#FFD700", "xtick.color": "#e6f1ff",
    "ytick.color": "#e6f1ff", "text.color": "#e6f1ff",
    "grid.color": "#8892b0", "legend.facecolor": "#112240",
    "legend.edgecolor": "#FFD700",
})

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False
if 'mc_results' not in st.session_state:
    st.session_state.mc_results = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = {}

# ============================================================================
# HELPER CLASSES
# ============================================================================

class GDPProjectionModel:
    """Linear regression model for GDP projection"""
    
    def __init__(self, base_year=2024):
        self.base_year = base_year
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.features = None
        self.coefficients = None
        self.r2_score = None
        
    def prepare_data(self, years, gdp_growth, predictors):
        X_scaled = self.scaler.fit_transform(predictors)
        y = np.array(gdp_growth)
        self.features = predictors.columns
        return X_scaled, y
    
    def fit(self, X_scaled, y):
        self.model.fit(X_scaled, y)
        self.coefficients = dict(zip(self.features, self.model.coef_))
        self.r2_score = self.model.score(X_scaled, y)
        return self
    
    def project(self, predictors_2026):
        X_2026 = self.scaler.transform(predictors_2026.reshape(1, -1))
        return self.model.predict(X_2026)[0]


class SectoralGDPModel:
    """Multi-sector GDP decomposition"""
    
    def __init__(self):
        self.sectors = {}
        self.weights = {}
        
    def add_sector(self, name, weight, base_growth):
        self.sectors[name] = {
            'weight': weight,
            'base_growth': base_growth,
            'elasticities': {}
        }
        self.weights[name] = weight
    
    def set_elasticities(self, sector, elasticity_dict):
        self.sectors[sector]['elasticities'] = elasticity_dict
    
    def project_growth(self, sector, variables):
        base = self.sectors[sector]['base_growth']
        elasticities = self.sectors[sector]['elasticities']
        growth = base
        for var, shock in variables.items():
            if var in elasticities:
                growth += elasticities[var] * shock
        return growth
    
    def aggregate_gdp(self, sector_growths):
        gdp_growth = 0
        for sector, growth in sector_growths.items():
            weight = self.weights[sector]
            gdp_growth += weight * growth
        return gdp_growth


class MonteCarloGDPSimulation:
    """Monte Carlo simulation for probability distribution"""
    
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations
        self.simulations = None
        
    def run_simulation(self, base_growth, scenarios_dict):
        self.simulations = []
        scenario_names = list(scenarios_dict.keys())
        probabilities = [scenarios_dict[s]['probability'] for s in scenario_names]
        
        for _ in range(self.num_simulations):
            scenario = np.random.choice(scenario_names, p=probabilities)
            base_sim = scenarios_dict[scenario]['growth']
            noise = np.random.normal(0, 0.15)
            self.simulations.append(base_sim + noise)
        
        self.simulations = np.array(self.simulations)
    
    def get_statistics(self):
        return {
            'mean': np.mean(self.simulations),
            'std': np.std(self.simulations),
            'median': np.median(self.simulations),
            'min': np.min(self.simulations),
            'max': np.max(self.simulations),
            'percentile_5': np.percentile(self.simulations, 5),
            'percentile_25': np.percentile(self.simulations, 25),
            'percentile_75': np.percentile(self.simulations, 75),
            'percentile_95': np.percentile(self.simulations, 95)
        }


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # ========================================================================
    # SIDEBAR PROFILE SECTION (ONLY NAME & COMPANY)
    # ========================================================================
    with st.sidebar:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{BLUE},{MID});border:1px solid rgba(255,215,0,.28);
             border-radius:14px;padding:16px;margin:4px 0 18px;text-align:center;">
            <div style="font-size:28px">📊</div>
            <div style="color:{GOLD};font-weight:800;letter-spacing:1px;font-size:12px;">THE MOUNTAIN PATH ACADEMY</div>
            <div style="color:#fff;font-size:15px;font-weight:700;margin-top:5px;">Prof. V. Ravichandran</div>
            <div style="color:{LB};font-size:11px;">World of Finance</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # MAIN HEADER
    # ========================================================================
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{BLUE},{MID});border-radius:16px;
         padding:22px 26px;border:1px solid rgba(255,215,0,.3);user-select:none;
         box-shadow:0 6px 24px rgba(0,0,0,.35);margin-bottom:10px;">
      <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
        <div style="font-size:34px">📊</div>
        <div style="flex:1;min-width:260px;">
          <div style="color:{GOLD};font-size:13px;font-weight:700;letter-spacing:2px;">THE MOUNTAIN PATH ACADEMY · WORLD OF FINANCE</div>
          <div style="color:#fff;font-size:26px;font-weight:800;line-height:1.15;margin-top:2px;">India GDP Projection {FORECAST_PERIOD}</div>
          <div style="color:{LB};font-size:14px;margin-top:3px;">Build scenarios · inspect quarterly growth · test sensitivities · compare institutions · export results</div>
        </div>
        <div style="text-align:right;min-width:150px;">
          <div style="color:{MUTED};font-size:12px;">Educational Series by</div>
          <div style="color:#fff;font-size:15px;font-weight:700;">Prof. V. Ravichandran</div>
          <a href="{LINK_ACADEMY}" target="_blank" style="color:{GOLD};font-size:12px;text-decoration:none;">themountainpathacademy.com ↗</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.markdown("## 🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "🗓️ Quarterly Projection", "⚙️ Scenario Builder", "📈 Sensitivity Analysis", 
         "📊 Institutional Comparison", "📥 Download Results"]
    )
    
    # Initialize model data
    # Setup sectoral model
    sectoral_model = SectoralGDPModel()
    # Working assumptions calibrated to aggregate to a 6.60% base case.
    sectoral_model.add_sector('Agriculture', 0.18, 3.50)
    sectoral_model.add_sector('Manufacturing', 0.27, 6.43)
    sectoral_model.add_sector('Services', 0.55, 7.70)
    
    sectoral_model.set_elasticities('Agriculture', {
        'Oil_Price_Change': -0.08,
        'Monsoon_Deviation': 0.20,
        'Food_Inflation': -0.10
    })
    
    sectoral_model.set_elasticities('Manufacturing', {
        'Oil_Price_Change': -0.10,
        'Capex_Growth_Change': 0.15,
        'Tariff_Impact': -0.50,
        'Global_Growth': 0.30
    })
    
    sectoral_model.set_elasticities('Services', {
        'Consumption_Growth': 0.20,
        'Interest_Rate_Change': -0.08,
        'Export_Growth': 0.25,
        'Global_Growth': 0.15
    })
    
    # ========================================================================
    # PAGE ROUTING
    # ========================================================================
    
    if page == "🏠 Dashboard":
        show_dashboard(sectoral_model)
    elif page == "🗓️ Quarterly Projection":
        show_quarterly_projection()
    elif page == "⚙️ Scenario Builder":
        show_scenario_builder(sectoral_model)
    elif page == "📈 Sensitivity Analysis":
        show_sensitivity_analysis()
    elif page == "📊 Institutional Comparison":
        show_institutional_comparison()
    elif page == "📥 Download Results":
        show_download_page()

    st.markdown(f"""
    <div style="margin-top:22px;background:linear-gradient(90deg,{BLUE},{MID});border-radius:16px;
         padding:20px 26px;border:1px solid rgba(255,215,0,.3);user-select:none;">
      <div style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center;">
        <div>
          <div style="color:{GOLD};font-size:15px;font-weight:800;">The Mountain Path — World of Finance</div>
          <div style="color:{LB};font-size:12.5px;margin-top:2px;">Bridging Theory with Practice · Excellence in Financial Education</div>
          <div style="color:{MUTED};font-size:11.5px;margin-top:6px;">Prof. V. Ravichandran · Visiting Professor &amp; Professor of Practice at Leading Business Schools · 28+ Years Corporate Finance &amp; Banking</div>
        </div>
        <div style="text-align:right;display:flex;flex-direction:column;gap:6px;">
          <a href="{LINK_ACADEMY}" target="_blank" style="color:{GOLD};font-weight:700;font-size:13px;text-decoration:none;">🌐 themountainpathacademy.com ↗</a>
          <a href="{LINK_LI}" target="_blank" style="color:{GOLD};font-weight:700;font-size:13px;text-decoration:none;">in · LinkedIn ↗</a>
          <a href="{LINK_GH}" target="_blank" style="color:{GOLD};font-weight:700;font-size:13px;text-decoration:none;">⌥ GitHub ↗</a>
        </div>
      </div>
      <div style="color:{MUTED};font-size:11px;margin-top:12px;border-top:1px solid rgba(255,255,255,.1);padding-top:8px;">Educational content only — not investment advice. Forecasts are model estimates and remain subject to revision.</div>
    </div>
    """, unsafe_allow_html=True)


def show_dashboard(sectoral_model):
    """Dashboard with base case scenario"""
    
    st.subheader("📊 Base Case Forecast")
    st.caption(
        f"Forecast period: {FORECAST_PERIOD} (April 2026-March 2027) · "
        f"Base case: {BASE_GDP_GROWTH:.1f}% · Data vintage: {DATA_AS_OF}"
    )
    
    # Base case variables
    base_variables = {
        'Oil_Price_Change': 0.0,
        'Monsoon_Deviation': 0.0,
        'Food_Inflation': 0.0,
        'Capex_Growth_Change': 0.0,
        'Tariff_Impact': 0.0,
        'Global_Growth': 0.0,
        'Consumption_Growth': 0.0,
        'Interest_Rate_Change': 0.0,
        'Export_Growth': 0.0
    }
    
    # Calculate base case
    sector_growths_base = {}
    for sector in sectoral_model.sectors.keys():
        sector_growths_base[sector] = sectoral_model.project_growth(sector, base_variables)
    
    gdp_base = sectoral_model.aggregate_gdp(sector_growths_base)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Base Case GDP Growth</strong><br>
            <span style='font-size: 2em; color: #003366;'>{gdp_base:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Agriculture</strong><br>
            <span style='font-size: 2em; color: #003366;'>{sector_growths_base['Agriculture']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Manufacturing</strong><br>
            <span style='font-size: 2em; color: #003366;'>{sector_growths_base['Manufacturing']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-box'>
            <strong>Services</strong><br>
            <span style='font-size: 2em; color: #003366;'>{sector_growths_base['Services']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # Quarterly forecast is also shown on the main dashboard so users do not
    # need to discover it through the sidebar navigation.
    st.markdown("---")
    st.subheader(f"🗓️ Quarterly Forecast — {FORECAST_PERIOD}")
    dashboard_quarters = pd.DataFrame({
        "Quarter": ["Q1 (Apr–Jun)", "Q2 (Jul–Sep)", "Q3 (Oct–Dec)", "Q4 (Jan–Mar)"],
        "Our Model (%)": [6.9, 6.3, 6.4, 6.8],
        "RBI Projection (%)": [7.0, 6.4, 6.5, 6.8],
    })

    q1, q2, q3, q4 = st.columns(4)
    for column, row in zip([q1, q2, q3, q4], dashboard_quarters.to_dict("records")):
        with column:
            st.metric(row["Quarter"], f"{row['Our Model (%)']:.1f}%",
                      delta=f"RBI {row['RBI Projection (%)']:.1f}%")

    st.dataframe(
        dashboard_quarters.style.format({
            "Our Model (%)": "{:.1f}",
            "RBI Projection (%)": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    qfig, qax = plt.subplots(figsize=(11, 4))
    qx = np.arange(len(dashboard_quarters))
    qax.plot(qx, dashboard_quarters["Our Model (%)"], marker="o", linewidth=2.5,
             color=DARKBLUE, label="Our model")
    qax.plot(qx, dashboard_quarters["RBI Projection (%)"], marker="s", linewidth=2,
             linestyle="--", color="#666666", label="RBI August 2026")
    qax.set_xticks(qx)
    qax.set_xticklabels(dashboard_quarters["Quarter"])
    qax.set_ylabel("Real GDP growth, YoY (%)", fontweight="bold")
    qax.grid(alpha=0.3)
    qax.legend()
    st.pyplot(qfig)
    st.caption(
        "Quarterly rates are year-over-year projections. Use “Quarterly Projection” "
        "in the sidebar to adjust each quarter."
    )
    
    st.markdown("---")
    
    # Scenario comparison
    st.subheader("📈 Scenario Comparison")
    
    # Upside scenario
    upside_variables = {
        'Oil_Price_Change': -15.0,
        'Monsoon_Deviation': 0.10,
        'Food_Inflation': -0.5,
        'Capex_Growth_Change': 5.0,
        'Tariff_Impact': 0.0,
        'Global_Growth': 0.5,
        'Consumption_Growth': 1.0,
        'Interest_Rate_Change': -0.50,
        'Export_Growth': 2.0
    }
    
    sector_growths_upside = {}
    for sector in sectoral_model.sectors.keys():
        sector_growths_upside[sector] = sectoral_model.project_growth(sector, upside_variables)
    
    gdp_upside = sectoral_model.aggregate_gdp(sector_growths_upside)
    
    # Downside scenario
    downside_variables = {
        'Oil_Price_Change': 20.0,
        'Monsoon_Deviation': -0.15,
        'Food_Inflation': 1.5,
        'Capex_Growth_Change': -8.0,
        'Tariff_Impact': -0.25,
        'Global_Growth': -1.0,
        'Consumption_Growth': -0.5,
        'Interest_Rate_Change': 0.25,
        'Export_Growth': -3.0
    }
    
    sector_growths_downside = {}
    for sector in sectoral_model.sectors.keys():
        sector_growths_downside[sector] = sectoral_model.project_growth(sector, downside_variables)
    
    gdp_downside = sectoral_model.aggregate_gdp(sector_growths_downside)
    
    # Display scenarios
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='scenario-box downside'>
            <strong>📉 Downside Case</strong><br>
            Probability: 20%<br>
            <span style='font-size: 1.8em;'>{gdp_downside:.2f}%</span><br>
            <small>Oil spike, tariffs, monsoon deficit</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='scenario-box base'>
            <strong>📊 Base Case</strong><br>
            Probability: 60%<br>
            <span style='font-size: 1.8em;'>{gdp_base:.2f}%</span><br>
            <small>Most likely outcome</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='scenario-box upside'>
            <strong>📈 Upside Case</strong><br>
            Probability: 20%<br>
            <span style='font-size: 1.8em;'>{gdp_upside:.2f}%</span><br>
            <small>Lower oil, strong capex, excess monsoon</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability-weighted forecast
    prob_weighted = (0.20 * gdp_downside) + (0.60 * gdp_base) + (0.20 * gdp_upside)
    
    st.info(f"**Probability-Weighted Forecast: {prob_weighted:.2f}%**")
    st.caption(
        f"Reference assumptions: oil ${BASE_OIL_PRICE:.2f}/bbl, RBI CPI inflation "
        f"{BASE_CPI_INFLATION:.1f}%, and central-government capex growth "
        f"{BASE_CAPEX_GROWTH:.1f}%. Scenario elasticities are model assumptions, not official estimates."
    )
    
    # Monte Carlo simulation
    st.subheader("🎲 Monte Carlo Simulation (10,000 runs)")
    
    scenarios = {
        'Upside': {'probability': 0.20, 'growth': gdp_upside},
        'Base': {'probability': 0.60, 'growth': gdp_base},
        'Downside': {'probability': 0.20, 'growth': gdp_downside}
    }
    
    mc_sim = MonteCarloGDPSimulation(num_simulations=10000)
    mc_sim.run_simulation(gdp_base, scenarios)
    stats_dict = mc_sim.get_statistics()
    
    # Display MC statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{stats_dict['mean']:.2f}%")
    
    with col2:
        st.metric("Median", f"{stats_dict['median']:.2f}%")
    
    with col3:
        st.metric("Std Dev", f"{stats_dict['std']:.2f}%")
    
    with col4:
        st.metric("Range", f"{stats_dict['min']:.2f}% - {stats_dict['max']:.2f}%")
    
    # Plot MC distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(mc_sim.simulations, bins=60, density=True, alpha=0.7,
                 color='#003366', edgecolor='black', linewidth=0.5)
    
    mu, sigma = stats_dict['mean'], stats_dict['std']
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, label='Normal Fit')
    axes[0].axvline(mu, color='#003366', linestyle='--', linewidth=2, label=f'Mean: {mu:.2f}%')
    axes[0].set_xlabel('GDP Growth Rate (%)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    axes[0].set_title('GDP Growth Distribution', fontsize=12, fontweight='bold', color=GOLD)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # CDF
    sorted_sims = np.sort(mc_sim.simulations)
    cumulative_prob = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    axes[1].plot(sorted_sims, cumulative_prob * 100, linewidth=2.5, color='#003366')
    axes[1].fill_between(sorted_sims, 0, cumulative_prob * 100, alpha=0.2, color='#003366')
    axes[1].set_xlabel('GDP Growth Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Cumulative Probability (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold', color=GOLD)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Percentile summary
    st.subheader("📊 Percentile Distribution")
    
    percentile_data = {
        'Percentile': ['5th', '25th', '50th (Median)', '75th', '95th'],
        'GDP Growth': [
            f"{stats_dict['percentile_5']:.2f}%",
            f"{stats_dict['percentile_25']:.2f}%",
            f"{stats_dict['median']:.2f}%",
            f"{stats_dict['percentile_75']:.2f}%",
            f"{stats_dict['percentile_95']:.2f}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)


def show_quarterly_projection():
    """Quarterly GDP projection path for FY2026-27."""

    st.subheader(f"🗓️ Quarterly GDP Projection — {FORECAST_PERIOD}")
    st.caption(
        "Real GDP growth, year over year. RBI benchmark is from its August 2026 "
        "monetary-policy projection; the model path averages 6.6% for the fiscal year."
    )

    quarters = ["Q1 (Apr–Jun)", "Q2 (Jul–Sep)", "Q3 (Oct–Dec)", "Q4 (Jan–Mar)"]
    rbi_projection = np.array([7.0, 6.4, 6.5, 6.8])
    model_base = np.array([6.9, 6.3, 6.4, 6.8])

    st.markdown("### Adjust the quarterly path")
    col1, col2, col3, col4 = st.columns(4)
    adjustments = []
    for column, quarter in zip([col1, col2, col3, col4], quarters):
        with column:
            adjustments.append(
                st.slider(
                    quarter,
                    min_value=-1.5,
                    max_value=1.5,
                    value=0.0,
                    step=0.1,
                    help="Adjustment in percentage points relative to the model base."
                )
            )

    custom_projection = model_base + np.array(adjustments)
    annual_model = float(np.mean(custom_projection))
    annual_rbi = float(np.mean(rbi_projection))

    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric("Adjusted FY Growth", f"{annual_model:.2f}%",
                  delta=f"{annual_model - BASE_GDP_GROWTH:+.2f} pp vs base")
    with metric2:
        st.metric("RBI FY Projection", f"{annual_rbi:.2f}%")
    with metric3:
        st.metric("Model–RBI Difference", f"{annual_model - annual_rbi:+.2f} pp")

    quarterly_df = pd.DataFrame({
        "Quarter": quarters,
        "Model Base (%)": model_base,
        "Adjusted Model (%)": custom_projection,
        "RBI Projection (%)": rbi_projection,
        "Adjustment (pp)": adjustments,
    })
    st.dataframe(
        quarterly_df.style.format({
            "Model Base (%)": "{:.1f}",
            "Adjusted Model (%)": "{:.1f}",
            "RBI Projection (%)": "{:.1f}",
            "Adjustment (pp)": "{:+.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(quarters))
    ax.plot(x, custom_projection, marker="o", linewidth=2.5, color=DARKBLUE,
            label="Adjusted model")
    ax.plot(x, rbi_projection, marker="s", linewidth=2, linestyle="--",
            color="#666666", label="RBI August 2026")
    ax.plot(x, model_base, linewidth=1.5, linestyle=":", color=LIGHTBLUE,
            label="Model base")
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)
    ax.set_ylabel("Real GDP growth, YoY (%)", fontweight="bold")
    ax.set_title(f"Quarterly Growth Path — {FORECAST_PERIOD}",
                 color=DARKBLUE, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.info(
        "The fiscal-year figure shown here is the simple average of the four quarterly "
        "year-over-year rates. This is an approximation: official annual GDP growth is "
        "calculated from annual GDP levels, so it need not equal that average exactly."
    )
    st.markdown(f"Source: [Reserve Bank of India]({SOURCES['RBI']}) · Data as of {DATA_AS_OF}")


def show_scenario_builder(sectoral_model):
    """Interactive scenario builder"""
    
    st.subheader("⚙️ Build Your Custom Scenario")
    
    st.markdown("Adjust the variables below to create your own scenario:")
    
    # Create 3 columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🛢️ Oil & Energy")
        oil_change = st.slider(
            "Oil Price Change (USD/bbl)",
            min_value=-15.0,
            max_value=25.0,
            value=0.0,
            step=0.5,
            help=f"Base: ${BASE_OIL_PRICE:.2f}/bbl (IMF July 2026). Negative = cheaper oil"
        )
        
        inflation_change = st.slider(
            "Food Inflation Change (%)",
            min_value=-1.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
    
    with col2:
        st.markdown("### 🌾 Agriculture & Climate")
        monsoon_dev = st.slider(
            "Monsoon Deviation (% of normal)",
            min_value=-0.20,
            max_value=0.20,
            value=0.0,
            step=0.05,
            help="Negative = deficit, Positive = excess"
        )
        
        capex_change = st.slider(
            "Capex Growth Change (%)",
            min_value=-25.0,
            max_value=25.0,
            value=0.0,
            step=1.0,
            help=f"Base: {BASE_CAPEX_GROWTH:.1f}% budgeted central-government capex growth"
        )
    
    with col3:
        st.markdown("### 🌍 External Factors")
        tariff_impact = st.slider(
            "Tariff Impact (%)",
            min_value=-0.50,
            max_value=0.0,
            value=0.0,
            step=0.05,
            help="Enter a downside adjustment; zero means no additional tariff shock"
        )
        
        global_growth = st.slider(
            "Global Growth Change (%)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
    
    st.markdown("---")
    
    # Additional variables
    col1, col2, col3 = st.columns(3)
    
    with col1:
        consumption_growth = st.slider(
            "Consumption Growth Change (%)",
            min_value=-1.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
    
    with col2:
        rate_change = st.slider(
            "Interest Rate Change (bps)",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=10.0,
            help="Negative = rate cuts"
        )
    
    with col3:
        export_growth = st.slider(
            "Export Growth Change (%)",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.5
        )
    
    st.markdown("---")
    
    # Calculate scenario
    if st.button("📊 Calculate Scenario", key="scenario_calc"):
        custom_variables = {
            'Oil_Price_Change': oil_change,
            'Monsoon_Deviation': monsoon_dev,
            'Food_Inflation': inflation_change,
            'Capex_Growth_Change': capex_change,
            'Tariff_Impact': tariff_impact / 100,
            'Global_Growth': global_growth,
            'Consumption_Growth': consumption_growth,
            'Interest_Rate_Change': rate_change / 100,
            'Export_Growth': export_growth
        }
        
        # Calculate sector growths
        sector_growths = {}
        for sector in sectoral_model.sectors.keys():
            sector_growths[sector] = sectoral_model.project_growth(sector, custom_variables)
        
        gdp_forecast = sectoral_model.aggregate_gdp(sector_growths)
        
        # Display results
        st.success("✅ Scenario Calculated!")
        
        # Results in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("GDP Growth", f"{gdp_forecast:.2f}%")
        
        with col2:
            st.metric("Agriculture", f"{sector_growths['Agriculture']:.2f}%")
        
        with col3:
            st.metric("Manufacturing", f"{sector_growths['Manufacturing']:.2f}%")
        
        with col4:
            st.metric("Services", f"{sector_growths['Services']:.2f}%")
        
        # Sector breakdown chart
        sector_data = pd.DataFrame({
            'Sector': list(sector_growths.keys()),
            'Growth': list(sector_growths.values())
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(sector_data['Sector'], sector_data['Growth'], color=['#003366', '#ADD8E6', '#FFD700'])
        ax.set_ylabel('Growth Rate (%)', fontweight='bold')
        ax.set_title('Sectoral Growth Breakdown', fontweight='bold', color=GOLD, fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(sector_data['Growth']):
            ax.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')
        
        st.pyplot(fig)
        
        # Save to session state
        st.session_state.forecast_data = {
            'gdp': gdp_forecast,
            'sectors': sector_growths,
            'variables': custom_variables,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def show_sensitivity_analysis():
    """Sensitivity analysis page"""
    
    st.subheader("📈 Sensitivity Analysis")
    
    st.markdown("Understand how different variables impact GDP growth.")
    
    base_growth = BASE_GDP_GROWTH
    
    sensitivities = {
        'Oil Prices (USD/bbl)': {'elasticity': -0.10, 'range': (-15, 15), 'description': 'Every $1/bbl change'},
        'Monsoon Index (% deviation)': {'elasticity': 0.20, 'range': (-10, 10), 'description': '10% deviation'},
        'Capex Growth (%)': {'elasticity': 0.08, 'range': (-20, 20), 'description': 'Capex growth change'},
        'Global Growth (%)': {'elasticity': 0.50, 'range': (-2, 2), 'description': 'Global GDP change'},
        'Trade/Tariff Shock (%)': {'elasticity': 0.50, 'range': (-2, 0), 'description': 'Modelled GDP downside shock'}
    }
    
    st.subheader("Sensitivity Coefficients")
    
    sensitivity_df = pd.DataFrame({
        'Variable': list(sensitivities.keys()),
        'Elasticity': [v['elasticity'] for v in sensitivities.values()],
        'Description': [v['description'] for v in sensitivities.values()]
    })
    
    st.dataframe(sensitivity_df, use_container_width=True)
    
    st.markdown("---")
    
    # Tornado chart
    st.subheader("🌪️ Sensitivity Tornado Chart")
    
    variables = list(sensitivities.keys())
    impacts = []
    
    for var_name in variables:
        data = sensitivities[var_name]
        min_shock, max_shock = data['range']
        min_impact = (data['elasticity'] * min_shock)
        max_impact = (data['elasticity'] * max_shock)
        impacts.append((min_impact, max_impact))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    y_pos = np.arange(len(variables))
    
    for i, (min_val, max_val) in enumerate(impacts):
        if min_val < 0:
            ax.barh(i, abs(min_val), left=-abs(min_val), height=0.6,
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        if max_val > 0:
            ax.barh(i, max_val, left=0, height=0.6,
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.axvline(0, color='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=10)
    ax.set_xlabel('GDP Growth Impact (Percentage Points)', fontsize=11, fontweight='bold')
    ax.set_title(f'GDP Growth Sensitivity - Tornado Chart (Base: {base_growth:.2f}%)',
                fontsize=12, fontweight='bold', color=GOLD)
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)
    
    # Interactive calculator
    st.markdown("---")
    st.subheader("🔧 Interactive Sensitivity Calculator")
    
    selected_var = st.selectbox("Select variable to analyze", list(sensitivities.keys()))
    
    var_data = sensitivities[selected_var]
    min_shock, max_shock = var_data['range']
    
    shock_value = st.slider(
        f"Shock to {selected_var}",
        min_value=int(min_shock),
        max_value=int(max_shock),
        value=0,
        step=1
    )
    
    impact = var_data['elasticity'] * shock_value
    new_growth = base_growth + impact
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Shock Value", f"{shock_value:.2f}")
    
    with col2:
        st.metric("Elasticity", f"{var_data['elasticity']:.3f}")
    
    with col3:
        st.metric("GDP Impact", f"{impact:+.2f} bps", delta=f"{impact:+.2f}")
    
    st.info(f"**New GDP Growth: {new_growth:.2f}%** (from base {base_growth:.2f}%)")


def show_institutional_comparison():
    """Institutional forecasts comparison"""
    
    st.subheader("📊 Institutional Forecasts Comparison")
    st.caption(f"Real GDP growth forecasts for {FORECAST_PERIOD}; latest available as of {DATA_AS_OF}.")
    
    institutions_data = {
        'Institution': ['Our Model (Base)', 'RBI (Aug 2026)', 'IMF (Jul 2026)',
                        'ADB (Jul 2026)', 'World Bank (Apr 2026)', 'OECD (Jun 2026)'],
        'Forecast (%)': [BASE_GDP_GROWTH, 6.70, 6.40, 6.60, 6.60, 6.30],
        'Type': ['Our Model', 'Institution', 'Institution', 'Institution', 'Institution', 'Institution']
    }
    
    forecast_df = pd.DataFrame(institutions_data)
    
    st.dataframe(forecast_df, use_container_width=True)
    
    our_base = forecast_df[forecast_df['Institution'] == 'Our Model (Base)']['Forecast (%)'].values[0]
    institutional_avg = forecast_df[forecast_df['Type'] == 'Institution']['Forecast (%)'].mean()
    difference = our_base - institutional_avg
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Our Base Case", f"{our_base:.2f}%")
    with col2:
        st.metric("Institutional Average", f"{institutional_avg:.2f}%")
    with col3:
        st.metric("Difference", f"{difference:+.2f}%")
    
    st.markdown("---")
    
    # Comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#003366' if x == 'Our Model' else '#666666' for x in forecast_df['Type']]
    ax.barh(forecast_df['Institution'], forecast_df['Forecast (%)'], color=colors, alpha=0.8)
    
    for i, (idx, row) in enumerate(forecast_df.iterrows()):
        ax.text(row['Forecast (%)'] + 0.1, i, f"{row['Forecast (%)']:.1f}%",
                va='center', fontweight='bold')
    
    ax.set_xlabel('GDP Growth Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{FORECAST_PERIOD} Real GDP Growth Forecast Comparison',
                fontsize=12, fontweight='bold', color=GOLD)
    ax.set_xlim([6.0, 7.1])
    ax.grid(axis='x', alpha=0.3)
    
    st.pyplot(fig)

    st.markdown("#### Sources and definitions")
    st.markdown(
        f"- [RBI August 2026 monetary policy]({SOURCES['RBI']}): 6.7%\n"
        f"- [IMF July 2026 WEO Update]({SOURCES['IMF']}): 6.4% on India's fiscal-year basis\n"
        f"- [ADB July 2026 Outlook]({SOURCES['ADB']}): 6.6%\n"
        f"- [World Bank April 2026 India Development Update]({SOURCES['World Bank']}): 6.6%\n"
        f"- [OECD June 2026 Economic Outlook]({SOURCES['OECD']}): 6.3%\n\n"
        "Institutional forecasts are not averaged with calendar-year figures."
    )


def show_download_page():
    """Download results page"""
    
    st.subheader("📥 Download Your Results")
    
    if not st.session_state.forecast_data:
        st.warning("⚠️ No custom scenario calculated yet. Go to 'Scenario Builder' first.")
        return
    
    st.success("✅ Custom scenario data available for download!")
    
    data = st.session_state.forecast_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("GDP Growth", f"{data['gdp']:.2f}%")
    with col2:
        st.metric("Generated", data['timestamp'])
    with col3:
        st.metric("Format", "Excel & CSV")
    
    st.markdown("---")
    
    # Summary data
    summary_data = {
        'Metric': ['GDP Growth', 'Agriculture', 'Manufacturing', 'Services'],
        'Value (%)': [
            data['gdp'],
            data['sectors'].get('Agriculture', 0),
            data['sectors'].get('Manufacturing', 0),
            data['sectors'].get('Services', 0)
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    variables_data = {
        'Variable': list(data['variables'].keys()),
        'Shock': list(data['variables'].values())
    }
    
    variables_df = pd.DataFrame(variables_data)
    
    # Create Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        variables_df.to_excel(writer, sheet_name='Variables', index=False)
    
    excel_buffer.seek(0)
    
    # Create CSV file
    csv_buffer = io.StringIO()
    summary_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"GDP_Forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.download_button(
            label="📄 Download CSV",
            data=csv_content,
            file_name=f"GDP_Forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.subheader("📋 Data Preview")
    
    tab1, tab2 = st.tabs(["Summary", "Variables"])
    
    with tab1:
        st.dataframe(summary_df, use_container_width=True)
    with tab2:
        st.dataframe(variables_df, use_container_width=True)


if __name__ == "__main__":
    main()
