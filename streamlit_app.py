import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Option Pricing Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px;
    width: auto;
    margin: 0 auto;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.metric-call {
    background-color: #e6f3ff;
    color: #0066cc;
}

.metric-put {
    background-color: #fff0e6;
    color: #cc6600;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    margin: 0;
}

.metric-label {
    font-size: 1.2rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (np.log(self.current_price / self.strike) + 
              (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (
                  self.volatility * np.sqrt(self.time_to_maturity)
              )
        d2 = d1 - self.volatility * np.sqrt(self.time_to_maturity)

        self.call_price = self.current_price * norm.cdf(d1) - (
            self.strike * np.exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(d2)
        )
        self.put_price = (
            self.strike * np.exp(-(self.interest_rate * self.time_to_maturity)) * norm.cdf(-d2)
        ) - self.current_price * norm.cdf(-d1)

        # Greeks
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        self.gamma = norm.pdf(d1) / (self.current_price * self.volatility * np.sqrt(self.time_to_maturity))
        self.vega = self.current_price * norm.pdf(d1) * np.sqrt(self.time_to_maturity)
        self.call_theta = -(self.current_price * norm.pdf(d1) * self.volatility / (2 * np.sqrt(self.time_to_maturity))) - \
                          self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        self.put_theta = -(self.current_price * norm.pdf(d1) * self.volatility / (2 * np.sqrt(self.time_to_maturity))) + \
                         self.interest_rate * self.strike * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)
        self.call_rho = self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        self.put_rho = -self.strike * self.time_to_maturity * np.exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2)

        return self.call_price, self.put_price

def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, option_type, purchase_price):
    pnl = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_model.current_price = spot
            bs_model.volatility = vol
            call_price, put_price = bs_model.calculate_prices()
            if option_type == 'call':
                pnl[i, j] = call_price - purchase_price
            else:
                pnl[i, j] = put_price - purchase_price
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pnl, xticklabels=spot_range.round(2), yticklabels=vol_range.round(2), 
                cmap='RdYlGn', center=0, ax=ax)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    ax.set_title(f'{option_type.capitalize()} Option PnL')
    return fig

# Sidebar
with st.sidebar:
    st.title("📊 Option Pricing Models")
    st.write("`Created by:`")
    linkedin_url = "www.linkedin.com/in/hajeeong"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Ha Jeong`</a>', unsafe_allow_html=True)

    model = st.radio("Select Model", ["Black-Scholes", "Monte Carlo"])
    
    current_price = st.number_input("Current Asset Price", value=100.00, min_value=0.01, step=0.01)
    strike = st.number_input("Strike Price", value=100.00, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.00, min_value=0.01, step=0.01)
    volatility = st.number_input("Volatility (σ)", value=0.20, min_value=0.01, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.00, step=0.01)
    
    call_purchase_price = st.number_input("Call Purchase Price", value=0.02, min_value=0.00, step=0.01)
    put_purchase_price = st.number_input("Put Purchase Price", value=0.02, min_value=0.00, step=0.01)

    st.markdown("---")
    st.subheader("Heatmap Parameters")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=80.00, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=120.00, step=0.01)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=0.10, step=0.01)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=0.30, step=0.01)

# Main content
st.title("Option Pricing Models")

# Input Parameters Display
st.subheader("Input Parameters")
input_data = pd.DataFrame({
    'Current Asset Price': [current_price],
    'Strike Price': [strike],
    'Time to Maturity (Years)': [time_to_maturity],
    'Volatility (σ)': [volatility],
    'Risk-Free Interest Rate': [interest_rate],
    'Call Purchase Price': [call_purchase_price],
    'Put Purchase Price': [put_purchase_price]
}).T
st.table(input_data)

# Calculate prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display option values
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Display Greeks
st.subheader("Option Greeks")
greeks_data = pd.DataFrame({
    'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
    'Call': [bs_model.call_delta, bs_model.gamma, bs_model.vega, bs_model.call_theta, bs_model.call_rho],
    'Put': [bs_model.put_delta, bs_model.gamma, bs_model.vega, bs_model.put_theta, bs_model.put_rho]
})
st.table(greeks_data.set_index('Greek').style.format("{:.4f}"))

# PnL Heatmaps
st.title("Options PnL - Interactive Heatmap")
st.info("Explore how option PnL changes with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, while maintaining a constant 'Strike Price'.")

spot_range = np.linspace(spot_min, spot_max, 20)
vol_range = np.linspace(vol_min, vol_max, 20)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Call Option PnL Heatmap")
    fig_call = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, 'call', call_purchase_price)
    st.pyplot(fig_call)

with col2:
    st.subheader("Put Option PnL Heatmap")
    fig_put = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, 'put', put_purchase_price)
    st.pyplot(fig_put)