import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from black_scholes import BlackScholes
from monte_carlo import monte_carlo_option_pricing
from binomial_tree import american_fast_tree, plot_binomial_tree_structure

# Page configuration
st.set_page_config(
    page_title="Option Pricing Calculator",
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

def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, option_type, purchase_price):
    """Create PnL heatmap for option prices"""
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
    linkedin_url = "https://www.linkedin.com/in/hajeeong"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Ha Jeong`</a>', unsafe_allow_html=True)

    model = st.radio("Select Model", ["Black-Scholes", "Monte Carlo", "Binomial Tree"])
    
    # Common parameters
    current_price = st.number_input("Current Asset Price", value=100.00, min_value=0.01, step=0.01)
    strike = st.number_input("Strike Price", value=100.00, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.00, min_value=0.01, step=0.01)
    volatility = st.number_input("Volatility (σ)", value=0.20, min_value=0.01, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.00, step=0.01)
    
    # Model-specific parameters
    if model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", value=1000, min_value=100, step=100)
    elif model == "Binomial Tree":
        steps = st.number_input("Number of Steps (N)", value=50, min_value=3, max_value=1000, step=1)
        u_factor = st.number_input("Up Factor (u)", value=1.1, min_value=1.01, max_value=2.0, step=0.01)
        d_factor = 1/u_factor
    
    call_purchase_price = st.number_input("Call Purchase Price", value=0.02, min_value=0.00, step=0.01)
    put_purchase_price = st.number_input("Put Purchase Price", value=0.02, min_value=0.00, step=0.01)

    # Heatmap parameters for Black-Scholes
    if model == "Black-Scholes":
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
input_cols = ['Current Asset Price', 'Strike Price', 'Time to Maturity (Years)', 
              'Volatility (σ)', 'Risk-Free Interest Rate', 'Call Purchase Price', 
              'Put Purchase Price']
if model == "Monte Carlo":
    input_cols.append('Number of Simulations')
elif model == "Binomial Tree":
    input_cols.append('Number of Steps (N)')
    input_cols.append('Up Factor (u)')

values = [current_price, strike, time_to_maturity, volatility, 
          interest_rate, call_purchase_price, put_purchase_price]
if model == "Monte Carlo":
    values.append(num_simulations)
elif model == "Binomial Tree":
    values.append(steps)
    values.append(u_factor)

input_data = pd.DataFrame(values, index=input_cols)
st.table(input_data)

# Calculate prices based on model
if model == "Black-Scholes":
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()
    
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

elif model == "Monte Carlo":
    call_price, price_paths_call = monte_carlo_option_pricing(
        "Call", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations
    )
    put_price, _ = monte_carlo_option_pricing(
        "Put", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations
    )
    
    # Display Monte Carlo visualization
    st.subheader("Monte Carlo Simulation Price Paths")
    df = pd.DataFrame(price_paths_call[:50].T)
    df.columns = [f'Path {i+1}' for i in range(50)]
    df['Time Steps'] = range(len(df))
    
    fig = px.line(df.melt(id_vars=['Time Steps'], var_name='Path', value_name='Stock Price'),
                  x='Time Steps', y='Stock Price', color='Path',
                  title='Simulated Price Paths')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ## Comparative Analysis: Black-Scholes vs. Monte Carlo Simulation
    
    1. **Accuracy:**
       - Black-Scholes: Provides exact solutions for European options under specific assumptions.
       - Monte Carlo: Can be more accurate for complex options or when Black-Scholes assumptions are violated.
    
    2. **Flexibility:**
       - Black-Scholes: Limited to European options with specific assumptions.
       - Monte Carlo: Can handle a wide variety of option types and underlying asset behaviors.
    
    3. **Computation Time:**
       - Black-Scholes: Very fast, provides instant results.
       - Monte Carlo: Can be time-consuming, especially for a large number of simulations.
    
    4. **Assumptions:**
       - Black-Scholes: Assumes constant volatility, no dividends, and log-normal distribution of returns.
       - Monte Carlo: Can incorporate more realistic assumptions like changing volatility or dividend payments.
    
    5. **Visualization:**
       - Black-Scholes: Provides a single price and Greeks.
       - Monte Carlo: Allows visualization of potential price paths, giving insight into the range of possible outcomes.
    """)

else:  # Binomial Tree
    call_price = american_fast_tree(strike, time_to_maturity, current_price, 
                                  interest_rate, steps, u_factor, d_factor, optType='C')
    put_price = american_fast_tree(strike, time_to_maturity, current_price, 
                                 interest_rate, steps, u_factor, d_factor, optType='P')
    
    if st.checkbox("Show Binomial Tree Structure"):
        viz_steps = min(steps, 5)
        fig = plot_binomial_tree_structure(current_price, u_factor, d_factor, viz_steps)
        st.pyplot(fig)

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