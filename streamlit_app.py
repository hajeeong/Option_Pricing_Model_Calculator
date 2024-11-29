import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps
from time import time

# Import your model classes and functions
from black_scholes import BlackScholes
from monte_carlo import monte_carlo_option_pricing

# Binomial Tree functions
def american_fast_tree(K, T, S0, r, N, u, d, optType='P'):
    dt = T/N
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)

    # initialise stock prices at maturity
    S = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))

    # option payoff
    if optType == 'P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)

    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        C[:i+1] = disc * ( q*C[1:i+2] + (1-q)*C[0:i+1] )
        C = C[:-1]
        if optType == 'P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)

    return C[0]

def plot_binomial_tree_interactive(S0, u, d, N):
    """Create an interactive plotly visualization of the binomial tree"""
    # Calculate stock prices at each node
    x_coords = []
    y_coords = []
    text = []
    x_edges = []
    y_edges = []
    
    for i in range(N+1):
        for j in range(i+1):
            price = S0 * (u**j) * (d**(i-j))
            x_coords.append(i)
            y_coords.append(price)
            text.append(f'Step: {i}<br>Price: ${price:.2f}')
            
            # Add edges
            if i < N:
                # Up movement
                x_edges.extend([i, i+1, None])
                y_edges.extend([price, price*u, None])
                # Down movement
                x_edges.extend([i, i+1, None])
                y_edges.extend([price, price*d, None])
    
    fig = go.Figure()
    
    # Add edges (lines)
    fig.add_trace(go.Scatter(
        x=x_edges,
        y=y_edges,
        mode='lines',
        line=dict(color='lightgray', width=1),
        hoverinfo='skip'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker=dict(size=10, color='blue'),
        text=text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Binomial Tree Structure',
        xaxis_title='Time Step',
        yaxis_title='Stock Price',
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

# Page configuration
st.set_page_config(
    page_title="Option Pricing Models",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar inputs
with st.sidebar:
    st.title("📊 Option Pricing Models")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/akshat-kulshreshtha-9314421a2/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Akshat Kulshreshtha`</a>', unsafe_allow_html=True)

    # Model selection (now defined in the sidebar)
    model = st.radio("Select Model", ["Black-Scholes", "Monte Carlo", "Binomial Tree"])

    # Common parameters
    current_price = st.number_input("Current Asset Price", value=100.00, min_value=0.01, step=0.01)
    strike = st.number_input("Strike Price", value=100.00, min_value=0.01, step=0.01)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, step=0.01)
    volatility = st.number_input("Volatility (σ)", value=0.20, min_value=0.01, step=0.01)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05, min_value=0.00, step=0.01)

    # Model-specific parameters
    if model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", value=1000, min_value=100, step=100)
    elif model == "Binomial Tree":
        steps = st.number_input("Number of Steps (N)", value=50, min_value=3, max_value=1000, step=1)
        u_factor = st.number_input("Up Factor (u)", value=1.1, min_value=1.01, max_value=2.0, step=0.01)
        d_factor = 1/u_factor

# Main content
st.title("Option Pricing Models")

# Calculate option prices based on selected model
if model == "Black-Scholes":
    bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
    call_price, put_price = bs_model.calculate_prices()
    
elif model == "Monte Carlo":
    call_price, _ = monte_carlo_option_pricing("Call", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations)
    put_price, _ = monte_carlo_option_pricing("Put", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations)
    
else:  # Binomial Tree
    call_price = american_fast_tree(strike, time_to_maturity, current_price, interest_rate, steps, u_factor, d_factor, optType='C')
    put_price = american_fast_tree(strike, time_to_maturity, current_price, interest_rate, steps, u_factor, d_factor, optType='P')
    
    if st.checkbox("Show Binomial Tree Structure", value=False):
        if steps > 20:
            st.warning("Tree visualization is limited to 20 steps for clarity. Reducing steps to 20 for visualization only.")
            viz_steps = 20
        else:
            viz_steps = steps
        
        fig = plot_binomial_tree_interactive(current_price, u_factor, d_factor, viz_steps)
        st.plotly_chart(fig, use_container_width=True)

# Display results
st.subheader("Option Prices")
col1, col2 = st.columns(2)

with col1:
    st.metric("Call Option Price", f"${call_price:.2f}")

with col2:
    st.metric("Put Option Price", f"${put_price:.2f}")

# Display input parameters
st.subheader("Input Parameters")
input_data = {
    "Parameter": ["Current Price", "Strike Price", "Time to Maturity", "Volatility", "Interest Rate"],
    "Value": [current_price, strike, time_to_maturity, volatility, interest_rate]
}
input_df = pd.DataFrame(input_data)
st.table(input_df)