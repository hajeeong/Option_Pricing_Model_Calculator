import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def monte_carlo_option_pricing(option_type, S, K, sigma, r, T, num_simulations):
    """Monte Carlo simulation for option pricing with price paths"""
    dt = T / 252  # Daily steps
    num_steps = int(252 * T)  # Number of time steps
    nudt = (r - 0.5 * sigma**2) * dt
    sidt = sigma * np.sqrt(dt)
    
    # Generate random paths
    Z = np.random.standard_normal((num_simulations, num_steps))
    S_t = np.zeros((num_simulations, num_steps + 1))
    S_t[:, 0] = S
    
    # Calculate price paths
    for t in range(1, num_steps + 1):
        S_t[:, t] = S_t[:, t-1] * np.exp(nudt + sidt * Z[:, t-1])
    
    # Calculate option payoffs
    if option_type == "Call":
        payoffs = np.maximum(S_t[:, -1] - K, 0)
    else:  # Put option
        payoffs = np.maximum(K - S_t[:, -1], 0)
    
    # Calculate option price
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price, S_t

def plot_price_paths(price_paths, current_price, num_paths_to_show=50):
    """Create an interactive plot of price paths"""
    # Convert price paths to DataFrame
    df = pd.DataFrame(price_paths[:num_paths_to_show].T)
    df.columns = [f'Path {i+1}' for i in range(num_paths_to_show)]
    df['Time Steps'] = range(len(df))
    
    # Melt the DataFrame for plotting
    df_melted = df.melt(id_vars=['Time Steps'], var_name='Path', value_name='Stock Price')
    
    # Create the plot using Plotly Express
    fig = px.line(df_melted, x='Time Steps', y='Stock Price', color='Path',
                  title='Simulated Price Paths')
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        hovermode='x unified',
        height=600
    )
    
    # Add horizontal line for initial price
    fig.add_hline(y=current_price, line_dash="dash", line_color="black", opacity=0.5)
    
    return fig

# Update your main Streamlit app
if model == "Monte Carlo":
    # Calculate option prices
    call_price, price_paths_call = monte_carlo_option_pricing(
        "Call", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations
    )
    put_price, _ = monte_carlo_option_pricing(
        "Put", current_price, strike, volatility, interest_rate, time_to_maturity, num_simulations
    )
    
    # Display option values with styling
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
    
    # Monte Carlo Simulation Price Paths
    st.subheader("Monte Carlo Simulation Price Paths")
    fig = plot_price_paths(price_paths_call, current_price)
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparative Analysis section
    st.subheader("Comparative Analysis: Black-Scholes vs. Monte Carlo Simulation")
    
    st.markdown("""
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