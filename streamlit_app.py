import numpy as np
from functools import wraps
from time import time
import plotly.graph_objects as go
import plotly.express as px

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
    stock_prices = []
    x_coords = []
    y_coords = []
    text = []
    
    for i in range(N+1):
        for j in range(i+1):
            price = S0 * (u**j) * (d**(i-j))
            x_coords.append(i)
            y_coords.append(price)
            text.append(f'Step: {i}<br>Price: ${price:.2f}')
            
            # Add edges
            if i < N:
                x_edges = []
                y_edges = []
                # Up movement
                x_edges.extend([i, i+1, None])
                y_edges.extend([price, price*u, None])
                # Down movement
                x_edges.extend([i, i+1, None])
                y_edges.extend([price, price*d, None])
                
    # Create the plot
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

# Add this to your existing streamlit app where you handle model selection
if model == "Binomial Tree":
    st.subheader("Binomial Tree Parameters")
    
    # Additional parameters for binomial tree
    steps = st.number_input("Number of Steps (N)", value=50, min_value=3, max_value=1000, step=1)
    u_factor = st.number_input("Up Factor (u)", value=1.1, min_value=1.01, max_value=2.0, step=0.01)
    d_factor = 1/u_factor
    
    # Calculate option prices using binomial tree
    call_price = american_fast_tree(strike, time_to_maturity, current_price, 
                                  interest_rate, steps, u_factor, d_factor, optType='C')
    put_price = american_fast_tree(strike, time_to_maturity, current_price, 
                                 interest_rate, steps, u_factor, d_factor, optType='P')
    
    # Display tree structure
    if st.checkbox("Show Binomial Tree Structure", value=False):
        if steps > 20:
            st.warning("Tree visualization is limited to 20 steps for clarity. Reducing steps to 20 for visualization only.")
            viz_steps = 20
        else:
            viz_steps = steps
        
        fig = plot_binomial_tree_interactive(current_price, u_factor, d_factor, viz_steps)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display convergence analysis
    if st.checkbox("Show Convergence Analysis", value=False):
        step_range = [10, 20, 50, 100, 200, 500]
        call_values = []
        put_values = []
        
        for n in step_range:
            call_values.append(american_fast_tree(strike, time_to_maturity, current_price, 
                                                interest_rate, n, u_factor, d_factor, optType='C'))
            put_values.append(american_fast_tree(strike, time_to_maturity, current_price, 
                                               interest_rate, n, u_factor, d_factor, optType='P'))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=step_range, y=call_values, name='Call Option', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=step_range, y=put_values, name='Put Option', line=dict(color='orange')))
        fig.update_layout(title='Option Price Convergence Analysis',
                         xaxis_title='Number of Steps',
                         yaxis_title='Option Price',
                         showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add comparison with other models
    if st.checkbox("Compare with Other Models", value=False):
        st.write("Model Comparison:")
        comparison_df = pd.DataFrame({
            'Model': ['Black-Scholes', 'Monte Carlo', 'Binomial Tree'],
            'Call Price': [
                BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate).calculate_prices()[0],
                monte_carlo_option_pricing("Call", current_price, strike, volatility, interest_rate, time_to_maturity, 1000)[0],
                call_price
            ],
            'Put Price': [
                BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate).calculate_prices()[1],
                monte_carlo_option_pricing("Put", current_price, strike, volatility, interest_rate, time_to_maturity, 1000)[0],
                put_price
            ]
        })
        st.table(comparison_df.style.format({'Call Price': '${:.2f}', 'Put Price': '${:.2f}'}))