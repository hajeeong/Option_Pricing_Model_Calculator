# import library
import numpy as np
from functools import wraps
from time import time

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Initialise parameters
S0 = 100      # initial stock price
K = 100       # strike price
T = 1         # time to maturity in years
r = 0.06      # annual risk-free rate
N = 3         # number of time steps
u = 1.1       # up-factor in binomial models
d = 1/u       # ensure recombining tree
optType = 'P' # Option Type 'C' or 'P'

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

@timing
def american_slow_tree(K,T,S0,r,N,u,d,optType='P'):
    #precompute values
    dt = T/N
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)

    # initialize stock prices at maturity
    S = np.zeros(N+1)
    for j in range(0, N+1):
        S[j] = S0 * u**j * d**(N-j)

    # option payoff
    C = np.zeros(N+1)
    for j in range(0, N+1):
        if optType == 'P':
            C[j] = max(0, K - S[j])
        else:
            C[j] = max(0, S[j] - K)

    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        for j in range(0,i+1):
            S = S0 * u**j * d**(i-j)
            C[j] = disc * ( q*C[j+1] + (1-q)*C[j] )
            if optType == 'P':
                C[j] = max(C[j], K - S)
            else:
                C[j] = max(C[j], S - K)

    return C[0]

american_slow_tree(K,T,S0,r,N,u,d,optType='P')

@timing
def american_fast_tree(K,T,S0,r,N,u,d,optType='P'):
    #precompute values
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

american_fast_tree(K,T,S0,r,N,u,d,optType='P')

for N in [3,50, 100, 1000, 5000]:
    american_fast_tree(K,T,S0,r,N,u,d,optType='P')
    american_slow_tree(K,T,S0,r,N,u,d,optType='P')

def plot_binomial_tree_structure(S0, u, d, N):
    """Plot the structure of the binomial tree"""
    # Calculate stock prices at each node
    stock_prices = []
    for i in range(N+1):
        prices = []
        for j in range(i+1):
            price = S0 * (u**j) * (d**(i-j))
            prices.append(price)
        stock_prices.append(prices)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot nodes and connections
    for i in range(N+1):
        x = [i] * len(stock_prices[i])
        y = stock_prices[i]
        
        # Plot nodes
        plt.scatter(x, y, c='blue', s=100)
        
        # Plot connections
        if i < N:
            for j in range(len(stock_prices[i])):
                plt.plot([i, i+1], 
                        [stock_prices[i][j], stock_prices[i+1][j]], 
                        'b--', alpha=0.3)
                plt.plot([i, i+1], 
                        [stock_prices[i][j], stock_prices[i+1][j+1]], 
                        'b--', alpha=0.3)
    
    plt.title(f'Binomial Tree Structure (N={N} steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.grid(True)
    return plt

def plot_performance_comparison(N_values=[3, 50, 100, 1000, 5000]):
    """Plot performance comparison between slow and fast implementations"""
    # Calculate execution times
    slow_times = []
    fast_times = []
    
    for N in N_values:
        ts = time()
        american_slow_tree(K,T,S0,r,N,u,d,optType='P')
        slow_times.append(time() - ts)
        
        ts = time()
        american_fast_tree(K,T,S0,r,N,u,d,optType='P')
        fast_times.append(time() - ts)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(N_values, slow_times, 'r-o', label='Slow Implementation')
    plt.plot(N_values, fast_times, 'g-o', label='Fast Implementation')
    plt.title('Performance Comparison: Slow vs Fast Implementation')
    plt.xlabel('Number of Steps (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    return plt

def plot_option_values(N_range=[3, 5, 10, 20, 50]):
    """Plot option values for different parameters"""
    strike_prices = np.linspace(80, 120, 20)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for N in N_range:
        option_values = []
        for K in strike_prices:
            value = american_fast_tree(K,T,S0,r,N,u,d,optType='P')
            option_values.append(value)
        
        ax.plot(strike_prices, [N]*len(strike_prices), option_values, 
                label=f'N={N}')
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Number of Steps (N)')
    ax.set_zlabel('Option Value')
    ax.set_title('American Put Option Values')
    ax.legend()
    return plt

# # Generate and display visualizations
# plt.style.use('default')  # Using default style instead of seaborn

# # 1. Plot binomial tree structure with improved styling
# fig1 = plot_binomial_tree_structure(S0, u, d, 4)
# plt.tight_layout()
# plt.show()

# # 2. Plot performance comparison with improved styling
# fig2 = plot_performance_comparison()
# plt.tight_layout()
# plt.show()

# # 3. Plot option values with improved styling
# fig3 = plot_option_values()
# plt.tight_layout()
# plt.show()

# # Additional analysis: Print summary statistics
# print("\nSummary Statistics:")
# print("-" * 50)
# print(f"Initial Stock Price (S0): ${S0}")
# print(f"Strike Price (K): ${K}")
# print(f"Risk-free Rate (r): {r*100}%")
# print(f"Up Factor (u): {u}")
# print(f"Down Factor (d): {d:.4f}")
# print("-" * 50)
# print("\nOption Values for different N:")
# for N in [3, 10, 50, 100]:
#     value = american_fast_tree(K,T,S0,r,N,u,d,optType='P')
#     print(f"N = {N:4d}: ${value:.4f}")

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def analyze_real_market(ticker="AAPL", days_history=180):
    """Analyze real market data and compare with binomial model"""
    # Fetch stock data
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_history)
    hist_data = stock.history(start=start_date, end=end_date)
    
    # Calculate historical volatility
    returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    hist_vol = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Get current option chain
    try:
        options = stock.option_chain(stock.options[0])  # Get nearest expiration date
        puts = options.puts
        current_price = hist_data['Close'][-1]
        
        # Calculate theoretical prices using our model
        theoretical_prices = []
        market_prices = []
        strikes = []
        
        # Calculate time to expiration in years
        expiry_date = datetime.strptime(stock.options[0], '%Y-%m-%d')
        T = (expiry_date - datetime.now()).days / 365
        
        # Calculate u and d based on historical volatility
        u = np.exp(hist_vol * np.sqrt(T/50))  # Using 50 steps
        d = 1/u
        
        # Compare market vs model prices
        for idx, put in puts.iterrows():
            strike = put['strike']
            if 0.8 * current_price <= strike <= 1.2 * current_price:  # Only strikes near current price
                model_price = american_fast_tree(
                    K=strike,
                    T=T,
                    S0=current_price,
                    r=0.05,  # Using 5% as risk-free rate
                    N=50,
                    u=u,
                    d=d,
                    optType='P'
                )
                theoretical_prices.append(model_price)
                market_prices.append(put['lastPrice'])
                strikes.append(strike)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        plt.plot(strikes, market_prices, 'bo-', label='Market Prices')
        plt.plot(strikes, theoretical_prices, 'ro-', label='Model Prices')
        plt.axvline(x=current_price, color='g', linestyle='--', label='Current Stock Price')
        plt.title(f'Put Option Prices: Market vs Model ({ticker})')
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot stock price history
        plt.figure(figsize=(12, 6))
        plt.plot(hist_data.index, hist_data['Close'], label='Stock Price')
        plt.plot(hist_data.index, hist_data['MA20'] if 'MA20' in hist_data else hist_data['Close'].rolling(20).mean(),
                label='20-day MA', alpha=0.7)
        plt.title(f'{ticker} Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print analysis
        print(f"\nReal Market Analysis for {ticker}:")
        print("-" * 50)
        print(f"Current Stock Price: ${current_price:.2f}")
        print(f"Historical Volatility: {hist_vol*100:.2f}%")
        print(f"Days to Expiration: {(expiry_date - datetime.now()).days}")
        print("\nModel vs Market Comparison:")
        print("Strike Price | Market Price | Model Price | Difference")
        print("-" * 60)
        for s, m, t in zip(strikes, market_prices, theoretical_prices):
            print(f"${s:10.2f} | ${m:11.2f} | ${t:10.2f} | ${abs(m-t):9.2f}")
            
    except Exception as e:
        print(f"Error fetching option data: {str(e)}")
        return None
    
# Example usage
analyze_real_market("AAPL")  # Analyze Apple stock
# You can try other stocks like:
# analyze_real_market("MSFT")  # Microsoft
# analyze_real_market("GOOGL")  # Google
# analyze_real_market("AMZN")  # Amazon