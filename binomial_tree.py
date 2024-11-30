import numpy as np
from functools import wraps
from time import time
import matplotlib.pyplot as plt

class BinomialTree:
    def __init__(self, strike, time_to_maturity, current_price, interest_rate, steps, up_factor, down_factor=None):
        """
        Initialize the Binomial Tree model.
        
        Parameters:
        -----------
        strike : float
            Strike price of the option
        time_to_maturity : float
            Time to maturity in years
        current_price : float
            Current price of the underlying asset
        interest_rate : float
            Risk-free interest rate
        steps : int
            Number of time steps in the tree
        up_factor : float
            Up movement factor
        down_factor : float, optional
            Down movement factor. If not provided, calculated as 1/up_factor
        """
        self.K = strike
        self.T = time_to_maturity
        self.S0 = current_price
        self.r = interest_rate
        self.N = steps
        self.u = up_factor
        self.d = 1/up_factor if down_factor is None else down_factor
        
        # Precompute common values
        self.dt = self.T/self.N
        self.q = (np.exp(self.r * self.dt) - self.d)/(self.u - self.d)
        self.disc = np.exp(-self.r * self.dt)

    def _timing_decorator(func):
        """Decorator to measure function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            print(f'Function {func.__name__} took {end_time - start_time:.4f} seconds')
            return result
        return wrapper

    @_timing_decorator
    def price_option_slow(self, option_type='P'):
        """
        Price an American option using the slow (traditional) implementation.
        
        Parameters:
        -----------
        option_type : str
            'C' for Call option, 'P' for Put option
            
        Returns:
        --------
        float : Option price
        """
        # Initialize stock prices at maturity
        S = np.zeros(self.N + 1)
        for j in range(0, self.N + 1):
            S[j] = self.S0 * self.u**j * self.d**(self.N-j)

        # Calculate option payoff
        C = np.zeros(self.N + 1)
        for j in range(0, self.N + 1):
            if option_type == 'P':
                C[j] = max(0, self.K - S[j])
            else:
                C[j] = max(0, S[j] - self.K)

        # Backward recursion through the tree
        for i in np.arange(self.N-1, -1, -1):
            for j in range(0, i+1):
                S = self.S0 * self.u**j * self.d**(i-j)
                C[j] = self.disc * (self.q * C[j+1] + (1-self.q) * C[j])
                if option_type == 'P':
                    C[j] = max(C[j], self.K - S)
                else:
                    C[j] = max(C[j], S - self.K)

        return C[0]

    @_timing_decorator
    def price_option_fast(self, option_type='P'):
        """
        Price an American option using the fast (vectorized) implementation.
        
        Parameters:
        -----------
        option_type : str
            'C' for Call option, 'P' for Put option
            
        Returns:
        --------
        float : Option price
        """
        # Initialize stock prices at maturity
        S = self.S0 * self.d**(np.arange(self.N,-1,-1)) * self.u**(np.arange(0,self.N+1,1))

        # Calculate option payoff
        if option_type == 'P':
            C = np.maximum(0, self.K - S)
        else:
            C = np.maximum(0, S - self.K)

        # Backward recursion through the tree
        for i in np.arange(self.N-1, -1, -1):
            S = self.S0 * self.d**(np.arange(i,-1,-1)) * self.u**(np.arange(0,i+1,1))
            C[:i+1] = self.disc * (self.q * C[1:i+2] + (1-self.q) * C[0:i+1])
            C = C[:-1]
            if option_type == 'P':
                C = np.maximum(C, self.K - S)
            else:
                C = np.maximum(C, S - self.K)

        return C[0]

    def plot_tree_structure(self):
        """
        Create a visual representation of the binomial tree structure.
        
        Returns:
        --------
        matplotlib.pyplot : Plot object
        """
        # Calculate stock prices at each node
        stock_prices = []
        for i in range(self.N + 1):
            prices = []
            for j in range(i + 1):
                price = self.S0 * (self.u**j) * (self.d**(i-j))
                prices.append(price)
            stock_prices.append(prices)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot nodes and connections
        for i in range(self.N + 1):
            x = [i] * len(stock_prices[i])
            y = stock_prices[i]
            
            # Plot nodes
            plt.scatter(x, y, c='blue', s=100)
            
            # Plot connections
            if i < self.N:
                for j in range(len(stock_prices[i])):
                    plt.plot([i, i+1], 
                            [stock_prices[i][j], stock_prices[i+1][j]], 
                            'b--', alpha=0.3)
                    plt.plot([i, i+1], 
                            [stock_prices[i][j], stock_prices[i+1][j+1]], 
                            'b--', alpha=0.3)
        
        plt.title(f'Binomial Tree Structure (N={self.N} steps)')
        plt.xlabel('Time Step')
        plt.ylabel('Stock Price')
        plt.grid(True)
        return plt

def american_fast_tree(K, T, S0, r, N, u, d, optType='P'):
    """
    Wrapper function for backward compatibility with the original implementation.
    """
    model = BinomialTree(K, T, S0, r, N, u, d)
    return model.price_option_fast(optType)

def plot_binomial_tree_structure(S0, u, d, N):
    """
    Wrapper function for backward compatibility with the original implementation.
    """
    model = BinomialTree(100, 1, S0, 0.05, N, u, d)  # Dummy values for K, T, r
    return model.plot_tree_structure()

if __name__ == "__main__":
    # Example usage
    model = BinomialTree(
        strike=100,
        time_to_maturity=1,
        current_price=100,
        interest_rate=0.06,
        steps=50,
        up_factor=1.1
    )
    
    # Price options
    put_price_slow = model.price_option_slow('P')
    put_price_fast = model.price_option_fast('P')
    call_price_slow = model.price_option_slow('C')
    call_price_fast = model.price_option_fast('C')
    
    print(f"\nOption Prices:")
    print(f"Put (Slow): ${put_price_slow:.4f}")
    print(f"Put (Fast): ${put_price_fast:.4f}")
    print(f"Call (Slow): ${call_price_slow:.4f}")
    print(f"Call (Fast): ${call_price_fast:.4f}")
    
    # Plot tree structure
    plt.style.use('default')
    model.plot_tree_structure()
    plt.show()