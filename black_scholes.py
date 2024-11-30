import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        dividend_yield: float = 0.0
    ):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.dividend_yield = dividend_yield

    def _calculate_d1_d2(self):
        S, K, T, r, q, sigma = (
            self.current_price,
            self.strike,
            self.time_to_maturity,
            self.interest_rate,
            self.dividend_yield,
            self.volatility,
        )
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def calculate_prices(self):  # Changed method name to match streamlit app
        S, K, T, r, q = (
            self.current_price,
            self.strike,
            self.time_to_maturity,
            self.interest_rate,
            self.dividend_yield,
        )
        d1, d2 = self._calculate_d1_d2()

        # Calculate option prices
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        # Calculate Greeks
        self.call_delta = np.exp(-q * T) * norm.cdf(d1)
        self.put_delta = -np.exp(-q * T) * norm.cdf(-d1)
        self.gamma = np.exp(-q * T) * norm.pdf(d1) / (S * self.volatility * np.sqrt(T))
        self.vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        self.call_theta = (-S * np.exp(-q * T) * norm.pdf(d1) * self.volatility / (2 * np.sqrt(T)) 
                          - r * K * np.exp(-r * T) * norm.cdf(d2) 
                          + q * S * np.exp(-q * T) * norm.cdf(d1))
        self.put_theta = (-S * np.exp(-q * T) * norm.pdf(d1) * self.volatility / (2 * np.sqrt(T)) 
                         + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                         - q * S * np.exp(-q * T) * norm.cdf(-d1))
        self.call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        self.put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return call_price, put_price

if __name__ == "__main__":
    # Example usage
    bs = BlackScholes(
        time_to_maturity=1,
        strike=100,
        current_price=100,
        volatility=0.2,
        interest_rate=0.05,
        dividend_yield=0.02
    )

    call_price, put_price = bs.calculate_prices()
    print(f"Call Price: ${call_price:.4f}")
    print(f"Put Price: ${put_price:.4f}")
    print("\nGreeks:")
    print(f"Call Delta: {bs.call_delta:.4f}")
    print(f"Put Delta: {bs.put_delta:.4f}")
    print(f"Gamma: {bs.gamma:.4f}")
    print(f"Vega: {bs.vega:.4f}")
    print(f"Call Theta: {bs.call_theta:.4f}")
    print(f"Put Theta: {bs.put_theta:.4f}")
    print(f"Call Rho: {bs.call_rho:.4f}")
    print(f"Put Rho: {bs.put_rho:.4f}")