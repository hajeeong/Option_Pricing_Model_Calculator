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

    def calculate_option_prices(self):
        S, K, T, r, q = (
            self.current_price,
            self.strike,
            self.time_to_maturity,
            self.interest_rate,
            self.dividend_yield,
        )
        d1, d2 = self._calculate_d1_d2()

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return call_price, put_price

    def calculate_greeks(self):
        S, K, T, r, q, sigma = (
            self.current_price,
            self.strike,
            self.time_to_maturity,
            self.interest_rate,
            self.dividend_yield,
            self.volatility,
        )
        d1, d2 = self._calculate_d1_d2()

        # Delta
        call_delta = np.exp(-q * T) * norm.cdf(d1)
        put_delta = -np.exp(-q * T) * norm.cdf(-d1)

        # Gamma
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Vega
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Theta
        call_theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                      - r * K * np.exp(-r * T) * norm.cdf(d2) 
                      + q * S * np.exp(-q * T) * norm.cdf(d1))
        put_theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2) 
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))

        # Rho
        call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'vega': vega,
            'call_theta': call_theta,
            'put_theta': put_theta,
            'call_rho': call_rho,
            'put_rho': put_rho
        }

    def run(self):
        try:
            self.call_price, self.put_price = self.calculate_option_prices()
            self.greeks = self.calculate_greeks()
            return True
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

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

    if bs.run():
        print(f"Call Price: {bs.call_price:.4f}")
        print(f"Put Price: {bs.put_price:.4f}")
        print("\nGreeks:")
        for greek, value in bs.greeks.items():
            print(f"{greek}: {value:.4f}")
    else:
        print("Failed to calculate option prices and Greeks.")