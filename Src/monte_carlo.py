import numpy as np
import pandas as pd

def run_monte_carlo(prices_series, days_ahead=30, simulations=1000):
    print("Initializing Monte Carlo...")
    
    daily_returns = prices_series.pct_change().dropna()
    
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    
    last_price = prices_series.iloc[-1]
    
    simulation_df = np.zeros((days_ahead, simulations))
    

    for x in range(simulations):
        count = 0
        daily_vol = np.random.normal(loc=mu, scale=sigma, size=days_ahead)
        price_series = [last_price]
        
        for y in daily_vol:
            price_series.append(price_series[-1] * (1 + y))
            
        simulation_df[:, x] = price_series[1:]
        
    print("Simulations completed.")
    return pd.DataFrame(simulation_df)

def calculate_risk_metrics(simulation_df):

    final_prices = simulation_df.iloc[-1]
    
    var_95 = np.percentile(final_prices, 5)
    var_99 = np.percentile(final_prices, 1)
    
    return var_95, var_99

if __name__ == "__main__":
    dummy_prices = pd.Series(np.random.normal(2000, 50, 100))
    #Test simulation
    results = run_monte_carlo(dummy_prices)
    #Test metrics
    v95, v99 = calculate_risk_metrics(results)
    
    print("\n--- RISK METRICS ---")
    print(f"95% Confidence Value: {v95:.2f}")
    print(f"99% Confidence Value: {v99:.2f}")