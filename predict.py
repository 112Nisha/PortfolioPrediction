# Abstracted prediction logic

import os
import pandas as pd
from pypfopt import expected_returns, risk_models

def compute_portfolio_stats(stocks, weights):
    # Load selected stocks
    dfs = []
    for stock in stocks:
        for fname in os.listdir("data"):
            if fname.lower().startswith(stock.lower()):
                df = pd.read_csv(os.path.join("data", fname), parse_dates=["Date"])
                dfs.append(df[["Date", "Close"]].rename(columns={"Close": stock}))
                break
    if len(dfs) != 2:
        return {
            'mean': 'N/A',
            'variance': 'N/A'
        }
    df = dfs[0].merge(dfs[1], on="Date").set_index("Date")
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    # Convert weights to float (actual money values)
    weights = [float(w) for w in weights]
    weights = weights[:2]
    total = sum(weights)
    if total == 0:
        return {
            'mean': 'N/A',
            'variance': 'N/A'
        }
    fractions = [w/total for w in weights]
    # Compute mean and variance for the portfolio (as fraction)
    mean_frac = sum(mu.values * fractions)
    variance_frac = float(fractions[0]**2 * S.iloc[0,0] + fractions[1]**2 * S.iloc[1,1] + 2*fractions[0]*fractions[1]*S.iloc[0,1])
    # Convert to money values
    mean_money = mean_frac * total
    risk_money = variance_frac * total
    return {
        'mean': f"{mean_money:.2f}",
        'variance': f"{risk_money:.2f}"
    }
