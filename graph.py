

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
import os

def generate_frontier_graph(stocks, weights):
    """
    Generates and saves the mean-variance, CVaR, and VaR frontiers, and marks common-optimal portfolios.
    Assumes stocks is a list of two stock symbols, weights is a list of two floats.
    Saves the main frontier plot to static/portfolio_frontier.png.
    """
    # Dynamically load selected stocks from /data/
    dfs = []
    for stock in stocks:
        # Find the matching file (case-insensitive, partial match)
        for fname in os.listdir("data"):
            if fname.lower().startswith(stock.lower()):
                df = pd.read_csv(os.path.join("data", fname), parse_dates=["Date"])
                dfs.append(df[["Date", "Close"]].rename(columns={"Close": stock}))
                break
    if len(dfs) != 2:
        # Not enough valid stocks selected
        return
    # Merge on Date
    df = dfs[0].merge(dfs[1], on="Date").set_index("Date")
    rets = df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(df)
    S  = risk_models.sample_cov(df)

    # helpers
    def portfolio_var(w, returns_df, alpha=0.05):
        pr = returns_df @ w
        return -np.percentile(pr, alpha*100)

    def portfolio_return(w, mu):
        return w @ mu

    def as_series(w, index):
        return pd.Series(w, index=index).sort_index()

    grid = np.linspace(mu.min()+1e-6, mu.max()-1e-6, 100)
    mv_pts, cvar_pts, var_pts = [], [], []
    w_tol = 1e-2
    num_assets = len(mu)
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets,])

    for r in grid:
        try:
            ef_mv = EfficientFrontier(mu, S)
            ef_mv.efficient_return(r)
            w_mv = as_series(ef_mv.clean_weights(), mu.index)
            ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
        except Exception:
            continue
        try:
            ef_c = EfficientCVaR(mu, rets)
            ef_c.efficient_return(r)
            w_c = as_series(ef_c.clean_weights(), mu.index)
            ret_c, cvar_risk = ef_c.portfolio_performance()
        except Exception:
            continue
        try:
            result = minimize(
                portfolio_var,
                initial_guess,
                args=(rets,),
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                    {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, mu) - r}
                ]
            )
            if result.success:
                w_var = as_series(result.x, mu.index)
                var_risk_val = portfolio_var(w_var, rets)
                ret_var = portfolio_return(w_var, mu)
                var_pts.append((var_risk_val, ret_var, w_var))
        except Exception:
            continue
        mv_pts.append((vol_mv, ret_mv, w_mv))
        cvar_pts.append((cvar_risk, ret_c, w_c))

    # Find common-optimal portfolios
    def weight_key(w, tol=1e-2):
        return tuple(np.round(w.values / tol) * tol)
    mv_dict   = {weight_key(w): (risk, ret, w) for risk, ret, w in mv_pts}
    cvar_dict = {weight_key(w): (risk, ret, w) for risk, ret, w in cvar_pts}
    var_dict  = {weight_key(w): (risk, ret, w) for risk, ret, w in var_pts}
    common_keys = set(mv_dict.keys()) & set(cvar_dict.keys()) & set(var_dict.keys())
    common_pts = []
    for k in common_keys:
        risk_mv, ret_mv, w_mv = mv_dict[k]
        risk_c, ret_c, w_c    = cvar_dict[k]
        risk_v, ret_v, w_v    = var_dict[k]
        common_pts.append({
            "ret": float(ret_mv),
            "w": w_mv,
            "mv_risk": float(risk_mv),
            "cvar_risk": float(risk_c),
            "var_risk": float(risk_v),
        })

    # Prepare Plotly traces for interactive graph
    traces = []
    for pts, name, color in zip([mv_pts, cvar_pts, var_pts],
                                ["Mean-Variance (Volatility)", "CVaR", "Mean-VaR"],
                                ["blue", "green", "red"]):
        risks = [x for x,_,_ in pts]
        returns = [y for _,y,_ in pts]
        splits = [w.values for _,_,w in pts]
        hover_texts = [f"Split: {w.index[0]}={w.values[0]:.2f}, {w.index[1]}={w.values[1]:.2f}<br>Risk: {risk:.4f}<br>Return: {ret:.4f}" for (risk, ret, w) in pts]
        traces.append(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+lines',
            name=name,
            marker=dict(color=color),
            text=hover_texts,
            hoverinfo='text'
        ))
    # Common points
    if common_pts:
        risks = [p["mv_risk"] for p in common_pts]
        returns = [p["ret"] for p in common_pts]
        splits = [p["w"].values for p in common_pts]
        hover_texts = [f"Split: {p['w'].index[0]}={p['w'].values[0]:.2f}, {p['w'].index[1]}={p['w'].values[1]:.2f}<br>Risk: {p['mv_risk']:.4f}<br>Return: {p['ret']:.4f}" for p in common_pts]
        traces.append(go.Scatter(
            x=risks,
            y=returns,
            mode='markers',
            name='Common weight on MV',
            marker=dict(symbol='x', size=12, color='orange'),
            text=hover_texts,
            hoverinfo='text'
        ))
    layout = go.Layout(
        title="Frontiers with common-weight portfolios marked",
        xaxis=dict(title="Risk (metric depends on curve)"),
        yaxis=dict(title="Expected Return"),
        hovermode='closest',
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig = go.Figure(data=traces, layout=layout)
    out_path = os.path.join("static", "portfolio_frontier.html")
    fig.write_html(out_path, include_plotlyjs='cdn')

    # Additional portfolio evolution plots can be added here using Plotly if needed

    # Return the path to the main graph for display
    return "/static/portfolio_frontier.html"
