import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize


def generate_frontier_graph(stocks, weights):
    """Generate the frontier HTML and return its absolute path.

    This function expects `stocks` to be a list of two stock symbols
    (matching files in the `data/` directory). It writes an interactive
    Plotly HTML file to `static/portfolio_frontier.html` and returns the
    absolute filesystem path so callers (like `app.py`) can open it.
    """
    # prepare output path as absolute so callers can open it reliably
    out_filename = "portfolio_frontier.html"
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", out_filename))

    # Load & prep for any two stocks
    dfs = []
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    for stock in stocks:
        for fname in os.listdir(data_dir):
            if fname.lower().startswith(stock.lower()):
                df = pd.read_csv(os.path.join(data_dir, fname), parse_dates=["Date"])
                dfs.append(df[["Date", "Close"]].rename(columns={"Close": stock}))
                break

    if len(dfs) != 2:
        # Not enough valid stocks selected - create a fallback HTML
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["No data available for selected stocks."], showlegend=False))
        fig.update_layout(title="Frontiers with common-weight portfolios marked")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.write_html(out_path, include_plotlyjs='cdn')
        return out_path

    df = dfs[0].merge(dfs[1], on="Date").set_index("Date")
    if df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["No data after merging selected stocks."], showlegend=False))
        fig.update_layout(title="Frontiers with common-weight portfolios marked")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.write_html(out_path, include_plotlyjs='cdn')
        return out_path

    rets = df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    def portfolio_var(w, returns_df, alpha=0.05):
        pr = returns_df @ w
        return -np.percentile(pr, alpha * 100)

    def portfolio_return(w, mu):
        return w @ mu

    def as_series(w, index):
        return pd.Series(w, index=index).sort_index()

    grid = np.linspace(mu.min() + 1e-6, mu.max() - 1e-6, 100)
    mv_pts, cvar_pts, var_pts = [], [], []
    num_assets = len(mu)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1.0 / num_assets])

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

    # Prepare Plotly traces for interactive graph
    traces = []
    for pts, name, color in zip([mv_pts, cvar_pts, var_pts],
                                ["Mean-Variance (Volatility)", "CVaR", "Mean-VaR"],
                                ["blue", "green", "red"]):
        if pts:
            risks = [x for x, _, _ in pts]
            returns = [y for _, y, _ in pts]
            splits = [w.values for _, _, w in pts]
            customdata = splits
            hovertemplate = (
                f"Risk: %{{x:.4f}}<br>Return: %{{y:.4f}}<br>"
                f"Split: {df.columns[0]}=%{{customdata[0]:.2f}}, {df.columns[1]}=%{{customdata[1]:.2f}}<extra></extra>"
            )
            traces.append(go.Scatter(
                x=risks,
                y=returns,
                mode='markers+lines',
                name=name,
                marker=dict(color=color),
                customdata=customdata,
                hovertemplate=hovertemplate
            ))

    if not traces:
        traces.append(go.Scatter(x=[0], y=[0], mode='text', text=["No efficient frontier points found."], showlegend=False))

    layout = go.Layout(
        title="Frontiers with common-weight portfolios marked",
        xaxis=dict(title="Risk (metric depends on curve)"),
        yaxis=dict(title="Expected Return"),
        hovermode='closest',
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig = go.Figure(data=traces, layout=layout)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs='cdn')

    return out_path
