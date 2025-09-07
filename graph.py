import os
import plotly.graph_objs as go
from portfolio import portfolio

def html_plot(stocks, points, risk, additional_points=None):
    traces = []
    
    # Helper to generate hovertemplate dynamically for N stocks
    def get_hovertemplate(stock_list):
        splits_str = ", ".join([f"{s}=%{{customdata[{i}]:.2f}}" for i, s in enumerate(stock_list)])
        return (
            f"Risk: %{{x:.4f}}<br>Return: %{{y:.4f}}<br>"
            f"Split: {splits_str}<extra></extra>"
        )
    
    # Add main frontier trace
    if points:
        risks = [x for x, _, _ in points]
        returns = [y for _, y, _ in points]
        splits = [w for _, _, w in points]
        hovertemplate = get_hovertemplate(stocks)
        traces.append(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+lines',
            name=risk,
            marker=dict(color="red"),
            customdata=splits,
            hovertemplate=hovertemplate
        ))
    else:
        traces.append(go.Scatter(
            x=[0], y=[0], mode='text',
            text=["No efficient frontier points found."],
            showlegend=False
        ))

    # Add additional points (if any)
    if additional_points:
        for ap in additional_points:
            pts = ap.get("points")
            label = ap.get("label", "Additional")
            color = ap.get("color", "blue")

            if pts:
                risks = [x for x, _, _ in pts]
                returns = [y for _, y, _ in pts]
                # splits = [w for _, _, w in pts]
                hovertemplate = get_hovertemplate(stocks)
                traces.append(go.Scatter(
                    x=risks,
                    y=returns,
                    mode='markers',
                    name=label,
                    marker=dict(color=color, size=10, symbol="diamond"),
                    customdata=None,
                    hovertemplate=hovertemplate
                ))

    layout = go.Layout(
        title=f"{risk} frontier",
        xaxis=dict(title=risk),
        yaxis=dict(title="Expected Return"),
        hovermode='closest',
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig = go.Figure(data=traces, layout=layout)
    html_str = fig.to_html(include_plotlyjs=False, full_html=False)
    return html_str


def generate_frontier_graph(pfo: portfolio):
    frontiers = []

    additional_variables = None
    if pfo.pf_variance_metrics["opt_variance"]:
        additional_variables = [
            {
                "points": [(pfo.pf_variance_metrics["user_variance"], pfo.pf_return, pfo.user_weights)],
                "label": "User Portfolio",
                "color": "blue",
            },
            {
                "points": [(pfo.pf_variance_metrics["opt_variance"], pfo.pf_return, pfo.pf_variance_metrics["opt_variance_weights"])],
                "label": "Optimal Split",
                "color": "green",
            }
        ]
    frontiers.append(html_plot(pfo.df.columns, pfo.mv_frontier_pts, "Variance", additional_variables))

    additional_variables = None
    if pfo.pf_var_metrics["opt_var"]:
        additional_variables = [
            {
                "points": [(pfo.pf_var_metrics["user_var"], pfo.pf_return, pfo.user_weights)],
                "label": "User Portfolio",
                "color": "blue",
            },
            {
                "points": [(pfo.pf_var_metrics["opt_var"], pfo.pf_return, pfo.pf_variance_metrics["opt_variance_weights"])],
                "label": "Optimal Split",
                "color": "green",
            }
        ]
    frontiers.append(html_plot(pfo.df.columns, pfo.var_frontier_pts, "VaR", additional_variables))

    additional_variables = None
    if pfo.pf_cvar_metrics["opt_cvar"]:
        additional_variables = [
            {
                "points": [(pfo.pf_cvar_metrics["user_cvar"], pfo.pf_return, pfo.user_weights)],
                "label": "User Portfolio",
                "color": "blue",
            },
            {
                "points": [(pfo.pf_cvar_metrics["opt_cvar"], pfo.pf_return, pfo.pf_variance_metrics["opt_variance_weights"])],
                "label": "Optimal Split",
                "color": "green",
            }
        ]
    frontiers.append(html_plot(pfo.df.columns, pfo.cvar_frontier_pts, "CVaR", additional_variables))

    return frontiers

def format_pf_stats(pfo: portfolio):
    user_metrics = {
        "mean": f"{pfo.pf_return}",
        "variance": f"{pfo.pf_variance_metrics['user_variance']}",
        "VaR": f"{pfo.pf_var_metrics['user_var']}",
        "CVar": f"{pfo.pf_cvar_metrics['user_cvar']}",
    }

    optimized_metrics = {
        "mean": f"{pfo.pf_return}",
        "variance": f"{pfo.pf_variance_metrics['opt_variance']}",
        "VaR": f"{pfo.pf_var_metrics['opt_var']}",
        "CVar": f"{pfo.pf_cvar_metrics['opt_cvar']}",
        "weights_variance": f"{pfo.pf_variance_metrics['opt_variance_weights']}",
        "weights_var": f"{pfo.pf_var_metrics['opt_var_weights']}",
        "weights_cvar": f"{pfo.pf_cvar_metrics['opt_cvar_weights']}",
    }

    return [user_metrics, optimized_metrics]


def backtest_plot(pfo: portfolio, weights, period='3M'):
    """Create a single backtest plot with four lines:
    - User portfolio
    - Variance-optimal portfolio (same return)
    - VaR-optimal portfolio (same return)
    - CVaR-optimal portfolio (same return)

    Returns HTML string (plotly) or None if not enough data.
    """
    import numpy as np
    import pandas as pd

    if pfo.df.empty:
        return None

    # ensure metrics computed
    try:
        pfo.portfolio_metrics(weights)
    except Exception:
        pass

    def to_array(w):
        if w is None:
            return None
        if isinstance(w, np.ndarray):
            return w
        if isinstance(w, pd.Series):
            return w.reindex(pfo.df.columns).to_numpy()
        if isinstance(w, dict):
            return pd.Series(w).reindex(pfo.df.columns).to_numpy()
        try:
            arr = np.array(w)
            if arr.shape[0] == len(pfo.df.columns):
                return arr
        except Exception:
            pass
        return None

    # rets = pfo.df.pct_change().dropna()
    end_date = pfo.df.index.max()
    try:
        months = int(str(period).upper().replace('M', ''))
        start_date = end_date - pd.DateOffset(months=months)
    except Exception:
        start_date = end_date - pd.DateOffset(months=3)

    rets_sub = pfo.rets[pfo.rets.index >= start_date]
    if rets_sub.empty:
        return None

    user_w = to_array(weights)
    var_w = to_array(getattr(pfo, 'pf_variance_metrics', {}).get('opt_variance_weights'))

    # VaR-opt weights may not have been computed earlier; try to retrieve or recompute
    varopt_raw = getattr(pfo, 'pf_var_metrics', {}).get('opt_var_weights')
    if varopt_raw is None:
        try:
            # attempt to recompute using the portfolio return
            if hasattr(pfo, 'pf_return'):
                res = pfo._optimize_var_for_return(pfo.pf_return)
                varopt_raw = res[2] if res else None
        except Exception:
            varopt_raw = None
    var_w_v = to_array(varopt_raw)
    if var_w_v is None:
        print("VaR-opt weights not available for backtest (optimizer failed or not computed).")

    cvar_w = to_array(getattr(pfo, 'pf_cvar_metrics', {}).get('opt_cvar_weights'))

    traces = []
    # define accessible color palette
    colors = {
        'user': '#1f77b4',      # blue
        'variance': '#ff7f0e',  # orange
        'var': '#2ca02c',       # green
        'cvar': '#d62728',      # red
    }

    # uniform hovertemplate for cumulative returns (rounded)
    hover_tmpl = 'Date: %{x}<br>Cumulative: %{y:.4f}<extra></extra>'

    # user
    if user_w is not None:
        pr_user = rets_sub.dot(user_w)
        cum_user = (1 + pr_user).cumprod()
        traces.append(go.Scatter(x=cum_user.index, y=cum_user.values, mode='lines+markers', name='User Portfolio', line=dict(width=3, color=colors['user']), marker=dict(size=4)))

    # variance-opt
    if var_w is not None:
        pr_var = rets_sub.dot(var_w)
        cum_var = (1 + pr_var).cumprod()
        traces.append(go.Scatter(x=cum_var.index, y=cum_var.values, mode='lines+markers', name='Variance-opt', line=dict(width=2, dash='dash', color=colors['variance']), marker=dict(size=3), hovertemplate=hover_tmpl))

    # VaR-opt
    if var_w_v is not None:
        pr_v = rets_sub.dot(var_w_v)
        cum_v = (1 + pr_v).cumprod()
        traces.append(go.Scatter(x=cum_v.index, y=cum_v.values, mode='lines+markers', name='VaR-opt', line=dict(width=2, dash='dot', color=colors['var']), marker=dict(size=3), hovertemplate=hover_tmpl))

    # CVaR-opt
    if cvar_w is not None:
        pr_c = rets_sub.dot(cvar_w)
        cum_c = (1 + pr_c).cumprod()
        traces.append(go.Scatter(x=cum_c.index, y=cum_c.values, mode='lines+markers', name='CVaR-opt', line=dict(width=2, dash='dashdot', color=colors['cvar']), marker=dict(size=3), hovertemplate=hover_tmpl))

    if not traces:
        traces.append(go.Scatter(x=[rets_sub.index.min(), rets_sub.index.max()], y=[1, 1], mode='lines', name='No data'))

    layout = go.Layout(title=f'Backtest over last {period}', xaxis=dict(title='Date'), yaxis=dict(title='Cumulative Return (base 1)'), legend=dict(orientation='h', x=0, y=1.1), margin=dict(l=40, r=40, t=60, b=40))
    fig = go.Figure(data=traces, layout=layout)
    return fig.to_html(include_plotlyjs=False, full_html=False)
