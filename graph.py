import os
import plotly.graph_objs as go
import plotly.colors as pc
from portfolio import portfolio
import numpy as np
import pandas as pd

# Helper to generate hovertemplate dynamically for N stocks
def get_hovertemplate(stock_list, label=None):
    splits_str = ", ".join([f"{s}=%{{customdata[{i}]}}" for i, s in enumerate(stock_list)])
    display = "Risk: %{x:.4f}<br>Return: %{y:.4f}<br>Split:" + splits_str + "<extra></extra>"
    if label is not None:
        display = label + "<br>" + display
    return (display)


def html_plot(stocks, points, risk_name, additional_points=None):
    traces = []

    # Add main frontier trace
    if points:
        risks, returns, weights = [], [], []
        for pt in points:
            risks.append(pt[0])
            returns.append(pt[1])
            weights.append(pt[2])

        hovertemplate = get_hovertemplate(stocks)
        traces.append(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+lines',
            name=risk_name,
            marker=dict(color="red"),
            customdata=weights,
            hovertemplate=hovertemplate
        ))
    else:
        traces.append(go.Scatter(
            x=[0], y=[0], mode='text',
            text=["No efficient frontier points found."],
            showlegend=False
        ))

    # Add additional points (if any)
    if additional_points is not None:
        # generate unique colors for each point
        colors = ['blue', 'green'] # replace this is logic to generate color if need arises

        for idx, pt in enumerate(additional_points):
            risk, ret, weight, label = pt[0], pt[1], pt[2], pt[3]
            color = colors[idx]

            hovertemplate = get_hovertemplate(stocks, label)
            traces.append(go.Scatter(
                x=[risk],
                y=[ret],
                name=label,
                mode='markers',
                marker=dict(color=color),
                customdata=[weight],
                hovertemplate=hovertemplate
            ))

    layout = go.Layout(
        title=f"{risk_name} frontier",
        xaxis=dict(title=risk_name),
        yaxis=dict(title="Expected Return"),
        hovermode='closest',
        legend=dict(x=0, y=1.1, orientation='h'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig = go.Figure(data=traces, layout=layout)
    html_str = fig.to_html(include_plotlyjs=False, full_html=False)
    return html_str


def generate_frontier_graph(pfo: portfolio, pf_metrics):
    frontiers = []

    if pf_metrics['opt_variance']:
        additional_variables = [
            (pf_metrics['opt_variance'], pf_metrics['return'], pf_metrics['opt_variance_weights'], "Minimum Variance portfolio"),
            (pf_metrics['user_variance'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")
        ]
    elif pf_metrics['opt_max_return']:
        additional_variables = [
            (pf_metrics['opt_max_return_variance'], pf_metrics['opt_max_return'], 
             pf_metrics['opt_max_return_weights'], "Maximum Return portfolio"),
        ]
    else: additional_variables = None
    # additional_variables = None
    frontiers.append(html_plot(pfo.df.columns, pfo.mv_frontier_pts, "Variance", additional_variables))

    if pf_metrics['opt_var']:
        additional_variables = [
            (pf_metrics['opt_var'], pf_metrics['return'], pf_metrics['opt_var_weights'], "Minimum VaR portfolio"),
            (pf_metrics['user_var'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")
        ]
    else: additional_variables = None
    # additional_variables = None

    frontiers.append(html_plot(pfo.df.columns, pfo.var_frontier_pts, "VaR", additional_variables))

    if pf_metrics['opt_cvar']:
        additional_variables = [
            (pf_metrics['opt_cvar'], pf_metrics['return'], pf_metrics['opt_cvar_weights'], "Minimum CVaR portfolio"),
            (pf_metrics['user_cvar'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")
        ]
    else: additional_variables = None
    # additional_variables = None

    frontiers.append(html_plot(pfo.df.columns, pfo.cvar_frontier_pts, "CVaR", additional_variables))

    # if pf_metrics['opt_max_return']:
    #     additional_variables = [
    #         (pf_metrics['opt_max_return_variance'], pf_metrics['opt_max_return'], 
    #          pf_metrics['opt_max_return_weights'], "Maximum Return portfolio"),
    #         # (pf_metrics['user_variance'], pf_metrics['return'], 
    #         #  pf_metrics['user_weights'], "User portfolio")
    #     ]
    # else: 
    #     additional_variables = None

    # frontiers.append(html_plot(pfo.df.columns, pfo.max_return_frontier_pts, 
    #                           "Variance (Max Return)", additional_variables))

    return frontiers


def backtest_plot(pfo: portfolio, weights, period='3M'):
    """Create a single backtest plot with four lines:
    - User portfolio
    - Variance-optimal portfolio (same return)
    - VaR-optimal portfolio (same return)
    - CVaR-optimal portfolio (same return)

    Returns HTML string (plotly) or None if not enough data.
    """

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
    pf_metrics = getattr(pfo, 'pf_metrics', {})
    variance_w = to_array(pf_metrics.get('opt_variance_weights'))
    var_w = to_array(pf_metrics.get('opt_var_weights'))
    cvar_w = to_array(pf_metrics.get('opt_cvar_weights'))

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
    if variance_w is not None:
        pr_var = rets_sub.dot(variance_w)
        cum_var = (1 + pr_var).cumprod()
        traces.append(go.Scatter(x=cum_var.index, y=cum_var.values, mode='lines+markers', name='Variance-opt', line=dict(width=2, dash='dash', color=colors['variance']), marker=dict(size=3), hovertemplate=hover_tmpl))

    # VaR-opt
    if var_w is not None:
        pr_v = rets_sub.dot(var_w)
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


def backtest_plots_from_series(series_dict, title_suffix=None):
    """Given a dict of series (keys: 'user','variance','var','cvar'),
    produce an HTML plot similar to backtest_plot. Returns HTML string.
    """
    if not series_dict:
        return None

    traces = []
    colors = {
        'user': '#1f77b4',
        'variance': '#ff7f0e',
        'var': '#2ca02c',
        'cvar': '#d62728',
    }

    hover_tmpl = 'Date: %{x}<br>Cumulative: %{y:.4f}<extra></extra>'

    mapping = [("user", "User Portfolio"), ("variance", "Variance-opt"), ("var", "VaR-opt"), ("cvar", "CVaR-opt")]
    for key, label in mapping:
        s = series_dict.get(key)
        if s is not None:
            traces.append(go.Scatter(x=s.index, y=s.values, mode='lines+markers', name=label, line=dict(width=2, color=colors.get(key)), marker=dict(size=3), hovertemplate=hover_tmpl))

    if not traces:
        traces.append(go.Scatter(x=[0, 1], y=[1, 1], mode='lines', name='No data'))

    title = 'Backtest'
    if title_suffix:
        title = f"{title} ({title_suffix})"

    layout = go.Layout(title=title, xaxis=dict(title='Date'), yaxis=dict(title='Cumulative Return (base 1)'), legend=dict(orientation='h', x=0, y=1.1), margin=dict(l=40, r=40, t=60, b=40))
    fig = go.Figure(data=traces, layout=layout)
    return fig.to_html(include_plotlyjs=False, full_html=False)


def generate_backtests_from_portfolio(pfo: portfolio, test_df, months_list=(1, 2, 3)):
    """Compute backtest series for requested months and return list of html strings.

    The function calls pfo.backtest_series for each months value and wraps them into
    HTML using backtest_plots_from_series.
    """
    htmls = []
    for m in months_list:
        s = pfo.backtest_series(test_df, months=m)
        html = backtest_plots_from_series(s, title_suffix=f"{m}M") if s is not None else None
        htmls.append(html)
    return htmls
