import os
import plotly.graph_objs as go
from portfolio import portfolio
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent GUI popups during plot generation

def get_hovertemplate(stock_list, label=None):
    splits_str = ", ".join([f"{s}=%{{customdata[{i}]}}" for i, s in enumerate(stock_list)])
    display = "Risk: %{x:.4f}<br>Return: %{y:.4f}<br>Split:" + splits_str + "<extra></extra>"
    if label is not None:
        display = label + "<br>" + display
    return display

def html_plot(stocks, points, risk_name, additional_points=None):
    traces = []
    if points:
        risks, returns, weights = [], [], []
        for pt in points:
            risks.append(pt[0])
            returns.append(pt[1])
            weights.append(pt[2])

        hovertemplate = get_hovertemplate(stocks)
        traces.append(go.Scatter(
            x=risks, y=returns, mode='markers+lines', name=risk_name,
            marker=dict(color="red"), customdata=weights, hovertemplate=hovertemplate
        ))
    else:
        traces.append(go.Scatter(x=[0], y=[0], mode='text',
                                 text=["No efficient frontier points found."], showlegend=False))

    if additional_points is not None:
        colors = ['blue', 'green']
        for idx, pt in enumerate(additional_points):
            risk, ret, weight, label = pt[0], pt[1], pt[2], pt[3]
            color = colors[idx % len(colors)]
            hovertemplate = get_hovertemplate(stocks, label)
            traces.append(go.Scatter(
                x=[risk], y=[ret], name=label, mode='markers', marker=dict(color=color),
                customdata=[weight], hovertemplate=hovertemplate
            ))

    layout = go.Layout(title=f"{risk_name} frontier", xaxis=dict(title=risk_name),
                       yaxis=dict(title="Expected Return"), hovermode='closest',
                       legend=dict(x=0, y=1.1, orientation='h'),
                       margin=dict(l=40, r=40, t=60, b=40))
    fig = go.Figure(data=traces, layout=layout)
    return fig.to_html(include_plotlyjs=False, full_html=False)

def generate_frontier_graph(pfo: portfolio, pf_metrics):
    frontiers = []
    if pf_metrics['opt_variance']:
        add_vars = [
            (pf_metrics['opt_variance'], pf_metrics['return'], pf_metrics['opt_variance_weights'], "Minimum Variance portfolio"),
            (pf_metrics['user_variance'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")]
    else:
        add_vars = None
    frontiers.append(html_plot(pfo.df.columns, pfo.mv_frontier_pts, "Variance", add_vars))

    if pf_metrics['opt_var']:
        add_vars = [
            (pf_metrics['opt_var'], pf_metrics['return'], pf_metrics['opt_var_weights'], "Minimum VaR portfolio"),
            (pf_metrics['user_var'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")]
    else:
        add_vars = None
    frontiers.append(html_plot(pfo.df.columns, pfo.var_frontier_pts, "VaR", add_vars))

    if pf_metrics['opt_cvar']:
        add_vars = [
            (pf_metrics['opt_cvar'], pf_metrics['return'], pf_metrics['opt_cvar_weights'], "Minimum CVaR portfolio"),
            (pf_metrics['user_cvar'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")]
    else:
        add_vars = None
    frontiers.append(html_plot(pfo.df.columns, pfo.cvar_frontier_pts, "CVaR", add_vars))

    if pf_metrics['opt_sharpe']:
        add_vars = [
            (pf_metrics['opt_sharpe'], pf_metrics['return'], pf_metrics['opt_sharpe_weights'], "Minimum Sharpe portfolio"),
            (-pfo._portfolio_sharpe(pf_metrics['user_weights']), pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")]
    else:
        add_vars = None
    frontiers.append(html_plot(pfo.df.columns, pfo.sharpe_frontier_pts, "Sharpe Ratio", add_vars))

    if pf_metrics['opt_maxdd']:
        add_vars = [
            (pf_metrics['opt_maxdd'], pf_metrics['return'], pf_metrics['opt_maxdd_weights'], "Minimum MaxDD portfolio"),
            (pf_metrics['user_maxdd'], pf_metrics['return'], pf_metrics['user_weights'], "User portfolio")]
    else:
        add_vars = None
    frontiers.append(html_plot(pfo.df.columns, pfo.maxdd_frontier_pts, "Max Drawdown", add_vars))

    return frontiers

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
        'sharpe': '#9467bd',
        'maxdd': '#8c564b',
    }

    hover_tmpl = 'Date: %{x}<br>Cumulative: %{y:.4f}<extra></extra>'

    mapping = [("user", "User Portfolio"), ("variance", "Variance-opt"), ("var", "VaR-opt"), ("cvar", "CVaR-opt"), ("sharpe", "Sharpe-opt"), ("maxdd", "MaxDD-opt")]
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

def generate_backtrader_plots(pfo, test_df_for_dates, train_metrics, months_list=(1, 2, 3)):
    """
    Generates Plotly HTML backtest charts for each period and portfolio type.
    """
    if test_df_for_dates.empty:
        print("Warning: Test DataFrame empty, cannot generate backtrader plots.")
        return {}

    start_date = test_df_for_dates.index.min()
    end_date = test_df_for_dates.index.max()

    def safe_to_dict(w):
        if w is None: return None
        if hasattr(w, 'to_dict'): return w.to_dict()
        if isinstance(w, (list, tuple, pd.Series, np.ndarray)):
            return {k: v for k, v in zip(pfo.stocks, w)}
        if isinstance(w, dict): return w
        return None

    all_weights = {
        "user": safe_to_dict(train_metrics.get("user_weights")),
        "variance": safe_to_dict(train_metrics.get("opt_variance_weights")),
        "var": safe_to_dict(train_metrics.get("opt_var_weights")),
        "cvar": safe_to_dict(train_metrics.get("opt_cvar_weights")),
        "sharpe": safe_to_dict(train_metrics.get("opt_sharpe_weights")),
        "maxdd": safe_to_dict(train_metrics.get("opt_maxdd_weights")),
    }

    html_plots = {}

    for months in months_list:
        month_label = f"{months}M"
        html_plots[month_label] = {}
        backtest_start = max(end_date - pd.DateOffset(months=months), start_date)

        if backtest_start >= end_date:
            continue

        # Collect series for all portfolios
        series_dict = {}
        for key, weights in all_weights.items():
            if not weights:
                continue
            try:
                df = pfo.run_backtest_backtrader(weights, backtest_start, end_date)
                if df is not None and not df.empty:
                    series = df["value"] / df["value"].iloc[0]
                    series_dict[key] = series
            except Exception as e:
                print(f"Error in backtest for {key}: {e}")
                import traceback; traceback.print_exc()

        # Convert to HTML plot
        if series_dict:
            html = backtest_plots_from_series(series_dict, title_suffix=month_label)
            html_plots[month_label] = html
        else:
            html_plots[month_label] = None

    return html_plots
