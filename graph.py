import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
from utils import Portfolio, OptimizationResultsContainer


def get_cmap(n, name='tab10'):
    cmap = plt.cm.get_cmap(name)
    colors = [cmap(i % cmap.N) for i in range(n)]
    return colors


def get_hovertemplate(stock_list, label=None):
    splits_str = ", ".join([f"{s}=%{{customdata[{i}]:.4f}}" for i, s in enumerate(stock_list)])
    display = "Risk: %{x:.4f}<br>Return: %{y:.4f}<br>Split:" + splits_str + "<extra></extra>"
    if label is not None:
        display = label + "<br>" + display
    return display


def html_plot(stocks, points_list, risk_name, additional_points=None):
    traces = []
    if points_list and any(points_list):
        colors = ['red', 'brown']
        for i, points in enumerate(points_list):
            risks, returns, weights = [], [], []
            for pt in points:
                risks.append(pt[0])
                returns.append(pt[1])
                weights.append(pt[2])
            label = "CML" if i == 1 else risk_name
            weights_list = [[w[s] for s in stocks] for w in weights]
            hovertemplate = get_hovertemplate(stocks)
            traces.append(go.Scatter(
                x=risks, y=returns, mode='markers+lines', name=label,
                marker=dict(color=colors[i]), customdata=weights_list, hovertemplate=hovertemplate
            ))
    else:
        traces.append(go.Scatter(x=[0], y=[0], mode='text',
                                 text=["No efficient frontier points found."], showlegend=False))

    if additional_points is not None:
        colors = ['blue', 'green', 'yellow', 'purple', 'cyan']
        for idx, pt in enumerate(additional_points):
            risk, ret, weight, label = pt[0], pt[1], pt[2], pt[3]
            weight = [weight[s] for s in stocks]
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

def generate_frontier_graph(pfo, user_pf: Portfolio = None, opt_for_risk: OptimizationResultsContainer = None, opt_for_return: OptimizationResultsContainer = None):
    frontiers = []

    # Helper function to reduce repetition
    def add_frontier(metric_name: str, frontier_points, risk_label: str):
        if opt_for_risk: risk_pf = getattr(opt_for_risk, metric_name)
        if opt_for_return: ret_pf = getattr(opt_for_return, metric_name)
        add_vars = [] # (x-axis risk, y-axis return, weights, label)

        if user_pf:
            add_vars.append(
                (getattr(user_pf, metric_name), user_pf.return_, user_pf.weights, "User Portfolio"),
            )

        if opt_for_risk and risk_pf.weights is not None:
            add_vars.append(
            (getattr(risk_pf, metric_name), risk_pf.return_, risk_pf.weights, f"Minimized {risk_label} Portfolio"),
            )

        if opt_for_return and ret_pf.weights is not None:
            add_vars.append(
                (getattr(ret_pf, metric_name), ret_pf.return_, ret_pf.weights, "Return-Optimized Portfolio")
            )

        main_frontier = [
            (getattr(p, metric_name), p.return_, p.weights) for p in frontier_points if isinstance(p, Portfolio)
        ]
        tangent_frontier = [
            (getattr(p.tangent, metric_name), p.tangent.return_, p.tangent.weights)
            for p in frontier_points
            if isinstance(p, Portfolio) and p.tangent is not None and p.tangent.weights is not None
        ]

        frontier_lists = [main_frontier]
        if tangent_frontier:
            frontier_lists.append(tangent_frontier)

        frontiers.append(html_plot(pfo.stocks, frontier_lists, risk_label, add_vars))

    # Call for each risk metric
    add_frontier("variance", pfo.mv_frontier_pts, "Variance")
    add_frontier("var", pfo.var_frontier_pts, "VaR")
    add_frontier("cvar", pfo.cvar_frontier_pts, "CVaR")
    add_frontier("sharpe", pfo.sharpe_frontier_pts, "Sharpe Ratio")
    add_frontier("maxdd", pfo.maxdd_frontier_pts, "Max Drawdown")

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
    dash_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    mapping = [
        ("user", "User Portfolio"), 
        ("variance_risk", "Variance-opt"), 
        ("var_risk", "VaR-opt"), 
        ("cvar_risk", "CVaR-opt"), 
        ("sharpe_risk", "Sharpe-opt"), 
        ("maxdd_risk", "MaxDD-opt"),
        ("variance_return", "Variance-return-opt"), 
        ("var_return", "VaR-return-opt"), 
        ("cvar_return", "CVaR-return-opt"), 
        ("sharpe_return", "Sharpe-return-opt"), 
        ("maxdd_return", "MaxDD-return-opt"),
    ]

    i = 0
    for key, label in mapping:
        s = series_dict.get(key)
        if s is not None:
            current_dash_style = dash_styles[i % len(dash_styles)]
            traces.append(go.Scatter(x=s.index, y=s.values, mode='lines+markers', name=label, line=dict(width=2, color=colors.get(key), dash=current_dash_style), marker=dict(size=3), hovertemplate=hover_tmpl))
            i += 1

    if not traces:
        traces.append(go.Scatter(x=[0, 1], y=[1, 1], mode='lines', name='No data'))

    title = 'Backtest'
    if title_suffix:
        title = f"{title} ({title_suffix})"

    layout = go.Layout(title=title, xaxis=dict(title='Date'), yaxis=dict(title='Cumulative Return (base 1)'), legend=dict(orientation='h', x=0, y=1.1), margin=dict(l=40, r=40, t=60, b=40))
    fig = go.Figure(data=traces, layout=layout)
    return fig.to_html(include_plotlyjs=False, full_html=False)

def generate_backtrader_plots(
    pfo,
    test_df_for_dates,
    opt_for_risk: OptimizationResultsContainer,
    opt_for_return: OptimizationResultsContainer,
    user_pf: Portfolio=None,
    months_list=(1, 2, 3)
):
    """
    Generates Plotly HTML backtest charts for each period and portfolio type.

    Arguments:
        pfo: Portfolio optimizer/backtester object with run_backtest_backtrader(weights, start, end)
        test_df_for_dates: DataFrame used to determine start/end of test period
        user_pf: Portfolio (user's portfolio)
        opt_for_risk: OptimizationResultsContainer (risk-optimized portfolios)
        opt_for_return: OptimizationResultsContainer (return-optimized portfolios)
    """
    if test_df_for_dates.empty:
        print("Warning: Test DataFrame empty, cannot generate backtrader plots.")
        return {}

    start_date = test_df_for_dates.index.min()
    end_date = test_df_for_dates.index.max()

    all_weights = {}

    # Build dictionary of all weight sets to backtest
    if isinstance(user_pf, Portfolio):
        all_weights["user"] = user_pf.weights

    for key, pf in opt_for_risk.items():
        if isinstance(pf, Portfolio):
            all_weights[f"{key}_risk"] = pf.weights

    for key, pf in opt_for_return.items():
        if isinstance(pf, Portfolio):
            all_weights[f"{key}_return"] = pf.weights

    html_plots = {}

    for months in months_list:
        month_label = f"{months}M"
        html_plots[month_label] = {}
        backtest_start = max(end_date - pd.DateOffset(months=months), start_date)

        if backtest_start >= end_date:
            continue

        # Collect normalized value series for all portfolios
        series_dict = {}
        for key, weights in all_weights.items():
            if not weights:
                continue
            try:
                df = pfo.run_backtest_backtrader(weights, backtest_start, end_date)
                if df is not None and not df.empty and "value" in df.columns:
                    series = df["value"] / df["value"].iloc[0]
                    series_dict[key] = series
            except Exception as e:
                print(f"Error in backtest for {key}: {e}")
                import traceback; traceback.print_exc()

        # Convert collected time series into Plotly HTML
        if series_dict:
            html = backtest_plots_from_series(series_dict, title_suffix=month_label)
            html_plots[month_label] = html
        else:
            html_plots[month_label] = None

    return html_plots
