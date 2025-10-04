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

    return frontiers

def generate_backtrader_plots(pfo, test_df_for_dates, train_metrics, months_list=(1, 2, 3)):
    BACKTEST_PLOTS_DIR = os.path.join("static", "backtest_plots")
    os.makedirs(BACKTEST_PLOTS_DIR, exist_ok=True)

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
        "User Portfolio": safe_to_dict(train_metrics.get("user_weights")),
        "Variance-Optimised": safe_to_dict(train_metrics.get("opt_variance_weights")),
        "VaR-Optimised": safe_to_dict(train_metrics.get("opt_var_weights")),
        "CVaR-Optimised": safe_to_dict(train_metrics.get("opt_cvar_weights")),
        "Max Return for Target Risk": safe_to_dict(train_metrics.get("opt_max_return_weights")),
    }

    period_plots = {}

    for months in months_list:
        month_label = f"{months}M"
        period_plots[month_label] = {}
        backtest_start = max(end_date - pd.DateOffset(months=months), start_date)

        if backtest_start >= end_date:
            for label in all_weights:
                period_plots[month_label][label] = None
            continue

        for label, weights_dict in all_weights.items():
            if not weights_dict:
                period_plots[month_label][label] = None
                continue

            filename = f"{label.replace(' ', '_').lower()}_{months}m_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.png"
            plot_path = os.path.join(BACKTEST_PLOTS_DIR, filename)

            try:
                result_path = pfo.run_backtest_backtrader(
                    weights_dict, backtest_start, end_date,
                    plot_filepath=plot_path, title=f"{label} ({months}M)"
                )
                if result_path:
                    period_plots[month_label][label] = os.path.relpath(result_path, 'static')
                else:
                    period_plots[month_label][label] = None
            except Exception as e:
                print(f"Error generating {label} {months}M backtest: {e}")
                import traceback; traceback.print_exc()
                period_plots[month_label][label] = None

    return period_plots
