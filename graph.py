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
