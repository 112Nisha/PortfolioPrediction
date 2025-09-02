import os
import plotly.graph_objs as go
from portfolio import portfolio

# TODO: ADD CODE TO PLOT THE POINT GIVEN BY THE USER ALSO (WITH ARROW?)
def html_plot(stocks, points, risk, input_pf=None):
    # Prepare Plotly traces for interactive graph
    traces = []
    for pts, name, color in zip([points],
                                [risk],
                                ["red"]):
        if pts:
            risks = [x for x, _, _ in pts]
            returns = [y for _, y, _ in pts]
            splits = [w.values for _, _, w in pts]
            customdata = splits
            # TODO: CHANGE THIS SO THAT IT CAN BE MORE THAN 2 STOCKS
            hovertemplate = (
                f"Risk: %{{x:.4f}}<br>Return: %{{y:.4f}}<br>"
                f"Split: {stocks[0]}=%{{customdata[0]:.2f}}, {stocks[1]}=%{{customdata[1]:.2f}}<extra></extra>"
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
    elif input_pf:
        traces.append(go.Scatter(x=[input_pf[0]], y=[input_pf[1]], mode='markers+lines', text=["No efficient frontier points found."], showlegend=False))

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
    """Generate the frontier HTML and return its absolute path.

    This function expects `stocks` to be a list of two stock symbols
    (matching files in the `data/` directory). It writes an interactive
    Plotly HTML file to `static/portfolio_frontier.html` and returns the
    absolute filesystem path so callers (like `app.py`) can open it.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='text', text=["No data available for selected stocks."], showlegend=False))
    fig.update_layout(title="Frontiers with common-weight portfolios marked")
    html_str = fig.to_html(include_plotlyjs=False, full_html=False)
    # return [html_str, html_str, html_str]
    frontiers = []

    # variance
    frontiers.append(html_plot(pfo.df.columns, pfo.mv_frontier_pts, "Variance"))

    # value at risk
    frontiers.append(html_plot(pfo.df.columns, pfo.var_frontier_pts, "VaR"))


    # conditional value at risk
    frontiers.append(html_plot(pfo.df.columns, pfo.cvar_frontier_pts, "CVaR"))

    return frontiers