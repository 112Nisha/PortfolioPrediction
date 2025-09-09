import os
from flask import Flask, render_template, request
from graph import generate_frontier_graph, backtest_plot
from portfolio import portfolio

app = Flask(__name__)
data_directory = './data'  # Replace with the desired path

@app.route('/', methods=['GET', 'POST'])
def index():
    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    if request.method == 'POST':
        stocks = request.form.getlist('stock')
        weights = request.form.getlist('weight')

        if len(stocks) != len(set(stocks)):
            error = "Your portfolio stocks must be unique."
            return render_template('index.html', error=error, stocks=stocks, weights=weights)

        if sum([float(w) for w in weights]) != 100:
            error = "Your portfolio weights must sum to 100%."
            return render_template('index.html', error=error, stocks=stocks, weights=weights)

        pfo = portfolio(stocks, weights)
        pfo.calculate_frontiers()
        pfo.portfolio_metrics(pfo.user_weights)
        graph_htmls = generate_frontier_graph(pfo)  # frontier graphs
        backtest_html = backtest_plot(pfo, pfo.user_weights, '3M')

        def format_weights(stats):
            for p in ["opt_variance_weights", "opt_var_weights", "opt_cvar_weights"]:
                if stats[p] is None:
                    stats[p] = "Could not optimise"
                else:
                    stats[p] = ", ".join(f"{idx}: {val}" for idx, val in stats[p].items())
            return stats
        stats = format_weights(pfo.pf_metrics)

        return render_template('index.html', stats=stats, graph_htmls=graph_htmls, backtest_html=backtest_html, stocks=stocks, weights=weights, stock_options=stocks_list)

    # GET request
    return render_template('index.html', stock_options=stocks_list)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
