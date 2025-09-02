from flask import Flask, render_template, request
from predict import compute_portfolio_stats
from graph import generate_frontier_graph, format_pf_stats
from portfolio import portfolio

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stocks = request.form.getlist('stock')
        weights = request.form.getlist('weight')
        # if len(stocks) != 2 or len(weights) != 2:
        #     error = "Please select two stocks and enter their weights."
        #     return render_template('index.html', error=error)

        stats = compute_portfolio_stats(stocks, weights)
        pfo = portfolio(stocks, weights)
        pfo.calculate_frontiers()
        pfo.portfolio_metrics(pfo.user_weights)
        stats = format_pf_stats(pfo)

        graph_htmls = generate_frontier_graph(pfo)  # This function saves the HTML file
        return render_template('index.html', stats=stats, graph_htmls=graph_htmls, stocks=stocks, weights=weights)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
