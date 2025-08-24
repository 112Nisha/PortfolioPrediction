from flask import Flask, render_template, request
from predict import compute_portfolio_stats
from graph import generate_frontier_graph

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stocks = request.form.getlist('stock')
        weights = request.form.getlist('weight')
        # Only allow two stocks
        if len(stocks) != 2 or len(weights) != 2:
            error = "Please select two stocks and enter their weights."
            return render_template('index.html', error=error)
        stats = compute_portfolio_stats(stocks, weights)
        generate_frontier_graph(stocks, weights)  # This function saves the image
        graph_url = '/static/portfolio_frontier.png'
        return render_template('index.html', stats=stats, graph_url=graph_url, stocks=stocks, weights=weights)
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
