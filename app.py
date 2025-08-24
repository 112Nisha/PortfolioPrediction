from flask import Flask, render_template, request
from predict import compute_portfolio_stats
from graph import generate_frontier_graph

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stocks = request.form.getlist('stock')
        weights = request.form.getlist('weight')
        if len(stocks) != 2 or len(weights) != 2:
            error = "Please select two stocks and enter their weights."
            return render_template('index.html', error=error)
        stats = compute_portfolio_stats(stocks, weights)
        graph_html_path = generate_frontier_graph(stocks, weights)  # This function saves the HTML file
        graph_html = None
        try:
            with open(graph_html_path, "r") as f:
                graph_html = f.read()
        except Exception:
            graph_html = None
        return render_template('index.html', stats=stats, graph_html=graph_html, stocks=stocks, weights=weights)
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
