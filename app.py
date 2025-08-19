from flask import Flask, render_template, request
from predict import compute_portfolio_stats
from graph import generate_frontier_graph

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stocks = request.form.getlist('stock')
        weights = request.form.getlist('weight')
        # Placeholder: call prediction and graph logic
        stats = compute_portfolio_stats(stocks, weights)
        # Assume graph.py generates a file in static/ named portfolio_frontier.png
        generate_frontier_graph(stocks, weights)  # This function saves the image
        graph_url = '/static/portfolio_frontier.png'  # Consistent naming convention
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
