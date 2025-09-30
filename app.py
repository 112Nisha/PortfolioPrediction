import os
from flask import Flask, render_template, request
import csv
from datetime import datetime

app = Flask(__name__)
data_directory = './data'  # Replace with the desired path


def format_weights_mv(stats):
    """Normalize/format weight-containing entries in stats for display.

    This helper is module-level so it can be used in both POST and tests.
    """
    if not isinstance(stats, dict):
        return stats
    for p in ["opt_variance_weights", "opt_var_weights", "opt_cvar_weights"]:
        if stats.get(p) is None:
            stats[p] = "Could not optimise"
        else:
            w = stats[p]
            try:
                if isinstance(w, dict):
                    stats[p] = ", ".join(f"{k}: {v:.4f}" for k, v in w.items())
                elif hasattr(w, 'items'):
                    stats[p] = ", ".join(f"{k}: {v:.4f}" for k, v in w.items())
                else:
                    stats[p] = str(w)
            except Exception:
                stats[p] = str(w)

    # create desired table
    table = {
        "return": stats["return"],
        "table": {"Variance": {"User Portfolio": stats["user_variance"], "Optimised Portfolio": stats["opt_variance"]},
        "VaR": {"User Portfolio": stats["user_var"], "Optimised Portfolio": stats["opt_var"]},
        "CVaR": {"User Portfolio": stats["user_cvar"], "Optimised Portfolio": stats["opt_cvar"]},},
    }

    return table


def format_weights_riskm(stats):
    """Normalize/format weight-containing entries in stats for display.

    This helper is module-level so it can be used in both POST and tests.
    """
    if not isinstance(stats, dict):
        return stats
    for p in ["opt_max_return_weights"]:
        if stats.get(p) is None:
            stats[p] = "Could not optimise"
        else:
            w = stats[p]
            try:
                if isinstance(w, dict):
                    stats[p] = ", ".join(f"{k}: {v:.4f}" for k, v in w.items())
                elif hasattr(w, 'items'):
                    stats[p] = ", ".join(f"{k}: {v:.4f}" for k, v in w.items())
                else:
                    stats[p] = str(w)
            except Exception:
                stats[p] = str(w)

    # create desired table
    table = {
        "return": stats["return"],
        "table": {"Variance": {"Optimised Portfolio": stats["user_variance"]},
        "VaR": {"Optimised Portfolio": stats["user_var"]},
        "CVaR": {"Optimised Portfolio": stats["user_cvar"]},}
    }

    return table


@app.route('/', methods=['GET'])
def index():
    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    # GET request
    # compute available months from CSVs without importing portfolio
    total_months = None
    try:
        min_date = None
        max_date = None
        for fname in os.listdir(data_directory):
            fpath = os.path.join(data_directory, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, 'r', newline='') as fh:
                    reader = csv.DictReader(fh)
                    if 'Date' not in reader.fieldnames:
                        continue
                    for row in reader:
                        dstr = row.get('Date')
                        if not dstr:
                            continue
                        try:
                            d = datetime.fromisoformat(dstr)
                        except Exception:
                            # try common format
                            try:
                                d = datetime.strptime(dstr, '%Y-%m-%d')
                            except Exception:
                                continue
                        if min_date is None or d < min_date:
                            min_date = d
                        if max_date is None or d > max_date:
                            max_date = d
            except Exception:
                continue
        if min_date is not None and max_date is not None:
            total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
    except Exception:
        total_months = None

    return render_template('index.html', stock_options=stocks_list, total_months=total_months)


@app.route('/mean_variance', methods=['POST'])
def mean_variance():
    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    if request.method == 'POST':
        # Collect form data
        stocks = request.form.getlist('stock')
        weights = request.form.getlist('weight')

        # optional train/test months from the user
        try:
            train_months = int(request.form.get('train_months', 36))
        except Exception:
            train_months = 36
        try:
            test_months = int(request.form.get('test_months', 3))
        except Exception:
            test_months = 3

        # basic form validation
        if len(stocks) != len(set(stocks)):
            error = "Your portfolio stocks must be unique."
            return render_template('index.html', error=error, stocks=stocks, weights=weights, stock_options=stocks_list)

        try:
            weight_vals = [float(w) for w in weights]
        except Exception:
            error = "Invalid weight values."
            return render_template('index.html', error=error, stocks=stocks, weights=weights, stock_options=stocks_list)

        if abs(sum(weight_vals) - 100.0) > 1e-6:
            error = "Your portfolio weights must sum to 100%."
            return render_template('index.html', error=error, stocks=stocks, weights=weights, stock_options=stocks_list)

        # Perform the heavy work inside try so we can show a clean error on failure
        # try:
        # lazy imports to avoid failing at module import time if heavy deps are missing
        from portfolio import portfolio
        from graph import generate_frontier_graph, generate_backtests_from_portfolio

        pfo = portfolio(stocks, weights)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(train_months, test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers()
        train_metrics = pfo.portfolio_metrics() # with train data
        test_metrics = pfo.portfolio_metrics(test_df) # with test_data
        graph_htmls = generate_frontier_graph(pfo, train_metrics)  # frontier graphs

        # create backtests for 1,2,3 months using full history and current opt weights
        backtest_htmls = generate_backtests_from_portfolio(pfo, test_df, months_list=(1, 2, 3))

        train_stats = format_weights_mv(train_metrics)
        test_stats = format_weights_mv(test_metrics)
        stats = {
            "train": train_stats,
            "test": test_stats,
            "weights": {
                "Variance-Optimised Weights": train_metrics["opt_variance_weights"],
                "VaR-Optimised Weights": train_metrics["opt_var_weights"],
                "CVaR-Optimised Weights": train_metrics["opt_cvar_weights"]
            }
        }

        return render_template('index.html', stats=stats, graph_htmls=graph_htmls, backtest_htmls=backtest_htmls, used_train=used_train, used_test=used_test, total_months=total_months, train_months=train_months, test_months=test_months, stocks=stocks, weights=weights, stock_options=stocks_list)
        # except Exception as e:
        #     # If heavy deps are missing or an error happens, show a friendly error and the GET view data
        #     return render_template('index.html', error=str(e), stock_options=stocks_list)

    return render_template('index.html', stock_options=stocks_list, total_months=total_months)

@app.route('/risk_metric', methods=['POST'])
def risk_metric():
    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    if request.method == 'POST':
        # Collect form data
        stocks = request.form.getlist('stock')
        risk_type = request.form.get("risk_metric")
        risk_value = request.form.get("risk_value")

        # optional train/test months from the user
        try:
            train_months = int(request.form.get('train_months', 36))
        except Exception:
            train_months = 36
        try:
            test_months = int(request.form.get('test_months', 3))
        except Exception:
            test_months = 3

        # basic form validation
        if len(stocks) != len(set(stocks)):
            error = "Your portfolio stocks must be unique."
            return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value)

        try:
            risk_value = float(risk_value)
        except (TypeError, ValueError):
            error = "Invalid risk value"
            return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value)
        print("target risk", risk_value)


        # Perform the heavy work inside try so we can show a clean error on failure
        # try:
        # lazy imports to avoid failing at module import time if heavy deps are missing
        from portfolio import portfolio
        from graph import generate_frontier_graph, backtest_plot, generate_backtests_from_portfolio

        pfo = portfolio(stocks)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(train_months, test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers(user_weights=False)

        # optimise target risk and set as user weights
        opt_w = pfo.optimize_max_return_for_volatility(risk_value)
        print(opt_w)
        if opt_w is None:
            print("error:", pfo.opt_max_return)
            error = f"Could not optimise on target risk {risk_value}: {pfo.opt_max_return}"
            return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value)
        pfo.user_weights = opt_w[2]

        train_metrics = pfo.portfolio_metrics() # with train data
        test_metrics = pfo.portfolio_metrics(test_df) # with test_data
        graph_htmls = generate_frontier_graph(pfo, train_metrics)  # frontier graphs

        # create backtests for 1,2,3 months using full history and current opt weights
        backtest_htmls = generate_backtests_from_portfolio(pfo, test_df, months_list=(1, 2, 3))

        train_stats = format_weights_riskm(train_metrics)
        test_stats = format_weights_riskm(test_metrics)

        stats = {"train": train_stats, "test": test_stats, "weights": {"Optimised return": train_metrics["opt_max_return_weights"]}}


        return render_template('index.html', stats=stats, graph_htmls=graph_htmls, backtest_htmls=backtest_htmls, used_train=used_train, used_test=used_test, total_months=total_months, train_months=train_months, test_months=test_months, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value)
        # except Exception as e:
        #     # If heavy deps are missing or an error happens, show a friendly error and the GET view data
        #     return render_template('index.html', error=str(e), stock_options=stocks_list)

    return render_template('index.html', stock_options=stocks_list, total_months=total_months)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
