import os
from flask import Flask, render_template, request, redirect, url_for
import csv
from datetime import datetime
from portfolio import portfolio, compute_test_metrics
from graph import generate_frontier_graph, generate_backtrader_plots
from utils import *

app = Flask(__name__)
data_directory = './data'  # Replace with the desired path


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

    if request.method != 'POST':
        return redirect(url_for('index'))

    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    # Collect form data
    stocks = request.form.getlist('stock')
    weight_percents = request.form.getlist('weight')

    mean_method = request.form.get("mean_method")
    cov_method = request.form.get("cov_method")

    # Get EWM span parameter
    try: ewm_span = int(request.form.get('ewm_span', 30))
    except Exception: ewm_span = 30

    # optional train/test months from the user
    try: train_months = int(request.form.get('train_months', 36))
    except Exception: train_months = 36
    try: test_months = int(request.form.get('test_months', 3))
    except Exception: test_months = 3

    # basic form validation
    if len(stocks) != len(set(stocks)):
        error = "Your portfolio stocks must be unique."
        return render_template('index.html', error=error, stocks=stocks, weights=weight_percents, stock_options=stocks_list, mean_method=mean_method, cov_method=cov_method)

    try:
        weights = [float(w) for w in weight_percents]
    except Exception:
        error = "Invalid weight values."
        return render_template('index.html', error=error, stocks=stocks, weights=weight_percents, stock_options=stocks_list, mean_method=mean_method, cov_method=cov_method)

    if abs(sum(weights) - 100.0) > 1e-6:
        error = "Your portfolio weights must sum to 100%."
        return render_template('index.html', error=error, stocks=stocks, weights=weight_percents, stock_options=stocks_list, mean_method=mean_method, cov_method=cov_method)
    weights = dict(zip(stocks, [float(w)/100 for w in weights]))


    try:
        pfo = portfolio(stocks, mean_method, cov_method, ewm_span=ewm_span)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(train_months, test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers()
        train_return_results, train_risk_results = pfo.optimize_user_portfolio(weights=weights)
        if train_return_results.variance.success is not None:
            return render_template('index.html', error=train_return_results.variance.success, stocks=stocks, weights=weight_percents, stock_options=stocks_list, mean_method=mean_method, cov_method=cov_method)
        if train_risk_results.variance.success is not None:
            return render_template('index.html', error=train_return_results.variance.success, stocks=stocks, weights=weight_percents, stock_options=stocks_list, mean_method=mean_method, cov_method=cov_method)

        test_return_results = compute_test_metrics(pfo, test_df=test_df, train_results=train_return_results)
        test_risk_results   = compute_test_metrics(pfo, test_df=test_df, train_results=train_risk_results)
        user_train_metrics = pfo.portfolio_metrics([weights])[0]
        user_test_metrics = pfo.portfolio_metrics([weights], df=test_df)[0]

        graph_htmls = generate_frontier_graph(pfo, user_pf=user_train_metrics, opt_for_risk=train_risk_results, opt_for_return=train_return_results)  # frontier graphs
        # We need test_df to determine the date ranges for backtrader
        backtest_plots_data = generate_backtrader_plots(pfo, test_df_for_dates=test_df, opt_for_risk=train_risk_results, opt_for_return=train_return_results, user_pf=user_train_metrics, months_list=(1, 2, 3))

        train_stats = format_weights_mv(stats=user_train_metrics, opt_return=train_return_results, opt_risk=train_risk_results)
        test_stats = format_weights_mv(stats=user_test_metrics, opt_return=test_return_results, opt_risk=test_risk_results)
        stats = {
            "train": train_stats,
            "test": test_stats,
            "weights": {
                "Variance-Optimised Weights": train_return_results.variance.weights,
                "VaR-Optimised Weights": train_return_results.var.weights,
                "CVaR-Optimised Weights": train_return_results.cvar.weights,
                "Sharpe-Optimised Weights": train_return_results.sharpe.weights,
                "MaxDD-Optimised Weights": train_return_results.maxdd.weights,

                "Return-Optimized Variance Weights": train_risk_results.variance.weights,
                "Return-Optimized VaR Weights": train_risk_results.var.weights,
                "Return-Optimized CVaR Weights": train_risk_results.cvar.weights,
                "Return-Optimized Sharpe Weights": train_risk_results.sharpe.weights,
                "Return-Optimized MaxDD Weights": train_risk_results.maxdd.weights,
            }
        }
        return render_template('index.html', stats=stats, graph_htmls=graph_htmls, backtest_plots_data=backtest_plots_data, used_train=used_train, used_test=used_test, total_months=total_months, train_months=train_months, test_months=test_months, stocks=stocks, weights=weights, stock_options=stocks_list, mean_method=mean_method, cov_method=cov_method)

    except Exception as e:
        print(e)
        return render_template('index.html', error=str(e), stock_options=stocks_list)


@app.route('/risk_opt', methods=['POST'])
def risk_opt():

    if request.method != 'POST':
        return redirect(url_for('index'))

    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    # Collect form data
    stocks = request.form.getlist('stock')
    risk_type = request.form.get("risk_metric")
    risk_value = request.form.get("risk_value")

    mean_method = request.form.get("mean_method")
    cov_method = request.form.get("cov_method")

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
        return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value, risktype=risk_type, mean_method=mean_method, cov_method=cov_method)

    try:
        risk_value = float(risk_value)
    except (TypeError, ValueError):
        error = "Invalid risk value"
        return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value, risktype=risk_type, mean_method=mean_method, cov_method=cov_method)
    print("target risk", risk_value)

    try:
        ewm_span = int(request.form.get('ewm_span', 30))
    except Exception:
        ewm_span = 30

    # Perform the heavy work inside try so we can show a clean error on failure
    try:
        pfo = portfolio(stocks, mean_method, cov_method, ewm_span=ewm_span)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(train_months, test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers()

        # optimise target risk and set as user weights
        if risk_type == 'variance': opt = pfo.optimize_max_return_for_volatility(risk_value)
        elif risk_type == 'cvar': opt = pfo.optimize_max_return_for_cvar(risk_value)
        elif risk_type == 'var': opt = pfo.optimize_max_return_for_var(risk_value)
        elif risk_type == 'sharpe': opt = pfo.optimize_max_return_for_sharpe(risk_value)
        elif risk_type == 'maxdd': opt = pfo.optimize_max_return_for_maxdd(risk_value)
        else:
            return render_template('index.html', error=f"Invalid risk metric {risk_type}", stocks=stocks, stock_options=stocks_list, riskvalue=risk_value, mean_method=mean_method, cov_method=cov_method)

        if isinstance(opt, str):
            print("error:", opt)
            error = f"Could not optimise on target risk {risk_value}: {opt}"
            return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value, risktype=risk_type, mean_method=mean_method, cov_method=cov_method)
        print(opt)

        train_metrics = pfo.portfolio_metrics([opt.weights])[0] # with train data
        test_metrics = pfo.portfolio_metrics([opt.weights], df=test_df)[0] # with test_data

        # Generate backtrader plots instead of old Plotly ones
        opt_container = OptimizationResultsContainer()
        opt_container.__setattr__(risk_type, train_metrics)
        backtest_plots_data = generate_backtrader_plots(pfo, test_df_for_dates=test_df, opt_for_risk=opt_container, opt_for_return=OptimizationResultsContainer(), months_list=(1, 2, 3))

        # for all additional points to be rendered
        for attr_name in OptimizationResultsContainer.model_fields.keys():
            setattr(opt_container, attr_name, train_metrics)        
        graph_htmls = generate_frontier_graph(pfo, opt_for_risk=opt_container)  # frontier graphs

        train_stats = format_weights_risk(train_metrics)
        test_stats = format_weights_risk(test_metrics)

        stats = {"train": train_stats, "test": test_stats, "weights": {"Optimised return": opt.weights}}

        return render_template('index.html', stats=stats, graph_htmls=graph_htmls, backtest_plots_data=backtest_plots_data, used_train=used_train, used_test=used_test, total_months=total_months, train_months=train_months, test_months=test_months, stocks=stocks, stock_options=stocks_list, riskvalue=risk_value, risktype=risk_type, mean_method=mean_method, cov_method=cov_method)

    except Exception as e:
        print(e)
        return render_template('index.html', error=str(e), stock_options=stocks_list)


@app.route('/return_opt', methods=['POST'])
def return_opt():

    if request.method != 'POST':
        return redirect(url_for('index'))

    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    # Collect form data
    stocks = request.form.getlist('stock')
    target_return = request.form.get("return_value")

    mean_method = request.form.get("mean_method")
    cov_method = request.form.get("cov_method")

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
        return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, returnval=target_return, mean_method=mean_method, cov_method=cov_method)

    try:
        target_return = float(target_return)/100
    except (TypeError, ValueError):
        error = "Invalid return value"
        return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, returnval=target_return, mean_method=mean_method, cov_method=cov_method)
    print("target return", target_return)


    # try:
    pfo = portfolio(stocks, mean_method, cov_method)

    # split and use train for optimisation
    train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(train_months, test_months)
    pfo.use_df(train_df)

    pfo.calculate_frontiers()

    # optimise target risk and set as user weights
    train_return_results, _ = pfo.optimize_user_portfolio(ret=True, risk=False, targetR=target_return)

    if all(port.success is not None for _, port in train_return_results.items()):
        bounds = (float(pfo.mu.min() + 1e-6)*100, float(pfo.mu.max() - 1e-6)*100)
        error = f"Could not optimise on target. Try with a return value in {bounds}"
        return render_template('index.html', error=error, stocks=stocks, stock_options=stocks_list, returnval=target_return*100, mean_method=mean_method, cov_method=cov_method)

    test_return_results = compute_test_metrics(pfo, test_df=test_df, train_results=train_return_results)

    graph_htmls = generate_frontier_graph(pfo, opt_for_return=train_return_results)  # frontier graphs
    # We need test_df to determine the date ranges for backtrader
    backtest_plots_data = generate_backtrader_plots(pfo, test_df_for_dates=test_df, opt_for_return=train_return_results, opt_for_risk=OptimizationResultsContainer(), months_list=(1, 2, 3))

    train_stats = format_weights_return(train_return_results)
    test_stats = format_weights_return(test_return_results)

    stats = {
        "train": train_stats,
        "test": test_stats,
        "weights": {
                "Variance-Optimised Weights": train_return_results.variance.weights,
                "VaR-Optimised Weights": train_return_results.var.weights,
                "CVaR-Optimised Weights": train_return_results.cvar.weights,
                "Sharpe-Optimised Weights": train_return_results.sharpe.weights,
                "MaxDD-Optimised Weights": train_return_results.maxdd.weights,
        }
    }

    return render_template('index.html', stats=stats, graph_htmls=graph_htmls, backtest_plots_data=backtest_plots_data, used_train=used_train, used_test=used_test, total_months=total_months, train_months=train_months, test_months=test_months, stocks=stocks, stock_options=stocks_list, returnval=target_return*100, mean_method=mean_method, cov_method=cov_method)

    # except Exception as e:
    #     return render_template('index.html', error=str(e), stock_options=stocks_list)



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run(debug=True)
