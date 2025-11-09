import os
from flask import Flask, render_template, request, redirect, url_for
import csv
from datetime import datetime
from portfolio import portfolio, compute_test_metrics
from graph import generate_frontier_graph, generate_backtrader_plots
from utils import *
from utils import extract_params

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

@app.route('/mean_variance', methods=['POST', 'GET'])
def mean_variance():

    if request.method != 'POST':
        return redirect(url_for('index'))

    ctx = extract_params(request, data_directory)
    if ctx.error is not None:
        return render_template('index.html', **ctx.model_dump(exclude_none=True))


    try:
        weights = [float(w) for w in ctx.weights]
    except Exception:
        ctx.error = "Invalid weight values."
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    if abs(sum(weights) - 100.0) > 1e-6:
        ctx.error = "Your portfolio weights must sum to 100%."
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    weights = dict(zip(ctx.stocks, [float(w)/100 for w in ctx.weights]))
    rf = ctx.risk_free / 100

    try:
        pfo = portfolio(ctx.stocks, ctx.mean_method, ctx.cov_method, ewm_span=ctx.ewm_span)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(ctx.train_months, ctx.test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers(rf=rf)
        train_return_results, train_risk_results = pfo.optimize_user_portfolio(weights=weights, rf=rf)

        if train_return_results.variance.error is not None:
            ctx.error = "Please choose another portfolio and try again."
            return render_template('index.html', **ctx.model_dump(exclude_none=True))

        if train_risk_results.variance.error is not None:
            ctx.error = "Please choose another portfolio and try again."
            return render_template('index.html', **ctx.model_dump(exclude_none=True))

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

        ctx = IndexContext(
            **ctx.model_dump(exclude_none=True),
            stats=stats, 
            graph_htmls=graph_htmls, 
            backtest_plots_data=backtest_plots_data,
            used_train=used_train, 
            used_test=used_test, 
            total_months=total_months, 
        )
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    except Exception as e:
        raise Exception(e)
        print(e)
        ctx.error = str(e)
        return render_template('index.html', **ctx.model_dump(exclude_none=True))


@app.route('/risk_opt', methods=['POST', 'GET'])
def risk_opt():

    if request.method != 'POST':
        return redirect(url_for('index'))

    ctx = extract_params(request, data_directory)
    if ctx.error is not None:
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    risk_type = ctx.risktype
    risk_value = ctx.riskvalue
    if risk_type is None or risk_value is None:
        ctx.error = "Please choose a risk type."
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    rf = ctx.risk_free / 100

    # Perform the heavy work inside try so we can show a clean error on failure
    try:
        pfo = portfolio(ctx.stocks, ctx.mean_method, ctx.cov_method, ewm_span=ctx.ewm_span)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(ctx.train_months, ctx.test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers(rf=rf)

        # optimise target risk and set as user weights
        if risk_type == 'variance': opt = pfo.optimize_max_return_for_volatility(risk_value)
        elif risk_type == 'cvar': opt = pfo.optimize_max_return_for_cvar(risk_value)
        elif risk_type == 'var': opt = pfo.optimize_max_return_for_var(risk_value)
        elif risk_type == 'sharpe': opt = pfo.optimize_max_return_for_sharpe(risk_value)
        elif risk_type == 'maxdd': opt = pfo.optimize_max_return_for_maxdd(risk_value)
        else:
            ctx.error = f"Invalid risk metric {risk_type}"
            return render_template('index.html', **ctx.model_dump(exclude_none=True))

        print(opt)
        if opt.error is not None:
            print("error:", opt.error)
            ctx.error = f"Could not optimise on target risk {risk_value}: {opt.error}"
            return render_template('index.html', **ctx.model_dump(exclude_none=True))

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

        train_stats = format_weights_risk(train_metrics, risk_type=risk_type)
        test_stats = format_weights_risk(test_metrics, risk_type=risk_type)

        stats = {"train": train_stats, "test": test_stats, "weights": {"Optimised return": opt.weights}}

        print(ctx)
        ctx = IndexContext(
            **ctx.model_dump(exclude_none=True),
            stats=stats, 
            graph_htmls=graph_htmls, 
            backtest_plots_data=backtest_plots_data,
            used_train=used_train, 
            used_test=used_test, 
            total_months=total_months, 
        )
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    except Exception as e:
        raise Exception(e)
        ctx.error = str(e)
        return render_template('index.html', **ctx.model_dump(exclude_none=True))


@app.route('/return_opt', methods=['POST', 'GET'])
def return_opt():

    if request.method != 'POST':
        return redirect(url_for('index'))

    ctx = extract_params(request, data_directory)
    if ctx.error is not None:
        return render_template('index.html', **ctx.model_dump(exclude_none=True))
    if ctx.returnval is None:
        ctx.error = "Invalid portfolio target return"
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    target_return = ctx.returnval / 100
    rf = ctx.risk_free / 100

    try:
        pfo = portfolio(ctx.stocks, ctx.mean_method, ctx.cov_method)

        # split and use train for optimisation
        train_df, test_df, used_train, used_test, total_months = pfo.split_train_test(ctx.train_months, ctx.test_months)
        pfo.use_df(train_df)

        pfo.calculate_frontiers(rf=rf)

        # optimise target risk and set as user weights
        train_return_results, _ = pfo.optimize_user_portfolio(ret=True, risk=False, targetR=target_return, rf=rf)
        print(train_return_results.items())

        if all(port.error is not None for _, port in train_return_results.items()):
            bounds = (float(pfo.mu.min() + 1e-6)*100, float(pfo.mu.max() - 1e-6)*100)
            ctx.error = f"Could not optimise on target. Try with a return value in {bounds}"
            return render_template('index.html', **ctx.model_dump(exclude_none=True))

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

        ctx = IndexContext(
            **ctx.model_dump(exclude_none=True),
            stats=stats, 
            graph_htmls=graph_htmls, 
            backtest_plots_data=backtest_plots_data,
            used_train=used_train, 
            used_test=used_test, 
            total_months=total_months, 
        )
        return render_template('index.html', **ctx.model_dump(exclude_none=True))

    except Exception as e:
        raise Exception(e)
        ctx.error = str(e)
        return render_template('index.html', **ctx.model_dump(exclude_none=True))


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Portfolio Optimization Flask App")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on (default: 8080)')
    args = parser.parse_args()

    # Use waitress, a production-ready WSGI server
    from waitress import serve
    print(f"Starting production server on http://{args.host}:{args.port}")
    serve(app, host=args.host, port=args.port)
