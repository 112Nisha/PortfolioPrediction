import os
import time
import numpy as np
import pandas as pd
from portfolio import portfolio

# -----------------------------
# CONFIG
# -----------------------------
MEAN_METHODS = [
    "mean_historical",
    "ewm_mean_historical",
    "capm",
]

COV_METHODS = [
    "sample",
    "semicovariance",
    "ewm",
    "ledoit_wolf",
    "oracle_approx",
]

RISK_METRICS = ["variance", "var", "cvar", "sharpe", "maxdd"]

LOOKBACK_WINDOWS = [
    (36, 3),   # 36 months train, 3 test
    (60, 6),   # 60 months train, 6 test
]

# -----------------------------
# SAFE RUN UTILITY
# -----------------------------
def safe_run(func):
    """Run func(), catch exception, and measure time."""
    t0 = time.time()
    try:
        result = func()
        return result, time.time() - t0, None
    except Exception as e:
        return None, None, str(e)

# -----------------------------
# RISK METRIC CALLER
# -----------------------------
def compute_risk(p, metric):
    if metric == "variance":
        return p._portfolio_vol(p.user_weights)
    if metric == "var":
        return p._portfolio_var(p.user_weights)
    if metric == "cvar":
        return p._portfolio_cvar(p.user_weights)
    if metric == "sharpe":
        return p._portfolio_sharpe(p.user_weights)
    if metric == "maxdd":
        return p._portfolio_max_drawdown(p.user_weights)
    raise ValueError(f"Unknown metric {metric}")

# -----------------------------
# RUN EXPERIMENTS
# -----------------------------
def run_single_experiment(stocks, mean_m, cov_m, train_m, test_m):
    """Runs: optimization, backtest, risk metrics, weight stability"""
    results = []

    # User weights = equal (100%)
    user_w = [100/len(stocks)] * len(stocks)

    p = portfolio(
        stocks,
        mean_method=mean_m,
        cov_method=cov_m,
        user_weights=user_w,
    )

    # SPLIT DATA
    train_df, test_df, _, _, _ = p.split_train_test(train_m, test_m)
    p.use_df(train_df)

    # RUN FRONTIERS + USER PORTFOLIO OPTIMISATION
    _, calc_time, calc_err = safe_run(lambda: p.calculate_frontiers())
    _, opt_time, opt_err = safe_run(lambda: p.optimize_user_portfolio())

    # CALCULATE METRICS
    pf_metrics = p.portfolio_metrics()

    # -----------------------------
    # WEIGHT STABILITY TEST
    # Compare weights from 1st half vs full history
    # -----------------------------
    try:
        half_cut = int(len(train_df) / 2)
        half_df = train_df.iloc[:half_cut]

        p2 = portfolio(
            stocks,
            mean_m,
            cov_m,
            user_weights=user_w,
        )
        p2.use_df(half_df)
        p2.calculate_frontiers()
        p2.optimize_user_portfolio()
        metrics_half = p2.portfolio_metrics()

        w_full = pf_metrics["opt_variance_weights"]
        w_half = metrics_half["opt_variance_weights"]

        if hasattr(w_full, "values") and hasattr(w_half, "values"):
            stability_score = np.mean(np.abs(w_full.values - w_half.values))
        else:
            stability_score = None
        weight_stability = stability_score

    except Exception as e:
        weight_stability = None

    # -----------------------------
    # BACKTEST
    # -----------------------------
    def run_backtest():
        weights_dict = {s: 1/len(stocks) for s in stocks}
        return p.run_backtest_backtrader(weights_dict)

    bt, bt_time, bt_err = safe_run(run_backtest)
    bt_return = None if bt is None else float(bt["value"].iloc[-1] / bt["value"].iloc[0] - 1)

    # -----------------------------
    # RECORD RISK METRICS
    # -----------------------------
    row = dict(
        stocks=str(stocks),
        mean_method=mean_m,
        cov_method=cov_m,
        train_months=train_m,
        test_months=test_m,
        calc_error=calc_err,
        opt_error=opt_err,
        calc_time=calc_time,
        opt_time=opt_time,
        weight_stability=weight_stability,
        backtest_error=bt_err,
        backtest_return=bt_return,
        backtest_time=bt_time,
    )

    for metric in RISK_METRICS:
        metric_val, metric_time, metric_err = safe_run(lambda: compute_risk(p, metric))
        row[f"{metric}_value"] = metric_val
        row[f"{metric}_error"] = metric_err
        row[f"{metric}_runtime"] = metric_time

    return row

# -----------------------------
# MAIN
# -----------------------------
def run_all():
    # Load list of tickers from data folder
    stocks = [
        f.split(".")[0]
        for f in os.listdir("./data")
        if f.lower().endswith(".csv")
    ]

    results = []
    print(f"✅ Found tickers: {stocks}\n")

    for mean_m in MEAN_METHODS:
        for cov_m in COV_METHODS:
            for (train_m, test_m) in LOOKBACK_WINDOWS:
                print(f"➡ Running {mean_m} + {cov_m} | train={train_m} test={test_m}")

                try:
                    row = run_single_experiment(stocks, mean_m, cov_m, train_m, test_m)
                except Exception as e:
                    row = {
                        "mean_method": mean_m,
                        "cov_method": cov_m,
                        "train_months": train_m,
                        "test_months": test_m,
                        "fatal_error": str(e),
                    }

                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("full_experiment_results.csv", index=False)
    print("\n✅ Saved: full_experiment_results.csv")
    print(df)

if __name__ == "__main__":
    run_all()
