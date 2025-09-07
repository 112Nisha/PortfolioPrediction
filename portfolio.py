import os
import numpy as np
import pandas as pd
from functools import reduce
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize


class portfolio():
    def __init__(self, stocks, user_weights):
        self.stocks = stocks
        self.user_weights = np.array([float(w)/100 for w in user_weights])

        self.df = self._load_data()  # will store dataframe with return information for all stocks.
        self.mu = None
        self.S = None
        self.rets = None
        if not self.df.empty:
            self.rets = self.df.pct_change().dropna()
            self.mu = expected_returns.mean_historical_return(self.df)
            self.S = risk_models.sample_cov(self.df)

        self.mv_frontier_pts = []
        self.cvar_frontier_pts = []
        self.var_frontier_pts = []

    def _load_data(self):
        dfs = []
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        for stock in self.stocks:
            for fname in os.listdir(data_dir):
                if fname.lower().startswith(stock.lower()):
                    df = pd.read_csv(os.path.join(data_dir, fname), parse_dates=["Date"])
                    dfs.append(df[["Date", "Close"]].rename(columns={"Close": stock}))
                    break
        if len(dfs) < len(self.stocks):
            print(f"Warning: Could not load data for all stocks: {self.stocks}")
            return pd.DataFrame()  # Return empty if not all data is found
        merged = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs)
        return merged.set_index("Date").dropna()


    def _portfolio_var(self, w, alpha=0.05):
        pr = self.rets @ w
        return -np.percentile(pr, alpha * 100)

    def _portfolio_cvar(self, w, alpha=0.05):
        var = self._portfolio_var(w, alpha)
        return self.rets[self.rets < var].mean().to_numpy()[0]

    def _portfolio_return(self, w):
        return w @ self.mu

    def _as_series(self, w):
        index = self.mu.index
        return pd.Series(w, index=index).sort_index()


    def _optimize_mv_for_return(self, target_return):
        try:
            ef_mv = EfficientFrontier(self.mu, self.S)
            ef_mv.efficient_return(target_return)
            w_mv = self._as_series(ef_mv.clean_weights())
            ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
            return vol_mv, ret_mv, w_mv
        except Exception:
            return None

    def _optimize_cvar_for_return(self, target_return):
        try:
            ef_c = EfficientCVaR(self.mu, self.rets)
            ef_c.efficient_return(target_return)
            # ensure weights are returned as a Series aligned with self.mu index
            # ef_c.clean_weights() returns a dict; _as_series will align by index
            w_c = self._as_series(ef_c.clean_weights())
            ret_c, cvar_risk = ef_c.portfolio_performance()
            return cvar_risk, ret_c, w_c
        except Exception:
            return None

    def _optimize_var_for_return(self, target_return):
        num_assets = len(self.mu)
        bounds = tuple((0, 1) for _ in range(num_assets))
        # Try multiple random restarts to improve robustness — VaR objective is non-convex
        rng = np.random.default_rng()
        attempts = 6
        last_message = None
        for attempt in range(attempts):
            if attempt == 0:
                initial_guess = np.array(num_assets * [1.0 / num_assets])
            else:
                # random positive weights normalized to sum to 1
                r = rng.random(num_assets)
                initial_guess = r / r.sum()

            try:
                result = minimize(
                    self._portfolio_var,
                    initial_guess,
                    args=(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[
                        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                        {'type': 'eq', 'fun': lambda weights: self._portfolio_return(weights) - target_return}
                    ],
                    options={'maxiter': 500}
                )
            except Exception as e:
                last_message = f"optimizer exception: {e}"
                continue

            if result is not None and getattr(result, 'success', False):
                w_var = self._as_series(result.x)
                var_risk_val = self._portfolio_var(w_var)
                ret_var = self._portfolio_return(w_var)
                return var_risk_val, ret_var, w_var
            else:
                # record last failure message
                last_message = getattr(result, 'message', str(result))

        # If we reach here, all attempts failed. Log for diagnostics and return None.
        print(f"_optimize_var_for_return failed after {attempts} attempts for target_return={target_return}; last message: {last_message}")
        return None


    def calculate_frontiers(self):
        if self.df.empty or self.mu is None or self.S is None:
            return

        grid = np.linspace(self.mu.min() + 1e-6, self.mu.max() - 1e-6, 100)
        self.mv_frontier_pts = [
            res for r in grid
            if (res := self._optimize_mv_for_return(r)) is not None
        ]
        self.cvar_frontier_pts = [
            res for r in grid
            if (res := self._optimize_cvar_for_return(r)) is not None
        ]
        self.var_frontier_pts = [
            res for r in grid
            if (res := self._optimize_var_for_return(r)) is not None
        ]

        print(len(self.cvar_frontier_pts), len(self.mv_frontier_pts), len(self.var_frontier_pts))

    def portfolio_metrics(self, weights):
        if weights is None or self.mu is None:
            return None

        # calculate risk metrics and corresponding points on frontiers.
        self.pf_return = self._portfolio_return(weights)

        opt_variance = self._optimize_mv_for_return(self.pf_return)
        self.pf_variance_metrics = {
            "user_variance": weights.T @ self.S @ weights,
            "opt_variance": opt_variance[0] if opt_variance else None,
            "opt_variance_weights": opt_variance[2] if opt_variance else None,
        }
        # print optimized weights for variance frontier (aligned by stock)
        if opt_variance and opt_variance[2] is not None:
            try:
                print("Variance-opt weights for user-return:", opt_variance[2].to_dict())
            except Exception:
                print("Variance-opt weights for user-return:", opt_variance[2])

        opt_var = self._optimize_var_for_return(self.pf_return)
        self.pf_var_metrics = {
            "user_var": self._portfolio_var(weights),
            "opt_var": opt_var[0] if opt_var else None,
            "opt_var_weights": opt_var[2] if opt_var else None,
        }
        # print optimized weights for VaR frontier
        if opt_var and opt_var[2] is not None:
            try:
                print("VaR-opt weights for user-return:", opt_var[2].to_dict())
            except Exception:
                print("VaR-opt weights for user-return:", opt_var[2])

        opt_cvar = self._optimize_cvar_for_return(self.pf_return)
        print(opt_cvar)
        self.pf_cvar_metrics = {
            "user_cvar": self._portfolio_cvar(weights),
            "opt_cvar": opt_cvar[0] if opt_cvar else None,
            "opt_cvar_weights": opt_cvar[2] if opt_cvar else None,
        }
        # print optimized weights for CVaR frontier
        if opt_cvar and opt_cvar[2] is not None:
            try:
                print("CVaR-opt weights for user-return:", opt_cvar[2].to_dict())
            except Exception:
                print("CVaR-opt weights for user-return:", opt_cvar[2])

    def backtest_plot(self, weights, period):
        # Backtesting plotting moved to `graph.backtest_plot` for modularity.
        raise NotImplementedError("backtest_plot moved to graph.backtest_plot")
