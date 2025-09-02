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
        self.user_weights = user_weights

        self.df = self._load_data() # will store dataframe with return information for all stocks.
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
            return pd.DataFrame() # Return empty if not all data is found
        merged = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs)
        return merged.set_index("Date").dropna()


    def _portfolio_var(self, w, returns_df, alpha=0.05):
        pr = returns_df @ w
        return -np.percentile(pr, alpha * 100)

    def _portfolio_return(self, w, mu):
        return w @ mu

    def _as_series(self, w, index):
        return pd.Series(w, index=index).sort_index()
    
    def _optimize_mv_for_return(self, target_return):
        try:
            ef_mv = EfficientFrontier(self.mu, self.S)
            ef_mv.efficient_return(target_return)
            w_mv = self._as_series(ef_mv.clean_weights(), self.mu.index)
            ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
            return vol_mv, ret_mv, w_mv
        except Exception as e:
            print(f"Optimization failed at return {target_return}: {e}")
            return None

    def _optimize_cvar_for_return(self, target_return):
        try:
            ef_c = EfficientCVaR(self.mu, self.rets)
            ef_c.efficient_return(target_return)
            w_c = self._as_series(ef_c.clean_weights(), self.mu.index)
            ret_c, cvar_risk = ef_c.portfolio_performance()
            return cvar_risk, ret_c, w_c
        except Exception:
            return None

    def _optimize_var_for_return(self, target_return):
        num_assets = len(self.mu)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array(num_assets * [1.0 / num_assets])
        try:
            result = minimize(
                self._portfolio_var,
                initial_guess,
                args=(self.rets,),
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                    {'type': 'eq', 'fun': lambda weights: self._portfolio_return(weights, self.mu) - target_return}
                ]
            )
            if result.success:
                w_var = self._as_series(result.x, self.mu.index)
                var_risk_val = self._portfolio_var(w_var, self.rets)
                ret_var = self._portfolio_return(w_var, self.mu)
                return var_risk_val, ret_var, w_var
        except Exception:
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
        pass

    def backtest_plot(self, weights, period):
        pass