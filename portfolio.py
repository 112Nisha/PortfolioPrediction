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

        # full_df holds the complete history (used for backtests). self.df
        # will be the working dataframe used for optimization (usually train set).
        self.full_df = self._load_data()  # full history
        self.df = self.full_df.copy()

        self.mu, self.S, self.rets = None, None, None
        if not self.df.empty:
            self.rets = self.df.pct_change().dropna()
            self.mu = expected_returns.mean_historical_return(self.df)
            self.S = risk_models.sample_cov(self.df)

        self.mv_frontier_pts = []
        self.cvar_frontier_pts = []
        self.var_frontier_pts = []

        self.pf_metrics = {}

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

    def split_train_test(self, train_months: int = 36, test_months: int = 3):
        """Split the available full data into a train and test partition by months.

        Defaults: train 36 months, test 3 months. The requested months are capped
        by the available history in `self.full_df`.

        Returns (train_df, test_df, used_train_months, used_test_months, total_months).
        """
        if self.full_df is None or self.full_df.empty:
            return pd.DataFrame(), pd.DataFrame(), 0, 0

        # total available months between first and last index (approx)
        start = self.full_df.index.min()
        end = self.full_df.index.max()
        total_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

        # cap requested months to available months
        req_train = int(train_months)
        req_test = int(test_months)
        if req_train + req_test > total_months:
            # give test its requested share (but capped) and reduce train
            req_test = min(req_test, total_months)
            req_train = max(total_months - req_test, 0)

        # compute cutoffs
        test_start = end - pd.DateOffset(months=req_test) + pd.DateOffset(days=1)
        train_start = test_start - pd.DateOffset(months=req_train)

        train_df = self.full_df[self.full_df.index >= train_start]
        train_df = train_df[train_df.index < test_start]
        test_df = self.full_df[self.full_df.index >= test_start]

        return train_df, test_df, req_train, req_test, total_months

    def use_df(self, df):
        """Replace working dataframe (used for optimization) and recompute mu/S/rets.

        Keep self.full_df unchanged so backtests operate on the full history.
        """
        self.df = df.copy()
        if not self.df.empty:
            self.rets = self.df.pct_change().dropna()
            self.mu = expected_returns.mean_historical_return(self.df)
            self.S = risk_models.sample_cov(self.df)
        else:
            self.rets, self.mu, self.S = None, None, None

    def _portfolio_var(self, w, alpha=0.05):
        pr = self.rets @ w
        return -np.percentile(pr, alpha * 100)

    def _portfolio_cvar(self, w, alpha=0.05):
        pr = self.rets @ w
        var = np.percentile(pr, alpha * 100)
        return -pr[pr <= var].mean()
        # return self.rets[self.rets < var].mean().to_numpy()[0]

    def _portfolio_return(self, w):
        return w @ self.mu

    def _as_series(self, w):
        index = self.mu.index
        return pd.Series(w, index=index).sort_index().round(2)

    def backtest_series(self, test_df, months=3):
        """Compute cumulative return series for user & optimized portfolios over
        the last `months` months from the full history.

        Returns a dict of pandas.Series (keys: 'user','variance','var','cvar') or
        None when insufficient data.
        """
        if self.full_df is None or self.full_df.empty:
            return None

        weights = self.user_weights
        # helper to coerce weight containers to numpy arrays aligned with columns
        def to_array(w):
            if w is None:
                return None
            if isinstance(w, np.ndarray):
                return w
            if isinstance(w, pd.Series):
                return w.reindex(self.full_df.columns).to_numpy()
            if isinstance(w, dict):
                return pd.Series(w).reindex(self.full_df.columns).to_numpy()
            try:
                arr = np.array(w)
                if arr.shape[0] == len(self.full_df.columns):
                    return arr
            except Exception:
                pass
            return None

        # end_date = self.full_df.index.max()
        # start_date = end_date - pd.DateOffset(months=int(months)) + pd.DateOffset(days=1)

        # rets = self.full_df.pct_change().dropna()
        # rets_sub = rets[rets.index >= start_date]
        # if rets_sub.empty:
        #     return None
        old_df = self.df.copy()
        self.use_df(test_df)

        user_w = to_array(weights)
        variance_w = to_array(self.opt_variance[2])
        var_w = to_array(self.opt_var[2])
        cvar_w = to_array(self.opt_cvar[2])

        out = {}
        if user_w is not None:
            pr_user = self.rets @ (user_w)
            out['user'] = (1 + pr_user).cumprod()
        if variance_w is not None:
            pr_var = self.rets @ (variance_w)
            out['variance'] = (1 + pr_var).cumprod()
        if var_w is not None:
            pr_v = self.rets @ (var_w)
            out['var'] = (1 + pr_v).cumprod()
        if cvar_w is not None:
            pr_c = self.rets @ (cvar_w)
            out['cvar'] = (1 + pr_c).cumprod()

        self.use_df(old_df)

        return out

    def _optimize_mv_for_return(self, target_return):
        try:
            ef_mv = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            ef_mv.efficient_return(target_return)
            w_mv = self._as_series(ef_mv.clean_weights())
            ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
            return (vol_mv, 100*ret_mv, w_mv)
        except Exception:
            return None

    def _optimize_cvar_for_return(self, target_return):
        try:
            ef_c = EfficientCVaR(self.mu, self.rets, weight_bounds=(-1,1))
            ef_c.efficient_return(target_return)
            # ensure weights are returned as a Series aligned with self.mu index
            # ef_c.clean_weights() returns a dict; _as_series will align by index
            w_c = self._as_series(ef_c.clean_weights())
            ret_c, cvar_risk = ef_c.portfolio_performance()
            return (cvar_risk, 100*ret_c, w_c)
        except Exception:
            return None

    def _optimize_var_for_return(self, target_return):
        num_assets = len(self.mu)
        bounds = tuple((-1, 1) for _ in range(num_assets))
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
                return (var_risk_val, 100*ret_var, w_var)
            else:
                # record last failure message
                last_message = getattr(result, 'message', str(result))

        # If we reach here, all attempts failed. Log for diagnostics and return None.
        print(f"_optimize_var_for_return failed after {attempts} attempts for target_return={target_return}; last message: {last_message}")
        return None

    def _optimize_max_return_for_variance(self, target_variance):
        """Find portfolio with maximum return for given variance constraint."""
        try:
            ef_max = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            # Use efficient_risk method with target volatility (sqrt of variance)
            target_volatility = np.sqrt(target_variance)
            ef_max.efficient_risk(target_volatility)
            w_max = self._as_series(ef_max.clean_weights())
            ret_max, vol_max, _ = ef_max.portfolio_performance()
            return (vol_max**2, 100*ret_max, w_max)  # return variance, not volatility
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
        
        if self.mv_frontier_pts:
            var_grid = np.linspace(
                min(pt[0]**2 for pt in self.mv_frontier_pts) + 1e-6,
                max(pt[0]**2 for pt in self.mv_frontier_pts) - 1e-6,
                50
            )
            self.max_return_frontier_pts = [
                res for v in var_grid
                if (res := self._optimize_max_return_for_variance(v)) is not None
            ]

        self.pf_return = self._portfolio_return(self.user_weights)
        self.pf_variance = self.user_weights.T @ self.S @ self.user_weights        
        self.opt_variance = self._optimize_mv_for_return(self.pf_return)
        self.opt_var = self._optimize_var_for_return(self.pf_return)
        self.opt_cvar = self._optimize_cvar_for_return(self.pf_return)
        self.opt_max_return = self._optimize_max_return_for_variance(self.pf_variance)

    def portfolio_metrics(self, df=None):
        weights = self.user_weights
        if weights is None or self.mu is None:
            return None

        if df is not None:
            old_df = self.df.copy()
            self.use_df(df)

        returns = self._portfolio_return(weights)
        user_variance = weights.T @ self.S @ weights

        # calculate user's portfolio volatility
        ef_user = EfficientFrontier(self.mu, self.S)
        ef_user.set_weights(dict(zip(self.stocks, weights)))
        _, user_vol, _ = ef_user.portfolio_performance(verbose=False)

        pf_metrics = {
            "return": round(returns*100, 4),
            "user_variance": user_vol, # weights.T @ self.S @ weights,
            "user_var": self._portfolio_var(weights),
            "user_cvar": self._portfolio_cvar(weights),

            "opt_variance": self.opt_variance[0] if self.opt_variance else None,
            "opt_var": self.opt_var[0] if self.opt_var else None,
            "opt_cvar": self.opt_cvar[0] if self.opt_cvar else None,
            "opt_max_return": self.opt_max_return[1] if self.opt_max_return else None,
            "opt_max_return_variance": self.opt_max_return[0] if self.opt_max_return else None,

            "user_weights": weights,
            "opt_variance_weights": self.opt_variance[2] if self.opt_variance else None,
            "opt_var_weights": self.opt_var[2] if self.opt_var else None,
            "opt_cvar_weights": self.opt_cvar[2] if self.opt_cvar else None,
            "opt_max_return_weights": self.opt_max_return[2] if self.opt_max_return else None,
        }

        if df is not None: # restore the old dataframe
            self.use_df(old_df)

        return pf_metrics
