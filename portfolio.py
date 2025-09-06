import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from functools import reduce
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize


class portfolio():
    def __init__(self, stocks, user_weights):
        self.stocks = stocks
        self.user_weights = np.array([float(w)/100 for w in user_weights])

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
        initial_guess = np.array(num_assets * [1.0 / num_assets])
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
                ]
            )
            if result.success:
                w_var = self._as_series(result.x)
                var_risk_val = self._portfolio_var(w_var)
                ret_var = self._portfolio_return(w_var)
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
        self.pf_return = self._portfolio_return(weights)

        opt_variance = self._optimize_mv_for_return(self.pf_return)
        self.pf_variance_metrics = {
            "user_variance": weights.T @ self.S @ weights,
            "opt_variance": opt_variance[0] if opt_variance else None,
            "opt_variance_weights": opt_variance[2] if opt_variance else None,
        }

        opt_var = self._optimize_var_for_return(self.pf_return)
        self.pf_var_metrics = {
            "user_var": self._portfolio_var(weights),
            "opt_var": opt_var[0] if opt_var else None,
            "opt_var_weights": opt_var[2] if opt_var else None,
        }

        opt_cvar = self._optimize_cvar_for_return(self.pf_return)
        print(opt_cvar)
        self.pf_cvar_metrics = {
            "user_cvar": self._portfolio_cvar(weights),
            "opt_cvar": opt_cvar[0] if opt_cvar else None,
            "opt_cvar_weights": opt_cvar[2] if opt_cvar else None,
        }

    def backtest_plot(self, weights, period):
        # period: expect a pandas DateOffset-friendly specification like '3M' for 3 months
        if self.df.empty:
            return None

        # Ensure the portfolio metrics for the provided weights are computed so we can find
        # equivalent points on each frontier for the user's return.
        try:
            self.portfolio_metrics(weights)
        except Exception:
            # ensure we don't crash backtesting if metrics fail
            pass

        # Helper to coerce various weight formats to numpy array aligned with self.df columns
        def to_array(w):
            if w is None:
                return None
            if isinstance(w, np.ndarray):
                return w
            if isinstance(w, pd.Series):
                # align to dataframe columns
                return w.reindex(self.df.columns).to_numpy()
            if isinstance(w, dict):
                return pd.Series(w).reindex(self.df.columns).to_numpy()
            # list-like
            try:
                arr = np.array(w)
                if arr.shape[0] == len(self.df.columns):
                    return arr
            except Exception:
                pass
            return None

        rets = self.df.pct_change().dropna()
        end_date = self.df.index.max()
        try:
            # support '3M' style period
            months = int(str(period).upper().replace('M', ''))
            start_date = end_date - pd.DateOffset(months=months)
        except Exception:
            # fallback to 3 months
            start_date = end_date - pd.DateOffset(months=3)

        rets_sub = rets[rets.index >= start_date]
        if rets_sub.empty:
            return None

        # Prepare user portfolio cumulative returns (if available)
        user_w = to_array(weights)
        pr_user = None
        cum_user = None
        if user_w is not None:
            pr_user = rets_sub.dot(user_w)
            cum_user = (1 + pr_user).cumprod()

        # For each risk measure, find the optimized weights (for the user's portfolio return) and backtest
        opt_var_w = None
        if hasattr(self, 'pf_variance_metrics'):
            opt_var_w = to_array(self.pf_variance_metrics.get('opt_variance_weights'))

        opt_v_w = None
        if hasattr(self, 'pf_var_metrics'):
            opt_v_w = to_array(self.pf_var_metrics.get('opt_var_weights'))

        opt_c_w = None
        if hasattr(self, 'pf_cvar_metrics'):
            opt_c_w = to_array(self.pf_cvar_metrics.get('opt_cvar_weights'))

        # Helper to build a figure comparing user vs an optimized portfolio
        def build_comparison_fig(opt_w, opt_name, include_user=True):
            traces_local = []
            title = f'Backtest over last {period} — {opt_name}'
            if include_user and cum_user is not None:
                traces_local.append(go.Scatter(x=cum_user.index, y=cum_user.values, mode='lines', name='User Portfolio', line=dict(width=2)))
            if opt_w is not None:
                pr_opt = rets_sub.dot(opt_w)
                cum_opt = (1 + pr_opt).cumprod()
                traces_local.append(go.Scatter(x=cum_opt.index, y=cum_opt.values, mode='lines', name=opt_name, line=dict(dash='dash')))
            if not traces_local:
                # nothing to show
                traces_local.append(go.Scatter(x=[rets_sub.index.min(), rets_sub.index.max()], y=[1, 1], mode='lines', name='No data'))
            layout = go.Layout(title=title, xaxis=dict(title='Date'), yaxis=dict(title='Cumulative Return (base 1)'), legend=dict(orientation='h', x=0, y=1.1), margin=dict(l=40, r=40, t=60, b=40))
            return go.Figure(data=traces_local, layout=layout)

        # Build three figures: Variance-opt, VaR-opt, CVaR-opt
        figs = []
        figs.append(build_comparison_fig(opt_var_w, 'Variance-opt'))
        figs.append(build_comparison_fig(opt_v_w, 'VaR-opt'))
        figs.append(build_comparison_fig(opt_c_w, 'CVaR-opt'))

        # Convert figures to HTML snippets. Include plotly.js in the first only.
        html_parts = []
        for i, fig in enumerate(figs):
            include_js = 'cdn' if i == 0 else False
            html_parts.append(fig.to_html(include_plotlyjs=include_js, full_html=False))

        # Wrap the three figures in a container
        combined = '<div class="backtest-container">' + '\n<hr/>' .join(html_parts) + '</div>'
        return combined
