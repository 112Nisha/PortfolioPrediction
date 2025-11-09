from pandas.core.frame import DataFrame

import os
import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
from utils import Portfolio, OptimizationResultsContainer
import backtrader as bt
from typing import List

def prepare_market_returns(csv_file='data/nifty50_data.csv'):
    market_df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
    if market_df.shape[1] > 1:
        market_df = market_df[['Close']]  
    return market_df

def compute_mu_and_cov(df, mean_method, cov_method, market_prices=None, risk_free_rate=0.0, ewm_span=None) -> tuple[np.ndarray | pd.Series]:
    """Compute expected returns (mu) and covariance matrix (S) based on chosen methods."""

    # --- MEAN (EXPECTED RETURN) ---
    if mean_method == "mean_historical":
        mu = expected_returns.mean_historical_return(df)
    elif mean_method == "ewm_mean_historical":
        if ewm_span:
            mu = expected_returns.ema_historical_return(df, span=ewm_span)
        else:
            mu = expected_returns.ema_historical_return(df)
    elif mean_method == "capm":
        # if market_prices is None:
        #     raise ValueError("CAPM requires market index prices (market_prices).")
        mu = expected_returns.capm_return(df, market_prices=None, risk_free_rate=risk_free_rate)
    else:
        raise ValueError(f"Unknown mean method: {mean_method}")

    # --- COVARIANCE / RISK MATRIX ---
    if cov_method == "sample":
        S = risk_models.sample_cov(df)
    elif cov_method == "semicovariance":
        S = risk_models.semicovariance(df)
    elif cov_method == "ewm":
        if ewm_span:
            S = risk_models.exp_cov(df, span=ewm_span)
        else:
            S = risk_models.exp_cov(df)
    elif cov_method == "mcd":
        S = risk_models.min_cov_determinant(df)
    elif cov_method == "ledoit_wolf":
        S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
    elif cov_method == "oracle_approx":
        S = risk_models.CovarianceShrinkage(df).oas()
    elif cov_method == "manual_shrinkage":
        shrinkage = risk_models.CovarianceShrinkage(df)
        S = shrinkage.shrinkage(0.1)  # You can tune this shrinkage intensity
    elif cov_method == "fix_psd":
        raw_cov = risk_models.sample_cov(df)
        S = risk_models.fix_nonpositive_semidefinite(raw_cov)
    elif cov_method == "cov_to_corr":
        raw_cov = risk_models.sample_cov(df)
        S = risk_models.cov_to_corr(raw_cov)
    else:
        raise ValueError(f"Unknown covariance method: {cov_method}")


    # mu = mu.to_numpy() if hasattr(mu, "to_numpy") else np.array(mu) # don't convert to ndarray for efficient frontier tickers to be correctly set.
    S = S.to_numpy() if hasattr(S, "to_numpy") else np.array(S)

    return mu, S


class BuyAndHold(bt.Strategy):
    """
    A simple strategy that applies fixed weights to a portfolio. Rebalancing is never done.
    """
    params = (('weights', None),)

    def __init__(self):
        if self.p.weights is None:
            raise ValueError("Weights must be provided to the BuyAndHold strategy")
        self.invested = False
        self.values = []

    def next(self):
        if not self.invested:
            total_value = self.broker.getvalue()
            for data in self.datas:
                stock_name = data._name
                weight = self.p.weights.get(stock_name, 0.0)

                target_value = total_value * weight
                self.order_target_value(data, target=target_value)

            self.invested = True
        self.values.append((self.datas[0].datetime.date(0), self.broker.getvalue()))


class portfolio():
    def __init__(self, stocks, mean_method, cov_method, ewm_span=None):
        self.stocks = stocks
        self.ewm_span = ewm_span

        # full_df holds the complete history (used for backtests). self.df
        # will be the working dataframe used for optimization (usually train set).
        self.full_df = self._load_data()  # full history
        self.df = self.full_df.copy()
        self.mean_method = mean_method
        self.cov_method = cov_method

        if self.df.empty:
            raise ValueError
        
        self.rets = expected_returns.returns_from_prices(self.df.dropna())
        self.mu, self.S = compute_mu_and_cov(self.df, mean_method, cov_method, ewm_span=self.ewm_span)

        self.mv_frontier_pts = []
        self.cvar_frontier_pts = []
        self.var_frontier_pts = []
        self.sharpe_frontier_pts = []
        self.maxdd_frontier_pts = []

        self.tangent_pfs = {}

        self.risk_field_map = {
            self._portfolio_vol: "variance",
            self._portfolio_var: "var",
            self._portfolio_cvar: "cvar",
            self._portfolio_max_drawdown: "maxdd"
        }



    def _load_data(self) -> DataFrame:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

        stock_dfs = {} # Will store individual stock OHLCV dataframes
        close_prices_for_merge = [] # Will store only close prices Series for merging

        for stock in self.stocks:
            found_stock_file = False
            for fname in os.listdir(data_dir):
                if fname.lower().startswith(stock.lower()):
                    df = pd.read_csv(os.path.join(data_dir, fname), parse_dates=["Date"])
                    df = df.set_index("Date").sort_index() # Set index and sort immediately

                    # Ensure OHLCV exists for backtrader
                    for col in ['Open', 'High', 'Low', 'Volume']:
                        if col not in df.columns: df[col] = df['Close'] # Fill missing with Close for simplicity

                    # Store the OHLCV dataframe for backtrader
                    stock_dfs[stock] = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                    # Store the Close price Series for merging into the main portfolio DF
                    close_prices_for_merge.append(df['Close'].rename(stock))
                    found_stock_file = True
                    break
            if not found_stock_file:
                print(f"Warning: Could not find data for stock: {stock}")


        if not close_prices_for_merge:
            print(f"Error: No stock data found for any of the selected stocks: {self.stocks}")
            return pd.DataFrame() # Return empty if no data was loaded at all

        merged_close_df = pd.concat(close_prices_for_merge, axis=1, join='inner')
        merged_close_df = merged_close_df.dropna()

        self.individual_stock_data = stock_dfs # Store individual data for backtrader
        print(f"DEBUG: _load_data - Loaded individual_stock_data for {len(self.individual_stock_data)} stocks. Example: {list(self.individual_stock_data.keys())[:2]}")
        if self.individual_stock_data:
            first_stock_df = next(iter(self.individual_stock_data.values()))
            print(f"DEBUG: First individual stock DF dates: {first_stock_df.index.min()} to {first_stock_df.index.max()}")

        return merged_close_df # Return the merged close prices for pypfopt

    def split_train_test(self, train_months: int = 36, test_months: int = 3):
        """Split the available full data into a train and test partition by months.

        Defaults: train 36 months, test 3 months. The requested months are capped
        by the available history in `self.full_df`.

        Returns (train_df, test_df, used_train_months, used_test_months, total_months).
        """
        if self.full_df is None or self.full_df.empty:
            return pd.DataFrame(), pd.DataFrame(), 0, 0, 0

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
        test_start_date = end - pd.DateOffset(months=req_test)
        train_start_date = test_start_date - pd.DateOffset(months=req_train)

        train_df = self.full_df[(self.full_df.index >= train_start_date) & (self.full_df.index < test_start_date)]
        test_df = self.full_df[self.full_df.index >= test_start_date]

        # Also prepare the individual stock data for backtrader
        train_individual_stock_data = {}
        test_individual_stock_data = {}

        for stock_name, stock_df in self.individual_stock_data.items():
            train_individual_stock_data[stock_name] = stock_df[(stock_df.index >= train_start_date) & (stock_df.index < test_start_date)]
            test_individual_stock_data[stock_name] = stock_df[stock_df.index >= test_start_date]

        self.train_individual_stock_data = train_individual_stock_data
        self.test_individual_stock_data = test_individual_stock_data

        print(f"DEBUG: split_train_test - Train period: {train_start_date.strftime('%Y-%m-%d')} to {test_start_date.strftime('%Y-%m-%d')}")
        print(f"DEBUG: split_train_test - Test period: {test_start_date.strftime('%Y-%m-%d')} to {self.full_df.index.max().strftime('%Y-%m-%d')}")
        print(f"DEBUG: split_train_test - self.test_individual_stock_data has {len(self.test_individual_stock_data)} stocks.")
        if self.test_individual_stock_data:
            first_test_stock_df = next(iter(self.test_individual_stock_data.values()))
            print(f"DEBUG: First test individual stock DF dates: {first_test_stock_df.index.min()} to {first_test_stock_df.index.max()} (rows: {len(first_test_stock_df)})")

        return train_df, test_df, req_train, req_test, total_months

    def use_df(self, df) -> None:
        """Replace working dataframe (used for optimization) and recompute mu/S/rets.

        Keep self.full_df unchanged so backtests operate on the full history.
        """
        self.df = df.copy()
        # if not self.df.empty:
        self.rets = expected_returns.returns_from_prices(self.df.dropna()) # self.df.pct_change().dropna() # Ensure rets is updated
        self.mu, self.S = compute_mu_and_cov(self.df, self.mean_method, self.cov_method, ewm_span=self.ewm_span)


    def _as_series(self, w: dict | np.ndarray) -> np.ndarray:
        if isinstance(w, dict):
            return np.array(list(w.values()), dtype=float)
        elif isinstance(w, np.ndarray):
            return w.astype(float)

    def _as_dict(self, w: np.ndarray) -> dict:
        return {stock: float(weight) for stock, weight in zip(self.stocks, w)}


    def _portfolio_return(self, w: dict | np.ndarray) -> float:
        return float(self._as_series(w) @ self.mu)

    def _portfolio_vol(self, w: dict | np.ndarray) -> float:
        if not isinstance(w, dict):
            w = self._as_dict(w)
        ef_user = EfficientFrontier(self.mu, self.S)
        ef_user.set_weights(w)
        _, user_vol, _ = ef_user.portfolio_performance(verbose=False)
        return float(user_vol)

    def _portfolio_var(self, w: dict | np.ndarray, alpha=0.05) -> float:
        pr = self.rets @ self._as_series(w)
        return float(-np.percentile(pr, alpha * 100))

    def _portfolio_cvar(self, w: dict | np.ndarray, alpha=0.05) -> float:
        pr = self.rets @ self._as_series(w)
        var = np.percentile(pr, alpha * 100)
        return float(-pr[pr <= var].mean())
        # return self.rets[self.rets < var].mean().to_numpy()[0]

    def _portfolio_sharpe(self, w: dict | np.ndarray, risk_free_rate=0.0) -> float:
        """Calculate negative Sharpe ratio (for minimization) using pypfopt"""
        # convert back to pd series so that pypfopt knows tickers in w.
        if not isinstance(w, dict):
            w = self._as_dict(w)
        ef = EfficientFrontier(self.mu, self.S)
        ef.set_weights(w)
        _, _, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
        return float(sharpe)  # Note: sharpe has to be maximised,  so negate for minimisation

    def _portfolio_max_drawdown(self, w: dict | np.ndarray) -> float:
        """Calculate maximum drawdown for given weights"""
        pr = self.rets @ self._as_series(w)
        cumulative = (1 + pr).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())  # Most negative value


    def _optimize_tangent_portfolio(self, rf, risk_fn) -> Portfolio:
        """
        Find the tangent (market) portfolio that maximizes (E[R]-Rf)/Risk.
        Works for any risk function (volatility, CVaR, etc.).
        Note: This doesn't exactly make sense for sharpe, I think.
        """
        num_assets = len(self.mu)
        bounds = tuple((-1, 1) for _ in range(num_assets))
        initial_guess = np.array(num_assets * [1.0 / num_assets])

        try:
            # Objective: maximize (E[R] - Rf)/risk, minimize negative
            def neg_generalized_sharpe(weights):
                port_ret = self._portfolio_return(weights)
                port_risk = risk_fn(weights)
                if port_risk == 0:
                    return np.inf
                return -((port_ret - rf) / port_risk)

            result = minimize(
                neg_generalized_sharpe,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                ],
                options={'maxiter': 500}
            )

            if result.success:
                w_opt = result.x
                ret_opt = self._portfolio_return(w_opt)
                risk_opt = risk_fn(w_opt)
                
                risk_field = self.risk_field_map.get(risk_fn, "variance")

                return Portfolio(**{risk_field: risk_opt}, return_=100 * ret_opt, weights=self._as_dict(w_opt))

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Portfolio(error=f"Tangent optimization failed: {e}")

    def _point_on_cml(self, tangent_pf, risk_fn, rf, target_return=None, target_risk=None) -> Portfolio:
        """
        Compute any point on the Capital Market Line (CML) numerically
        for arbitrary risk measures.
        
        If target_return or target_risk is given, the function tries
        to find the closest achievable combination by searching
        over allocations between the risk-free and tangent portfolios.
        """
        R_t = tangent_pf.return_ / 100
        w_t = np.array(list(tangent_pf.weights.values()))
        risk_field = self.risk_field_map.get(risk_fn)

        # search range for allocation to tangent portfolio (a)
        a_grid = np.linspace(0.2, 1.5, 300)  # allows leverage or short RF
        candidates = []

        for a in a_grid:
            w_mix = a * w_t
            R_mix = rf + a * (R_t - rf)
            # Compute risk numerically using given risk_fn
            risk_mix = risk_fn(w_mix)
            candidates.append((a, R_mix, risk_mix, w_mix))

        candidates = np.array(candidates, dtype=object)

        # selection logic
        if target_return is not None:
            idx = np.argmin(np.abs(candidates[:,1] - target_return))
        elif target_risk is not None:
            idx = np.argmin(np.abs(candidates[:,2] - target_risk))
        else:
            idx = np.argmax(candidates[:,1])  # pick highest return if nothing specified

        a_sel, R_sel, risk_sel, w_sel = candidates[idx]
        w_rf = 1 - a_sel

        weights_dict = {**{k: v for k, v in zip(tangent_pf.weights.keys(), w_sel)}, "R": w_rf}

        return Portfolio(
            weights=weights_dict,
            return_=100 * R_sel,
            **{risk_field: risk_sel},
        )

    def _add_tangent(self, base: Portfolio, risk_fn, rf, target_return = None, target_risk = None) -> Portfolio:
        if rf is None:
            return base

        tangent = self.tangent_pfs.get(risk_fn, None)
        if tangent is None:
            tangent = self._optimize_tangent_portfolio(rf, risk_fn)
            self.tangent_pfs[risk_fn] = tangent

        tangent = self._point_on_cml(tangent, risk_fn, rf, target_return, target_risk)
        if (tangent is not None) and (tangent.error is None):
            base.tangent = tangent
        return base


    def _optimize_mv_for_return(self, target_return, rf=None) -> Portfolio:
        try:
            ef_mv = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            ef_mv.efficient_return(target_return)
            w_mv = ef_mv.clean_weights()
            ret_mv, vol_mv, _ = ef_mv.portfolio_performance()
            base = Portfolio(
            weights=w_mv,
            return_=100*ret_mv,
            variance=vol_mv,
            var=self._portfolio_var(w_mv),
            cvar=self._portfolio_cvar(w_mv),
            sharpe=self._portfolio_sharpe(w_mv),
            maxdd=self._portfolio_max_drawdown(w_mv)
            )
            return self._add_tangent(base, self._portfolio_vol, rf, target_return=target_return)

        except Exception as e:
            return Portfolio(error=str(e))

    def _optimize_cvar_for_return(self, target_return, rf=None) -> Portfolio:
        try:
            ef_c = EfficientCVaR(self.mu, self.rets, weight_bounds=(-1,1))
            ef_c.efficient_return(target_return)
            w_c = ef_c.clean_weights()
            ret_c, cvar_risk = ef_c.portfolio_performance()
            base = Portfolio(
            weights=w_c,
            return_=100*ret_c,
            variance=self._portfolio_vol(w_c),
            var=self._portfolio_var(w_c),
            cvar=cvar_risk,
            sharpe=self._portfolio_sharpe(w_c),
            maxdd=self._portfolio_max_drawdown(w_c)
            )
            return self._add_tangent(base, self._portfolio_cvar, rf, target_return=target_return)

        except Exception as e:
            return Portfolio(error=str(e))

    def _optimize_var_for_return(self, target_return, rf=None) -> Portfolio:
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
                # raise Exception(e)
                last_message = f"optimizer exception: {e}"
                continue

            if result is not None and result.success:
                w_var = result.x
                base = Portfolio(
                weights=self._as_dict(w_var),
                return_=100*self._portfolio_return(w_var),
                variance=self._portfolio_vol(w_var),
                var=self._portfolio_var(w_var),
                cvar=self._portfolio_cvar(w_var),
                sharpe=self._portfolio_sharpe(w_var),
                maxdd=self._portfolio_max_drawdown(w_var)
                )
                return self._add_tangent(base, self._portfolio_var, rf, target_return=target_return)

            else:
                # record last failure message
                last_message = getattr(result, 'message', str(result))

        # If we reach here, all attempts failed. Log for diagnostics and return None.
        print(f"_optimize_var_for_return failed after {attempts} attempts for target_return={target_return}; last message: {last_message}")
        return Portfolio(error=str(f"_optimize_var_for_return failed after {attempts} attempts for target_return={target_return}; last message: {last_message}"))

    def _optimize_sharpe_for_return(self, target_return):
        """Optimize for maximum Sharpe ratio at given return level"""
        num_assets = len(self.mu)
        bounds = tuple((-1, 1) for _ in range(num_assets))
        initial_guess = np.array(num_assets * [1.0 / num_assets])
        
        try:
            result = minimize(
                lambda w: -self._portfolio_sharpe(w),
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                    {'type': 'eq', 'fun': lambda weights: self._portfolio_return(weights) - target_return}
                ],
                options={'maxiter': 500}
            )

            if result.success:
                w_sharpe = result.x
                return Portfolio(
                weights=self._as_dict(w_sharpe),
                return_=100*self._portfolio_return(w_sharpe),
                variance=self._portfolio_vol(w_sharpe),
                var=self._portfolio_var(w_sharpe),
                cvar=self._portfolio_cvar(w_sharpe),
                sharpe=self._portfolio_sharpe(w_sharpe),
                maxdd=self._portfolio_max_drawdown(w_sharpe)
                )
            else:
                return Portfolio(error=f"Sharpe optimization failed: {result.message}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            # raise Exception(e)
            return Portfolio(error=f"Sharpe optimization failed: {e}")

    def _optimize_maxdd_for_return(self, target_return, rf=None) -> Portfolio:
        """Optimize for minimum max drawdown at given return level"""
        num_assets = len(self.mu)
        bounds = tuple((-1, 1) for _ in range(num_assets))
        
        rng = np.random.default_rng()
        attempts = 6
        
        for attempt in range(attempts):
            if attempt == 0:
                initial_guess = np.array(num_assets * [1.0 / num_assets])
            else:
                r = rng.random(num_assets)
                initial_guess = r / r.sum()
            
            try:
                result = minimize(
                    lambda w: -self._portfolio_max_drawdown(w),  # Minimize negative = maximize
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[
                        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                        {'type': 'eq', 'fun': lambda weights: self._portfolio_return(weights) - target_return}
                    ],
                    options={'maxiter': 500}
                )
                
                if result.success:
                    w_maxdd = result.x
                    base = Portfolio(
                    weights=self._as_dict(w_maxdd),
                    return_=100*self._portfolio_return(w_maxdd),
                    variance=self._portfolio_vol(w_maxdd),
                    var=self._portfolio_var(w_maxdd),
                    cvar=self._portfolio_cvar(w_maxdd),
                    sharpe=self._portfolio_sharpe(w_maxdd),
                    maxdd=self._portfolio_max_drawdown(w_maxdd)
                    )
                    return self._add_tangent(base, self._portfolio_max_drawdown, rf, target_return=target_return)

            except Exception:
                continue
        
        return Portfolio(error=f"Max drawdown optimization failed for {target_return}")


    def optimize_max_return_for_volatility(self, target_volatility) -> Portfolio:
        """Find portfolio with maximum return for given volatility constraint."""
        try:
            # First, check the min/max possible volatility
            ef_min = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            ef_min.min_volatility()
            _, min_vol, _ = ef_min.portfolio_performance()

            max_ret_vol = max(p.variance for p in self.mv_frontier_pts)

            print(f"Feasible volatility range: [{min_vol:.4f}, {max_ret_vol:.4f}]")
            print(f"Target volatility: {target_volatility:.4f}")

            if target_volatility < min_vol or target_volatility > max_ret_vol:
                return Portfolio(error=f"Target is outside feasible range: ({min_vol}, {max_ret_vol})")

            ef_max = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            ef_max.efficient_risk(target_volatility)

            w_max = ef_max.clean_weights()
            ret_max, vol_max, _ = ef_max.portfolio_performance()

            return Portfolio(
            weights=w_max,
            return_=100*ret_max,
            variance=vol_max,
            var=self._portfolio_var(w_max),
            cvar=self._portfolio_cvar(w_max),
            sharpe=self._portfolio_sharpe(w_max),
            maxdd=self._portfolio_max_drawdown(w_max)
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Portfolio(error=str(e))

    def optimize_max_return_for_cvar(self, target_cvar, confidence_level=0.95) -> Portfolio:
        """Find portfolio with maximum return for given CVaR constraint."""
        try:
            # Initialize EfficientCVaR optimizer
            ef_min = EfficientCVaR(self.mu, self.rets, beta=confidence_level, weight_bounds=(-1, 1))
            ef_min.min_cvar()
            _, min_cvar = ef_min.portfolio_performance()

            max_cvar = max(p.cvar for p in self.cvar_frontier_pts)

            print(f"Feasible CVaR range: [{min_cvar:.4f}, {max_cvar:.4f}]")
            print(f"Target CVaR: {target_cvar:.4f}")

            # Check feasibility
            if target_cvar < min_cvar or target_cvar > max_cvar:
                return Portfolio(error=f"Target is outside feasible range: ({min_cvar}, {max_cvar})")

            # Optimize for given CVaR constraint
            ef_opt = EfficientCVaR(self.mu, self.rets, beta=confidence_level, weight_bounds=(-1, 1))
            ef_opt.efficient_risk(target_cvar)

            w_opt = ef_opt.clean_weights()
            ret_opt, cvar_opt = ef_opt.portfolio_performance()
            
            return Portfolio(
            weights=w_opt,
            return_=100*float(ret_opt),
            variance=self._portfolio_vol(w_opt),
            var=self._portfolio_var(w_opt),
            cvar=cvar_opt,
            sharpe=self._portfolio_sharpe(w_opt),
            maxdd=self._portfolio_max_drawdown(w_opt)
        )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Portfolio(error=str(e))

    def optimize_max_return_for_var(self, target_var, confidence_level=0.95) -> Portfolio:
        """Find portfolio with maximum return for given VaR constraint."""
        try:
            min_var = min(p.var for p in self.var_frontier_pts)
            max_var = max(p.var for p in self.var_frontier_pts)

            print(f"Feasible VaR range: [{min_var:.4f}, {max_var:.4f}]")
            print(f"Target VaR: {target_var:.4f}")

            if target_var < min_var or target_var > max_var:
                return Portfolio(error=f"Target is outside feasible range: ({min_var}, {max_var})")
        
            # Optimization settings
            n_assets = len(self.mu)
            initial_guess = np.ones(n_assets) / n_assets
            bounds = tuple((-1, 1) for _ in range(n_assets))

            # Objective: maximize return = minimize negative expected return
            def neg_expected_return(weights):
                return -self._portfolio_return(weights)

            # Constraints: sum of weights = 1, VaR <= target
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: target_var - self._portfolio_var(w)}
            ]

            result = minimize(
                neg_expected_return,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'disp': False}
            )

            if not result.success:
                final_var = self._portfolio_var(result.x)
                if abs(final_var - target_var) / target_var < 0.01:  # Within 1%
                    print("Warning: Optimizer didn't converge but solution is close")
                else:
                    raise ValueError(f"Optimization failed: {result.message}")

            w_opt = result.x
            var_opt = self._portfolio_var(result.x)
            ret_opt = self._portfolio_return(result.x)

            return Portfolio(
            weights=self._as_dict(w_opt),
            return_=100*self._portfolio_return(w_opt),
            variance=self._portfolio_vol(w_opt),
            var=self._portfolio_var(w_opt),
            cvar=self._portfolio_cvar(w_opt),
            sharpe=self._portfolio_sharpe(w_opt),
            maxdd=self._portfolio_max_drawdown(w_opt)
        )

        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Portfolio(error=str(e))

    def optimize_max_return_for_sharpe(self, target_sharpe, risk_free_rate=0.0):
        """Find portfolio with maximum return for given Sharpe ratio constraint."""
        try:
            min_sharpe = min(p.sharpe for p in self.sharpe_frontier_pts)
            max_sharpe = max(p.sharpe for p in self.sharpe_frontier_pts)
            
            print(f"Feasible Sharpe range: [{min_sharpe:.4f}, {max_sharpe:.4f}]")
            print(f"Target Sharpe: {target_sharpe:.4f}")
            
            if target_sharpe < min_sharpe or target_sharpe > max_sharpe:
                return Portfolio(error=f"Target is outside feasible range: ({min_sharpe}, {max_sharpe})")
            
            n_assets = len(self.mu)
            initial_guess = np.ones(n_assets) / n_assets
            bounds = tuple((-1, 1) for _ in range(n_assets))
            
            def neg_expected_return(weights):
                return -self._portfolio_return(weights)
            
            def sharpe_constraint(weights):
                return self._portfolio_sharpe(weights) - target_sharpe
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': sharpe_constraint}
            ]
            
            result = minimize(
                neg_expected_return,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                w_opt = result.x
                ret_opt = self._portfolio_return(result.x)
                sharpe_opt = self._portfolio_sharpe(result.x, risk_free_rate)
                
                return Portfolio(
                weights=self._as_dict(w_opt),
                return_=100*self._portfolio_return(w_opt),
                variance=self._portfolio_vol(w_opt),
                var=self._portfolio_var(w_opt),
                cvar=self._portfolio_cvar(w_opt),
                sharpe=self._portfolio_sharpe(w_opt, risk_free_rate),
                maxdd=self._portfolio_max_drawdown(w_opt)
            )
            else:
                return Portfolio(error=f"Sharpe optimisation failed: {result.message}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Portfolio(error=f"Sharpe optimisation failed: {e}")

    def optimize_max_return_for_maxdd(self, target_maxdd) -> Portfolio:
        """Find portfolio with maximum return for given max drawdown constraint."""
        try:
            min_maxdd = min(p.maxdd for p in self.maxdd_frontier_pts)
            max_maxdd = max(p.maxdd for p in self.maxdd_frontier_pts)
            
            print(f"Feasible MaxDD range: [{min_maxdd:.4f}, {max_maxdd:.4f}]")
            print(f"Target MaxDD: {target_maxdd:.4f}")
            
            if target_maxdd < min_maxdd or target_maxdd > max_maxdd:
                return Portfolio(error=f"Target is outside feasible range: ({min_maxdd}, {max_maxdd})")
            
            n_assets = len(self.mu)
            initial_guess = np.ones(n_assets) / n_assets
            bounds = tuple((-1, 1) for _ in range(n_assets))
            
            def neg_expected_return(weights):
                return -self._portfolio_return(weights)
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: self._portfolio_max_drawdown(w) - target_maxdd}
            ]
            
            result = minimize(
                neg_expected_return,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500}
            )
            
            if result.success:
                w_opt = result.x
                ret_opt = self._portfolio_return(result.x)
                maxdd_opt = self._portfolio_max_drawdown(result.x)
                
                return Portfolio(
                weights=self._as_dict(w_opt),
                return_=100*self._portfolio_return(w_opt),
                variance=self._portfolio_vol(w_opt),
                var=self._portfolio_var(w_opt),
                cvar=self._portfolio_cvar(w_opt),
                sharpe=self._portfolio_sharpe(w_opt),
                maxdd=self._portfolio_max_drawdown(w_opt)
            )
            else:
                raise ValueError(f"Optimization failed: {result.message}")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Portfolio(error=str(e))


    def calculate_frontiers(self, rf=None) -> None:
        if self.df.empty or self.mu is None or self.S is None:
            return

        grid = np.linspace(self.mu.min() + 1e-6, self.mu.max() - 1e-6, 100)

        # Map frontier attribute names to the corresponding optimization function
        frontier_map = {
            "mv_frontier_pts": lambda r: self._optimize_mv_for_return(r, rf=rf),
            "cvar_frontier_pts": lambda r: self._optimize_cvar_for_return(r, rf=rf),
            "var_frontier_pts": lambda r: self._optimize_var_for_return(r, rf=rf),
            "sharpe_frontier_pts": lambda r: self._optimize_sharpe_for_return(r), # Special case
            "maxdd_frontier_pts": lambda r: self._optimize_maxdd_for_return(r, rf=rf),
        }

        for attr_name, opt_func in frontier_map.items():
            setattr(
                self,
                attr_name,
                [
                    res for r in grid
                    if (res := opt_func(r)) is not None and getattr(res, "error", None) is None
                ]
            )


    def optimize_user_portfolio(self, ret: bool=True, risk: bool=True, weights:dict | None = None, targetR: float | None =None, rf: float | None = None) -> tuple[OptimizationResultsContainer]:
        opt_for_return, opt_for_risk = OptimizationResultsContainer(), OptimizationResultsContainer()

        # optimised for return
        if ret:
            if (weights is None) and (targetR is None): 
                opt_for_return.variance.error = "Either weights or target return required for return optimisation."
                return opt_for_return, opt_for_risk

            if weights is not None:
                pf_return = self._portfolio_return(weights)
            else:
                pf_return = targetR

            opt_for_return = OptimizationResultsContainer(
                variance = self._optimize_mv_for_return(pf_return, rf),
                var = self._optimize_var_for_return(pf_return, rf),
                cvar = self._optimize_cvar_for_return(pf_return, rf),
                sharpe = self._optimize_sharpe_for_return(pf_return),
                maxdd = self._optimize_maxdd_for_return(pf_return, rf),
            )

        # optimised for risk
        if risk:
            if weights is None:
                opt_for_risk.variance.error = "Weight argument required for risk optimisation"
                return opt_for_return, opt_for_risk

            w = weights
            vol = self._portfolio_vol(w)
            var = self._portfolio_var(w)
            cvar = self._portfolio_cvar(w)
            sharpe = self._portfolio_sharpe(w)
            mdd = self._portfolio_max_drawdown(w)

            opt_for_risk = OptimizationResultsContainer(
                variance = self.optimize_max_return_for_volatility(vol),
                var = self.optimize_max_return_for_cvar(cvar),
                cvar = self.optimize_max_return_for_var(var),
                sharpe = self.optimize_max_return_for_sharpe(sharpe),
                maxdd = self.optimize_max_return_for_maxdd(mdd),
            )

        return opt_for_return, opt_for_risk


    def portfolio_metrics(self, weights: List[dict], df=None) -> List[Portfolio]:
        if df is not None:
            old_df = self.df.copy()
            self.use_df(df)

        metrics = []
        for weight in weights:
            metrics.append(
                Portfolio(
                    weights = weight,
                    variance = self._portfolio_vol(weight),
                    var = self._portfolio_var(weight),
                    cvar = self._portfolio_cvar(weight),
                    sharpe = self._portfolio_sharpe(weight),
                    maxdd = self._portfolio_max_drawdown(weight),
                    return_ = 100*(self._portfolio_return(weight)),
                )
            )

        if df is not None: # restore the old dataframe
            self.use_df(old_df)

        return metrics


    # TODO: move this to test.py. dataframes can be passed as an argument, as they are the only self parameter used.
    def run_backtest_backtrader(self, weights: dict, start_date=None, end_date=None):
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        cerebro.broker.set_shortcash(True)


        # Add data feeds
        data_to_use = self.test_individual_stock_data
        for stock_name, df in data_to_use.items():
            if df.empty:
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if start_date and end_date:
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            if df.empty:
                continue

            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.reset_index(names=['datetime'])
            data = bt.feeds.PandasData(
                dataname=df,
                datetime='datetime',
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=-1
            )
            cerebro.adddata(data, name=stock_name)

        cerebro.addstrategy(BuyAndHold, weights=weights)

        # Run
        strategies = cerebro.run()
        strat = strategies[0]

        # Return as DataFrame
        values_df = pd.DataFrame(strat.values, columns=['date', 'value']).set_index('date')
        return values_df



def compute_test_metrics(pfo: portfolio, test_df: pd.Series, train_results: OptimizationResultsContainer) -> OptimizationResultsContainer:
    test_results = OptimizationResultsContainer()

    for metric_name, pf in train_results.items():
        if isinstance(pf, Portfolio) and pf.weights is not None:
            test_portfolio = pfo.portfolio_metrics([pf.weights], df=test_df)[0] # Compute metrics for this portfolio’s weights
            setattr(test_results, metric_name, test_portfolio)
        else:
            setattr(test_results, metric_name, pf)

    return test_results
