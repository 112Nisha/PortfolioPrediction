import os
import numpy as np
import pandas as pd
from functools import reduce
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
import backtrader as bt


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
    def __init__(self, stocks, user_weights=None):
        self.stocks = stocks

        if user_weights is not None:
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

        self.opt_variance = False
        self.opt_var = False
        self.opt_cvar = False
        self.opt_max_return = False
        self.pf_metrics = {}

    def _load_data(self):
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

    def use_df(self, df):
        """Replace working dataframe (used for optimization) and recompute mu/S/rets.

        Keep self.full_df unchanged so backtests operate on the full history.
        """
        self.df = df.copy()
        if not self.df.empty:
            self.rets = self.df.pct_change().dropna() # Ensure rets is updated
            self.mu = expected_returns.mean_historical_return(self.df)
            self.S = risk_models.sample_cov(self.df)
        else:
            self.rets, self.mu, self.S = None, None, None

    def _as_series(self, w):
        index = self.mu.index
        return pd.Series(w, index=index).sort_index().round(2)


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


    def optimize_max_return_for_volatility(self, target_volatility):
        """Find portfolio with maximum return for given volatility constraint."""
        try:
            # First, check the min/max possible volatility
            ef_min = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            ef_min.min_volatility()
            _, min_vol, _ = ef_min.portfolio_performance()

            max_ret_vol = max(risk for risk, _, _ in self.mv_frontier_pts)

            print(f"Feasible volatility range: [{min_vol:.4f}, {max_ret_vol:.4f}]")
            print(f"Target volatility: {target_volatility:.4f}")

            if target_volatility < min_vol or target_volatility > max_ret_vol:
                self.opt_max_return = f"Target is outside feasible range: ({min_vol}, {max_ret_vol})"
                return None

            ef_max = EfficientFrontier(self.mu, self.S, weight_bounds=(-1,1))
            ef_max.efficient_risk(target_volatility)

            w_max = self._as_series(ef_max.clean_weights())
            ret_max, vol_max, _ = ef_max.portfolio_performance()
            self.opt_max_return = (vol_max, 100*ret_max, w_max)

            return (vol_max, 100*ret_max, w_max)
        except Exception as e:
            self.opt_max_return = e
            import traceback
            traceback.print_exc()
            return None

    def optimize_max_return_for_cvar(self, target_cvar, confidence_level=0.95):
        """Find portfolio with maximum return for given CVaR constraint."""
        try:
            # Initialize EfficientCVaR optimizer
            ef_min = EfficientCVaR(self.mu, self.rets, beta=confidence_level, weight_bounds=(-1, 1))
            ef_min.min_cvar()
            _, min_cvar = ef_min.portfolio_performance()

            max_cvar = max(risk for risk, _, _ in self.cvar_frontier_pts)

            print(f"Feasible CVaR range: [{min_cvar:.4f}, {max_cvar:.4f}]")
            print(f"Target CVaR: {target_cvar:.4f}")

            # Check feasibility
            if target_cvar < min_cvar or target_cvar > max_cvar:
                self.opt_max_return = f"Target is outside feasible range: ({min_cvar}, {max_cvar})"
                return None

            # Optimize for given CVaR constraint
            ef_opt = EfficientCVaR(self.mu, self.rets, beta=confidence_level, weight_bounds=(-1, 1))
            ef_opt.efficient_risk(target_cvar)

            w_opt = self._as_series(ef_opt.clean_weights())
            ret_opt, cvar_opt = ef_opt.portfolio_performance()

            self.opt_max_return = (cvar_opt, 100 * ret_opt, w_opt)
            return (cvar_opt, 100 * ret_opt, w_opt)

        except Exception as e:
            self.opt_max_return = e
            import traceback
            traceback.print_exc()
            return None


    def optimize_max_return_for_var(self, target_var, confidence_level=0.95):
        """Find portfolio with maximum return for given VaR constraint."""
        try:
            min_var = min(risk for risk, _, _ in self.var_frontier_pts)
            max_var = max((risk for risk, _, _ in self.var_frontier_pts))

            print(f"Feasible VaR range: [{min_var:.4f}, {max_var:.4f}]")
            print(f"Target VaR: {target_var:.4f}")

            if target_var < min_var or target_var > max_var:
                self.opt_max_return = f"Target is outside feasible range: ({min_var}, {max_var})"
                return None
        
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

            w_opt = self._as_series(dict(zip(self.mu.index, result.x)))
            var_opt = self._portfolio_var(result.x)
            ret_opt = self._portfolio_return(result.x)

            self.opt_max_return_var = (var_opt, 100 * ret_opt, w_opt)
            return (var_opt, 100 * ret_opt, w_opt)

        
        except Exception as e:
            self.opt_max_return = e
            import traceback
            traceback.print_exc()
            return None
 



    def calculate_frontiers(self, user_weights=True):
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
     
        if user_weights:
            pf_return = self._portfolio_return(self.user_weights)
            self.opt_variance = self._optimize_mv_for_return(pf_return)
            self.opt_var = self._optimize_var_for_return(pf_return)
            self.opt_cvar = self._optimize_cvar_for_return(pf_return)


    def portfolio_metrics(self, df=None):
        weights = self.user_weights
        if weights is None or self.mu is None:
            return None

        if df is not None:
            old_df = self.df.copy()
            self.use_df(df)

        returns = self._portfolio_return(weights)

        # calculate user's portfolio volatility
        ef_user = EfficientFrontier(self.mu, self.S)
        ef_user.set_weights(dict(zip(self.stocks, weights)))
        _, user_vol, _ = ef_user.portfolio_performance(verbose=False)

        pf_metrics = {
            "return": returns,
            "user_variance": user_vol,
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


    # TODO: move this to test.py. dataframes can be passed as an argument, as they are the only self parameter used.
    def run_backtest_backtrader(self, weights_dict, start_date=None, end_date=None):
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)
        cerebro.broker.set_shortcash(True)

        cerebro.addstrategy(BuyAndHold, weights=weights_dict)

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

        # Run
        strategies = cerebro.run()
        strat = strategies[0]

        # Return as DataFrame
        values_df = pd.DataFrame(strat.values, columns=['date', 'value']).set_index('date')
        return values_df
