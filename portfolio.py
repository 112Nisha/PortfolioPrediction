import os
import numpy as np
import pandas as pd
from functools import reduce
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
import backtrader as bt

# portfolio.py - new class definition (put this before the portfolio class)

class FixedWeightsStrategy(bt.Strategy):
    """
    A simple strategy that applies fixed weights to a portfolio.
    """
    params = (('weights', None),)

    def __init__(self):
        if self.p.weights is None:
            raise ValueError("Weights must be provided to the FixedWeightsStrategy")

        self.order_targets = {}
        for i, data in enumerate(self.datas):
            stock_name = data._name
            if stock_name not in self.p.weights:
                # Handle cases where a stock might be in data but not in weights
                self.order_targets[stock_name] = 0.0
                print(f"Warning: Stock {stock_name} not found in provided weights, setting target to 0.")
            else:
                self.order_targets[stock_name] = self.p.weights[stock_name]

        self.rebalance_date = None # To control rebalancing frequency if needed


    def next(self):
        if self.rebalance_date is None or self.data.datetime.date() >= self.rebalance_date:
            for i, data in enumerate(self.datas):
                stock_name = data._name
                target_weight = self.order_targets.get(stock_name, 0.0)
                self.order_target_percent(data, target_weight)

            # Rebalance monthly (or adjust as needed)
            current_date = self.data.datetime.date()
            # Set next rebalance date to the first day of the next month
            # This is a simple monthly rebalance. Can be made more sophisticated.
            if current_date.month == 12:
                self.rebalance_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                self.rebalance_date = current_date.replace(month=current_date.month + 1, day=1)

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
            weights = ef_max.efficient_risk(target_volatility)

            w_max = self._as_series(ef_max.clean_weights())
            ret_max, vol_max, _ = ef_max.portfolio_performance()
            self.opt_max_return = (vol_max, 100*ret_max, w_max)

            return (vol_max, 100*ret_max, w_max)
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

        old_df = self.df.copy()
        self.use_df(test_df)

        out = {}

        try:
            user_w = to_array(weights)
            if user_w is not None:
                pr_user = self.rets @ (user_w)
                out['user'] = (1 + pr_user).cumprod()
        except:
            print("Failed to generate backtest for user portfolio")


        try:
            variance_w = to_array(self.opt_variance[2])
            if variance_w is not None:
                pr_var = self.rets @ (variance_w)
                out['variance'] = (1 + pr_var).cumprod()
        except:
            print("Failed to generate backtest for variance opt portfolio")


        try:
            var_w = to_array(self.opt_var[2])
            if var_w is not None:
                pr_v = self.rets @ (var_w)
                out['var'] = (1 + pr_v).cumprod()
        except:
            print("Failed to generate backtest for var opt portfolio")


        try:
            cvar_w = to_array(self.opt_cvar[2])
            if cvar_w is not None:
                pr_c = self.rets @ (cvar_w)
                out['cvar'] = (1 + pr_c).cumprod()
        except:
            print("Failed to generate backtest for cvar opt portfolio")

        self.use_df(old_df)

        return out

    def run_backtest_backtrader(self, weights_dict, start_date=None, end_date=None, plot_filepath="backtest.png", title="Backtest"):
        """
        Runs a backtrader backtest for a given set of weights and saves the plot.
        weights_dict: a dictionary of {stock_name: weight_percentage (0 to 1)}
        """
        cerebro = bt.Cerebro()

        # Set starting cash
        cerebro.broker.setcash(100000.0) # $100,000 starting cash

        # Add our strategy
        cerebro.addstrategy(FixedWeightsStrategy, weights=weights_dict)

        # Add data feeds
        # Use the test_individual_stock_data which was split earlier
        data_to_use = self.test_individual_stock_data # This should be the dictionary of DataFrames

        if not data_to_use:
            print("ERROR: run_backtest_backtrader - No individual stock data available in self.test_individual_stock_data. This should not happen if split_train_test ran correctly.")
            return None

        print(f"DEBUG: run_backtest_backtrader - Attempting to add data for {len(data_to_use)} stocks for period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Filter data by start and end dates if provided
        added_data_count = 0
        for stock_name, df in data_to_use.items():
            if not df.empty:
                # Ensure the DataFrame index is a DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Filter the DataFrame based on start_date and end_date
                df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

                if not df_filtered.empty:
                    print(f"DEBUG: Adding data for {stock_name}. Dates: {df_filtered.index.min().strftime('%Y-%m-%d')} to {df_filtered.index.max().strftime('%Y-%m-%d')}, Rows: {len(df_filtered)}")

                    df_filtered_for_bt = df_filtered.copy()

                    # Ensure all OHLCV columns are lowercase first
                    df_filtered_for_bt.columns = [col.lower() for col in df_filtered_for_bt.columns]

                    # Reset the index, and explicitly name the new datetime column 'datetime'
                    df_filtered_for_bt = df_filtered_for_bt.reset_index(names=['datetime']) # Names the new column 'datetime'


                    data = bt.feeds.PandasData(
                        dataname=df_filtered_for_bt, # Now has 'datetime' as a regular column
                        datetime='datetime',         # Refers to the 'datetime' column
                        open='open',                 # Lowercase column name
                        high='high',                 # Lowercase column name
                        low='low',                   # Lowercase column name
                        close='close',               # Lowercase column name
                        volume='volume',             # Lowercase column name
                        openinterest=-1
                    )
                    cerebro.adddata(data, name=stock_name)
                    added_data_count += 1
                else:
                    print(f"WARNING: No filtered data for {stock_name} within {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}. Original DF dates: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            else:
                print(f"WARNING: Empty DataFrame for stock {stock_name} in self.test_individual_stock_data, skipping.")

        if added_data_count == 0:
            print(f"ERROR: run_backtest_backtrader - No data feeds successfully added to Cerebro for period {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return None
        else:
            print(f"DEBUG: run_backtest_backtrader - Successfully added {added_data_count} data feeds to Cerebro.")

        # Add observers for plotting
        cerebro.addobserver(bt.observers.Broker)
        cerebro.addobserver(bt.observers.Trades)
        cerebro.addobserver(bt.observers.BuySell)
        cerebro.addobserver(bt.observers.Value) # Portfolio value over time

        # Run the engine
        print(f'Running Backtrader Backtest from {start_date} to {end_date} for title: {title}...')
        cerebro.run()

        # Plotting (requires matplotlib and saves to a file)
        try:
            import matplotlib
            matplotlib.use("Agg")  # prevent GUI popup

            figs = cerebro.plot(style='candlestick', numfigs=1, iplot=False)
            plt = matplotlib.pyplot
            plt.close('all')  # close any open figures
            fig = figs[0][0]
            fig.suptitle(title, y=1.02)
            fig.savefig(plot_filepath, bbox_inches='tight', dpi=150)
            plt.close(fig)

            return plot_filepath

        except Exception as e:
            print(f"Error during backtrader plotting: {e}")
            import traceback
            traceback.print_exc()
            return None
