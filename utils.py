from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import os

__all__ = [
    "format_weights_mv",
    "format_weights_risk",
    "format_weights_return",
    "Portfolio",
    "OptimizationResultsContainer",
    "IndexContext",
]

class IndexContext(BaseModel):
    stats: Optional[Any] = None
    graph_htmls: Optional[List[str]] = None
    backtest_plots_data: Optional[Any] = None
    used_train: Optional[Any] = None
    used_test: Optional[Any] = None
    total_months: Optional[int] = None
    stocks: Optional[List[str]] = None
    riskvalue: Optional[float] = None
    risktype: Optional[str] = None
    returnval: Optional[float] = None
    error: Optional[str] = None
    weights: Optional[List[Any]] = None

    mean_method: str
    cov_method: str
    stock_options: List[str]
    train_months: int
    test_months: int
    risk_free: Optional[float] = None
    ewm_span: int



class Portfolio(BaseModel):
    # Note: None default only exists so that indexing OptimizationResultsContainer never fails.
    error: Optional[str] = None

    weights: Optional[dict] = None
    return_: Optional[float] = None
    variance: Optional[float] = None
    var: Optional[float] = None
    cvar: Optional[float] = None
    sharpe: Optional[float] = None
    maxdd: Optional[float] = None

    tangent: Optional["Portfolio"] = None

# stores Portfolio optimised for each risk metric 
class OptimizationResultsContainer(BaseModel):
    variance: Optional[Portfolio] = Field(default_factory=Portfolio)
    var: Optional[Portfolio] = Field(default_factory=Portfolio)
    cvar: Optional[Portfolio] = Field(default_factory=Portfolio)
    sharpe: Optional[Portfolio] = Field(default_factory=Portfolio)
    maxdd: Optional[Portfolio] = Field(default_factory=Portfolio)
 
    def items(self):
        return {
            "variance": self.variance,
            "var": self.var,
            "cvar": self.cvar,
            "sharpe": self.sharpe,
            "maxdd": self.maxdd
        }.items()


def extract_params(request, data_directory) -> IndexContext:
    files = os.listdir(data_directory)
    stocks_list = [f.split('.')[0] for f in files if os.path.isfile(os.path.join(data_directory, f))]

    # Collect form data
    stocks = request.form.getlist('stock')
    try: rf = float(request.form.getlist('risk_free'))
    except: rf = 7.0 # later make a better default? fetch an actual value?

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

    ctx = IndexContext(
        stocks=stocks, 
        stock_options=stocks_list, 
        mean_method=mean_method, 
        cov_method=cov_method,
        train_months=train_months, 
        test_months=test_months,
        risk_free=rf,
        ewm_span=ewm_span
    )

    # basic form validation
    if len(stocks) != len(set(stocks)):
        ctx.error = "Your portfolio stocks must be unique."

    # check for unique stocks
    try: 
        weight_percents = request.form.getlist('weight')
        if len(weight_percents) > 0:
            ctx.weights = weight_percents
    except: pass

    try:
        ctx.risktype = request.form.get("risk_metric")
        ctx.riskvalue = float(request.form.get("risk_value"))
    except: pass

    try:
        ctx.returnval = float(request.form.get("return_value"))
    except: pass


    return ctx


def format_weights_mv(stats: Portfolio, opt_risk: OptimizationResultsContainer, opt_return: OptimizationResultsContainer):

    # create desired table
    table = {
        "return": stats.return_,
        "tables": [
        { # optimised return table
            "Variance": {"User Portfolio": stats.variance, "Optimised Portfolio": opt_return.variance.variance},
            "VaR": {"User Portfolio": stats.var, "Optimised Portfolio": opt_return.var.var},
            "CVaR": {"User Portfolio": stats.cvar, "Optimised Portfolio": opt_return.cvar.cvar},
            "Sharpe Ratio": {"User Portfolio": stats.sharpe, "Optimised Portfolio": opt_return.sharpe.sharpe},
            "Max Drawdown": {"User Portfolio": stats.maxdd, "Optimised Portfolio": opt_return.maxdd.maxdd},
        },
        { # optimised risk portfolio
            "Optimized Variance": {"Return": opt_risk.variance.return_, "Risk": opt_risk.variance.variance},
            "Optimized VaR": {"Return": opt_risk.var.return_, "Risk": opt_risk.var.var},
            "Optimized CVaR": {"Return": opt_risk.cvar.return_, "Risk": opt_risk.cvar.cvar},
            "Optimized Sharpe Ratio": {"Return": opt_risk.sharpe.return_, "Risk": opt_risk.sharpe.sharpe},
            "Optimized Max Drawdown": {"Return": opt_risk.maxdd.return_, "Risk": opt_risk.maxdd.maxdd},
        },
        ]
    }
    return table

def format_weights_risk(opt_risk: Portfolio):
    # create desired table
    table = {
        "return": opt_risk.return_,
        "tables": [
        { # optimised portfolio table
            "Variance": {"Optimised Portfolio": opt_risk.variance},
            "VaR": {"Optimised Portfolio": opt_risk.var},
            "CVaR": {"Optimised Portfolio": opt_risk.cvar},
            "Sharpe Ratio": {"Optimised Portfolio": opt_risk.sharpe},
            "Max Drawdown": {"Optimised Portfolio": opt_risk.maxdd},
        },
        ]
    }
    return table

def format_weights_return(opt_return: OptimizationResultsContainer):
    # create desired table
    table = {
        "return": None,
        "tables": [
        { # optimised portfolio table
            "Variance": {"Optimised Portfolio": opt_return.variance.variance, "Return": opt_return.variance.return_ },
            "VaR": {"Optimised Portfolio": opt_return.var.var, "Return": opt_return.var.return_ },
            "CVaR": {"Optimised Portfolio": opt_return.cvar.cvar, "Return": opt_return.cvar.return_ },
            "Sharpe Ratio": {"Optimised Portfolio": opt_return.sharpe.sharpe, "Return": opt_return.sharpe.return_ },
            "Max Drawdown": {"Optimised Portfolio": opt_return.maxdd.maxdd, "Return": opt_return.maxdd.return_ },
        },
        ]
    }
    return table
