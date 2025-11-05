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
    table = {
        "return": stats.return_,
        "tables": [
            {
                "User Portfolio": {
                    "Weights": stats.weights,
                    "Return": round(stats.return_, 2),
                    "Variance": round(stats.variance, 2),
                    "VaR": round(stats.var, 2),
                    "CVaR": round(stats.cvar, 2),
                    "Sharpe Ratio": round(stats.sharpe, 2),
                    "Max Drawdown": round(stats.maxdd, 2),
                    "Description": "Your current portfolio"
                },
                "Variance-Optimized": {
                    "Weights": opt_return.variance.weights,
                    "Return": f"<strong>{opt_return.variance.return_}</strong>",
                    "Variance": f"<strong>{opt_return.variance.variance}</strong>",
                    "VaR": opt_return.variance.var,
                    "CVaR": opt_return.variance.cvar,
                    "Sharpe Ratio": opt_return.variance.sharpe,
                    "Max Drawdown": opt_return.variance.maxdd,
                    "Description": "Minimized variance for same return"
                },
                "VaR-Optimized": {
                    "Weights": opt_return.var.weights,
                    "Return": f"<strong>{opt_return.var.return_}</strong>",
                    "Variance": opt_return.var.variance,
                    "VaR": f"<strong>{opt_return.var.var}</strong>",
                    "CVaR": opt_return.var.cvar,
                    "Sharpe Ratio": opt_return.var.sharpe,
                    "Max Drawdown": opt_return.var.maxdd,
                    "Description": "Minimized VaR for same return"
                },
                "CVaR-Optimized": {
                    "Weights": opt_return.cvar.weights,
                    "Return": f"<strong>{opt_return.cvar.return_}</strong>",
                    "Variance": opt_return.cvar.variance,
                    "VaR": opt_return.cvar.var,
                    "CVaR": f"<strong>{opt_return.cvar.cvar}</strong>",
                    "Sharpe Ratio": opt_return.cvar.sharpe,
                    "Max Drawdown": opt_return.cvar.maxdd,
                    "Description": "Minimized CVaR for same return"
                },
                "Sharpe-Optimized": {
                    "Weights": opt_return.sharpe.weights,
                    "Return": f"<strong>{opt_return.sharpe.return_}</strong>",
                    "Variance": opt_return.sharpe.variance,
                    "VaR": opt_return.sharpe.var,
                    "CVaR": opt_return.sharpe.cvar,
                    "Sharpe Ratio": f"<strong>{opt_return.sharpe.sharpe}</strong>",
                    "Max Drawdown": opt_return.sharpe.maxdd,
                    "Description": "Maximized Sharpe ratio for same return"
                },
                "MaxDD-Optimized": {
                    "Weights": opt_return.maxdd.weights,
                    "Return": f"<strong>{opt_return.maxdd.return_}</strong>",
                    "Variance": opt_return.maxdd.variance,
                    "VaR": opt_return.maxdd.var,
                    "CVaR": opt_return.maxdd.cvar,
                    "Sharpe Ratio": opt_return.maxdd.sharpe,
                    "Max Drawdown": f"<strong>{opt_return.maxdd.maxdd}</strong>",
                    "Description": "Minimized max drawdown for same return"
                },
                "Max Return (Variance)": {
                    "Weights": opt_risk.variance.weights,
                    "Return": f"<strong>{opt_risk.variance.return_}</strong>",
                    "Variance": f"<strong>{opt_risk.variance.variance}</strong>",
                    "VaR": opt_risk.variance.var,
                    "CVaR": opt_risk.variance.cvar,
                    "Sharpe Ratio": opt_risk.variance.sharpe,
                    "Max Drawdown": opt_risk.variance.maxdd,
                    "Description": "Max return for user's variance level"
                },
                "Max Return (VaR)": {
                    "Weights": opt_risk.var.weights,
                    "Return": f"<strong>{opt_risk.var.return_}</strong>",
                    "Variance": opt_risk.var.variance,
                    "VaR": f"<strong>{opt_risk.var.var}</strong>",
                    "CVaR": opt_risk.var.cvar,
                    "Sharpe Ratio": opt_risk.var.sharpe,
                    "Max Drawdown": opt_risk.var.maxdd,
                    "Description": "Max return for user's VaR level"
                },
                "Max Return (CVaR)": {
                    "Weights": opt_risk.cvar.weights,
                    "Return": f"<strong>{opt_risk.cvar.return_}</strong>",
                    "Variance": opt_risk.cvar.variance,
                    "VaR": opt_risk.cvar.var,
                    "CVaR": f"<strong>{opt_risk.cvar.cvar}</strong>",
                    "Sharpe Ratio": opt_risk.cvar.sharpe,
                    "Max Drawdown": opt_risk.cvar.maxdd,
                    "Description": "Max return for user's CVaR level"
                },
                "Max Return (Sharpe)": {
                    "Weights": opt_risk.sharpe.weights,
                    "Return": f"<strong>{opt_risk.sharpe.return_}</strong>",
                    "Variance": opt_risk.sharpe.variance,
                    "VaR": opt_risk.sharpe.var,
                    "CVaR": opt_risk.sharpe.cvar,
                    "Sharpe Ratio": f"<strong>{opt_risk.sharpe.sharpe}</strong>",
                    "Max Drawdown": opt_risk.sharpe.maxdd,
                    "Description": "Max return for user's Sharpe ratio level"
                },
                "Max Return (MaxDD)": {
                    "Weights": opt_risk.maxdd.weights,
                    "Return": f"<strong>{opt_risk.maxdd.return_}</strong>",
                    "Variance": opt_risk.maxdd.variance,
                    "VaR": opt_risk.maxdd.var,
                    "CVaR": opt_risk.maxdd.cvar,
                    "Sharpe Ratio": opt_risk.maxdd.sharpe,
                    "Max Drawdown": f"<strong>{opt_risk.maxdd.maxdd}</strong>",
                    "Description": "Max return for user's max drawdown level"
                }
            }
        ]
    }
    return table

def format_weights_risk(opt_risk):
    def safe(pf, field, digits=2):
        if pf is None:
            return "N/A"
        val = getattr(pf, field, None)
        if val is None:
            return "N/A"
        try:
            return round(float(val), digits)
        except Exception:
            return val

    # Mapping helpers to extract portfolios from a container (or fallback to single Portfolio)
    if isinstance(opt_risk, OptimizationResultsContainer):
        p_variance = opt_risk.variance
        p_var = opt_risk.var
        p_cvar = opt_risk.cvar
        p_sharpe = opt_risk.sharpe
        p_maxdd = opt_risk.maxdd
        # there's no explicit "user" portfolio inside a container; pick the first non-empty portfolio as user
        user_pf = next((p for p in (p_variance, p_var, p_cvar, p_sharpe, p_maxdd) if p and (p.weights or p.return_ or p.variance)), Portfolio())
    else:
        # opt_risk is a Portfolio -> treat it as the user portfolio; use it for all optimized columns if no container available
        user_pf = opt_risk
        p_variance = opt_risk
        p_var = opt_risk
        p_cvar = opt_risk
        p_sharpe = opt_risk
        p_maxdd = opt_risk

    table = {
        "return": safe(user_pf, "return_", 2) if user_pf else "N/A",
        "tables": [
            {
                "User Portfolio": {
                    "Weights": user_pf.weights or {},
                    "Return": safe(user_pf, "return_"),
                    "Variance": safe(user_pf, "variance"),
                    "VaR": safe(user_pf, "var"),
                    "CVaR": safe(user_pf, "cvar"),
                    "Sharpe Ratio": safe(user_pf, "sharpe"),
                    "Max Drawdown": safe(user_pf, "maxdd"),
                    "Description": "Your current portfolio"
                },
                "Variance-Optimized": {
                    "Weights": p_variance.weights or {},
                    "Return": f"<strong>{safe(p_variance, 'return_')}</strong>",
                    "Variance": f"<strong>{safe(p_variance, 'variance')}</strong>",
                    "VaR": safe(p_variance, "var"),
                    "CVaR": safe(p_variance, "cvar"),
                    "Sharpe Ratio": safe(p_variance, "sharpe"),
                    "Max Drawdown": safe(p_variance, "maxdd"),
                    "Description": "Minimized variance / Max return per variance"
                },
                "VaR-Optimized": {
                    "Weights": p_var.weights or {},
                    "Return": f"<strong>{safe(p_var, 'return_')}</strong>",
                    "Variance": safe(p_var, "variance"),
                    "VaR": f"<strong>{safe(p_var, 'var')}</strong>",
                    "CVaR": safe(p_var, "cvar"),
                    "Sharpe Ratio": safe(p_var, "sharpe"),
                    "Max Drawdown": safe(p_var, "maxdd"),
                    "Description": "Minimized VaR / Max return per VaR"
                },
                "CVaR-Optimized": {
                    "Weights": p_cvar.weights or {},
                    "Return": f"<strong>{safe(p_cvar, 'return_')}</strong>",
                    "Variance": safe(p_cvar, "variance"),
                    "VaR": safe(p_cvar, "var"),
                    "CVaR": f"<strong>{safe(p_cvar, 'cvar')}</strong>",
                    "Sharpe Ratio": safe(p_cvar, "sharpe"),
                    "Max Drawdown": safe(p_cvar, "maxdd"),
                    "Description": "Minimized CVaR / Max return per CVaR"
                },
                "Sharpe-Optimized": {
                    "Weights": p_sharpe.weights or {},
                    "Return": f"<strong>{safe(p_sharpe, 'return_')}</strong>",
                    "Variance": safe(p_sharpe, "variance"),
                    "VaR": safe(p_sharpe, "var"),
                    "CVaR": safe(p_sharpe, "cvar"),
                    "Sharpe Ratio": f"<strong>{safe(p_sharpe, 'sharpe')}</strong>",
                    "Max Drawdown": safe(p_sharpe, "maxdd"),
                    "Description": "Maximized Sharpe / Max return per Sharpe"
                },
                "MaxDD-Optimized": {
                    "Weights": p_maxdd.weights or {},
                    "Return": f"<strong>{safe(p_maxdd, 'return_')}</strong>",
                    "Variance": safe(p_maxdd, "variance"),
                    "VaR": safe(p_maxdd, "var"),
                    "CVaR": safe(p_maxdd, "cvar"),
                    "Sharpe Ratio": safe(p_maxdd, "sharpe"),
                    "Max Drawdown": f"<strong>{safe(p_maxdd, 'maxdd')}</strong>",
                    "Description": "Minimized Max Drawdown / Max return per MaxDD"
                }
            }
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
