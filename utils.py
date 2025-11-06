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
                    "Return": f"<strong>{round(opt_return.variance.return_, 2)}</strong>",
                    "Variance": f"<strong>{round(opt_return.variance.variance, 2)}</strong>",
                    "VaR": round(opt_return.variance.var, 2),
                    "CVaR": round(opt_return.variance.cvar, 2),
                    "Sharpe Ratio": round(opt_return.variance.sharpe, 2),
                    "Max Drawdown": round(opt_return.variance.maxdd, 2),
                    "Description": "Minimized variance for same return"
                },
                "VaR-Optimized": {
                    "Weights": opt_return.var.weights,
                    "Return": f"<strong>{round(opt_return.var.return_, 2)}</strong>",
                    "Variance": round(opt_return.var.variance, 2),
                    "VaR": f"<strong>{round(opt_return.var.var, 2)}</strong>",
                    "CVaR": round(opt_return.var.cvar, 2),
                    "Sharpe Ratio": round(opt_return.var.sharpe, 2),
                    "Max Drawdown": round(opt_return.var.maxdd, 2),
                    "Description": "Minimized VaR for same return"
                },
                "CVaR-Optimized": {
                    "Weights": opt_return.cvar.weights,
                    "Return": f"<strong>{round(opt_return.cvar.return_, 2)}</strong>",
                    "Variance": round(opt_return.cvar.variance, 2),
                    "VaR": round(opt_return.cvar.var, 2),
                    "CVaR": f"<strong>{round(opt_return.cvar.cvar, 2)}</strong>",
                    "Sharpe Ratio": round(opt_return.cvar.sharpe, 2),
                    "Max Drawdown": round(opt_return.cvar.maxdd, 2),
                    "Description": "Minimized CVaR for same return"
                },
                "Sharpe-Optimized": {
                    "Weights": opt_return.sharpe.weights,
                    "Return": f"<strong>{round(opt_return.sharpe.return_, 2)}</strong>",
                    "Variance": round(opt_return.sharpe.variance, 2),
                    "VaR": round(opt_return.sharpe.var, 2),
                    "CVaR": round(opt_return.sharpe.cvar, 2),
                    "Sharpe Ratio": f"<strong>{round(opt_return.sharpe.sharpe, 2)}</strong>",
                    "Max Drawdown": round(opt_return.sharpe.maxdd, 2),
                    "Description": "Maximized Sharpe ratio for same return"
                },
                "MaxDD-Optimized": {
                    "Weights": opt_return.maxdd.weights,
                    "Return": f"<strong>{round(opt_return.maxdd.return_, 2)}</strong>",
                    "Variance": round(opt_return.maxdd.variance, 2),
                    "VaR": round(opt_return.maxdd.var, 2),
                    "CVaR": round(opt_return.maxdd.cvar, 2),
                    "Sharpe Ratio": round(opt_return.maxdd.sharpe, 2),
                    "Max Drawdown": f"<strong>{round(opt_return.maxdd.maxdd, 2)}</strong>",
                    "Description": "Minimized max drawdown for same return"
                },
                "Max Return (Variance)": {
                    "Weights": opt_risk.variance.weights,
                    "Return": f"<strong>{round(opt_risk.variance.return_, 2)}</strong>",
                    "Variance": f"<strong>{round(opt_risk.variance.variance, 2)}</strong>",
                    "VaR": round(opt_risk.variance.var, 2),
                    "CVaR": round(opt_risk.variance.cvar, 2),
                    "Sharpe Ratio": round(opt_risk.variance.sharpe, 2),
                    "Max Drawdown": round(opt_risk.variance.maxdd, 2),
                    "Description": "Max return for user portfolio's variance level"
                },
                "Max Return (VaR)": {
                    "Weights": opt_risk.var.weights,
                    "Return": f"<strong>{round(opt_risk.var.return_, 2)}</strong>",
                    "Variance": round(opt_risk.var.variance, 2),
                    "VaR": f"<strong>{round(opt_risk.var.var, 2)}</strong>",
                    "CVaR": round(opt_risk.var.cvar, 2),
                    "Sharpe Ratio": round(opt_risk.var.sharpe, 2),
                    "Max Drawdown": round(opt_risk.var.maxdd, 2),
                    "Description": "Max return for user portfolio's VaR level"
                },
                "Max Return (CVaR)": {
                    "Weights": opt_risk.cvar.weights,
                    "Return": f"<strong>{round(opt_risk.cvar.return_, 2)}</strong>",
                    "Variance": round(opt_risk.cvar.variance, 2),
                    "VaR": round(opt_risk.cvar.var, 2),
                    "CVaR": f"<strong>{round(opt_risk.cvar.cvar, 2)}</strong>",
                    "Sharpe Ratio": round(opt_risk.cvar.sharpe, 2),
                    "Max Drawdown": round(opt_risk.cvar.maxdd, 2),
                    "Description": "Max return for user portfolio's CVaR level"
                },
                "Max Return (Sharpe)": {
                    "Weights": opt_risk.sharpe.weights,
                    "Return": f"<strong>{round(opt_risk.sharpe.return_, 2)}</strong>",
                    "Variance": round(opt_risk.sharpe.variance, 2),
                    "VaR": round(opt_risk.sharpe.var, 2),
                    "CVaR": round(opt_risk.sharpe.cvar, 2),
                    "Sharpe Ratio": f"<strong>{round(opt_risk.sharpe.sharpe, 2)}</strong>",
                    "Max Drawdown": round(opt_risk.sharpe.maxdd, 2),
                    "Description": "Max return for user portfolio's Sharpe ratio level"
                },
                "Max Return (MaxDD)": {
                    "Weights": opt_risk.maxdd.weights,
                    "Return": f"<strong>{round(opt_risk.maxdd.return_, 2)}</strong>",
                    "Variance": round(opt_risk.maxdd.variance, 2),
                    "VaR": round(opt_risk.maxdd.var, 2),
                    "CVaR": round(opt_risk.maxdd.cvar, 2),
                    "Sharpe Ratio": round(opt_risk.maxdd.sharpe, 2),
                    "Max Drawdown": f"<strong>{round(opt_risk.maxdd.maxdd, 2)}</strong>",
                    "Description": "Max return for user portfolio's max drawdown level"
                }
            }
        ]
    }
    return table

def format_weights_risk(opt_risk, risk_type: str = "variance"):
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
                "Portfolio": {
                    "Weights": user_pf.weights or {},
                    "Return": safe(user_pf, "return_") if risk_type != "return" else f"<strong>{safe(user_pf, "return_")}</strong>",
                    "Variance": safe(user_pf, "variance") if risk_type != "variance" else f"<strong>{safe(user_pf, "variance")}</strong>",
                    "VaR": safe(user_pf, "var") if risk_type != "var" else f"<strong>{safe(user_pf, "var")}</strong>",
                    "CVaR": safe(user_pf, "cvar") if risk_type != "cvar" else f"<strong>{safe(user_pf, "cvar")}</strong>",
                    "Sharpe Ratio": safe(user_pf, "sharpe") if risk_type != "sharpe" else f"<strong>{safe(user_pf, "sharpe")}</strong>",
                    "Max Drawdown": safe(user_pf, "maxdd") if risk_type != "maxdd" else f"<strong>{safe(user_pf, "maxdd")}</strong>",
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
