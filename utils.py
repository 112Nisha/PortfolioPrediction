from typing import Dict, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd

__all__ = [
    "format_weights_mv",
    "format_weights_risk",
    "format_weights_return",
    "Portfolio",
    "OptimizationResultsContainer"
]

class Portfolio(BaseModel):
    # Note: None default only exists so that indexing OptimizationResultsContainer never fails.
    success: Optional[str] = None

    weights: Optional[dict] = None
    return_: Optional[float] = None
    variance: Optional[float] = None
    var: Optional[float] = None
    cvar: Optional[float] = None
    sharpe: Optional[float] = None
    maxdd: Optional[float] = None

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


def safe_metric_attr(container, metric: str, attr: str):
    """
    Returns the attribute of a metric safely.
    If the metric is None or a string, returns None.
    """
    pf = getattr(container, metric, None)
    if isinstance(pf, Portfolio):
        return getattr(pf, attr, None)
    return None



def format_weights(w): # Try to delete this
    """
    Normalize/format weight-containing entries in stats for display.
    """
    try:
        if isinstance(w, dict):
            return ", ".join(f"{k}: {v:.4f}" for k, v in w.items())
        elif hasattr(w, 'items'):
            return ", ".join(f"{k}: {v:.4f}" for k, v in w.items())
        else:
            return str(w)
    except Exception:
        return str(w)

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
