from pydantic import BaseModel
from portfolio import portfolio
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt import EfficientFrontier, risk_models, expected_returns

class pfo_metrics(BaseModel):
    pfo : portfolio

