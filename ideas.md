## TO-DOs
- short selling results in could not optimise. Issue with code or params?
- maximise return for a given risk

- Add docs.
- Create a maximum limit of 5 stocks and select only from a list of stocks.


## DONE
- Plot the point corresponding to the weights they gave + arrow to the point we recommend.
- Show the metrics for the recommended portfolio properly.
- Make sure graphs work for more than 2 stocks.
- Add backtesting graph
- Three separate frontier graphs, not one.
- Create interactive frontiers for variance,var,cvar
- POSTPROCESS WEIGHTS!!!


## Outputs
1. Efficent frontier, where the point they gave is, where the point we are giving is (arrow)
3. Minimum variance portfolio, possibly with desired return value.
4. Backtesting graphs

Should we include other risk measures??

## Considerations
1. What estimators? -- shrinkage/sample? historical mean/capm?
	Maybe give advanced options that allows the users to decide which one.
2. How often do we update our data? We will need some kind of regular job for this that runs on the server.
3. Data source? Yahoo is slow, has rate limits and is "for personal use only".
4. For the recommendation, do we ask them if they want to preserve return? or do we give the MVP. The former I think. -- optimise with respect to which risk measure? -- advanced option?
5. Should we also show Sharpe ratio
6. We have to do something about how slow it is.

Transaction costs.
Later: include other strategies like the ones prabhas used


## Structure
- portfolio.py: frontier, metric calcs
- graphs.py: all plotting calcs
