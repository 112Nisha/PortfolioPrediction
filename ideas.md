## TO-DOs
- Three separate frontier graphs, not one.
- Plot the point corresponding to the weights they gave + arrow to the point we recommend.
- Add backtesting graph

- Change so that user can enter any stock code and we download.
- Add code to periodically update our data.


## DONE
- Create interactive frontiers for variance,var,cvar

POSTPROCESS WEIGHTS!!!

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
- predict.py: 
- 