# PortfolioPrediction

A Flask web app for portfolio mean-variance analysis and rebalancing suggestions.

## Features
- Web portal for users to input their stock choices and weights
- Computes historical mean and variance for the portfolio (abstracted)
- Displays the mean-variance frontier graph (abstracted)
- Suggests portfolio rebalancing for higher mean and/or lower variance (abstracted)

## Usage
2. **Run the app locally**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

## File Structure
```
PortfolioPrediction/
├── app.py
├── deploy.sh
├── freeze.py
├── graph.py
├── predict.py
├── README.md
├── static
│   └── style.css
└── templates
    └── index.html
```
