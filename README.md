# PortfolioPrediction

A Flask web app for portfolio mean-variance analysis and rebalancing suggestions.

## Features
- Web portal for users to input their stock choices and weights
- Computes historical mean and variance for the portfolio (abstracted)
- Displays the mean-variance frontier graph (abstracted)
- Suggests portfolio rebalancing for higher mean and/or lower variance (abstracted)
- Ready for static deployment to GitHub Pages

## Usage
1. **Install dependencies**
   ```bash
   pip install Flask Frozen-Flask
   ```
2. **Run the app locally**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000` in your browser.

3. **Deploy to GitHub Pages**
    - Generate the static site:
       ```bash
       python freeze.py
       ```
    - Or run the deploy script:
       ```bash
       bash deploy.sh
       ```
    - Push the generated `build` folder to your GitHub repository.
    - Set GitHub Pages source to `/build` in repository settings.

## Customization
- Add your prediction logic in `predict.py`
- Add your graph generation logic in `graph.py`

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
