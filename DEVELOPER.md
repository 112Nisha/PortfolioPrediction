# PortfolioPrediction Developer Documentation

## Project Structure
```
PortfolioPrediction/
├── app.py                # Main Flask app, routes and logic
├── predict.py            # Portfolio stats calculation (abstracted)
├── graph.py              # Graph generation (abstracted)
├── freeze.py             # Static site generator using Frozen-Flask
├── deploy.sh             # Deployment script for GitHub Pages
├── templates/            # Jinja2 HTML templates
│   ├── base.html         # Shared layout (header, nav, footer)
│   ├── index.html        # Home page
│   ├── about.html        # About page
│   └── info.html         # Info page
├── static/               # Static assets (CSS, images, generated graphs)
│   ├── style.css         # Main stylesheet
│   └── portfolio_frontier.png # Example graph (generated)
├── README.md             # User documentation
└── DEVELOPER.md          # Developer documentation
```

## How It Works
- **app.py**: Defines all routes. Handles form submission, calls `predict.py` and `graph.py`, and passes results to templates.
- **predict.py**: Implement `compute_portfolio_stats(stocks, weights)` to calculate mean, variance, and suggestions.
- **graph.py**: Implement `generate_frontier_graph(stocks, weights)` to generate and save a graph image (e.g., `static/portfolio_frontier.png`).
- **Templates**: Use Jinja2 inheritance (`base.html`) for consistent layout. All pages extend `base.html`.
- **Static Files**: Place CSS, images, and generated graphs in `static/`.
- **freeze.py**: Freezes the site for static hosting. Yields `/` for the home page.
- **deploy.sh**: Installs dependencies, runs `freeze.py`, and moves output to `docs/` for GitHub Pages.

## Adding Pages
1. Create a new template in `templates/` (e.g., `newpage.html`).
2. Add a route in `app.py`:
   ```python
   @app.route('/newpage')
   def newpage():
       return render_template('newpage.html')
   ```
3. Add a link to the nav in `base.html` if needed.

## Customizing Portfolio Logic
- Edit `predict.py` and `graph.py` to implement your own math, data fetching, and graphing.
- Use any Python libraries (e.g., pandas, matplotlib) as needed.

## Styling
- Edit `static/style.css` for colors, layout, and responsive design.
- The header/footer use IIIT Hyderabad’s blue gradient for branding.

## Static Site Generation
- Run `python freeze.py` to generate the static site in `build/`.
- Move or rename `build/` to `docs/` for GitHub Pages.

## Deployment
- Push the `docs/` folder to your GitHub repository.
- Set GitHub Pages source to `/docs` in repository settings.

## Tips
- Use Flask’s `url_for('static', filename='...')` for static assets in templates.
- Use Jinja2 blocks for flexible template inheritance.
- Keep placeholder content and update as needed.

## Contact
- For questions, contact the lead developer or IIIT Hyderabad faculty.
