#!/bin/bash
# Deploy Flask app as static site for GitHub Pages using Frozen-Flask
# NOTE: Only static content can be hosted on GitHub Pages. This script uses Frozen-Flask to export static files.

pip install Flask Frozen-Flask

# Generate static site using freeze.py
python freeze.py

# Move build to docs for GitHub Pages
rm -rf docs
mv build docs

echo "Static site generated in ./docs. Push to GitHub and set Pages source to /docs folder."
