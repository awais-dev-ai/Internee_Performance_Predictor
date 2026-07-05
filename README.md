# Intern Performance Predictor
Machine learning model to predict intern performance.

## Project Layout

- `src/` - reusable ML and preprocessing code
- `ui/` - Flask application, templates, and static files
- `scripts/` - training and utility scripts
- `tests/` - automated tests
- `notebooks/` - exploratory analysis and reporting
- `data/`, `models/`, `reports/` - generated artifacts

## Run the pipeline

```bash
python main.py
```

## Run the Flask UI

```bash
flask --app app run
```

## Production entrypoint

```bash
waitress-serve wsgi:app
```