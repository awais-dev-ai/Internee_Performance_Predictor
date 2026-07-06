# Intern Performance Predictor

![Python](https://img.shields.io/badge/python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![CI](https://github.com/awais-dev-ai/Internee_Performance_Predictor/workflows/CI/badge.svg)

**Predict intern performance and flag struggling/excellent interns using ML with class imbalance handling.**

## Business Problem

Companies need to identify:
- **Struggling interns** (≤ 39 score) → coaching intervention
- **Excellent interns** (≥ 74 score) → advanced assignments
- **Average interns** (40-73) → standard track

Challenge: In real data, Struggle and Excel interns are rare (~15% each), making them hard to predict accurately.

## Solution

**Multi-layer class imbalance strategy:**
1. **15/70/15 data distribution** — mimics real intern populations
2. **Oversampling with jitter** — duplicates minority samples with small noise
3. **Sample-weighted training** — gives higher weight to Struggle/Excel during training
4. **Stratified splitting** — preserves class proportions in train/test sets
5. **Composite model selection** — balances RMSE + balanced accuracy
6. **Threshold optimization** — grid search finds optimal cutoffs (not hardcoded 40/75)

## Results

| Metric | Value |
|--------|-------|
| **Model** | XGBoost |
| **RMSE** | 4.07 |
| **MAE** | 3.03 |
| **R²** | 0.965 |
| **Accuracy** | 0.927 |
| **Balanced Accuracy** | 0.944 |
| **Macro F1** | 0.938 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| **Struggle** | 0.95 | **1.00** | 0.97 | 56 |
| **Average** | 0.92 | 0.92 | 0.92 | 181 |
| **Excel** | 0.93 | 0.91 | 0.92 | 163 |

**Key insight:** Struggle recall = 1.00 — the model catches every struggling intern.

## Web Interface

![App Screenshot](images/app-screenshot.png)

*The Flask web app lets users input task completion hours, feedback rating, and attendance percentage to get instant predictions. Input validation ensures all fields are filled with valid numeric values within expected ranges.*

## Quick Start

### Option 1: Run Locally (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Train and run the app
python app.py
# Open http://127.0.0.1:5000
```

### Option 2: Run with Docker

```bash
# Build and start
docker-compose up

# Open http://localhost:5000
```

### Option 3: Train Only (CLI)

```bash
python main.py
```

## Features

- **Regression model** predicts continuous performance score (0-100)
- **Classification layer** buckets into Excel/Average/Struggle
- **Class imbalance handling** — sample weights, oversampling, stratified split
- **Threshold optimization** — grid search for best classification cutoffs
- **Explainability** — SHAP values + feature importance
- **Web interface** — Flask form for real-time predictions with input validation (empty field checks, numeric type enforcement, range bounds)
- **Unit tests** — 16 tests covering data, models, preprocessing, and web app

## Project Structure

```
├── .github/workflows/
│   └── ci.yml              # CI/CD: runs tests + training on push
├── .dockerignore
├── Dockerfile              # Container definition
├── docker-compose.yml      # One-command deployment
├── requirements.txt        # Python dependencies
├── app.py                  # Flask entry point
├── main.py                 # Training pipeline (CLI)
├── src/
│   ├── __init__.py
│   ├── data_generation.py     # Synthetic data with 15/70/15 distribution
│   ├── preprocessing.py       # Stratified splitting, validation
│   ├── model_training.py      # Sample weights, composite selection
│   ├── evaluation.py          # Threshold optimization
│   ├── interpretation.py     # Feature importance
│   └── eda.py                 # EDA helpers
├── ui/
│   └── __init__.py            # Flask app factory
├── notebooks/
│   └── Intern_Performance_Analysis.ipynb  # Full analysis with plots
├── tests/                    # 16 unit tests
└── data/                     # Generated datasets
```

## Notebook Visualizations

The notebook ([`Intern_Performance_Analysis.ipynb`](notebooks/Intern_Performance_Analysis.ipynb)) includes:

### 1. Data Distribution
- Histograms of all features and target
- Correlation heatmap
- Scatter plots (feature vs performance)

### 2. Model Comparison
- Actual vs Predicted scatter plots for Random Forest & XGBoost
- Residual analysis (residuals vs predicted, histogram, Q-Q plot)

### 3. Model Interpretation
- **SHAP summary plot** — feature contributions
- **Feature importance bar chart** — built-in sklearn/XGBoost importances

### 4. Classification Results
- Confusion matrix (Excel/Average/Struggle)
- Per-class precision, recall, F1
- Threshold optimization visualization (optional)

## Architecture

### Training Pipeline

```
generate_synthetic_data()
    └─ 15% Struggle + 70% Average + 15% Excel
    └─ Oversample minority classes with Gaussian jitter
       ↓
train_test_split_data(stratify=True)
    └─ Preserves 15/70/15 in both train and test
       ↓
train_candidate_models(use_sample_weights=True)
    └─ Random Forest + XGBoost
    └─ Inverse class frequency weights (Struggle/Excel get higher weight)
       ↓
select_best_model(alpha=0.5)
    └─ Composite: 0.5 * (1 - normalized_rmse) + 0.5 * balanced_accuracy
       ↓
optimize_thresholds(metric="macro_f1")
    └─ Grid search: struggle_range=[30-50], excel_range=[65-85]
    └─ Finds Struggle ≤ 39, Excel ≥ 74
       ↓
save_model_artifacts()  # model + metadata with thresholds
```

### Inference Flow

```
User Input (Flask form)
    ↓  Input validation (empty fields, numeric type, range bounds)
    ↓
prepare_prediction_frame()  # clean + validate
    ↓
model.predict()  # XGBoost
    ↓
classify_performance(thresholds from metadata)
    ↓
Return: score + category (Struggle/Average/Excel)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12 |
| **ML** | XGBoost, Random Forest (scikit-learn) |
| **Web** | Flask 3.0 |
| **Explainability** | SHAP |
| **Testing** | pytest |
| **CI/CD** | GitHub Actions |
| **Deployment** | Docker + Docker Compose |

## Key Design Decisions

### Why oversampling instead of SMOTE?
- **Simpler** — no need for nearest-neighbor computation
- **Effective** — Gaussian jitter creates natural variation
- **Fast** — works well with tree-based models

### Why composite model selection?
- **Balanced** — considers both regression accuracy and classification quality
- **Ensures** — minority classes aren't sacrificed for overall RMSE

### Why grid search for thresholds?
- **Adaptive** — finds optimal cutoffs for your data
- **Business-aware** — optimizes for Macro F1, not just accuracy
- **Transparent** — you can see the exact thresholds used

### Why fixed hyperparameters in production?
- **Fast** — no CV overhead on startup
- **Sufficient** — defaults are already well-tuned from notebook experiments
- **Maintainable** — simpler code, easier to understand

## Business Impact

### Before (Baseline / Unoptimized approach)
- 25/50/25 distribution
- No imbalance handling
- Struggle recall: ~60-70%
- Hardcoded thresholds (40/75)

### After (Optimized pipeline)
- 15/70/15 distribution
- Full imbalance strategy
- **Struggle recall: 100%**
- Optimized thresholds (39/74)

## Future Improvements

- Replace synthetic data with real intern records
- Add SHAP waterfall plots for individual predictions
- Implement model monitoring for data drift
- Add authentication for production deployment
- A/B testing framework for threshold strategies

## License

MIT License — feel free to use this project for learning or as a portfolio piece.

## Author

**Awais Ahmad**  
Email: awaisahmad.dev.ai@gmail.com  
LinkedIn: [linkedin.com/in/awaisahmad-dev-ai](https://www.linkedin.com/in/awaisahmad-dev-ai/)

Built to demonstrate production-grade ML engineering, class-imbalance handling, and full-stack deployment.