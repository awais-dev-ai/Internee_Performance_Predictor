#!/usr/bin/env python3
"""Update README.md metrics table from model_metadata.json.

Usage:
    python scripts/update_readme.py

Reads models/model_metadata.json and replaces the content between
<!-- BEGIN METRICS --> and <!-- END METRICS --> markers in README.md
with the current metrics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_metadata() -> dict:
    json_path = Path("models/model_metadata.json")
    if not json_path.exists():
        raise FileNotFoundError(
            "models/model_metadata.json not found. Run 'python main.py' first."
        )
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_metrics_table(meta: dict) -> str:
    """Build the markdown metrics table from metadata dict."""
    model_name = meta.get("model_name", "Unknown")
    metrics = meta.get("metrics", {})
    classification = meta.get("classification", {})
    per_class = classification.get("per_class", {})
    opt = meta.get("threshold_optimization", {})

    rmse = _fmt(metrics.get("rmse"), 2)
    mae = _fmt(metrics.get("mae"), 2)
    r2 = _fmt(metrics.get("r2"), 3)
    accuracy = _fmt(classification.get("accuracy"), 3)
    balanced_acc = _fmt(classification.get("balanced_accuracy"), 3)
    macro_f1 = _fmt(classification.get("macro_f1"), 3)
    struggle_f1 = _fmt(per_class.get("Struggle", {}).get("f1"), 3)
    average_f1 = _fmt(per_class.get("Average", {}).get("f1"), 3)
    excel_f1 = _fmt(per_class.get("Excel", {}).get("f1"), 3)
    struggle_th = _fmt(opt.get("struggle_threshold"), 0)
    excel_th = _fmt(opt.get("excel_threshold"), 0)

    return f"""<!-- BEGIN METRICS -->
| Metric | Value |
|--------|-------|
| **Model** | `{model_name}` |
| **RMSE** | `{rmse}` |
| **MAE** | `{mae}` |
| **R²** | `{r2}` |
| **Accuracy** | `{accuracy}` |
| **Balanced Accuracy** | `{balanced_acc}` |
| **Macro F1** | `{macro_f1}` |
| **Struggle F1** | `{struggle_f1}` |
| **Average F1** | `{average_f1}` |
| **Excel F1** | `{excel_f1}` |
| **Optimal Struggle Threshold** | `≤ {struggle_th}` |
| **Optimal Excel Threshold** | `≥ {excel_th}` |
<!-- END METRICS -->"""


def _fmt(val, decimals: int) -> str:
    """Format a value with the given number of decimal places."""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def update_readme(new_table: str) -> None:
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    pattern = r"<!-- BEGIN METRICS -->.*?<!-- END METRICS -->"
    if not re.search(pattern, content, flags=re.DOTALL):
        raise RuntimeError(
            "README.md is missing <!-- BEGIN METRICS --> / <!-- END METRICS --> markers. "
            "Please add them around the Results table."
        )

    updated = re.sub(pattern, new_table.strip(), content, flags=re.DOTALL)
    readme_path.write_text(updated, encoding="utf-8")
    print("README.md updated with latest metrics.")


if __name__ == "__main__":
    meta = load_metadata()
    table = build_metrics_table(meta)
    update_readme(table)