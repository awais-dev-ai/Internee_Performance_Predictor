"""Feature importance and model explanation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
	"""Return a sorted feature-importance table for tree-based models."""

	if hasattr(model, "feature_importances_"):
		importances = np.asarray(model.feature_importances_, dtype=float)
	elif hasattr(model, "get_booster"):
		score_map = model.get_booster().get_score(importance_type="weight")
		importances = np.array([float(score_map.get(feature, 0.0)) for feature in feature_names])
	else:
		raise ValueError("Model does not expose feature importances.")

	table = pd.DataFrame({"feature": feature_names, "importance": importances})
	table = table.sort_values("importance", ascending=False, ignore_index=True)
	return table