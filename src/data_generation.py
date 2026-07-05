"""Functions to generate synthetic intern performance data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("task_completion_hrs", "feedback_rating", "attendance_pct")
TARGET_COLUMN = "performance_score"


CLASS_SPECS = {
	"Struggle": {
		"count": 0.25,
		"task_completion_hrs": (12.0, 20.0, 2.1),
		"feedback_rating": (1.0, 2.6, 0.45),
		"attendance_pct": (50.0, 76.0, 5.5),
		"noise": 5.0,
	},
	"Average": {
		"count": 0.50,
		"task_completion_hrs": (6.5, 13.5, 2.0),
		"feedback_rating": (2.2, 4.0, 0.55),
		"attendance_pct": (68.0, 95.0, 6.0),
		"noise": 4.0,
	},
	"Excel": {
		"count": 0.25,
		"task_completion_hrs": (2.0, 8.0, 1.7),
		"feedback_rating": (3.4, 5.0, 0.4),
		"attendance_pct": (82.0, 100.0, 4.5),
		"noise": 3.5,
	},
}


def _sample_uniform(rng: np.random.Generator, bounds: tuple[float, float], size: int) -> np.ndarray:
	return rng.uniform(bounds[0], bounds[1], size)


def _sample_truncated_normal(
	rng: np.random.Generator,
	mean: float,
	std: float,
	bounds: tuple[float, float],
	size: int,
) -> np.ndarray:
	values = rng.normal(mean, std, size)
	return values.clip(bounds[0], bounds[1])


def _build_class_block(
	rng: np.random.Generator,
	label: str,
	size: int,
	start_index: int,
) -> pd.DataFrame:
	spec = CLASS_SPECS[label]
	task = _sample_truncated_normal(
		rng,
		mean=(spec["task_completion_hrs"][0] + spec["task_completion_hrs"][1]) / 2,
		std=spec["task_completion_hrs"][2],
		bounds=spec["task_completion_hrs"][:2],
		size=size,
	)
	feedback = _sample_truncated_normal(
		rng,
		mean=(spec["feedback_rating"][0] + spec["feedback_rating"][1]) / 2,
		std=spec["feedback_rating"][2],
		bounds=spec["feedback_rating"][:2],
		size=size,
	)
	attendance = _sample_truncated_normal(
		rng,
		mean=(spec["attendance_pct"][0] + spec["attendance_pct"][1]) / 2,
		std=spec["attendance_pct"][2],
		bounds=spec["attendance_pct"][:2],
		size=size,
	)
	noise = rng.normal(0, spec["noise"], size)

	performance_score = (
		(10 - task) * 3.5
		+ feedback * 12
		+ attendance * 0.4
		+ noise
	)
	performance_score = np.clip(performance_score, 0, 100).round(1)

	return pd.DataFrame(
		{
			"intern_id": [f"INT{index:03d}" for index in range(start_index, start_index + size)],
			"task_completion_hrs": task.round(2),
			"feedback_rating": feedback.round(2),
			"attendance_pct": attendance.round(1),
			TARGET_COLUMN: performance_score,
		}
	)


def generate_synthetic_data(
	n_samples: int = 1000,
	seed: int = 42,
	save_path: str | Path | None = None,
	class_balance: bool = True,
) -> pd.DataFrame:
	"""Generate a realistic synthetic intern performance dataset."""

	rng = np.random.default_rng(seed)

	if class_balance:
		counts = {
			label: int(round(spec["count"] * n_samples))
			for label, spec in CLASS_SPECS.items()
		}
		delta = n_samples - sum(counts.values())
		if delta != 0:
			counts["Average"] += delta

		blocks = []
		start_index = 1
		for label in ("Struggle", "Average", "Excel"):
			size = counts[label]
			if size <= 0:
				continue
			blocks.append(_build_class_block(rng, label, size=size, start_index=start_index))
			start_index += size

		df = pd.concat(blocks, ignore_index=True)
		df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
		df["intern_id"] = [f"INT{index:03d}" for index in range(1, len(df) + 1)]
	else:
		intern_ids = [f"INT{index:03d}" for index in range(1, n_samples + 1)]

		task_completion_hrs = rng.normal(8, 3, n_samples).clip(2, 20)
		feedback_rating = rng.uniform(1.0, 5.0, n_samples)
		attendance_pct = rng.normal(85, 10, n_samples).clip(50, 100)
		noise = rng.normal(0, 4, n_samples)

		performance_score = (
			(10 - task_completion_hrs) * 3.5
			+ feedback_rating * 12
			+ attendance_pct * 0.4
			+ noise
		)
		performance_score = np.clip(performance_score, 0, 100).round(1)

		df = pd.DataFrame(
			{
				"intern_id": intern_ids,
				"task_completion_hrs": task_completion_hrs.round(2),
				"feedback_rating": feedback_rating.round(2),
				"attendance_pct": attendance_pct.round(1),
				TARGET_COLUMN: performance_score,
			}
		)

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(save_path, index=False)

	return df


def inject_missing_values(
	df: pd.DataFrame,
	missing_rate: float = 0.05,
	seed: int = 42,
	columns: tuple[str, ...] = FEATURE_COLUMNS + (TARGET_COLUMN,),
) -> pd.DataFrame:
	"""Inject missing values into a copy of a dataframe for robustness tests."""

	if not 0 <= missing_rate < 1:
		raise ValueError("missing_rate must be in the range [0, 1).")

	rng = np.random.default_rng(seed)
	corrupted = df.copy()

	for column in columns:
		if column not in corrupted.columns:
			continue

		mask = rng.random(len(corrupted)) < missing_rate
		corrupted.loc[mask, column] = np.nan

	return corrupted