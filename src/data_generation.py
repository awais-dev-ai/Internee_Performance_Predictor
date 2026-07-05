"""Functions to generate synthetic intern performance data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ("task_completion_hrs", "feedback_rating", "attendance_pct")
TARGET_COLUMN = "performance_score"


CLASS_SPECS = {
    "Struggle": {
        "count": 0.15,
        "task_completion_hrs": (12.0, 20.0, 2.1),
        "feedback_rating": (1.0, 2.6, 0.45),
        "attendance_pct": (50.0, 76.0, 5.5),
        "noise": 3.0,  # Reduced from 5.0 for cleaner signal at tails
    },
    "Average": {
        "count": 0.70,
        "task_completion_hrs": (6.5, 13.5, 2.0),
        "feedback_rating": (2.2, 4.0, 0.55),
        "attendance_pct": (68.0, 95.0, 6.0),
        "noise": 4.0,
    },
    "Excel": {
        "count": 0.15,
        "task_completion_hrs": (2.0, 8.0, 1.7),
        "feedback_rating": (3.4, 5.0, 0.4),
        "attendance_pct": (82.0, 100.0, 4.5),
        "noise": 2.5,  # Reduced from 3.5 for cleaner signal at tails
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


def _oversample_minority(
    df: pd.DataFrame,
    label_column: str,
    target_proportions: dict[str, float],
    rng: np.random.Generator,
    jitter_scale: float = 0.05,
) -> pd.DataFrame:
    """Oversample minority classes by duplicating with small Gaussian jitter.

    This brings minority class proportions closer to the majority class
    without creating exact duplicates, improving model learning at the tails.
    """
    from src.evaluation import classify_performance

    # Assign class labels based on performance score
    labels = classify_performance(df[label_column].values)
    df = df.copy()
    df["_class_label"] = labels

    class_counts = df["_class_label"].value_counts()
    max_count = class_counts.max()

    oversampled_parts = [df]
    for class_name, target_pct in target_proportions.items():
        current_count = class_counts.get(class_name, 0)
        target_count = int(max_count * target_pct / target_proportions.get("Average", 1.0))
        n_to_add = target_count - current_count

        if n_to_add <= 0:
            continue

        class_df = df[df["_class_label"] == class_name].copy()
        n_available = len(class_df)
        if n_available == 0:
            continue

        # Sample with replacement to get desired count
        indices = rng.integers(0, n_available, size=n_to_add)
        synthetic = class_df.iloc[indices].copy()

        # Add small Gaussian jitter to feature columns (not the target)
        for col in FEATURE_COLUMNS:
            col_std = synthetic[col].std() if synthetic[col].std() > 0 else 1.0
            jitter = rng.normal(0, col_std * jitter_scale, size=n_to_add)
            synthetic[col] = (synthetic[col] + jitter).clip(
                *CLASS_SPECS[class_name][col][:2] if class_name in CLASS_SPECS else (0, 100)
            )

        oversampled_parts.append(synthetic)

    result = pd.concat(oversampled_parts, ignore_index=True)
    result.drop(columns=["_class_label"], inplace=True)
    return result


def generate_synthetic_data(
    n_samples: int = 2000,
    seed: int = 42,
    save_path: str | Path | None = None,
    class_balance: bool = True,
    oversample_minority: bool = True,
) -> pd.DataFrame:
    """Generate a realistic synthetic intern performance dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate (default 2000 for richer minority classes).
    seed : int
        Random seed for reproducibility.
    save_path : str or Path, optional
        If provided, save the dataset to this CSV path.
    class_balance : bool
        If True, use CLASS_SPECS proportions (15/70/15). If False, generate uniformly.
    oversample_minority : bool
        If True, oversample Struggle and Excel classes to improve model learning.
    """
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

        # Oversample minority classes to improve model learning
        if oversample_minority:
            # Extract just the count proportions from CLASS_SPECS
            proportions = {label: spec["count"] for label, spec in CLASS_SPECS.items()}
            df = _oversample_minority(
                df,
                label_column=TARGET_COLUMN,
                target_proportions=proportions,
                rng=rng,
                jitter_scale=0.05,
            )
            # Re-assign intern IDs after oversampling
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