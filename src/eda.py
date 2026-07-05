"""Functions for exploratory data analysis."""

from __future__ import annotations

import pandas as pd


def data_overview(df: pd.DataFrame) -> dict[str, object]:
	"""Return compact dataset summary statistics."""

	return {
		"shape": df.shape,
		"missing_values": int(df.isna().sum().sum()),
		"columns": list(df.columns),
		"numeric_summary": df.select_dtypes(include="number").describe().to_dict(),
	}


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a numeric correlation matrix."""

	numeric_df = df.select_dtypes(include="number")
	return numeric_df.corr()