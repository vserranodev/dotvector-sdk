from pysindy import SINDy, STLSQ, PolynomialLibrary, SmoothedFiniteDifference
from pathlib import Path
import numpy as np
import pandas as pd


def patterns(data, threshold=1e-2, degree=2):
    numeric_data = data.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.dropna(axis=0)
    if numeric_data.empty:
        raise ValueError("No numeric columns to train SINDy.")

    model = SINDy(
        optimizer=STLSQ(threshold=threshold, normalize_columns=True),
        feature_library=PolynomialLibrary(degree=degree),
        differentiation_method=SmoothedFiniteDifference(),
        discrete_time=True,
    )
    model.fit(x=numeric_data.to_numpy(dtype=np.float64), t=1)
    return model.coefficients()

if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parents[1] / "tests" / "demo_data.csv"
    data = pd.read_csv(csv_path)
    coef = patterns(data)
    nonzero_mask = np.abs(coef) > 1e-12
    total_terms = coef.size
    nonzero_terms = int(nonzero_mask.sum())
    density = nonzero_terms / total_terms if total_terms else 0.0
    empty_equations = int((nonzero_mask.sum(axis=1) == 0).sum())
    max_abs_coef = float(np.max(np.abs(coef))) if total_terms else 0.0

    print(f"Coefficient matrix shape: {coef.shape}")
    print(f"Non-zero terms: {nonzero_terms}/{total_terms}")
    print(f"Sparsity density: {density:.4f}")
    print(f"Empty equations: {empty_equations}")
    print(f"Maximum |coefficient|: {max_abs_coef:.4f}")

    