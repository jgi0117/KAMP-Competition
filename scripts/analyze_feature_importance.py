"""Group-permutation importance of saved tree models on the held-out Test period."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_tree_models as trees  # noqa: E402


def main():
    data = trees.dp.load_data()
    X, y, indices, keep = trees.make_windows(data)
    split = trees.dp.get_final_split(data)
    selected = np.flatnonzero(split.eval[indices] & keep)
    features = trees.dp.FEATURE_COLUMNS
    original = X[selected]
    actual = y[selected]
    rng = np.random.default_rng(42)
    rows = []
    for model_name in ("xgboost", "lightgbm"):
        model = joblib.load(ROOT / f"results/tree_models/models/{model_name}_seed42.joblib")
        baseline = float(np.sqrt(np.mean((actual - model.predict(original)) ** 2)))
        for column, name in enumerate(features):
            increases = []
            for _ in range(5):
                changed = original.copy()
                permutation = rng.permutation(len(original))
                changed[:, column::len(features)] = original[permutation][:, column::len(features)]
                rmse = float(np.sqrt(np.mean((actual - model.predict(changed)) ** 2)))
                increases.append(rmse - baseline)
            rows.append({"model": model_name, "feature": name, "base_rmse": baseline,
                         "rmse_increase_mean": float(np.mean(increases)),
                         "rmse_increase_std": float(np.std(increases, ddof=1))})
    output = ROOT / "docs/report/evidence/grouped_permutation_importance.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).sort_values(["model", "rmse_increase_mean"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
