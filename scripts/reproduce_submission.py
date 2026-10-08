"""Regenerate the comparison, evidence, charts, and completed HWPX report."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trees", action="store_true",
                        help="Reuse completed searches and refit both tree models")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--force-tree-search",
        action="store_true",
        help="With --train-trees, ignore saved grid-search CSVs and rerun all candidate/folds",
    )
    parser.add_argument("--capture-dashboard", action="store_true",
                        help="Refresh the dashboard screenshot using Playwright")
    args = parser.parse_args()

    run("verify_cleaned_data.py")
    if args.force_tree_search and not args.train_trees:
        parser.error("--force-tree-search requires --train-trees")
    if args.train_trees:
        train_args = ["--models", "lightgbm", "xgboost", "--n-jobs", str(args.n_jobs)]
        if args.force_tree_search:
            train_args.append("--force-search")
        run("train_tree_models.py", *train_args)
    run("build_model_comparison.py")
    run("build_report_evidence.py")
    run("analyze_feature_importance.py")
    run("build_eda_evidence.py")
    run("make_report_figures.py")
    if args.capture_dashboard:
        run("capture_dashboard.py")
    for filename in ("03_dashboard.png", "12_dashboard_causes.png",
                     "13_dashboard_model.png", "14_dashboard_actions.png"):
        screenshot = ROOT / "docs/report/figures" / filename
        if not screenshot.is_file():
            raise FileNotFoundError(f"Dashboard screenshot is required: {screenshot}")
    run("fill_result_report.py")
    run("verify_report.py")


if __name__ == "__main__":
    main()
