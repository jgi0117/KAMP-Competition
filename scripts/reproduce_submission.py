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
                        help="Complete both 108-candidate searches and refit the tree models")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--capture-dashboard", action="store_true",
                        help="Refresh the dashboard screenshot using Playwright")
    args = parser.parse_args()

    run("verify_cleaned_data.py")
    if args.train_trees:
        run("train_base9_trees.py", "--models", "lightgbm", "xgboost",
            "--n-jobs", str(args.n_jobs))
    run("build_base9_comparison.py")
    run("build_report_evidence.py")
    run("analyze_feature_importance.py")
    run("make_report_figures.py")
    if args.capture_dashboard:
        run("capture_dashboard.py")
    screenshot = ROOT / "report/figures/03_dashboard.png"
    if not screenshot.is_file():
        raise FileNotFoundError(f"Dashboard screenshot is required: {screenshot}")
    run("fill_result_report.py")
    run("verify_report.py")


if __name__ == "__main__":
    main()
