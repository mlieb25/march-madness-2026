#!/usr/bin/env python3
"""
Run the Full pipeline in order: data-pull → etl → phase3_model_search → phase4_calibration
→ phase5_ensemble → phase6_simulation. Exits with a clear message if a prior step failed
or a required artifact is missing.
"""
import os
import sys
import subprocess

DATA_DIR = "data"
STEPS = [
    ("data-pull", "python3 data-pull.py", [
        "data/barttorvik_historical.csv",
        "data/fivethirtyeight_forecasts.csv",
    ]),
    ("etl", "python3 etl.py", [
        "data/ml_training_data.csv",
        "data/ml_inference_data_2026.csv",
    ]),
    ("phase3_model_search", "python3 phase3_model_search.py", [
        "data/phase3_top_models.json",
    ]),
    ("phase4_calibration", "python3 phase4_calibration.py", [
        "data/phase4_oof_probs.csv",
        "data/phase4_inference_probs.csv",
        "data/phase4_best_combos.json",
    ]),
    ("phase5_ensemble", "python3 phase5_ensemble.py", [
        "data/phase5_ensemble_probs.csv",
    ]),
    ("phase6_simulation", "python3 phase6_simulation.py", [
        "data/phase6_team_round_probs.csv",
        "data/phase6_brackets.json",
    ]),
]


def main():
    print("=" * 60)
    print("March Madness Full Pipeline (run_all.py)")
    print("=" * 60)

    for name, cmd, required_after in STEPS:
        print(f"\n── {name} ──")
        ret = subprocess.run(cmd, shell=True)
        if ret.returncode != 0:
            print(f"\n[!] {name} failed (exit code {ret.returncode}). Fix the error above and re-run.")
            sys.exit(ret.returncode)

        for path in required_after:
            if not os.path.isfile(path):
                print(f"\n[!] Expected artifact missing after {name}: {path}")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Full pipeline complete.")
    print("  Dashboard: cd app && streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
