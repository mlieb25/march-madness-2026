#!/usr/bin/env python3
"""
March Madness ML Pipeline - Complete Execution Script (V2)
============================================================
Orchestrates the entire machine learning pipeline from data preparation
through final predictions using the enhanced master dataset.

Pipeline Phases:
    Phase 0: Master Dataset Creation (build_master_dataset.py)
    Phase 1: ETL - Data Integration (etl_v2.py)
    Phase 2: Baseline Models & Benchmarks
    Phase 3: Model Search & Hyperparameter Tuning
    Phase 4: Calibration
    Phase 5: Ensemble & BMA
    Phase 6: Tournament Simulation

Author: Mitchell Liebrecht
Date: March 15, 2026
"""

import sys
import subprocess
import time
from pathlib import Path
import json

print("=" * 80)
print("MARCH MADNESS ML PIPELINE V2 - COMPLETE EXECUTION")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

SKIP_IF_EXISTS = True  # Skip phases if output files already exist
STOP_ON_ERROR = True   # Stop pipeline if any phase fails

PHASES = [
    {
        'name': 'Phase 0: Master Dataset Creation',
        'script': 'build_master_dataset.py',
        'outputs': ['processed_data/team_season_stats.csv', 
                    'processed_data/tournament_games_features.csv'],
        'required': True,
        'description': 'Build comprehensive master dataset from Kaggle data'
    },
    {
        'name': 'Phase 1: Enhanced ETL',
        'script': 'etl_v2.py',
        'outputs': ['data/ml_training_data_v2.csv', 
                    'data/ml_inference_data_2026_v2.csv'],
        'required': True,
        'description': 'Integrate master dataset with external sources'
    },
    {
        'name': 'Phase 2: Baseline Models',
        'script': 'phase2_baselines.py',
        'outputs': ['data/phase2_results.csv',
                    'data/phase2_bar_to_beat.json'],
        'required': False,
        'description': 'Establish baseline performance benchmarks'
    },
    {
        'name': 'Phase 3: Model Search',
        'script': 'phase3_model_search.py',
        'outputs': ['data/phase3_top_models.json',
                    'data/phase3_search_results.csv'],
        'required': False,
        'description': 'Hyperparameter tuning and model selection'
    },
    {
        'name': 'Phase 4: Calibration',
        'script': 'phase4.py',
        'outputs': ['data/calibrated_predictions_2026.csv',
                    'data/phase4_oof_probs.csv'],
        'required': False,
        'description': 'Probability calibration for reliable predictions'
    },
    {
        'name': 'Phase 5: Ensemble',
        'script': 'phase5_ensemble.py',
        'outputs': ['data/phase5_ensemble_probs.csv',
                    'data/phase5_submission.csv'],
        'required': False,
        'description': 'Bayesian model averaging and stacking'
    },
    {
        'name': 'Phase 6: Tournament Simulation',
        'script': 'phase6_simulation.py',
        'outputs': ['data/phase6_bracket.csv',
                    'data/phase6_win_probabilities.csv'],
        'required': False,
        'description': 'Monte Carlo tournament simulation'
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_outputs_exist(outputs):
    """Check if all output files exist."""
    for output in outputs:
        if not Path(output).exists():
            return False
    return True

def run_phase(phase_info):
    """Execute a single phase of the pipeline."""
    name = phase_info['name']
    script = phase_info['script']
    outputs = phase_info['outputs']
    required = phase_info['required']
    
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")
    print(f"Script: {script}")
    print(f"Description: {phase_info['description']}")
    
    # Check if we can skip
    if SKIP_IF_EXISTS and check_outputs_exist(outputs):
        print(f"\n✓ Outputs already exist, skipping...")
        print(f"  Files: {outputs}")
        return True
    
    # Check if script exists
    if not Path(script).exists():
        print(f"\n✗ Script not found: {script}")
        if required:
            print(f"  This is a required phase!")
            return False
        else:
            print(f"  Skipping optional phase...")
            return True
    
    # Execute
    print(f"\nExecuting...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per phase
        )
        
        elapsed = time.time() - start_time
        
        # Show output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✓ Phase completed successfully in {elapsed:.1f}s")
            
            # Verify outputs were created
            missing = [o for o in outputs if not Path(o).exists()]
            if missing:
                print(f"\n! Warning: Expected outputs not created: {missing}")
                if required:
                    return False
            
            return True
        else:
            print(f"\n✗ Phase failed with exit code {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ Phase timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Phase failed with exception: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete pipeline."""
    start_time = time.time()
    
    print("Pipeline Configuration:")
    print(f"  Skip if exists: {SKIP_IF_EXISTS}")
    print(f"  Stop on error: {STOP_ON_ERROR}")
    print(f"  Total phases: {len(PHASES)}")
    print()
    
    results = []
    
    for i, phase in enumerate(PHASES, 1):
        print(f"\n\n{'#' * 80}")
        print(f"EXECUTING {i}/{len(PHASES)}: {phase['name']}")
        print(f"{'#' * 80}")
        
        success = run_phase(phase)
        
        results.append({
            'phase': phase['name'],
            'script': phase['script'],
            'success': success,
            'required': phase['required']
        })
        
        if not success:
            if phase['required'] and STOP_ON_ERROR:
                print(f"\n\n{'!' * 80}")
                print(f"PIPELINE HALTED: Required phase failed")
                print(f"{'!' * 80}")
                break
            elif not phase['required']:
                print(f"\nOptional phase failed, continuing...")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    elapsed = time.time() - start_time
    
    print(f"\n\n{'=' * 80}")
    print("PIPELINE EXECUTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total time: {elapsed/60:.1f} minutes\n")
    
    print("Phase Results:")
    print(f"{'Phase':<40} {'Status':<10} {'Required'}")
    print("-" * 80)
    
    for result in results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        required = "Yes" if result['required'] else "No"
        print(f"{result['phase']:<40} {status:<10} {required}")
    
    # Save summary
    summary = {
        'total_time_seconds': elapsed,
        'phases_run': len(results),
        'phases_successful': sum(1 for r in results if r['success']),
        'phases_failed': sum(1 for r in results if not r['success']),
        'results': results
    }
    
    with open('data/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPipeline summary saved to: data/pipeline_summary.json")
    
    # Check for final outputs
    print("\n" + "=" * 80)
    print("KEY OUTPUT FILES")
    print("=" * 80)
    
    key_outputs = [
        ('Master Dataset', 'processed_data/team_season_stats.csv'),
        ('Training Data', 'data/ml_training_data_v2.csv'),
        ('Inference Data', 'data/ml_inference_data_2026_v2.csv'),
        ('Baseline Results', 'data/phase2_results.csv'),
        ('Top Models', 'data/phase3_top_models.json'),
        ('Calibrated Predictions', 'data/calibrated_predictions_2026.csv'),
        ('Ensemble Predictions', 'data/phase5_ensemble_probs.csv'),
        ('Kaggle Submission', 'data/phase5_submission.csv'),
        ('Bracket', 'data/phase6_bracket.csv')
    ]
    
    for name, path in key_outputs:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists} {name:<25} {path}")
    
    print()
    
    # Success check
    required_failed = sum(1 for r in results if r['required'] and not r['success'])
    if required_failed == 0:
        print("✓ All required phases completed successfully!")
        print("\nYou can now:")
        print("  1. Review baseline results: data/phase2_results.csv")
        print("  2. Check top models: data/phase3_top_models.json")
        print("  3. Submit predictions: data/phase5_submission.csv")
        print("  4. View bracket: data/phase6_bracket.csv")
        return 0
    else:
        print(f"✗ {required_failed} required phase(s) failed")
        print("  Check error messages above for details")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
