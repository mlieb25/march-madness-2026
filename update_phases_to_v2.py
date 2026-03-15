#!/usr/bin/env python3
"""
Automatic Phase Script Updater for V2
======================================
Updates phase2-6 scripts to use enhanced V2 datasets.

This script creates backup copies and modifies data paths to use:
  - ml_training_data_v2.csv (instead of ml_training_data.csv)
  - ml_inference_data_2026_v2.csv (instead of ml_inference_data_2026.csv)

Author: Mitchell Liebrecht
Date: March 15, 2026
"""

import re
import shutil
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("PHASE SCRIPTS V2 UPDATER")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS_TO_UPDATE = [
    'phase2_baselines.py',
    'phase3_model_search.py',
    'phase4.py',
    'phase4_calibration.py',
    'phase5_ensemble.py',
    'phase6_simulation.py',
    'models.py'
]

REPLACEMENTS = [
    # Training data
    (r'"data/ml_training_data\.csv"', '"data/ml_training_data_v2.csv"'),
    (r"'data/ml_training_data\.csv'", "'data/ml_training_data_v2.csv'"),
    
    # Inference data
    (r'"data/ml_inference_data_2026\.csv"', '"data/ml_inference_data_2026_v2.csv"'),
    (r"'data/ml_inference_data_2026\.csv'", "'data/ml_inference_data_2026_v2.csv'"),
]

BACKUP_DIR = Path('backups_v1')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_backup(file_path):
    """Create timestamped backup of file."""
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = BACKUP_DIR / backup_name
    shutil.copy2(file_path, backup_path)
    return backup_path

def update_file(file_path, replacements):
    """Apply replacements to file content."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    changes_made = []
    
    for pattern, replacement in replacements:
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            changes_made.append((pattern, len(matches)))
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True, changes_made
    else:
        return False, []

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("This script will update your phase scripts to use V2 datasets.")
    print()
    print("Changes:")
    print("  - ml_training_data.csv → ml_training_data_v2.csv")
    print("  - ml_inference_data_2026.csv → ml_inference_data_2026_v2.csv")
    print()
    print(f"Backups will be saved to: {BACKUP_DIR}/")
    print()
    
    # Check if V2 data exists
    v2_train = Path('data/ml_training_data_v2.csv')
    v2_infer = Path('data/ml_inference_data_2026_v2.csv')
    
    if not v2_train.exists() or not v2_infer.exists():
        print("✗ V2 datasets not found!")
        print()
        print("Please run these first:")
        print("  1. python build_master_dataset.py")
        print("  2. python etl_v2.py")
        print()
        return
    
    print("✓ V2 datasets found")
    print()
    
    # Confirm
    response = input("Proceed with updates? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print()
    print("Updating scripts...")
    print()
    
    updated_count = 0
    skipped_count = 0
    
    for script_name in SCRIPTS_TO_UPDATE:
        script_path = Path(script_name)
        
        if not script_path.exists():
            print(f"  ✗ {script_name} - Not found, skipping")
            skipped_count += 1
            continue
        
        # Create backup
        backup_path = create_backup(script_path)
        print(f"  ✓ Created backup: {backup_path}")
        
        # Update file
        updated, changes = update_file(script_path, REPLACEMENTS)
        
        if updated:
            print(f"  ✓ Updated {script_name}")
            for pattern, count in changes:
                print(f"      - {count} occurrence(s) of pattern updated")
            updated_count += 1
        else:
            print(f"  - {script_name} - No changes needed")
            skipped_count += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("UPDATE COMPLETE")
    print("=" * 80)
    print()
    print(f"Scripts updated:  {updated_count}")
    print(f"Scripts skipped:  {skipped_count}")
    print(f"Backups saved to: {BACKUP_DIR}/")
    print()
    
    if updated_count > 0:
        print("Next steps:")
        print("  1. Review changes in updated scripts")
        print("  2. Run phase2_baselines.py to test with V2 data")
        print("  3. If issues occur, restore from backups/")
        print()
        print("To restore a backup:")
        print(f"  cp {BACKUP_DIR}/phase2_baselines_*.py phase2_baselines.py")
    else:
        print("No updates were needed. Scripts may already use V2 data.")
    print()

if __name__ == "__main__":
    main()
