#!/usr/bin/env python3
"""Test loading CSV files directly (simulating database exports) and processing them."""

import pandas as pd
from privbayes_enhanced import EnhancedPrivBayesAdapter
import os

def test_csv_loading():
    """Test loading CSV files and processing them."""
    print("=" * 80)
    print("Testing CSV File Loading (Database Export Simulation)")
    print("=" * 80)
    print()
    
    test_files = [
        'test_data/ecommerce_database_export.csv',
        'test_data/employee_database_export.csv',
        'test_data/sensor_database_export.csv',
        'test_data/financial_database_export.csv',
    ]
    
    results = {}
    
    for csv_file in test_files:
        if not os.path.exists(csv_file):
            print(f"⚠ Skipping {csv_file} (not found)")
            continue
        
        print(f"Testing: {csv_file}")
        print("-" * 80)
        
        try:
            # Load CSV (simulating database export)
            print("Loading CSV file...")
            df = pd.read_csv(csv_file)
            print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"  Data types after CSV load:")
            for col, dtype in df.dtypes.items():
                print(f"    {col}: {dtype}")
            
            # Check if datetime columns are strings (common in CSV exports)
            datetime_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to detect if it's a datetime string
                    sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                    if sample and isinstance(sample, str):
                        try:
                            pd.to_datetime(sample)
                            datetime_cols.append(col)
                        except:
                            pass
            
            if datetime_cols:
                print(f"  Detected potential datetime string columns: {datetime_cols}")
            
            # Fit model (should auto-detect and convert types)
            print("\nFitting Enhanced PrivBayes model...")
            model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
            model.fit(df)
            print("  ✓ Model fitted successfully")
            
            # Generate synthetic data
            print("Generating synthetic data...")
            synthetic = model.sample(n_samples=min(200, len(df)))
            print(f"  ✓ Generated {len(synthetic)} synthetic samples")
            
            # Check column preservation
            print("\nColumn validation:")
            print(f"  Original columns: {len(df.columns)}")
            print(f"  Synthetic columns: {len(synthetic.columns)}")
            print(f"  Columns match: {list(df.columns) == list(synthetic.columns)}")
            
            # Check data types
            print("\nData type comparison:")
            for col in df.columns[:5]:  # Show first 5
                orig_type = str(df[col].dtype)
                synth_type = str(synthetic[col].dtype)
                match = "✓" if orig_type == synth_type or (orig_type.startswith('datetime') and synth_type == 'int64') else "⚠"
                print(f"  {match} {col}: {orig_type} → {synth_type}")
            
            # Privacy report
            report = model.privacy_report()
            print(f"\nPrivacy budget: {report['epsilon_total']:.6f} ε")
            
            print(f"\n✓ {os.path.basename(csv_file)} test PASSED\n")
            results[csv_file] = True
            
        except Exception as e:
            print(f"\n✗ {os.path.basename(csv_file)} test FAILED")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[csv_file] = False
            print()
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    for csv_file, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{os.path.basename(csv_file):40s}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_csv_loading()
    exit(0 if success else 1)

