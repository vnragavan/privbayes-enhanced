#!/usr/bin/env python3
"""Simple test script to test Enhanced PrivBayes with your data."""

import pandas as pd
import os
from privbayes_enhanced import EnhancedPrivBayesAdapter


def test_with_data(data_path='data/adult.csv', n_samples=500, epsilon=1.0):
    """
    Test Enhanced PrivBayes with your data file.
    
    Args:
        data_path: Path to your CSV file
        n_samples: Number of synthetic samples to generate
        epsilon: Privacy budget
    """
    print("=" * 80)
    print("Testing Enhanced PrivBayes with Your Data")
    print("=" * 80)
    print()
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Original data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print()
    
    # Create model
    print("Creating Enhanced PrivBayes model...")
    model = EnhancedPrivBayesAdapter(
        epsilon=epsilon,
        delta=1e-6,
        seed=42,
        cat_keep_all_nonzero=True  # Minimize __UNK__ tokens
    )
    
    # Fit model
    print("Fitting model to data...")
    model.fit(data)
    print("✓ Model fitted successfully")
    print()
    
    # Generate synthetic data
    print(f"Generating {n_samples} synthetic samples...")
    synthetic = model.sample(n_samples=n_samples)
    print(f"✓ Generated synthetic data shape: {synthetic.shape}")
    print()
    
    # Check for __UNK__ tokens
    print("Checking for __UNK__ tokens...")
    unk_count = 0
    for col in synthetic.columns:
        if synthetic[col].dtype == 'object':
            unk_in_col = (synthetic[col].astype(str) == '__UNK__').sum()
            if unk_in_col > 0:
                print(f"  Warning: {unk_in_col} __UNK__ tokens in column '{col}'")
                unk_count += unk_in_col
    
    if unk_count == 0:
        print("  ✓ No __UNK__ tokens found")
    else:
        print(f"  ⚠ Total __UNK__ tokens: {unk_count}")
    print()
    
    # Privacy report
    print("Privacy Report:")
    report = model.privacy_report()
    print(f"  Total epsilon used: {report['epsilon_total']:.4f}")
    print(f"  Delta used: {report['delta']:.2e}")
    print(f"  Epsilon for structure: {report['eps_struct']:.4f}")
    print(f"  Epsilon for CPT: {report['eps_cpt']:.4f}")
    print(f"  Epsilon for discovery: {report['eps_disc']:.4f}")
    print()
    
    # Basic statistics comparison
    print("Basic Statistics Comparison:")
    print(f"  Original data size: {len(data)}")
    print(f"  Synthetic data size: {len(synthetic)}")
    print()
    
    # Show sample of synthetic data
    print("Sample of synthetic data (first 5 rows):")
    print(synthetic.head())
    print()
    
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    
    return model, synthetic, report


if __name__ == "__main__":
    import sys
    
    # Allow command line arguments
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'data/adult.csv'
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    test_with_data(data_path=data_path, n_samples=n_samples, epsilon=epsilon)

