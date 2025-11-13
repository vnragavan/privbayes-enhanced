#!/usr/bin/env python3
"""Test utility and privacy metrics."""

import pandas as pd
import numpy as np
from privbayes_enhanced import EnhancedPrivBayesAdapter
from privbayes_enhanced.metrics import compute_utility_metrics, compute_privacy_metrics, print_utility_report, print_privacy_report

def main():
    print("=" * 80)
    print("Testing Utility and Privacy Metrics")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    real_data = pd.read_csv('data/adult.csv')
    print(f"Loaded {len(real_data)} rows, {len(real_data.columns)} columns")
    print()
    
    # Create and fit model
    print("Creating and fitting model...")
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        delta=1e-6,
        seed=42,
        cat_keep_all_nonzero=True
    )
    model.fit(real_data)
    print("Model fitted")
    print()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = model.sample(n_samples=min(1000, len(real_data)))
    print(f"Generated {len(synthetic_data)} synthetic samples")
    print()
    
    # Compute utility metrics
    print("Computing utility metrics...")
    utility_metrics = model.evaluate_utility(real_data, synthetic_data, verbose=True)
    print()
    
    # Compute privacy metrics
    print("Computing privacy metrics...")
    privacy_metrics = model.evaluate_privacy(real_data, synthetic_data, verbose=True)
    print()
    
    # Privacy report
    print("Privacy Budget Report:")
    privacy_report = model.privacy_report()
    print(f"  Total epsilon used: {privacy_report['epsilon_total']:.6f}")
    print(f"  Delta used: {privacy_report['delta']:.2e}")
    print()
    
    print("=" * 80)
    print("Metrics test completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()

