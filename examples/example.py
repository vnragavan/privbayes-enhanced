#!/usr/bin/env python3
"""Basic example showing how to use Enhanced PrivBayes."""

import pandas as pd
import numpy as np
from privbayes_enhanced import EnhancedPrivBayesAdapter

def create_sample_data(n=1000):
    """Generate some fake data to work with."""
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 20000, n).clip(0),
        'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n),
        'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
        'married': np.random.choice([True, False], n),
        'target': np.random.choice(['A', 'B', 'C'], n)
    })
    return data

def main():
    print("=" * 80)
    print("Enhanced PrivBayes Examples")
    print("=" * 80)
    print()
    
    # Create some sample data
    print("Creating sample data...")
    data = create_sample_data(n=1000)
    print(f"Original data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print()
    
    # Basic usage
    print("=" * 80)
    print("Example 1: Basic setup")
    print("=" * 80)
    model1 = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        delta=1e-6,
        seed=42,
        cat_keep_all_nonzero=True
    )
    
    print("Fitting model...")
    model1.fit(data)
    
    print("Generating synthetic data...")
    synthetic1 = model1.sample(n_samples=500)
    print(f"Synthetic data shape: {synthetic1.shape}")
    
    report1 = model1.privacy_report()
    print(f"\nPrivacy Report:")
    print(f"  Total epsilon used: {report1['epsilon_total']:.4f}")
    print(f"  Delta used: {report1['delta']:.2e}")
    print(f"  Epsilon for structure: {report1['eps_struct']:.4f}")
    print(f"  Epsilon for CPT: {report1['eps_cpt']:.4f}")
    print(f"  Epsilon for discovery: {report1['eps_disc']:.4f}")
    print()
    
    # Using public knowledge
    print("=" * 80)
    print("Example 2: Using public knowledge")
    print("=" * 80)
    model2 = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        public_categories={
            'state': ['CA', 'NY', 'TX', 'FL', 'IL'],
            'education': ['HS', 'BS', 'MS', 'PhD'],
            'target': ['A', 'B', 'C']
        },
        public_bounds={
            'age': [0, 120],
            'income': [0, 200000]
        },
        label_columns=['target'],
        cat_keep_all_nonzero=True
    )
    
    print("Fitting model with public knowledge...")
    model2.fit(data)
    
    print("Generating synthetic data...")
    synthetic2 = model2.sample(n_samples=500)
    print(f"Synthetic data shape: {synthetic2.shape}")
    
    # Check for __UNK__ tokens
    unk_count = 0
    for col in synthetic2.columns:
        if synthetic2[col].dtype == 'object':
            unk_in_col = (synthetic2[col].astype(str) == '__UNK__').sum()
            if unk_in_col > 0:
                print(f"  Warning: {unk_in_col} __UNK__ tokens in column '{col}'")
                unk_count += unk_in_col
    
    if unk_count == 0:
        print("  ✓ No __UNK__ tokens found (all categories preserved)")
    print()
    
    # More advanced settings
    print("=" * 80)
    print("Example 3: Advanced settings")
    print("=" * 80)
    model3 = EnhancedPrivBayesAdapter(
        epsilon=2.0,
        delta=1e-6,
        temperature=1.5,  # Higher temperature for more privacy
        max_parents=3,
        bins_per_numeric=50,
        eps_split={"structure": 0.3, "cpt": 0.7},
        label_columns=['target'],
        cat_keep_all_nonzero=True
    )
    
    print("Fitting model with advanced configuration...")
    model3.fit(data)
    
    print("Generating synthetic data...")
    synthetic3 = model3.sample(n_samples=1000)
    print(f"Synthetic data shape: {synthetic3.shape}")
    
    report3 = model3.privacy_report()
    print(f"\nPrivacy Report:")
    print(f"  Total epsilon used: {report3['epsilon_total']:.4f}")
    print(f"  Temperature: {report3['temperature']:.2f}")
    print()
    
    print("=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()


