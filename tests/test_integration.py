"""Integration tests for Enhanced PrivBayes."""

import pytest
import pandas as pd
import numpy as np
from privbayes_enhanced import EnhancedPrivBayesAdapter
from privbayes_enhanced.synthesizer import PrivBayesSynthesizerEnhanced


def test_end_to_end_workflow(sample_data):
    """Test complete workflow from data to synthetic data."""
    # Create model
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        delta=1e-6,
        seed=42,
        cat_keep_all_nonzero=True
    )
    
    # Fit model
    model.fit(sample_data)
    
    # Generate synthetic data
    synthetic = model.sample(n_samples=len(sample_data))
    
    # Verify output
    assert synthetic.shape == sample_data.shape
    assert set(synthetic.columns) == set(sample_data.columns)
    
    # Check privacy report
    report = model.privacy_report()
    assert report['epsilon_total'] > 0
    assert report['delta'] > 0


def test_reproducibility(sample_data):
    """Test that same seed produces same results."""
    model1 = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model1.fit(sample_data)
    synthetic1 = model1.sample(n_samples=100)
    
    model2 = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model2.fit(sample_data)
    synthetic2 = model2.sample(n_samples=100)
    
    # With same seed, should get same results
    pd.testing.assert_frame_equal(synthetic1, synthetic2)


def test_different_seeds_produce_different_results(sample_data):
    """Test that different seeds produce different results."""
    model1 = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model1.fit(sample_data)
    synthetic1 = model1.sample(n_samples=100)
    
    model2 = EnhancedPrivBayesAdapter(epsilon=1.0, seed=123)
    model2.fit(sample_data)
    synthetic2 = model2.sample(n_samples=100)
    
    # With different seeds, should get different results
    # (very unlikely to be identical)
    assert not synthetic1.equals(synthetic2)


def test_advanced_configuration(sample_data):
    """Test advanced configuration options."""
    model = EnhancedPrivBayesAdapter(
        epsilon=2.0,
        delta=1e-6,
        temperature=1.5,
        max_parents=3,
        bins_per_numeric=50,
        eps_split={"structure": 0.3, "cpt": 0.7},
        label_columns=['target'],
        public_categories={
            'state': ['CA', 'NY', 'TX', 'FL', 'IL'],
            'education': ['HS', 'BS', 'MS', 'PhD'],
            'target': ['A', 'B', 'C']
        },
        public_bounds={
            'age': [0, 120],
            'income': [0, 200000]
        },
        cat_keep_all_nonzero=True,
        seed=42
    )
    
    model.fit(sample_data)
    synthetic = model.sample(n_samples=500)
    
    assert synthetic.shape[0] == 500
    report = model.privacy_report()
    assert report['epsilon_total'] > 0
    assert report['temperature'] == 1.5


def test_small_dataset(small_data):
    """Test with small dataset."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(small_data)
    
    synthetic = model.sample(n_samples=50)
    
    assert synthetic.shape[0] == 50
    assert set(synthetic.columns) == set(small_data.columns)


def test_large_synthetic_sample(sample_data):
    """Test generating more synthetic samples than original data."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(sample_data)
    
    # Generate 10x more samples
    synthetic = model.sample(n_samples=len(sample_data) * 10)
    
    assert synthetic.shape[0] == len(sample_data) * 10
    assert set(synthetic.columns) == set(sample_data.columns)


def test_data_with_nans(sample_data):
    """Test handling of data with NaN values."""
    # Add some NaN values
    data_with_nans = sample_data.copy()
    data_with_nans.loc[0:10, 'income'] = np.nan
    data_with_nans.loc[5:15, 'state'] = None
    
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(data_with_nans)
    
    synthetic = model.sample(n_samples=200)
    
    assert synthetic.shape[0] == 200
    # Should handle NaNs gracefully


def test_datetime_columns():
    """Test handling of datetime columns."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B'], 100)
    })
    
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(data)
    
    synthetic = model.sample(n_samples=50)
    
    assert synthetic.shape[0] == 50
    assert 'date' in synthetic.columns


def test_boolean_columns():
    """Test handling of boolean columns."""
    data = pd.DataFrame({
        'flag': np.random.choice([True, False], 200),
        'value': np.random.normal(0, 1, 200)
    })
    
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(data)
    
    synthetic = model.sample(n_samples=100)
    
    assert synthetic.shape[0] == 100
    # Boolean columns should be handled
    assert 'flag' in synthetic.columns

