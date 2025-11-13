"""Tests for EnhancedPrivBayesAdapter."""

import pytest
import pandas as pd
import numpy as np
from privbayes_enhanced import EnhancedPrivBayesAdapter


def test_adapter_basic_fit_and_sample(sample_data):
    """Test basic fit and sample functionality."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(sample_data)
    
    synthetic = model.sample(n_samples=500)
    
    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape[0] == 500
    assert set(synthetic.columns) == set(sample_data.columns)


def test_adapter_default_sample_size(sample_data):
    """Test that sample() uses original data size by default."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(sample_data)
    
    synthetic = model.sample()
    
    assert synthetic.shape[0] == len(sample_data)


def test_adapter_privacy_report(sample_data):
    """Test privacy report generation."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, delta=1e-6, seed=42)
    model.fit(sample_data)
    
    report = model.privacy_report()
    
    assert isinstance(report, dict)
    assert 'epsilon_total' in report
    assert 'delta' in report
    assert 'eps_struct' in report
    assert 'eps_cpt' in report
    assert 'eps_disc' in report
    assert report['epsilon_total'] > 0


def test_adapter_with_public_categories(sample_data):
    """Test adapter with public categories."""
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        public_categories={
            'state': ['CA', 'NY', 'TX', 'FL', 'IL'],
            'education': ['HS', 'BS', 'MS', 'PhD']
        },
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(n_samples=200)
    
    # Check that no __UNK__ tokens appear in columns with public categories
    for col in ['state', 'education']:
        if col in synthetic.columns:
            assert '__UNK__' not in synthetic[col].astype(str).values


def test_adapter_with_label_columns(sample_data):
    """Test adapter with label columns."""
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        label_columns=['target'],
        public_categories={'target': ['A', 'B', 'C']},
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(n_samples=200)
    
    # Label columns should not have __UNK__
    assert '__UNK__' not in synthetic['target'].astype(str).values
    assert set(synthetic['target'].unique()) <= {'A', 'B', 'C'}


def test_adapter_temperature(sample_data):
    """Test adapter with temperature parameter."""
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        temperature=1.5,
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(n_samples=200)
    
    assert synthetic.shape[0] == 200
    report = model.privacy_report()
    assert report['temperature'] == 1.5


def test_adapter_cat_keep_all_nonzero(sample_data):
    """Test adapter with cat_keep_all_nonzero option."""
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        cat_keep_all_nonzero=True,
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(n_samples=200)
    
    # With cat_keep_all_nonzero=True, should minimize __UNK__ tokens
    unk_count = 0
    for col in synthetic.columns:
        if synthetic[col].dtype == 'object':
            unk_count += (synthetic[col].astype(str) == '__UNK__').sum()
    
    # Should have very few or no UNK tokens
    assert unk_count < len(synthetic) * 0.1  # Less than 10% UNK


def test_adapter_different_epsilon_values(sample_data):
    """Test adapter with different epsilon values."""
    for eps in [0.1, 0.5, 1.0, 2.0]:
        model = EnhancedPrivBayesAdapter(epsilon=eps, seed=42)
        model.fit(sample_data)
        
        synthetic = model.sample(n_samples=100)
        report = model.privacy_report()
        
        assert synthetic.shape[0] == 100
        assert report['epsilon_total'] > 0


def test_adapter_error_if_not_fitted():
    """Test that sampling without fitting raises error."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0)
    
    with pytest.raises(ValueError, match="must be fitted"):
        model.sample()


def test_adapter_with_public_bounds(sample_data):
    """Test adapter with public bounds."""
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        public_bounds={
            'age': [0, 120],
            'income': [0, 200000]
        },
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(n_samples=200)
    
    # Check that values are within bounds
    assert (synthetic['age'] >= 0).all()
    assert (synthetic['age'] <= 120).all()
    assert (synthetic['income'] >= 0).all()


def test_adapter_mixed_data_types(mixed_data):
    """Test adapter with mixed data types."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(mixed_data)
    
    synthetic = model.sample(n_samples=100)
    
    assert synthetic.shape == (100, len(mixed_data.columns))
    assert set(synthetic.columns) == set(mixed_data.columns)

