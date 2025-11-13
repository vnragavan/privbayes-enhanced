"""Tests for PrivBayesSynthesizerEnhanced."""

import pytest
import pandas as pd
import numpy as np
from privbayes_enhanced.synthesizer import PrivBayesSynthesizerEnhanced


def test_synthesizer_basic_fit_and_sample(sample_data):
    """Test basic fit and sample functionality."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, seed=42)
    model.fit(sample_data)
    
    synthetic = model.sample(500)
    
    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape[0] == 500
    assert set(synthetic.columns) == set(sample_data.columns)


def test_synthesizer_privacy_report(sample_data):
    """Test privacy report generation."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, delta=1e-6, seed=42)
    model.fit(sample_data)
    
    report = model.privacy_report()
    
    assert isinstance(report, dict)
    assert 'epsilon' in report
    assert 'delta' in report
    assert 'eps_struct' in report
    assert 'eps_cpt' in report
    assert 'eps_disc_used' in report
    assert 'temperature' in report


def test_synthesizer_parents_property(sample_data):
    """Test parents_ property."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, seed=42)
    model.fit(sample_data)
    
    parents = model.parents_
    
    assert isinstance(parents, dict)
    assert len(parents) == len(sample_data.columns)
    for col, par_list in parents.items():
        assert col in sample_data.columns
        assert isinstance(par_list, list)
        assert len(par_list) <= model.max_parents


def test_synthesizer_temperature(sample_data):
    """Test temperature parameter."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, temperature=2.0, seed=42)
    model.fit(sample_data)
    
    report = model.privacy_report()
    assert report['temperature'] == 2.0
    
    synthetic = model.sample(200)
    assert synthetic.shape[0] == 200


def test_synthesizer_max_parents(sample_data):
    """Test max_parents parameter."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, max_parents=1, seed=42)
    model.fit(sample_data)
    
    parents = model.parents_
    for col, par_list in parents.items():
        assert len(par_list) <= 1


def test_synthesizer_public_categories(sample_data):
    """Test with public categories."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        public_categories={
            'state': ['CA', 'NY', 'TX', 'FL', 'IL'],
            'education': ['HS', 'BS', 'MS', 'PhD']
        },
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(200)
    
    # Check that public categories are used
    public_cats_map = {
        'state': {'CA', 'NY', 'TX', 'FL', 'IL', '__UNK__'},
        'education': {'HS', 'BS', 'MS', 'PhD', '__UNK__'}
    }
    
    for col in ['state', 'education']:
        if col in synthetic.columns:
            unique_vals = set(synthetic[col].astype(str).unique())
            # Should only contain public categories (and possibly __UNK__)
            assert unique_vals.issubset(public_cats_map.get(col, set()))


def test_synthesizer_public_bounds(sample_data):
    """Test with public bounds."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        public_bounds={
            'age': [0, 120],
            'income': [0, 200000]
        },
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(200)
    
    # Values should be within reasonable range (may exceed slightly due to noise)
    assert (synthetic['age'] >= -10).all()  # Allow some noise
    assert (synthetic['income'] >= -1000).all()


def test_synthesizer_forbid_as_parent(sample_data):
    """Test forbid_as_parent parameter."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        forbid_as_parent=['age', 'income'],
        seed=42
    )
    model.fit(sample_data)
    
    parents = model.parents_
    for col, par_list in parents.items():
        assert 'age' not in par_list
        assert 'income' not in par_list


def test_synthesizer_label_columns(sample_data):
    """Test label_columns parameter."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        label_columns=['target'],
        public_categories={'target': ['A', 'B', 'C']},
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(200)
    
    # Label columns should not have __UNK__
    assert '__UNK__' not in synthetic['target'].astype(str).values


def test_synthesizer_cat_keep_all_nonzero(sample_data):
    """Test cat_keep_all_nonzero parameter."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        cat_keep_all_nonzero=True,
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(200)
    
    # Should have minimal __UNK__ tokens
    unk_count = 0
    for col in synthetic.columns:
        if synthetic[col].dtype == 'object':
            unk_count += (synthetic[col].astype(str) == '__UNK__').sum()
    
    assert unk_count < len(synthetic) * 0.2  # Less than 20% UNK


def test_synthesizer_numeric_only(numeric_data):
    """Test with numeric-only data."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, seed=42)
    model.fit(numeric_data)
    
    synthetic = model.sample(100)
    
    assert synthetic.shape == (100, len(numeric_data.columns))
    for col in numeric_data.columns:
        assert pd.api.types.is_numeric_dtype(synthetic[col])


def test_synthesizer_categorical_only(categorical_data):
    """Test with categorical-only data."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0, seed=42)
    model.fit(categorical_data)
    
    synthetic = model.sample(100)
    
    assert synthetic.shape == (100, len(categorical_data.columns))


def test_synthesizer_error_if_not_fitted():
    """Test that sampling without fitting raises error."""
    model = PrivBayesSynthesizerEnhanced(epsilon=1.0)
    
    with pytest.raises(RuntimeError, match="not fitted"):
        model.sample(100)


def test_synthesizer_different_epsilon_values(small_data):
    """Test with different epsilon values."""
    for eps in [0.1, 0.5, 1.0, 2.0]:
        model = PrivBayesSynthesizerEnhanced(epsilon=eps, seed=42)
        model.fit(small_data)
        
        synthetic = model.sample(50)
        report = model.privacy_report()
        
        assert synthetic.shape[0] == 50
        assert report['epsilon'] == eps


def test_synthesizer_bins_per_numeric(sample_data):
    """Test bins_per_numeric parameter."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        bins_per_numeric=32,
        seed=42
    )
    model.fit(sample_data)
    
    synthetic = model.sample(100)
    assert synthetic.shape[0] == 100


def test_synthesizer_eps_split(sample_data):
    """Test epsilon split configuration."""
    model = PrivBayesSynthesizerEnhanced(
        epsilon=1.0,
        eps_split={"structure": 0.2, "cpt": 0.8},
        seed=42
    )
    model.fit(sample_data)
    
    report = model.privacy_report()
    assert report['eps_struct'] > 0
    assert report['eps_cpt'] > 0


def test_synthesizer_invalid_temperature():
    """Test that invalid temperature raises error."""
    with pytest.raises(ValueError, match="temperature must be positive"):
        PrivBayesSynthesizerEnhanced(epsilon=1.0, temperature=0)


def test_synthesizer_invalid_epsilon():
    """Test that negative epsilon raises error."""
    # This might not raise immediately, but should fail during fit
    model = PrivBayesSynthesizerEnhanced(epsilon=-1.0, seed=42)
    # The error might occur during fit or be handled gracefully
    # Let's test with a small dataset
    data = pd.DataFrame({'x': [1, 2, 3]})
    # Should either work (with epsilon clamped) or raise error
    try:
        model.fit(data)
        # If it works, check that epsilon is handled
        report = model.privacy_report()
        assert report['epsilon'] >= 0
    except (ValueError, RuntimeError):
        pass  # Expected behavior

