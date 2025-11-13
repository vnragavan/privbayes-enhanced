"""Shared test fixtures for Enhanced PrivBayes tests."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 20000, n).clip(0),
        'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n),
        'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
        'married': np.random.choice([True, False], n),
        'target': np.random.choice(['A', 'B', 'C'], n)
    })
    return data


@pytest.fixture
def small_data():
    """Create small dataset for quick tests."""
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x': np.random.randint(0, 10, n),
        'y': np.random.choice(['A', 'B'], n),
        'z': np.random.normal(0, 1, n)
    })
    return data


@pytest.fixture
def numeric_data():
    """Create numeric-only dataset."""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(5, 2, n),
        'x3': np.random.randint(0, 100, n)
    })
    return data


@pytest.fixture
def categorical_data():
    """Create categorical-only dataset."""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'cat1': np.random.choice(['A', 'B', 'C'], n),
        'cat2': np.random.choice(['X', 'Y'], n),
        'cat3': np.random.choice(['1', '2', '3', '4'], n)
    })
    return data


@pytest.fixture
def mixed_data():
    """Create mixed-type dataset."""
    np.random.seed(42)
    n = 150
    data = pd.DataFrame({
        'numeric': np.random.normal(0, 1, n),
        'categorical': np.random.choice(['A', 'B'], n),
        'boolean': np.random.choice([True, False], n),
        'integer': np.random.randint(0, 10, n)
    })
    return data

