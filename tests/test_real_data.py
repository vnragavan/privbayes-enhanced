"""Tests using real data from the data folder."""

import pytest
import pandas as pd
import os
from privbayes_enhanced import EnhancedPrivBayesAdapter
from privbayes_enhanced.synthesizer import PrivBayesSynthesizerEnhanced


@pytest.fixture
def adult_data():
    """Load the adult dataset from data folder."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'adult.csv')
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def test_adult_data_basic(adult_data):
    """Test basic functionality with adult dataset."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(adult_data)
    
    synthetic = model.sample(n_samples=100)
    
    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape[0] == 100
    assert set(synthetic.columns) == set(adult_data.columns)


def test_adult_data_with_public_knowledge(adult_data):
    """Test with public knowledge about adult dataset."""
    # Common public knowledge about the Adult dataset
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        public_categories={
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 
                         'Federal-gov', 'Local-gov', 'State-gov', 
                         'Without-pay', 'Never-worked'],
            'education': ['Bachelors', 'Some-college', '11th', 'HS-grad',
                         'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                         '7th-8th', '12th', 'Masters', '1st-4th', '10th',
                         'Doctorate', '5th-6th', 'Preschool'],
            'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married',
                              'Separated', 'Widowed', 'Married-spouse-absent',
                              'Married-AF-spouse'],
            'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                            'Other-relative', 'Unmarried'],
            'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
                     'Other', 'Black'],
            'sex': ['Female', 'Male'],
            'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico',
                              'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)',
                              'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
                              'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
                              'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                              'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                              'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                              'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
                              'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                              'Holand-Netherlands'],
            'income': ['<=50K', '>50K']
        },
        public_bounds={
            'age': [0, 120],
            'fnlwgt': [0, 1e7],
            'education-num': [1, 16],
            'capital-gain': [0, 1e6],
            'capital-loss': [0, 1e6],
            'hours-per-week': [0, 100]
        },
        label_columns=['income'],
        cat_keep_all_nonzero=True,
        seed=42
    )
    
    model.fit(adult_data)
    
    synthetic = model.sample(n_samples=500)
    
    # Check that no __UNK__ tokens in label column
    assert '__UNK__' not in synthetic['income'].astype(str).values
    
    # Check that values are within reasonable bounds
    assert (synthetic['age'] >= 0).all()
    assert (synthetic['age'] <= 120).all()
    
    # Check privacy report
    report = model.privacy_report()
    assert report['epsilon_total'] > 0


def test_adult_data_privacy_report(adult_data):
    """Test privacy report with adult dataset."""
    model = EnhancedPrivBayesAdapter(epsilon=1.0, delta=1e-6, seed=42)
    model.fit(adult_data)
    
    report = model.privacy_report()
    
    assert isinstance(report, dict)
    assert 'epsilon_total' in report
    assert 'delta' in report
    assert report['epsilon_total'] > 0


def test_adult_data_different_epsilon(adult_data):
    """Test with different epsilon values on adult dataset."""
    for eps in [0.5, 1.0, 2.0]:
        model = EnhancedPrivBayesAdapter(epsilon=eps, seed=42)
        model.fit(adult_data)
        
        synthetic = model.sample(n_samples=100)
        report = model.privacy_report()
        
        assert synthetic.shape[0] == 100
        assert report['epsilon_total'] > 0


def test_adult_data_subsample(adult_data):
    """Test with a subsample of adult data for faster testing."""
    # Use first 1000 rows for faster testing
    small_data = adult_data.head(1000)
    
    model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
    model.fit(small_data)
    
    synthetic = model.sample(n_samples=500)
    
    assert synthetic.shape[0] == 500
    assert set(synthetic.columns) == set(small_data.columns)


def test_adult_data_temperature(adult_data):
    """Test temperature parameter with adult dataset."""
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        temperature=1.5,
        seed=42
    )
    model.fit(adult_data)
    
    synthetic = model.sample(n_samples=200)
    report = model.privacy_report()
    
    assert synthetic.shape[0] == 200
    assert report['temperature'] == 1.5

