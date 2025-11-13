# Testing Guide for Enhanced PrivBayes

This guide explains how to test the Enhanced PrivBayes code with your data.

## Quick Start

### 1. Install Dependencies

First, install the required packages:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov  # For running tests
```

### 2. Test with Your Data

The easiest way to test with your data is using the provided test script:

```bash
python test_with_data.py
```

Or specify your data file:

```bash
python test_with_data.py data/adult.csv 1000 1.0
# Arguments: <data_file> <n_samples> <epsilon>
```

### 3. Run Automated Tests

Run all automated tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_adapter.py
pytest tests/test_real_data.py  # Tests with your adult.csv data
```

Run with coverage report:

```bash
pytest --cov=privbayes_enhanced --cov-report=html
```

## Test Structure

The test suite includes:

1. **Unit Tests** (`tests/test_adapter.py`, `tests/test_synthesizer.py`)
   - Test individual components
   - Test various configuration options
   - Test error handling

2. **Integration Tests** (`tests/test_integration.py`)
   - Test complete workflows
   - Test reproducibility
   - Test edge cases

3. **Real Data Tests** (`tests/test_real_data.py`)
   - Test with your actual data files
   - Test with public knowledge configurations
   - Test privacy reports

## Testing Your Own Data

### Using the Test Script

1. Place your CSV file in the `data/` folder (or provide full path)
2. Run the test script:

```python
python test_with_data.py path/to/your/data.csv 500 1.0
```

### Using Python Directly

```python
import pandas as pd
from privbayes_enhanced import EnhancedPrivBayesAdapter

# Load your data
data = pd.read_csv('data/your_data.csv')

# Create and fit model
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    delta=1e-6,
    seed=42,
    cat_keep_all_nonzero=True
)
model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=1000)

# Check privacy report
report = model.privacy_report()
print(f"Epsilon used: {report['epsilon_total']}")
```

### With Public Knowledge

If you know public information about your data (e.g., all possible states, age bounds), provide it:

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'state': ['CA', 'NY', 'TX', ...],  # All possible states
        'education': ['HS', 'BS', 'MS', ...]
    },
    public_bounds={
        'age': [0, 120],      # Known bounds
        'income': [0, 1e6]
    },
    label_columns=['target'],  # Target variable (no hashing)
    cat_keep_all_nonzero=True
)
model.fit(data)
synthetic = model.sample(n_samples=1000)
```

## What to Check

When testing, verify:

1. **Model Fits Successfully**: No errors during `fit()`
2. **Synthetic Data Generated**: Correct shape and columns
3. **Privacy Budget**: Check `privacy_report()` to see epsilon usage
4. **No Excessive __UNK__ Tokens**: With `cat_keep_all_nonzero=True`, should be minimal
5. **Data Types Preserved**: Numeric columns remain numeric, categorical remain categorical
6. **Reproducibility**: Same seed produces same results

## Common Issues

### Issue: Too many __UNK__ tokens

**Solution**: 
- Use `cat_keep_all_nonzero=True` (default in adapter)
- Provide `public_categories` for known domains
- Use `label_columns` for target variables

### Issue: Out of memory

**Solution**:
- Reduce `bins_per_numeric` (default 16)
- Reduce `max_parents` (default 2)
- Use smaller dataset for testing

### Issue: Values outside expected range

**Solution**:
- Provide `public_bounds` for numeric columns
- Check that bounds are reasonable

## Running Specific Tests

```bash
# Test only adapter
pytest tests/test_adapter.py

# Test only with real data
pytest tests/test_real_data.py

# Test with specific marker
pytest -m integration

# Run tests in parallel (faster)
pytest -n auto

# Show print statements
pytest -s

# Stop on first failure
pytest -x
```

## Continuous Integration

For CI/CD, you can run:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest --cov=privbayes_enhanced --cov-report=term-missing

# Check coverage threshold
pytest --cov=privbayes_enhanced --cov-fail-under=80
```

## Example Test Output

When you run `python test_with_data.py`, you should see:

```
================================================================================
Testing Enhanced PrivBayes with Your Data
================================================================================

Loading data from: data/adult.csv
Original data shape: (32561, 15)
Columns: ['age', 'workclass', 'fnlwgt', 'education', ...]

Creating Enhanced PrivBayes model...
Fitting model to data...
✓ Model fitted successfully

Generating 500 synthetic samples...
✓ Generated synthetic data shape: (500, 15)

Checking for __UNK__ tokens...
  ✓ No __UNK__ tokens found

Privacy Report:
  Total epsilon used: 1.0000
  Delta used: 1.00e-06
  Epsilon for structure: 0.2700
  Epsilon for CPT: 0.6300
  Epsilon for discovery: 0.1000

================================================================================
Test completed successfully!
================================================================================
```

## Next Steps

After testing:

1. Adjust `epsilon` based on your privacy requirements
2. Provide `public_categories` and `public_bounds` if available
3. Tune `temperature` for QI-linkage reduction (T>1 = more privacy)
4. Use `label_columns` for target variables
5. Generate larger synthetic datasets as needed

