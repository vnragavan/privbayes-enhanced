# Quick Start Guide

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## How to Use

### 1. Activate the Virtual Environment

```bash
source venv/bin/activate
```

### 2. Command Line Interface

Generate synthetic data from the command line:

```bash
# Basic usage
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0

# With privacy report
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --privacy-report

# Generate more samples
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --n-samples 5000

# With public knowledge
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --public-categories state:CA,NY,TX,FL,IL \
  --public-bounds age:0,120
```

See [CLI_USAGE.md](CLI_USAGE.md) for complete CLI documentation.

### 3. Run Tests

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_real_data.py -v
```

### 4. Use in Python Code

```python
import pandas as pd
from privbayes_enhanced import EnhancedPrivBayesAdapter

# Load your data
data = pd.read_csv('data/adult.csv')

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

## Running Tests

The test suite includes:
- Unit tests for adapter and synthesizer
- Integration tests for complete workflows
- Real data tests
- Edge case handling

## Next Steps

1. Adjust privacy budget: modify `epsilon` based on your privacy requirements
2. Add public knowledge: provide `public_categories` and `public_bounds` if available
3. Tune parameters: adjust `temperature`, `max_parents`, etc. for your use case
4. Generate larger datasets: use `sample(n_samples=...)` to generate more synthetic data

## Project Structure

- `venv/` - Virtual environment (activate with `source venv/bin/activate`)
- `tests/` - Test suite
- `examples/` - Example scripts
- `docs/` - Documentation

## Deactivate Virtual Environment

When done:

```bash
deactivate
```

