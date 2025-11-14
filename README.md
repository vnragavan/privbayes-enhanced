# Enhanced PrivBayes

Standalone implementation of a differentially private Bayesian network synthesizer. Generates synthetic data while preserving privacy guarantees and maintaining data utility.

## What it does

- Differential privacy with (ε,δ)-DP guarantees and proper budget tracking
- Handles numeric, categorical, boolean, datetime, and mixed data types
- **Automatic categorical name preservation**: By default, all categorical columns preserve actual category names (no hash buckets like B000, B001)
- Temperature-based sampling to reduce QI linkage risks
- Privacy-preserving categorical discovery using DP heavy hitters
- Smooth sensitivity bounds for numeric columns
- Option to keep all categories (reduces `__UNK__` tokens)
- Built-in metrics for utility, privacy, QI linkage, inference attacks, and DP compliance

## Installation

### Windows

```cmd
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Linux / macOS

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

**Note:** The code works on both Windows and Linux/macOS. All commands are cross-platform except for virtual environment activation.

### Running Without Installation (Windows)

You can also run the code **without installing** the package. See [docs/RUNNING_WITHOUT_INSTALLATION.md](docs/RUNNING_WITHOUT_INSTALLATION.md) for details.

**Quick example:**
```cmd
REM Set PYTHONPATH and run
set PYTHONPATH=%CD% && python -m privbayes_enhanced.cli data\adult.csv -o synthetic.csv --epsilon 1.0
```

Or use the helper script:
```cmd
run_cli.bat data\adult.csv -o synthetic.csv --epsilon 1.0
```

**Note:** You still need to install dependencies: `pip install -r requirements.txt`

## Quick Start

### Command Line

The `privbayes` command works the same on Windows, Linux, and macOS:

**Windows:**
```cmd
# Basic usage
privbayes data\adult.csv -o synthetic.csv --epsilon 1.0

# Generate data and all metrics
privbayes data\adult.csv -o synthetic.csv --epsilon 1.0 --all-metrics

# With public knowledge (use quotes for multi-line)
privbayes data\adult.csv -o synthetic.csv --epsilon 1.0 --public-categories state:CA,NY,TX,FL,IL --public-bounds age:0,120

# Disable auto-detection, use manual label columns
privbayes data\adult.csv -o synthetic.csv --epsilon 1.0 --no-auto-detect-label-columns --label-columns income,education

# Generate more samples
privbayes data\adult.csv -o synthetic.csv --epsilon 1.0 --n-samples 5000
```

**Linux / macOS:**
```bash
# Basic usage
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0

# Generate data and all metrics
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --all-metrics

# With public knowledge
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --public-categories state:CA,NY,TX,FL,IL \
  --public-bounds age:0,120

# Disable auto-detection, use manual label columns
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --no-auto-detect-label-columns --label-columns income,education

# Generate more samples
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --n-samples 5000
```

**Note:** On Windows, use backslashes (`\`) for paths. On Linux/macOS, use forward slashes (`/`). The `privbayes` command itself works identically on all platforms.

Full CLI docs are in [docs/CLI_USAGE.md](docs/CLI_USAGE.md).

### Python API

```python
from privbayes_enhanced import EnhancedPrivBayesAdapter
import pandas as pd

# Load data
data = pd.read_csv('data/adult.csv')

# Create and fit model (default: all categoricals preserve actual names)
model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=1000)

# Customize categorical handling
# Option 1: Disable auto-detection, use manual label columns
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    auto_detect_label_columns=False,  # Disable auto-detection
    label_columns=['income', 'education']  # Only these preserve names
)

# Option 2: Use default (all categoricals preserve names)
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    auto_detect_label_columns=True  # Default: all categoricals as labels
)

# Evaluate metrics
utility_metrics = model.evaluate_utility(data, synthetic)
privacy_metrics = model.evaluate_privacy(data, synthetic)
```

More examples in [docs/QUICK_START.md](docs/QUICK_START.md).

## Project Structure

```
privbayes-enhanced-standalone/
├── privbayes_enhanced/    # Main package
├── tests/                 # Test suite
├── examples/              # Example scripts
├── docs/                  # Documentation
├── data/                  # Sample data
└── test_data/            # Test datasets
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details on the project organization.

## Documentation

- [Quick Start](docs/QUICK_START.md) - Getting started
- [CLI Usage](docs/CLI_USAGE.md) - Command line reference
- [Metrics](docs/METRICS.md) - Understanding the metrics
- [Database Compatibility](docs/DATABASE_COMPATIBILITY.md) - Working with database exports
- [Privacy Audit](docs/PRIVACY_AUDIT.md) - Privacy evaluation
- [Testing](docs/TESTING.md) - Running tests
- [Hash Buckets Explained](docs/HASH_BUCKETS_EXPLAINED.md) - Understanding categorical hashing
- [Customizing Default Behavior](docs/CUSTOMIZING_DEFAULT_BEHAVIOR.md) - Changing default categorical handling

## Examples

Check out the `examples/` directory:

**Windows:**
```cmd
# Basic example
python examples\example.py

# Generate all metrics
python examples\generate_all_metrics.py

# K-anonymity demonstration
python examples\demo_k_anonymity.py
```

**Linux / macOS:**
```bash
# Basic example
python examples/example.py

# Generate all metrics
python examples/generate_all_metrics.py

# K-anonymity demonstration
python examples/demo_k_anonymity.py
```

## Testing

Run the test suite (works the same on all platforms):

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=privbayes_enhanced

# Run specific test file
pytest tests/test_adapter.py
```

**Note:** The `pytest` command works identically on Windows, Linux, and macOS.

More testing info in [docs/TESTING.md](docs/TESTING.md).

## Requirements

**Platform Support:** Windows, Linux, and macOS

**Python:** 3.8 or higher

**Dependencies:**
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0 (optional, needed for downstream metrics)

Full list in [requirements.txt](requirements.txt).

All dependencies are cross-platform and work on Windows, Linux, and macOS.

