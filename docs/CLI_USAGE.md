# Command-Line Interface Usage

Enhanced PrivBayes can be run directly from the command line using the `privbayes` command.

## Installation

After installing the package (with `pip install -e .`), the `privbayes` command will be available in your PATH.

## Basic Usage

### Generate Synthetic Data

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0
```

This will:
- Load data from `data/adult.csv`
- Fit a model with epsilon=1.0
- Generate synthetic data with the same number of rows
- Save to `synthetic.csv`

### Specify Number of Samples

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --n-samples 5000
```

### With Public Knowledge

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --public-categories state:CA,NY,TX,FL,IL \
  --public-categories education:HS,BS,MS,PhD \
  --public-bounds age:0,120 \
  --public-bounds income:0,200000 \
  --label-columns income
```

### Privacy Report

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --privacy-report
```

Output:
```
================================================================================
Privacy Report
================================================================================
Total epsilon used: 1.000000
Delta used: 1.00e-06
Epsilon for structure: 0.270000
Epsilon for CPT: 0.630000
Epsilon for discovery: 0.100000
Temperature: 1.00
================================================================================
```

### Save Privacy Report to File

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --save-report report.json
```

## Advanced Usage

### Save and Load Models

Save a fitted model:

```bash
privbayes data/adult.csv --fit-model model.pkl --epsilon 1.0
```

Load a saved model and generate more samples:

```bash
privbayes --load-model model.pkl -o synthetic.csv --n-samples 10000
```

This is useful when you want to:
- Fit once, generate many times
- Share models without sharing data
- Generate different sized datasets from the same model

### Temperature for QI-Linkage Reduction

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --temperature 1.5
```

Higher temperature (T>1) reduces linkage attacks but may reduce utility.

### Epsilon Split

Control how epsilon is allocated:

```bash
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --eps-split 0.3:0.7
```

This allocates 30% to structure learning and 70% to CPT estimation.

### Configuration File

Create a JSON config file:

```json
{
  "epsilon": 1.0,
  "delta": 1e-6,
  "seed": 42,
  "temperature": 1.5,
  "max_parents": 3,
  "bins_per_numeric": 32,
  "public_categories": {
    "state": ["CA", "NY", "TX", "FL", "IL"],
    "education": ["HS", "BS", "MS", "PhD"]
  },
  "public_bounds": {
    "age": [0, 120],
    "income": [0, 200000]
  },
  "label_columns": ["income"],
  "eps_split": {
    "structure": 0.3,
    "cpt": 0.7
  }
}
```

Use it:

```bash
privbayes data/adult.csv -o synthetic.csv --config config.json
```

## Command-Line Options

### Input/Output

- `input_file` - Input CSV file (required unless using `--load-model`)
- `-o, --output` - Output CSV file for synthetic data
- `--fit-model` - Save fitted model to pickle file
- `--load-model` - Load fitted model from pickle file

### Privacy Parameters

- `--epsilon` - Privacy budget (default: 1.0)
- `--delta` - Privacy parameter (default: 1e-6)
- `--seed` - Random seed (default: 42)

### Sampling

- `--n-samples` - Number of synthetic samples (default: same as input)
- `--temperature` - Temperature for sampling (default: 1.0)

### Public Knowledge

- `--public-categories` - Public categories as `col:val1,val2,...` (can repeat)
- `--public-bounds` - Public bounds as `col:min,max` (can repeat)
- `--label-columns` - Label columns (comma-separated)

### Model Configuration

- `--max-parents` - Maximum parents in Bayesian network (default: 2)
- `--bins-per-numeric` - Number of bins for numeric columns (default: 16)
- `--eps-split` - Epsilon split as `structure:CPT` (e.g., 0.3:0.7)
- `--cat-keep-all-nonzero` - Keep all observed categories (default: True)
- `--no-cat-keep-all-nonzero` - Disable keeping all categories

### Output Options

- `--privacy-report` - Print privacy report to stdout
- `--save-report` - Save privacy report to JSON file
- `-v, --verbose` - Verbose output

### Configuration

- `--config` - JSON configuration file path

## Examples

### Example 1: Quick Test

```bash
privbayes data/adult.csv -o test.csv --epsilon 1.0 --n-samples 100
```

### Example 2: Production Run with Public Knowledge

```bash
privbayes data/adult.csv \
  -o synthetic_adult.csv \
  --epsilon 2.0 \
  --n-samples 50000 \
  --public-categories state:CA,NY,TX,FL,IL \
  --public-categories education:HS,BS,MS,PhD \
  --public-bounds age:0,120 \
  --public-bounds income:0,200000 \
  --label-columns income \
  --temperature 1.5 \
  --privacy-report \
  --save-report privacy.json \
  --verbose
```

### Example 3: Fit Once, Generate Many Times

```bash
# Fit and save model
privbayes data/adult.csv --fit-model adult_model.pkl --epsilon 1.0

# Generate different sized datasets
privbayes --load-model adult_model.pkl -o synthetic_1k.csv --n-samples 1000
privbayes --load-model adult_model.pkl -o synthetic_10k.csv --n-samples 10000
privbayes --load-model adult_model.pkl -o synthetic_100k.csv --n-samples 100000
```

### Example 4: Using Configuration File

```bash
# Create config.json (see above for format)
privbayes data/adult.csv -o synthetic.csv --config config.json --verbose
```

## Tips

1. **Use `--verbose`** to see progress and details
2. **Save models** if you need to generate multiple datasets
3. **Provide public knowledge** to minimize `__UNK__` tokens
4. **Use `--privacy-report`** to verify epsilon usage
5. **Start with small `--n-samples`** for testing
6. **Use `--temperature > 1`** for additional privacy protection

## Troubleshooting

### Command not found

Make sure you've installed the package:
```bash
pip install -e .
```

And activated your virtual environment if using one:
```bash
source venv/bin/activate
```

### File not found

Use absolute paths or paths relative to your current directory:
```bash
privbayes /full/path/to/data.csv -o /full/path/to/output.csv
```

### Too many UNK tokens

- Use `--public-categories` for known domains
- Use `--label-columns` for target variables
- Ensure `--cat-keep-all-nonzero` is enabled (default)

