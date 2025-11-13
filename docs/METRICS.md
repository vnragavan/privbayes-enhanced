# Utility and Privacy Metrics

Enhanced PrivBayes includes metrics for evaluating both the **utility** (data quality) and **privacy** characteristics of synthetic data.

## Overview

### Privacy Metrics

The privacy metrics are computed by the `privacy_report()` method and include:

- **Epsilon (ε)**: Total privacy budget used
- **Delta (δ)**: Privacy parameter for (ε,δ)-DP
- **Epsilon Allocation**: Breakdown of epsilon usage:
  - Structure learning
  - CPT (Conditional Probability Table) estimation
  - Metadata discovery (bounds, categories)
- **Mechanism Type**: Pure ε-DP or (ε,δ)-DP
- **Temperature**: QI-linkage reduction parameter

### Utility Metrics

The utility metrics compare real and synthetic data to assess data quality:

1. **Numeric Statistics**:
   - Mean error (relative)
   - Standard deviation error (relative)
   - Median error (relative)
   - Range preservation (min/max)
   - **Marginal Distributional Errors**:
     - Kolmogorov-Smirnov statistic (0 = identical, 1 = completely different)
     - Wasserstein distance (Earth Mover's Distance)
     - Jensen-Shannon divergence on binned data

2. **Categorical Similarity**:
   - Jensen-Shannon divergence (0 = identical, 1 = completely different)
   - Total Variation Distance (0 = identical, 1 = completely different)
   - Coverage (percentage of real categories present in synthetic data)
   - Unique value counts

3. **Correlation Preservation**:
   - Correlation of correlations (how well pairwise correlations are preserved)
   - Mean absolute error of correlations

4. **Privacy Heuristics**:
   - Exact match rate (lower is better)
   - UNK token rate

## Usage

### Python API

```python
import pandas as pd
from privbayes_enhanced import EnhancedPrivBayesAdapter

# Load data
data = pd.read_csv('data/adult.csv')

# Create and fit model
model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=1000)

# Evaluate utility metrics
utility_metrics = model.evaluate_utility(data, synthetic, verbose=True)

# Evaluate privacy metrics
privacy_metrics = model.evaluate_privacy(data, synthetic, verbose=True)

# Get privacy budget report
privacy_report = model.privacy_report()
print(f"Epsilon used: {privacy_report['epsilon_total']}")
```

### Command Line

```bash
# Generate synthetic data and evaluate utility
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --evaluate-utility

# Evaluate both utility and privacy metrics
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --evaluate-utility --evaluate-privacy

# Save metrics to JSON file
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --evaluate-utility --evaluate-privacy --save-metrics metrics.json
```

## Metric Details

### Privacy Budget Report

```python
{
    "epsilon_total": 1.0,
    "delta": 1e-6,
    "eps_struct": 0.27,
    "eps_cpt": 0.63,
    "eps_disc": 0.10,
    "temperature": 1.0
}
```

### Utility Metrics Structure

```python
{
    "summary": {
        "n_real_samples": 30162,
        "n_synthetic_samples": 1000,
        "n_numeric_columns": 6,
        "n_categorical_columns": 9,
        "mean_numeric_mean_error": 0.05,
        "mean_numeric_ks_statistic": 0.12,
        "mean_numeric_wasserstein_distance": 5.23,
        "mean_categorical_jsd": 0.15,
        "mean_categorical_tv_distance": 0.08,
        "mean_categorical_coverage": 0.95
    },
    "marginal_distributional_errors": {
        "age": {
            "type": "numeric",
            "kolmogorov_smirnov": 0.12,
            "wasserstein_distance": 5.23,
            "jensen_shannon_divergence": 0.08
        },
        "workclass": {
            "type": "categorical",
            "jensen_shannon_divergence": 0.12,
            "total_variation_distance": 0.08
        },
        ...
    },
    "numeric_statistics": {
        "age": {
            "mean_error": 0.02,
            "std_error": 0.05,
            "median_error": 0.03,
            "real_mean": 38.44,
            "synth_mean": 38.52,
            "kolmogorov_smirnov_statistic": 0.12,
            "kolmogorov_smirnov_pvalue": 0.05,
            "wasserstein_distance": 5.23,
            "jensen_shannon_divergence": 0.08
        },
        ...
    },
    "categorical_similarity": {
        "workclass": {
            "jensen_shannon_divergence": 0.12,
            "total_variation_distance": 0.08,
            "coverage": 0.95,
            "real_unique_count": 8,
            "synth_unique_count": 8
        },
        ...
    },
    "correlation_preservation": {
        "correlation_of_correlations": 0.89,
        "mean_absolute_error": 0.05
    }
}
```

### Privacy Metrics Structure

```python
{
    "exact_match_rate": 0.001,
    "n_exact_matches": 1,
    "unk_token_rate": 0.02,
    "n_unk_tokens": 20,
    "privacy_budget": {
        "epsilon_total": 1.0,
        "delta": 1e-6,
        "mechanism": "pure"
    }
}
```

## Interpreting Metrics

### Utility Metrics

- **Mean/Std Error < 0.1**: Good preservation of statistics
- **Kolmogorov-Smirnov Statistic < 0.2**: Good numeric distribution similarity
- **Wasserstein Distance**: Lower is better (depends on data scale)
- **Jensen-Shannon Divergence < 0.2**: Good distribution similarity (both numeric and categorical)
- **Total Variation Distance < 0.2**: Good categorical distribution similarity
- **Coverage > 0.9**: Most categories preserved
- **Correlation of Correlations > 0.8**: Good correlation preservation

### Privacy Metrics

- **Exact Match Rate < 0.01**: Very few exact copies (good for privacy)
- **UNK Token Rate**: Lower is better (indicates better category discovery)
- **Epsilon**: Lower values provide stronger privacy guarantees

## Example Output

```
================================================================================
Utility Metrics Report
================================================================================

Dataset Summary:
  Real samples: 30,162
  Synthetic samples: 1,000
  Numeric columns: 6
  Categorical columns: 9

Marginal Distributional Errors (Per Column):
  Average KS statistic (numeric): 0.1234 (0=identical, 1=different)
  Average Wasserstein distance (numeric): 5.23
  Average JSD (categorical): 0.1234 (0=identical, 1=different)

Numeric Column Statistics:
  Average mean error: 0.0234
  Average std error: 0.0456
  Average KS statistic: 0.1234
  Average Wasserstein distance: 5.23

  age:
    Mean error: 0.0021
    Std error: 0.0123
    Real mean: 38.44, Synthetic mean: 38.52
    KS statistic: 0.1234 (p=0.05)
    Wasserstein distance: 5.23
    JSD (binned): 0.08

Categorical Distribution Similarity:
  Average Jensen-Shannon divergence: 0.1234
    (0 = identical, 1 = completely different)
  Average coverage: 95.23%

Correlation Preservation:
  Correlation of correlations: 0.8912
    (1.0 = perfect preservation, 0 = no preservation)
  Mean absolute error: 0.0456

================================================================================
Privacy Metrics Report
================================================================================

Exact Match Rate: 0.0010
  (Lower is better - indicates fewer exact copies of real data)
  Exact matches: 1

UNK Token Rate: 0.0200
  UNK tokens: 20

Privacy Budget:
  Epsilon: 1.000000
  Delta: 1.00e-06
  Mechanism: pure

================================================================================
```

## Best Practices

1. **Always check privacy budget**: Ensure epsilon usage matches your requirements
2. **Evaluate utility after generation**: Use metrics to assess if synthetic data meets your needs
3. **Compare multiple runs**: Generate multiple synthetic datasets and compare metrics
4. **Use public knowledge**: Providing `public_categories` and `public_bounds` improves utility
5. **Balance privacy and utility**: Lower epsilon = better privacy but potentially lower utility

## Advanced Usage

### Custom Metrics

You can also use the metrics functions directly:

```python
from privbayes_enhanced.metrics import (
    compute_utility_metrics,
    compute_privacy_metrics,
    print_utility_report,
    print_privacy_report
)

# Compute metrics
utility = compute_utility_metrics(real_data, synthetic_data)
privacy = compute_privacy_metrics(real_data, synthetic_data, privacy_report)

# Print reports
print_utility_report(utility, verbose=True)
print_privacy_report(privacy)
```

### Saving Metrics

```python
import json

# Get all metrics
utility = model.evaluate_utility(data, synthetic, verbose=False)
privacy = model.evaluate_privacy(data, synthetic, verbose=False)
privacy_budget = model.privacy_report()

# Combine and save
all_metrics = {
    'utility': utility,
    'privacy': privacy,
    'privacy_budget': privacy_budget
}

with open('metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)
```

