# Downstream Task Metrics

## Overview

Downstream task metrics evaluate how well synthetic data preserves relationships needed for **machine learning tasks** like classification and regression. Unlike statistical similarity metrics, downstream metrics test whether the synthetic data can be used effectively for the same ML tasks as the real data.

## What Are Downstream Metrics?

Downstream metrics answer the question: **"Can I train a model on synthetic data and get similar performance to training on real data?"**

### Key Concepts

1. **Train on Synthetic, Test on Real**: Models are trained on synthetic data, then evaluated on a held-out test set from real data
2. **Performance Gap**: The difference in model performance between training on real vs synthetic data
3. **Feature Importance Preservation**: Whether the same features are important in models trained on real vs synthetic data

## Metrics Computed

### For Classification Tasks

- **Accuracy**: Percentage of correct predictions - **Higher is better** (1.0 = perfect)
- **F1-Score**: Harmonic mean of precision and recall (weighted average for multi-class) - **Higher is better** (1.0 = perfect)
- **ROC-AUC**: Area under the ROC curve (binary classification only) - **Higher is better** (1.0 = perfect)
- **Performance Gap**: Difference in accuracy between real and synthetic training - **Lower is better** (closer to 0)

### For Regression Tasks

- **MSE (Mean Squared Error)**: Average squared prediction error - **Lower is better** (closer to 0)
- **MAE (Mean Absolute Error)**: Average absolute prediction error - **Lower is better** (closer to 0)
- **R² Score**: Coefficient of determination - **Higher is better** (1.0 = perfect, 0 = baseline)
- **Performance Gap**: Difference in R² score between real and synthetic training - **Lower is better** (closer to 0)

### Feature Importance

- **Cosine Similarity**: How similar are feature importances between real and synthetic models - **Higher is better** (1.0 = identical)
- **Top Features**: Which features are most important in each model - **Should match between real and synthetic**

## Usage

### Python API

```python
from privbayes_enhanced import EnhancedPrivBayesAdapter
import pandas as pd

# Load and fit model
data = pd.read_csv('data/adult.csv')
model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=1000)

# Evaluate downstream metrics
downstream_metrics = model.evaluate_downstream(
    real_data=data,
    synthetic_data=synthetic,
    target_column='income',  # Auto-detected if None
    task_type='classification',  # Auto-detected if None
    verbose=True
)
```

### Command Line

```bash
privbayes fit data/adult.csv --epsilon 1.0 --output model.pkl
privbayes sample model.pkl --n-samples 1000 --output synthetic.csv
privbayes evaluate-downstream data/adult.csv synthetic.csv --target income
```

## Example Output

```
================================================================================
Downstream Task Performance Report
================================================================================

Target Column: income
Task Type: classification
Models Tested: RandomForest, LogisticRegression

================================================================================
Model Performance Comparison
================================================================================

RandomForest:
  Real Data Performance:
    accuracy: 0.8523 (Higher is better, 1.0 = perfect)
    f1_score: 0.8412 (Higher is better, 1.0 = perfect)
    roc_auc: 0.9123 (Higher is better, 1.0 = perfect)
  Synthetic Data Performance:
    accuracy: 0.8234 (Higher is better, 1.0 = perfect)
    f1_score: 0.8123 (Higher is better, 1.0 = perfect)
    roc_auc: 0.8901 (Higher is better, 1.0 = perfect)
  Performance Gap: 0.0289 (relative: 3.39%) (Lower is better, closer to 0)

LogisticRegression:
  Real Data Performance:
    accuracy: 0.8234 (Higher is better, 1.0 = perfect)
    f1_score: 0.8123 (Higher is better, 1.0 = perfect)
    roc_auc: 0.8901 (Higher is better, 1.0 = perfect)
  Synthetic Data Performance:
    accuracy: 0.8012 (Higher is better, 1.0 = perfect)
    f1_score: 0.7890 (Higher is better, 1.0 = perfect)
    roc_auc: 0.8723 (Higher is better, 1.0 = perfect)
  Performance Gap: 0.0222 (relative: 2.70%) (Lower is better, closer to 0)

================================================================================
Feature Importance Preservation
================================================================================

RandomForest:
  Cosine Similarity: 0.9234 (Higher is better, 1.0 = identical, 0 = different)

================================================================================
Summary
================================================================================

Average Accuracy Gap: 0.0256
Best Preserving Model: LogisticRegression
================================================================================
```

## Interpretation

### Performance Gaps

- **< 0.01 (1%)**: Excellent - synthetic data preserves ML patterns very well (**Lower is better**, closer to 0)
- **0.01 - 0.05 (1-5%)**: Good - synthetic data is usable for ML tasks (**Lower is better**)
- **0.05 - 0.10 (5-10%)**: Acceptable - some utility loss but still usable (**Lower is better**)
- **> 0.10 (10%)**: Poor - significant utility loss for ML tasks (**Lower is better**)

### Feature Importance Similarity

- **> 0.9**: Excellent - same features are important (**Higher is better**, 1.0 = identical)
- **0.7 - 0.9**: Good - similar feature importance (**Higher is better**)
- **0.5 - 0.7**: Acceptable - some differences in feature importance (**Higher is better**)
- **< 0.5**: Poor - different features are important (**Higher is better**)

## Why This Matters

### Use Cases

1. **ML Model Training**: Can synthetic data replace real data for training?
2. **Feature Engineering**: Are the same features important in synthetic data?
3. **Model Validation**: Does synthetic data preserve predictive relationships?
4. **Data Augmentation**: Can synthetic data improve model performance?

### Advantages Over Statistical Metrics

- **Task-Specific**: Evaluates utility for actual ML use cases
- **Relationship Preservation**: Tests whether predictive relationships are preserved
- **Practical Relevance**: Directly measures whether synthetic data is usable for ML

## Limitations

1. **Requires scikit-learn**: Must install `scikit-learn>=1.0.0`
2. **Computational Cost**: Training multiple models takes time
3. **Model-Dependent**: Results depend on which models are tested
4. **Task-Specific**: Only evaluates utility for the specified task

## Best Practices

1. **Use Multiple Models**: Test both tree-based (RandomForest) and linear (LogisticRegression/LinearRegression) models
2. **Adequate Sample Size**: Ensure synthetic data has enough samples for training
3. **Proper Train/Test Split**: Use a held-out test set from real data
4. **Task Selection**: Choose the task type that matches your use case
5. **Compare with Baseline**: Compare synthetic performance to real data performance

## Integration with Other Metrics

Downstream metrics complement other evaluation metrics:

- **Utility Metrics**: Statistical similarity (KS, JSD, correlations)
- **Privacy Metrics**: Exact match rate, UNK tokens
- **QI Linkage**: Re-identification risk
- **Inference Attack**: Attribute inference risk
- **DP Audit**: Formal privacy guarantees

Together, these provide a complete evaluation of synthetic data quality and privacy.

