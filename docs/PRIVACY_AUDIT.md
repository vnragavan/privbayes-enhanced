# Privacy Audit and Advanced Privacy Metrics

Enhanced PrivBayes includes privacy audit tools and advanced privacy metrics for evaluating QI linkage attacks, inference attacks, and differential privacy compliance.

## Overview

### QI Linkage Risk Metrics

Quasi-Identifier (QI) linkage attacks occur when an attacker uses publicly available information (like age, zipcode, gender) to match synthetic records to real individuals.

Metrics computed:
- **Exact QI Match Rate**: Percentage of synthetic QI combinations that exist in real data - **Lower is better** (fewer matches = better privacy)
- **K-Anonymity Violations**: Number of QI combinations with ≤k matches in real data - **Lower is better** (fewer violations = better k-anonymity)
- **Linkage Risk Rate**: Percentage of risky QI combinations (low frequency in real data) - **Lower is better** (lower risk = better privacy)
- **QI Combination Statistics**: Unique combinations in real vs synthetic data - **Closer match is better**

### Inference Attack Risk Metrics

Inference attacks occur when an attacker tries to infer sensitive attributes (like income, disease status) from quasi-identifiers.

Metrics computed:
- **Unique Inference Rate**: Percentage of QI combinations that uniquely determine sensitive values - **Lower is better** (fewer unique inferences = better privacy)
- **Distribution Error**: KL divergence between real and synthetic conditional distributions - **Lower is better** (closer to 0 = better preservation)
- **Per-Column Analysis**: Risk assessment for each sensitive column - **Lower risk is better**

### DP Compliance Audit

**Important**: The audit verifies **implementation compliance** with the reference design, not formal mathematical proofs. See [AUDIT_GUARANTEES.md](AUDIT_GUARANTEES.md) for details on what the audit guarantees and its limitations.

Verifies that the implementation follows DP-compliant patterns by checking:
- Epsilon accounting (sum doesn't exceed budget)
- Delta bounds (within acceptable range)
- Mechanism type (pure ε-DP or (ε,δ)-DP)
- Configuration privacy leaks
- Metadata DP compliance

#### Reference Design Checklist

The audit includes a checklist based on the reference PrivBayes design:

1. **Numeric bounds**: Smooth-sensitivity DP quantiles for (α,1-α); no private clamping; optional clamp to coarse public box
2. **Binning**: Fixed bin counts on DP bounds (pure post-processing)
3. **Categorical domain**: DP hash-bucket heavy hitters (noised counts) + `__UNK__`; bounded alphabet
4. **Structure utilities**: MI computed from DP joint counts; selection by top-u or EM over these DP utilities
5. **Sensitivity use**: Count sensitivity = 1 under add/remove; Laplace scales 1/ε per cell; no ad-hoc rescaling
6. **CPT estimation**: Laplace(1/ε_var) to CPT counts, clip ≥ 0, smooth, normalize per row
7. **Composition**: Explicit split (ε_disc, ε_struct, ε_cpt); δ tracked for smooth bounds; no fold-back of ε_disc
8. **Hyperparameter tuning**: Heuristics depend only on (n,d,ε); no raw statistics; can fix n to public bound if needed
9. **Logging**: Privacy ledger only (privacy_report); no raw min/max, no raw MI, no unnoised counts
10. **Adjacency**: Add/remove (unbounded) explicitly recorded; sensitivity and noise calibrated accordingly

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

# Evaluate QI linkage risk
qi_metrics = model.evaluate_qi_linkage(
    data, synthetic,
    qi_columns=['age', 'sex', 'race'],  # Optional: auto-detected if None
    k=5,  # k-anonymity parameter
    verbose=True
)

# Evaluate inference attack risk
inference_metrics = model.evaluate_inference_attack(
    data, synthetic,
    sensitive_columns=['income'],  # Optional: auto-detected if None
    qi_columns=['age', 'sex', 'race'],  # Optional: auto-detected if None
    verbose=True
)

# Audit DP compliance
audit_results = model.audit_dp_compliance(strict=True, verbose=True)
```

### Command Line

```bash
# Evaluate QI linkage risk
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --evaluate-qi-linkage \
  --qi-columns age,sex,race

# Evaluate inference attack risk
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --evaluate-inference \
  --sensitive-columns income \
  --qi-columns age,sex,race

# Audit DP compliance
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --audit-dp

# All privacy evaluations
privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \
  --evaluate-privacy \
  --evaluate-qi-linkage \
  --evaluate-inference \
  --audit-dp \
  --qi-columns age,sex,race \
  --sensitive-columns income \
  --save-metrics privacy_metrics.json
```

## Metric Details

### QI Linkage Risk Metrics

```python
{
    "qi_columns": ["age", "sex", "race"],
    "n_qi_columns": 3,
    "unique_real_qi_combinations": 150,
    "unique_synth_qi_combinations": 120,
    "exact_qi_match_rate": 0.85,
    "n_exact_qi_matches": 102,
    "k_anonymity_violation_rate": 0.15,
    "k_anonymity_violations": 18,
    "k_parameter": 5,
    "linkage_risk_rate": 0.10,
    "n_linkage_risky_combinations": 12
}
```

**Interpretation** (all metrics: **Lower is better**):
- **Exact QI Match Rate < 0.5**: Good (fewer exact matches = better privacy)
- **K-Anonymity Violation Rate < 0.2**: Good (most combinations have k+ matches = better k-anonymity)
- **Linkage Risk Rate < 0.1**: Good (few risky combinations = lower linkage attack risk)

### Inference Attack Risk Metrics

```python
{
    "sensitive_columns": ["income"],
    "qi_columns": ["age", "sex", "race"],
    "inference_risks": {
        "income": {
            "unique_inference_rate": 0.12,
            "n_unique_inferences": 18,
            "n_qi_combinations": 150,
            "avg_distribution_error": 0.05
        }
    },
    "avg_unique_inference_rate": 0.12
}
```

**Interpretation** (all metrics: **Lower is better**):
- **Unique Inference Rate < 0.2**: Good (few unique inferences possible = better privacy)
- **Distribution Error < 0.1**: Good (similar conditional distributions = better utility preservation)

### DP Compliance Audit

```python
{
    "passed": True,
    "warnings": [
        "Delta (1.00e-06) is relatively large. Consider delta < 1e-5 for strong privacy."
    ],
    "errors": [],
    "checks": {
        "epsilon_accounting": {
            "configured": 1.0,
            "actual": 0.95,
            "sum": 0.95,
            "within_budget": True
        },
        "delta_bounds": {
            "delta": 1e-6,
            "acceptable": True
        },
        "mechanism": {
            "type": "pure",
            "valid": True
        },
        "temperature": {
            "value": 1.5,
            "recommended": True
        },
        "metadata_dp": {
            "dp_used": True,
            "mode": "dp_bounds_smooth",
            "compliant": True
        }
    }
}
```

## Example Output

### QI Linkage Risk Report

```
================================================================================
QI Linkage Risk Report
================================================================================

Quasi-Identifier Columns: age, sex, race
Number of QI columns: 3

QI Combination Statistics:
  Unique real QI combinations: 150
  Unique synthetic QI combinations: 120
  Exact QI matches: 102
  Exact QI match rate: 0.8500

K-Anonymity Analysis (k=5):
  K-anonymity violations: 18
  Violation rate: 0.1500 (Lower is better - indicates better k-anonymity protection)

Linkage Risk:
  Risky QI combinations: 12
  Linkage risk rate: 0.1000 (Lower is better - indicates lower linkage attack risk)

================================================================================
```

### Inference Attack Risk Report

```
================================================================================
Inference Attack Risk Report
================================================================================

Sensitive Columns: income
QI Columns: age, sex, race

Average Unique Inference Rate: 0.1200 (Lower is better - indicates lower inference attack risk)

Per-Column Inference Risks:

  income:
    Unique inference rate: 0.1200
    Unique inferences: 18
    QI combinations: 150
    Avg distribution error: 0.0500

================================================================================
```

### DP Compliance Audit Report

```
================================================================================
Differential Privacy Compliance Audit
================================================================================

Overall Status: PASSED

================================================================================
Reference Design Checklist Compliance
================================================================================

Compliance: 10/10 items passed (100.0%)

  ✓ [PASS] Numeric bounds: Smooth-sensitivity DP quantiles for (α,1-α); no private clamping
  ✓ [PASS] Binning: Fixed bin counts on DP bounds (pure post-processing)
  ✓ [PASS] Categorical domain: DP hash-bucket heavy hitters (noised counts) + __UNK__
  ✓ [PASS] Structure utilities: MI computed from DP joint counts
  ✓ [PASS] Sensitivity use: Count sensitivity = 1 under add/remove; Laplace scales 1/ε
  ✓ [PASS] CPT estimation: Laplace(1/ε_var) to CPT counts, clip ≥ 0, smooth, normalize
  ✓ [PASS] Composition: Explicit split (ε_disc, ε_struct, ε_cpt); δ tracked; no fold-back
  ✓ [PASS] Hyperparameter tuning: Heuristics depend only on (n,d,ε); no raw statistics
  ✓ [PASS] Logging: Privacy ledger only (privacy_report); no raw statistics
  ✓ [PASS] Adjacency: Add/remove (unbounded) explicitly recorded

================================================================================
Detailed Checks:
  ✓ epsilon_accounting:
      configured: 1.0
      actual: 0.95
      sum: 0.95
      within_budget: True
  ✓ delta_bounds:
      delta: 1e-06
      acceptable: True
  ✓ mechanism:
      type: (ε,δ)-DP
      valid: True
  ✓ temperature:
      value: 1.5
      recommended: True
  ✓ metadata_dp:
      dp_used: True
      mode: dp_bounds_smooth
      compliant: True

================================================================================
```

## Best Practices

1. **Always run DP audit**: Verify compliance before deploying
2. **Evaluate QI linkage**: Especially important for datasets with demographic data
3. **Check inference attacks**: Critical for sensitive attributes
4. **Use temperature > 1**: Reduces QI linkage risk
5. **Specify QI columns**: More accurate risk assessment
6. **Monitor k-anonymity**: Lower violation rates indicate better protection

## Reducing Privacy Risks

### For QI Linkage

- **Increase temperature**: `temperature > 1.0` flattens distributions
- **Use public knowledge**: Provide `public_categories` and `public_bounds`
- **Coarsen QI columns**: Reduce granularity of QI attributes
- **Forbid QI as parents**: Use `forbid_as_parent` parameter

### For Inference Attacks

- **Increase epsilon**: More budget allows better utility without unique inferences
- **Use label columns**: Protect sensitive attributes with `label_columns`
- **Increase temperature**: Reduces correlation between QI and sensitive attributes
- **Provide public categories**: For sensitive columns when possible

### For DP Compliance

- **Use strict_dp=True**: Ensures all operations are DP-compliant
- **Avoid original_data_bounds**: Don't reveal exact data ranges
- **Monitor epsilon usage**: Ensure sum doesn't exceed budget
- **Use appropriate delta**: Keep delta < 1e-5 for strong privacy

## Advanced Usage

### Custom QI Detection

```python
# Manually specify QI columns
qi_metrics = model.evaluate_qi_linkage(
    data, synthetic,
    qi_columns=['age', 'zipcode', 'gender'],
    k=10  # 10-anonymity
)
```

### Custom Sensitive Columns

```python
# Manually specify sensitive columns
inference_metrics = model.evaluate_inference_attack(
    data, synthetic,
    sensitive_columns=['income', 'disease_status'],
    qi_columns=['age', 'zipcode']
)
```

### Strict Audit

```python
# Fail on any violations
audit_results = model.audit_dp_compliance(strict=True)

if not audit_results['passed']:
    print("DP compliance failed!")
    for error in audit_results['errors']:
        print(f"  Error: {error}")
```

