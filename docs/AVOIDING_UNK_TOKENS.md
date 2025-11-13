# How to Avoid UNK Tokens

This guide explains what causes `__UNK__` tokens and how to completely avoid them in your synthetic data.

## What Are UNK Tokens?

`__UNK__` (unknown) tokens appear in categorical columns when:
1. The categorical domain is **private** (not provided as public knowledge)
2. The system uses **DP heavy hitters** to discover categories privately
3. Some categories fall outside the **top-K buckets** kept in the vocabulary
4. These categories get replaced with `__UNK__` for privacy

## Why UNK Tokens Occur

When a categorical column has an **unknown domain** (no `public_categories` provided):

1. **Hashing**: Values are hashed into buckets (B000, B001, etc.) for privacy
2. **DP Heavy Hitters**: Only the top-K most frequent buckets are kept
3. **UNK for Others**: Values in non-top-K buckets become `__UNK__`

This is a **privacy-preserving mechanism** - it prevents revealing rare categories that could leak information.

## Requirements to Avoid UNK Tokens

### ✅ Method 1: Use `public_categories` (BEST - No UNK, No DP Cost)

**When to use**: You know all possible values for a categorical column (e.g., US states, ISO country codes, days of week).

**How it works**: Provides the complete domain upfront, so no DP discovery needed.

**Example**:
```python
from privbayes_enhanced import EnhancedPrivBayesAdapter

model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'state': ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
        'education': ['HS', 'BS', 'MS', 'PhD', 'Assoc'],
        'marital_status': ['Single', 'Married', 'Divorced', 'Widowed']
    }
)
```

**Result**: ✅ **Zero UNK tokens** for these columns, **no epsilon cost** for discovery.

**Requirements**:
- You must know **all possible values** for the column
- Values must be **exact matches** (case-sensitive)
- Works for any number of categories

---

### ✅ Method 2: Use `label_columns` (BEST for Target Variables)

**When to use**: For target/outcome variables (e.g., income, diagnosis, class labels).

**How it works**: Label columns never use hashing, never get UNK tokens.

**Example**:
```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    label_columns=['income', 'diagnosis', 'target'],
    public_categories={
        'income': ['<=50K', '>50K'],  # Required when using label_columns
        'diagnosis': ['benign', 'malignant'],
        'target': ['class_A', 'class_B', 'class_C']
    }
)
```

**Result**: ✅ **Zero UNK tokens** for label columns.

**Requirements**:
- Must provide `public_categories` for each label column
- Best for **target/outcome variables**
- Protects important columns from UNK

---

### ✅ Method 3: Use `cat_keep_all_nonzero=True` (Universal Strategy)

**When to use**: When you want to keep **all observed categories** from training data.

**How it works**: Keeps all buckets with non-zero noisy counts (not just top-K).

**Example**:
```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    cat_keep_all_nonzero=True  # Default in adapter
)
```

**Result**: ✅ **Near-zero UNK tokens** (only for categories never seen in training).

**Requirements**:
- **Default in `EnhancedPrivBayesAdapter`**
- DP-safe (uses noisy counts)
- Uses more memory (keeps all buckets)
- May still have UNK if categories appear in test but not training

**Limitation**: Only keeps categories **observed in training data**. If new categories appear in test data, they'll be UNK.

---

### ⚠️ Method 4: Increase `cat_topk` (Partial Solution)

**When to use**: When you can't use public_categories but want fewer UNK tokens.

**How it works**: Keeps more top-K buckets (uses more epsilon for discovery).

**Example**:
```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    cat_topk=50,  # Default is 28, increase to keep more categories
    cat_topk_overrides={'column_name': 100}  # Per-column control
)
```

**Result**: ⚠️ **Fewer UNK tokens**, but may not eliminate them completely.

**Requirements**:
- Uses **more epsilon** for categorical discovery
- Trade-off: Less epsilon for structure/CPT learning
- May still have UNK for rare categories

**Limitation**: Cannot guarantee zero UNK tokens.

---

### ⚠️ Method 5: Increase `eps_disc` (Partial Solution)

**When to use**: When you want to allocate more privacy budget to category discovery.

**How it works**: Allocates more epsilon specifically for discovering categories.

**Example**:
```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    eps_disc=0.3,  # Default is ~0.1, increase for more discovery
    eps_split={'disc': 0.3, 'struct': 0.3, 'cpt': 0.4}
)
```

**Result**: ⚠️ **Fewer UNK tokens**, but may not eliminate them completely.

**Requirements**:
- Reduces epsilon available for structure/CPT learning
- Trade-off between discovery and model quality
- May still have UNK for rare categories

**Limitation**: Cannot guarantee zero UNK tokens.

---

## Complete Solution: Combine Methods

For **zero UNK tokens**, combine methods:

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    # Method 1: Public categories for known domains
    public_categories={
        'state': ['CA', 'NY', 'TX', ...],  # All US states
        'education': ['HS', 'BS', 'MS', 'PhD'],
    },
    # Method 2: Label columns for targets
    label_columns=['income', 'diagnosis'],
    public_categories={
        'income': ['<=50K', '>50K'],
        'diagnosis': ['benign', 'malignant']
    },
    # Method 3: Keep all observed categories
    cat_keep_all_nonzero=True  # Default
)
```

**Result**: ✅ **Zero UNK tokens** for all categorical columns.

---

## Summary Table

| Method | UNK Tokens | DP Cost | Memory | Best For |
|--------|------------|---------|--------|----------|
| `public_categories` | ✅ **0** | ✅ **None** | Low | Known domains |
| `label_columns` | ✅ **0** | ✅ **None** | Low | Target variables |
| `cat_keep_all_nonzero=True` | ✅ **~0** | Low | Higher | All categories |
| Increase `cat_topk` | ⚠️ Fewer | Higher | Higher | Partial solution |
| Increase `eps_disc` | ⚠️ Fewer | Higher | Same | Partial solution |

---

## Quick Reference

### Zero UNK Tokens (Recommended)

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'column1': ['all', 'possible', 'values'],
        'column2': ['complete', 'domain']
    },
    label_columns=['target'],
    public_categories={'target': ['all', 'target', 'values']},
    cat_keep_all_nonzero=True  # Default
)
```

### Minimal UNK Tokens (When Public Categories Unknown)

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    cat_keep_all_nonzero=True,  # Default - keeps all observed
    cat_topk=50,  # Increase from default 28
    eps_disc=0.2  # Allocate more for discovery
)
```

---

## Checking for UNK Tokens

After generating synthetic data:

```python
# Count UNK tokens
unk_count = 0
for col in synthetic_data.columns:
    if synthetic_data[col].dtype == 'object':
        unk_count += (synthetic_data[col].astype(str) == '__UNK__').sum()

print(f"Total UNK tokens: {unk_count}")
print(f"UNK rate: {unk_count / len(synthetic_data) / len(synthetic_data.select_dtypes(include=['object']).columns) * 100:.2f}%")
```

Or use the privacy metrics:

```python
privacy_metrics = model.evaluate_privacy(real_data, synthetic_data)
print(f"UNK token rate: {privacy_metrics['unk_token_rate']:.4f}")
print(f"Number of UNK tokens: {privacy_metrics['n_unk_tokens']}")
```

---

## Best Practices

1. **Always use `public_categories`** when you know the domain (US states, ISO codes, etc.)
2. **Use `label_columns`** for target/outcome variables
3. **Keep `cat_keep_all_nonzero=True`** (default) to minimize UNK
4. **Check UNK rates** after generation to verify
5. **Combine methods** for complete UNK elimination

---

## Example: Breast Cancer Dataset

```python
from privbayes_enhanced import EnhancedPrivBayesAdapter
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = [data.target_names[t] for t in data.target]

# Create model with zero UNK guarantee
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    label_columns=['target'],
    public_categories={'target': ['benign', 'malignant']},  # Required
    cat_keep_all_nonzero=True  # Default
)

model.fit(df)
synthetic = model.sample(n_samples=100000)

# Verify: Check UNK tokens
unk_count = (synthetic['target'].astype(str) == '__UNK__').sum()
print(f"UNK tokens in target: {unk_count}")  # Should be 0
```

**Result**: ✅ **0 UNK tokens** because:
- `target` is a `label_column` (no hashing)
- `public_categories` provided for target
- All numeric columns don't use UNK (they're discretized)

---

## Troubleshooting

### Still Getting UNK Tokens?

1. **Check if `public_categories` values match exactly** (case-sensitive)
2. **Verify `label_columns` have `public_categories`** (required)
3. **Ensure `cat_keep_all_nonzero=True`** (default in adapter)
4. **Check for typos** in category names
5. **Verify column names** match between data and configuration

### UNK Tokens in Test Data?

If you see UNK tokens in **test/evaluation data** but not training:
- This is **expected** - UNK appears for categories not seen during training
- Use `public_categories` to include all possible test categories
- Or ensure training data includes all categories that might appear in test

---

## Conclusion

**To completely avoid UNK tokens:**

1. ✅ Use `public_categories` for known domains
2. ✅ Use `label_columns` for target variables (with `public_categories`)
3. ✅ Keep `cat_keep_all_nonzero=True` (default)
4. ✅ Verify with metrics after generation

**Result**: Zero UNK tokens with no additional DP cost for known domains.

