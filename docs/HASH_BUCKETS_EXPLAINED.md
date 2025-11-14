# Why Hash Buckets (B000, B001, etc.) Appear in Synthetic Data

## Overview

Hash buckets like `B000`, `B001`, `B030`, `B042` appear in synthetic data when categorical columns have **too many unique values** to be treated as public categories. This is a **differential privacy mechanism** that protects individual values while still allowing the model to learn patterns.

## Why Hashing Happens

### 1. **Privacy Protection**
- For columns with many unique values (e.g., names, descriptions, IDs), revealing the exact set of categories would leak information about individuals in the training data
- Hashing values into buckets (B000, B001, etc.) preserves privacy while allowing the model to learn distributions

### 2. **Auto-Detection Threshold**
The adapter automatically treats categorical columns as "public categories" (preserving actual names) if they have **≤ 100 unique values**. Columns with **> 100 unique values** are hashed.

### 3. **Example from Your Data**

| Column | Unique Values | Status | Result |
|--------|--------------|--------|--------|
| `grade` | 5 | ≤ 100 | ✓ Actual names (A, B, C, D, F) |
| `status` | 2 | ≤ 100 | ✓ Actual names (Y, N) |
| `city` | 8 | ≤ 100 | ✓ Actual names (Sydney, Paris, etc.) |
| `country` | 8 | ≤ 100 | ✓ Actual names (Germany, Italy, etc.) |
| `category` | 4 | ≤ 100 | ✓ Actual names (Type A, Type B, etc.) |
| `department` | 5 | ≤ 100 | ✓ Actual names (Sales, Marketing, etc.) |
| `product` | 100 | = 100 | ✓ Actual names (Product_0, Product_1, etc.) |
| `name` | 875 | > 100 | ✗ Hash buckets (B030, B003, etc.) |
| `description` | 891 | > 100 | ✗ Hash buckets (B042, B026, etc.) |
| `notes` | 286 | > 100 | ✗ Hash buckets (B020, B034, etc.) |

## How Hash Buckets Work

1. **During Training (`fit()`)**:
   - Values are hashed using BLAKE2b hash function
   - Each value maps to a bucket: `B000`, `B001`, `B002`, ..., `B063` (default 64 buckets)
   - Only the **top-K most frequent buckets** are kept (default K=28)
   - The model learns the distribution of hash buckets, not individual values

2. **During Generation (`sample()`)**:
   - The model generates hash bucket codes (e.g., `B030`, `B042`)
   - These are decoded back to the bucket names
   - **Original values are lost** - you only get the hash bucket names

## Solutions to Get Actual Category Names

### Solution 1: Use `label_columns` (Best for Target Variables)

If a column is a **label/target variable** (e.g., income, diagnosis), you can mark it as a label column. Label columns **never use hashing** and always preserve actual category names.

```python
from privbayes_enhanced.adapter import EnhancedPrivBayesAdapter

model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    label_columns=['name', 'description']  # No hashing for these
)
model.fit(df)
synth = model.sample(1000)
```

**Trade-off**: Uses more epsilon budget, but preserves actual names.

### Solution 2: Manually Provide `public_categories`

If you know all possible values for a column (e.g., from a public domain), you can provide them explicitly:

```python
# Get all unique values from your data (or from a public source)
all_names = df['name'].dropna().unique().tolist()
all_descriptions = df['description'].dropna().unique().tolist()

model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'name': all_names,
        'description': all_descriptions
    }
)
model.fit(df)
synth = model.sample(1000)
```

**Trade-off**: 
- ✅ Preserves actual names
- ⚠️ If the domain is truly private (e.g., actual person names), this may reduce privacy
- ✅ If the domain is public knowledge (e.g., US states, ISO codes), this is safe

### Solution 3: Increase Auto-Detection Threshold

You can modify the threshold in `privbayes_enhanced/adapter.py` (currently 100):

```python
# In adapter.py, line ~132
if len(unique_vals) <= 200:  # Changed from 100 to 200
    self.model.public_categories[col] = sorted([str(v) for v in unique_vals if pd.notna(v)])
```

**Trade-off**: 
- ✅ More columns will preserve actual names
- ⚠️ Higher threshold may reduce privacy for large domains
- ⚠️ Uses more memory for large category lists

### Solution 4: Use `cat_keep_all_nonzero=True` (Already Enabled)

This is already enabled by default in the adapter. It keeps **all observed hash buckets** instead of just top-K, which minimizes `__UNK__` tokens but still produces hash buckets (not actual names).

## When to Use Each Solution

| Scenario | Recommended Solution |
|----------|---------------------|
| Target/label variable (e.g., income, diagnosis) | `label_columns` |
| Public domain (e.g., US states, ISO country codes) | `public_categories` |
| Private domain, but need actual names | `label_columns` (if acceptable) or increase threshold |
| Acceptable to have hash buckets | No change needed (current behavior) |

## Current Behavior Summary

- **Small domains (≤ 100 unique values)**: ✅ Actual category names preserved
- **Large domains (> 100 unique values)**: ✗ Hash buckets (B000, B001, etc.)
- **All `__UNK__` tokens**: ✅ Replaced with NULL (NaN) in post-processing

## Example: Fixing Hash Buckets for Specific Columns

```python
from privbayes_enhanced.adapter import EnhancedPrivBayesAdapter
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')

# Option A: Use label_columns (if these are target variables)
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    label_columns=['name', 'description', 'notes']
)
model.fit(df)
synth = model.sample(1000)

# Option B: Provide public_categories (if domain is public knowledge)
all_names = df['name'].dropna().unique().tolist()
all_descriptions = df['description'].dropna().unique().tolist()
all_notes = df['notes'].dropna().unique().tolist()

model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'name': all_names,
        'description': all_descriptions,
        'notes': all_notes
    }
)
model.fit(df)
synth = model.sample(1000)

# Verify results
print("Sample values:")
print(f"  name: {synth['name'].head(5).tolist()}")
print(f"  description: {synth['description'].head(5).tolist()}")
# Should show actual names, not B000, B001, etc.
```

## Privacy Note

**Important**: If you use `label_columns` or `public_categories` for columns with truly private values (e.g., actual person names), you are revealing the exact set of values that appeared in your training data. This may reduce privacy guarantees. Use these options only when:
- The column is a target variable you need to preserve exactly
- The domain is public knowledge (e.g., US states, ISO codes)
- You accept the privacy trade-off for utility

