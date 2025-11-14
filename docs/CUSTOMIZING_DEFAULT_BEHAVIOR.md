# Customizing Default Behavior

## Current Default Behavior

By default, the adapter **automatically treats all categorical columns as label columns**. This means:
- ✅ All categorical columns preserve actual category names (no hash buckets like B000, B001)
- ✅ No hashing is applied to categorical columns
- ⚠️ Uses more epsilon budget (label columns don't use hashing)

## How to Change the Default Behavior

### Option 1: Disable Auto-Detection (Use Manual Control)

If you want to manually control which columns are treated as label columns, you can:

1. **Modify the adapter code** to disable auto-detection, or
2. **Override after initialization** by explicitly setting `label_columns`

**Method A: Modify adapter.py**

In `privbayes_enhanced/adapter.py`, comment out or remove the auto-detection code (around lines 100-145):

```python
# Comment out or remove this section:
# # Auto-detect all categorical columns and treat them as label columns by default
# if not hasattr(self.model, 'label_columns') or self.model.label_columns is None:
#     self.model.label_columns = []
# ... (rest of auto-detection code)
```

**Method B: Override after initialization**

```python
from privbayes_enhanced.adapter import EnhancedPrivBayesAdapter
import pandas as pd

df = pd.read_csv('your_data.csv')

# Initialize model
model = EnhancedPrivBayesAdapter(epsilon=1.0)

# Manually set label_columns (overrides auto-detection)
model.model.label_columns = ['grade', 'status']  # Only these will be label columns

# Or set to empty list to disable label columns entirely
# model.model.label_columns = []

model.fit(df)
synth = model.sample(1000)
```

### Option 2: Add a Configuration Parameter

You can add a parameter to control this behavior. Here's how:

**Step 1: Modify `__init__` method in `adapter.py`:**

```python
def __init__(
    self,
    epsilon: float = 1.0,
    delta: Optional[float] = 1e-6,
    seed: int = 42,
    temperature: float = 1.0,
    cpt_smoothing: float = 1.5,
    label_columns: Optional[list] = None,
    public_categories: Optional[dict] = None,
    cat_keep_all_nonzero: bool = True,
    auto_detect_label_columns: bool = True,  # NEW PARAMETER
    **kwargs
):
    """Initialize the adapter.
    
    Args:
        ...
        auto_detect_label_columns: If True, automatically treat all categorical columns
            as label columns (default: True). If False, only use manually specified
            label_columns.
    """
    ...
    self.auto_detect_label_columns = auto_detect_label_columns
    ...
```

**Step 2: Modify `fit()` method to check this flag:**

```python
def fit(self, X: pd.DataFrame, y=None, column_constraints: Optional[dict] = None):
    ...
    
    # Only auto-detect if enabled
    if self.auto_detect_label_columns:
        # Auto-detect all categorical columns and treat them as label columns by default
        if not hasattr(self.model, 'label_columns') or self.model.label_columns is None:
            self.model.label_columns = []
        # ... (rest of auto-detection code)
    else:
        # Use only manually specified label_columns
        if not hasattr(self.model, 'label_columns') or self.model.label_columns is None:
            self.model.label_columns = []
    ...
```

**Step 3: Use the new parameter:**

```python
# Disable auto-detection
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    auto_detect_label_columns=False,  # Disable auto-detection
    label_columns=['grade', 'status']  # Only these will be label columns
)
model.fit(df)

# Or enable auto-detection (default)
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    auto_detect_label_columns=True  # Default behavior
)
model.fit(df)
```

### Option 3: Change the Threshold (Hybrid Approach)

Instead of treating ALL categoricals as label columns, you can modify the threshold to only treat small-domain categoricals as label columns:

**In `adapter.py`, modify the auto-detection logic:**

```python
# Detect categorical columns with small domain (e.g., <= 50 unique values)
auto_label_columns = []
existing_labels = set(self.model.label_columns)
for col in X.columns:
    if col not in datetime_candidate_cols:
        if X[col].dtype == 'object':
            unique_vals = X[col].dropna().unique()
            # Only treat small-domain categoricals as label columns
            if len(unique_vals) <= 50:  # Threshold: 50 instead of all
                if col not in existing_labels:
                    auto_label_columns.append(col)
```

### Option 4: Use Public Categories Instead

Instead of using label columns, you can use `public_categories` for specific columns:

```python
from privbayes_enhanced.adapter import EnhancedPrivBayesAdapter
import pandas as pd

df = pd.read_csv('your_data.csv')

# Get unique values for columns you want to preserve
all_grades = df['grade'].dropna().unique().tolist()
all_statuses = df['status'].dropna().unique().tolist()

# Use public_categories instead of label_columns
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'grade': all_grades,
        'status': all_statuses
    }
    # Don't set label_columns - let large domains use hashing
)
model.fit(df)
synth = model.sample(1000)
```

## Comparison of Approaches

| Approach | Pros | Cons |
|---------|------|------|
| **Current (All as labels)** | Simple, preserves all names | Uses more epsilon, may reveal domain |
| **Manual control** | Full control, efficient | Requires manual specification |
| **Threshold-based** | Balanced approach | Still hashes large domains |
| **Public categories** | Flexible per-column | Requires knowing all values |

## Recommended Approach

For most use cases, we recommend **Option 2** (adding a configuration parameter) because:
- ✅ Maintains backward compatibility (default: True)
- ✅ Gives users control when needed
- ✅ Clean API design
- ✅ Easy to use

## Example: Complete Implementation

Here's a complete example of adding the configuration parameter:

```python
# In adapter.py __init__
def __init__(
    self,
    epsilon: float = 1.0,
    delta: Optional[float] = 1e-6,
    seed: int = 42,
    temperature: float = 1.0,
    cpt_smoothing: float = 1.5,
    label_columns: Optional[list] = None,
    public_categories: Optional[dict] = None,
    cat_keep_all_nonzero: bool = True,
    auto_detect_label_columns: bool = True,  # NEW
    **kwargs
):
    ...
    self.auto_detect_label_columns = auto_detect_label_columns
    ...

# In adapter.py fit()
def fit(self, X: pd.DataFrame, y=None, column_constraints: Optional[dict] = None):
    ...
    
    # Auto-detect all categorical columns as label columns (if enabled)
    if self.auto_detect_label_columns:
        # ... existing auto-detection code ...
    else:
        # Use only manually specified label_columns
        if not hasattr(self.model, 'label_columns') or self.model.label_columns is None:
            self.model.label_columns = []
        if label_columns:  # From __init__
            self.model.label_columns = list(set(self.model.label_columns + label_columns))
    ...
```

## Usage Examples

```python
# Example 1: Use default (all categoricals as labels)
model = EnhancedPrivBayesAdapter(epsilon=1.0)
model.fit(df)

# Example 2: Disable auto-detection, use manual control
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    auto_detect_label_columns=False,
    label_columns=['grade', 'status']  # Only these
)
model.fit(df)

# Example 3: Disable auto-detection, no label columns
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    auto_detect_label_columns=False
    # No label_columns - will use hashing for all categoricals
)
model.fit(df)
```

