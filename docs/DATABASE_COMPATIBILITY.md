# Database Compatibility Guide

## Overview

Enhanced PrivBayes is designed to work with CSV files exported from databases, automatically handling various data types commonly found in database exports.

## Supported Data Types

### ✅ Fully Supported

1. **DATETIME / TIMESTAMP**
   - Formats: `2023-01-15 10:30:00`, `2023-01-15T10:30:00`, ISO formats
   - Handled: Automatically detected and converted to numeric (nanoseconds since epoch)
   - Example: `order_date`, `hire_date`, `transaction_date`

2. **DECIMAL / NUMERIC / FLOAT**
   - Formats: `123.45`, `999.99`, `0.00`
   - Handled: Preserved as float64, discretized into bins
   - Example: `unit_price`, `total_amount`, `temperature`

3. **INTEGER / BIGINT**
   - Formats: `12345`, `999999`
   - Handled: Preserved as int64
   - Example: `customer_id`, `employee_id`, `quantity`

4. **VARCHAR / TEXT / STRING**
   - Formats: Any string values
   - Handled: Treated as categorical, uses DP heavy hitters
   - Example: `product_name`, `status`, `department`

5. **BOOLEAN**
   - Formats: `True/False`, `1/0`, `Y/N`
   - Handled: Converted to numeric (0/1) or kept as boolean
   - Example: `is_manager`, `is_fraud`

### 🔄 Automatic Type Detection

The code automatically detects and converts:

1. **String-formatted dates**: If 95%+ of values in an object column can be parsed as datetime, it's converted to numeric
2. **Numeric strings**: If 95%+ of values in an object column can be converted to numeric, it's converted
3. **Datetime columns**: Automatically converted to int64 (nanoseconds)

## Test Results

We've tested with various database-exported CSV files:

### ✅ E-commerce Transactions
- **Types**: DATETIME, DECIMAL, VARCHAR, INTEGER
- **Columns**: order_date, shipping_date, unit_price, total_amount, product_name, status
- **Result**: ✓ PASSED

### ✅ Employee Records
- **Types**: DATETIME, VARCHAR, INTEGER, BOOLEAN
- **Columns**: hire_date, last_promotion_date, department, salary, is_manager
- **Result**: ✓ PASSED

### ✅ Sensor Data
- **Types**: DATETIME, DECIMAL, INTEGER, VARCHAR
- **Columns**: timestamp, temperature, humidity, pressure, status_code
- **Result**: ✓ PASSED

### ✅ Financial Transactions
- **Types**: DATETIME, DECIMAL, VARCHAR, INTEGER
- **Columns**: transaction_date, amount, balance_after, currency, transaction_type
- **Result**: ✓ PASSED

## Usage Examples

### Basic Usage

```python
from privbayes_enhanced import EnhancedPrivBayesAdapter
import pandas as pd

# Load CSV exported from database
data = pd.read_csv('database_export.csv')

# Create and fit model (automatic type detection)
model = EnhancedPrivBayesAdapter(epsilon=1.0, seed=42)
model.fit(data)

# Generate synthetic data
synthetic = model.sample(n_samples=1000)

# Save synthetic data
synthetic.to_csv('synthetic_data.csv', index=False)
```

### Command Line

```bash
# Direct CSV processing
privbayes database_export.csv -o synthetic_data.csv --epsilon 1.0

# With all metrics
privbayes database_export.csv -o synthetic_data.csv --epsilon 1.0 --all-metrics
```

## Type Conversion Details

### Datetime Handling

**Original (CSV)**: `2023-01-15 10:30:00` (string)
**After Loading**: `datetime64[ns]` (pandas)
**After Processing**: `int64` (nanoseconds since epoch)
**Synthetic Output**: `int64` (can be converted back to datetime if needed)

### Decimal Handling

**Original (CSV)**: `123.45` (DECIMAL(10,2))
**After Loading**: `float64` (pandas)
**After Processing**: Discretized into bins
**Synthetic Output**: `float64` (within learned distribution)

### Varchar Handling

**Original (CSV)**: `'Product Name'` (VARCHAR(255))
**After Loading**: `object` (pandas)
**After Processing**: Categorical with DP heavy hitters
**Synthetic Output**: `object` (preserves categories, may include `__UNK__` for rare values)

## Best Practices

### 1. Datetime Columns

If you have datetime columns, the code will automatically detect and convert them. However, for better control:

```python
# Option 1: Let code auto-detect (recommended)
model.fit(data)

# Option 2: Pre-process if needed
data['order_date'] = pd.to_datetime(data['order_date'])
model.fit(data)
```

### 2. Decimal Precision

For high-precision decimals, consider rounding or using public bounds:

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_bounds={
        'price': [0.0, 1000.0],  # Known bounds
    }
)
```

### 3. Categorical Columns

For categorical columns with known domains, use public categories:

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={
        'status': ['Pending', 'Shipped', 'Delivered', 'Cancelled'],
        'currency': ['USD', 'EUR', 'GBP', 'JPY'],
    }
)
```

### 4. Boolean Columns

Boolean columns are automatically handled. They're converted to numeric (0/1) internally but can be kept as boolean:

```python
# Boolean columns are automatically detected and handled
# No special configuration needed
```

## Limitations

1. **Very Large Integers**: May lose precision if outside int64 range
2. **High Precision Decimals**: Discretized into bins (configurable)
3. **Very Long Strings**: Treated as categorical (may get `__UNK__` if rare)
4. **Mixed Types in Column**: If column has mixed types, may not auto-detect correctly

## Troubleshooting

### Issue: Datetime columns not detected

**Solution**: Ensure datetime strings are in a standard format. The code detects if 95%+ of values are parseable.

```python
# Check if datetime parsing works
pd.to_datetime(data['date_column'], errors='coerce').notna().mean()
# Should be >= 0.95
```

### Issue: Decimal precision lost

**Solution**: Use more bins or public bounds:

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    bins_per_numeric=32,  # More bins = more precision
)
```

### Issue: Too many `__UNK__` tokens

**Solution**: Use public categories or increase epsilon:

```python
model = EnhancedPrivBayesAdapter(
    epsilon=1.0,
    public_categories={'column_name': ['val1', 'val2', ...]},
    cat_keep_all_nonzero=True,  # Keep all categories
)
```

## Summary

✅ **Works with**: DATETIME, DECIMAL, VARCHAR, INTEGER, BOOLEAN
✅ **Auto-detects**: String-formatted dates, numeric strings
✅ **Preserves**: Column structure, data relationships
✅ **Tested with**: E-commerce, employee, sensor, financial data

The code is **generic enough** to work with most database-exported CSV files without modification!

