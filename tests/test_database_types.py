#!/usr/bin/env python3
"""Test Enhanced PrivBayes with database-exported CSV files containing various data types."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from privbayes_enhanced import EnhancedPrivBayesAdapter
import os

def create_test_datasets():
    """Create test datasets simulating database exports."""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Dataset 1: E-commerce transactions (datetime, decimal, varchar, integer)
    print("Creating Dataset 1: E-commerce transactions...")
    ecommerce_data = {
        'order_id': [f'ORD-{i:06d}' for i in range(1, n_samples + 1)],
        'order_date': pd.date_range('2023-01-01', periods=n_samples, freq='1h'),
        'customer_id': np.random.randint(1000, 9999, n_samples),
        'product_name': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], n_samples),
        'quantity': np.random.randint(1, 10, n_samples),
        'unit_price': np.round(np.random.uniform(10.50, 999.99, n_samples), 2),  # DECIMAL(10,2)
        'total_amount': None,  # Will calculate
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'], n_samples),
        'shipping_date': None,  # Will calculate
        'status': np.random.choice(['Pending', 'Shipped', 'Delivered', 'Cancelled'], n_samples),
        'discount_percent': np.round(np.random.uniform(0, 25.5, n_samples), 2),  # DECIMAL(5,2)
    }
    
    df_ecommerce = pd.DataFrame(ecommerce_data)
    df_ecommerce['total_amount'] = (df_ecommerce['unit_price'] * df_ecommerce['quantity'] * 
                                   (1 - df_ecommerce['discount_percent'] / 100)).round(2)
    shipping_delays = pd.Series([pd.Timedelta(days=int(d)) for d in np.random.randint(1, 7, n_samples)])
    df_ecommerce['shipping_date'] = df_ecommerce['order_date'] + shipping_delays
    
    # Dataset 2: Employee records (datetime, varchar, integer, boolean)
    print("Creating Dataset 2: Employee records...")
    employee_data = {
        'employee_id': np.random.randint(10000, 99999, n_samples),
        'first_name': np.random.choice(['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana'], n_samples),
        'last_name': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], n_samples),
        'email': [f"user{i}@company.com" for i in range(n_samples)],
        'hire_date': pd.date_range('2020-01-01', periods=n_samples, freq='1D'),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'], n_samples),
        'salary': np.random.randint(40000, 150000, n_samples),
        'is_manager': np.random.choice([True, False], n_samples),
        'years_experience': np.random.randint(0, 20, n_samples),
        'last_promotion_date': None,  # Will calculate
    }
    
    df_employee = pd.DataFrame(employee_data)
    promotion_delays = pd.Series([pd.Timedelta(days=int(d)) for d in np.random.randint(0, 1000, n_samples)])
    df_employee['last_promotion_date'] = df_employee['hire_date'] + promotion_delays
    
    # Dataset 3: Sensor data (datetime, decimal, integer, varchar)
    print("Creating Dataset 3: Sensor data...")
    sensor_data = {
        'sensor_id': np.random.choice(['SENSOR-001', 'SENSOR-002', 'SENSOR-003', 'SENSOR-004'], n_samples),
        'timestamp': pd.date_range('2024-01-01 00:00:00', periods=n_samples, freq='5min'),
        'temperature': np.round(np.random.uniform(18.5, 25.8, n_samples), 2),  # DECIMAL(5,2)
        'humidity': np.round(np.random.uniform(30.0, 80.5, n_samples), 2),  # DECIMAL(5,2)
        'pressure': np.round(np.random.uniform(980.0, 1020.5, n_samples), 2),  # DECIMAL(6,2)
        'reading_count': np.random.randint(1, 1000, n_samples),
        'status_code': np.random.choice(['OK', 'WARNING', 'ERROR', 'MAINTENANCE'], n_samples),
        'location': np.random.choice(['Building-A', 'Building-B', 'Building-C'], n_samples),
    }
    
    df_sensor = pd.DataFrame(sensor_data)
    
    # Dataset 4: Financial transactions (datetime, decimal, varchar, integer)
    print("Creating Dataset 4: Financial transactions...")
    financial_data = {
        'transaction_id': [f'TXN-{i:08d}' for i in range(1, n_samples + 1)],
        'transaction_date': pd.date_range('2023-06-01', periods=n_samples, freq='30min'),
        'account_number': [f'ACC-{np.random.randint(100000, 999999)}' for _ in range(n_samples)],
        'transaction_type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer', 'Payment'], n_samples),
        'amount': np.round(np.random.uniform(10.00, 50000.00, n_samples), 2),  # DECIMAL(10,2)
        'balance_after': None,  # Will calculate
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], n_samples),
        'merchant_name': np.random.choice(['Amazon', 'Walmart', 'Target', 'Best Buy', 'Other'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),  # Boolean as integer
    }
    
    df_financial = pd.DataFrame(financial_data)
    df_financial['balance_after'] = np.cumsum(df_financial['amount']) + 10000
    
    # Save all datasets
    datasets = {
        'ecommerce': df_ecommerce,
        'employee': df_employee,
        'sensor': df_sensor,
        'financial': df_financial,
    }
    
    os.makedirs('test_data', exist_ok=True)
    for name, df in datasets.items():
        filepath = f'test_data/{name}_database_export.csv'
        df.to_csv(filepath, index=False)
        print(f"  Saved {filepath} ({len(df)} rows, {len(df.columns)} columns)")
        print(f"    Column types: {df.dtypes.to_dict()}")
    
    return datasets


def test_with_dataset(name: str, df: pd.DataFrame, epsilon: float = 1.0):
    """Test Enhanced PrivBayes with a dataset."""
    print("\n" + "=" * 80)
    print(f"Testing with {name} dataset")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print()
    
    try:
        # Create and fit model
        print("Fitting model...")
        model = EnhancedPrivBayesAdapter(epsilon=epsilon, seed=42)
        model.fit(df)
        print("✓ Model fitted successfully")
        
        # Generate synthetic data
        print("Generating synthetic data...")
        synthetic = model.sample(n_samples=min(500, len(df)))
        print(f"✓ Generated {len(synthetic)} synthetic samples")
        
        # Check data types
        print("\nOriginal data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        print("\nSynthetic data types:")
        for col in synthetic.columns:
            print(f"  {col}: {synthetic[col].dtype}")
        
        # Check for any issues
        print("\nData validation:")
        print(f"  Original columns: {len(df.columns)}")
        print(f"  Synthetic columns: {len(synthetic.columns)}")
        print(f"  Columns match: {list(df.columns) == list(synthetic.columns)}")
        
        # Check for NaN/UNK
        unk_count = 0
        for col in synthetic.columns:
            if synthetic[col].dtype == 'object':
                unk_count += (synthetic[col].astype(str) == '__UNK__').sum()
        
        print(f"  UNK tokens: {unk_count}")
        
        # Privacy report
        report = model.privacy_report()
        print(f"\nPrivacy budget used: {report['epsilon_total']:.6f} ε")
        
        print(f"\n✓ {name} dataset test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ {name} dataset test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing Enhanced PrivBayes with Database-Exported CSV Files")
    print("=" * 80)
    print()
    
    # Create test datasets
    datasets = create_test_datasets()
    
    # Test each dataset
    results = {}
    for name, df in datasets.items():
        results[name] = test_with_dataset(name, df, epsilon=1.0)
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

