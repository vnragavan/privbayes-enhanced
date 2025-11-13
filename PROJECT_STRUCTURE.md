# Project Structure

This document describes the organization of the Enhanced PrivBayes project.

## Directory Structure

```
privbayes-enhanced-standalone/
├── privbayes_enhanced/          # Main package
│   ├── __init__.py              # Package initialization
│   ├── adapter.py               # High-level adapter interface
│   ├── synthesizer.py           # Core DP synthesizer
│   ├── metrics.py               # Utility and privacy metrics
│   ├── privacy_audit.py          # Privacy audit and QI linkage
│   ├── downstream_metrics.py    # Downstream task metrics
│   └── cli.py                   # Command-line interface
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── test_adapter.py          # Adapter tests
│   ├── test_synthesizer.py      # Synthesizer tests
│   ├── test_integration.py     # Integration tests
│   ├── test_real_data.py       # Real data tests
│   ├── test_audit_guarantees.py # Audit tests
│   ├── test_csv_loading.py      # CSV loading tests
│   ├── test_database_types.py   # Database type tests
│   ├── test_metrics.py         # Metrics tests
│   └── test_with_data.py       # Data loading tests
│
├── examples/                     # Example scripts
│   ├── README.md                # Examples documentation
│   ├── example.py               # Basic usage example
│   ├── generate_all_metrics.py # Complete metrics example
│   └── demo_k_anonymity.py     # K-anonymity demonstration
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation index
│   ├── QUICK_START.md           # Quick start guide
│   ├── CLI_USAGE.md             # CLI documentation
│   ├── METRICS.md               # Metrics documentation
│   ├── DOWNSTREAM_METRICS.md    # Downstream metrics guide
│   ├── PRIVACY_AUDIT.md         # Privacy audit guide
│   ├── DATABASE_COMPATIBILITY.md # Database compatibility
│   ├── DP_CHECKLIST.md           # DP compliance checklist
│   ├── AUDIT_GUARANTEES.md      # Audit guarantees
│   ├── K_ANONYMITY_EXPLAINED.md  # K-anonymity guide
│   └── TESTING.md                # Testing guide
│
├── data/                        # Sample data (gitignored)
│   ├── .gitkeep
│   └── adult.csv                # Sample dataset
│
├── test_data/                   # Test datasets (gitignored)
│   ├── .gitkeep
│   └── *.csv                    # Test CSV files
│
├── venv/                        # Virtual environment (gitignored)
│
├── .gitignore                   # Git ignore rules
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # Main project README
└── PROJECT_STRUCTURE.md         # This file
```

## Key Directories

### `privbayes_enhanced/`
The main package containing all core functionality:
- **adapter.py**: High-level interface for users
- **synthesizer.py**: Core differential privacy implementation
- **metrics.py**: Utility and privacy evaluation metrics
- **privacy_audit.py**: Privacy risk assessment tools
- **downstream_metrics.py**: ML task performance metrics
- **cli.py**: Command-line interface

### `tests/`
Test suite includes:
- Unit tests for each module
- Integration tests
- Real data tests
- Database compatibility tests

### `examples/`
Example scripts demonstrating usage:
- Basic examples
- Advanced usage patterns
- Demonstrations

### `docs/`
Complete documentation:
- User guides
- API documentation
- Privacy and security guides
- Development guides

### `data/` and `test_data/`
Data directories (gitignored):
- Sample datasets
- Test data files
- Generated synthetic data

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Test files**: `test_*.py`
- **Documentation**: `UPPER_CASE.md` or `Title_Case.md`
- **Config files**: `lowercase.ini` or `lowercase.txt`

## Best Practices

1. **Code organization**: All source code in `privbayes_enhanced/`
2. **Tests**: All tests in `tests/` directory
3. **Examples**: All examples in `examples/` directory
4. **Documentation**: All docs in `docs/` directory
5. **Data**: Sample/test data in `data/` and `test_data/`
6. **Generated files**: Excluded via `.gitignore`

## Adding New Files

When adding new files:

1. **New modules**: Add to `privbayes_enhanced/`
2. **New tests**: Add to `tests/` with `test_` prefix
3. **New examples**: Add to `examples/` with descriptive name
4. **New docs**: Add to `docs/` with descriptive name
5. **Update**: Update relevant README files

## Ignored Files

The following are excluded from version control (see `.gitignore`):

- `__pycache__/` - Python cache
- `*.pyc` - Compiled Python files
- `venv/` - Virtual environment
- `*.egg-info/` - Package metadata
- `.pytest_cache/` - Pytest cache
- `*.pkl` - Pickle files
- `*.json` - Generated metrics files
- `data/*.csv` - Data files
- `test_data/*.csv` - Test data files

