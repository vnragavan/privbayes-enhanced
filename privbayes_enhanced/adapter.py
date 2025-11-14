"""High-level adapter for Enhanced PrivBayes."""

from typing import Optional, List
import warnings
import pandas as pd
import numpy as np
from .synthesizer import PrivBayesSynthesizerEnhanced


class EnhancedPrivBayesAdapter:
    """Wrapper around PrivBayesSynthesizerEnhanced with a simpler interface."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: Optional[float] = 1e-6,
        seed: int = 42,
        temperature: float = 1.0,
        cpt_smoothing: float = 1.5,  # DP-safe CPT smoothing (post-processing)
        label_columns: Optional[list] = None,
        public_categories: Optional[dict] = None,
        cat_keep_all_nonzero: bool = True,  # Universal strategy: keep all categories
        **kwargs
    ):
        """Initialize the adapter.
        
        Args:
            epsilon: Privacy budget (default 1.0)
            delta: Privacy parameter, default 1e-6
            seed: Random seed for reproducibility
            temperature: Sampling temperature, >1 reduces linkage risk (default 1.0)
            label_columns: Columns to treat as labels (no hashing, no UNK tokens)
            public_categories: Dict of {column: [list of categories]} for known domains
        """
        self.epsilon = epsilon
        self.delta = delta or 1e-6
        self.seed = seed
        self.temperature = temperature
        self.cpt_smoothing = cpt_smoothing
        self.kwargs = kwargs
        
        # Merge public_categories into kwargs if provided
        model_kwargs = dict(kwargs)
        if public_categories:
            if 'public_categories' in model_kwargs:
                model_kwargs['public_categories'].update(public_categories)
            else:
                model_kwargs['public_categories'] = public_categories
        
        # Add label_columns if provided
        if label_columns:
            model_kwargs['label_columns'] = label_columns
        
        # Initialize Enhanced PrivBayes with universal strategy (keeps all categories)
        model_kwargs['cat_keep_all_nonzero'] = cat_keep_all_nonzero
        self.model = PrivBayesSynthesizerEnhanced(
            epsilon=epsilon,
            delta=self.delta,
            seed=seed,
            temperature=temperature,
            cpt_smoothing=cpt_smoothing,
            **model_kwargs
        )
        
        self._fitted = False
        self._real_data = None
        self._original_columns = None  # Track original column order and names
        self._datetime_columns = {}  # Track columns that were originally datetime
        self._datetime_formats = {}  # Track datetime format strings for preservation
        self._column_constraints = {}  # Track schema constraints: max_length, min_value, max_value
        
    def fit(self, X: pd.DataFrame, y=None, column_constraints: Optional[dict] = None):
        """Fit the model to data.
        
        Converts datetimes to numeric, coerces numeric strings, then fits the model.
        
        Args:
            X: Training data
            y: Not used (for sklearn compatibility)
            column_constraints: Optional dict of {column: {'max_length': int, 'min_value': int, 'max_value': int}}
                Example: {'Personnummer_slett': {'max_length': 5}, 'PersonId': {'min_value': 0}}
        """
        X2 = X.copy()
        # Store original column order and names
        self._original_columns = list(X.columns)
        self._datetime_columns = {}  # Reset datetime tracking
        self._datetime_formats = {}  # Reset datetime format tracking
        self._column_constraints = column_constraints or {}  # Store schema constraints
        
        # Auto-detect constraints from data if not provided
        if not self._column_constraints:
            self._column_constraints = self._detect_constraints(X)
        
        # Set public_bounds for non-negative columns to prevent DP noise from making bounds negative
        # This is DP-safe because it's public knowledge (e.g., IDs are always >= 0)
        # Merge with existing public_bounds if any
        if not hasattr(self.model, 'public_bounds') or self.model.public_bounds is None:
            self.model.public_bounds = {}
        
        for col, constraints in self._column_constraints.items():
            if 'min_value' in constraints and constraints['min_value'] >= 0:
                # If column should be non-negative, set public lower bound to 0
                # This prevents DP noise from making the lower bound negative
                if col not in self.model.public_bounds:
                    # Get max from training data or use a reasonable default
                    if col in X.columns:
                        max_val = constraints.get('max_value', float(X[col].max()))
                    else:
                        max_val = constraints.get('max_value', 1e6)
                    self.model.public_bounds[col] = [0.0, float(max_val)]
        
        for c in X2.columns:
            # Convert datetime/timedelta to nanoseconds for processing
            if pd.api.types.is_datetime64_any_dtype(X2[c]):
                # Store original datetime type
                self._datetime_columns[c] = 'datetime64[ns]'
                # Detect format from original data (if it was a string column)
                if c in X.columns and X[c].dtype == 'object':
                    # Try to detect format from first non-null value
                    sample_val = X[c].dropna().iloc[0] if len(X[c].dropna()) > 0 else None
                    if sample_val is not None:
                        self._datetime_formats[c] = self._detect_datetime_format(str(sample_val))
                X2[c] = X2[c].astype('int64')
            elif pd.api.types.is_timedelta64_dtype(X2[c]):
                # Store original timedelta type
                self._datetime_columns[c] = 'timedelta64[ns]'
                X2[c] = X2[c].astype('int64')
            # Try parsing string columns as datetime (handles CSV exports)
            elif X2[c].dtype == 'object':
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, 
                                                message='.*Could not infer format.*')
                        dt_parsed = pd.to_datetime(X2[c], errors='coerce')
                    if dt_parsed.notna().mean() >= 0.95:
                        # Store that this was a datetime string
                        self._datetime_columns[c] = 'datetime64[ns]'
                        # Detect and store the format from original string data
                        sample_val = X[c].dropna().iloc[0] if len(X[c].dropna()) > 0 else None
                        if sample_val is not None:
                            self._datetime_formats[c] = self._detect_datetime_format(str(sample_val))
                        X2[c] = dt_parsed.astype('int64')
                        continue
                except (ValueError, TypeError, OverflowError):
                    pass
                # Fall back to numeric coercion
                s = pd.to_numeric(X2[c], errors='coerce')
                if s.notna().mean() >= 0.95:
                    X2[c] = s
        
        self._real_data = X2
        self.model.fit(X2)
        
        self._fitted = True
        return self
    
    def sample(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data. Defaults to same size as training data."""
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")
        
        if n_samples is None:
            n_samples = len(self._real_data)
        
        # Generate synthetic data
        synthetic_df = self.model.sample(n_samples, seed=self.seed)
        
        # Apply schema constraints BEFORE datetime conversion (so we can clip negative timestamps)
        synthetic_df = self._apply_constraints(synthetic_df)
        
        # Convert datetime columns back to readable format (after constraints applied)
        for col, dtype in self._datetime_columns.items():
            if col in synthetic_df.columns:
                if dtype == 'datetime64[ns]':
                    # Convert nanoseconds (int64) back to datetime
                    dt_series = pd.to_datetime(synthetic_df[col], errors='coerce')
                    
                    # Apply original format if detected
                    if col in self._datetime_formats:
                        fmt = self._datetime_formats[col]
                        if fmt:
                            # Format as string with original format, then convert back to datetime
                            # This preserves the format when saved to CSV
                            synthetic_df[col] = dt_series.dt.strftime(fmt)
                        else:
                            synthetic_df[col] = dt_series
                    else:
                        synthetic_df[col] = dt_series
                elif dtype == 'timedelta64[ns]':
                    # Convert nanoseconds (int64) back to timedelta
                    synthetic_df[col] = pd.to_timedelta(synthetic_df[col], errors='coerce')
        
        # Ensure column order and names match original data exactly
        if self._original_columns is not None:
            # Reorder columns to match original order
            # Add any missing columns (shouldn't happen, but be safe)
            missing_cols = [col for col in self._original_columns if col not in synthetic_df.columns]
            if missing_cols:
                # Fill missing columns with NaN (shouldn't happen in normal operation)
                for col in missing_cols:
                    synthetic_df[col] = np.nan
            
            # Remove any extra columns (shouldn't happen, but be safe)
            extra_cols = [col for col in synthetic_df.columns if col not in self._original_columns]
            if extra_cols:
                synthetic_df = synthetic_df.drop(columns=extra_cols)
            
            # Reorder to match original column order
            synthetic_df = synthetic_df[self._original_columns]
        
        return synthetic_df
    
    def privacy_report(self) -> dict:
        """Return privacy parameters."""
        report = self.model.privacy_report()
        return {
            "epsilon_total": report.get("epsilon_total_actual", self.epsilon),
            "delta": report.get("delta_used", self.delta),
            "eps_struct": report.get("eps_struct", 0.0),
            "eps_cpt": report.get("eps_cpt", 0.0),
            "eps_disc": report.get("eps_disc_used", 0.0),
            "implementation": "Enhanced PrivBayes",
            "temperature": float(self.temperature),
            "note": "Enhanced PrivBayes with temperature-based sampling and QI-linkage reduction"
        }
    
    def evaluate_utility(
        self,
        real_data: pd.DataFrame,
        synthetic_data: Optional[pd.DataFrame] = None,
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> dict:
        """Compare real and synthetic data to measure utility.
        
        If synthetic_data isn't provided, generates it first.
        """
        from .metrics import compute_utility_metrics, print_utility_report
        
        if synthetic_data is None:
            if n_samples is None:
                n_samples = len(real_data)
            synthetic_data = self.sample(n_samples=n_samples)
        
        metrics = compute_utility_metrics(real_data, synthetic_data)
        
        if verbose:
            print_utility_report(metrics, verbose=verbose)
        
        return metrics
    
    def evaluate_privacy(
        self,
        real_data: pd.DataFrame,
        synthetic_data: Optional[pd.DataFrame] = None,
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate privacy-related metrics.
        
        These are heuristic metrics. True privacy guarantees come from
        the differential privacy mechanism.
        
        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset (generated if None)
            n_samples: Number of samples to generate if synthetic_data is None
            verbose: Print detailed report
        
        Returns:
            Dictionary of privacy metrics
        """
        from .metrics import compute_privacy_metrics, print_privacy_report
        
        if synthetic_data is None:
            if n_samples is None:
                n_samples = len(real_data)
            synthetic_data = self.sample(n_samples=n_samples)
        
        privacy_report = self.privacy_report()
        metrics = compute_privacy_metrics(real_data, synthetic_data, privacy_report)
        
        if verbose:
            print_privacy_report(metrics)
        
        return metrics
    
    def evaluate_qi_linkage(
        self,
        real_data: pd.DataFrame,
        synthetic_data: Optional[pd.DataFrame] = None,
        qi_columns: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        k: int = 1,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate QI (Quasi-Identifier) linkage attack risk.
        
        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset (generated if None)
            qi_columns: List of quasi-identifier columns (auto-detected if None)
            n_samples: Number of samples to generate if synthetic_data is None
            k: k-anonymity parameter (default: 1)
            verbose: Print detailed report
        
        Returns:
            Dictionary of QI linkage risk metrics
        """
        from .privacy_audit import compute_qi_linkage_risk, print_qi_linkage_report
        
        if synthetic_data is None:
            if n_samples is None:
                n_samples = len(real_data)
            synthetic_data = self.sample(n_samples=n_samples)
        
        metrics = compute_qi_linkage_risk(real_data, synthetic_data, qi_columns, k)
        
        if verbose:
            print_qi_linkage_report(metrics)
        
        return metrics
    
    def evaluate_inference_attack(
        self,
        real_data: pd.DataFrame,
        synthetic_data: Optional[pd.DataFrame] = None,
        sensitive_columns: Optional[List[str]] = None,
        qi_columns: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate inference attack risk.
        
        Args:
            real_data: Original real dataset
            synthetic_data: Synthetic dataset (generated if None)
            sensitive_columns: List of sensitive columns (auto-detected if None)
            qi_columns: List of quasi-identifier columns (auto-detected if None)
            n_samples: Number of samples to generate if synthetic_data is None
            verbose: Print detailed report
        
        Returns:
            Dictionary of inference attack risk metrics
        """
        from .privacy_audit import compute_inference_attack_risk, print_inference_attack_report
        
        if synthetic_data is None:
            if n_samples is None:
                n_samples = len(real_data)
            synthetic_data = self.sample(n_samples=n_samples)
        
        metrics = compute_inference_attack_risk(real_data, synthetic_data, sensitive_columns, qi_columns)
        
        if verbose:
            print_inference_attack_report(metrics)
        
        return metrics
    
    def evaluate_downstream(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate downstream task performance metrics.
        
        Trains machine learning models on both real and synthetic data,
        then compares their performance on a held-out test set from real data.
        
        Args:
            real_data: Original real dataset
            synthetic_data: Generated synthetic dataset
            target_column: Name of target column (auto-detected if None)
            task_type: 'classification' or 'regression' (auto-detected if None)
            verbose: Print detailed report
        
        Returns:
            Dictionary of downstream metrics including:
            - Model performance gaps (real vs synthetic training)
            - Feature importance preservation
            - Summary statistics
        """
        from .downstream_metrics import compute_downstream_metrics, print_downstream_report
        
        metrics = compute_downstream_metrics(
            real_data=real_data,
            synthetic_data=synthetic_data,
            target_column=target_column,
            task_type=task_type
        )
        
        if verbose:
            print_downstream_report(metrics)
        
        return metrics
    
    def audit_dp_compliance(
        self,
        strict: bool = True,
        verbose: bool = True
    ) -> dict:
        """
        Audit differential privacy compliance of the model against reference design checklist.
        
        Args:
            strict: If True, fail on any violations
            verbose: Print detailed report
        
        Returns:
            Dictionary of audit results with checklist compliance
        """
        from .privacy_audit import audit_dp_compliance, print_dp_audit_report
        
        privacy_report = self.model.privacy_report()  # Get full report from model
        model_config = {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'temperature': self.temperature,
            'strict_dp': getattr(self.model, 'strict_dp', True),
            'original_data_bounds': getattr(self.model, 'original_data_bounds', None),
            'bins_per_numeric': getattr(self.model, 'bins_per_numeric', 16),
            'cat_keep_all_nonzero': getattr(self.model, 'cat_keep_all_nonzero', True),
            'cpt_smoothing': getattr(self.model, 'cpt_smoothing', 1.5),
        }
        
        audit_results = audit_dp_compliance(model_config, privacy_report, self.model, strict)
        
        if verbose:
            print_dp_audit_report(audit_results)
        
        return audit_results
    
    def _detect_constraints(self, X: pd.DataFrame) -> dict:
        """Auto-detect constraints from training data.
        
        Detects:
        - String max length for categorical columns
        - Integer min/max bounds (especially non-negative for IDs)
        - Integer digit count (max number of digits)
        - Float decimal places (precision)
        """
        constraints = {}
        
        for col in X.columns:
            col_constraints = {}
            
            # For string/categorical columns, detect length patterns
            if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                # Convert to string and analyze length patterns
                str_lengths = X[col].astype(str).str.len()
                str_lengths = str_lengths[str_lengths > 0]  # Exclude empty strings
                
                if len(str_lengths) > 0:
                    min_len = int(str_lengths.min())
                    max_len = int(str_lengths.max())
                    
                    col_constraints['min_length'] = min_len
                    col_constraints['max_length'] = max_len
                    
                    # If all values have the same length, it's a fixed-length column (like char(n))
                    if min_len == max_len:
                        col_constraints['fixed_length'] = min_len
                    
                    # Detect if values are mostly the same length (e.g., 95%+ have same length)
                    length_counts = str_lengths.value_counts()
                    most_common_length = length_counts.idxmax()
                    most_common_count = length_counts.max()
                    if most_common_count / len(str_lengths) >= 0.95:
                        # Most values have the same length - treat as preferred length
                        col_constraints['preferred_length'] = int(most_common_length)
            
            # For integer columns, detect bounds, digit count, and check if non-negative
            elif pd.api.types.is_integer_dtype(X[col]):
                numeric_vals = pd.to_numeric(X[col], errors='coerce')
                numeric_vals = numeric_vals[numeric_vals.notna()]
                
                if len(numeric_vals) > 0:
                    min_val = int(numeric_vals.min())
                    max_val = int(numeric_vals.max())
                    
                    # If all values are non-negative, set min_value to 0
                    if min_val >= 0:
                        col_constraints['min_value'] = 0
                    else:
                        col_constraints['min_value'] = min_val
                    
                    col_constraints['max_value'] = max_val
                    
                    # Detect max number of digits (for integer precision)
                    # Count digits in all non-null integer values
                    int_vals = numeric_vals.astype(int)
                    digit_counts = [len(str(abs(v))) for v in int_vals if pd.notna(v)]
                    if digit_counts:
                        max_digits = max(digit_counts)
                        col_constraints['max_digits'] = max_digits
            
            # For numeric (float) columns, detect decimal places and non-negative
            elif pd.api.types.is_numeric_dtype(X[col]):
                numeric_vals = pd.to_numeric(X[col], errors='coerce')
                numeric_vals = numeric_vals[numeric_vals.notna()]
                
                if len(numeric_vals) > 0:
                    min_val = float(numeric_vals.min())
                    # If all values are non-negative, set min_value to 0
                    if min_val >= 0:
                        col_constraints['min_value'] = 0
                    
                    # Detect decimal places (precision)
                    # Convert to string and count decimal places
                    decimal_places = []
                    for v in numeric_vals:
                        if pd.notna(v):
                            v_str = str(v)
                            if '.' in v_str:
                                # Count digits after decimal point
                                dec_part = v_str.split('.')[1]
                                # Remove scientific notation (e.g., '1e-5')
                                if 'e' in dec_part.lower():
                                    continue
                                decimal_places.append(len(dec_part))
                            else:
                                decimal_places.append(0)
                    
                    if decimal_places:
                        max_decimal_places = max(decimal_places)
                        # Only track if there's consistent precision (at least 50% have same precision)
                        if max_decimal_places > 0:
                            col_constraints['decimal_places'] = max_decimal_places
            
            if col_constraints:
                constraints[col] = col_constraints
        
        return constraints
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply schema constraints to synthetic data.
        
        Enforces:
        - String max length (truncates or pads)
        - Integer min/max bounds (clips values)
        - Non-negative values for ID columns
        - Integer digit count (max digits)
        - Float decimal places (precision)
        """
        df = df.copy()
        
        for col, constraints in self._column_constraints.items():
            if col not in df.columns:
                continue
            
            # Apply string length constraints for categorical columns
            if df[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[col]):
                # Handle fixed-length columns (like char(n))
                if 'fixed_length' in constraints:
                    fixed_len = constraints['fixed_length']
                    # Pad or truncate to exact length
                    df[col] = df[col].astype(str).str[:fixed_len].str.ljust(fixed_len, ' ')
                
                # Handle preferred length (most values have same length)
                elif 'preferred_length' in constraints:
                    preferred_len = constraints['preferred_length']
                    min_len = constraints.get('min_length', 0)
                    max_len = constraints.get('max_length', preferred_len)
                    # Truncate if too long, pad if too short (within min-max range)
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str[:max_len]
                    # Pad to preferred length if shorter than min
                    if min_len > 0:
                        df[col] = df[col].str.ljust(min_len, ' ')
                
                # Handle variable length (min-max range)
                elif 'max_length' in constraints or 'min_length' in constraints:
                    min_len = constraints.get('min_length', 0)
                    max_len = constraints.get('max_length', None)
                    
                    df[col] = df[col].astype(str)
                    
                    # Truncate if too long
                    if max_len is not None:
                        df[col] = df[col].str[:max_len]
                    
                    # Pad if too short (to minimum length)
                    if min_len > 0:
                        df[col] = df[col].str.ljust(min_len, ' ')
            
            # Apply integer bounds
            if pd.api.types.is_numeric_dtype(df[col]):
                # Apply min_value constraint (especially for non-negative IDs)
                if 'min_value' in constraints:
                    min_val = constraints['min_value']
                    df[col] = df[col].clip(lower=min_val)
                
                # Apply max_value constraint
                if 'max_value' in constraints:
                    max_val = constraints['max_value']
                    df[col] = df[col].clip(upper=max_val)
                
                # Apply integer digit count constraint
                if 'max_digits' in constraints and pd.api.types.is_integer_dtype(df[col]):
                    max_digits = constraints['max_digits']
                    # Ensure values don't exceed max_digits
                    # Clip to range that fits in max_digits
                    max_val_for_digits = 10 ** max_digits - 1
                    df[col] = df[col].clip(upper=max_val_for_digits)
                    # Round to integers
                    df[col] = df[col].astype(int)
                
                # Apply decimal places constraint for floats
                if 'decimal_places' in constraints and not pd.api.types.is_integer_dtype(df[col]):
                    decimal_places = constraints['decimal_places']
                    # Round to specified decimal places
                    df[col] = df[col].round(decimal_places)
            
            # Ensure dates are valid (not negative timestamps)
            # This runs BEFORE datetime conversion, so values are still numeric (nanoseconds)
            if col in self._datetime_columns and pd.api.types.is_numeric_dtype(df[col]):
                min_timestamp = pd.Timestamp('1900-01-01').value  # nanoseconds since epoch
                df[col] = df[col].clip(lower=min_timestamp)
        
        return df
    
    def _detect_datetime_format(self, date_str: str) -> Optional[str]:
        """Detect datetime format from a sample string.
        
        Returns format string compatible with strftime/strptime, or None if not detected.
        """
        import re
        from datetime import datetime
        
        # Common datetime format patterns
        format_patterns = [
            # Format: 2019-10-01 00:00:00.000
            (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$', '%Y-%m-%d %H:%M:%S.%f'),
            # Format: 2019-10-01 00:00:00
            (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
            # Format: 2019-10-01T00:00:00.000
            (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}$', '%Y-%m-%dT%H:%M:%S.%f'),
            # Format: 2019-10-01T00:00:00
            (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$', '%Y-%m-%dT%H:%M:%S'),
            # Format: 2019-10-01
            (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
            # Format: 01/10/2019
            (r'^\d{2}/\d{2}/\d{4}', '%d/%m/%Y'),
            # Format: 10/01/2019 (US format)
            (r'^\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),
        ]
        
        # Try to match pattern
        for pattern, fmt in format_patterns:
            if re.match(pattern, date_str):
                # Verify format works by trying to parse
                try:
                    datetime.strptime(date_str, fmt)
                    return fmt
                except ValueError:
                    continue
        
        # Try pandas auto-parsing and infer format
        try:
            dt = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(dt):
                # Try common formats
                common_formats = [
                    '%Y-%m-%d %H:%M:%S.%f',  # 2019-10-01 00:00:00.000
                    '%Y-%m-%d %H:%M:%S',     # 2019-10-01 00:00:00
                    '%Y-%m-%dT%H:%M:%S.%f',  # 2019-10-01T00:00:00.000
                    '%Y-%m-%dT%H:%M:%S',     # 2019-10-01T00:00:00
                    '%Y-%m-%d',              # 2019-10-01
                ]
                for fmt in common_formats:
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        if parsed == dt.to_pydatetime():
                            return fmt
                    except ValueError:
                        continue
        except Exception:
            pass
        
        return None


