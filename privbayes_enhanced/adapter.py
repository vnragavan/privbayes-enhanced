"""High-level adapter for Enhanced PrivBayes."""

from typing import Optional, List
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
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the model to data.
        
        Converts datetimes to numeric, coerces numeric strings, then fits the model.
        """
        X2 = X.copy()
        for c in X2.columns:
            # Convert datetime/timedelta to nanoseconds for processing
            if pd.api.types.is_datetime64_any_dtype(X2[c]):
                X2[c] = X2[c].astype('int64')
            elif pd.api.types.is_timedelta64_dtype(X2[c]):
                X2[c] = X2[c].astype('int64')
            # Try parsing string columns as datetime (handles CSV exports)
            elif X2[c].dtype == 'object':
                try:
                    dt_parsed = pd.to_datetime(X2[c], errors='coerce')
                    if dt_parsed.notna().mean() >= 0.95:
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


