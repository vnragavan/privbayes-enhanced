"""Metrics for checking how good the synthetic data is."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def compute_utility_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare real vs synthetic data to see how similar they are.
    
    Checks means, stds, distributions (KS test, Wasserstein), correlations,
    and whether categories are preserved. Auto-detects column types if not given.
    """
    metrics = {}
    
    # Auto-detect column types if not provided
    if categorical_columns is None:
        categorical_columns = [
            col for col in real_data.columns
            if not pd.api.types.is_numeric_dtype(real_data[col])
        ]
    
    if numeric_columns is None:
        numeric_columns = [
            col for col in real_data.columns
            if pd.api.types.is_numeric_dtype(real_data[col])
        ]
    
    # Remove __UNK__ from categorical columns for analysis
    cat_cols_clean = [col for col in categorical_columns if col in real_data.columns]
    num_cols_clean = [col for col in numeric_columns if col in real_data.columns]
    
    # 1. Numeric column statistics
    numeric_stats = {}
    for col in num_cols_clean:
        if col not in real_data.columns or col not in synthetic_data.columns:
            continue
        
        real_col = pd.to_numeric(real_data[col], errors='coerce').dropna()
        synth_col = pd.to_numeric(synthetic_data[col], errors='coerce').dropna()
        
        if len(real_col) == 0 or len(synth_col) == 0:
            continue
        
        # Compute marginal distributional errors
        # Kolmogorov-Smirnov statistic (0 = identical, 1 = completely different)
        try:
            ks_stat, ks_pvalue = ks_2samp(real_col, synth_col)
        except Exception:
            ks_stat, ks_pvalue = np.nan, np.nan
        
        # Wasserstein distance (Earth Mover's Distance)
        try:
            wass_dist = wasserstein_distance(real_col, synth_col)
        except Exception:
            wass_dist = np.nan
        
        # Jensen-Shannon divergence on binned data
        # Create histograms with same bins
        try:
            min_val = min(real_col.min(), synth_col.min())
            max_val = max(real_col.max(), synth_col.max())
            n_bins = min(50, max(10, int(np.sqrt(len(real_col)))))
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            real_hist, _ = np.histogram(real_col, bins=bins)
            synth_hist, _ = np.histogram(synth_col, bins=bins)
            
            # Normalize to probabilities
            real_probs = real_hist / (real_hist.sum() + 1e-10)
            synth_probs = synth_hist / (synth_hist.sum() + 1e-10)
            
            jsd = float(jensenshannon(real_probs, synth_probs))
        except Exception:
            jsd = np.nan
        
        numeric_stats[col] = {
            'mean_error': abs(real_col.mean() - synth_col.mean()) / (abs(real_col.mean()) + 1e-10),
            'std_error': abs(real_col.std() - synth_col.std()) / (abs(real_col.std()) + 1e-10),
            'median_error': abs(real_col.median() - synth_col.median()) / (abs(real_col.median()) + 1e-10),
            'min_preserved': real_col.min() <= synth_col.max() and synth_col.min() <= real_col.max(),
            'max_preserved': real_col.max() >= synth_col.min() and synth_col.max() >= real_col.min(),
            'real_mean': float(real_col.mean()),
            'synth_mean': float(synth_col.mean()),
            'real_std': float(real_col.std()),
            'synth_std': float(synth_col.std()),
            # Marginal distributional errors
            'kolmogorov_smirnov_statistic': float(ks_stat) if not np.isnan(ks_stat) else None,
            'kolmogorov_smirnov_pvalue': float(ks_pvalue) if not np.isnan(ks_pvalue) else None,
            'wasserstein_distance': float(wass_dist) if not np.isnan(wass_dist) else None,
            'jensen_shannon_divergence': float(jsd) if not np.isnan(jsd) else None,
        }
    
    metrics['numeric_statistics'] = numeric_stats
    
    # 2. Categorical distribution similarity (Jensen-Shannon divergence)
    categorical_jsd = {}
    for col in cat_cols_clean:
        if col not in real_data.columns or col not in synthetic_data.columns:
            continue
        
        real_col = real_data[col].astype(str).fillna('__MISSING__')
        synth_col = synthetic_data[col].astype(str).fillna('__MISSING__')
        
        # Get all unique values
        all_values = set(real_col.unique()) | set(synth_col.unique())
        all_values = sorted([v for v in all_values if v != '__UNK__'])  # Exclude UNK for comparison
        
        if len(all_values) == 0:
            continue
        
        # Compute probability distributions
        real_counts = real_col.value_counts()
        synth_counts = synth_col.value_counts()
        
        real_probs = np.array([real_counts.get(v, 0) for v in all_values], dtype=float)
        synth_probs = np.array([synth_counts.get(v, 0) for v in all_values], dtype=float)
        
        # Normalize
        real_probs = real_probs / (real_probs.sum() + 1e-10)
        synth_probs = synth_probs / (synth_probs.sum() + 1e-10)
        
        # Jensen-Shannon divergence (0 = identical, 1 = completely different)
        jsd = float(jensenshannon(real_probs, synth_probs))
        
        # Coverage: percentage of real categories present in synthetic
        real_unique = set(real_col.unique())
        synth_unique = set(synth_col.unique())
        coverage = len(real_unique & synth_unique) / max(len(real_unique), 1)
        
        # Total Variation Distance (L1 distance between distributions)
        tv_distance = float(0.5 * np.sum(np.abs(real_probs - synth_probs)))
        
        categorical_jsd[col] = {
            'jensen_shannon_divergence': jsd,
            'total_variation_distance': tv_distance,  # 0 = identical, 1 = completely different
            'coverage': coverage,
            'real_unique_count': len(real_unique),
            'synth_unique_count': len(synth_unique),
            'overlap_count': len(real_unique & synth_unique),
        }
    
    metrics['categorical_similarity'] = categorical_jsd
    
    # 2.5. Marginal distributional error summary (for all columns)
    marginal_errors = {}
    
    # Add numeric column marginal errors
    for col, stats in numeric_stats.items():
        marginal_errors[col] = {
            'type': 'numeric',
            'kolmogorov_smirnov': stats.get('kolmogorov_smirnov_statistic'),
            'wasserstein_distance': stats.get('wasserstein_distance'),
            'jensen_shannon_divergence': stats.get('jensen_shannon_divergence'),
        }
    
    # Add categorical column marginal errors
    for col, stats in categorical_jsd.items():
        marginal_errors[col] = {
            'type': 'categorical',
            'jensen_shannon_divergence': stats.get('jensen_shannon_divergence'),
            'total_variation_distance': stats.get('total_variation_distance'),
        }
    
    metrics['marginal_distributional_errors'] = marginal_errors
    
    # 3. Correlation preservation (for numeric columns)
    if len(num_cols_clean) > 1:
        real_numeric = real_data[num_cols_clean].select_dtypes(include=[np.number])
        synth_numeric = synthetic_data[num_cols_clean].select_dtypes(include=[np.number])
        
        # Only compute if we have enough columns
        if real_numeric.shape[1] > 1 and synth_numeric.shape[1] > 1:
            real_corr = real_numeric.corr().values
            synth_corr = synth_numeric.corr().values
            
            # Flatten and remove diagonal
            mask = ~np.eye(real_corr.shape[0], dtype=bool)
            real_corr_flat = real_corr[mask]
            synth_corr_flat = synth_corr[mask]
            
            # Correlation of correlations (how well correlations are preserved)
            if len(real_corr_flat) > 0 and np.std(real_corr_flat) > 1e-10:
                corr_corr = float(np.corrcoef(real_corr_flat, synth_corr_flat)[0, 1])
                corr_mae = float(np.mean(np.abs(real_corr_flat - synth_corr_flat)))
                
                metrics['correlation_preservation'] = {
                    'correlation_of_correlations': corr_corr if not np.isnan(corr_corr) else 0.0,
                    'mean_absolute_error': corr_mae,
                }
    
    # 4. Overall summary statistics
    metrics['summary'] = {
        'n_real_samples': len(real_data),
        'n_synthetic_samples': len(synthetic_data),
        'n_numeric_columns': len(num_cols_clean),
        'n_categorical_columns': len(cat_cols_clean),
    }
    
    # Compute average metrics
    if numeric_stats:
        metrics['summary']['mean_numeric_mean_error'] = np.mean([
            v['mean_error'] for v in numeric_stats.values()
        ])
        metrics['summary']['mean_numeric_std_error'] = np.mean([
            v['std_error'] for v in numeric_stats.values()
        ])
        # Average marginal distributional errors for numeric columns
        ks_vals = [v['kolmogorov_smirnov_statistic'] for v in numeric_stats.values() 
                   if v.get('kolmogorov_smirnov_statistic') is not None]
        wass_vals = [v['wasserstein_distance'] for v in numeric_stats.values() 
                     if v.get('wasserstein_distance') is not None]
        jsd_num_vals = [v['jensen_shannon_divergence'] for v in numeric_stats.values() 
                        if v.get('jensen_shannon_divergence') is not None]
        
        if ks_vals:
            metrics['summary']['mean_numeric_ks_statistic'] = np.mean(ks_vals)
        if wass_vals:
            metrics['summary']['mean_numeric_wasserstein_distance'] = np.mean(wass_vals)
        if jsd_num_vals:
            metrics['summary']['mean_numeric_jsd'] = np.mean(jsd_num_vals)
    
    if categorical_jsd:
        metrics['summary']['mean_categorical_jsd'] = np.mean([
            v['jensen_shannon_divergence'] for v in categorical_jsd.values()
        ])
        metrics['summary']['mean_categorical_tv_distance'] = np.mean([
            v['total_variation_distance'] for v in categorical_jsd.values()
        ])
        metrics['summary']['mean_categorical_coverage'] = np.mean([
            v['coverage'] for v in categorical_jsd.values()
        ])
    
    return metrics


def compute_privacy_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    privacy_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check privacy heuristics like exact matches and UNK tokens.
    
    These are just heuristics - real privacy comes from the DP mechanism itself.
    """
    metrics = {}
    
    # 1. Exact match rate (lower is better for privacy)
    # Check if any synthetic row exactly matches a real row
    exact_matches = 0
    real_str = real_data.astype(str).apply(lambda x: '|'.join(x), axis=1)
    synth_str = synthetic_data.astype(str).apply(lambda x: '|'.join(x), axis=1)
    
    exact_matches = len(set(real_str) & set(synth_str))
    exact_match_rate = exact_matches / max(len(synthetic_data), 1)
    
    metrics['exact_match_rate'] = exact_match_rate
    metrics['n_exact_matches'] = exact_matches
    
    # 2. UNK token rate (indicator of privacy-preserving categorical discovery)
    unk_count = 0
    total_categorical_values = 0
    for col in synthetic_data.columns:
        if synthetic_data[col].dtype == 'object':
            col_str = synthetic_data[col].astype(str)
            unk_count += (col_str == '__UNK__').sum()
            total_categorical_values += len(col_str)
    
    unk_rate = unk_count / max(total_categorical_values, 1)
    metrics['unk_token_rate'] = unk_rate
    metrics['n_unk_tokens'] = int(unk_count)
    
    # 3. Privacy report integration
    if privacy_report:
        metrics['privacy_budget'] = {
            'epsilon_total': privacy_report.get('epsilon_total_actual', privacy_report.get('epsilon_total', 0)),
            'delta': privacy_report.get('delta_used', privacy_report.get('delta', 0)),
            'mechanism': privacy_report.get('mechanism', 'unknown'),
        }
    
    return metrics


def print_utility_report(metrics: Dict[str, Any], verbose: bool = True) -> None:
    """Print a formatted utility metrics report."""
    print("=" * 80)
    print("Utility Metrics Report")
    print("=" * 80)
    
    summary = metrics.get('summary', {})
    print(f"\nDataset Summary:")
    print(f"  Real samples: {summary.get('n_real_samples', 0):,}")
    print(f"  Synthetic samples: {summary.get('n_synthetic_samples', 0):,}")
    print(f"  Numeric columns: {summary.get('n_numeric_columns', 0)}")
    print(f"  Categorical columns: {summary.get('n_categorical_columns', 0)}")
    
    # Marginal distributional errors summary
    marginal_errors = metrics.get('marginal_distributional_errors', {})
    if marginal_errors:
        print(f"\nMarginal Distributional Errors (Per Column):")
        numeric_cols = [col for col, err in marginal_errors.items() if err.get('type') == 'numeric']
        cat_cols = [col for col, err in marginal_errors.items() if err.get('type') == 'categorical']
        
        if numeric_cols:
            avg_ks = np.mean([marginal_errors[col].get('kolmogorov_smirnov') 
                             for col in numeric_cols 
                             if marginal_errors[col].get('kolmogorov_smirnov') is not None])
            avg_wass = np.mean([marginal_errors[col].get('wasserstein_distance') 
                               for col in numeric_cols 
                               if marginal_errors[col].get('wasserstein_distance') is not None])
            if not np.isnan(avg_ks):
                print(f"  Average KS statistic (numeric): {avg_ks:.4f} (0=identical, 1=different)")
            if not np.isnan(avg_wass):
                print(f"  Average Wasserstein distance (numeric): {avg_wass:.4f}")
        
        if cat_cols:
            avg_jsd = np.mean([marginal_errors[col].get('jensen_shannon_divergence') 
                              for col in cat_cols 
                              if marginal_errors[col].get('jensen_shannon_divergence') is not None])
            if not np.isnan(avg_jsd):
                print(f"  Average JSD (categorical): {avg_jsd:.4f} (0=identical, 1=different)")
    
    # Numeric statistics
    numeric_stats = metrics.get('numeric_statistics', {})
    if numeric_stats:
        print(f"\nNumeric Column Statistics:")
        if 'mean_numeric_mean_error' in summary:
            print(f"  Average mean error: {summary['mean_numeric_mean_error']:.4f}")
        if 'mean_numeric_std_error' in summary:
            print(f"  Average std error: {summary['mean_numeric_std_error']:.4f}")
        if 'mean_numeric_ks_statistic' in summary:
            print(f"  Average KS statistic: {summary['mean_numeric_ks_statistic']:.4f}")
        if 'mean_numeric_wasserstein_distance' in summary:
            print(f"  Average Wasserstein distance: {summary['mean_numeric_wasserstein_distance']:.4f}")
        
        if verbose:
            for col, stats in numeric_stats.items():
                print(f"\n  {col}:")
                print(f"    Mean error: {stats['mean_error']:.4f}")
                print(f"    Std error: {stats['std_error']:.4f}")
                print(f"    Real mean: {stats['real_mean']:.2f}, Synthetic mean: {stats['synth_mean']:.2f}")
                if stats.get('kolmogorov_smirnov_statistic') is not None:
                    print(f"    KS statistic: {stats['kolmogorov_smirnov_statistic']:.4f} (p={stats.get('kolmogorov_smirnov_pvalue', 0):.4f})")
                if stats.get('wasserstein_distance') is not None:
                    print(f"    Wasserstein distance: {stats['wasserstein_distance']:.4f}")
                if stats.get('jensen_shannon_divergence') is not None:
                    print(f"    JSD (binned): {stats['jensen_shannon_divergence']:.4f}")
    
    # Categorical similarity
    cat_sim = metrics.get('categorical_similarity', {})
    if cat_sim:
        print(f"\nCategorical Distribution Similarity:")
        if 'mean_categorical_jsd' in summary:
            print(f"  Average Jensen-Shannon divergence: {summary['mean_categorical_jsd']:.4f}")
            print(f"    (0 = identical, 1 = completely different)")
        if 'mean_categorical_coverage' in summary:
            print(f"  Average coverage: {summary['mean_categorical_coverage']:.2%}")
        
        if verbose:
            for col, stats in cat_sim.items():
                print(f"\n  {col}:")
                print(f"    JSD: {stats['jensen_shannon_divergence']:.4f}")
                print(f"    TV distance: {stats.get('total_variation_distance', 0):.4f}")
                print(f"    Coverage: {stats['coverage']:.2%}")
                print(f"    Real unique: {stats['real_unique_count']}, Synthetic unique: {stats['synth_unique_count']}")
    
    # Correlation preservation
    corr = metrics.get('correlation_preservation', {})
    if corr:
        print(f"\nCorrelation Preservation:")
        print(f"  Correlation of correlations: {corr.get('correlation_of_correlations', 0):.4f}")
        print(f"    (1.0 = perfect preservation, 0 = no preservation)")
        print(f"  Mean absolute error: {corr.get('mean_absolute_error', 0):.4f}")
    
    print("\n" + "=" * 80)


def print_privacy_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted privacy metrics report."""
    print("=" * 80)
    print("Privacy Metrics Report")
    print("=" * 80)
    
    print(f"\nExact Match Rate: {metrics.get('exact_match_rate', 0):.4f}")
    print(f"  (Lower is better - indicates fewer exact copies of real data)")
    print(f"  Exact matches: {metrics.get('n_exact_matches', 0)}")
    
    print(f"\nUNK Token Rate: {metrics.get('unk_token_rate', 0):.4f}")
    print(f"  UNK tokens: {metrics.get('n_unk_tokens', 0)}")
    
    privacy_budget = metrics.get('privacy_budget', {})
    if privacy_budget:
        print(f"\nPrivacy Budget:")
        print(f"  Epsilon: {privacy_budget.get('epsilon_total', 0):.6f}")
        print(f"  Delta: {privacy_budget.get('delta', 0):.2e}")
        print(f"  Mechanism: {privacy_budget.get('mechanism', 'unknown')}")
    
    print("\n" + "=" * 80)

