"""Check if synthetic data works for ML tasks like classification/regression."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_downstream_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,  # 'classification' or 'regression'
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """See if you can train ML models on synthetic data and get similar results.
    
    Trains models on both real and synthetic data, then compares performance.
    Auto-detects target column and task type if not specified.
    """
    if not SKLEARN_AVAILABLE:
        return {
            'error': 'scikit-learn not available. Install with: pip install scikit-learn',
            'available': False
        }
    
    metrics = {
        'available': True,
        'target_column': None,
        'task_type': None,
        'models_tested': [],
        'performance_gaps': {},
        'feature_importance_similarity': {},
    }
    
    # Auto-detect target column
    if target_column is None:
        # Common target column names
        for col in ['income', 'target', 'label', 'class', 'outcome', 'y']:
            if col in real_data.columns:
                target_column = col
                break
        
        # If not found, use last column as fallback
        if target_column is None:
            target_column = real_data.columns[-1]
    
    if target_column not in real_data.columns:
        metrics['error'] = f'Target column "{target_column}" not found in data'
        return metrics
    
    metrics['target_column'] = target_column
    
    # Prepare features and target
    feature_cols = [col for col in real_data.columns if col != target_column]
    
    if len(feature_cols) == 0:
        metrics['error'] = 'No feature columns found'
        return metrics
    
    # Prepare real data
    X_real = real_data[feature_cols].copy()
    y_real = real_data[target_column].copy()
    
    # Prepare synthetic data
    if target_column not in synthetic_data.columns:
        metrics['error'] = f'Target column "{target_column}" not found in synthetic data'
        return metrics
    
    X_synth = synthetic_data[feature_cols].copy()
    y_synth = synthetic_data[target_column].copy()
    
    # Auto-detect task type
    if task_type is None:
        if pd.api.types.is_numeric_dtype(y_real):
            # Check if it's binary (0/1) or multi-class
            unique_vals = y_real.nunique()
            if unique_vals <= 10:  # Likely classification
                task_type = 'classification'
            else:  # Likely regression
                task_type = 'regression'
        else:
            task_type = 'classification'
    
    metrics['task_type'] = task_type
    
    # Encode categorical features and target
    X_real_encoded, X_synth_encoded, encoders = _encode_features(X_real, X_synth)
    
    if task_type == 'classification':
        y_real_encoded, y_synth_encoded, target_encoder = _encode_target(y_real, y_synth)
    else:
        y_real_encoded = y_real.values
        y_synth_encoded = y_synth.values
        target_encoder = None
    
    # Split real data for testing
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real_encoded, y_real_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_real_encoded if task_type == 'classification' else None
    )
    
    # Train models on real and synthetic data
    if task_type == 'classification':
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10),
            'LogisticRegression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
    else:
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10),
            'LinearRegression': LinearRegression()
        }
    
    performance_gaps = {}
    feature_importance_similarity = {}
    
    for model_name, model in models.items():
        # Train on real data
        model_real = _clone_model(model)
        model_real.fit(X_real_train, y_real_train)
        score_real = _evaluate_model(model_real, X_real_test, y_real_test, task_type)
        
        # Train on synthetic data
        model_synth = _clone_model(model)
        model_synth.fit(X_synth_encoded, y_synth_encoded)
        score_synth = _evaluate_model(model_synth, X_real_test, y_real_test, task_type)
        
        # Compute performance gap
        if task_type == 'classification':
            gap = score_real['accuracy'] - score_synth['accuracy']
            relative_gap = gap / max(score_real['accuracy'], 0.01)
        else:
            gap = score_real['r2'] - score_synth['r2']
            relative_gap = gap / max(abs(score_real['r2']), 0.01) if score_real['r2'] != 0 else gap
        
        performance_gaps[model_name] = {
            'real_performance': score_real,
            'synthetic_performance': score_synth,
            'absolute_gap': gap,
            'relative_gap': relative_gap,
        }
        
        # Feature importance similarity (for tree-based models)
        if hasattr(model_real, 'feature_importances_') and hasattr(model_synth, 'feature_importances_'):
            real_importance = model_real.feature_importances_
            synth_importance = model_synth.feature_importances_
            
            # Compute cosine similarity
            importance_similarity = _cosine_similarity(real_importance, synth_importance)
            
            feature_importance_similarity[model_name] = {
                'cosine_similarity': float(importance_similarity),
                'real_importance': {col: float(imp) for col, imp in zip(feature_cols, real_importance)},
                'synthetic_importance': {col: float(imp) for col, imp in zip(feature_cols, synth_importance)},
            }
    
    metrics['models_tested'] = list(models.keys())
    metrics['performance_gaps'] = performance_gaps
    metrics['feature_importance_similarity'] = feature_importance_similarity
    
    # Summary statistics
    if task_type == 'classification':
        avg_gap = np.mean([pg['absolute_gap'] for pg in performance_gaps.values()])
        metrics['summary'] = {
            'average_accuracy_gap': float(avg_gap),
            'best_model': min(performance_gaps.items(), key=lambda x: abs(x[1]['absolute_gap']))[0],
        }
    else:
        avg_gap = np.mean([pg['absolute_gap'] for pg in performance_gaps.values()])
        metrics['summary'] = {
            'average_r2_gap': float(avg_gap),
            'best_model': min(performance_gaps.items(), key=lambda x: abs(x[1]['absolute_gap']))[0],
        }
    
    return metrics


def _encode_features(X_real: pd.DataFrame, X_synth: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Encode categorical features to numeric."""
    encoders = {}
    X_real_encoded = X_real.copy()
    X_synth_encoded = X_synth.copy()
    
    for col in X_real.columns:
        if X_real[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_real[col]):
            le = LabelEncoder()
            # Fit on combined data to ensure all categories are seen
            combined = pd.concat([X_real[col], X_synth[col]], ignore_index=True).astype(str)
            le.fit(combined)
            
            X_real_encoded[col] = le.transform(X_real[col].astype(str))
            X_synth_encoded[col] = le.transform(X_synth[col].astype(str))
            encoders[col] = le
    
    return X_real_encoded.values, X_synth_encoded.values, encoders


def _encode_target(y_real: pd.Series, y_synth: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode categorical target to numeric."""
    le = LabelEncoder()
    combined = pd.concat([y_real, y_synth], ignore_index=True).astype(str)
    le.fit(combined)
    
    y_real_encoded = le.transform(y_real.astype(str))
    y_synth_encoded = le.transform(y_synth.astype(str))
    
    return y_real_encoded, y_synth_encoded, le


def _clone_model(model):
    """Create a copy of a model."""
    from sklearn.base import clone
    return clone(model)


def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str) -> Dict[str, float]:
    """Run model on test data and return accuracy/R²/etc."""
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
        }
        
        # F1 score (works for binary and multi-class)
        try:
            metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted'))
        except:
            metrics['f1_score'] = 0.0
        
        # ROC-AUC (only for binary classification)
        if len(np.unique(y_test)) == 2:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
            except:
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
    else:
        metrics = {
            'mse': float(mean_squared_error(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
        }
    
    return metrics


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def print_downstream_report(metrics: Dict[str, Any], verbose: bool = True) -> None:
    """Print a formatted report of downstream metrics."""
    if not metrics.get('available', False):
        print("Downstream metrics not available (scikit-learn not installed)")
        return
    
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print("=" * 80)
    print("Downstream Task Performance Report")
    print("=" * 80)
    print()
    
    print(f"Target Column: {metrics['target_column']}")
    print(f"Task Type: {metrics['task_type']}")
    print(f"Models Tested: {', '.join(metrics['models_tested'])}")
    print()
    
    print("=" * 80)
    print("Model Performance Comparison")
    print("=" * 80)
    print()
    
    for model_name, perf in metrics['performance_gaps'].items():
        print(f"{model_name}:")
        print(f"  Real Data Performance:")
        for metric, value in perf['real_performance'].items():
            if value is not None:
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
        
        print(f"  Synthetic Data Performance:")
        for metric, value in perf['synthetic_performance'].items():
            if value is not None:
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
        
        gap = perf['absolute_gap']
        rel_gap = perf['relative_gap']
        
        if metrics['task_type'] == 'classification':
            print(f"  Performance Gap: {gap:.4f} (relative: {rel_gap:.2%})")
            print(f"    (Lower gap is better - synthetic data preserves classification patterns)")
        else:
            print(f"  Performance Gap: {gap:.4f} (relative: {rel_gap:.2%})")
            print(f"    (Lower gap is better - synthetic data preserves regression patterns)")
        print()
    
    if metrics.get('feature_importance_similarity'):
        print("=" * 80)
        print("Feature Importance Preservation")
        print("=" * 80)
        print()
        
        for model_name, importance_data in metrics['feature_importance_similarity'].items():
            if hasattr(importance_data, 'get'):
                similarity = importance_data.get('cosine_similarity', 0)
                print(f"{model_name}:")
                print(f"  Cosine Similarity: {similarity:.4f} (1.0 = identical, 0 = different)")
                print()
    
    if 'summary' in metrics:
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        summary = metrics['summary']
        if 'average_accuracy_gap' in summary:
            print(f"Average Accuracy Gap: {summary['average_accuracy_gap']:.4f}")
        elif 'average_r2_gap' in summary:
            print(f"Average R² Gap: {summary['average_r2_gap']:.4f}")
        print(f"Best Preserving Model: {summary.get('best_model', 'N/A')}")
        print()
    
    print("=" * 80)

