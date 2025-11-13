#!/usr/bin/env python3
"""Generate all available metrics: utility, privacy, QI linkage, inference, downstream, and DP audit."""

import pandas as pd
import json
import os
from privbayes_enhanced import EnhancedPrivBayesAdapter

def main():
    print("=" * 80)
    print("Enhanced PrivBayes - Complete Metrics Generation")
    print("=" * 80)
    print()
    
    # Load data
    data_path = '../data/adult.csv'
    if not os.path.exists(data_path):
        # Try alternative path
        data_path = 'data/adult.csv'
        if not os.path.exists(data_path):
            print(f"Error: Data file not found: {data_path}")
            print("Please run from project root or ensure data/adult.csv exists")
            return
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data)} rows, {len(data.columns)} columns")
    print(f"  Columns: {', '.join(data.columns[:5])}...")
    print()
    
    # Create and fit model
    print("Creating Enhanced PrivBayes model...")
    model = EnhancedPrivBayesAdapter(
        epsilon=1.0,
        delta=1e-6,
        seed=42,
        temperature=1.0,
        cat_keep_all_nonzero=True
    )
    print("  Epsilon: 1.0")
    print("  Delta: 1e-6")
    print("  Seed: 42")
    print()
    
    print("Fitting model to data...")
    model.fit(data)
    print("✓ Model fitted successfully")
    print()
    
    # Generate synthetic data
    n_samples = min(1000, len(data))
    print(f"Generating {n_samples} synthetic samples...")
    synthetic = model.sample(n_samples=n_samples)
    print(f"✓ Generated {len(synthetic)} synthetic samples")
    print()
    
    # Privacy Budget Report
    print("=" * 80)
    print("1. PRIVACY BUDGET REPORT")
    print("=" * 80)
    privacy_report = model.privacy_report()
    print(f"Total epsilon used: {privacy_report['epsilon_total']:.6f}")
    print(f"Delta used: {privacy_report['delta']:.2e}")
    print(f"Epsilon for structure: {privacy_report['eps_struct']:.6f}")
    print(f"Epsilon for CPT: {privacy_report['eps_cpt']:.6f}")
    print(f"Epsilon for discovery: {privacy_report['eps_disc']:.6f}")
    print(f"Temperature: {privacy_report['temperature']:.2f}")
    print()
    
    # Utility Metrics
    print("=" * 80)
    print("2. UTILITY METRICS")
    print("=" * 80)
    utility_metrics = model.evaluate_utility(data, synthetic, verbose=True)
    print()
    
    # Privacy Metrics
    print("=" * 80)
    print("3. PRIVACY METRICS")
    print("=" * 80)
    privacy_metrics = model.evaluate_privacy(data, synthetic, verbose=True)
    print()
    
    # QI Linkage Risk
    print("=" * 80)
    print("4. QI LINKAGE RISK METRICS")
    print("=" * 80)
    qi_columns = ['age', 'sex', 'race']  # Common QI columns for adult dataset
    qi_metrics = model.evaluate_qi_linkage(
        data, synthetic,
        qi_columns=qi_columns,
        k=5,  # 5-anonymity
        verbose=True
    )
    print()
    
    # Inference Attack Risk
    print("=" * 80)
    print("5. INFERENCE ATTACK RISK METRICS")
    print("=" * 80)
    inference_metrics = model.evaluate_inference_attack(
        data, synthetic,
        sensitive_columns=['income'],  # Income is sensitive
        qi_columns=qi_columns,
        verbose=True
    )
    print()
    
    # DP Compliance Audit
    print("=" * 80)
    print("6. DP COMPLIANCE AUDIT")
    print("=" * 80)
    audit_results = model.audit_dp_compliance(strict=True, verbose=True)
    print()
    
    # Downstream Task Metrics
    print("=" * 80)
    print("7. DOWNSTREAM TASK METRICS")
    print("=" * 80)
    downstream_metrics = model.evaluate_downstream(
        data, synthetic,
        target_column='income',  # Income is the target for adult dataset
        verbose=True
    )
    print()
    
    # Save all metrics to JSON
    print("=" * 80)
    print("8. SAVING ALL METRICS")
    print("=" * 80)
    all_metrics = {
        'privacy_budget': privacy_report,
        'utility': utility_metrics,
        'privacy': privacy_metrics,
        'qi_linkage': qi_metrics,
        'inference_attack': inference_metrics,
        'dp_audit': audit_results,
        'downstream': downstream_metrics,
    }
    
    output_file = 'all_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"✓ All metrics saved to: {output_file}")
    print()
    
    # Summary
    print("=" * 80)
    print("METRICS GENERATION SUMMARY")
    print("=" * 80)
    print(f"✓ Privacy budget: {privacy_report['epsilon_total']:.6f} ε, {privacy_report['delta']:.2e} δ")
    
    utility_summary = utility_metrics.get('summary', {})
    if 'mean_numeric_ks_statistic' in utility_summary:
        print(f"✓ Average KS statistic (numeric): {utility_summary['mean_numeric_ks_statistic']:.4f}")
    if 'mean_categorical_jsd' in utility_summary:
        print(f"✓ Average JSD (categorical): {utility_summary['mean_categorical_jsd']:.4f}")
    
    privacy_summary = privacy_metrics
    print(f"✓ Exact match rate: {privacy_summary.get('exact_match_rate', 0):.4f}")
    print(f"✓ UNK token rate: {privacy_summary.get('unk_token_rate', 0):.4f}")
    
    qi_summary = qi_metrics
    if 'exact_qi_match_rate' in qi_summary:
        print(f"✓ QI exact match rate: {qi_summary['exact_qi_match_rate']:.4f}")
        print(f"✓ K-anonymity violation rate: {qi_summary.get('k_anonymity_violation_rate', 0):.4f}")
    
    inference_summary = inference_metrics
    if 'avg_unique_inference_rate' in inference_summary:
        print(f"✓ Average unique inference rate: {inference_summary['avg_unique_inference_rate']:.4f}")
    
    audit_summary = audit_results.get('checklist_summary', {})
    print(f"✓ DP compliance: {audit_summary.get('passed_items', 0)}/{audit_summary.get('total_items', 0)} items passed")
    
    downstream_summary = downstream_metrics
    if downstream_summary.get('available', False) and 'summary' in downstream_summary:
        if 'average_accuracy_gap' in downstream_summary['summary']:
            print(f"✓ Average accuracy gap: {downstream_summary['summary']['average_accuracy_gap']:.4f}")
        elif 'average_r2_gap' in downstream_summary['summary']:
            print(f"✓ Average R² gap: {downstream_summary['summary']['average_r2_gap']:.4f}")
    elif not downstream_summary.get('available', False):
        print("⚠ Downstream metrics not available (scikit-learn not installed)")
    print()
    
    print("=" * 80)
    print("All metrics generated successfully!")
    print(f"Full results saved to: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

