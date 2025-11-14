#!/usr/bin/env python3
"""CLI for Enhanced PrivBayes."""

import argparse
import sys
import os
import json
import pandas as pd
from typing import Optional
from . import EnhancedPrivBayesAdapter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced PrivBayes: Differentially Private Bayesian Network Synthesizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  privbayes data/adult.csv -o synthetic.csv --epsilon 1.0

  # Generate data and all metrics in one command
  privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --all-metrics

  # With public knowledge
  privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 \\
    --public-categories state:CA,NY,TX,FL,IL \\
    --public-bounds age:0,120

  # Generate more samples
  privbayes data/adult.csv -o synthetic.csv --epsilon 1.0 --n-samples 5000

  # Save model and load later
  privbayes data/adult.csv --fit-model model.pkl --epsilon 1.0
  privbayes --load-model model.pkl -o synthetic.csv --n-samples 1000
        """
    )
    
    # Input/Output
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input CSV file path (required if not loading model)'
    )
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='Output CSV file path for synthetic data'
    )
    parser.add_argument(
        '--fit-model',
        dest='model_file',
        help='Save fitted model to this file (pickle format)'
    )
    parser.add_argument(
        '--load-model',
        dest='load_model_file',
        help='Load fitted model from this file (pickle format)'
    )
    
    # Privacy parameters
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1.0,
        help='Privacy budget epsilon (default: 1.0)'
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=1e-6,
        help='Privacy parameter delta (default: 1e-6)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--n-samples',
        type=int,
        dest='n_samples',
        help='Number of synthetic samples to generate (default: same as input)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for sampling (T>1 reduces linkage, default: 1.0)'
    )
    
    # Public knowledge
    parser.add_argument(
        '--public-categories',
        dest='public_categories',
        help='Public categories as col:val1,val2,... (can be repeated)'
    )
    parser.add_argument(
        '--public-bounds',
        dest='public_bounds',
        help='Public bounds as col:min,max (can be repeated)'
    )
    parser.add_argument(
        '--label-columns',
        dest='label_columns',
        help='Label columns (comma-separated, no hashing, no UNK)'
    )
    parser.add_argument(
        '--auto-detect-label-columns',
        dest='auto_detect_label_columns',
        action='store_true',
        default=True,
        help='Automatically treat all categorical columns as label columns (preserves actual names, no hash buckets). Default: True'
    )
    parser.add_argument(
        '--no-auto-detect-label-columns',
        dest='auto_detect_label_columns',
        action='store_false',
        help='Disable automatic label column detection. Only use manually specified --label-columns'
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        help='JSON configuration file path'
    )
    
    # Other options
    parser.add_argument(
        '--cat-keep-all-nonzero',
        action='store_true',
        default=True,
        help='Keep all observed categories (minimize UNK tokens, default: True)'
    )
    parser.add_argument(
        '--no-cat-keep-all-nonzero',
        dest='cat_keep_all_nonzero',
        action='store_false',
        help='Disable keeping all categories'
    )
    parser.add_argument(
        '--max-parents',
        type=int,
        default=2,
        help='Maximum parents in Bayesian network (default: 2)'
    )
    parser.add_argument(
        '--bins-per-numeric',
        type=int,
        default=16,
        help='Number of bins for numeric columns (default: 16)'
    )
    parser.add_argument(
        '--eps-split',
        help='Epsilon split as structure:CPT (e.g., 0.3:0.7)'
    )
    
    # Output options
    parser.add_argument(
        '--privacy-report',
        action='store_true',
        help='Print privacy report to stdout'
    )
    parser.add_argument(
        '--save-report',
        dest='report_file',
        help='Save privacy report to JSON file'
    )
    parser.add_argument(
        '--evaluate-utility',
        action='store_true',
        help='Evaluate utility metrics (requires input file)'
    )
    parser.add_argument(
        '--evaluate-privacy',
        action='store_true',
        help='Evaluate privacy metrics (requires input file)'
    )
    parser.add_argument(
        '--evaluate-qi-linkage',
        action='store_true',
        help='Evaluate QI linkage attack risk (requires input file)'
    )
    parser.add_argument(
        '--evaluate-inference',
        action='store_true',
        help='Evaluate inference attack risk (requires input file)'
    )
    parser.add_argument(
        '--audit-dp',
        action='store_true',
        help='Audit differential privacy compliance'
    )
    parser.add_argument(
        '--save-metrics',
        dest='metrics_file',
        help='Save utility and privacy metrics to JSON file'
    )
    parser.add_argument(
        '--all-metrics',
        action='store_true',
        help='Generate all metrics (utility, privacy, QI linkage, inference attack, downstream, DP audit)'
    )
    parser.add_argument(
        '--target-column',
        dest='target_column',
        help='Target column for downstream metrics (auto-detected if not specified)'
    )
    parser.add_argument(
        '--qi-columns',
        dest='qi_columns',
        help='Comma-separated list of quasi-identifier columns for QI linkage and inference attack'
    )
    parser.add_argument(
        '--sensitive-columns',
        dest='sensitive_columns',
        help='Comma-separated list of sensitive columns for inference attack'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def parse_public_categories(categories_str: Optional[str]) -> dict:
    """Parse public categories from string format: col:val1,val2,..."""
    if not categories_str:
        return {}
    
    result = {}
    # Support multiple --public-categories arguments
    if isinstance(categories_str, list):
        parts = categories_str
    else:
        parts = [categories_str]
    
    for part in parts:
        if ':' not in part:
            continue
        col, vals = part.split(':', 1)
        col = col.strip()
        vals = [v.strip() for v in vals.split(',')]
        if col not in result:
            result[col] = []
        result[col].extend(vals)
    
    return result


def parse_public_bounds(bounds_str: Optional[str]) -> dict:
    """Parse public bounds from string format: col:min,max"""
    if not bounds_str:
        return {}
    
    result = {}
    # Support multiple --public-bounds arguments
    if isinstance(bounds_str, list):
        parts = bounds_str
    else:
        parts = [bounds_str]
    
    for part in parts:
        if ':' not in part:
            continue
        col, bounds = part.split(':', 1)
        col = col.strip()
        try:
            min_val, max_val = bounds.split(',')
            result[col] = [float(min_val.strip()), float(max_val.strip())]
        except ValueError:
            print(f"Warning: Invalid bounds format for {col}: {bounds}", file=sys.stderr)
    
    return result


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def save_model(model: EnhancedPrivBayesAdapter, filepath: str):
    """Save model to pickle file."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> EnhancedPrivBayesAdapter:
    """Load model from pickle file."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Load config file if provided (overrides command-line args)
    config = {}
    if args.config:
        config = load_config(args.config)
        if args.verbose:
            print(f"Loaded configuration from {args.config}")
    
    # Check if loading model or fitting new one
    if args.load_model_file:
        if args.verbose:
            print(f"Loading model from {args.load_model_file}...")
        model = load_model(args.load_model_file)
        if args.verbose:
            print("Model loaded successfully")
    else:
        # Need input file to fit model
        if not args.input_file:
            print("Error: Input file required (or use --load-model)", file=sys.stderr)
            sys.exit(1)
        
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        
        # Load data
        if args.verbose:
            print(f"Loading data from {args.input_file}...")
        try:
            data = pd.read_csv(args.input_file)
            if args.verbose:
                print(f"Loaded {len(data)} rows, {len(data.columns)} columns")
        except Exception as e:
            print(f"Error loading data: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Build model parameters
        model_kwargs = {
            'epsilon': config.get('epsilon', args.epsilon),
            'delta': config.get('delta', args.delta),
            'seed': config.get('seed', args.seed),
            'temperature': config.get('temperature', args.temperature),
            'cat_keep_all_nonzero': config.get('cat_keep_all_nonzero', args.cat_keep_all_nonzero),
            'max_parents': config.get('max_parents', args.max_parents),
            'bins_per_numeric': config.get('bins_per_numeric', args.bins_per_numeric),
        }
        
        # Parse public categories
        pub_cats = config.get('public_categories', {})
        if args.public_categories:
            parsed_cats = parse_public_categories(args.public_categories)
            pub_cats.update(parsed_cats)
        if pub_cats:
            model_kwargs['public_categories'] = pub_cats
        
        # Parse public bounds
        pub_bounds = config.get('public_bounds', {})
        if args.public_bounds:
            parsed_bounds = parse_public_bounds(args.public_bounds)
            pub_bounds.update(parsed_bounds)
        if pub_bounds:
            model_kwargs['public_bounds'] = pub_bounds
        
        # Parse label columns
        if args.label_columns:
            model_kwargs['label_columns'] = [c.strip() for c in args.label_columns.split(',')]
        elif 'label_columns' in config:
            model_kwargs['label_columns'] = config['label_columns']
        
        # Handle auto_detect_label_columns
        if 'auto_detect_label_columns' not in model_kwargs:
            model_kwargs['auto_detect_label_columns'] = args.auto_detect_label_columns
        elif 'auto_detect_label_columns' in config:
            model_kwargs['auto_detect_label_columns'] = config['auto_detect_label_columns']
        
        # Parse epsilon split
        if args.eps_split:
            s, c = args.eps_split.split(':')
            model_kwargs['eps_split'] = {
                'structure': float(s),
                'cpt': float(c)
            }
        elif 'eps_split' in config:
            model_kwargs['eps_split'] = config['eps_split']
        
        # Create and fit model
        if args.verbose:
            print("Creating Enhanced PrivBayes model...")
            print(f"  Epsilon: {model_kwargs['epsilon']}")
            print(f"  Delta: {model_kwargs['delta']}")
            print(f"  Seed: {model_kwargs['seed']}")
        
        model = EnhancedPrivBayesAdapter(**model_kwargs)
        
        if args.verbose:
            print("Fitting model to data...")
        model.fit(data)
        
        if args.verbose:
            print("Model fitted successfully")
        
        # Save model if requested
        if args.model_file:
            save_model(model, args.model_file)
    
    # Generate synthetic data
    if args.output_file or args.privacy_report or args.report_file:
        n_samples = args.n_samples
        if n_samples is None and not args.load_model_file:
            # Use original data size if not specified
            n_samples = len(data)
        elif n_samples is None:
            n_samples = 1000  # Default for loaded models
        
        if args.verbose:
            print(f"Generating {n_samples} synthetic samples...")
        
        synthetic = model.sample(n_samples=n_samples)
        
        if args.verbose:
            print(f"Generated {len(synthetic)} synthetic samples")
        
        # Save synthetic data
        if args.output_file:
            synthetic.to_csv(args.output_file, index=False)
            print(f"Synthetic data saved to {args.output_file}")
        
        # Privacy report
        report = model.privacy_report()
        
        if args.privacy_report:
            print("\n" + "=" * 80)
            print("Privacy Report")
            print("=" * 80)
            print(f"Total epsilon used: {report['epsilon_total']:.6f}")
            print(f"Delta used: {report['delta']:.2e}")
            print(f"Epsilon for structure: {report['eps_struct']:.6f}")
            print(f"Epsilon for CPT: {report['eps_cpt']:.6f}")
            print(f"Epsilon for discovery: {report['eps_disc']:.6f}")
            print(f"Temperature: {report['temperature']:.2f}")
            print("=" * 80)
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Privacy report saved to {args.report_file}")
        
        # Evaluate utility and privacy metrics if requested
        all_metrics = {}
        
        # If --all-metrics is set, enable all evaluation flags
        if args.all_metrics:
            args.evaluate_utility = True
            args.evaluate_privacy = True
            args.evaluate_qi_linkage = True
            args.evaluate_inference = True
            args.audit_dp = True
            if not args.metrics_file:
                args.metrics_file = 'all_metrics.json'
        
        if args.evaluate_utility or args.evaluate_privacy or args.evaluate_qi_linkage or args.evaluate_inference or args.all_metrics:
            if args.load_model_file:
                print("Warning: Cannot evaluate metrics when loading model without original data.", file=sys.stderr)
                print("Provide input_file to evaluate metrics.", file=sys.stderr)
            else:
                if args.evaluate_utility:
                    if args.verbose:
                        print("\nEvaluating utility metrics...")
                    from .metrics import compute_utility_metrics, print_utility_report
                    utility_metrics = compute_utility_metrics(data, synthetic)
                    print_utility_report(utility_metrics, verbose=args.verbose)
                    all_metrics['utility'] = utility_metrics
                
                if args.evaluate_privacy:
                    if args.verbose:
                        print("\nEvaluating privacy metrics...")
                    from .metrics import compute_privacy_metrics, print_privacy_report
                    privacy_metrics = compute_privacy_metrics(data, synthetic, report)
                    print_privacy_report(privacy_metrics)
                    all_metrics['privacy'] = privacy_metrics
                
                if args.evaluate_qi_linkage:
                    if args.verbose:
                        print("\nEvaluating QI linkage risk...")
                    qi_cols = [c.strip() for c in args.qi_columns.split(',')] if args.qi_columns else None
                    qi_metrics = model.evaluate_qi_linkage(data, synthetic, qi_columns=qi_cols, verbose=args.verbose)
                    all_metrics['qi_linkage'] = qi_metrics
                
                if args.evaluate_inference:
                    if args.verbose:
                        print("\nEvaluating inference attack risk...")
                    sens_cols = [c.strip() for c in args.sensitive_columns.split(',')] if args.sensitive_columns else None
                    qi_cols = [c.strip() for c in args.qi_columns.split(',')] if args.qi_columns else None
                    inference_metrics = model.evaluate_inference_attack(
                        data, synthetic,
                        sensitive_columns=sens_cols,
                        qi_columns=qi_cols,
                        verbose=args.verbose
                    )
                    all_metrics['inference_attack'] = inference_metrics
                
                # Downstream metrics (only if --all-metrics or explicitly requested)
                if args.all_metrics:
                    if args.verbose:
                        print("\nEvaluating downstream task performance...")
                    try:
                        downstream_metrics = model.evaluate_downstream(
                            data, synthetic,
                            target_column=args.target_column,
                            verbose=args.verbose
                        )
                        all_metrics['downstream'] = downstream_metrics
                    except Exception as e:
                        if args.verbose:
                            print(f"Warning: Downstream metrics failed: {e}", file=sys.stderr)
                        all_metrics['downstream'] = {'error': str(e), 'available': False}
                
                if args.metrics_file:
                    all_metrics['privacy_budget'] = report
                    with open(args.metrics_file, 'w') as f:
                        json.dump(all_metrics, f, indent=2, default=str)
                    print(f"\nMetrics saved to {args.metrics_file}")
        
        # DP audit (can be done without input file)
        if args.audit_dp:
            if args.verbose:
                print("\nAuditing DP compliance...")
            audit_results = model.audit_dp_compliance(verbose=args.verbose)
            all_metrics['dp_audit'] = audit_results
            if args.metrics_file:
                # Update metrics file with audit results
                try:
                    with open(args.metrics_file, 'r') as f:
                        existing_metrics = json.load(f)
                    existing_metrics['dp_audit'] = audit_results
                    with open(args.metrics_file, 'w') as f:
                        json.dump(existing_metrics, f, indent=2, default=str)
                except:
                    # If file doesn't exist or can't be read, write all metrics
                    all_metrics['dp_audit'] = audit_results
                    with open(args.metrics_file, 'w') as f:
                        json.dump(all_metrics, f, indent=2, default=str)
                print(f"\nMetrics saved to {args.metrics_file}")
    else:
        if args.verbose:
            print("No output specified. Use -o/--output to save synthetic data.")


if __name__ == '__main__':
    main()

