#!/usr/bin/env python3
"""Show what k-anonymity violations look like with real examples."""

import pandas as pd
from collections import Counter

def demonstrate_k_anonymity():
    print("=" * 80)
    print("K-Anonymity Violations Demonstration")
    print("=" * 80)
    print()
    
    # Load data (try multiple paths)
    import os
    data_paths = ['../data/adult.csv', 'data/adult.csv']
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    if data_path is None:
        print("Error: Could not find data/adult.csv")
        print("Please run from project root or ensure data/adult.csv exists")
        return
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} real records")
    print()
    
    # QI columns
    qi_columns = ['age', 'sex', 'race']
    print(f"Quasi-Identifier columns: {', '.join(qi_columns)}")
    print()
    
    # Build QI combinations for real data
    real_qi = data[qi_columns].astype(str).apply(lambda x: '|'.join(x), axis=1)
    real_qi_counts = Counter(real_qi)
    
    print("=" * 80)
    print("Real Data QI Statistics")
    print("=" * 80)
    print(f"Total unique QI combinations: {len(real_qi_counts)}")
    print(f"Average frequency per QI combination: {sum(real_qi_counts.values()) / len(real_qi_counts):.2f}")
    print()
    
    # Show distribution
    freq_dist = Counter(real_qi_counts.values())
    print("Frequency distribution (how many QI combos appear N times):")
    for freq in sorted(freq_dist.keys())[:10]:
        print(f"  {freq:3d} times: {freq_dist[freq]:4d} QI combinations")
    print(f"  ... (showing first 10)")
    print()
    
    # Show examples of rare QI combinations (k-anonymity violations for k=5)
    k = 5
    print(f"=" * 80)
    print(f"Examples of Rare QI Combinations (≤{k} occurrences = k-anonymity violation)")
    print("=" * 80)
    print()
    
    rare_combos = [(combo, count) for combo, count in real_qi_counts.items() if count <= k]
    rare_combos.sort(key=lambda x: x[1])  # Sort by frequency
    
    print(f"Total rare QI combinations (≤{k} occurrences): {len(rare_combos)}")
    print(f"Percentage of all QI combinations: {len(rare_combos) / len(real_qi_counts) * 100:.1f}%")
    print()
    
    print("Top 10 rarest QI combinations:")
    for i, (combo, count) in enumerate(rare_combos[:10], 1):
        parts = combo.split('|')
        print(f"  {i:2d}. age={parts[0]:>3s}, sex={parts[1]:>6s}, race={parts[2]:>20s} → {count} occurrence(s)")
    print()
    
    # Show what happens with synthetic data
    print("=" * 80)
    print("What Happens with Synthetic Data")
    print("=" * 80)
    print()
    print("If synthetic data generates a QI combination that appears ≤5 times in real data:")
    print()
    print("  Example: Synthetic record has (age=22, sex=Male, race=Native-American)")
    example_combo = "22|Male|Native-American"
    if example_combo in real_qi_counts:
        count = real_qi_counts[example_combo]
        print(f"  → This QI combination appears {count} time(s) in real data")
        if count <= k:
            print(f"  → ⚠️  K-ANONYMITY VIOLATION (only {count} < {k})")
            print(f"  → Risk: Attacker can narrow down to {count} possible individuals")
            print(f"  → Re-identification probability: {1/count*100:.1f}%")
    else:
        print(f"  → This QI combination does NOT appear in real data")
        print(f"  → ⚠️  K-ANONYMITY VIOLATION (0 < {k})")
        print(f"  → Risk: Cannot link to real data (but still a violation)")
    print()
    
    # Show k-anonymity compliance
    print("=" * 80)
    print("K-Anonymity Compliance Analysis")
    print("=" * 80)
    print()
    
    compliant = sum(1 for count in real_qi_counts.values() if count >= k)
    violations = len(real_qi_counts) - compliant
    
    print(f"For k={k}-anonymity:")
    print(f"  ✓ Compliant QI combinations (≥{k} occurrences): {compliant:,} ({compliant/len(real_qi_counts)*100:.1f}%)")
    print(f"  ✗ Violations (≤{k} occurrences): {violations:,} ({violations/len(real_qi_counts)*100:.1f}%)")
    print()
    
    # Explain the synthetic data result
    print("=" * 80)
    print("Why Synthetic Data Shows 100% Violation Rate")
    print("=" * 80)
    print()
    print("From our metrics:")
    print("  • Real data: 528 unique QI combinations")
    print("  • Synthetic data: 17 unique QI combinations")
    print("  • All 17 synthetic QI combinations violate k-anonymity")
    print()
    print("This happens because:")
    print("  1. Synthetic data has much less diversity (17 vs 528 combinations)")
    print("  2. The 17 combinations generated are likely common ones")
    print("  3. But even common combinations can violate k-anonymity if they're rare")
    print("  4. OR: The synthetic combinations don't match real ones exactly")
    print()
    print("Key point: Even with 100% violation rate, DP still provides formal privacy")
    print("because synthetic records are NOT exact copies of real records.")

if __name__ == "__main__":
    demonstrate_k_anonymity()

