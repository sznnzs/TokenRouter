# plot_results.py
"""Plotting utilities for TokenRouter experimental results."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_dir):
    """Load results from directory"""
    results_file = os.path.join(results_dir, "results.json")
    analysis_file = os.path.join(results_dir, "results_analysis.json")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
    else:
        analysis = None
    
    return results, analysis

def plot_pareto_frontier(analysis, dataset_name, save_path):
    """Plot accuracy vs LLM token ratio Pareto frontier"""
    plt.figure(figsize=(10, 8))
    
    # Extract data for different method types
    methods = analysis[dataset_name]
    
    # Categorize methods
    router_methods = []
    baseline_methods = []
    pure_methods = []
    
    for method, stats in methods.items():
        point = {
            'name': method,
            'accuracy': stats['accuracy'],
            'llm_ratio': stats['llm_token_ratio']
        }
        
        if 'TokenRouter' in method:
            router_methods.append(point)
        elif method in ['LLM-Only', 'SLM-Only']:
            pure_methods.append(point)
        else:
            baseline_methods.append(point)
    
    # Plot different method types
    if router_methods:
        x = [m['llm_ratio'] * 100 for m in router_methods]
        y = [m['accuracy'] * 100 for m in router_methods]
        plt.scatter(x, y, s=200, marker='o', label='TokenRouter', zorder=5)
        
        # Connect TokenRouter points
        sorted_methods = sorted(router_methods, key=lambda m: m['llm_ratio'])
        x_sorted = [m['llm_ratio'] * 100 for m in sorted_methods]
        y_sorted = [m['accuracy'] * 100 for m in sorted_methods]
        plt.plot(x_sorted, y_sorted, 'b-', alpha=0.5, linewidth=2)
    
    if baseline_methods:
        x = [m['llm_ratio'] * 100 for m in baseline_methods]
        y = [m['accuracy'] * 100 for m in baseline_methods]
        plt.scatter(x, y, s=150, marker='^', label='Baselines', zorder=3)
    
    if pure_methods:
        for m in pure_methods:
            x = m['llm_ratio'] * 100
            y = m['accuracy'] * 100
            plt.scatter(x, y, s=300, marker='*', label=m['name'], zorder=4)
    
    plt.xlabel('LLM Token Ratio (%)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'{dataset_name}: Accuracy vs Computational Cost Trade-off', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for m in router_methods:
        if 'R30' in m['name'] or 'R20' in m['name'] or 'R50' in m['name']:
            plt.annotate(m['name'].split('-')[1], 
                        (m['llm_ratio'] * 100, m['accuracy'] * 100),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ablation_study(analysis, dataset_name, save_path):
    """Plot ablation study results"""
    ablation_methods = [
        'TokenRouter-Full', 'TokenRouter-NoKalman',
        'TokenRouter-NoAdaptive', 'TokenRouter-NoCommitment'
    ]
    
    if not all(m in analysis[dataset_name] for m in ablation_methods):
        print(f"Ablation methods not found for {dataset_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    methods = []
    accuracies = []
    llm_ratios = []
    
    for method in ablation_methods:
        stats = analysis[dataset_name][method]
        methods.append(method.replace('TokenRouter-', ''))
        accuracies.append(stats['accuracy'] * 100)
        llm_ratios.append(stats['llm_token_ratio'] * 100)
    
    # Accuracy comparison
    x = np.arange(len(methods))
    bars1 = ax1.bar(x, accuracies, color=['green', 'orange', 'blue', 'red'])
    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Ablation Study: Accuracy Impact', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}', ha='center', va='bottom')
    
    # LLM ratio comparison
    bars2 = ax2.bar(x, llm_ratios, color=['green', 'orange', 'blue', 'red'])
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('LLM Token Ratio (%)', fontsize=12)
    ax2.set_title('Ablation Study: Computational Cost', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    
    # Add value labels
    for bar, ratio in zip(bars2, llm_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ratio:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy_distribution(results, dataset_name, save_path):
    """Plot entropy distribution analysis"""
    # Collect all entropy values from TokenRouter runs
    all_entropies = []
    llm_entropies = []
    slm_entropies = []
    
    for method, data in results[dataset_name].items():
        if 'TokenRouter' in method and data['metrics']:
            for metric in data['metrics']:
                if 'entropy_history' in metric:
                    all_entropies.extend(metric['entropy_history'])
    
    if not all_entropies:
        print(f"No entropy data found for {dataset_name}")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(all_entropies, bins=50, alpha=0.7, density=True)
    plt.xlabel('Entropy', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Entropy Distribution', fontsize=14)
    
    # Box plot by percentiles
    plt.subplot(2, 2, 2)
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    perc_values = [np.percentile(all_entropies, p) for p in percentiles]
    plt.plot(percentiles, perc_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Percentile', fontsize=12)
    plt.ylabel('Entropy Value', fontsize=12)
    plt.title('Entropy Percentiles', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Time series example (first problem)
    plt.subplot(2, 1, 2)
    target_key = 'TokenRouter-R30' if results[dataset_name] and 'TokenRouter-R30' in results[dataset_name] else None
    if target_key:
        first_metric = results[dataset_name][target_key]['metrics'][0]
        if 'entropy_history' in first_metric:
            entropy_series = first_metric['entropy_history'][:100]  # First 100 tokens
            plt.plot(entropy_series, linewidth=2)
            plt.xlabel('Token Position', fontsize=12)
            plt.ylabel('Entropy', fontsize=12)
            plt.title('Example Entropy Time Series', fontsize=14)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_table(analysis, dataset_name, save_path):
    """Create a comparison table plot"""
    # Select key methods for comparison
    key_methods = ['SLM-Only', 'LLM-Only', 'Random-P50', 'FixedThreshold-T3.0',
                   'TokenRouter-R20', 'TokenRouter-R30', 'TokenRouter-R40']
    
    data = []
    for method in key_methods:
        if method in analysis[dataset_name]:
            stats = analysis[dataset_name][method]
            data.append({
                'Method': method,
                'Accuracy (%)': f"{stats['accuracy']*100:.1f}",
                'LLM Ratio (%)': f"{stats['llm_token_ratio']*100:.1f}",
                'Avg Tokens': f"{stats['avg_total_tokens']:.0f}",
                'Efficiency': f"{stats['accuracy']/(stats['llm_token_ratio']+0.01):.2f}"
            })
    
    if not data:
        print(f"Key methods not found for {dataset_name}")
        return
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(f'{dataset_name}: Method Comparison', fontsize=16, pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot TokenRouter results")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results, analysis = load_results(args.results_dir)
    
    if analysis is None:
        print("Analysis file not found. Please run experiments first.")
        return
    
    # Generate plots for each dataset
    for dataset in analysis.keys():
        print(f"\nGenerating plots for {dataset}...")
        
        # Pareto frontier
        plot_path = os.path.join(args.output_dir, f'{dataset}_pareto_frontier.png')
        plot_pareto_frontier(analysis, dataset, plot_path)
        print(f"  - Saved Pareto frontier plot to {plot_path}")
        
        # Ablation study
        plot_path = os.path.join(args.output_dir, f'{dataset}_ablation.png')
        plot_ablation_study(analysis, dataset, plot_path)
        print(f"  - Saved ablation study plot to {plot_path}")
        
        # Entropy distribution
        plot_path = os.path.join(args.output_dir, f'{dataset}_entropy_dist.png')
        plot_entropy_distribution(results, dataset, plot_path)
        print(f"  - Saved entropy distribution plot to {plot_path}")
        
        # Comparison table
        plot_path = os.path.join(args.output_dir, f'{dataset}_comparison_table.png')
        plot_comparison_table(analysis, dataset, plot_path)
        print(f"  - Saved comparison table to {plot_path}")
    
    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
