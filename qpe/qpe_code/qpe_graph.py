# qpe_graph.py

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def plot(counts, best_k, all_k, phi_est, n, shots, output_dir):
    """
    Plot QPE results histogram with counts instead of probabilities.
    
    Args:
        counts (dict): Dictionary {bitstring: counts}
        best_k (int): Index of the most probable outcome
        all_k (list): List of all possible k values
        phi_est (float): Estimated phase value
        n (int): Number of phase qubits
        shots (int): Total number of measurements
        output_dir (str): Directory to save the plot
    """
    # Create figure with dynamic width based on number of outcomes
    fig_width = max(10, 2**n * 0.6)  # Ensure minimum width for small n
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    # Color scheme - highlight the most probable result
    colors = ['#1f77b4'] * len(counts)  # Matplotlib blue
    colors[best_k] = '#d62728'  # Matplotlib red
    
    # Create bar plot
    bars = ax.bar(all_k, counts, 
                  color=colors, 
                  width=0.85, 
                  edgecolor='white', 
                  linewidth=1.0, 
                  zorder=2,
                  alpha=0.8)
    
    # Add count labels on top of bars
    for k, count in enumerate(counts):
        if count > 0:
            ax.text(k, count + shots * 0.01, f"{count}",
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='black')
    
    # Configure x-axis with phase labels
    labels = [f"{k}/{2**n}\n({k/2**n:.3f})" for k in all_k]
    ax.set_xticks(all_k)
    ax.set_xticklabels(labels, rotation=45, fontsize=min(10, 120/2**n))
    ax.set_xlabel("Phase Estimate (k / 2ⁿ)", fontsize=12, fontweight='bold')
    
    # Configure y-axis
    ax.set_ylabel("Counts", fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, which='major', color='lightgray', 
                  linestyle='-', linewidth=0.7, zorder=1)
    
    # Set title
    ax.set_title(f"Quantum Phase Estimation Results (n={n} qubits)", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Remove top and right spines for cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    
    # Set background color
    ax.set_facecolor('whitesmoke')
    fig.patch.set_facecolor('white')
    
    # Adjust y-limit to accommodate labels
    ax.set_ylim(0, max(counts) * 1.25)
    
    # Create information box
    info_text = f"""Estimation Results:
• Estimated phase: φ = {1 - phi_est:.6f}
• Most probable: k = {best_k}
• Total shots: {shots}
• Precision: 1/2ⁿ = {1/2**n:.6f}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightsteelblue", 
                     alpha=0.8, edgecolor='navy'))
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.8, label='All outcomes'),
        Patch(facecolor='#d62728', alpha=0.8, label=f'Most probable: k={best_k}')
    ]
    
    ax.legend(handles=legend_elements, 
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2,
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=11)
    
    # Adjust layout to accommodate legend and labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save high-quality plot
    plt.savefig(output_dir, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Plot saved: {output_dir}")
    print('=' * 60)

    return phi_est