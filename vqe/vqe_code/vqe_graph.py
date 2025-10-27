# vqe_graph

import os
import matplotlib.pyplot as plt

def plot(result, elapsed, energies, min_energy, molecule_name, output_dir, best_rep, save_plot=True):
    """
    Plot VQE energy convergence.

    Args:
        result (float): Final VQE energy.
        elapsed (float): Time elapsed during VQE.
        energies (list): List of energies during optimization.
        min_energy (float): Analytical minimum energy.
        molecule_name (str): Molecule name.
        output_dir (str): Directory to save plot.
        best_rep (int): Iteration of the best energy.
        save_plot (bool): Whether to save plot to file.
    """
    plt.figure(figsize=(12, 8))
    iterations = range(len(energies))
    plt.plot(iterations, energies, 'b-', linewidth=2, label='VQE Energy', alpha=0.8)

    # Plot analytical and final energies
    plt.axhline(y=min_energy, color='red', linestyle='--', linewidth=2, label=f'Analytical Minimum: {min_energy:.6f}')
    plt.axhline(y=result, color='blue', linestyle='--', linewidth=2, label=f'Final VQE: {result:.6f}')

    # Highlight start, end, and best iteration
    plt.scatter([0], [energies[0]], color='blue', s=100, label=f'Start: {energies[0]:.6f}')
    plt.scatter([len(energies)-1], [energies[-1]], color='blue', s=100, label=f'End: {energies[-1]:.6f}')
    plt.scatter([best_rep], [result], color='red', s=100, label=f'Best Iteration: {result:.6f}')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title(f'VQE Convergence for {molecule_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Display stats on plot
    error = abs(result - min_energy)
    error_percent = (error / abs(min_energy)) * 100
    stats_text = f'''Statistics:
• Iterations: {len(energies)}
• Best: {result:.6f} Hartree = {(result * 27.211386245988):.6f} eV
• Absolute Error: {error:.6f} Hartree = {(error * 27.211386245988):.6f} eV
• Error %: {error_percent:.3f}%
• Time elapsed: {elapsed:.6f} s'''
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_plot:
        plt.savefig(os.path.join(output_dir, "vqe_convergence.png"), dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {os.path.join(output_dir, 'vqe_convergence.png')}\n" + '='*50)

    plt.show()