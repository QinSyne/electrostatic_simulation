import sys
import os
import numpy as np
import copy
import time
import matplotlib.pyplot as plt

# Add parent directory to path to import from electrostatic_simulation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver import LaplaceSolver
from visualizer import plot_results
from HPC_Paper_Project.solver_numba import LaplaceSolverNumba

def main():
    # 1. Initialize
    NX, NY = 100, 100  # Slightly larger for better resolution in paper
    
    # Setup base configuration
    # We will use the same setup logic as test_point_source.py but applied to both solvers
    
    # --- Serial Solver Setup ---
    solver_serial = LaplaceSolver(NX, NY)
    # Ground plate (bottom)
    solver_serial.set_boundary_line((0, 5), (NX, 5), 0.0, thickness=2.0)
    # High voltage tip (needle from top)
    solver_serial.set_boundary_line((NX//2, NY-1), (NX//2, NY//2), 1000.0, thickness=1.0)
    solver_serial.set_boundary_point(NX//2, NY//2 - 1, 1000.0)
    
    # --- Parallel Solver Setup ---
    solver_parallel = LaplaceSolverNumba(NX, NY)
    # Ground plate (bottom)
    solver_parallel.set_boundary_line((0, 5), (NX, 5), 0.0, thickness=2.0)
    # High voltage tip (needle from top)
    solver_parallel.set_boundary_line((NX//2, NY-1), (NX//2, NY//2), 1000.0, thickness=1.0)
    solver_parallel.set_boundary_point(NX//2, NY//2 - 1, 1000.0)
    
    # 2. Solve
    print("Running Serial Solver (NumPy)...")
    start_time = time.time()
    solver_serial.solve(method='sor', max_iter=5000, tol=1e-5, omega=1.8)
    print(f"Serial Solver Time: {time.time() - start_time:.4f}s")
    
    print("Running Parallel Solver (Numba)...")
    start_time = time.time()
    solver_parallel.solve(max_iter=5000, tol=1e-5, omega=1.8)
    print(f"Parallel Solver Time: {time.time() - start_time:.4f}s")
    
    # 3. Verify Correctness
    diff = np.abs(solver_serial.V - solver_parallel.V)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"\n--- Verification ---")
    print(f"Max Difference: {max_diff:.6e}")
    print(f"Mean Difference: {mean_diff:.6e}")
    
    if max_diff < 1e-3:
        print("SUCCESS: Results are identical (within tolerance).")
    else:
        print("WARNING: Results differ significantly.")

    # 4. Generate Plots for Paper
    save_dir = os.path.join(os.path.dirname(__file__), 'res_paper')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Plot Serial
    plot_results(solver_serial, title="Serial Solver (NumPy)", 
                 save_path=os.path.join(save_dir, "plot_serial.png"))
                 
    # Plot Parallel
    plot_results(solver_parallel, title="Parallel Solver (Numba)", 
                 save_path=os.path.join(save_dir, "plot_parallel.png"))
                 
    # Plot Difference Heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap='viridis', origin='lower')
    plt.colorbar(label='Absolute Difference (V)')
    plt.title(f"Difference Map (Max Diff: {max_diff:.2e})")
    plt.savefig(os.path.join(save_dir, "plot_difference.png"))
    plt.close()
    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    main()
