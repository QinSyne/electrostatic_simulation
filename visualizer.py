import numpy as np
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，避免弹出窗口
import matplotlib.pyplot as plt

def plot_results(solver, title="Electrostatic Simulation", save_path=None):
    """
    可视化电势分布和电场分布 (单算法)
    """
    V = solver.V
    Ex, Ey = solver.calculate_field()
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    im = ax[0].imshow(V, cmap='inferno', origin='lower', extent=[0, solver.nx, 0, solver.ny])
    ax[0].set_title(f"{title} - Potential (V)")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.colorbar(im, ax=ax[0], label='Potential (V)')
    
    x = np.arange(solver.nx)
    y = np.arange(solver.ny)
    X, Y = np.meshgrid(x, y)
    E_mag = np.sqrt(Ex**2 + Ey**2)
    
    st = ax[1].streamplot(X, Y, Ex, Ey, color=E_mag, cmap='autumn', density=1.5, linewidth=1)
    ax[1].set_title(f"{title} - Field Lines")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_xlim(0, solver.nx)
    ax[1].set_ylim(0, solver.ny)
    fig.colorbar(st.lines, ax=ax[1], label='|E|')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_comparison_results(solver_jacobi, solver_sor, hist_jacobi, hist_sor, title="Comparison", save_path=None):
    """
    绘制 6 张图对比 Jacobi 和 SOR 算法
    Row 1: Jacobi Potential, Jacobi Field
    Row 2: SOR Potential, SOR Field
    Row 3: Convergence Line Plot, Performance Bar Chart
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"{title} - Jacobi vs SOR", fontsize=16)
    
    # Helper to plot potential and field
    def plot_row(ax_row, solver, algo_name):
        V = solver.V
        Ex, Ey = solver.calculate_field()
        
        # Potential
        im = ax_row[0].imshow(V, cmap='inferno', origin='lower', extent=[0, solver.nx, 0, solver.ny])
        ax_row[0].set_title(f"{algo_name} - Potential")
        ax_row[0].set_xlabel("x")
        ax_row[0].set_ylabel("y")
        fig.colorbar(im, ax=ax_row[0], label='V')
        
        # Field
        x = np.arange(solver.nx)
        y = np.arange(solver.ny)
        X, Y = np.meshgrid(x, y)
        E_mag = np.sqrt(Ex**2 + Ey**2)
        
        st = ax_row[1].streamplot(X, Y, Ex, Ey, color=E_mag, cmap='autumn', density=1.5, linewidth=1)
        ax_row[1].set_title(f"{algo_name} - Field Lines")
        ax_row[1].set_xlabel("x")
        ax_row[1].set_ylabel("y")
        ax_row[1].set_xlim(0, solver.nx)
        ax_row[1].set_ylim(0, solver.ny)
        fig.colorbar(st.lines, ax=ax_row[1], label='|E|')

    # Row 1: Jacobi
    plot_row(axs[0], solver_jacobi, "Jacobi")
    
    # Row 2: SOR
    plot_row(axs[1], solver_sor, "SOR")
    
    # Row 3, Col 1: Convergence Line Plot
    ax_conv = axs[2, 0]
    ax_conv.plot(hist_jacobi, label='Jacobi', linewidth=2)
    ax_conv.plot(hist_sor, label='SOR', linewidth=2)
    ax_conv.set_yscale('log')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Error (Log Scale)')
    ax_conv.set_title('Convergence Speed Comparison')
    ax_conv.grid(True, which="both", ls="-", alpha=0.5)
    ax_conv.legend()
    
    # Row 3, Col 2: Bar Chart (Iterations & Final Error)
    ax_bar = axs[2, 1]
    labels = ['Jacobi', 'SOR']
    iters = [len(hist_jacobi), len(hist_sor)]
    final_errors = [hist_jacobi[-1], hist_sor[-1]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot Iterations (Left Axis)
    ax_bar.set_title('Performance Metrics')
    bar1 = ax_bar.bar(x - width/2, iters, width, label='Iterations', color='skyblue')
    ax_bar.set_ylabel('Iterations', color='skyblue')
    ax_bar.tick_params(axis='y', labelcolor='skyblue')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    
    # Plot Final Error (Right Axis)
    ax_bar2 = ax_bar.twinx()
    bar2 = ax_bar2.bar(x + width/2, final_errors, width, label='Final Error', color='salmon')
    ax_bar2.set_ylabel('Final Error', color='salmon')
    ax_bar2.tick_params(axis='y', labelcolor='salmon')
    ax_bar2.set_yscale('log') # Error is usually small
    
    # Legend
    # Combine legends
    lines1, labels1 = ax_bar.get_legend_handles_labels()
    lines2, labels2 = ax_bar2.get_legend_handles_labels()
    ax_bar.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存至: {save_path}")
        plt.close()
    else:
        plt.show()

    plt.xlabel('Iteration')
    plt.ylabel('Error (Log Scale)')
    plt.title('Convergence Analysis: Jacobi vs SOR')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()
