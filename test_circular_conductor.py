import numpy as np
from solver import LaplaceSolver
from visualizer import plot_comparison_results
import os
import copy

def main():
    # 1. 初始化
    NX, NY = 60, 60
    solver = LaplaceSolver(NX, NY)
    
    # 2. 设置边界条件：圆形导体
    # 中心带电圆柱 (正极)
    solver.set_boundary_circle((30, 30), 8.0, 100.0)
    
    # 左下圆柱 (负极)
    solver.set_boundary_circle((15, 15), 5.0, -100.0)
    
    # 右上圆柱 (负极)
    solver.set_boundary_circle((45, 45), 5.0, -100.0)
    
    # 3. 求解 (对比 Jacobi 和 SOR)
    print("正在求解圆形导体模型...")
    
    solver_jacobi = copy.deepcopy(solver)
    solver_sor = copy.deepcopy(solver)
    
    print("运行 Jacobi...")
    hist_jacobi = solver_jacobi.solve(method='jacobi', max_iter=3000, tol=1e-5)
    
    print("运行 SOR...")
    hist_sor = solver_sor.solve(method='sor', max_iter=3000, tol=1e-5, omega=1.8)
    
    # 4. 可视化并保存
    save_dir = "res"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plot_comparison_results(solver_jacobi, solver_sor, 
                            hist_jacobi, hist_sor,
                            title="Circular Conductors", 
                            save_path=os.path.join(save_dir, "circular_conductor.png"))

if __name__ == "__main__":
    main()
