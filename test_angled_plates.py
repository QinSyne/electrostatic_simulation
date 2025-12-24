import numpy as np
from solver import LaplaceSolver
from visualizer import plot_comparison_results
import os
import copy

def main():
    # 1. 初始化
    NX, NY = 80, 80
    solver = LaplaceSolver(NX, NY)
    
    # 2. 设置边界条件：倾斜板电容器 (扇形边缘)

    solver.set_boundary_line((30, 10), (60, 30), 100.0, thickness=2.0)
    solver.set_boundary_line((10, 30), (30, 60), -100.0, thickness=2.0)
    
    # 3. 求解 (对比 Jacobi 和 SOR)
    print("正在求解倾斜板电容器模型...")
    
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
                            title="Angled Plates Capacitor", 
                            save_path=os.path.join(save_dir, "angled_plates.png"))

if __name__ == "__main__":
    main()
