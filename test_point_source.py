import numpy as np
from solver import LaplaceSolver
from visualizer import plot_comparison_results
import os
import copy
import time

def main():
    # 1. 初始化
    NX, NY = 60, 60
    solver = LaplaceSolver(NX, NY)
    
    # 2. 设置边界条件：尖端放电模型
    # 接地平板 (底部)
    solver.set_boundary_line((0, 5), (60, 5), 0.0, thickness=2.0)
    
    # 高压尖端 (顶部垂下的针)
    # 针身
    solver.set_boundary_line((30, 59), (30, 35), 1000.0, thickness=1.0)
    # 针尖 (强化一点)
    solver.set_boundary_point(30, 34, 1000.0)
    
    # 3. 求解 (对比 Jacobi 和 SOR)
    print("正在求解尖端放电模型...")
    
    solver_jacobi = copy.deepcopy(solver)
    solver_sor = copy.deepcopy(solver)
    
    print("运行 Jacobi...")
    start = time.time()
    hist_jacobi = solver_jacobi.solve(method='jacobi', max_iter=3000, tol=1e-5)
    print(f"Jacobi 用时: {time.time() - start:.2f} 秒")
    print("运行 SOR...")
    start = time.time()
    hist_sor = solver_sor.solve(method='sor', max_iter=3000, tol=1e-5, omega=1.8)
    print(f"SOR 用时: {time.time() - start:.2f} 秒")
    # 4. 可视化并保存
    save_dir = "res"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plot_comparison_results(solver_jacobi, solver_sor, 
                            hist_jacobi, hist_sor,
                            title="Tip Discharge Model", 
                            save_path=os.path.join(save_dir, "point_source.png"))

if __name__ == "__main__":
    main()
