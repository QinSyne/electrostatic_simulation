import sys
import os
import time
import numpy as np

# 添加父目录到路径，以便导入原来的 solver.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import LaplaceSolver
from solver_numba import LaplaceSolverNumba

def run_benchmark():
    # 设置较大的网格以体现 Numba 优势
    SIZE = 500
    MAX_ITER = 10000
    
    print(f"=== 基准测试: 网格大小 {SIZE}x{SIZE} ===")
    
    # 1. 测试原版 Solver (SOR)
    print("\n正在运行原版 Solver (Python/NumPy)...")
    solver_orig = LaplaceSolver(SIZE, SIZE)
    solver_orig.set_boundary_condition((SIZE//4, 3*SIZE//4), (SIZE//4, 3*SIZE//4), 10.0)
    
    start_time = time.time()
    solver_orig.solve(method='sor', max_iter=MAX_ITER, tol=1e-4)
    orig_time = time.time() - start_time
    print(f"原版耗时: {orig_time:.4f} 秒")
    
    # 2. 测试 Numba Solver
    print("\n正在运行 Numba Solver (JIT Compiled)...")
    # 预热 (JIT 编译需要时间，第一次运行会包含编译时间)
    print("预热中 (编译 JIT 代码)...")
    warmup = LaplaceSolverNumba(50, 50)
    warmup.solve(max_iter=10)
    
    solver_numba = LaplaceSolverNumba(SIZE, SIZE)
    solver_numba.set_boundary_condition((SIZE//4, 3*SIZE//4), (SIZE//4, 3*SIZE//4), 10.0)
    
    start_time = time.time()
    solver_numba.solve(max_iter=MAX_ITER, tol=1e-4)
    numba_time = time.time() - start_time
    print(f"Numba 耗时: {numba_time:.4f} 秒")
    
    # 3. 结果对比
    speedup = orig_time / numba_time
    print(f"\n=== 结果 ===")
    print(f"加速比: {speedup:.2f}x")
    if speedup > 10:
        print("🚀 这是一个巨大的提升！")
    else:
        print("提升显著，但还有优化空间。")

if __name__ == "__main__":
    run_benchmark()
