# 第四课：Python 并行实践 (Hands-On Optimization)

恭喜你答对了！**通信时间 (Communication Overhead)** 确实属于串行部分（或者说是无法完美并行的开销）。如果通信太慢，根据 Amdahl 定律，它将成为整个系统的性能瓶颈。这就是为什么超级计算机需要极其昂贵的高速互连网络（如 InfiniBand）。

现在，让我们离开理论，进入最激动人心的环节——**让你的代码飞起来**。

我们将使用 **Numba** 库。它是一个 JIT (Just-In-Time) 编译器，可以将你的 Python 函数在运行时编译成优化的机器码（就像 C++ 或 Fortran 一样快），而且支持自动并行化。

---

## 1. 准备工作：建立基准 (Baseline)

在优化之前，我们必须知道现在有多慢。
我们需要测量 `solver.py` 在当前状态下的运行时间。

请在终端运行以下命令（确保安装了 `tqdm` 用于显示进度，如果没有可以忽略）：

```bash
# 运行现有的测试脚本，观察大概速度
python test_point_source.py
```

记下大概的感觉，或者你可以修改 `test_point_source.py`，在 `solver.solve()` 前后加上计时代码：

```python
import time
start = time.time()
solver.solve(...)
print(f"耗时: {time.time() - start:.4f} 秒")
```

---

## 2. 引入 Numba：从解释器到编译器

Python 慢是因为它是**解释型**语言，而且有 **GIL (全局解释器锁)**，限制了多线程。
Numba 可以绕过这些限制。

### 安装 Numba

你需要先安装它：
```bash
pip install numba
```

### 核心魔法：`@jit`
我们只需要在函数前面加一个装饰器 `@jit(nopython=True)`。

但是，Numba 最擅长的是**显式的循环**，而不是 NumPy 的向量化操作（因为向量化操作会产生大量的临时数组，消耗内存带宽）。
还记得我们在第二课讲的“内存墙”吗？**显式循环可以让我们在一个循环内做完所有事，不用反复读写内存。**

---

## 3. 实战：编写 `solver_numba.py`

我们将创建一个新的求解器文件，利用 Numba 的并行功能。

请在 `electrostatic_simulation` 目录下新建文件 `solver_numba.py`，并填入以下代码。
注意观察 `update_red_black` 函数，我们使用了 `prange` (parallel range) 来替代普通的 `range`，这告诉 Numba：“这里的循环可以分配给多个核心同时跑！”

```python
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def update_red_black(V, boundary_mask, omega, is_black):
    """
    使用 Numba 加速的红黑更新内核
    parallel=True 开启自动并行化
    """
    ny, nx = V.shape
    
    # prange 会自动将循环分配给多个 CPU 核心
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            # 判断红黑: (i + j) % 2 == 0 是红，== 1 是黑
            if (i + j) % 2 == is_black:
                if not boundary_mask[i, j]:
                    # SOR 更新公式
                    v_new = 0.25 * (V[i-1, j] + V[i+1, j] + V[i, j-1] + V[i, j+1])
                    V[i, j] = V[i, j] + omega * (v_new - V[i, j])
    
    return V

@jit(nopython=True)
def compute_error(V, V_old):
    """计算最大误差"""
    return np.max(np.abs(V - V_old))

class LaplaceSolverNumba:
    def __init__(self, size_x, size_y, dx=1.0):
        self.nx = size_x
        self.ny = size_y
        self.V = np.zeros((self.ny, self.nx))
        self.boundary_mask = np.zeros((self.ny, self.nx), dtype=bool)

    def set_boundary_condition(self, x_range, y_range, voltage):
        # ... (与原版相同，省略以节省篇幅，你可以直接复制原版的逻辑)
        x0, x1 = x_range
        y0, y1 = y_range
        self.V[y0:y1, x0:x1] = voltage
        self.boundary_mask[y0:y1, x0:x1] = True
        
    def set_boundary_point(self, x, y, voltage):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.V[iy, ix] = voltage
            self.boundary_mask[iy, ix] = True

    def solve(self, max_iter=10000, tol=1e-4, omega=1.5):
        print(f"开始 Numba 加速求解: 网格={self.nx}x{self.ny}")
        
        for i in range(max_iter):
            V_old = self.V.copy()
            
            # 1. 更新红点 (is_black=0)
            self.V = update_red_black(self.V, self.boundary_mask, omega, 0)
            
            # 2. 更新黑点 (is_black=1)
            self.V = update_red_black(self.V, self.boundary_mask, omega, 1)
            
            # 每 100 次检查一次收敛，减少计算误差的开销
            if i % 100 == 0:
                err = compute_error(self.V, V_old)
                if err < tol:
                    print(f"收敛于迭代 {i}, 误差 {err}")
                    break
        return []
```

---

## 4. 任务：对比测试

1.  **安装 Numba**。
2.  **创建 `solver_numba.py`** (你可以参考上面的代码，或者直接让我帮你创建)。
3.  **修改测试脚本**：创建一个 `test_benchmark.py`，同时运行原版 `solver` 和 `solver_numba`，对比它们的时间。

你会发现，随着网格变大（比如 $500 \times 500$），Numba 版本可能会比原版快 **10 倍甚至 100 倍**！这就是并行计算和编译优化的力量。
