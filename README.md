# 静电场数值模拟 (Electrostatic Field Simulation)
完成于2025.12.24
## 项目简介
本项目是一个基于 Python 的静电场数值模拟工具，专为计算机科学与物理学交叉学习设计。它利用**有限差分法 (Finite Difference Method, FDM)** 求解二维空间中的拉普拉斯方程 ($\nabla^2 V = 0$)，从而模拟不同电极形状下的电势分布和电场分布。

项目重点在于对比两种数值迭代算法的性能：
1.  **Jacobi 迭代法**：基础算法，便于理解并行计算原理。
2.  **SOR (Successive Over-Relaxation) 迭代法**：进阶算法，展示了数值分析中通过数学技巧加速收敛的威力。

## 功能特性
- **多种边界条件**：支持矩形、任意角度直线、点电荷、圆形导体等形状的电极设置。
- **算法性能对比**：自动运行 Jacobi 和 SOR 算法，并生成收敛速度对比图（迭代次数 vs 误差）。
- **可视化输出**：生成包含电势热力图、电场流线图、收敛曲线和性能柱状图的综合报表。

## 文件结构
```
electrostatic_simulation/
├── solver.py           # 核心求解器 (LaplaceSolver 类)
├── visualizer.py       # 可视化模块 (绘图与保存)
├── test_parallel_plates.py   # 测试案例：平行板电容器
├── test_angled_plates.py     # 测试案例：倾斜板电容器
├── test_point_source.py      # 测试案例：尖端放电
├── test_circular_conductor.py # 测试案例：圆形导体
└── res/                # 结果输出目录 (存放生成的图片)
```

## 快速开始
### 1. 环境准备
确保安装了 Python 3 以及以下依赖库：
```bash
pip install numpy matplotlib
```
### 2. 运行模拟
可以运行以下任意一个测试脚本，程序会自动在 `res/` 目录下生成对应的分析图片。
*   **平行板电容器模型**：
    ```bash
    python test_parallel_plates.py
    ```
*   **倾斜板电容器模型**：
    ```bash
    python test_angled_plates.py
    ```
*   **尖端放电模型**：
    ```bash
    python test_point_source.py
    ```
*   **圆形导体模型**：
    ```bash
    python test_circular_conductor.py
    ```

### 3. 查看结果
运行完成后，打开 `res/` 文件夹，你将看到如下图片：
- `parallel_plates.png`
- `angled_plates.png`
- `point_source.png`
- `circular_conductor.png`

每张图片包含 6 个子图：
- **第一行**：Jacobi 算法计算得到的电势分布和电场线。
- **第二行**：SOR 算法计算得到的电势分布和电场线（两者应在物理上一致）。
- **第三行**：
    - **左图**：收敛曲线对比（对数坐标）。你会看到 SOR 的误差下降速度远快于 Jacobi。
    - **右图**：性能柱状图。直观展示迭代次数的巨大差异（通常 SOR 快 10 倍以上）。

## 核心算法原理
### 离散化 (Discretization)
将连续的拉普拉斯方程 $\frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} = 0$ 在网格上离散化，得到核心更新公式：
$$ V_{i,j} = \frac{1}{4} (V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1}) $$
即某点的电势等于其周围四个邻居的平均值。

### 迭代求解
- **Jacobi**：$V_{new} = \text{Average}(V_{old})$。利用矩阵切片实现并行计算。
- **SOR**：$V_{new} = (1-\omega)V_{old} + \omega \times \text{Average}(V_{new/old})$。引入松弛因子 $\omega$ (本项目取 1.8) 来“预测”变化趋势，从而大幅加速收敛。


