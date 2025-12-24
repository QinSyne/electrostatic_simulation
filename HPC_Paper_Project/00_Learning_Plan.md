# 高性能计算与并行架构学习路径 (HPC Learning Path)

欢迎来到高性能计算（HPC）的世界！这个文件夹将作为你的“科研笔记本”，记录从零基础到完成学术论文的全过程。

我们将以你现有的 `electrostatic_simulation` 项目为基石，一步步探索如何让它跑得更快、规模更大。

## 阶段一：基础概念与数学原理 (已完成)
**目标**：理解“为什么计算会慢”以及“并行计算的理论极限”。
- [x] **01_Basics_and_Math.md**: 
    - 串行 vs 并行
    - 核心数学推导：阿姆达尔定律 (Amdahl's Law)
    - 复杂度分析：你的 `solver.py` 到底有多慢？

## 阶段二：计算机体系结构初探 (已完成)
**目标**：理解硬件是如何支持计算的。
- [x] **02_Architecture_DeepDive.md**:
    - CPU 的构造：核心 (Core)、缓存 (Cache) 与 向量化 (SIMD)
    - GPU 的构造：为什么它适合做科学计算？
    - 内存墙 (Memory Wall) 问题

## 阶段三：并行算法设计 (已完成)
**目标**：学习如何改造算法以适应并行环境。
- [x] **03_Parallel_Algorithms.md**:
    - 数据依赖性分析 (Data Dependency)
    - 你的代码中的“红黑排序 (Red-Black Ordering)”是什么原理？
    - 区域分解法 (Domain Decomposition)

## 阶段四：Python 并行实践 (已完成)
**目标**：动手修改代码，体验加速。
- [x] **04_Hands_On_Optimization.md**:
    - 使用 `Numba` 进行 JIT 编译加速
    - (进阶) 简单的 GPU 加速体验

## 阶段五：学术写作 (当前阶段)
**目标**：将探索过程转化为论文。
- [x] **05_Paper_Outline.md**: 拟定论文大纲
- [ ] **06_Drafting.md**: 撰写正文

---
**学习建议**：
1. 不要急于求成，先理解数学原理。
2. 随时回顾你的 `solver.py`，思考每一行代码背后的计算代价。
