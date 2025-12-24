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
        x0, x1 = x_range
        y0, y1 = y_range
        x0, x1 = max(0, x0), min(self.nx, x1)
        y0, y1 = max(0, y0), min(self.ny, y1)
        self.V[y0:y1, x0:x1] = voltage
        self.boundary_mask[y0:y1, x0:x1] = True

    def set_boundary_point(self, x, y, voltage):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.V[iy, ix] = voltage
            self.boundary_mask[iy, ix] = True

    def set_boundary_line(self, start, end, voltage, thickness=1.0):
        """
        设置线形边界 (支持任意角度的斜线)
        :param start: (x, y) 起点坐标
        :param end: (x, y) 终点坐标
        :param voltage: 电势值
        :param thickness: 线条宽度 (默认为1.0)
        """
        x0, y0 = start
        x1, y1 = end
        
        # 创建网格坐标矩阵 (向量化计算，避免循环)
        Y, X = np.ogrid[:self.ny, :self.nx]
        
        # 计算每个网格点到线段的距离
        dx = x1 - x0
        dy = y1 - y0
        length_sq = dx**2 + dy**2
        
        if length_sq == 0:
            # 起点终点重合，退化为点
            dist = np.sqrt((X - x0)**2 + (Y - y0)**2)
        else:
            # 计算投影参数 t = ((P-A) . (B-A)) / |B-A|^2
            # P是网格点(X,Y), A是起点(x0,y0), B是终点
            t = ((X - x0) * dx + (Y - y0) * dy) / length_sq
            
            # 限制 t 在 [0, 1] 之间 (限制在线段范围内，而不是无限长直线)
            t = np.clip(t, 0, 1)
            
            # 找到线段上距离网格点最近的点
            nearest_x = x0 + t * dx
            nearest_y = y0 + t * dy
            
            # 计算欧几里得距离
            dist = np.sqrt((X - nearest_x)**2 + (Y - nearest_y)**2)
            
        # 找到距离小于等于半宽度的点，标记为边界
        mask = dist <= (thickness / 2.0)
        
        self.V[mask] = voltage
        self.boundary_mask[mask] = True

    def calculate_field(self):
        """
        根据电势 V 计算电场 E = -grad(V)
        :return: Ex, Ey 矩阵
        """
        # 使用 numpy 的 gradient 函数计算梯度
        # 注意：E = -dV/dx, -dV/dy
        # np.gradient 返回 (d/dy, d/dx)
        Gy, Gx = np.gradient(self.V, 1.0) # dx=1.0
        Ex = -Gx
        Ey = -Gy
        return Ex, Ey

    def solve(self, max_iter=10000, tol=1e-4, omega=1.5):
        # print(f"开始 Numba 加速求解: 网格={self.nx}x{self.ny}")
        
        history = []
        for i in range(max_iter):
            V_old = self.V.copy()
            
            # 1. 更新红点 (is_black=0)
            self.V = update_red_black(self.V, self.boundary_mask, omega, 0)
            
            # 2. 更新黑点 (is_black=1)
            self.V = update_red_black(self.V, self.boundary_mask, omega, 1)
            
            # 每 100 次检查一次收敛，减少计算误差的开销
            if i % 100 == 0:
                err = compute_error(self.V, V_old)
                history.append(err)
                if err < tol:
                    # print(f"收敛于迭代 {i}, 误差 {err}")
                    break
        return history
