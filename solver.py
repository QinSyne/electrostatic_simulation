import numpy as np
class LaplaceSolver:
    def __init__(self, size_x, size_y, dx=1.0):
        """
        初始化求解器
        :param size_x: 网格 x 方向大小
        :param size_y: 网格 y 方向大小
        :param dx: 网格间距 (物理距离)
        """
        self.nx = size_x
        self.ny = size_y
        self.dx = dx
        # V 是电势分布矩阵
        self.V = np.zeros((self.ny, self.nx))
        # mask 标记边界条件位置 (True 表示该点是固定的边界条件，不参与迭代)
        self.boundary_mask = np.zeros((self.ny, self.nx), dtype=bool)

    def set_boundary_condition(self, x_range, y_range, voltage):
        """
        设置边界条件 (例如金属板、围墙等)
        :param x_range: tuple (x_start, x_end)
        :param y_range: tuple (y_start, y_end)
        :param voltage: 该区域的电势值
        """
        x0, x1 = x_range
        y0, y1 = y_range
        
        # 确保索引在范围内
        x0, x1 = max(0, x0), min(self.nx, x1)
        y0, y1 = max(0, y0), min(self.ny, y1)

        self.V[y0:y1, x0:x1] = voltage
        self.boundary_mask[y0:y1, x0:x1] = True

    def set_boundary_line(self, start, end, voltage, thickness=1.0):
        """
        设置线形边界 (支持任意角度的斜线)
        :param start: (x, y) 起点坐标
        :param end: (x, y) 终点坐标
        :param voltage: 电势值
        :param thickness: 线条宽度 (默认为1.0)
        """
        x0, y0 = start#这里start是一个元组，包含起点的x和y坐标
        x1, y1 = end#这里end是一个元组，包含终点的x和y坐标
        
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

    def set_boundary_point(self, x, y, voltage):
        """
        设置单点边界 (模拟尖端)
        :param x: x 坐标
        :param y: y 坐标
        :param voltage: 电势值
        """
        # 转换为整数索引
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.V[iy, ix] = voltage
            self.boundary_mask[iy, ix] = True

    def set_boundary_circle(self, center, radius, voltage):
        """
        设置圆形边界
        :param center: (x, y) 圆心
        :param radius: 半径
        :param voltage: 电势值
        """
        cx, cy = center
        Y, X = np.ogrid[:self.ny, :self.nx]
        
        # 计算到圆心的距离平方
        dist_sq = (X - cx)**2 + (Y - cy)**2
        
        # 标记圆内的点
        mask = dist_sq <= radius**2
        self.V[mask] = voltage
        self.boundary_mask[mask] = True

    def solve(self, method='sor', max_iter=10000, tol=1e-4, omega=1.5):
        """
        求解拉普拉斯方程
        :param method: 'jacobi' 或 'sor' (Successive Over-Relaxation)
        :param max_iter: 最大迭代次数
        :param tol: 收敛容差 (当变化小于此值时停止)
        :param omega: 松弛因子 (仅用于 SOR, 1 < omega < 2)
        :return: 迭代历史 (误差列表)
        """
        history = []
        
        print(f"开始求解: 方法={method}, 网格大小={self.nx}x{self.ny}")
        
        for i in range(max_iter):
            V_old = self.V.copy()
            
            if method == 'jacobi':
                # Jacobi 迭代: V_new[i,j] 只依赖于 V_old
                # V(i,j) = 1/4 * (V(i+1,j) + V(i-1,j) + V(i,j+1) + V(i,j-1))
                # 使用 numpy 切片加速计算，避免慢速的 python 循环
                
                # 计算上下左右四个邻居的平均值
                # 这种写法对应于离散化的拉普拉斯算子
                V_neighbors = 0.25 * (V_old[1:-1, 0:-2] +  # 左
                                      V_old[1:-1, 2:] +    # 右
                                      V_old[0:-2, 1:-1] +  # 上
                                      V_old[2:, 1:-1])     # 下               
                # 更新内部点 (不包括最外层边界，除非最外层也是待求区域，这里假设最外层是自然边界或固定边界)
                # 为了简单，我们假设整个网格边缘是固定的 0V 或者由 set_boundary_condition 设定
                # 这里我们只更新非边界掩码的区域
                # 构造更新后的矩阵
                V_new = V_old.copy()
                V_new[1:-1, 1:-1] = V_neighbors
                
                # 恢复固定边界条件的值 (因为上面的矩阵操作可能会覆盖掉内部的固定边界)
                # 注意：这里利用 mask 恢复那些被强制设定的电位点
                np.putmask(V_new, self.boundary_mask, self.V)
                
                self.V = V_new

            elif method == 'sor':
                # SOR 迭代 (包含 Gauss-Seidel): 使用最新的值进行更新，并引入松弛因子
                # 由于 Python 循环太慢，这里使用 Numba 加速或者简单的棋盘格更新(Red-Black ordering)
                # 为了代码简单易懂且不引入额外依赖(如numba)，我们用纯 Numpy 实现“棋盘格”更新法
                
                
                # 1. 更新“红”格子 (i+j 为偶数)
                self._update_checkerboard(0, omega)
                # 2. 更新“黑”格子 (i+j 为奇数)
                self._update_checkerboard(1, omega)
            
            # 计算误差 (最大变化量)
            diff = np.abs(self.V - V_old)
            # 忽略边界点上的“变化” (它们应该是0)
            diff[self.boundary_mask] = 0
            error = np.max(diff)
            history.append(error)
            
            if error < tol:
                print(f"收敛于第 {i} 次迭代, 误差: {error:.2e}")
                break
        else:
            print(f"达到最大迭代次数 {max_iter}, 最终误差: {error:.2e}")
        return history

    def _update_checkerboard(self, offset, omega):
        """
        棋盘格更新法 (Red-Black Update) 用于 SOR/Gauss-Seidel 的向量化实现
        offset: 0 或 1
        """
        # V[i, j] = (1-omega)*V[i,j] + omega/4 * (sum of neighbors)
        
        V = self.V
        # 创建切片视图
        # 我们只更新内部点 [1:-1, 1:-1]
        
        # 计算邻居平均值
        # 注意：这里直接用 self.V (即最新的值)，体现了 Gauss-Seidel 的特性
        neighbors = 0.25 * (V[1:-1, 0:-2] + V[1:-1, 2:] + V[0:-2, 1:-1] + V[2:, 1:-1])
        
        # 计算 SOR 更新值
        new_val = (1 - omega) * V[1:-1, 1:-1] + omega * neighbors
        
        # 创建棋盘掩码
        y_idx, x_idx = np.ogrid[1:self.ny-1, 1:self.nx-1]
        checker_mask = (y_idx + x_idx) % 2 == offset
        
        # 只更新那些“不是固定边界”的点
        # self.boundary_mask[1:-1, 1:-1] 对应内部区域的边界掩码
        # ~self.boundary_mask[...] 表示“非边界”
        active_mask = checker_mask & (~self.boundary_mask[1:-1, 1:-1])
        
        # 只更新对应颜色的、且非边界的格子
        V[1:-1, 1:-1][active_mask] = new_val[active_mask]
        

    def calculate_field(self):
        """
        根据电势 V 计算电场 E = -grad(V)
        :return: Ex, Ey 矩阵
        """
        # 使用 numpy 的 gradient 函数计算梯度
        # 注意：E = -dV/dx, -dV/dy
        # np.gradient 返回 (d/dy, d/dx)
        Gy, Gx = np.gradient(self.V, self.dx)
        Ex = -Gx
        Ey = -Gy
        return Ex, Ey
