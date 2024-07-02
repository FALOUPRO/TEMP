import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# 创建极坐标图
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# 生成数据
theta = np.linspace(0, 2.*np.pi, 100)
r = np.abs(np.sin(5*theta))

# 使用colormap生成颜色数组
cmap = get_cmap('viridis')  # 选择colormap
colors = cmap(np.linspace(0, 1, len(theta)))

# 绘制变色填充
for i in range(len(theta)-1):
    ax.fill_between(theta[i:i+2], 0, r[i:i+2], color=colors[i], alpha=0.7)

plt.show()