import sys
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class UiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('UI.ui', self)  # 加载UI文件
        
        self.figure = plt.figure()  # 创建一个图形对象
        self.canvas = FigureCanvas(self.figure)  # 创建一个画布对象
        
        layout = QVBoxLayout()  # 创建一个垂直布局对象
        layout.addWidget(self.canvas)  # 将画布添加到布局中
        self.map.setLayout(layout)  # 设置 map 的布局为垂直布局
        
        self.plot()  # 调用 plot 方法绘制图形
        self.show()  # 显示窗口

    def plot(self):
        ax = self.figure.add_subplot(111, projection='3d')  # 创建一个3D子图对象
        x = np.arange(-8, 8, 0.1)  # 创建一个包含从-8到8的间隔为0.1的数组
        y = np.arange(-8, 8, 0.1)  # 创建一个包含从-8到8的间隔为0.1的数组
        x, y = np.meshgrid(x, y)  # 创建一个网格
        r = np.sqrt(x ** 2 + y ** 2)  # 计算每个点到原点的距离
        z = np.sin(r)  # 计算每个点的sin值
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))  # 绘制3D曲面图
        self.canvas.draw()  # 刷新画布

app = QApplication(sys.argv)  # 创建一个应用程序对象
window = UiWindow()  # 创建一个窗口对象
sys.exit(app.exec_())  # 运行应用程序
