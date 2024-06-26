import pandas as pd  # 导入 pandas 库，用于数据处理
from PyQt5.QtWidgets import QApplication, QMainWindow  # 导入 PyQt5 库，用于创建 GUI 应用
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # 导入 Matplotlib 库，用于在 PyQt5 应用中绘制图形
from matplotlib.figure import Figure  # 导入 Figure 类，用于创建图形
from matplotlib.font_manager import FontProperties
import numpy as np  # 导入 numpy 库，用于数值计算
import sys  # 导入 sys 库，用于处理 Python 运行时环境的参数和函数

# 创建一个字体属性对象，指定字体名称为'SimHei'，这是一个支持中文的字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

# 读取 Excel 文件
sheet_name = '倾翻力矩计算'  # Excel 文件中的表格名称
skiprows = 4  # 跳过的行数
nrows = 1298  # 读取的行数
usecols = [1, 18]  # 读取的列
df = pd.read_excel('G.xlsx', sheet_name=sheet_name, skiprows=skiprows, nrows=nrows, usecols=usecols, header=None)
df.columns = ['B', 'S']  # 为数据框指定列名
# df['B'] = np.rad2deg(df['B'])  # 将弧度转换为度数

# 创建一个用于绘制极坐标图的画布类
class Canvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()  # 创建一个新的图形
        self.axes = fig.add_subplot(111, projection='polar', theta_offset=np.pi/2)  # 在图形上添加一个极坐标子图
        # self.axes.set_theta_offset(np.pi/2)  # 设置0度（极径）的偏移量为π/2弧度，使其在上方
        self.axes.set_theta_direction(-1)  # 设置角度的方向为顺时针
        super(Canvas, self).__init__(fig)  # 初始化父类
        self.setParent(parent)  # 设置父窗口

        self.plot(df['B'], df['S'])  # 绘制极坐标图

    def plot(self, x, y):
        self.axes.clear()  # 清除子图上的内容
        self.axes.plot(x, y)  # 在子图上绘制数据
        self.axes.set_title('极限吊装载荷', fontproperties=font)  # 设置图形的标题
        self.draw()  # 更新图形

         # 打印 'B' 和 'S' 列的值
        print('B:', x)
        print('S:', y)

# 创建一个主窗口类
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)  # 初始化父类

        self.canvas = Canvas(self)  # 创建一个画布
        self.setCentralWidget(self.canvas)  # 将画布设置为主窗口的中心部件

# 创建一个 PyQt5 应用
app = QApplication(sys.argv)
window = MainWindow()  # 创建一个主窗口
window.show()  # 显示主窗口
sys.exit(app.exec_())  # 运行应用，并在应用退出时返回状态码