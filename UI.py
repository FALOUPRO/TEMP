from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QWidget, QSplitter, QVBoxLayout, QAction, QLineEdit
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd

class UIControl(QMainWindow):
      def __init__(self, result_df):
        super().__init__()

        # 创建一个 QAction
        self.action = QAction("Action", self)
        self.action.triggered.connect(self.handle_triggered)

        # 创建一个 QMainWindow
        self.window = QMainWindow()

        self.drawtable(result_df)
        self.drawgraph(result_df)

        # 创建一个 QSplitter 并将表格视图和曲线图添加到其中
        splitter = QSplitter()
        splitter.addWidget(self.table_view)
        splitter.addWidget(self.widget)
        splitter.addWidget(self.polar_view)
        # 将 QSplitter 添加到主窗口
        # self.window.setCentralWidget(splitter)
        self.setCentralWidget(splitter)
        self.show()  # 显示窗口

      def drawtable(self, result_df):
        super().__init__()
        # 创建一个 QTableView
        table_view = QTableView()

        # 创建一个 QStandardItemModel
        model = QStandardItemModel()

        # 添加 DataFrame 的列名作为模型的标题
        model.setHorizontalHeaderLabels(result_df.columns)

        # 遍历 DataFrame 的每一行
        for index, row in result_df.iterrows():
            # 创建一个空列表来保存这一行的数据
            items = []
            
            # 遍历这一行的每一列
            for value in row:
                # 创建一个 QStandardItem 并将值转换为字符串
                item = QStandardItem(str(value))
                
                # 将这个项目添加到列表中
                items.append(item)
            
            # 将这一行的数据添加到模型中
            model.appendRow(items)

        # 将模型设置为表格视图的模型
        table_view.setModel(model)


      #   # 将表格视图添加到主窗口
      #   self.setCentralWidget(table_view)

      def handle_triggered(self):
         print("Action triggered!")

      def drawgraph(self, result_df):
        # 创建一个新的 figure，2行4列
        fig, axs = plt.subplots(4, 2, figsize=(10, 20))

        # 调整子图之间的间距
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        # 在子图中绘制曲线图
        axs[0, 0].plot(result_df['Angle'], result_df['MAXLOAD'])
        axs[0, 0].set_title('MAXLOAD')

        axs[0, 1].plot(result_df['Angle'], result_df['FTLOAD'].apply(lambda x: pd.Series(x)))
        axs[0, 1].set_title('ftload')

        axs[1, 0].plot(result_df['Angle'], result_df['MR'])
        axs[1, 0].set_title('MR')

        axs[1, 1].plot(result_df['Angle'], result_df['LR'])
        axs[1, 1].set_title('LR')

        axs[2, 0].plot(result_df['Angle'], result_df['X0'])
        axs[2, 0].set_title('x0')

        axs[2, 1].plot(result_df['Angle'], result_df['Y0'])
        axs[2, 1].set_title('y0')

        axs[3, 0].plot(result_df['Angle'], result_df['X1'])
        axs[3, 0].set_title('x1')

        axs[3, 1].plot(result_df['Angle'], result_df['Y1'])
        axs[3, 1].set_title('y1')

        # 创建一个 FigureCanvas 并将 figure 添加到其中
        canvas = FigureCanvas(fig)

        # 创建一个 QWidget 来保存 FigureCanvas
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        widget.setLayout(layout)

        # 创建一个新的 figure
        fig_polar = plt.figure(figsize=(9, 9))
        # 创建一个极坐标子图
        ax_polar = fig_polar.add_subplot(1, 1, 1, polar=True, theta_offset=np.pi/2)
        # 绘制极坐标图形，注意角度需要转换为弧度
        ax_polar.plot(np.radians(result_df['Angle']), result_df['DLOAD'])
        # 设置标题
        ax_polar.set_title('DLOAD', va='bottom')
        canvas = FigureCanvas(fig_polar)
        # 创建一个 QWidget 来保存 FigureCanvas
        polar_view = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        polar_view.setLayout(layout)

# # 创建一个 QApplication
# app = QApplication([])

# # 创建一个 QMainWindow
# window = QMainWindow()

# # 创建一个 QTableView
# table_view = QTableView()

# # 创建一个 QStandardItemModel
# model = QStandardItemModel()

# # 添加 DataFrame 的列名作为模型的标题
# model.setHorizontalHeaderLabels(result_df.columns)

# # 遍历 DataFrame 的每一行
# for index, row in result_df.iterrows():
#     # 创建一个空列表来保存这一行的数据
#     items = []
    
#     # 遍历这一行的每一列
#     for value in row:
#         # 创建一个 QStandardItem 并将值转换为字符串
#         item = QStandardItem(str(value))
        
#         # 将这个项目添加到列表中
#         items.append(item)
    
#     # 将这一行的数据添加到模型中
#     model.appendRow(items)

# # 将模型设置为表格视图的模型
# table_view.setModel(model)

# # 将表格视图添加到主窗口
# window.setCentralWidget(table_view)

# # 绘制曲线图

# # 创建一个新的 figure，2行4列
# fig, axs = plt.subplots(4, 2, figsize=(10, 20))

# # 调整子图之间的间距
# plt.subplots_adjust(hspace=0.5, wspace=0.3)

# # 在子图中绘制曲线图
# axs[0, 0].plot(result_df['Angle'], result_df['MAXLOAD'])
# axs[0, 0].set_title('MAXLOAD')

# axs[0, 1].plot(result_df['Angle'], result_df['FTLOAD'].apply(lambda x: pd.Series(x)))
# axs[0, 1].set_title('ftload')

# axs[1, 0].plot(result_df['Angle'], result_df['MR'])
# axs[1, 0].set_title('MR')

# axs[1, 1].plot(result_df['Angle'], result_df['LR'])
# axs[1, 1].set_title('LR')

# axs[2, 0].plot(result_df['Angle'], result_df['X0'])
# axs[2, 0].set_title('x0')

# axs[2, 1].plot(result_df['Angle'], result_df['Y0'])
# axs[2, 1].set_title('y0')

# axs[3, 0].plot(result_df['Angle'], result_df['X1'])
# axs[3, 0].set_title('x1')

# axs[3, 1].plot(result_df['Angle'], result_df['Y1'])
# axs[3, 1].set_title('y1')

# # 创建一个 FigureCanvas 并将 figure 添加到其中
# canvas = FigureCanvas(fig)

# # 创建一个 QWidget 来保存 FigureCanvas
# widget = QWidget()
# layout = QVBoxLayout()
# layout.addWidget(canvas)
# widget.setLayout(layout)

# # 创建一个新的 figure
# fig_polar = plt.figure(figsize=(9, 9))
# # 创建一个极坐标子图
# ax_polar = fig_polar.add_subplot(1, 1, 1, polar=True, theta_offset=np.pi/2)
# # 绘制极坐标图形，注意角度需要转换为弧度
# ax_polar.plot(np.radians(result_df['Angle']), result_df['DLOAD'])
# # 设置标题
# ax_polar.set_title('DLOAD', va='bottom')

# canvas = FigureCanvas(fig_polar)
# # 创建一个 QWidget 来保存 FigureCanvas
# polar_view = QWidget()
# layout = QVBoxLayout()
# layout.addWidget(canvas)
# polar_view.setLayout(layout)

# # 创建一个 QSplitter 并将表格视图和 QWidget 添加到其中
# splitter = QSplitter()
# splitter.addWidget(table_view)
# splitter.addWidget(widget)
# splitter.addWidget(polar_view)
# # 将 QSplitter 添加到主窗口
# window.setCentralWidget(splitter)

# # 显示主窗口
# window.show()

# print(DISTANCE.MASS(rpoint, MCAR, MCARpoint, MDRV, MDRVpoint, 0))
# # 运行应用程序
# sys.exit(app.exec_())