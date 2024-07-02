import sys
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, QObject
from UI import UIControl
from calculate import CALCU
import numpy as np
import pandas as pd

FT = [(3250,5000),(-3250,5250),(-3250,-2250),(3250,-2000)]
FTMAXLOAD = (35,35,35,35)
K = (100000, 100000, 100000, 100000)
angle = 0
MCAR = 20
MDRV = 15
MCARpoint = (0, 1065)
MDRVpoint = (0, 3000)
loadpoint = (0, 13650)
rpoint = (0, 0)
deta = 0.1
tolerance = 0.1
tol = 0.00001

# 示例数据
data = {
    "Angle": np.linspace(0, 360, 100),  # 生成0到360度的100个点
    "MAXLOAD": np.random.rand(100) * 100,  # 随机生成MAXLOAD数据
    "FTLOAD": np.random.rand(100) * 100,  # 随机生成FTLOAD数据
    "MR": np.random.rand(100) * 50,  # 随机生成MR数据
    "LR": np.random.rand(100) * 50,  # 随机生成LR数据
    "X0": np.random.rand(100) * 10,  # 随机生成X0数据
    "Y0": np.random.rand(100) * 10,  # 随机生成Y0数据
    "X1": np.random.rand(100) * 10,  # 随机生成X1数据
    "Y1": np.random.rand(100) * 10,  # 随机生成Y1数据
    "DLOAD": np.random.rand(100) * 100  # 随机生成DLOAD数据
}

# 创建DataFrame
output_data = pd.DataFrame(data)

# 需求：main函数应该怎么构建，设计一个基础架构，需要与一个ui模块进行通讯，交换输入输出与控制信息；还需要控制计算模块，按ui的触发信息决定什么时候依照输入计算输出
class UiWindow(QMainWindow):
    # 定义一个信号，用于在计算完成后更新UI
    update_ui_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        df = pd.DataFrame()  # 创建一个空的 DataFrame
        # inp = (FT, FTMAXLOAD, K, angle, MCAR, MDRV, rpoint, MCARpoint, MDRVpoint, loadpoint, rpoint, deta, tolerance, tol)
        inp = {
        'FT': FT,
        'FTMAXLOAD': FTMAXLOAD,
        'K': K,
        'angle': angle,
        'MCAR': MCAR,
        'MDRV': MDRV,
        'MCARpoint': MCARpoint,
        'MDRVpoint': MDRVpoint,
        'loadpoint': loadpoint,
        'rpoint': rpoint,
        'deta': deta,
        'tolerance': tolerance,
        'tol': tol
        }
        # 使用线程执行计算任务，避免阻塞UI
        self.compute_task = CALCU(inp)  # 创建一个ComputeModule类的实例
        threading.Thread(target=self.compute_task).start()  # 创建一个线程，执行compute_task方法
        # OUTPUT = self.compute_task.compute(inp)  # 使用输入数据计算输出数据
        self.ui_module = UIControl(output_data)  # 创建一个UIControl类的实例
        self.ui_module.triggered.connect(self.control_compute_task)  # 将ui_module的triggered信号连接到control_compute_module槽函数

    def handle_ui_input(self):
        input_data = self.ui_module.get_input()  # 从ui_module获取输入数据
        if input_data:
            self.control_compute_task()  # 如果有输入数据，则调用control_compute_module方法
        return input_data  # 返回输入数据

    def handle_ui_output(self, output_data):
        self.ui_module.set_output(output_data)  # 在ui_module中设置输出数据

    def control_compute_task(self):
        input_data = self.handle_ui_input()  # 从ui_module获取输入数据
        output_data = self.compute_task.compute(input_data)  # 使用输入数据计算输出数据
        self.handle_ui_output(output_data)  # 在ui_module中设置输出数据

if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建一个QApplication类的实例
    window = UiWindow()  # 创建一个UiWindow类的实例
    window.show()  # 显示UiWindow
    sys.exit(app.exec_())  # 启动应用程序事件循环，并在完成后退出程序
