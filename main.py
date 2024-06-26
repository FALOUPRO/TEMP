import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from UI import UIControl
from calculate import CALCU
import pandas as pd

# 需求：main函数应该怎么构建，设计一个基础架构，需要与一个ui模块进行通讯，交换输入输出与控制信息；还需要控制计算模块，按ui的触发信息决定什么时候依照输入计算输出
class UiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        df = pd.DataFrame()  # 创建一个空的 DataFrame
        self.ui_module = UIControl(df)  # 创建一个UIControl类的实例
        self.compute_module = CALCU()  # 创建一个ComputeModule类的实例
        self.ui_module.triggered.connect(self.control_compute_module)  # 将ui_module的triggered信号连接到control_compute_module槽函数

    def handle_ui_input(self):
        input_data = self.ui_module.get_input()  # 从ui_module获取输入数据
        if input_data:
            self.control_compute_module()  # 如果有输入数据，则调用control_compute_module方法
        return input_data  # 返回输入数据

    def handle_ui_output(self, output_data):
        self.ui_module.set_output(output_data)  # 在ui_module中设置输出数据

    def control_compute_module(self):
        input_data = self.handle_ui_input()  # 从ui_module获取输入数据
        output_data = self.compute_module.compute(input_data)  # 使用输入数据计算输出数据
        self.handle_ui_output(output_data)  # 在ui_module中设置输出数据

if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建一个QApplication类的实例
    window = UiWindow()  # 创建一个UiWindow类的实例
    window.show()  # 显示UiWindow
    sys.exit(app.exec_())  # 启动应用程序事件循环，并在完成后退出程序
