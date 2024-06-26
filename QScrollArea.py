import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QScrollArea

class ScrollAreaExample(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('自动适应窗口')  # 设置窗口标题
        self.setGeometry(100, 100, 400, 300)  # 设置窗口大小和位置

        scroll_area = QScrollArea()  # 创建一个QScrollArea对象
        scroll_content = QWidget()  # 创建一个QWidget对象，用于放置可滚动内容
        layout = QVBoxLayout(scroll_content)  # 创建一个垂直布局管理器，并将其应用于scroll_content

        for i in range(50):  # 循环创建50个按钮
            button = QPushButton(f'按钮 {i}')  # 创建一个按钮
            layout.addWidget(button)  # 将按钮添加到布局中

        scroll_area.setWidget(scroll_content)  # 将scroll_content设置为scroll_area的子部件
        scroll_area.setWidgetResizable(True)  # 设置内容部件自动调整大小以适应scroll_area的大小

        main_layout = QVBoxLayout(self)  # 创建一个垂直布局管理器，并将其应用于主窗口
        main_layout.addWidget(scroll_area)  # 将scroll_area添加到主布局中

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    ex = ScrollAreaExample()  # 创建ScrollAreaExample对象
    ex.show()  # 显示窗口
    sys.exit(app.exec_())  # 运行应用程序
