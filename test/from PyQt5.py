from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
from PyQt5.QtCore import Qt, pyqtSlot
import sys

class UiWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建一个 QPushButton 对象，设置其文本为 "Click me"
        self.button = QPushButton("Click me", self)

        # 将 QPushButton 的 clicked 信号连接到 self.on_button_clicked 槽函数
        self.button.clicked.connect(self.on_button_clicked)

    @pyqtSlot()
    def on_button_clicked(self):
        print("Button clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UiWindow()
    window.show()
    sys.exit(app.exec_())