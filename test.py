import sys
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from UI import UIControl  # 假设UI.py中的类名为UIControl
import pandas as pd

@pytest.fixture(scope="module")
def app():
    return QApplication(sys.argv)

@pytest.fixture
def ui_control():
    # 创建测试数据
    data = {'Angle': [0, 45, 90], 'MAXLOAD': [100, 150, 200]}
    df = pd.DataFrame(data)
    return UIControl(df)

def test_table_view(app, ui_control):
    # 验证表格视图是否正确显示数据
    model = ui_control.table_view.model()
    assert model.rowCount() == 3  # 假设有3行数据
    assert model.columnCount() == 2  # 假设有2列数据
    assert model.data(model.index(0, 0)) == '0'  # 验证第一行第一列的数据

def test_graph(app, ui_control):
    # 验证图表是否被创建
    assert ui_control.widget is not None
    # 更详细的图表测试可以根据实际情况添加

# 运行pytest进行测试
if __name__ == "__main__":
    pytest.main([__file__])