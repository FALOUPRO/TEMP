import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QWidget
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QSplitter, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# 假设你的数据保存在 'data.xlsx' 这个 Excel 文件中
df = pd.read_excel('G.xlsx')

# 创建新的列 "FT"，这个列是脚盘的坐标
FT = [(3250,5000),(-3250,5250),(-3250,-2250),(3250,-2000)]
FTMAXLOAD = (35,35,35,35)
K = (100000, 100000, 100000, 100000)
angle = 0
MCAR = 20
MDRV = 15
rpoint = (0, 0)
MCARpoint = (0, 1065)
MDRVpoint = (0, 3000)
loadpoint = (0, 13650)
rpoint = (0, 0)
deta = 0.1
tolerance = 0.1
tol = 0.00001

class DISTANCE:
    def calculate_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def GARM(ft1, ft2, loadpoint):
        x1, y1 = ft1
        x2, y2 = ft2
        x, y = loadpoint
        return abs((y1 - y2) * x + (x2 - x1) * y + x1 * y2 - y1 * x2) / math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
    
    def load(rpoint, loadpoint, angle):
        x0, y0 = rpoint
        x1, y1 = loadpoint
        angle = math.radians(angle)

        r = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        theta = math.asin((x1-x0)/r)
        x = x0 + r * math.sin(angle + theta)
        y = y0 + r * math.cos(angle + theta)
        return (x, y), r, theta
    
    def MASS(rpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle):
        x0, y0 = rpoint
        x1, y1 = MCARpoint
        x2, y2 = MDRVpoint
        MC=MCAR
        MD=MDRV
        angle = math.radians(angle)

        XC=(x1-x0)*MC/(MC+MD)+x0
        YC=(y1-y0)*MC/(MC+MD)+y0
        r = ((x2 - x0)**2 + (y2 - y0)**2)**0.5*(MD/(MC+MD))
        theta = math.asin(x2/((x2 - x0)**2 + (y2 - y0)**2)**0.5)
        x = XC + r * math.sin(angle + theta)
        y = YC + r * math.cos(angle + theta)
        return (x, y), (XC, YC), r, theta
    
    def MAXLOAD(angle, MCAR, MDRV, FT, rpoint, MCARpoint, MDRVpoint, loadpoint):
        FWLOAD=0
        BHLOAD=0
        RLOAD=0
        LLOAD=0
         # 计算 MASS 的值，并只取前两个返回值
        mass_loc = DISTANCE.MASS(rpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle)[0]
        load_loc = DISTANCE.load(rpoint, loadpoint, angle)[0]
        angle = math.radians(angle)
        x0, y0 = mass_loc
        x1, y1 = load_loc

        if math.cos(angle)>=0:
            # 向前
            FWLOAD=(MCAR+MDRV)*DISTANCE.GARM(FT[0], FT[1], mass_loc)/DISTANCE.GARM(FT[0], FT[1], load_loc)
            if math.sin(angle)>=0:
                # 右前
                RLOAD=(MCAR+MDRV)*DISTANCE.GARM(FT[0], FT[3], mass_loc)/DISTANCE.GARM(FT[0], FT[3], load_loc)
                case = 'fr'
                mr = DISTANCE.GARM(FT[0], FT[3], mass_loc)
                lr = DISTANCE.GARM(FT[0], FT[3], load_loc)
            else:
                # 左前
                LLOAD=(MCAR+MDRV)*DISTANCE.GARM(FT[1], FT[2], mass_loc)/DISTANCE.GARM(FT[1], FT[2], load_loc)
                case = 'fl'
                mr = DISTANCE.GARM(FT[1], FT[2], mass_loc)
                lr = DISTANCE.GARM(FT[1], FT[2], load_loc)
        else:
            # 向后
            FWLOAD=(MCAR+MDRV)*DISTANCE.GARM(FT[2], FT[3], mass_loc)/DISTANCE.GARM(FT[2], FT[3], load_loc)
            if math.sin(angle)>=0:
                # 右后
                RLOAD=(MCAR+MDRV)*DISTANCE.GARM(FT[0], FT[3], mass_loc)/DISTANCE.GARM(FT[0], FT[3], load_loc)
                case = 'br'
                mr = DISTANCE.GARM(FT[0], FT[3], mass_loc)
                lr = DISTANCE.GARM(FT[0], FT[3], load_loc)
            else:
                # 左后
                LLOAD=(MCAR+MDRV)*DISTANCE.GARM(FT[1], FT[2], mass_loc)/DISTANCE.GARM(FT[1], FT[2], load_loc)
                case = 'bl'
                mr = DISTANCE.GARM(FT[1], FT[2], mass_loc)
                lr = DISTANCE.GARM(FT[1], FT[2], load_loc)

        # 取值最小的一个
        loads = [FWLOAD, BHLOAD, RLOAD, LLOAD]
        filtered_loads = [load for load in loads if load != 0]
        MAXLOAD = min(filtered_loads) if filtered_loads else 0
        return MAXLOAD, case, mr, lr, x0, y0, x1, y1
    
    def feetload(mutimass, FC, FT, K, rpoint, angle):
        (x, y), load, G = mutimass
        # FC_dir = {
        #     12: ([(x1, y1), (x2, y2), (x3, y3), (x0, y0)],[(x0, y0), (x1, y1), (x2, y2), (x3, y3)]),
        #     23: ([(x0, y0), (x1, y1), (x2, y2), (x3, y3)],[(x3, y3), (x0, y0), (x1, y1), (x2, y2)]),
        #     34: ([(x3, y3), (x0, y0), (x1, y1), (x2, y2)],[(x2, y2), (x3, y3), (x0, y0), (x1, y1)]),
        #     41: ([(x2, y2), (x3, y3), (x0, y0), (x1, y1)],[(x1, y1), (x2, y2), (x3, y3), (x0, y0)])
        # }
        # [(x1, y1), (x2, y2), (x3, y3), (x0, y0)],[(A0, B0), (A1, B1), (A2, B2), (A3, B3)]=FC_dir.get(FC,(FT))

        if FC == 12:
            (x1, y1), (x2, y2), (x3, y3), (x0, y0) = FT
            (A0, B0), (A1, B1), (A2, B2), (A3, B3) = FT
        elif FC == 23:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = FT
            (A3, B3), (A0, B0), (A1, B1), (A2, B2) = FT
        elif FC == 34:
            (x3, y3), (x0, y0), (x1, y1), (x2, y2) = FT
            (A2, B2), (A3, B3), (A0, B0), (A1, B1) = FT
        elif FC == 41:
            (x2, y2), (x3, y3), (x0, y0), (x1, y1) = FT
            (A1, B1), (A2, B2), (A3, B3), (A0, B0) = FT
        # T0, T1, T2, T3 = FTMAXLOAD
        F0, F1, F2, F3 = 0, 0, 0, 0 # 初始化 F0, F1, F2, F3
        FE0, FE1, FE2, FE3 = 0, 0, 0, 0 # 初始化 FE0, FE1, FE2, FE3
        XC = -(-x1*x2*y0 + x2*x3*y0 + x0*x3*y1 - x2*x3*y1 + x0*x1*y2 - x0*x3*y2 - x0*x1*y3 + x1*x2*y3)/(x1*y0 - x3*y0 - x0*y1 + x2*y1 - x1*y2 + x3*y2 +x0*y3 - x2*y3)
        YC = -(-(-x2*y0 + x0*y2)*(y1 - y3) + (y0 - y2)*(-x3*y1 + x1*y3))/((-x1 + x3)*(y0 - y2) - (-x0 + x2)*(y1 - y3))
        # 创建对角线交点坐标
        
        # F0*X0 + F1*X1 + F2*X2 = load*x
        # F0*Y0 + F1*Y1 + F2*Y2 = load*y
        # F0 + F1 + F2 = load
        D = np.linalg.det(np.array([[x0, x1, x2], [y0, y1, y2], [1, 1, 1]]))
        D0 = np.linalg.det(np.array([[load*x, x1, x2], [load*y, y1, y2], [load, 1, 1]]))
        D1 = np.linalg.det(np.array([[x0, load*x, x2], [y0, load*y, y2], [1, load, 1]]))
        D2 = np.linalg.det(np.array([[x0, x1, load*x], [y0, y1, load*y], [1, 1, load]]))
        F0 = D0 / D
        F1 = D1 / D
        F2 = D2 / D
        DT0 = F0 / K[0]
        DT1 = F1 / K[1]
        DT2 = F2 / K[2]
        DTC = (DT2-DT0)*DISTANCE.calculate_distance((x0, y0), (XC, YC))/DISTANCE.calculate_distance((x0, y0), (x2, y2))+DT0
        CG = (DT1-DTC)*DISTANCE.calculate_distance((x3, y3), (XC, YC))/DISTANCE.calculate_distance((x1, y1), (XC, YC))+DTC

        R = np.linalg.det(np.array([[A0, A1, A2], [B0, B1, B2], [1, 1, 1]]))
        R0 = np.linalg.det(np.array([[load*x, A1, A2], [load*y, B1, B2], [load, 1, 1]]))
        R1 = np.linalg.det(np.array([[A0, load*x, A2], [B0, load*y, B2], [1, load, 1]]))
        R2 = np.linalg.det(np.array([[A0, A1, load*x], [B0, B1, load*y], [1, 1, load]]))
        FE0 = R0 / R
        FE1 = R1 / R
        FE2 = R2 / R
        DE0 = FE0 / K[0]
        DE1 = FE1 / K[1]
        DE2 = FE2 / K[2]
        DEC = (DE2-DE0)*DISTANCE.calculate_distance((A0, B0), (XC, YC))/DISTANCE.calculate_distance((A0, B0), (A2, B2))+DE0
        CGE = (DE1-DEC)*DISTANCE.calculate_distance((A3, B3), (XC, YC))/DISTANCE.calculate_distance((A1, B1), (XC, YC))+DEC
        if CG >=0:
            F3 = 0
        if CGE >=0:
            FE3 = 0
        if CG < 0:
            A = 1/K[0]*(DISTANCE.calculate_distance((x0, y0), (XC, YC))/DISTANCE.calculate_distance((x0, y0), (x2, y2))-1)
            B = 1/K[1]*(1-DISTANCE.calculate_distance((x1, y1), (XC, YC))/DISTANCE.calculate_distance((x1, y1), (x3, y3)))
            C = -1/K[2]*DISTANCE.calculate_distance((x0, y0), (XC, YC))/DISTANCE.calculate_distance((x2, y2), (x0, y0))
            D = 1/K[3]*DISTANCE.calculate_distance((x1, y1), (XC, YC))/DISTANCE.calculate_distance((x1, y1), (x3, y3))
            # F0*X0 + F1*X1 + F2*X2 + F3*X3 = load*x
            # F0*Y0 + F1*Y1 + F2*Y2 + F3*Y3 = load*y
            # F0 + F1 + F2 + F3 = load
            # A*F0 + B*F1 + C*F2 + D*F3 = 0
            D = np.linalg.det(np.array([[x0, x1, x2, x3], [y0, y1, y2, y3], [1, 1, 1, 1], [A, B, C, D]]))
            D0 = np.linalg.det(np.array([[load*x, x1, x2, x3], [load*y, y1, y2, y3], [load, 1, 1, 1], [0, B, C, D]]))
            D1 = np.linalg.det(np.array([[x0, load*x, x2, x3], [y0, load*y, y2, y3], [1, load, 1, 1], [A, 0, C, D]]))
            D2 = np.linalg.det(np.array([[x0, x1, load*x, x3], [y0, y1, load*y, y3], [1, 1, load, 1], [A, B, 0, D]]))
            D3 = np.linalg.det(np.array([[x0, x1, x2, load*x], [y0, y1, y2, load*y], [1, 1, 1, load], [A, B, C, 0]]))
            F0 = D0 / D
            F1 = D1 / D
            F2 = D2 / D
            F3 = D3 / D
            print('k')
        if CGE < 0:
            A = 1/K[0]*(DISTANCE.calculate_distance((A0, B0), (XC, YC))/DISTANCE.calculate_distance((A0, B0), (A2, B2))-1)
            B = 1/K[1]*(1-DISTANCE.calculate_distance((A1, B1), (XC, YC))/DISTANCE.calculate_distance((A1, B1), (A3, B3)))
            C = -1/K[2]*DISTANCE.calculate_distance((A0, B0), (XC, YC))/DISTANCE.calculate_distance((A2, B2), (A0, B0))
            D = 1/K[3]*DISTANCE.calculate_distance((A1, B1), (XC, YC))/DISTANCE.calculate_distance((A1, B1), (A3, B3))
            # F0*X0 + F1*X1 + F2*X2 + F3*X3 = load*x
            # F0*Y0 + F1*Y1 + F2*Y2 + F3*Y3 = load*y
            # F0 + F1 + F2 + F3 = load
            # A*F0 + B*F1 + C*F2 + D*F3 = 0
            R = np.linalg.det(np.array([[A0, A1, A2, A3], [B0, B1, B2, B3], [1, 1, 1, 1], [A, B, C, D]]))
            R0 = np.linalg.det(np.array([[load*x, A1, A2, A3], [load*y, B1, B2, B3], [load, 1, 1, 1], [0, B, C, D]]))
            R1 = np.linalg.det(np.array([[A0, load*x, A2, A3], [B0, load*y, B2, B3], [1, load, 1, 1], [A, 0, C, D]]))
            R2 = np.linalg.det(np.array([[A0, A1, load*x, A3], [B0, B1, load*y, B3], [1, 1, load, 1], [A, B, 0, D]]))
            R3 = np.linalg.det(np.array([[A0, A1, A2, load*x], [B0, B1, B2, load*y], [1, 1, 1, load], [A, B, C, 0]]))
            FE0 = R0 / R
            FE1 = R1 / R
            FE2 = R2 / R
            FE3 = R3 / R
            print('ke')

        if FC == 12:
            if DTC > DEC:
                FFT = (FE0, FE1, FE2, FE3)
            else:
                FFT = (F1, F2, F3, F0)
        elif FC == 23:
            if DTC > DEC:
                FFT = (FE3, FE0, FE1, FE2)
            else:
                FFT = (F0, F1, F2, F3)
        elif FC == 34:
            if DTC > DEC:
                FFT = (FE2, FE3, FE0, FE1)
            else:
                FFT = (F3, F0, F1, F2)
        elif FC == 41:
            if DTC > DEC:
                FFT = (FE1, FE2, FE3, FE0)
            else:
                FFT = (F2, F3, F0, F1)
        
        # print(XC, YC)
        # print(DT0, DT1, DT2, DE0, DE1, DE2)
        # print(DTC, DEC)
        # print (FFT)
        return FFT, (x, y), load, angle
    
    def FTLOAD(mutimass, tol, tolerance, FT, FTMAXLOAD, rpoint, angle):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = FT
        (x, y), load, G = mutimass
        tol = tolerance
        DS1 = DISTANCE.GARM((x1, y1), (x2, y2), (x, y))
        DS2 = DISTANCE.GARM((x2, y2), (x3, y3), (x, y))
        DS3 = DISTANCE.GARM((x3, y3), (x4, y4), (x, y))
        DS4 = DISTANCE.GARM((x4, y4), (x1, y1), (x, y))
        if DS1 <= tol:
            F1 = load * DISTANCE.calculate_distance((x2, y2), (x, y)) / DISTANCE.calculate_distance((x1, y1), (x2, y2))
            F2 = load * DISTANCE.calculate_distance((x1, y1), (x, y)) / DISTANCE.calculate_distance((x1, y1), (x2, y2))
            F3 = 0
            F4 = 0
            FC = 12
        elif DS2 <= tol:
            F1 = 0
            F2 = load * DISTANCE.calculate_distance((x3, y3), (x, y)) / DISTANCE.calculate_distance((x2, y2), (x3, y3))
            F3 = load * DISTANCE.calculate_distance((x2, y2), (x, y)) / DISTANCE.calculate_distance((x2, y2), (x3, y3))
            F4 = 0
            FC = 23
        elif DS3 <= tol:
            F1 = 0
            F2 = 0
            F3 = load * DISTANCE.calculate_distance((x4, y4), (x, y)) / DISTANCE.calculate_distance((x3, y3), (x4, y4))
            F4 = load * DISTANCE.calculate_distance((x3, y3), (x, y)) / DISTANCE.calculate_distance((x3, y3), (x4, y4))
            FC = 34
        elif DS4 <= tol:
            F1 = load * DISTANCE.calculate_distance((x4, y4), (x, y)) / DISTANCE.calculate_distance((x1, y1), (x4, y4))
            F2 = 0
            F3 = 0
            F4 = load * DISTANCE.calculate_distance((x1, y1), (x, y)) / DISTANCE.calculate_distance((x1, y1), (x4, y4))
            FC = 41
        if F1 > FTMAXLOAD[0] or F2 > FTMAXLOAD[1] or F3 > FTMAXLOAD[2] or F4 > FTMAXLOAD[3]:
            DLOAD = G
            DLOAD_min = 0
            DLOAD_max = G
            while DLOAD_max - DLOAD_min > tol:  # tol是你的精度要求
                DLOAD = (DLOAD_min + DLOAD_max) / 2
                Rmutimass = DISTANCE.mutimass(DLOAD, rpoint, loadpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle)
                (F1, F2, F3, F4) = DISTANCE.feetload(Rmutimass, FC, FT, FTMAXLOAD, rpoint, angle)[0]
                if F1 > FTMAXLOAD[0] or F2 > FTMAXLOAD[1] or F3 > FTMAXLOAD[2] or F4 > FTMAXLOAD[3]:
                    DLOAD_max = DLOAD
                else:
                    DLOAD_min = DLOAD
            DLOAD = DLOAD_min
            Rmutimass = DISTANCE.mutimass(DLOAD, rpoint, loadpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle)
            (F1, F2, F3, F4) = DISTANCE.feetload(Rmutimass, FC, FT, FTMAXLOAD, rpoint, angle)[0]
        else:
            DLOAD = G
        return (F1, F2, F3, F4), (x, y), DLOAD, angle
    
    def mutimass(load, rpoint, loadpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle):
        x0, y0 = DISTANCE.MASS(rpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle)[0]
        x1, y1 = DISTANCE.load(rpoint, loadpoint, angle)[0]
        M = MCAR + MDRV
        G = load
        x = (M*x0+G*x1)/(M+G)
        y = (M*y0+G*y1)/(M+G)
        return (x, y), load + M, G
    
def plot_ftloads(ftloads):
    # 创建笛卡尔坐标图
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111)

    # 对于每个ftload结果，绘制曲线
    for ftload in ftloads:
        (F1, F2, F3, F4) = ftload
        # 绘制曲线
        ax.plot(angle, [F1, F2, F3, F4])
# 指定角度范围和步长
start_angle = 0
end_angle = 360
step_size = deta

# 创建一个空列表来存储 maxload 值
maxload_values = []
angle_values = []
case_values = []
mr_values = []
lr_values = []
x0_values = []
y0_values = []
x1_values = []
y1_values = []
ftload_values = []
dload_values = []

# 在指定的步长下遍历角度范围
for angle in np.arange(start_angle, end_angle + deta, step_size):
    # 计算当前角度下的 MAXLOAD 和 case
    MAXLOAD, case, mr, lr, x0, y0, x1, y1 = DISTANCE.MAXLOAD(angle, MCAR, MDRV, FT, rpoint, MCARpoint, MDRVpoint, loadpoint)
    (F1, F2, F3, F4),(C1, C2),DLOAD = DISTANCE.FTLOAD(DISTANCE.mutimass(MAXLOAD, rpoint, loadpoint, MCAR, MCARpoint, MDRV, MDRVpoint, angle), tol, tolerance, FT, FTMAXLOAD, rpoint, angle)[0:3]
    ftload = [F1, F2, F3, F4]
    # 将 MAXLOAD 值添加到列表中
    maxload_values.append(MAXLOAD)
    dload_values.append(DLOAD)
    # 将角度值添加到列表中
    angle_values.append(angle)
    case_values.append(case)
    mr_values.append(mr)
    lr_values.append(lr)
    x0_values.append(F1)
    y0_values.append(F2)
    x1_values.append(F3)
    y1_values.append(F4)
    ftload_values.append(ftload)

# 创建一个新的 DataFrame
result_df = pd.DataFrame({'Angle': angle_values, 'MAXLOAD': maxload_values, 'DLOAD':dload_values, 'Case': case_values, 'MR': mr_values, 'LR': lr_values, 'X0': x0_values, 'Y0': y0_values, 'X1': x1_values, 'Y1': y1_values, 'FTLOAD': ftload_values})

# 将 result_df 添加到原始 DataFrame 中
# df = pd.concat([df, result_df], axis=1)

# 打印 DataFrame
print(result_df)

# 创建一个 QApplication
app = QApplication([])

# 创建一个 QMainWindow
window = QMainWindow()

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

# 将表格视图添加到主窗口
window.setCentralWidget(table_view)

# 绘制曲线图

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

# 创建一个 QSplitter 并将表格视图和 QWidget 添加到其中
splitter = QSplitter()
splitter.addWidget(table_view)
splitter.addWidget(widget)
splitter.addWidget(polar_view)
# 将 QSplitter 添加到主窗口
window.setCentralWidget(splitter)

# 显示主窗口
window.show()

print(DISTANCE.MASS(rpoint, MCAR, MCARpoint, MDRV, MDRVpoint, 0))
# 运行应用程序
sys.exit(app.exec_())
