# -*-coding:utf-8-*-
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import numpy as np
from ui3 import Ui_form
from utils import epoch_buffer

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

time_i = 400  # 定时器间隔刷新时间

# 创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black',tight_layout=True)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.fig.clf()  # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用
        self.axes1 = self.fig.add_subplot(1, 1, 1)
        self.axes1.set_facecolor('black')
        self.axes1.set_ylim(-150, 150)
        self.axes1.set_xlim(0, 400)
        # 设置边框
        self.axes1.spines['right'].set_color('none')
        self.axes1.spines['top'].set_color('none')
        self.axes1.spines['left'].set_color('yellow')
        self.axes1.spines['bottom'].set_color('yellow')
        # 设置坐标轴位置
        self.axes1.spines['bottom'].set_position(('data', 0))
        self.axes1.spines['left'].set_position(('data', -2))
        # self.figure.tight_layout(pad=-1.5)
        self.fig.canvas.draw()  # 画布重绘
        self.fig.canvas.flush_events()  # 画布刷新


class FFTFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=40):
        # 第一步：创建一个创建Figure
        self.fig2 = Figure(figsize=(width, height), dpi=dpi, facecolor='black', tight_layout=True)
        # 第二步：在父类中激活Figure窗口
        super(FFTFigure, self).__init__(self.fig2)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.fig2.clf()  # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用
        self.axes1 = self.fig2.add_subplot(1, 1, 1)
        self.axes1.set_facecolor('black')
        self.axes1.set_ylim(-0.1, 1)
        # 设置边框
        self.axes1.spines['right'].set_color('none')
        self.axes1.spines['top'].set_color('none')
        self.axes1.spines['left'].set_color('yellow')
        self.axes1.spines['bottom'].set_color('yellow')
        # 设置坐标轴位置
        self.axes1.spines['bottom'].set_position(('data', 0))
        self.axes1.spines['left'].set_position(('data', 0))
        self.fig2.canvas.draw()  # 画布重绘
        self.fig2.canvas.flush_events()  # 画布刷新


class MainDialogImgBW(QMainWindow, Ui_form):
    def __init__(self):
        super(MainDialogImgBW, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Sleep Stage Scoring")
        self.setMinimumSize(0, 0)
        self.i = 0
        self.j = 0
        self.stage = 4
        self.dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.class_dict = {0: 'W', 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
        # 第五步：定义MyFigure类的一个实例
        self.F = MyFigure(width=10, height=5, dpi=80)  # 50精确，400就很粗
        self.FFT = FFTFigure(width=6, height=5, dpi=40)
        # 添加定时器,动态显示
        self.timer_re = QTimer(self)
        self.timer_re.start(time_i)  # ms
        self.timer_re.timeout.connect(self.auto_plot)

        # 第六步：在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F, 0, 0)
        # 在GUI中添加FFT实例
        self.gridlayout = QGridLayout(self.groupBox_4)  # 继承容器groupBox_4
        self.gridlayout.addWidget(self.FFT, 0, 0)


    def auto_plot(self):
        if epoch_buffer.get_data_state(self.i, time_i):  # 50=time_i / 10
            x, sample_rate = epoch_buffer.get_data(400, 40, 10, self.i)
            self.F.axes1.set_title('EEG_FPZ_CZ')
            self.F.axes1.cla()
            self.F.axes1.plot(x, 'r')
            # 设置坐标轴颜色、刻度、位置
            self.F.axes1.tick_params(axis='x', colors='yellow')
            self.F.axes1.tick_params(axis='y', colors='yellow')
            self.F.axes1.set_ylim(-150, 150)
            self.F.axes1.set_xlim(0, 400)
            self.F.axes1.spines['bottom'].set_position(('data', 0))
            self.F.axes1.spines['left'].set_position(('data', -2))
            self.F.axes1.set_ylabel("Amplitude(uV)", color="yellow", fontsize=10)
            self.F.axes1.set_xlabel("Frequency(Hz)", color="yellow", fontsize=10)
            self.F.draw()

            self.FFT.axes1.cla()
            N = 200
            x_ = N/2 * np.linspace(0, 1, N / 2)
            transformed = 1.0 / N * np.abs(np.fft.fft(x, N))
            self.FFT.axes1.plot(x_, transformed[0:N / 2], )
            self.FFT.axes1.legend(["channel 1"], loc='upper right', facecolor='white')  # label
            self.FFT.axes1.set_title("FFT", color='red', fontsize=20, pad=-20)
            self.FFT.axes1.set_ylim(-1.0, 10.0)
            self.FFT.axes1.tick_params(axis='x', colors='yellow', labelsize=15)
            self.FFT.axes1.tick_params(axis='y', colors='yellow')
            self.FFT.axes1.spines['bottom'].set_position(('data', 0))
            self.FFT.axes1.spines['left'].set_position(('data', 0))
            self.FFT.axes1.set_ylabel("Amplitude(uV)", color="yellow", fontsize=20)
            self.FFT.axes1.set_xlabel("Frequency(Hz)", color="yellow", fontsize=20)
            self.FFT.draw()

            # classes
            if epoch_buffer.get_label_state(self.j):
                self.stage = epoch_buffer.get_label(self.j)
                self.dict[self.stage] = self.dict[self.stage] + 1
                self.classes(self.stage, sample_rate)
                self.j = self.j + 1
            self.i = self.i + 1
        else:
            self.F.axes1.set_title('EEG_FPZ_CZ')
            # self.F.axes1.cla()
            # 设置坐标轴颜色、刻度、位置
            self.F.axes1.tick_params(axis='x', colors='yellow')
            self.F.axes1.tick_params(axis='y', colors='yellow')
            self.F.axes1.set_ylim(-150, 150)
            self.F.axes1.set_xlim(0, 400)
            self.F.axes1.spines['bottom'].set_position(('data', 0))
            self.F.axes1.spines['left'].set_position(('data', -2))
            self.F.axes1.set_ylabel("Amplitude(uV)", color="yellow", fontsize=10)
            self.F.axes1.set_xlabel("Frequency(Hz)", color="yellow", fontsize=10)
            self.F.draw()
            # self.FFT.axes1.cla()
            self.FFT.axes1.set_title("FFT", color='red', fontsize=20, pad=-20)
            self.FFT.axes1.set_ylim(-1.0, 10.0)
            self.FFT.axes1.tick_params(axis='x', colors='yellow', labelsize=15)
            self.FFT.axes1.tick_params(axis='y', colors='yellow')
            self.FFT.axes1.spines['bottom'].set_position(('data', 0))
            self.FFT.axes1.spines['left'].set_position(('data', 0))
            self.FFT.axes1.set_ylabel("Amplitude(uV)", color="yellow", fontsize=20)
            self.FFT.axes1.set_xlabel("Frequency(Hz)", color="yellow", fontsize=20)
            self.FFT.draw()


    def classes(self, stage, sample_rate):
        self.lineEdit.setText(str(self.dict[0]))
        self.lineEdit_2.setText(str(self.dict[1]))
        self.lineEdit_3.setText(str(self.dict[2]))
        self.lineEdit_4.setText(str(self.dict[3]))
        self.lineEdit_5.setText(str(self.dict[4]))
        self.lineEdit_6.setText(self.class_dict[stage])

        self.lineEdit_7.setText(str(sample_rate))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    sys.exit(app.exec_())
