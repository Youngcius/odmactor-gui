import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtChart import QChartView, QChart, QLineSeries, QValueAxis

# 继承关系：QGraphicsItem --> QGraphicsObject --> QGraphicsWidget --> QChart --> QPolarChart

# import TimeTagger as tt

from ui import odmactor_window
from utils.sequence import *
from utils.asg import ASG

timeUnitDict = {'s': 1, 'ms': C.milli, 'ns': C.nano, 'ps': C.pico}


class OdmactorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(OdmactorGUI, self).__init__()
        self.ui = odmactor_window.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)

        self._buildUI()

        # self.scheduler =
        self.initSequenceFigure()
        self.initPhotonCountFigure()

        # Initialize hardware resources
        # self.asg = ASG()
        # self.tagger = tt.TimeTagger()
        # self.tagger: tt.TimeTagger = None
        self.photonCountConfig = {
            'channels': [1],
            'binwidth': 0,
            'n_values': 0
        }
        # self.counter = tt.Counter(self.tagger, **self.photonCountConfig)
        # self.counter.stop()

    def _buildUI(self):
        # self.setCentralWidget(self.ui.Splitter)

        #########################
        # Status bar
        self.labelInstrStatus = QtWidgets.QLabel(self)
        self.labelInstrStatus.setText('Instrument Status:')
        self.ui.statusbar.addWidget(self.labelInstrStatus)

        self.progressBar = QtWidgets.QProgressBar(self)
        self.progressBar.setMinimumWidth(200)
        self.progressBar.setMinimum(5)
        self.progressBar.setValue(30)
        self.progressBar.setMaximum(50)
        self.progressBar.setFormat('%p%')  # %p%, %v
        self.ui.statusbar.addPermanentWidget(self.progressBar)

        #########################
        # ASG sequence table
        # self.ui.tableWidgetSequence .... 初始化

    def initSequenceFigure(self):
        seq_data = [
            [10, 20, 30, 10],
            [20, 10, 10, 30],
            [0, 0],
            [0, 0],
            [35, 35]
        ]
        self.seq_fig = seq_to_fig(seq_data)
        self.seq_fig_canvas = FigureCanvas(self.seq_fig)
        # FigureCanvas.
        self.seq_fig_canvas.setParent(self.ui.groupBoxSequenceVisualization)
        # add_axes_to_fig(seq_data, self.seq_fig, 1, 1,1)
        # ax1 = self.seq_fig.add_subplot(1, 1, 1)
        # xmax = 5
        # x = np.linspace(0.0, xmax, 200)
        # y = np.cos(2 * np.pi * x) * np.exp(-x)

        # ax1.plot(x, y)
        # ax1.set_title("曲线之间填充")
        # ax1.set_xlabel('时间(sec)')
        # ax1.set_ylabel('响应幅度')
        # self.ui.groupBoxSequenceVisualization.setCentralWidget(self.seq_fig_canvas)
        # gridLayout_tmp = QtWidgets.QGridLayout(self.ui.groupBoxSequenceVisualization)

        # gridLayout_tmp.addWidget(self.seq_fig_canvas, 0, 0, 1, 1)
        # self.seq_fig.

    #
    # def initStatictialFigure(self):
    #     # mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei']  # 汉字字体
    #     plt.style.use('seaborn')
    #     self.stat_fig = plt.figure(figsize=(8, 5))
    #     self.statFigureCanvas = FigureCanvas(self.stat_fig)
    #     self.statFigureCanvas.setParent(self.ui.groupBoxStatistics)
    #     gridLayout_tmp = QtWidgets.QGridLayout(self.ui.groupBoxStatistics)
    #     gridLayout_tmp.addWidget(self.statFigureCanvas, 0, 0, 1, 1)
    #     self.stat_fig.suptitle('Statistical Information of Current Dataset')
    #     self.stat_ax1 = self.stat_fig.add_subplot(1, 2, 1)  # 分子中原子数
    #     self.stat_ax2 = self.stat_fig.add_subplot(1, 2, 2)  # 某个选定属性的统计图（hist）
    #     self.stat_ax1.set_title('Number of Atoms')
    #     self.stat_ax2.set_title('Selected Property')

    ########################
    # Photon counting
    def initPhotonCountFigure(self):
        """
        Demo
        """
        chart = QChart()
        chartView = QChartView(self.ui.scrollArea3)
        chartView.setChart(chart)

        # 创建曲线序列
        series0 = QLineSeries()
        series1 = QLineSeries()
        series0.setName("Sin曲线")
        series1.setName("Cos曲线")
        chart.addSeries(series0)  # 序列添加到图表
        chart.addSeries(series1)

        # 序列添加数值
        t = 0
        intv = 0.1
        pointCount = 100
        for i in range(pointCount):
            y1 = math.cos(t)
            series0.append(t, y1)
            y2 = 1.5 * math.sin(t + 20)
            series1.append(t, y2)
            t = t + intv

        ##创建坐标轴
        axisX = QValueAxis()  # X 轴
        axisX.setRange(0, 10)  # 设置坐标轴范围
        axisX.setTitleText("time(secs)")  # 标题
        ##    axisX.setLabelFormat("%.1f")     #标签格式
        ##    axisX.setTickCount(11)           #主分隔个数
        ##    axisX.setMinorTickCount(4)
        ##    axisX.setGridLineVisible(false)

        axisY = QValueAxis()  # Y 轴
        axisY.setRange(-2, 2)
        axisY.setTitleText("value")
        ##    axisY.setTickCount(5)
        ##    axisY.setMinorTickCount(4)
        ##    axisY.setLabelFormat("%.2f")     #标签格式
        ##    axisY.setGridLineVisible(false)

        # 为序列设置坐标轴
        chart.setAxisX(axisX, series0)  # 为序列设置坐标轴
        chart.setAxisY(axisY, series0)

        chart.setAxisX(axisX, series1)  # 为序列设置坐标轴
        chart.setAxisY(axisY, series1)

    ########################
    # About information
    @pyqtSlot()
    def on_actionAbout_triggered(self):
        txt = 'Version:\t {}\n'.format('1.0.0')
        txt += 'Date:\t {}\n'.format('March, 2022')
        txt += 'Author:\t {}\n'.format('Zhaohui Yang')
        txt += 'Email:\t {}\n'.format('zhy@email.arizona.edu')
        txt += 'Brief:\t {}\n'.format('A soft-hardware-integrated operation software for ODMR and spin manipulation. '
                                      'Anyone is welcome to contact the author!')
        QtWidgets.QMessageBox.about(self, 'Software Information', txt)

    @pyqtSlot()
    def on_actionHelp_triggered(self):

        manual_website = 'https://github.com/Youngcius/odmactor/three/master/asset/doc/manual.md'
        txt = 'Please refer the <a href = {}>online user manual</a>. ' \
              'Or contact the author whose email is on the "About" page.'.format(manual_website)
        QtWidgets.QMessageBox.about(self, 'Help', txt)

    @pyqtSlot()
    def on_actionQtInfo_triggered(self):
        QtWidgets.QMessageBox.aboutQt(self)

    ########################
    # Photon count parameter updating
    # @pyqtSlot()

    # @pyqtSlot()
    # def on_spinBoxBinwidth_valueChange
    # TODO: on/off 切换状态时候
    @pyqtSlot(bool)
    def on_pushButtonPhotonCountOnOff_clicked(self, checked):
        """
        :param checked: if True, reload parameters to start counting; otherwise, stop counting
        """
        if checked:
            unit = timeUnitDict[self.ui.comboBoxPhotonCountTimeUnit.currentText()]
            self.photonCountConfig['binwidth'] = unit * self.ui.spinBoxBinwidth.value() / C.pico
            self.photonCountConfig['n_values'] = self.ui.spinBoxCountNumber.value()
            # self.photonCountConfig['channels'] = ......... TODO: 左边的 apd channel （Tagger 的channel）
            self.counter = tt.Counter(*self.photonCountConfig)
            self.counter.start()
        else:
            self.counter.stop()

    @pyqtSlot()
    def on_pushButtonPhotonCountRefresh_clicked(self):
        pass

    @pyqtSlot()
    def on_pushButtonPhotonCountSaveData_clicked(self):
        """
        Save data in form of JSON file in default
        """
        pass

    ########################
    # ASG sequences configuration
    @pyqtSlot()
    def on_pushButtonASGOpen_clicked(self):
        self.asg.connect()

    @pyqtSlot()
    def on_pushButtonASGClose_clicked(self):
        self.asg.close_device()

    @pyqtSlot()
    def on_pushButtonASGStart_clicked(self):
        self.asg.start()

    @pyqtSlot()
    def on_pushButtonASGStop_clicked(self):
        self.asg.stop()

    @pyqtSlot()
    def on_pushButtonASGLoad_clicked(self):
        # 1) seq = .........
        # self.asg.download_ASG_pulse_data(seq)
        # 用写好的函数 scheduler中
        # 2) 可视化
        pass

    @pyqtSlot()
    def on_pushButtonASGClear_clicked(self):
        self.ui.tableWidgetSequence.clear()

    @pyqtSlot()
    def on_pushButtonASGAdd_clicked(self):

        # self.ui.tableWidgetSequence.add colume
        pass
