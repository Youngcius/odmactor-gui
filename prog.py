import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtChart import QChartView, QChart, QLineSeries, QValueAxis

# 继承关系：QGraphicsItem --> QGraphicsObject --> QGraphicsWidget --> QChart --> QPolarChart

import TimeTagger as tt

from instrument import ASG, Microwave, Laser
import utils
from utils.sequence import *

from ui import odmactor_window

timeUnitDict = {'s': 1, 'ms': C.milli, 'ns': C.nano, 'ps': C.pico}
freqUnitDict = {'Hz': 1, 'KHz': C.kilo, 'MHz': C.mega, 'GHz': C.giga}


class OdmactorGUI(QtWidgets.QMainWindow):
    """
    Mainwindow configuration of Odmactor
    """

    # def on_tableWidgetSequence_change...........(获取table 数据):
    #     self.
    def initInstruments(self):
        self.laser = Laser()
        self.asg = ASG()

        if tt.scanTimeTagger():
            self.tagger = tt.TimeTagger()
        else:
            self.tagger = None

        try:
            self.mw = Microwave()
        except ValueError:
            self.mw = None

    ###########################################
    # Instruments configuration
    ################
    # Laser
    @pyqtSlot()
    def on_radioButtonLaserCW_clicked(self):
        self.laserMode = 'CW'

    @pyqtSlot()
    def on_radioButtonLaserPulse_clicked(self):
        self.laserMode = 'Pulse'

    @pyqtSlot(bool)
    def on_pushButtonLaserOnOff_clicked(self, checked):
        if checked:
            self.laser.start()
        else:
            self.laser.stop()

    @pyqtSlot(float)
    def on_doubleSpinBoxLaserPower_vaueChanged(self, power):
        self.laser.set_power(power)

    # Microwave
    @pyqtSlot(float)
    def on_doubleSpinBoxMicrowaveFrequency_valueChanged(self, freq):
        freq *= freqUnitDict[self.ui.comboBoxMicrowaveFrequencyUnit.currentText()]
        self.mw.set_frequency(freq)

    @pyqtSlot(float)
    def on_doubleSpinBoxMicrowavePower_valueChanged(self, power):
        # = self.ui.doubleSpinBoxMicrowavePower.value()
        if self.ui.comboBoxMicrowavePowerUnit.currentText() == 'mW':
            power *= utils.mW_to_dBm(power)
        self.mw.set_power(power)

    @pyqtSlot()
    def on_pushButtonMicrowaveOnOff_clicked(self, checked):
        if checked:
            self.mw.start()
        else:
            self.mw.stop()

    # PI pulse
    @pyqtSlot(float)
    def on_doubleSpinBoxMicrowavePiPulseDuration_valueChanged(self, duration):
        # self.piPulse['duration'] = self.ui.doubleSpinBoxMicrowavePiPulseDuration.value()
        self.piPulse['duration'] = duration * timeUnitDict[self.ui.comboBoxMicrowavePiPulseDurationUnit.currentText()]
        print('pi duration', duration)

    @pyqtSlot(float)
    def on_doubleSpinBoxMicrowavePiPulseFrequency_valueChanged(self, freq):
        self.piPulse['frequency'] = freq * freqUnitDict[self.ui.comboBoxMicrowavePiPulseFrequencyUnit.currentText()]

    @pyqtSlot(float)
    def on_doubleSpinBoxMicrowavePiPulsePower_valueChanged(self, power):
        self.piPulse['power'] = power
        if self.ui.comboBoxMicrowavePiPulsePowerUnit.currentText() == 'mW':
            self.piPulse['power'] = utils.mW_to_dBm(self.piPulse['power'])

    # ASG channels
    @pyqtSlot()
    def on_comboBoxASGLaser_valueChanged(self):
        self.asgChannels['laser'] = int(self.ui.comboBoxASGLaser.currentText())

    @pyqtSlot()
    def on_comboBoxASGMicrowave_valueChanged(self):
        self.asgChannels['mw'] = int(self.ui.comboBoxASGMicrowave.currentText())
        pass

    @pyqtSlot()
    def on_comboBoxASGAPD_valueChanged(self):
        self.asgChannels['apd'] = int(self.ui.comboBoxASGAPD.currentText())
        pass

    @pyqtSlot()
    def on_comboBoxASGTagger_valueChanged(self):
        self.asgChannels['tagger'] = int(self.ui.comboBoxASGTagger.currentText())

    # Tagger channels
    @pyqtSlot()
    def on_comboBoxTaggerAPD_valueChanged(self):
        self.taggerChannels['apd'] = int(self.ui.comboBoxTaggerAPD.currentText())

    @pyqtSlot()
    def on_comboBoxTaggerASG_valueChanged(self):
        self.taggerChannels['asg'] = int(self.ui.comboBoxTaggerTrigger.currentText())

    def fetchParameters(self):

        # ASG sequences from table widget
        self.sequences = []
        for i in range(self.ui.tableWidgetSequence.rowCount()):
            self.sequences.append([])
            for j in range(self.ui.tableWidgetSequence.columnCount()):
                self.sequences[i].append(int(self.ui.tableWidgetSequence.item(i, j).text()))
        # self.sequences[0] = [10, 0]
        # self.sequences[-1] = [10, 0]

        # Laser, MW, ASG, tagger counter # TODO: 似乎没必要预读取和设置仪器参数

        # initialize other necessary parameters
        self.piPulse = {'frequency': 0, 'duration': 0, 'power': 0}
        self.asgChannels = {
            'laser': int(self.ui.comboBoxASGLaser.currentText()),
            'mw': int(self.ui.comboBoxASGMicrowave.currentText()),
            'apd': int(self.ui.comboBoxTaggerAPD.currentText()),
            'tagger': int(self.ui.comboBoxASGTagger.currentText())
        }
        self.taggerChannels = {
            'apd': int(self.ui.comboBoxTaggerAPD.currentText()),
            'asg': int(self.ui.comboBoxTaggerTrigger.currentText())
        }

    def initCharts(self):
        # visualize ASG sequences
        self.seqFigCanvas = FigureCanvas(seq_to_fig(self.sequences))  # TODO: check 是不是只更新 seqFigCanvas 就行了？
        self.layoutSequenceVisualization = QtWidgets.QVBoxLayout(self.ui.widgetSequenceVisualization)
        self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
        self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutSequenceVisualization.setSpacing(0)

        # initialize frequency-domain ODMR chart

        # initialized time-domain ODMR chart
        pass

    def __init__(self):
        super(OdmactorGUI, self).__init__()
        self.ui = odmactor_window.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)

        # initialize other UI components
        self.buildUI()

        # fetch parameters from initial UI
        self.fetchParameters()

        # initial charts
        self.initCharts()

        # initialize instrument instances
        self.initInstruments()
        self.checkInstruments()

        # initialize data variables

        # self.scheduler =
        # self.scheduler =
        self.initPhotonCountChart()

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

    def checkInstruments(self):
        """
        Check whether necessary instruments are ready or not, and show the information on the status bar
        """
        txt = self.labelInstrStatus.text()

        # txt += ': There is no '
        # QtWidgets.QMessageBox.about(self, '', txt)

    def buildUI(self):
        # status bar
        self.labelInstrStatus = QtWidgets.QLabel(self)
        self.labelInstrStatus.setText('Instrument Status:')
        self.ui.statusbar.addWidget(self.labelInstrStatus)

        # progress bar
        self.progressBar = QtWidgets.QProgressBar(self)
        self.progressBar.setMinimumWidth(200)
        self.progressBar.setMinimum(5)
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(50)
        self.progressBar.setFormat('%p%')  # %p%, %v
        self.ui.statusbar.addPermanentWidget(self.progressBar)

        # table widget
        for i in range(self.ui.tableWidgetSequence.rowCount()):
            for j in range(self.ui.tableWidgetSequence.columnCount()):
                item = QtWidgets.QTableWidgetItem(str(0))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.ui.tableWidgetSequence.setItem(i, j, item)
        #########################
        # ASG sequence table
        # self.ui.tableWidgetSequence .... 初始化

    # Photon counting
    def initPhotonCountChart(self):
        """
        Demo
        """
        chart = QChart()
        # chartView = QChartView(self.ui.scrollArea3)
        chartView = self.ui.chartviewPhotonCount
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
            unit = timeUnitDict[self.ui.comboBoxBinwidthUnit.currentText()]
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
