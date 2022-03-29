import sys
import os
import json
import numpy as np
import scipy.constants as C
import TimeTagger as tt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui, QtChart
from PyQt5.QtCore import Qt, pyqtSlot

import scheduler
import utils
from instrument import ASG, Microwave, Laser
from utils.sequence import seq_to_fig

from ui import odmactor_window

# 继承关系：QGraphicsItem --> QGraphicsObject --> QGraphicsWidget --> QChart --> QPolarChart

timeUnitDict = {'s': 1, 'ms': C.milli, 'ns': C.nano, 'ps': C.pico}
freqUnitDict = {'Hz': 1, 'KHz': C.kilo, 'MHz': C.mega, 'GHz': C.giga}
schedulerModes = ['CW', 'Pulse', 'Ramsey', 'Rabi', 'Relaxation']


class OdmactorGUI(QtWidgets.QMainWindow):
    """
    Mainwindow configuration of Odmactor
    """

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
        # self.schedulers = {
        #     'CW': scheduler.CWScheduler(),
        #     'Pulse': scheduler.PulseScheduler(),
        #     'Ramsey': scheduler.RamseyScheduler(),
        #     'Rabi': scheduler.RabiScheduler(),
        #     'Relaxation': scheduler.RelaxationScheduler()
        # }
        # self.scheduler =
        # self.scheduler =

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

    def initInstruments(self):
        """
        Initialize instances representing specific instruments, i.e, Laser, Microwave, ASG, Tagger, etc.
        """
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

    def initCharts(self):
        """
        Initialize charts of measurement results, i.e., Photon Count, Sequences, Frequency-domain ODMR, Time-domain ODMR
        """
        # visualize ASG sequences
        self.seqFigCanvas = FigureCanvas(seq_to_fig(self.sequences))  # TODO: check 是不是只更新 seqFigCanvas 就行了？
        self.layoutSequenceVisualization = QtWidgets.QVBoxLayout(self.ui.widgetSequenceVisualization)
        self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
        self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutSequenceVisualization.setSpacing(0)

        # initialize photon count chart
        self.chartPhotonCount = QtChart.QChart()
        self.ui.chartviewPhotonCount.setChart(self.chartPhotonCount)

        # 创建曲线序列
        series0 = QtChart.QLineSeries()
        series1 = QtChart.QLineSeries()
        series0.setName("Sin曲线")
        series1.setName("Cos曲线")
        self.chartPhotonCount.addSeries(series0)  # 序列添加到图表
        self.chartPhotonCount.addSeries(series1)

        # 序列添加数值
        t = 0
        intv = 0.1
        pointCount = 100
        for i in range(pointCount):
            y1 = np.cos(t)
            series0.append(t, y1)
            y2 = 1.5 * np.sin(t + 20)
            series1.append(t, y2)
            t = t + intv

        ##创建坐标轴
        axisX = QtChart.QValueAxis()  # X 轴
        axisX.setRange(0, 10)  # 设置坐标轴范围
        axisX.setTitleText("time(secs)")  # 标题
        axisX.setLabelFormat("%.1f")  # 标签格式
        axisX.setTickCount(11)  # 主分隔个数
        axisX.setMinorTickCount(4)
        # axisX.setGridLineVisible(False)

        axisY = QtChart.QValueAxis()  # Y 轴
        axisY.setRange(-2, 2)
        axisY.setTitleText("value")
        axisY.setTickCount(5)
        axisY.setMinorTickCount(4)
        axisY.setLabelFormat("%.2f")  # 标签格式
        # axisY.setGridLineVisible(False)

        # 为序列设置坐标轴
        self.chartPhotonCount.setAxisX(axisX, series0)  # 为序列设置坐标轴
        self.chartPhotonCount.setAxisY(axisY, series0)

        self.chartPhotonCount.setAxisX(axisX, series1)  # 为序列设置坐标轴
        self.chartPhotonCount.setAxisY(axisY, series1)

        # initialize frequency-domain ODMR chart
        self.chartODMRFrequency = QtChart.QChart()
        self.ui.chartviewODMRFrequency.setChart(self.chartODMRFrequency)

        # initialized time-domain ODMR chart
        self.chartODMRTime = QtChart.QChart()
        self.ui.chartviewODMRTime.setChart(self.chartODMRTime)

    def fetchParameters(self):
        """
        Fetch necessary parameters from UI components and initialize them into properties of the class
        """

        # ASG sequences from table widget

        self.sequences = []  # unit: ns
        self.fetchSequencesfromTableWidget()
        self.sequences[0] = [10, 0]
        self.sequences[-1] = [10, 0]

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

    def checkInstruments(self):
        """
        Check whether necessary instruments are ready or not, and show the information on the status bar
        """
        txt = self.labelInstrStatus.text()

        # txt += ': There is no '
        # QtWidgets.QMessageBox.about(self, '', txt)
        pass

    ###########################################
    # Menu bar
    ################
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

    ###########################################
    # Generic instruments configuration
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

    @pyqtSlot()
    def on_pushButtonLaserConnect_clicked(self):
        self.laser.connect()

    @pyqtSlot()
    def on_pushButtonLaserClose_clicked(self):
        self.laser.close()

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

    @pyqtSlot()
    def on_pushButtonMicrowaveConnect_clicked(self):
        self.mw.connect()

    @pyqtSlot()
    def on_pushButtonMicrowaveClose_clicked(self):
        self.mw.close()

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

    ###########################################
    # ODMR measurement parameters
    ################
    # Exclusive groups
    @pyqtSlot(bool)
    def on_groupBoxODMRFrequency_clicked(self, checked):
        if checked:
            self.ui.groupBoxODMRTime.setChecked(False)

    @pyqtSlot(bool)
    def on_groupBoxODMRTime_clicked(self, checked):
        if checked:
            self.ui.groupBoxODMRFrequency.setChecked(False)

    # Operation
    @pyqtSlot()
    def on_pushButtonODMRLoadSequences_clicked(self):
        # 读取 sequences data 并且 load
        # TODO: check 合法的 pulse duration 输入值
        self.odmrSeqConfig = {
            'N': self.ui.spinBoxODMRPeriodNumber.value(),
            'withReference': self.ui.checkBoxODMRWithReference.isChecked(),
            'MicrowaveOnOff': self.ui.checkBoxODMRMicrowaveOnOff.isChecked(),
            'laserInit': self.ui.spinBoxODMRLaserTime.value(),
            'laserMicrowaveInterval': self.ui.spinBoxODMRLaserTime.value(),
            'microwaveInit': self.ui.spinBoxODMRMicrowaveTime.value(),
            'microwaveReadoutInterval': self.ui.spinBoxODMRMicrowaveReadoutInterval.value(),
            'previousReadoutInterval': self.ui.spinBoxODMRPreReadoutInterval.value(),
            'signalReadout': self.ui.spinBoxODMRSignalReadoutTime.value(),
            'signalReferenceInterval': self.ui.spinBoxODMRSignalReferenceInterval.value(),
            'periodInterval': self.ui.spinBoxODMRPeriodInterval.value()
        }
        # self.scheduler.configure_odmr_seq(
        #     # TODO: check data
        # )

        ls = [
            self.ui.checkBoxASGLaserTTL.isChecked(),
            self.ui.checkBoxASGMicrowaveTTL.isChecked(),
            self.ui.checkBoxASGAPDTTL.isChecked(),
            self.ui.checkBoxASGTaggerTTL.isChecked(),
        ]  # TODO: 放在哪里合适呢

        self.sequences = self.schedulers[self.schedulerMode].sequences
        self.feedSequencesToTabkeWidget()
        self.updateSequenceChart()
        self.asg.load_data(self.sequences)

    @pyqtSlot()
    def on_pushButtonODMRStartDetecting_clicked(self):
        # TODO
        # run 完要不要 close？
        # start button 按下去时候 freqs、times才有用

        # frequencies for scanning

        unit_freq = freqUnitDict[self.ui.comboBoxODMRFrequencyUnit.currentText()]
        freq_start = self.ui.doubleSpinBoxODMRFrequencyStart.value() * unit_freq
        freq_end = self.ui.doubleSpinBoxODMRFrequencyEnd.value() * unit_freq
        freq_step = self.ui.doubleSpinBoxODMRFrequencyStep.value() * unit_freq

        # times for scanning
        unit_time = timeUnitDict[self.ui.comboBoxODMRTimeUnit.currentText()]
        time_start = self.ui.doubleSpinBoxODMRTimeStart.value() * unit_time
        time_end = self.ui.doubleSpinBoxODMRTimeEnd.value() * unit_time
        time_step = self.ui.doubleSpinBoxODMRTimeStep.value() * unit_time

        self.scheduler.run_scanning()

        if self.schedulerMode in ['CW', 'Pulse']:
            # 跳转到 frequency-domain panel TODO
            # 更新 chart
            if self.ui.radioButtonODMRFrequencyShowCount.isChecked():
                pass
            else:
                pass
        else:
            # 更新 chart
            pass

    @pyqtSlot()
    def on_pushButtonODMRSaveData(self):
        # TODO
        pass

    # ODMR cheduler mode
    def connectScheduler(self, mode: str):
        """
        Allocate hardware resources to one specific scheduler
        """
        self.schedulerMode = mode
        if mode not in schedulerModes:
            raise ValueError('{} is not a supported scheduler type'.format(mode))
        for k, scheduler in self.schedulers.keys():
            if k != mode:
                scheduler.close()
        self.schedulers[mode].connect()

    @pyqtSlot()
    def on_radioButtonODMRCW_clicked(self):
        self.connectScheduler('CW')

    @pyqtSlot()
    def on_radioButtonODMRPulse_clicked(self):
        self.connectScheduler('Pulse')

    @pyqtSlot()
    def on_radioButtonODMRRamsey_clicked(self):
        self.connectScheduler('Ramsey')

    @pyqtSlot()
    def on_radioButtonODMRRabi_clicked(self):
        self.connectScheduler('Rabi')

    @pyqtSlot()
    def on_radioButtonODMRRelaxation_clicked(self):
        self.connectScheduler('Relaxation')

        ########################

    # Photon count parameter updating

    # @pyqtSlot()
    # def on_spinBoxBinwidth_valueChange
    #

    ###########################################
    # Photon count configuration
    ################
    @pyqtSlot(bool)
    def on_pushButtonPhotonCountOnOff_clicked(self, checked):
        """
        :param checked: if True, reload parameters to start counting; otherwise, stop counting
        """
        # TODO: on/off 切换状态时候 才读取 binwidth、count number，
        if checked:
            unit = timeUnitDict[self.ui.comboBoxBinwidthUnit.currentText()]
            self.photonCountConfig['channels'] = [int(self.ui.comboBoxASGAPD.currentText())]  # TODO 设置几个comboBox 数据互斥
            self.photonCountConfig['binwidth'] = unit * self.ui.spinBoxBinwidth.value() / C.pico
            self.photonCountConfig['n_values'] = self.ui.spinBoxCountNumber.value()
            self.counter = tt.Counter(self.tagger, *self.photonCountConfig)
            self.counter.start()

            # TODO: 持续返回数据，定时刷新曲线，rate?!
            if self.ui.radioButtonPhotonCountRate.isChecked():
                self.counter.getDataNormalized()
            else:
                self.counter.getData()
        else:
            self.counter.stop()

    @pyqtSlot()
    def on_pushButtonPhotonCountRefresh_clicked(self):
        # 清空chart
        pass

    @pyqtSlot()
    def on_pushButtonPhotonCountSaveData_clicked(self):
        """
        Save data in form of JSON file in default
        """
        pass

    ###########################################
    # ASG sequences configuration
    ################
    def fetchSequencesfromTableWidget(self):
        self.sequences.clear()
        for i in range(self.ui.tableWidgetSequence.rowCount()):
            self.sequences.append([])
            for j in range(self.ui.tableWidgetSequence.columnCount()):
                self.sequences[i].append(int(self.ui.tableWidgetSequence.item(i, j).text()))

    def feedSequencesToTabkeWidget(self):
        for i, seq in enumerate(self.sequences):
            for j, val in enumerate(seq):
                self.ui.tableWidgetSequence.item(i, j).setText(str(val))

    def updateSequenceChart(self):
        self.seqFigCanvas = FigureCanvas(
            seq_to_fig(self.sequences))  # TODO: check 是不是只更新 seqFigCanvas 就行了？,因为 seqFigCanvas 已经被 add 到 layout里面了

    @pyqtSlot()
    def on_pushButtonASGOpen_clicked(self):
        c = self.asg.connect()
        if c == 1:
            print('ASG连接成功')
        else:
            print('ASG连接失败！')

    @pyqtSlot()
    def on_pushButtonASGClose_clicked(self):
        self.asg.close()

    @pyqtSlot()
    def on_pushButtonASGStart_clicked(self):
        self.asg.start()

    @pyqtSlot()
    def on_pushButtonASGStop_clicked(self):
        self.asg.stop()

    @pyqtSlot()
    def on_pushButtonASGLoad_clicked(self):
        # 1) seq = .........
        self.fetchSequencesfromTableWidget()
        self.asg.load_data(self.sequences)
        # self.asg.download_ASG_pulse_data(seq)
        # 用写好的函数 scheduler中
        # 2) 可视化
        self.updateSequenceChart()

    @pyqtSlot()
    def on_pushButtonASGClear_clicked(self):
        self.ui.tableWidgetSequence.clear()

    @pyqtSlot()
    def on_pushButtonASGAdd_clicked(self):
        # self.ui.tableWidgetSequence add 2 columns
        pass

    ###########################################
    # Frequency-domain measurement panel (CW, Pulse)
    ################
    @pyqtSlot()
    def on_pushButtonODMRFrequencyInterrupt_clicked(self):
        # interrupt
        pass

    @pyqtSlot()
    def on_pushButtonODMRFrequencyFit_clicked(self):
        # 必须等run结束曲线画出来后拟合
        pass

    ###########################################
    # Time-domain measurement panel (Ramsey, Rabi, Relaxation)
    ################
    @pyqtSlot()
    def on_pushButtonODMRTimeInterrupt_clicked(self):
        # interrupt
        pass

    @pyqtSlot()
    def on_pushButtonODMRTimeFit_clicked(self):
        # 必须等run结束曲线画出来后拟合
        pass
