import sys
import os
import json
import time
import threading
import numpy as np
import scipy.constants as C
import TimeTagger as tt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui, QtChart
from PyQt5.QtCore import Qt, pyqtSlot, QThread

import scheduler
import utils
from instrument import ASG, Microwave, Laser, LockInAmplifier
from utils.sequence import seq_to_fig

from ui import odmactor_window

# 继承关系：QGraphicsItem --> QGraphicsObject --> QGraphicsWidget --> QChart --> QPolarChart

timeUnitDict = {'s': 1, 'ms': C.milli, 'ns': C.nano, 'ps': C.pico}
freqUnitDict = {'Hz': 1, 'KHz': C.kilo, 'MHz': C.mega, 'GHz': C.giga}
frequencyDomainModes = ['CW', 'Pulse']
timeDomainModes = ['Ramsey', 'Rabi', 'Relaxation']
schedulerModes = frequencyDomainModes + timeDomainModes


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

        # initialize instrument instances
        self.initInstruments()
        self.checkInstruments()

        # initialize data variables
        self.schedulers = {mode: getattr(scheduler, mode + 'Scheduler') for mode in schedulerModes}
        # self.schedulers = {
        #     'CW': scheduler.CWScheduler(),
        #     'Pulse': scheduler.PulseScheduler(),
        #     'Ramsey': scheduler.RamseyScheduler(),
        #     'Rabi': scheduler.RabiScheduler(),
        #     'Relaxation': scheduler.RelaxationScheduler()
        # }

        # photon count config (tagger counter measurement class)
        self.updatePhotonCountConfig()
        if self.tagger:
            self.counter = tt.Counter(self.tagger, **self.photonCountConfig)
            self.counter.stop()
        else:
            self.counter = None

        # initial charts
        self.initCharts()

    def initInstruments(self):
        """
        Initialize instances representing specific instruments, i.e, Laser, Microwave, ASG, Tagger, etc.
        """
        self.laser = Laser()
        self.asg = ASG()

        if tt.scanTimeTagger():
            self.tagger = tt.createTimeTagger()
        else:
            self.tagger = None

        try:
            self.mw = Microwave()
        except:
            self.mw = None

        try:
            self.lockin = LockInAmplifier()
        except:
            self.lockin = None

    def updatePhotonCountChart(self):
        # x-axis and y-axis
        binwidth_sec = self.photonCountConfig['binwidth'] * C.pico
        self.axisXPhotonCount.setRange(0, binwidth_sec * self.photonCountConfig['n_values'])
        if self.ui.radioButtonPhotonCountRate.isChecked():
            self.axisYPhotonCount.setTitleText("Count rate")
        else:
            self.axisYPhotonCount.setTitleText("Count number")

        while True:
            if self.ui.radioButtonPhotonCountRate.isChecked():
                counts = self.counter.getData().ravel() / self.photonCountConfig['binwidth'] / C.pico
            else:
                counts = self.counter.getData().rave()
            self.seriesPhotonCount.removePoints(0, self.seriesPhotonCount.count())
            # self.chartPhotonCount.removeSeries(self.seriesPhotonCount)
            for i, c in enumerate(counts):
                self.seriesPhotonCount.append(i * binwidth_sec, c)
            time.sleep(0.1)
    # def initCharts(self):
    #     """
    #     Initialize charts of measurement results, i.e., Photon Count, Sequences, Frequency-domain ODMR, Time-domain ODMR
    #     """
    #     # visualize ASG sequences
    #     self.seqFigCanvas = FigureCanvas(seq_to_fig(self.sequences))  # TODO: check 是不是只更新 seqFigCanvas 就行了？
    #     self.layoutSequenceVisualization = QtWidgets.QVBoxLayout(self.ui.widgetSequenceVisualization)
    #     self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
    #     self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
    #     self.layoutSequenceVisualization.setSpacing(0)
    #
    #     # initialize photon count chart
    #     self.chartPhotonCount = QtChart.QChart()
    #     self.ui.chartviewPhotonCount.setChart(self.chartPhotonCount)
    #
    #     # 创建曲线序列
    #     series0 = QtChart.QLineSeries()
    #     series1 = QtChart.QLineSeries()
    #     series0.setName("Sin曲线")
    #     series1.setName("Cos曲线")
    #     self.chartPhotonCount.addSeries(series0)  # 序列添加到图表
    #     self.chartPhotonCount.addSeries(series1)
    #
    #     # 序列添加数值
    #     t = 0
    #     intv = 0.1
    #     pointCount = 100
    #     for i in range(pointCount):
    #         y1 = np.cos(t)
    #
    #         series0.append(t, y1)
    #
    #         y2 = 1.5 * np.sin(t + 20)
    #         series1.append(t, y2)
    #         t = t + intv
    #
    #
    #     ##创建坐标轴
    #     axisX = QtChart.QValueAxis()  # X 轴
    #     axisX.setRange(0, 10)  # 设置坐标轴范围
    #     axisX.setTitleText("time(secs)")  # 标题
    #     axisX.setLabelFormat("%.1f")  # 标签格式
    #     axisX.setTickCount(11)  # 主分隔个数
    #     axisX.setMinorTickCount(4)
    #     # axisX.setGridLineVisible(False)
    #
    #     axisY = QtChart.QValueAxis()  # Y 轴
    #     axisY.setRange(-2, 2)
    #     axisY.setTitleText("value")
    #     axisY.setTickCount(5)
    #     axisY.setMinorTickCount(4)
    #     axisY.setLabelFormat("%.2f")  # 标签格式
    #     # axisY.setGridLineVisible(False)
    #
    #     # 为序列设置坐标轴
    #     self.chartPhotonCount.setAxisX(axisX, series0)  # 为序列设置坐标轴
    #     self.chartPhotonCount.setAxisY(axisY, series0)
    #
    #     self.chartPhotonCount.setAxisX(axisX, series1)  # 为序列设置坐标轴
    #     self.chartPhotonCount.setAxisY(axisY, series1)
    #
    #     # initialize frequency-domain ODMR chart
    #     self.chartODMRFrequency = QtChart.QChart()
    #     self.ui.chartviewODMRFrequency.setChart(self.chartODMRFrequency)
    #
    #     # initialized time-domain ODMR chart
    #     self.chartODMRTime = QtChart.QChart()
    #     self.ui.chartviewODMRTime.setChart(self.chartODMRTime)

    def initCharts(self):
        """
        Initialize charts of measurement results, i.e., Photon Count, Sequences, Frequency-domain ODMR, Time-domain ODMR
        """
        ###################################
        # visualize ASG sequences
        self.seqFigCanvas = FigureCanvas(seq_to_fig(self.sequences))  # TODO: check 是不是只更新 seqFigCanvas 就行了？
        self.layoutSequenceVisualization = QtWidgets.QVBoxLayout(self.ui.widgetSequenceVisualization)
        self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
        self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutSequenceVisualization.setSpacing(0)

        ###################################
        # initialize photon count chart
        self.chartPhotonCount = QtChart.QChart()
        self.ui.chartviewPhotonCount.setChart(self.chartPhotonCount)
        self.seriesPhotonCount = QtChart.QLineSeries()
        self.seriesPhotonCount.setName('Channel {} counting'.format(self.ui.comboBoxTaggerAPD.currentText()))
        self.chartPhotonCount.addSeries(self.seriesPhotonCount)

        self.axisXPhotonCount = QtChart.QValueAxis()  # X axis
        self.axisXPhotonCount.setTitleText('Time (s)')
        self.axisXPhotonCount.setTickCount(11)  # 主分隔个数
        self.axisXPhotonCount.setMinorTickCount(4)  # 次刻度数
        self.axisXPhotonCount.setLabelFormat("%.1f")

        self.axisYPhotonCount = QtChart.QValueAxis()  # Y axis
        self.axisYPhotonCount.setTickCount(5)
        self.axisYPhotonCount.setMinorTickCount(4)
        self.axisYPhotonCount.setLabelFormat("%.2f")  # 标签格式
        # axisY.setGridLineVisible(False)

        # add axis on series
        self.chartPhotonCount.setAxisX(self.axisXPhotonCount, self.seriesPhotonCount)
        self.chartPhotonCount.setAxisY(self.axisYPhotonCount, self.seriesPhotonCount)

        ###################################
        # initialize frequency-domain ODMR chart
        self.chartODMRFrequency = QtChart.QChart()
        self.ui.chartviewODMRFrequency.setChart(self.chartODMRFrequency)

        ###################################
        # initialized time-domain ODMR chart
        self.chartODMRTime = QtChart.QChart()
        self.ui.chartviewODMRTime.setChart(self.chartODMRTime)

    def fetchParameters(self):
        """
        Fetch necessary parameters from UI components and initialize them into properties of the class
        """
        # ASG sequences from table widget
        self.sequences = []  # unit: ns
        self.fetchSequencesfromTableWidget()  # sequences with each element equal to 0 has been set in self.buildUI()
        # self.sequences[0] = [10, 0]
        # self.sequences[-1] = [10, 0]

        # Laser, MW, ASG, tagger counter # TODO: 似乎没必要预读取和设置仪器参数
        # TODO: 似乎不需要 laserConfig MWConfig之类的字段

        # initialize other necessary parameters
        self.updatePiPulse()
        self.updateASGChannels()
        self.updateTaggerChannels()

    def updatePiPulse(self):
        self.piPulse = {
            'frequency': self.ui.doubleSpinBoxMicrowavePiPulseFrequency.value(),
            'duration': self.ui.doubleSpinBoxMicrowavePiPulseDuration.value(),
            'power': self.ui.doubleSpinBoxMicrowavePiPulsePower.value()
        }
        self.piPulse['frequency'] *= freqUnitDict[self.ui.comboBoxMicrowavePiPulseFrequencyUnit.currentText()]
        self.piPulse['duration'] *= timeUnitDict[self.ui.comboBoxMicrowavePiPulseDurationUnit.currentText()]
        if self.ui.comboBoxMicrowavePiPulsePowerUnit.currentText() == 'mW':
            self.piPulse['power'] = utils.mW_to_dBm(self.piPulse['power'])

    def updateASGChannels(self):
        self.asgChannels = {
            'laser': int(self.ui.comboBoxASGLaser.currentText()),
            'mw': int(self.ui.comboBoxASGMicrowave.currentText()),
            'apd': int(self.ui.comboBoxTaggerAPD.currentText()),
            'tagger': int(self.ui.comboBoxASGTagger.currentText())
        }

    def updateTaggerChannels(self):
        self.taggerChannels = {
            'apd': int(self.ui.comboBoxTaggerAPD.currentText()),
            'asg': int(self.ui.comboBoxTaggerTrigger.currentText())
        }

    def updatePhotonCountConfig(self):
        unit = timeUnitDict[self.ui.comboBoxBinwidthUnit.currentText()]
        self.photonCountConfig = {
            'channels': [[int(self.ui.comboBoxTaggerAPD.currentText())]],
            'binwidth': unit * self.ui.spinBoxBinwidth.value() / C.pico,  # unit: ps
            'n_values': self.ui.spinBoxCountNumber.value()
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
        # self.ui.tableWidgetSequence.horizontalHeader().setStyleSheet('QHeaderView::section{background:lightblue;}')
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

        laserStatus = 'ready'  # Laser is not implemented in programs
        asgStatus = 'connected' if self.asg.connect() else 'None'
        taggerStatus = 'connected' if self.tagger else 'None'
        if self.mw is None or self.mw.is_connection_active() is False:
            mwStatus = 'None'
        else:
            mwStatus = 'connected'

        txt += 'Laser: {}, Microwave: {}, ASG: {}, Tagger: {}'.format(laserStatus, mwStatus, asgStatus, taggerStatus)
        self.labelInstrStatus.setText(txt)

    # def updateInstrumentsStatus(self):

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
            self.labelInstrStatus.setText('Laser: started')
        else:
            self.laser.stop()
            self.labelInstrStatus.setText('Laser: stopped')

    @pyqtSlot()
    def on_pushButtonLaserConnect_clicked(self):
        self.laser.connect()
        self.labelInstrStatus.setText('Laser: connected')  # TODO: connect fail还是success呢？？？

    @pyqtSlot()
    def on_pushButtonLaserClose_clicked(self):
        self.laser.close()
        self.labelInstrStatus.setText('Laser: closed')

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
        if self.ui.comboBoxMicrowavePowerUnit.currentText() == 'mW':
            power *= utils.mW_to_dBm(power)
        self.mw.set_power(power)

    @pyqtSlot()
    def on_pushButtonMicrowaveOnOff_clicked(self, checked):
        if checked:
            self.mw.start()
            self.labelInstrStatus.setText('Microwave: started')
        else:
            self.mw.stop()
            self.labelInstrStatus.setText('Microwave: stopped')

    @pyqtSlot()
    def on_pushButtonMicrowaveConnect_clicked(self):
        self.mw.connect()
        self.labelInstrStatus.setText('Microwave: connected')

    @pyqtSlot()
    def on_pushButtonMicrowaveClose_clicked(self):
        self.mw.close()
        self.labelInstrStatus.setText('Microwave: closed')

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

    @pyqtSlot()
    def on_comboBoxASGAPD_valueChanged(self):
        self.asgChannels['apd'] = int(self.ui.comboBoxASGAPD.currentText())

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
        """
        Fetch sequences parameters to generate ASG sequences, load it into ASG and visualize
        """
        # TODO: check 合法的 pulse duration 输入值
        self.odmrSeqConfig = {
            'N': self.ui.spinBoxODMRPeriodNumber.value(),
            'withReference': self.ui.checkBoxODMRWithReference.isChecked(),
            'MicrowaveOnOff': self.ui.checkBoxODMRMicrowaveOnOff.isChecked(),
            'laserInit': self.ui.spinBoxODMRLaserTime.value(),
            'laserMicrowaveInterval': self.ui.spinBoxODMRLaserTime.value(),
            'microwaveTime': self.ui.spinBoxODMRMicrowaveTime.value(),
            'microwaveReadoutInterval': self.ui.spinBoxODMRMicrowaveReadoutInterval.value(),
            'previousReadoutInterval': self.ui.spinBoxODMRPreReadoutInterval.value(),
            'signalReadout': self.ui.spinBoxODMRSignalReadoutTime.value(),
            'signalReferenceInterval': self.ui.spinBoxODMRSignalReferenceInterval.value(),
            'periodInterval': self.ui.spinBoxODMRPeriodInterval.value()
        }

        self.schedulers[self.schedulerMode].set_asg_sequences_ttl(
            laser_ttl=1 if self.ui.checkBoxASGLaserTTL.isChecked() else 0,
            mw_ttl=1 if self.ui.checkBoxASGMicrowaveTTL.isChecked() else 0,
            apd_ttl=1 if self.ui.checkBoxASGAPDTTL.isChecked() else 0,
            tagger_ttl=1 if self.ui.checkBoxASGTaggerTTL.isChecked() else 0,
        )
        self.schedulers[self.schedulerMode].configure_odmr_seq(
            t_init=self.odmrSeqConfig['laserInit'],
            t_mw=self.odmrSeqConfig['microwaveTime'],
            t_read_sig=self.odmrSeqConfig['signalReadout'],
            inter_init_mw=self.odmrSeqConfig['laserMicrowaveInterval'],
            pre_read=self.odmrSeqConfig['previousReadoutInterval'],
            inter_read=self.odmrSeqConfig['signalReferenceInterval'],
            inter_period=self.odmrSeqConfig['periodInterval'],
            N=self.odmrSeqConfig['N'],
        )

        if self.schedulerMode in timeDomainModes:
            self.schedulers[self.schedulerMode].gene_pseudo_detect_seq()
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

        if self.schedulerMode in frequencyDomainModes:
            self.schedulers[self.schedulerMode].set_mw_freqs(freq_start, freq_end, freq_step)
            # 跳转到 frequency-domain panel TODO
            # 更新 chart
            if self.ui.radioButtonODMRFrequencyShowCount.isChecked():
                pass
            else:
                pass
        if self.schedulerMode in timeDomainModes:
            self.schedulers[self.schedulerMode].set_delay_times(time_start, time_end, time_step)
            # 更新 chart
            pass

    @pyqtSlot()
    def on_pushButtonODMRSaveData(self):
        # TODO
        self.counter.getData()
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

    ###########################################
    # Photon count configuration
    ################

    # # ============================
    # N = 1000
    # lockin = LockInAmplifier()
    #
    # # create a figure widget and a plot
    # fig_trace = go.FigureWidget()
    # # fig_trace.add_scatter(x=trace.getIndex(), y=trace.getData()[0])
    # fig_trace.add_scatter(x=range(N), y=lockin.get_data_with_time(num=N))
    #
    # async def update_trace():
    #     """Update the plot every 0.1 s"""
    #     while True:
    #         # fig_trace.data[0].y = trace.getData()[0]
    #         fig_trace.data[0].y = lockin.get_data_with_time(num=N)
    #
    #         await asyncio.sleep(0.1)
    #
    # # If this cell is re-excecuted and there was a previous task, stop it first to avoid a dead daemon
    # try:
    #     task_trace.cancel()
    # except:
    #     pass
    #
    # loop = asyncio.get_event_loop()
    # task_trace = loop.create_task(update_trace())
    #
    # # create a stop button
    # button_trace_stop = Button(description='stop')
    # button_trace_stop.on_click(lambda a: task_trace.cancel())
    #
    # display(fig_trace, button_trace_stop)
    # async def update(self) -> None:

    @pyqtSlot(bool)
    def on_pushButtonPhotonCountOnOff_clicked(self, checked):
        """
        :param checked: if True, reload parameters to start counting; otherwise, stop counting
        """
        self.tagger.setTestSignal(int(self.ui.comboBoxTaggerAPD.currentText()), True)  # TODO: delete this
        if checked:
            self.updatePhotonCountConfig()
            self.counter = tt.Counter(self.tagger, *self.photonCountConfig)
            t = threading.Thread(target=self.updatePhotonCountChart)

            # TODO: 持续返回数据，定时刷新曲线，rate?!
            self.counter.start()
            t.start()
        else:
            try:
                t.wait()
            except:
                pass
            self.counter.stop()

    @pyqtSlot()
    def on_pushButtonPhotonCountRefresh_clicked(self):
        # 清空chart，重新开始
        # self.chartPhotonCount.clearFocus()
        pass

    @pyqtSlot()
    def on_pushButtonPhotonCountSaveData_clicked(self):
        """
        Save data in form of JSON file in default
        """
        pass

    @pyqtSlot()
    def on_pushButtonTaggerConnect_clicked(self):
        """
        Re-connect Time Tagger
        """
        try:
            self.tagger.getSerial()
        except:
            self.tagger = tt.createTimeTagger()
        self.labelInstrStatus.setText('Tagger: connected')

    @pyqtSlot()
    def on_pushButtonTaggerClose_clicked(self):
        """
        Close connection with Tagger
        """
        if isinstance(self.tagger, tt.TimeTagger):
            tt.freeTimeTagger(self.tagger)
        self.labelInstrStatus.setText('Tagger: closed')

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
            self.labelInstrStatus.setText('ASG: connect success')
        else:
            self.labelInstrStatus.setText('ASG: connect fail')

    @pyqtSlot()
    def on_pushButtonASGClose_clicked(self):
        self.asg.close()
        self.labelInstrStatus.setText('ASG: closed')

    @pyqtSlot()
    def on_pushButtonASGStart_clicked(self):
        self.asg.start()
        self.labelInstrStatus.setText('ASG: started')

    @pyqtSlot()
    def on_pushButtonASGStop_clicked(self):
        self.asg.stop()
        self.labelInstrStatus.setText('ASG: stopped')

    @pyqtSlot()
    def on_pushButtonASGLoad_clicked(self):
        # 1) fetch sequences data
        self.fetchSequencesfromTableWidget()
        self.asg.load_data(self.sequences)
        # 用写好的函数 scheduler中 ........................
        # 2) load sequences into current scheduler
        self.schedulers[self.schedulerMode].config_sequences(self.sequences)
        # 3) visualize seuqnces
        self.updateSequenceChart()

    @pyqtSlot()
    def on_pushButtonASGClear_clicked(self):
        self.ui.tableWidgetSequence.clear()

    @pyqtSlot()
    def on_pushButtonASGAdd_clicked(self):
        # self.ui.tableWidgetSequence add 2 columns
        rowCount = self.ui.tableWidgetSequence.rowCount()
        originColumnCount = self.ui.tableWidgetSequence.columnCount()
        self.ui.tableWidgetSequence.setColumnCount(originColumnCount + 2)
        self.ui.tableWidgetSequence.setHorizontalHeaderLabels(['High', 'Low'] * int(originColumnCount / 2 + 1))
        for i in range(rowCount):
            for j in range(originColumnCount, originColumnCount + 2):
                item = QtWidgets.QTableWidgetItem(str(0))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.ui.tableWidgetSequence.setItem(i, j, item)

    ###########################################
    # Frequency-domain measurement panel (CW, Pulse)
    ################
    @pyqtSlot()
    def on_pushButtonODMRFrequencyInterrupt_clicked(self):
        # interrupt
        # 要不要开辟新的 thread TODO
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

# TODO 1: 先画图于 odmr spectrum chart，在做 real-time 的 chart
# TODO 2: 先跑单线程的，在考虑为 scheduler 开辟新的 thread
