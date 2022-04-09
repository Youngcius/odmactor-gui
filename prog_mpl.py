import sys
import os
import json
import time
import datetime
import asyncio
import threading

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as C
import TimeTagger as tt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui, QtChart
from PyQt5.QtCore import Qt, pyqtSlot, QThread, QTimer

import scheduler
import utils
from instrument import ASG, Microwave, Laser, LockInAmplifier
from utils.sequence import seq_to_fig

from ui import odmactor_window_mpl

# 继承关系：QGraphicsItem --> QGraphicsObject --> QGraphicsWidget --> QChart --> QPolarChart

timeUnitDict = {'s': 1, 'ms': C.milli, 'us': C.micro, 'ns': C.nano, 'ps': C.pico}
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
        self.ui = odmactor_window_mpl.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)

        # initialize other UI components
        self.buildUI()

        # initialize instrument instances
        self.initInstruments()
        self.checkInstruments()

        # fetch parameters from initial UI
        self.fetchParameters()

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
        Initialize instances representing specific instruments, i.e, Laser, Microwave, ASG, Tagger, Lock-in, etc.
        """
        self.laser = Laser()
        self.asg = ASG()
        print(self.asg.connect(), 'asg 初始化了')

        if tt.scanTimeTagger():
            self.tagger = tt.createTimeTagger()
        else:
            self.tagger = None

        try:
            self.mw = Microwave()
        except:
            self.mw = None

        # try:
        #     self.lockin = LockInAmplifier()
        # except:
        #     self.lockin = None
        self.lockin = None

    def releaseInstruments(self):
        """
        Close all instruments, i.e., Laser, Microwave, ASG, Tagger, Lock-in, etc.
        """
        self.laser.close()
        self.mw.close()
        self.asg.close()
        if isinstance(self.tagger, tt.TimeTagger):
            tt.freeTimeTagger(self.tagger)

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
        self.countFigCanvas = FigureCanvas(plt.figure())
        self.layoutPhotonCountVisualization = QtWidgets.QVBoxLayout(self.ui.widgetPhotonCountVisualization)
        self.layoutPhotonCountVisualization.addWidget(self.countFigCanvas)
        self.layoutPhotonCountVisualization.setContentsMargins(0,0,0,0)
        self.layoutPhotonCountVisualization.setSpacing(0)
        self.axesPhotonCount= self.countFigCanvas.figure.subplots()
        self.axesPhotonCount.set_xlabel('Time point')
        self.timerPhotonCount = self.countFigCanvas.new_timer(100, [(self.updatePhotonCountChart, (), {})])
        #
        #
        # self._static_ax = static_canvas.figure.subplots()
        # f, ax = plt.subplots()
        # ax.set_xalbel('')
        # t = np.linspace(0, 10, 501)
        # self._static_ax.plot(t, np.tan(t), ".")
        #
        # self._dynamic_ax = dynamic_canvas.figure.subplots()
        # self._timer = dynamic_canvas.new_timer(100, [(self._update_canvas, (), {})])
        # self._timer.start()

        #
        # ###################################
        # # initialize photon count chart
        # # data field: chart, series, axisX, axisY
        # self.chartPhotonCount = QtChart.QChart()
        # self.ui.chartviewPhotonCount.setChart(self.chartPhotonCount)
        # # self.ui.chartviewPhotonCount.addWidget
        # self.seriesPhotonCount = QtChart.QLineSeries()
        # self.scatterPhotonCount = QtChart.QScatterSeries()
        # self.seriesPhotonCount.setName('Channel {} counting'.format(self.ui.comboBoxTaggerAPD.currentText()))
        # self.chartPhotonCount.addSeries(self.seriesPhotonCount)
        # self.chartPhotonCount.addSeries(self.scatterPhotonCount)
        #
        # self.axisXPhotonCount = QtChart.QValueAxis()  # X axis: Time
        # self.axisXPhotonCount.setTitleText('Time')
        # self.axisXPhotonCount.setTickCount(11)  # 主分隔个数
        # self.axisXPhotonCount.setMinorTickCount(4)  # 次刻度数
        # self.axisXPhotonCount.setLabelFormat("%.1f")
        #
        # self.axisYPhotonCount = QtChart.QValueAxis()  # Y axis: count or count rate
        # self.axisYPhotonCount.setTickCount(5)
        # self.axisYPhotonCount.setMinorTickCount(4)
        # self.axisYPhotonCount.setLabelFormat("%.2f")  # 标签格式
        # # axisY.setGridLineVisible(False)
        #
        # # add axis on series
        # self.chartPhotonCount.setAxisX(self.axisXPhotonCount, self.seriesPhotonCount)
        # self.chartPhotonCount.setAxisY(self.axisYPhotonCount, self.seriesPhotonCount)
        #
        # self.chartPhotonCount.setAxisX(self.axisXPhotonCount, self.scatterPhotonCount)
        # self.chartPhotonCount.setAxisY(self.axisYPhotonCount, self.scatterPhotonCount)
        #
        # # set timer to update chart
        # self.timerPhotonCount = QTimer()
        # self.timerPhotonCount.timeout.connect(self.updatePhotonCountChart)

        ###################################
        # initialize frequency-domain ODMR chart
        self.chartODMRFrequency = QtChart.QChart()
        self.ui.chartviewODMRFrequency.setChart(self.chartODMRFrequency)
        self.seriesODMRFrequency = QtChart.QLineSeries()  # 有参考时就返回对比度，否则就是计数
        self.chartODMRFrequency.addSeries(self.seriesODMRFrequency)

        self.axisXODMRFrequency = QtChart.QValueAxis()  # X axis: frequency
        self.axisXODMRFrequency.setTitleText('Frequency')
        self.axisXODMRFrequency.setTickCount(11)
        self.axisXODMRFrequency.setMinorTickCount(4)
        self.axisXODMRFrequency.setLabelFormat("%.1f")

        self.axisYODMRFrequency = QtChart.QValueAxis()  # Y axis: count or contrast
        self.axisYODMRFrequency.setTickCount(5)
        self.axisYODMRFrequency.setMinorTickCount(4)
        self.axisYODMRFrequency.setLabelFormat("%.2f")  # 标签格式

        self.chartODMRFrequency.setAxisX(self.axisXODMRFrequency, self.seriesODMRFrequency)
        self.chartODMRFrequency.setAxisY(self.axisYODMRFrequency, self.seriesODMRFrequency)

        self.timerODMRFrequency = QTimer()
        self.timerODMRFrequency.timeout.connect(self.updateODMRFrequencyChart)

        ###################################
        # initialized time-domain ODMR chart
        self.chartODMRTime = QtChart.QChart()
        self.ui.chartviewODMRTime.setChart(self.chartODMRTime)
        self.seriesODMRTime = QtChart.QLineSeries()
        self.chartODMRTime.addSeries(self.seriesODMRTime)

        self.axisXODMRTime = QtChart.QValueAxis()  # X axis: time
        self.axisXODMRTime.setTitleText('Time')
        self.axisXODMRTime.setTickCount(11)
        self.axisXODMRTime.setMinorTickCount(4)
        self.axisXODMRTime.setLabelFormat("%.1f")

        self.axisYODMRTime = QtChart.QValueAxis()  # T axis
        self.axisYODMRTime.setTickCount(5)
        self.axisYODMRTime.setMinorTickCount(4)
        self.axisYODMRTime.setLabelFormat("%.2f")  # 标签格式

        self.chartODMRTime.setAxisX(self.axisXODMRTime, self.seriesODMRTime)
        self.chartODMRTime.setAxisY(self.axisYODMRTime, self.seriesODMRTime)

        self.timerODMRTime = QTimer()
        self.timerODMRTime.timeout.connect(self.updateODMRTimeChart)

    def fetchParameters(self):
        """
        Fetch necessary parameters from UI components and initialize them into properties of the class
        """
        # ASG sequences from table widget
        self.sequences = []  # unit: ns
        self.fetchSequencesfromTableWidget()  # sequences with each element equal to 0 has been set in self.buildUI()
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
            'channels': [int(self.ui.comboBoxTaggerAPD.currentText())],
            'binwidth': int(unit * self.ui.spinBoxBinwidth.value() / C.pico),  # unit: ps
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
        # self.progressBar.setMinimum(5)
        # self.progressBar.setValue(101)
        print(self.progressBar.maximum(), self.progressBar.minimum(), self.progressBar.value())
        # self.progressBar.setMaximum(50)
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
        if isinstance(self.mw, Microwave) and self.mw.connect():
            if checked:
                self.mw.start()
                self.labelInstrStatus.setText('Microwave: started')
            else:
                self.mw.stop()
                self.labelInstrStatus.setText('Microwave: stopped')
        else:
            self.labelInstrStatus.setText(color_str('Microwave: not connected'))

    @pyqtSlot()
    def on_pushButtonMicrowaveConnect_clicked(self):
        if isinstance(self.mw, Microwave):
            if self.mw.connect():
                self.labelInstrStatus.setText('Microwave: connected')
            else:
                self.labelInstrStatus.setText(color_str('Microwave: connect fail'))
        else:
            try:
                self.mw = Microwave()
                self.labelInstrStatus.setText('Microwave: connected')
            except:
                self.labelInstrStatus.setText(color_str('Microwave: connect fail'))

    @pyqtSlot()
    def on_pushButtonMicrowaveClose_clicked(self):
        if isinstance(self.mw, Microwave):
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

    def loop_demo(self):
        self.i = 0
        while True:
            time.sleep(1)
            self.i += 1
            print(self.i)

    @pyqtSlot()
    def on_pushButtonODMRStartDetecting_clicked(self):
        # 微波参数 --> 设置序列 ------> 频率范围 --> counting setting
        # TODO: 不同 update pi pulse, asg channels 和 tagger cannels
        if self.ui.groupBoxODMRFrequency.isChecked():
            self.startFrequencyDomainDetecting()
        else:
            self.startTimeDomainDetecting()

    def startFrequencyDomainDetecting(self):
        """
        Start frequency-domain ODMR detecting experiments, i.e., CW or Pulse
        """
        # configure parameters
        self.seriesODMRFrequency.removePoints(0, self.seriesODMRFrequency.count())
        self.seriesODMRFrequency.setName('{} Spectrum'.format(self.schedulerMode))

        # frequencies for scanning
        unit_freq = freqUnitDict[self.ui.comboBoxODMRFrequencyUnit.currentText()]
        freq_start = self.ui.doubleSpinBoxODMRFrequencyStart.value() * unit_freq
        freq_end = self.ui.doubleSpinBoxODMRFrequencyEnd.value() * unit_freq
        freq_step = self.ui.doubleSpinBoxODMRFrequencyStep.value() * unit_freq
        self.schedulers[self.schedulerMode].set_mw_freqs(freq_start, freq_end, freq_step)
        self.progressBar.setMaximum(len(self.schedulers[self.schedulerMode].frequencies))

        self.schedulers[self.schedulerMode].configure_odmr_seq()
        self.schedulers[self.schedulerMode].configure_tagger_counting(
            apd_channel=self.taggerChannels['apd'],
            asg_channel=self.taggerChannels['asg'],
            reader='counter' if self.schedulerMode == 'CW' else 'cmb'
        )

        # freqs = self.schedulers[self.schedulerMode].frequencies

        # conduct ODMR scheduling and update real-time chart
        self.releaseInstruments()
        self.schedulers[self.schedulerMode].connect()
        print('成功！')

        t = threading.Thread(target=self.schedulers[self.schedulerMode].run_scanning)  # TODO: 先连接
        t.start()
        self.timerODMRFrequency.start(1000)
        t.join()
        self.timerODMRFrequency.stop()
        self.progressBar.setValue(-1)
        # self.schedulers[self.schedulerMode].close()

    def startTimeDomainDetecting(self):
        """
        Start time-domain ODMR detecting experiments, i.e., Ramsey, Rabi, Relaxation
        """
        # configure chart parameters
        self.seriesODMRTime.removePoints(0, self.seriesODMRTime.count())
        self.seriesODMRFrequency.setName('{} Result'.format(self.schedulerMode))

        # configure pi pulse
        self.schedulers[self.schedulerMode].pi_pulse['freqs'] = self.piPulse['frequency']
        self.schedulers[self.schedulerMode].pi_pulse['power'] = self.piPulse['power']
        self.schedulers[self.schedulerMode].pi_pulse['time'] = self.piPulse['duration']

        # times for scanning
        unit_time = timeUnitDict[self.ui.comboBoxODMRTimeUnit.currentText()]
        time_start = self.ui.doubleSpinBoxODMRTimeStart.value() * unit_time
        time_end = self.ui.doubleSpinBoxODMRTimeEnd.value() * unit_time
        time_step = self.ui.doubleSpinBoxODMRTimeStep.value() * unit_time
        self.schedulers[self.schedulerMode].set_delay_times(time_start, time_end, time_step)
        self.progressBar.setMaximum(len(self.schedulers[self.schedulerMode].times))

        # conduct ODMR scheduling and update real-time chart
        t = threading.Thread(target=self.schedulers[self.schedulerMode].run_scanning)
        t.start()
        self.timerODMRTime.start(1000)
        t.join()
        self.timerODMRTime.stop()
        self.progressBar.setValue(-1)

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
    @pyqtSlot(bool)
    def on_pushButtonPhotonCountOnOff_clicked(self, checked):
        """
        :param checked: if True, reload parameters to start counting; otherwise, stop counting
        """
        self.tagger.setTestSignal(int(self.ui.comboBoxTaggerAPD.currentText()), True)  # TODO: delete this
        self.updatePhotonCountConfig()
        try:
            self.counter = tt.Counter(self.tagger, **self.photonCountConfig)
        except:
            self.labelInstrStatus.setText(color_str('No Time Tagger to detect photons'))

        # self.axisXPhotonCount.setTitleText(
        #     'Time ({} {})'.format(self.ui.spinBoxBinwidth.value(), self.ui.comboBoxBinwidthUnit.currentText()))
        # self.axisXPhotonCount.setRange(0, self.photonCountConfig['n_values'])
        # if self.ui.radioButtonPhotonCountRate.isChecked():
        #     self.axisYPhotonCount.setTitleText("Count rate")
        # else:
        #     self.axisYPhotonCount.setTitleText("Count number")

        if checked:
            self.counter.start()
            self.timerPhotonCount.start(100)
        else:
            self.counter.stop()
            self.timerPhotonCount.stop()

    @pyqtSlot()
    def on_pushButtonPhotonCountRefresh_clicked(self):
        """
        Clear series data of the chart, restart counting
        """
        self.seriesPhotonCount.removePoints(0, self.seriesPhotonCount.count())
        try:
            self.counter.clear()
        except:
            self.labelInstrStatus.setText(color_str('No Time Tagger to detect photons'))

    @pyqtSlot()
    def on_pushButtonPhotonCountSaveData_clicked(self):
        """
        Save data in form of JSON file in default
        """
        # 暂时只支持一个通道，保存的数据是单个字典 --> 单个 JavaScript object --> JSON file
        timestamp = datetime.datetime.now()
        counts = self.counter.getData().ravel()
        data = {
            'channel': self.photonCountConfig['channels'][0],
            'time': (self.counter.getIndex() * C.pico).tolist(),
            'count': counts.tolist(),
            'count rate (1/s)': (counts / self.photonCountConfig['binwidth'] / C.pico).tolist(),
            'timestamp': str(timestamp),
        }
        fname = 'odmactor-counts_' + timestamp.strftime('%Y-%m-%d_%H-%M-%S') + '.json'
        fname = os.path.join(os.path.expanduser('~'), 'Downloads', fname)  # 暂时只考虑 windows 的文件路径
        with open(fname, 'w') as f:
            json.dump(data, f)
        self.labelInstrStatus.setText('File has been saved in {}'.format(fname))

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
        if self.asg is not None:
            self.sequences = self.asg.normalize_data(self.sequences)

    def feedSequencesToTabkeWidget(self):
        for i, seq in enumerate(self.sequences):
            for j, val in enumerate(seq):
                self.ui.tableWidgetSequence.item(i, j).setText(str(val))

    def updateSequenceChart(self):
        """
        Update sequences chart and add it to the layout widget
        """
        self.layoutSequenceVisualization.removeWidget(self.seqFigCanvas)
        self.seqFigCanvas = FigureCanvas(seq_to_fig(self.sequences))
        self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
        self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutSequenceVisualization.setSpacing(0)

    def updatePhotonCountChart(self):



        # # ======================================
        # self.seriesPhotonCount.removePoints(0, self.seriesPhotonCount.count())
        # self.scatterPhotonCount.removePoints(0, self.scatterPhotonCount.count())
        #
        # counts = self.counter.getData().ravel()
        # if self.ui.radioButtonPhotonCountRate.isChecked():
        #     counts = counts / self.photonCountConfig['binwidth'] / C.pico
        # cmin, cmax = min(counts), max(counts)
        # delta = cmax - cmin
        # self.axisYPhotonCount.setRange(cmin - 0.05 * delta, cmax + 0.05 * delta)
        #
        # for i, c in enumerate(counts):
        #     self.seriesPhotonCount.append(i, c)
        #     self.scatterPhotonCount.append(i,c)


        # # ======================================
        self.axesPhotonCount.clear()
        counts = self.counter.getData().ravel()
        if self.ui.radioButtonPhotonCountRate.isChecked():
            counts = counts / self.photonCountConfig['binwidth'] / C.pico
        self.axesPhotonCount.plot(counts)
        self.axesPhotonCount.figure.canvas.draw()

        # self._dynamic_ax.clear()
        # t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
        # self._dynamic_ax.plot(t, np.sin(t + time.time()) + np.random.rand(len(t)))
        # self._dynamic_ax.figure.canvas.draw()


    def updateODMRFrequencyChart(self):
        """
        Update frequency-domain ODMR results periodically
        """
        # update series

        # update progress bar
        freqs = self.schedulers[self.schedulerMode].frequencies
        cur_freq = self.schedulers[self.schedulerMode].cur_freq
        self.progressBar.setValue(freqs.index(cur_freq) + 1)

    def updateODMRTimeChart(self):
        """
        Update time-domain ODMR results periodically
        """
        # update series

        # update progress bar
        times = self.schedulers[self.schedulerMode].times
        cur_time = self.schedulers[self.schedulerMode].cur_time
        self.progressBar.setValue(times.index(cur_time) + 1)

    @pyqtSlot()
    def on_pushButtonASGOpen_clicked(self):
        c = self.asg.connect()
        if c == 1:
            self.labelInstrStatus.setText('ASG: connect success')
        else:
            self.labelInstrStatus.setText(color_str('ASG: connect fail'))

    @pyqtSlot()
    def on_pushButtonASGClose_clicked(self):
        if self.asg.close():
            self.labelInstrStatus.setText('ASG: closed')
        else:
            self.labelInstrStatus.setText(color_str('ASG: close fail'))

    @pyqtSlot()
    def on_pushButtonASGStart_clicked(self):
        if self.asg.start():
            self.labelInstrStatus.setText('ASG: started')
        else:
            self.labelInstrStatus.setText(color_str('ASG: start fail'))

    @pyqtSlot()
    def on_pushButtonASGStop_clicked(self):
        if self.asg.stop():
            self.labelInstrStatus.setText('ASG: stopped')
        else:
            self.labelInstrStatus.setText(color_str('ASG: stop fail'))

    @pyqtSlot()
    def on_pushButtonASGLoad_clicked(self):
        # 1) fetch sequences data and visualize
        self.fetchSequencesfromTableWidget()
        try:
            self.asg.load_data(self.sequences)
            self.updateSequenceChart()
            self.labelInstrStatus.setText('ASG: sequences loaded')
        except:
            self.labelInstrStatus.setText(color_str('ASG: abnormal sequence, not loaded'))
        # 2) load sequences into current scheduler
        try:
            self.schedulers[self.schedulerMode].config_sequences(self.sequences)
        except:
            self.labelInstrStatus.setText(color_str('ASG: abnormal sequence, not loaded'))

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
        self.timerODMRFrequency.stop()
        try:
            self.schedulers[self.schedulerMode].close()
            self.labelInstrStatus.setText('{} Scheduler: interrupted'.format(self.schedulerMode))
        except:
            self.labelInstrStatus.setText(color_str('{} Scheduler: interrupted'.format(self.schedulerMode)))

    @pyqtSlot()
    def on_pushButtonODMRFrequencyFit_clicked(self):
        # 必须等run结束曲线画出来后拟合
        if not self.timerODMRFrequency.isActive() and ...:
            self.labelInstrStatus.setText(color_str('Data fitting: done'))
        else:
            self.labelInstrStatus.setText(color_str('Data fitting: no data to fit'))

    ###########################################
    # Time-domain measurement panel (Ramsey, Rabi, Relaxation)
    ################
    @pyqtSlot()
    def on_pushButtonODMRTimeInterrupt_clicked(self):
        # interrupt
        self.timerODMRTime.stop()
        try:
            self.schedulers[self.schedulerMode].close()
            self.labelInstrStatus.setText('{} Scheduler: interrupted'.format(self.schedulerMode))
        except:
            self.labelInstrStatus.setText(color_str('{} Scheduler: interrupted'.format(self.schedulerMode)))

    @pyqtSlot()
    def on_pushButtonODMRTimeFit_clicked(self):
        # 必须等run结束曲线画出来后拟合
        if not self.timerODMRTime.isActive() and ...:
            self.labelInstrStatus.setText(color_str('Data fitting: done'))
        else:
            self.labelInstrStatus.setText(color_str('Data fitting: no data to fit'))


def color_str(string, color='red'):
    """
    Convert plaintext to color text in for of HTML labels
    """
    return '<font color={}>{}</font>'.format(color, string)
