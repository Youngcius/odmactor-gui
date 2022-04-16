import os
import json
import time
import datetime
import threading
import nidaqmx
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as C
import TimeTagger as tt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore, QtGui, QtChart
from PyQt5.QtCore import Qt, pyqtSlot, QThread, QTimer

import scheduler
import utils
from instrument import ASG, Microwave, Laser, LockInAmplifier
from utils.sequence import sequences_to_figure

from ui import odmactor_window

# 继承关系：QGraphicsItem --> QGraphicsObject --> QGraphicsWidget --> QChart --> QPolarChart

timeUnitDict = {'s': 1, 'ms': C.milli, 'us': C.micro, 'ns': C.nano, 'ps': C.pico}
freqUnitDict = {'Hz': 1, 'KHz': C.kilo, 'MHz': C.mega, 'GHz': C.giga}
frequencyDomainModes = ['CW', 'Pulse']
timeDomainModes = ['Ramsey', 'Rabi', 'Relaxation']
schedulerModes = frequencyDomainModes + timeDomainModes

plt.style.use('seaborn-pastel')


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

        # initialize instrument instances
        self.initInstruments()
        self.checkInstruments()

        # fetch parameters from initial UI
        self.fetchParameters()

        # initialize data variables
        self.schedulers = {
            mode: getattr(scheduler, mode + 'Scheduler')(
                laser=self.laser, mw=self.mw, tagger=self.tagger, asg=self.asg, epoch_omit=5, use_lockin=self.useLockin
            ) for mode in schedulerModes
        }

        # photon count config (tagger counter measurement class)
        self.updatePhotonCountConfig()
        if self.tagger:  # initialize Counter on Tagger
            self.counter = tt.Counter(self.tagger, **self.photonCountConfig)
            self.counter.stop()
        else:
            self.counter = None
        if self.lockin:  # initialize DAQ task on Lockin
            self.daqtask = nidaqmx.Task()
            self.daqtask.ai_channels.add_ai_voltage_chan("Dev1/ai0")
            self.daqcache = []
        else:
            self.daqtask = None

        # initial charts
        self.initCharts()

    def initInstruments(self):
        """
        Initialize instances representing specific instruments, i.e, Laser, Microwave, ASG, Tagger, Lock-in, etc.
        """
        self.laser = Laser()
        self.asg = ASG()

        try:
            self.mw = Microwave()
            self.ui.doubleSpinBoxMicrowaveFrequency.setValue(self.mw.get_frequency() / C.giga)
            self.ui.doubleSpinBoxMicrowavePower.setValue(self.mw.get_power())
        except:
            self.mw = None

        if tt.scanTimeTagger():
            self.tagger = tt.createTimeTagger()
        else:
            self.tagger = None

        try:
            self.lockin = LockInAmplifier()
        except:
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
        self.seqFigCanvas = FigureCanvas(sequences_to_figure(self.sequences))
        self.layoutSequenceVisualization = QtWidgets.QVBoxLayout(self.ui.widgetSequenceVisualization)
        self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
        self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutSequenceVisualization.setSpacing(0)

        ###################################
        # initialize photon count chart: figure canvas, toolbar, axes, timer
        self.canvasPhotonCount = FigureCanvas(plt.figure())  # figure canvas
        self.naviBarPhotonCount = NavigationToolbar(self.canvasPhotonCount, self.ui.widgetPhotonCountVisualization)
        self.layoutPhotonCountVisualization = QtWidgets.QVBoxLayout(self.ui.widgetPhotonCountVisualization)
        self.layoutPhotonCountVisualization.addWidget(self.naviBarPhotonCount)
        self.layoutPhotonCountVisualization.addWidget(self.canvasPhotonCount)
        self.layoutPhotonCountVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutPhotonCountVisualization.setSpacing(0)
        self.axesPhotonCount = self.canvasPhotonCount.figure.subplots()
        self.axesPhotonCount.set_xlabel('Time (s)', fontsize=13)
        self.axesPhotonCount.set_ylabel('Count', fontsize=13)
        self.timerPhotonCount = self.canvasPhotonCount.new_timer(100, [(self.updatePhotonCountChart, (), {})])

        ###################################
        # initialize frequency-domain ODMR chart: figure canvas, toolbar, axes, timer
        self.canvasODMRFrequency = FigureCanvas(plt.figure())  # figure canvas
        self.naviBarODMRFrequency = NavigationToolbar(self.canvasODMRFrequency,
                                                      self.ui.widgetODMRFrequencyVisualization)
        self.layoutODMRFrequencyVisualization = QtWidgets.QVBoxLayout(self.ui.widgetODMRFrequencyVisualization)
        self.layoutODMRFrequencyVisualization.addWidget(self.naviBarODMRFrequency)
        self.layoutODMRFrequencyVisualization.addWidget(self.canvasODMRFrequency)
        self.layoutODMRFrequencyVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutODMRFrequencyVisualization.setSpacing(0)
        self.axesODMRFrequency = self.canvasODMRFrequency.figure.subplots()
        self.axesODMRFrequency.set_xlabel('Frequency (Hz)', fontsize=13)
        self.axesODMRFrequency.set_ylabel('Count/Contrast', fontsize=13)
        self.timerODMRFrequency = self.canvasODMRFrequency.new_timer(100, [(self.updateODMRFrequencyChart, (), {})])

        ###################################
        # initialized time-domain ODMR chart
        self.canvasODMRTime = FigureCanvas(plt.figure())
        self.naviBarODMRTime = NavigationToolbar(self.canvasODMRTime, self.ui.widgetODMRTimeVisualization)
        self.layoutODMRTimeVisualization = QtWidgets.QVBoxLayout(self.ui.widgetODMRTimeVisualization)
        self.layoutODMRTimeVisualization.addWidget(self.naviBarODMRTime)
        self.layoutODMRTimeVisualization.addWidget(self.canvasODMRTime)
        self.layoutODMRTimeVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutODMRTimeVisualization.setSpacing(0)
        self.axesODMRTime = self.canvasODMRTime.figure.subplots()
        self.axesODMRTime.set_xlabel('Time (s)', fontsize=13)
        self.axesODMRTime.set_ylabel('Count/Contrast', fontsize=13)
        self.timerODMRTime = self.canvasODMRTime.new_timer(100, [(self.updateODMRTimeChart, (), {})])

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
            'tagger': int(self.ui.comboBoxASGTagger.currentText()),
            'mw_sync': int(self.ui.comboBoxASGMicrowaveSync.currentText()),
            'lockin_sync': int(self.ui.comboBoxASGLockinSync.currentText())
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
        self.daqcache = [0] * self.photonCountConfig['n_values']

    def buildUI(self):
        # status bar
        self.labelInstrStatus = QtWidgets.QLabel(self)
        self.labelInstrStatus.setText('Instrument Status:')
        self.ui.statusbar.addWidget(self.labelInstrStatus)

        # progress bar
        self.progressBar = QtWidgets.QProgressBar(self)
        self.progressBar.setMinimumWidth(200)
        self.progressBar.setFormat('%p%')  # %p%, %v
        self.ui.statusbar.addPermanentWidget(self.progressBar)

        # table widget
        # self.ui.tableWidgetSequence.horizontalHeader().setStyleSheet('QHeaderView::section{background:lightblue;}')
        for i in range(self.ui.tableWidgetSequence.rowCount()):
            for j in range(self.ui.tableWidgetSequence.columnCount()):
                item = QtWidgets.QTableWidgetItem(str(0))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.ui.tableWidgetSequence.setItem(i, j, item)

        # other settings
        self.ui.groupBoxODMRFrequency.setChecked(True)
        self.ui.radioButtonODMRCW.setChecked(True)
        self.schedulerMode = 'CW'
        self.ui.radioButtonUseLockin.setChecked(True)
        self.ui.radioButtonUseTagger.setChecked(False)
        self.useLockin = True

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
    @pyqtSlot(bool)
    def on_radioButtonLaserCW_clicked(self, checked):
        if checked:
            self.laserMode = 'CW'

    @pyqtSlot(bool)
    def on_radioButtonLaserPulse_clicked(self, checked):
        if checked:
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

    @pyqtSlot(bool)
    def on_pushButtonMicrowaveOnOff_clicked(self, checked):
        if isinstance(self.mw, Microwave) and self.mw.connect():
            if checked:
                self.mw.start()
                self.labelInstrStatus.setText('Microwave: started')
                if self.ui.groupBoxMicrowaveTimer.isChecked():
                    time.sleep(self.ui.doubleSpinBoxMicrowaveTimerTime.value() * timeUnitDict[
                        self.ui.comboBoxMicrowaveTimerTimeUnit.currentText()])
                    self.mw.stop()
                    self.ui.pushButtonMicrowaveOnOff.setChecked(False)
                    self.labelInstrStatus.setText('Microwave: stopped')
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

    @pyqtSlot()
    def on_comboBoxASGMicrowaveSync_valueChanged(self):
        self.asgChannels['mw_sync'] = int(self.ui.comboBoxASGMicrowaveSync.currentText())

    @pyqtSlot()
    def on_comboBoxASGLockinSync_valueChanged(self):
        self.asgChannels['lockin_sync'] = int(self.ui.comboBoxASGLockinSync.currentText())

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
    @pyqtSlot(bool)
    def on_radioButtonUseLockin_clicked(self, checked):
        self.useLockin = checked

    @pyqtSlot(bool)
    def on_radioButtonUseTagger_clicked(self, checked):
        self.useLockin = not checked

    @pyqtSlot()
    def on_pushButtonODMRLoadSequences_clicked(self):
        """
        Fetch sequences parameters to generate ASG sequences, load it into ASG and visualize
        """
        self.schedulers[self.schedulerMode].connect()
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
        self.schedulers[self.schedulerMode].with_ref = self.odmrSeqConfig['withReference']
        self.schedulers[self.schedulerMode].mw_on_off = self.odmrSeqConfig['MicrowaveOnOff']
        self.schedulers[self.schedulerMode].set_asg_sequences_ttl(
            laser_ttl=1 if self.ui.checkBoxASGLaserTTL.isChecked() else 0,
            mw_ttl=1 if self.ui.checkBoxASGMicrowaveTTL.isChecked() else 0,
            apd_ttl=1 if self.ui.checkBoxASGAPDTTL.isChecked() else 0,
            tagger_ttl=1 if self.ui.checkBoxASGTaggerTTL.isChecked() else 0,
        )
        self.schedulers[self.schedulerMode].use_lockin = self.useLockin

        if self.schedulerMode == 'CW':
            period = max(self.odmrSeqConfig['laserInit'], self.odmrSeqConfig['microwaveTime'])
            self.schedulers[self.schedulerMode].configure_odmr_seq(period=period, N=self.odmrSeqConfig['N'])
        else:
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
        # 微波参数 --> 设置序列 ------> 频率范围 --> counting setting
        self.schedulers[self.schedulerMode].connect()
        if self.ui.groupBoxODMRFrequency.isChecked():
            self.startFrequencyDomainDetecting()
        else:
            self.startTimeDomainDetecting()

    def startFrequencyDomainDetecting(self):
        """
        Start frequency-domain ODMR detecting experiments, i.e., CW or Pulse
        """
        # frequencies for scanning
        unit_freq = freqUnitDict[self.ui.comboBoxODMRFrequencyUnit.currentText()]
        freq_start = self.ui.doubleSpinBoxODMRFrequencyStart.value() * unit_freq
        freq_end = self.ui.doubleSpinBoxODMRFrequencyEnd.value() * unit_freq
        freq_step = self.ui.doubleSpinBoxODMRFrequencyStep.value() * unit_freq
        self.schedulers[self.schedulerMode].set_mw_freqs(freq_start, freq_end, freq_step)
        self.progressBar.setMaximum(len(self.schedulers[self.schedulerMode].frequencies))
        if self.useLockin:
            self.schedulers[self.schedulerMode].configure_lockin_counting(
                freq=self.ui.spinBoxODMRSyncFrequency.value()
            )
        else:
            self.schedulers[self.schedulerMode].configure_tagger_counting(
                apd_channel=self.taggerChannels['apd'],
                asg_channel=self.taggerChannels['asg'],
                reader='counter' if self.schedulerMode == 'CW' else 'cbm'
            )

        t = threading.Thread(target=self.schedulers[self.schedulerMode].run_scanning)
        t.start()
        self.timerODMRFrequency.start()

    def startTimeDomainDetecting(self):
        """
        Start time-domain ODMR detecting experiments, i.e., Ramsey, Rabi, Relaxation
        """
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
        self.timerODMRTime.start()

    @pyqtSlot()
    def on_pushButtonODMRSaveData_clicked(self):
        # TODO
        # self.counter.getData()
        # pass
        self.labelInstrStatus.setText('Saved in {}'.format(self.schedulers[self.schedulerMode].output_dir))

    # ODMR cheduler mode
    def connectScheduler(self, mode: str):
        """
        Allocate hardware resources to one specific scheduler
        """
        self.schedulerMode = mode
        if mode not in schedulerModes:
            raise ValueError('{} is not a supported scheduler type'.format(mode))
        # for k, scheduler in self.schedulers.keys():
        #     if k != mode:
        #         scheduler.close() # TODO: 不用释放仪器，因为仪器变量是 shared
        # self.schedulers[mode].connect()

    @pyqtSlot(bool)
    def on_radioButtonODMRCW_clicked(self, checked):
        if checked:
            try:
                self.connectScheduler('CW')
                self.labelInstrStatus.setText('CW Scheduler: ready')
            except:
                self.labelInstrStatus.setText(color_str('CW Scheduler: not ready'))

    @pyqtSlot(bool)
    def on_radioButtonODMRPulse_clicked(self, checked):
        if checked:
            try:
                self.connectScheduler('Pulse')
                self.labelInstrStatus.setText('Pulse Scheduler: ready')
            except:
                self.labelInstrStatus.setText(color_str('Pulse Scheduler: not ready'))

    @pyqtSlot(bool)
    def on_radioButtonODMRRamsey_clicked(self, checked):
        if checked:
            try:
                self.connectScheduler('Ramsey')
                self.labelInstrStatus.setText('Ramsey Scheduler: ready')
            except:
                self.labelInstrStatus.setText(color_str('Ramsey Scheduler: not ready'))

    @pyqtSlot(bool)
    def on_radioButtonODMRRabi_clicked(self, checked):
        if checked:
            try:
                self.connectScheduler('Rabi')
                self.labelInstrStatus.setText('Rabi Scheduler: ready')
            except:
                self.labelInstrStatus.setText(color_str('Rabi Scheduler: not ready'))

    @pyqtSlot(bool)
    def on_radioButtonODMRRelaxation_clicked(self, checked):
        if checked:
            try:
                self.connectScheduler('Relaxation')
                self.labelInstrStatus.setText('Relaxation Scheduler: ready')
            except:
                self.labelInstrStatus.setText(color_str('Relaxation Scheduler: not ready'))

    ###########################################
    # Photon count configuration
    ################
    @pyqtSlot(bool)
    def on_pushButtonPhotonCountOnOff_clicked(self, checked):
        """
        Start to count using Time Tagger of Lockin Amplifier
        :param checked: if True, reload parameters to start counting; otherwise, stop counting
        """
        # self.tagger.setTestSignal(int(self.ui.comboBoxTaggerAPD.currentText()), True)  # TODO: delete this
        if checked:
            self.updatePhotonCountConfig()
            if self.useLockin:  # using Lockin Amplifier
                pass
            else:  # using Time Tagger
                try:
                    self.counter = tt.Counter(self.tagger, **self.photonCountConfig)
                    self.counter.start()
                except:
                    self.labelInstrStatus.setText(color_str('No Time Tagger to detect photons'))
            self.timerPhotonCount.start()
        else:
            if self.useLockin:  # using Lockin Amplifier
                pass
            else:  # using Time Tagger
                try:
                    self.counter.stop()
                except:
                    pass
            self.timerPhotonCount.stop()

    @pyqtSlot()
    def on_pushButtonPhotonCountRefresh_clicked(self):
        """
        Clear series data of the chart, restart counting
        """
        self.axesPhotonCount.clear()
        self.axesODMRFrequency.set_xlabel('Time (s)', fontsize=13)
        if self.ui.radioButtonPhotonCountRate.isChecked():
            self.axesODMRFrequency.set_ylabel('Count rate', fontsize=13)
        else:
            self.axesODMRFrequency.set_ylabel('Count number', fontsize=13)

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
        if self.useLockin:
            data = {
                'channel': self.photonCountConfig['channels'][0],
                'time': (np.arange(0, self.photonCountConfig['n_values']) * 0.1).tolist(),
                'magnitude': self.daqcache,
                'timestamp': str(timestamp),
                'device': 'Lockin Amplifier'
            }
        else:
            counts = self.counter.getData().ravel()
            data = {
                'channel': self.photonCountConfig['channels'][0],
                'time': (self.counter.getIndex() * C.pico).tolist(),
                'count': counts.tolist(),
                'count rate (1/s)': (counts / self.photonCountConfig['binwidth'] / C.pico).tolist(),
                'timestamp': str(timestamp),
                'device': 'Time Tagger'
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
        self.seqFigCanvas = FigureCanvas(sequences_to_figure(self.sequences))
        self.layoutSequenceVisualization.addWidget(self.seqFigCanvas)  # 添加FigureCanvas对象
        self.layoutSequenceVisualization.setContentsMargins(0, 0, 0, 0)
        self.layoutSequenceVisualization.setSpacing(0)

    def updatePhotonCountChart(self):
        """
        Update the real-time photon couting chart
        ---
        1. with Time Tagger
            read `n_values` counts to update the last `n_values` counts, there might be data overlapping
        2. with Lockin Amplifier:
            read new data with fixed length and average aggregation to obtain a single value,
            then add it into the cache queue for every fixed time interval, 0.1 second in default
        """
        self.axesPhotonCount.clear()
        self.axesPhotonCount.set_xlabel('Time (s)', fontsize=13)
        if self.useLockin:
            self.daqcache.append(np.mean(self.daqtask.read(number_of_samples_per_channel=100)))
            self.daqcache.pop(0)
            counts = self.daqcache
            times = np.arange(0, self.photonCountConfig['n_values']) * 0.1
            self.axesPhotonCount.set_ylabel('Magnitude', fontsize=13)
        else:
            counts = self.counter.getData().ravel()
            times = self.counter.getIndex() * C.pico
            if self.ui.radioButtonPhotonCountRate.isChecked():
                counts = counts / self.photonCountConfig['binwidth'] / C.pico
                self.axesPhotonCount.set_ylabel('Count rate', fontsize=13)
            else:
                self.axesPhotonCount.set_ylabel('Count number', fontsize=13)
        self.axesPhotonCount.plot(times, counts)
        self.axesPhotonCount.figure.canvas.draw()


    def updateODMRFrequencyChart(self):
        """
        Update frequency-domain ODMR results periodically
        ---
        1. with Time Tagger
            read one value in each ASG operation period, totally N values
        2. with Lock-in Amplifier
            read M values after the last ASG operation period, M is not necessarily equal to N
        """
        # update series
        self.axesODMRFrequency.clear()
        self.axesODMRFrequency.set_title('{} Spectrum'.format(self.schedulerMode), fontsize=15)
        self.axesODMRFrequency.set_xlabel('Frequency (Hz)', fontsize=13)
        freqs = self.schedulers[self.schedulerMode].frequencies
        sig = self.schedulers[self.schedulerMode].cur_data
        if self.ui.checkBoxODMRWithReference.isChecked():  # plot contrast
            ref = self.schedulers[self.schedulerMode].cur_data_ref
            length = len(ref)
            contrast = [s / r for s, r in zip(sig[:length], ref)]
            self.axesODMRFrequency.plot(freqs[:length], contrast, 'o-')
            self.axesODMRFrequency.set_ylabel('Contrast', fontsize=13)
        else:  # plot count
            length = len(sig)
            self.axesODMRFrequency.plot(freqs[:length], sig, 'o-')
            self.axesODMRFrequency.set_ylabel('Count', fontsize=13)
        self.axesODMRFrequency.figure.canvas.draw()

        # update progress bar
        cur_freq = self.schedulers[self.schedulerMode].cur_freq
        if cur_freq in freqs:
            self.progressBar.setValue(freqs.index(cur_freq) + 1)

        if not self.schedulers[self.schedulerMode].is_running:
            self.labelInstrStatus.setText(f'{self.schedulerMode} Scheduler: done. '
                                          f'Data has been saved in {self.schedulers[self.schedulerMode].output_fname}')
            self.timerODMRFrequency.stop()
            self.progressBar.setValue(-1)

    def updateODMRTimeChart(self):
        """
        Update time-domain ODMR results periodically
        ---
        1. with Time Tagger
            read one value in each ASG operation period, totally N values
        2. with Lock-in Amplifier
            read M values after the last ASG operation period, M is not necessarily equal to N
        """
        # update series
        self.axesODMRTime.clear()
        self.axesODMRTime.set_title('{} Measurement'.format(self.schedulerMode), fontsize=15)
        self.axesODMRTime.set_xlabel('Time (s)', fontsize=13)
        times = self.schedulers[self.schedulerMode].times
        sig = self.schedulers[self.schedulerMode].data_ref
        if self.ui.checkBoxODMRWithReference.isChecked():  # plot contrast
            ref = self.schedulers[self.schedulerMode].cur_data_ref
            length = len(ref)
            contrast = [s / r for s, r in zip(sig[:length], ref)]
            self.axesODMRFrequency.plot(times[:length], contrast, 'o-')
            self.axesODMRTime.set_ylabel('Contrast', fontsize=13)
        else:  # plot count
            length = len(sig)
            self.axesODMRFrequency.plot(time[:length], sig, 'o-')
            self.axesODMRFrequency.set_ylabel('Count', fontsize=13)
        self.axesODMRTime.figure.canvas.draw()

        # update progress bar
        cur_time = self.schedulers[self.schedulerMode].cur_time
        if cur_time in times:
            self.progressBar.setValue(times.index(cur_time) + 1)

        if not self.schedulers[self.schedulerMode].is_running:
            self.labelInstrStatus.setText(f'{self.schedulerMode} Scheduler: done. '
                                          f'Data has been saved in {self.schedulers[self.schedulerMode].output_fname}')
            self.timerODMRTime.stop()
            self.progressBar.setValue(-1)

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
        for i in range(self.ui.tableWidgetSequence.rowCount()):
            for j in range(self.ui.tableWidgetSequence.columnCount()):
                self.ui.tableWidgetSequence.item(i, j).setText(str(0))
        self.sequences = [[0, 0] for _ in range(8)]
        self.updateSequenceChart()

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

    @pyqtSlot(int)
    def on_spinBoxODMRSyncFrequency_valueChanged(self, freq):
        self.schedulers[self.schedulerMode].sync_freq = freq

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
