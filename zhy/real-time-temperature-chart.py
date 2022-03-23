import random
import math
import time
import threading

from PyQt5.QtChart import (QAreaSeries, QBarSet, QChart, QChartView,
                           QLineSeries, QPieSeries, QScatterSeries, QSplineSeries,
                           QStackedBarSeries, QValueAxis)
from PyQt5.QtCore import (
    pyqtSlot, QPoint, QPointF, Qt
)
from PyQt5.Qt import (QApplication, QWidget, QPushButton,
                      QThread, QMutex, pyqtSignal)
from PyQt5.QtGui import QColor, QPainter, QPalette
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout,
                             QLabel, QSizePolicy, QWidget, QPushButton)


class RealTimeChart(QChart):
    def __init__(self, parent=None):
        super(RealTimeChart, self).__init__(parent)
        self.xRange = 500
        self.sampleRate = 1
        self.counter = 0
        self.seriesList = []
        self.legend().show()

        self.axisX = QValueAxis()
        self.axisX.setRange(0, self.xRange)
        self.addAxis(self.axisX, Qt.AlignBottom)
        # self.setAxisX(self.axisX, series)

        self.axisY = QValueAxis()
        self.axisY.setRange(500, 600)
        self.addAxis(self.axisY, Qt.AlignLeft)
        # self.setAxisY(self.axisY, series)

        self.series = QLineSeries()
        self.series.setName("生成300~1000随机数")
        self.series.setUseOpenGL(True)
        self.addSeries(self.series)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

    def handleUpdate(self, ydata):

        if (self.counter < self.xRange):
            for i in range(self.sampleRate):
                self.series.append(self.counter + i, ydata)
        else:

            points = self.series.pointsVector()
            y_temp = [0] * (len(points) - self.sampleRate)
            for i in range(len(points) - self.sampleRate):
                points[i].setY(points[i + self.sampleRate].y())
                y_temp[i] = points[i + self.sampleRate].y()
            for i in range(self.sampleRate):
                points[len(points) - (self.sampleRate - i)].setY(ydata)
            self.series.replace(points)
            self.axisY.setRange(min(y_temp), max(y_temp))

        self.series.setUseOpenGL(True)
        self.counter += self.sampleRate


class WorkThread(QThread):
    _signal_updateUI = pyqtSignal()

    def __init__(self, parent=None):
        super(WorkThread, self).__init__(parent)
        self.qmut = QMutex()
        self.isexit = False

    def run(self):
        while (True):
            self.qmut.lock()
            if (self.isexit):
                break
            self.qmut.unlock()

            self._signal_updateUI.emit()
            time.sleep(0.01)
        self.qmut.unlock()

    def stop(self):
        # 改变线程状态与终止
        self.qmut.lock()
        self.isexit = True
        self.qmut.unlock()
        self.wait()


class MyWidget(QWidget):

    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent)

        self.mythread = WorkThread()

        # self.m_charts = []
        self.pushubutton = QPushButton()
        self.pushubutton.setText('start')
        self.pushubutton_stop = QPushButton()
        self.pushubutton_stop.setText('stop')

        self.connectSignals()

        # Create the layout.
        baseLayout = QGridLayout()
        settingsLayout = QHBoxLayout()
        settingsLayout.addWidget(self.pushubutton)
        settingsLayout.addWidget(self.pushubutton_stop)
        settingsLayout.addStretch()
        baseLayout.addLayout(settingsLayout, 0, 0, 1, 3)

        self.myChart = RealTimeChart()
        chartView = QChartView(self.myChart)
        chartView.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        baseLayout.addWidget(chartView)
        # self.m_charts.append(chartView)
        self.setLayout(baseLayout)
        self.pushubutton_stop.setEnabled(False)

    def pushubutton_clicked(self):
        self.mythread.start()
        self.pushubutton.setEnabled(False)
        self.pushubutton_stop.setEnabled(True)

    def pushubutton_stop_clicked(self):
        self.mythread.stop()
        self.mythread.isexit = False
        self.pushubutton.setEnabled(True)
        self.pushubutton_stop.setEnabled(False)

    def connectSignals(self):
        self.pushubutton.clicked.connect(self.pushubutton_clicked)
        self.pushubutton_stop.clicked.connect(self.pushubutton_stop_clicked)
        self.mythread._signal_updateUI.connect(self.threading_slot)

    def threading_slot(self):
        yint = random.randint(300, 1000)
        self.myChart.handleUpdate(yint)
        # 实时刷新界面
        QApplication.processEvents()


if __name__ == '__main__':
    import sys

    from PyQt5.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    window = QMainWindow()
    widget = MyWidget()
    window.setCentralWidget(widget)
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())
