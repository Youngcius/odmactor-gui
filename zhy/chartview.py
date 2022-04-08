import sys
import random
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart, QChartView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QDateTime, Qt, QTimer


class ChartView(QChartView, QChart):
    def __init__(self, *args, **kwargs):
        super(ChartView, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        self.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.chart_init()
        self.timer_init()

    def timer_init(self):
        # 使用QTimer，一定时间触发一次，更新数据
        self.timer = QTimer()
        self.timer.timeout.connect(self.drawLine)
        self.timer.start(100)  # unit: ms

    def chart_init(self):
        self.chart = QChart()
        self.series = QSplineSeries()
        # 设置曲线名称
        self.series.setName("实时数据")
        # 把曲线添加到QChart的实例中
        self.chart.addSeries(self.series)
        # 声明并初始化X轴，Y轴
        self.dtaxisX = QDateTimeAxis()
        self.vlaxisY = QValueAxis()
        # 设置坐标轴显示范围
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        self.vlaxisY.setMin(0)
        self.vlaxisY.setMax(1500)
        # 设置X轴时间样式
        self.dtaxisX.setFormat("MM月dd hh:mm:ss")
        # 设置坐标轴上的格点
        self.dtaxisX.setTickCount(6)
        self.vlaxisY.setTickCount(11)
        # 设置坐标轴名称
        self.dtaxisX.setTitleText("时间")
        self.vlaxisY.setTitleText("量程")
        # 设置网格不显示
        self.vlaxisY.setGridLineVisible(False)
        # 把坐标轴添加到chart中
        self.chart.addAxis(self.dtaxisX, Qt.AlignBottom)
        self.chart.addAxis(self.vlaxisY, Qt.AlignLeft)
        # 把曲线关联到坐标轴
        self.series.attachAxis(self.dtaxisX)
        self.series.attachAxis(self.vlaxisY)

        self.setChart(self.chart)

    def drawLine(self):
        """
        更新数据重新绘图
        """
        # 获取当前时间
        bjtime = QDateTime.currentDateTime()
        # 更新X轴坐标
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        # 当曲线上的点超出X轴的范围时，移除最早的点
        if (self.series.count() > 149):
            self.series.removePoints(0, self.series.count() - 149)
        # 产生随即数
        yint = random.randint(0, 1500)
        # 添加数据到曲线末端
        self.series.append(bjtime.toMSecsSinceEpoch(), yint)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    view = ChartView()
    view.show()
    sys.exit(app.exec_())
