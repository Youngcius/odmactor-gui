import sys
from ui import odmactor_window
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSlot


class OdmactorGUI(QtWidgets.QMainWindow):
    """
    Mainwindow configuration of Odmactor
    """

    def __init__(self):
        super(OdmactorGUI, self).__init__()
        self.ui = odmactor_window.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)

    @pyqtSlot(float)
    def on_spinBoxODMRSyncFrequency_valueChanged(self, freq):
        print(freq)

    @pyqtSlot(float)
    def on_doubleSpinBoxMicrowavePower_valueChanged(self, power):
        print('power', power)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = OdmactorGUI()
    window.show()
    sys.exit(app.exec())
