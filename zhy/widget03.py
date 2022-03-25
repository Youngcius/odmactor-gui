import sys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot
from ui import ui03


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = ui03.Ui_MainWindow()
        self.ui.setupUi(self)
        self._buildUI()

    def _buildUI(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setText('文件名')
        self.ui.statusbar.addWidget(self.label)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # helloWidget = QtWidgets.QWidget()
    # ui = untitled.Ui_Form()
    # ui.setupUi(helloWidget)
    helloWidget = MyWindow()
    helloWidget.show()
    sys.exit(app.exec())
