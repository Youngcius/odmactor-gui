import sys
from ui import odmactor_window
from PyQt5 import QtWidgets, QtCore, QtGui


class OdmactorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(OdmactorWindow, self).__init__()
        self.ui = odmactor_window.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)

        # self.scheduler =


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = OdmactorWindow()
    window.show()
    sys.exit(app.exec())
