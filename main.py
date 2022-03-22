import sys
from PyQt5 import QtWidgets
from ui import odmactor_window


class OdmactorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(OdmactorWindow, self).__init__()
        self.ui = odmactor_window.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    window = OdmactorWindow()
    window.show()
    sys.exit(app.exec())
