import sys
from PyQt5 import QtWidgets
from prog import OdmactorGUI


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = OdmactorGUI()
    window.show()
    sys.exit(app.exec())
