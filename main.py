import sys
from PyQt5 import QtWidgets
# from prog import OdmactorGUI
from prog_mpl import OdmactorGUI


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = OdmactorGUI()
    window.show()
    sys.exit(app.exec())
