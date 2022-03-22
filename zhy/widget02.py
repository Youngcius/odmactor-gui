import sys

from PyQt5 import QtWidgets
from ui import untitled

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    helloWidget = QtWidgets.QWidget()
    ui = untitled.Ui_Form()
    ui.setupUi(helloWidget)
    helloWidget.show()
    sys.exit(app.exec())
