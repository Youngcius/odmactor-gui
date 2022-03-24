import sys
from ui import odmactor_window
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSlot


class OdmactorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(OdmactorGUI, self).__init__()
        self.ui = odmactor_window.Ui_OdmactorMainWindow()
        self.ui.setupUi(self)

        # self.scheduler =
