import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSlot
from ui import ui02


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()
        self.ui = ui02.Ui_Form()
        self.ui.setupUi(self)


    # @QtCore.pyqtSlot()
    # def on_toolButton_clicked(self):
    #     print('clicked this button')
    @pyqtSlot()
    def on_pushButtonLeft_clicked(self):
        self.ui.lineEdit.setAlignment(Qt.AlignLeft)
        self.ui.pushButtonRight.setChecked(False)
        self.ui.pushButtonCenter.setChecked(False)

    @pyqtSlot()
    def on_pushButtonRight_clicked(self):
        self.ui.lineEdit.setAlignment(Qt.AlignRight)
        self.ui.pushButtonCenter.setChecked(False)
        self.ui.pushButtonLeft.setChecked(False)

    @pyqtSlot()
    def on_pushButtonCenter_clicked(self):
        self.ui.lineEdit.setAlignment(Qt.AlignCenter)
        self.ui.pushButtonLeft.setChecked(False)
        self.ui.pushButtonRight.setChecked(False)

    @pyqtSlot(bool)
    def on_pushButtonBold_clicked(self, checked):
        font = self.ui.lineEdit.font()
        font.setBold(checked)
        self.ui.lineEdit.setFont(font)

    @pyqtSlot(bool)
    def on_pushButtonItalic_clicked(self, checked):
        font = self.ui.lineEdit.font()
        font.setItalic(checked)
        self.ui.lineEdit.setFont(font)

    @pyqtSlot(bool)
    def on_pushButtonUnderline_clicked(self, checked):
        font = self.ui.lineEdit.font()
        font.setUnderline(checked)
        self.ui.lineEdit.setFont(font)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # helloWidget = QtWidgets.QWidget()
    # ui = untitled.Ui_Form()
    # ui.setupUi(helloWidget)
    helloWidget = MyWidget()
    helloWidget.show()
    sys.exit(app.exec())
