import sys

from PyQt5 import QtWidgets, QtCore
from ui import ui01


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super(MyWidget, self).__init__()
        self.ui = ui01.Ui_Form()
        self.ui.setupUi(self)
        # self.radio = QtWidgets.QRadioButton(self)
        # QtCore.QMetaObject.connectSlotsByName(self)
        # self.radio.setText('====')
        # self.ui.radio

    @QtCore.pyqtSlot(bool)
    def on_radioButton_clicked(self, checked):
        # print('clicked:', self.radio.isChecked())
        pass

    @QtCore.pyqtSlot()
    def on_toolButton_clicked(self):
        print('clicked this button')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # helloWidget = QtWidgets.QWidget()
    # ui = untitled.Ui_Form()
    # ui.setupUi(helloWidget)
    helloWidget = MyWidget()
    helloWidget.show()
    sys.exit(app.exec())
