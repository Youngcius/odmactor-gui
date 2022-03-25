import os
import sys
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWebEngineWidgets


import molecule
import data_process
import datetime

# QtCore.QUrl
################
# web view
################

import webbrowser
webbrowser.open('view.html')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        self.setWindowTitle('显示网页')
        self.resize(800, 800)
        # 新建一个QWebEngineView()对象
        self.qwebengine = QtWebEngineWidgets.QWebEngineView(self)
        # 设置网页在窗口中显示的位置和大小
        # self.qwebengine.setGeometry(20, 20, 600, 600)

        # 在QWebEngineView中加载网址
        # self.qwebengine.load(QtCore.QUrl(r"https://www.csdn.net/"))
        with open('view.html', 'r') as f:
            txt = f.read()
        self.qwebengine.setHtml(txt)
        print(txt)
        self.setCentralWidget(self.qwebengine)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()