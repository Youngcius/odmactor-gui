# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\molecule.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 948)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.toolBox = QtWidgets.QToolBox(self.splitter)
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 331, 833))
        self.page.setObjectName("page")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBoxMolecule = QtWidgets.QGroupBox(self.page)
        self.groupBoxMolecule.setObjectName("groupBoxMolecule")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBoxMolecule)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButtonWebView = QtWidgets.QPushButton(self.groupBoxMolecule)
        self.pushButtonWebView.setObjectName("pushButtonWebView")
        self.gridLayout_2.addWidget(self.pushButtonWebView, 0, 0, 1, 2)
        self.labelNo = QtWidgets.QLabel(self.groupBoxMolecule)
        self.labelNo.setObjectName("labelNo")
        self.gridLayout_2.addWidget(self.labelNo, 1, 0, 1, 1)
        self.spinBoxNo = QtWidgets.QSpinBox(self.groupBoxMolecule)
        self.spinBoxNo.setObjectName("spinBoxNo")
        self.gridLayout_2.addWidget(self.spinBoxNo, 1, 1, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBoxMolecule)
        self.groupBoxDataset = QtWidgets.QGroupBox(self.page)
        self.groupBoxDataset.setObjectName("groupBoxDataset")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBoxDataset)
        self.gridLayout.setObjectName("gridLayout")
        self.labelHistType = QtWidgets.QLabel(self.groupBoxDataset)
        self.labelHistType.setObjectName("labelHistType")
        self.gridLayout.addWidget(self.labelHistType, 0, 0, 1, 1)
        self.comboBoxHistType = QtWidgets.QComboBox(self.groupBoxDataset)
        self.comboBoxHistType.setObjectName("comboBoxHistType")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.comboBoxHistType.addItem("")
        self.gridLayout.addWidget(self.comboBoxHistType, 0, 1, 1, 1)
        self.labelSplit = QtWidgets.QLabel(self.groupBoxDataset)
        self.labelSplit.setObjectName("labelSplit")
        self.gridLayout.addWidget(self.labelSplit, 1, 0, 1, 1)
        self.doubleSpinBoxSplit = QtWidgets.QDoubleSpinBox(self.groupBoxDataset)
        self.doubleSpinBoxSplit.setObjectName("doubleSpinBoxSplit")
        self.gridLayout.addWidget(self.doubleSpinBoxSplit, 1, 1, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBoxDataset)
        self.groupBox_4 = QtWidgets.QGroupBox(self.page)
        self.groupBox_4.setObjectName("groupBox_4")
        self.formLayout = QtWidgets.QFormLayout(self.groupBox_4)
        self.formLayout.setObjectName("formLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.verticalLayout_2.addWidget(self.groupBox_4)
        self.toolBox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 331, 833))
        self.page_2.setObjectName("page_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.page_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.page_2)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_6.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBoxModelSelect = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxModelSelect.setObjectName("comboBoxModelSelect")
        self.comboBoxModelSelect.addItem("")
        self.comboBoxModelSelect.addItem("")
        self.comboBoxModelSelect.addItem("")
        self.comboBoxModelSelect.addItem("")
        self.comboBoxModelSelect.addItem("")
        self.gridLayout_6.addWidget(self.comboBoxModelSelect, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_6.addWidget(self.label_7, 1, 0, 1, 1)
        self.comboBoxPropertyPredict = QtWidgets.QComboBox(self.groupBox)
        self.comboBoxPropertyPredict.setObjectName("comboBoxPropertyPredict")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.comboBoxPropertyPredict.addItem("")
        self.gridLayout_6.addWidget(self.comboBoxPropertyPredict, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_3 = QtWidgets.QGroupBox(self.page_2)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.checkBoxUseNew = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBoxUseNew.setObjectName("checkBoxUseNew")
        self.gridLayout_16.addWidget(self.checkBoxUseNew, 0, 0, 1, 1)
        self.checkBoxUseExisted = QtWidgets.QCheckBox(self.groupBox_3)
        self.checkBoxUseExisted.setObjectName("checkBoxUseExisted")
        self.gridLayout_16.addWidget(self.checkBoxUseExisted, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.page_2)
        self.groupBox_2.setObjectName("groupBox_2")
        self.formLayout_3 = QtWidgets.QFormLayout(self.groupBox_2)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setObjectName("label")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.toolBox.addItem(self.page_2, "")
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.groupBoxGraph2D = QtWidgets.QGroupBox(self.tab)
        self.groupBoxGraph2D.setObjectName("groupBoxGraph2D")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBoxGraph2D)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.scrollArea1 = QtWidgets.QScrollArea(self.groupBoxGraph2D)
        self.scrollArea1.setWidgetResizable(True)
        self.scrollArea1.setObjectName("scrollArea1")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 283, 196))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.labelFigure1 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.labelFigure1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelFigure1.setAlignment(QtCore.Qt.AlignCenter)
        self.labelFigure1.setObjectName("labelFigure1")
        self.gridLayout_15.addWidget(self.labelFigure1, 0, 0, 1, 1)
        self.scrollArea1.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_12.addWidget(self.scrollArea1, 0, 0, 1, 1)
        self.scrollArea2 = QtWidgets.QScrollArea(self.groupBoxGraph2D)
        self.scrollArea2.setWidgetResizable(True)
        self.scrollArea2.setObjectName("scrollArea2")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 283, 196))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_3)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.labelFigure2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.labelFigure2.setAlignment(QtCore.Qt.AlignCenter)
        self.labelFigure2.setObjectName("labelFigure2")
        self.gridLayout_7.addWidget(self.labelFigure2, 0, 0, 1, 1)
        self.scrollArea2.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout_12.addWidget(self.scrollArea2, 1, 0, 1, 1)
        self.scrollArea1_2 = QtWidgets.QScrollArea(self.groupBoxGraph2D)
        self.scrollArea1_2.setWidgetResizable(True)
        self.scrollArea1_2.setObjectName("scrollArea1_2")
        self.scrollAreaWidgetContents_4 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_4.setGeometry(QtCore.QRect(0, 0, 283, 196))
        self.scrollAreaWidgetContents_4.setObjectName("scrollAreaWidgetContents_4")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_4)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.labelFigure1_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_4)
        self.labelFigure1_2.setAlignment(QtCore.Qt.AlignCenter)
        self.labelFigure1_2.setObjectName("labelFigure1_2")
        self.gridLayout_14.addWidget(self.labelFigure1_2, 0, 0, 1, 1)
        self.scrollArea1_2.setWidget(self.scrollAreaWidgetContents_4)
        self.gridLayout_12.addWidget(self.scrollArea1_2, 2, 0, 1, 1)
        self.scrollArea2_2 = QtWidgets.QScrollArea(self.groupBoxGraph2D)
        self.scrollArea2_2.setWidgetResizable(True)
        self.scrollArea2_2.setObjectName("scrollArea2_2")
        self.scrollAreaWidgetContents_5 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_5.setGeometry(QtCore.QRect(0, 0, 283, 196))
        self.scrollAreaWidgetContents_5.setObjectName("scrollAreaWidgetContents_5")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_5)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.labelFigure2_2 = QtWidgets.QLabel(self.scrollAreaWidgetContents_5)
        self.labelFigure2_2.setAlignment(QtCore.Qt.AlignCenter)
        self.labelFigure2_2.setObjectName("labelFigure2_2")
        self.gridLayout_11.addWidget(self.labelFigure2_2, 0, 0, 1, 1)
        self.scrollArea2_2.setWidget(self.scrollAreaWidgetContents_5)
        self.gridLayout_12.addWidget(self.scrollArea2_2, 3, 0, 1, 1)
        self.gridLayout_13.addWidget(self.groupBoxGraph2D, 0, 0, 2, 1)
        self.groupBoxMoleculeInfo = QtWidgets.QGroupBox(self.tab)
        self.groupBoxMoleculeInfo.setObjectName("groupBoxMoleculeInfo")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBoxMoleculeInfo)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtWidgets.QLabel(self.groupBoxMoleculeInfo)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 1)
        self.pushButtonPositions = QtWidgets.QPushButton(self.groupBoxMoleculeInfo)
        self.pushButtonPositions.setObjectName("pushButtonPositions")
        self.gridLayout_3.addWidget(self.pushButtonPositions, 0, 2, 1, 1)
        self.lineEditSmiles = QtWidgets.QLineEdit(self.groupBoxMoleculeInfo)
        self.lineEditSmiles.setObjectName("lineEditSmiles")
        self.gridLayout_3.addWidget(self.lineEditSmiles, 0, 1, 1, 1)
        self.lineEditInChI = QtWidgets.QLineEdit(self.groupBoxMoleculeInfo)
        self.lineEditInChI.setObjectName("lineEditInChI")
        self.gridLayout_3.addWidget(self.lineEditInChI, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBoxMoleculeInfo)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 1, 0, 1, 1)
        self.pushButtonCharges = QtWidgets.QPushButton(self.groupBoxMoleculeInfo)
        self.pushButtonCharges.setObjectName("pushButtonCharges")
        self.gridLayout_3.addWidget(self.pushButtonCharges, 1, 2, 1, 1)
        self.tableWidgetProperties = QtWidgets.QTableWidget(self.groupBoxMoleculeInfo)
        self.tableWidgetProperties.setObjectName("tableWidgetProperties")
        self.tableWidgetProperties.setColumnCount(2)
        self.tableWidgetProperties.setRowCount(15)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidgetProperties.setHorizontalHeaderItem(1, item)
        self.tableWidgetProperties.horizontalHeader().setDefaultSectionSize(90)
        self.tableWidgetProperties.horizontalHeader().setMinimumSectionSize(90)
        self.tableWidgetProperties.verticalHeader().setDefaultSectionSize(20)
        self.tableWidgetProperties.verticalHeader().setMinimumSectionSize(19)
        self.gridLayout_3.addWidget(self.tableWidgetProperties, 2, 0, 1, 3)
        self.gridLayout_13.addWidget(self.groupBoxMoleculeInfo, 0, 1, 1, 1)
        self.groupBoxGraph3D = QtWidgets.QGroupBox(self.tab)
        self.groupBoxGraph3D.setObjectName("groupBoxGraph3D")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBoxGraph3D)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.scrollArea3 = QtWidgets.QScrollArea(self.groupBoxGraph3D)
        self.scrollArea3.setWidgetResizable(True)
        self.scrollArea3.setObjectName("scrollArea3")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 313, 384))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.labelFigure3 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.labelFigure3.setAlignment(QtCore.Qt.AlignCenter)
        self.labelFigure3.setObjectName("labelFigure3")
        self.gridLayout_8.addWidget(self.labelFigure3, 0, 0, 1, 1)
        self.scrollArea3.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_10.addWidget(self.scrollArea3, 0, 0, 1, 1)
        self.gridLayout_13.addWidget(self.groupBoxGraph3D, 1, 1, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupBoxStatistics = QtWidgets.QGroupBox(self.tab_3)
        self.groupBoxStatistics.setObjectName("groupBoxStatistics")
        self.gridLayout_4.addWidget(self.groupBoxStatistics, 0, 0, 1, 1)
        self.groupBoxTrainTest = QtWidgets.QGroupBox(self.tab_3)
        self.groupBoxTrainTest.setObjectName("groupBoxTrainTest")
        self.gridLayout_17 = QtWidgets.QGridLayout(self.groupBoxTrainTest)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.label_8 = QtWidgets.QLabel(self.groupBoxTrainTest)
        self.label_8.setObjectName("label_8")
        self.gridLayout_17.addWidget(self.label_8, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBoxTrainTest, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.groupBoxModelView = QtWidgets.QGroupBox(self.tab_2)
        self.groupBoxModelView.setObjectName("groupBoxModelView")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBoxModelView)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.groupBoxModelView)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_3.addWidget(self.label_9)
        self.gridLayout_5.addWidget(self.groupBoxModelView, 0, 0, 1, 1)
        self.groupBoxResult = QtWidgets.QGroupBox(self.tab_2)
        self.groupBoxResult.setObjectName("groupBoxResult")
        self.gridLayout_5.addWidget(self.groupBoxResult, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_9.addWidget(self.splitter, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 23))
        self.menubar.setObjectName("menubar")
        self.menu_F = QtWidgets.QMenu(self.menubar)
        self.menu_F.setObjectName("menu_F")
        self.menu_M = QtWidgets.QMenu(self.menubar)
        self.menu_M.setObjectName("menu_M")
        self.menu = QtWidgets.QMenu(self.menu_M)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../icons/图神经网络.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.menu.setIcon(icon)
        self.menu.setObjectName("menu")
        self.menu_A = QtWidgets.QMenu(self.menubar)
        self.menu_A.setObjectName("menu_A")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpenFile = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../icons/文档.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpenFile.setIcon(icon1)
        self.actionOpenFile.setObjectName("actionOpenFile")
        self.actionOpenFolder = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../icons/文件.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpenFolder.setIcon(icon2)
        self.actionOpenFolder.setObjectName("actionOpenFolder")
        self.actionClose = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../icons/退出.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionClose.setIcon(icon3)
        self.actionClose.setObjectName("actionClose")
        self.actionSave = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../icons/保存.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave.setIcon(icon4)
        self.actionSave.setObjectName("actionSave")
        self.actionSaveAs = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../icons/另存为.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSaveAs.setIcon(icon5)
        self.actionSaveAs.setObjectName("actionSaveAs")
        self.actionTrain = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("../icons/训练.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTrain.setIcon(icon6)
        self.actionTrain.setObjectName("actionTrain")
        self.actionPredict = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("../icons/测试.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPredict.setIcon(icon7)
        self.actionPredict.setObjectName("actionPredict")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("../icons/帮助.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionHelp.setIcon(icon8)
        self.actionHelp.setObjectName("actionHelp")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("../icons/关于.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbout.setIcon(icon9)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAboutQT = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("../icons/qt.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAboutQT.setIcon(icon10)
        self.actionAboutQT.setObjectName("actionAboutQT")
        self.actionGraphSAGE = QtWidgets.QAction(MainWindow)
        self.actionGraphSAGE.setObjectName("actionGraphSAGE")
        self.actionGCN = QtWidgets.QAction(MainWindow)
        self.actionGCN.setObjectName("actionGCN")
        self.actionXGBoost = QtWidgets.QAction(MainWindow)
        self.actionXGBoost.setObjectName("actionXGBoost")
        self.menu_F.addAction(self.actionOpenFile)
        self.menu_F.addAction(self.actionOpenFolder)
        self.menu_F.addAction(self.actionSave)
        self.menu_F.addAction(self.actionSaveAs)
        self.menu_F.addAction(self.actionClose)
        self.menu.addAction(self.actionGraphSAGE)
        self.menu.addAction(self.actionGCN)
        self.menu.addAction(self.actionXGBoost)
        self.menu_M.addAction(self.actionTrain)
        self.menu_M.addAction(self.actionPredict)
        self.menu_M.addAction(self.menu.menuAction())
        self.menu_A.addAction(self.actionAbout)
        self.menu_A.addAction(self.actionHelp)
        self.menu_A.addAction(self.actionAboutQT)
        self.menubar.addAction(self.menu_F.menuAction())
        self.menubar.addAction(self.menu_M.menuAction())
        self.menubar.addAction(self.menu_A.menuAction())

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(1)
        self.tabWidget.setCurrentIndex(1)
        self.actionClose.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "分子属性预测系统"))
        self.groupBoxMolecule.setTitle(_translate("MainWindow", "Molecule"))
        self.pushButtonWebView.setText(_translate("MainWindow", "浏览器中打开（观察交互图形）"))
        self.labelNo.setText(_translate("MainWindow", "所观察分子：NO."))
        self.groupBoxDataset.setTitle(_translate("MainWindow", "Dataset"))
        self.labelHistType.setText(_translate("MainWindow", "统计信息"))
        self.comboBoxHistType.setItemText(0, _translate("MainWindow", "旋转常量 A"))
        self.comboBoxHistType.setItemText(1, _translate("MainWindow", "旋转常量 B"))
        self.comboBoxHistType.setItemText(2, _translate("MainWindow", "旋转常量 C"))
        self.comboBoxHistType.setItemText(3, _translate("MainWindow", "偶极矩"))
        self.comboBoxHistType.setItemText(4, _translate("MainWindow", "各向同性极化率"))
        self.comboBoxHistType.setItemText(5, _translate("MainWindow", "HOMO"))
        self.comboBoxHistType.setItemText(6, _translate("MainWindow", "LUMO"))
        self.comboBoxHistType.setItemText(7, _translate("MainWindow", "带隙"))
        self.comboBoxHistType.setItemText(8, _translate("MainWindow", "空间最概然位置"))
        self.comboBoxHistType.setItemText(9, _translate("MainWindow", "零点振动能"))
        self.comboBoxHistType.setItemText(10, _translate("MainWindow", "内能(0K)"))
        self.comboBoxHistType.setItemText(11, _translate("MainWindow", "内能(RT)"))
        self.comboBoxHistType.setItemText(12, _translate("MainWindow", "焓(RT)"))
        self.comboBoxHistType.setItemText(13, _translate("MainWindow", "Gibbs自由能(RT)"))
        self.comboBoxHistType.setItemText(14, _translate("MainWindow", "热容(RT)"))
        self.labelSplit.setText(_translate("MainWindow", "Train:Total"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Molecule Manipulation"))
        self.label_2.setText(_translate("MainWindow", "功能改进中..."))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page), _translate("MainWindow", "Molecule"))
        self.groupBox.setTitle(_translate("MainWindow", "Model Setting"))
        self.label_3.setText(_translate("MainWindow", "模型选择"))
        self.comboBoxModelSelect.setItemText(0, _translate("MainWindow", "GraphSAGE"))
        self.comboBoxModelSelect.setItemText(1, _translate("MainWindow", "GCN"))
        self.comboBoxModelSelect.setItemText(2, _translate("MainWindow", "MPNN"))
        self.comboBoxModelSelect.setItemText(3, _translate("MainWindow", "RandomForest"))
        self.comboBoxModelSelect.setItemText(4, _translate("MainWindow", "XGBoost"))
        self.label_7.setText(_translate("MainWindow", "预测属性"))
        self.comboBoxPropertyPredict.setItemText(0, _translate("MainWindow", "偶极矩"))
        self.comboBoxPropertyPredict.setItemText(1, _translate("MainWindow", "各向同性极化率"))
        self.comboBoxPropertyPredict.setItemText(2, _translate("MainWindow", "HOMO"))
        self.comboBoxPropertyPredict.setItemText(3, _translate("MainWindow", "LUMO"))
        self.comboBoxPropertyPredict.setItemText(4, _translate("MainWindow", "带隙"))
        self.comboBoxPropertyPredict.setItemText(5, _translate("MainWindow", "空间最概然位置"))
        self.comboBoxPropertyPredict.setItemText(6, _translate("MainWindow", "零点振动能"))
        self.comboBoxPropertyPredict.setItemText(7, _translate("MainWindow", "内能(0K)"))
        self.comboBoxPropertyPredict.setItemText(8, _translate("MainWindow", "内能(RT)"))
        self.comboBoxPropertyPredict.setItemText(9, _translate("MainWindow", "焓(RT)"))
        self.comboBoxPropertyPredict.setItemText(10, _translate("MainWindow", "Gibbs自由能(RT)"))
        self.comboBoxPropertyPredict.setItemText(11, _translate("MainWindow", "热容(RT)"))
        self.groupBox_3.setTitle(_translate("MainWindow", "New or Existed"))
        self.checkBoxUseNew.setText(_translate("MainWindow", "基于数据集重新训练"))
        self.checkBoxUseExisted.setText(_translate("MainWindow", "基于已有模型预测"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Parameter Regulation"))
        self.label.setText(_translate("MainWindow", "功能改进中..."))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Model"))
        self.groupBoxGraph2D.setTitle(_translate("MainWindow", "分子图结构"))
        self.labelFigure1.setText(_translate("MainWindow", "Figure1"))
        self.labelFigure2.setText(_translate("MainWindow", "Figure2"))
        self.labelFigure1_2.setText(_translate("MainWindow", "Figure1"))
        self.labelFigure2_2.setText(_translate("MainWindow", "Figure2"))
        self.groupBoxMoleculeInfo.setTitle(_translate("MainWindow", "分子属性信息"))
        self.label_4.setText(_translate("MainWindow", "Smiles表示"))
        self.pushButtonPositions.setText(_translate("MainWindow", "原子空间位置(...)"))
        self.label_5.setText(_translate("MainWindow", "InChI表示"))
        self.pushButtonCharges.setText(_translate("MainWindow", "原子核电荷数(...)"))
        item = self.tableWidgetProperties.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "旋转常量 A"))
        item = self.tableWidgetProperties.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "旋转常量 B"))
        item = self.tableWidgetProperties.verticalHeaderItem(2)
        item.setText(_translate("MainWindow", "旋转常量 C"))
        item = self.tableWidgetProperties.verticalHeaderItem(3)
        item.setText(_translate("MainWindow", "偶极矩"))
        item = self.tableWidgetProperties.verticalHeaderItem(4)
        item.setText(_translate("MainWindow", "各向同性极化率"))
        item = self.tableWidgetProperties.verticalHeaderItem(5)
        item.setText(_translate("MainWindow", "HOMO"))
        item = self.tableWidgetProperties.verticalHeaderItem(6)
        item.setText(_translate("MainWindow", "LUMO"))
        item = self.tableWidgetProperties.verticalHeaderItem(7)
        item.setText(_translate("MainWindow", "带隙"))
        item = self.tableWidgetProperties.verticalHeaderItem(8)
        item.setText(_translate("MainWindow", "空间最概然位置"))
        item = self.tableWidgetProperties.verticalHeaderItem(9)
        item.setText(_translate("MainWindow", "零点振动能"))
        item = self.tableWidgetProperties.verticalHeaderItem(10)
        item.setText(_translate("MainWindow", "内能(0K)"))
        item = self.tableWidgetProperties.verticalHeaderItem(11)
        item.setText(_translate("MainWindow", "内能(RT)"))
        item = self.tableWidgetProperties.verticalHeaderItem(12)
        item.setText(_translate("MainWindow", "焓(RT)"))
        item = self.tableWidgetProperties.verticalHeaderItem(13)
        item.setText(_translate("MainWindow", "Gibbs自由能(RT)"))
        item = self.tableWidgetProperties.verticalHeaderItem(14)
        item.setText(_translate("MainWindow", "热容(RT)"))
        item = self.tableWidgetProperties.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Value"))
        item = self.tableWidgetProperties.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Unit"))
        self.groupBoxGraph3D.setTitle(_translate("MainWindow", "三维结构"))
        self.labelFigure3.setText(_translate("MainWindow", "Figure3"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "分子信息"))
        self.groupBoxStatistics.setTitle(_translate("MainWindow", "统计信息"))
        self.groupBoxTrainTest.setTitle(_translate("MainWindow", "训练/预测"))
        self.label_8.setText(_translate("MainWindow", "功能改进中..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "数据集信息"))
        self.groupBoxModelView.setTitle(_translate("MainWindow", "模型视图"))
        self.label_9.setText(_translate("MainWindow", "功能改进中..."))
        self.groupBoxResult.setTitle(_translate("MainWindow", "预测结果"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "网络模型"))
        self.menu_F.setTitle(_translate("MainWindow", "文件(&F)"))
        self.menu_M.setTitle(_translate("MainWindow", "模型(&M)"))
        self.menu.setTitle(_translate("MainWindow", "模型说明"))
        self.menu_A.setTitle(_translate("MainWindow", "关于(&A)"))
        self.actionOpenFile.setText(_translate("MainWindow", "打开文件"))
        self.actionOpenFile.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionOpenFolder.setText(_translate("MainWindow", "打开目录"))
        self.actionOpenFolder.setShortcut(_translate("MainWindow", "Ctrl+Shift+O"))
        self.actionClose.setText(_translate("MainWindow", "退出"))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+W"))
        self.actionSave.setText(_translate("MainWindow", "保存"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionSaveAs.setText(_translate("MainWindow", "另存为"))
        self.actionSaveAs.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))
        self.actionTrain.setText(_translate("MainWindow", "训练"))
        self.actionTrain.setShortcut(_translate("MainWindow", "Ctrl+T"))
        self.actionPredict.setText(_translate("MainWindow", "预测"))
        self.actionPredict.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionHelp.setText(_translate("MainWindow", "帮助"))
        self.actionAbout.setText(_translate("MainWindow", "关于"))
        self.actionAboutQT.setText(_translate("MainWindow", "Qt Info"))
        self.actionGraphSAGE.setText(_translate("MainWindow", "GraphSAGE"))
        self.actionGCN.setText(_translate("MainWindow", "GCN"))
        self.actionXGBoost.setText(_translate("MainWindow", "XGBoost"))

