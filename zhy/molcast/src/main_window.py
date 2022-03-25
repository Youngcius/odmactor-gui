import os
import sys
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWebEngineWidgets
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import molecule
import data_process
import datetime
import webbrowser
from model_process import MoleculePredictor
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt

# QtCore.QUrl
################
# web view
################
# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(QtWidgets.QMainWindow, self).__init__()
#         self.setWindowTitle('显示网页')
#         self.resize(800, 800)
#         # 新建一个QWebEngineView()对象
#         self.qwebengine = QtWebEngineWidgets.QWebEngineView(self)
#         # 设置网页在窗口中显示的位置和大小
#         self.qwebengine.setGeometry(20, 20, 600, 600)
#         # 在QWebEngineView中加载网址
#         self.qwebengine.load(QtCore.QUrl(r"https://www.csdn.net/"))

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     win = MainWindow()
#     win.show()
#     app.exec()


# 存储模型默认名字（或者代表了参数文件）
names_model = ['GraphSAGE.pth', 'GCN.pth', 'XGBoost.pth']
indices_model = {k: v for k, v in enumerate(names_model)}


class MoleculeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MoleculeWindow, self).__init__()
        self.ui = molecule.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('../icons/分子.svg'))
        # web tab page
        # self.webview = QtWebEngineWidgets.QWebEngineView(self)
        # self.ui.tab_2.addWidget(self.webview)
        # self.ui.tabWidget.addTab(self.webview)
        # self.model = MoleculePredictor(type=self.ui.comboBoxModelSelect.currentText())  # 模型初始化
        self.models_existed = {'GraphSAGE': MoleculePredictor('GraphSAGE'), 'GCN': MoleculePredictor('GCN'),
                               'XGBoost': MoleculePredictor('XGBoost')}
        self.models_new = {'GraphSAGE': MoleculePredictor('GraphSAGE'), 'GCN': MoleculePredictor('GCN'),
                           'XGBoost': MoleculePredictor('XGBoost')}
        self.predictor = self.models_existed['GraphSAGE']
        self.initModels('../output')
        self.initTabWidget()
        self.initStatusbar()
        self.initToolBox()
        self.initStatictialFigure()
        self.loaded = False  # 表示数据未载入
        self.autoLoadDataset()  ####################################

    def autoLoadDataset(self):
        # folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择分子数据文件', '../')
        # self.qm9 = data_process.QM9Dataset(raw_dir=folder)
        # 初始化一些UI控件属性
        # print(self.qm9.dict['Smiles'])
        # print(self.qm9.prop_names)
        folder = 'E:/VSCode/Python/moleculer-property/data/dsgdb9nsd.xyz'
        self.qm9 = data_process.QM9Dataset(raw_dir=folder)
        self.ui.spinBoxNo.setMinimum(0)
        self.ui.spinBoxNo.setMaximum(len(self.qm9.dict['atom_nums']) - 1)
        self.ui.doubleSpinBoxSplit.setMaximum(1.0)
        self.ui.doubleSpinBoxSplit.setMinimum(0.0)
        self.ui.doubleSpinBoxSplit.setSingleStep(0.01)
        self.ui.doubleSpinBoxSplit.setValue(0.7)
        self.ui.statusbar.addWidget(QtWidgets.QLabel('Current dataset folder: {}'.format(folder)))
        self.loaded = True  # 已经有数据集载入
        self.plot_stat_ax1()

    def initStatictialFigure(self):
        # mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei']  # 汉字字体
        plt.style.use('seaborn')
        self.stat_fig = plt.figure(figsize=(8, 5))
        self.statFigureCanvas = FigureCanvas(self.stat_fig)
        self.statFigureCanvas.setParent(self.ui.groupBoxStatistics)
        gridLayout_tmp = QtWidgets.QGridLayout(self.ui.groupBoxStatistics)
        gridLayout_tmp.addWidget(self.statFigureCanvas, 0, 0, 1, 1)
        self.stat_fig.suptitle('Statistical Information of Current Dataset')
        self.stat_ax1 = self.stat_fig.add_subplot(1, 2, 1)  # 分子中原子数
        self.stat_ax2 = self.stat_fig.add_subplot(1, 2, 2)  # 某个选定属性的统计图（hist）
        self.stat_ax1.set_title('Number of Atoms')
        self.stat_ax2.set_title('Selected Property')

    # def

    ###################
    # 核心方法：训练 & 预测
    ####################

    def initModels(self, model_dir):
        fnames = os.listdir(model_dir)
        for fn in fnames:
            if fn in names_model:
                # self.models_old[fn.split('.')[0]]    <--- 'fn'   ..........
                pass

    def train(self, model, dataloader, epoch=6, lr=0.005):
        for e in epoch:
            for i, (train_data, train_labels) in enumerate(dataloader):
                # 进度条示意
                pass

    def validate(self, model):
        pass

    def predict(self, model):
        pass

    def initTabWidget(self):
        pass

    def initToolBox(self):
        self.ui.checkBoxUseExisted.setChecked(True)
        self.ui.checkBoxUseNew.setChecked(False)
        self.checkGroup = QtWidgets.QButtonGroup(self)
        self.checkGroup.addButton(self.ui.checkBoxUseNew)
        self.checkGroup.addButton(self.ui.checkBoxUseExisted)
        self.checkGroup.setExclusive(True)

    def initStatusbar(self):
        pass

    # 模型选择
    @QtCore.pyqtSlot(int)
    def on_comboBoxModelSelect_currentIndexChanged(self, index: int):
        """
        0: GraphSAGE
        1: GCN
        2: XGBoost
        """
        # self.predictor = self.models[indices_model[index]]
        print('model:', index, self.ui.comboBoxModelSelect.currentText())

    # 模型描述
    @QtCore.pyqtSlot()
    def on_actionGraphSAGE_triggered(self):
        # self.ui.comboBoxModelSelect.setCurrentText('GraphSAGE')
        txt = ''
        txt += 'Description:\t{}\n'.format('......................................')
        txt += 'Reference:\t{}\n'.format('An article.............................')
        txt += 'Github:\t{}\n'.format('....................')
        QtWidgets.QMessageBox.about(self, 'GraphSAGE图神经网络', txt)

    # 预测属性选择：选取不同类型的 Dataloader (已经都预先生成）
    @QtCore.pyqtSlot()
    def on_comboBoxPropertyPredict_currentIndexChanged(self, index: int):
        """
        index: 0, ..., 12
        """
        index += 3


    @QtCore.pyqtSlot()
    def on_actionGraphSAGE_triggered(self):
        # self.ui.comboBoxModelSelect.setCurrentText('GraphSAGE')
        txt = ''
        txt += 'Description:\t{}\n'.format('......................................')
        txt += 'Reference:\t{}\n'.format('An article.............................')
        txt += 'Github:\t{}\n'.format('....................')

        QtWidgets.QMessageBox.about(self, 'GCN', txt)

    @QtCore.pyqtSlot()
    def on_actionGraphSAGE_triggered(self):
        # self.ui.comboBoxModelSelect.setCurrentText('GraphSAGE')
        txt = ''
        txt += 'Description:\t{}\n'.format('......................................')
        txt += 'Reference:\t{}\n'.format('An article.............................')
        txt += 'Github:\t{}\n'.format('....................')

        QtWidgets.QMessageBox.about(self, 'XGBoost机器学习模型', txt)

    # @QtCore.pyqtSlot()
    # def on_actionGCN_triggered(self):
    #     self.ui.comboBoxModelSelect.setCurrentText('GCN')
    #
    # @QtCore.pyqtSlot()
    # def on_actionXGBoost_triggered(self):
    #     self.ui.comboBoxModelSelect.setCurrentText('XGBoost')

    @QtCore.pyqtSlot()
    def on_actionOpenFile_triggered(self):
        print('打开分子数据文件啦')
        fmt_filter = 'XYZ文件(*.xyz);; MOL文件(*.xyz);; Smile文件(.smile)'
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择分子数据文件', '../', fmt_filter)
        print(fname)

    @QtCore.pyqtSlot()
    def on_actionOpenFolder_triggered(self):
        print('选择分子数据文件夹')
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择分子数据文件', '../')
        self.qm9 = data_process.QM9Dataset(raw_dir=folder)
        # 初始化一些UI控件属性
        # print(self.qm9.dict['Smiles'])
        # print(self.qm9.prop_names)
        self.ui.spinBoxNo.setMinimum(0)
        self.ui.spinBoxNo.setMaximum(len(self.qm9.dict['atom_nums']) - 1)
        self.ui.doubleSpinBoxSplit.setMaximum(1.0)
        self.ui.doubleSpinBoxSplit.setMinimum(0.0)
        self.ui.doubleSpinBoxSplit.setSingleStep(0.01)
        self.ui.doubleSpinBoxSplit.setValue(0.7)
        self.ui.statusbar.addWidget(QtWidgets.QLabel('Current dataset folder: {}'.format(folder)))
        self.loaded = True  # 已经有数据集载入
        self.plot_stat_ax1()

    def plot_stat_ax1(self):
        if self.loaded:
            self.stat_ax1.clear()
            self.stat_ax1.hist(self.qm9.dict['atom_nums'], bins=15)
            self.stat_ax1.set_xlabel('Atoms number')
            self.stat_fig.savefig('stat_fig.png')

    def plot_stat_ax2(self, prop_idx=None):
        if prop_idx is None or not self.loaded:
            return
        else:
            self.stat_ax2.clear()
            self.stat_ax2.set_title('Histogram of {} Distribution'.format(self.qm9.prop_names[prop_idx]))
            self.stat_ax2.hist([prop[prop_idx] for prop in self.qm9.dict['properties']], bins=15)
            self.stat_ax2.set_xlabel(self.qm9.prop_names[prop_idx])
            self.stat_fig.savefig('stat_fig.png')

    # 统计信息，TabWidget中第二页（tab_3）
    @QtCore.pyqtSlot(int)
    def on_comboBoxHistType_currentIndexChanged(self, index: int):
        """
        index: 0, ..., 15
        """
        print('hist:', index, self.ui.comboBoxHistType.currentText())
        print('type:', type(index))
        self.plot_stat_ax2(prop_idx=index)

    # 一些功能按键
    @QtCore.pyqtSlot()
    def on_pushButtonWebView_clicked(self):
        if self.loaded:
            idx = self.ui.spinBoxNo.value()
            v = view(self.qm9.dict['all_atoms'][idx], viewer='x3d')
            with open('view.html', 'w') as f:
                f.write(v.data)
            webbrowser.open('view.html')

    @QtCore.pyqtSlot(int)
    def on_spinBoxNo_valueChanged(self, idx: int):
        # mol1 = Chem.MolFromInchi(self.qm9.dict['InChI'][idx][0])
        # mol2 = Chem.MolFromInchi(self.qm9.dict['InChI'][idx][1])
        mol1 = Chem.MolFromSmiles(self.qm9.dict['Smiles'][idx][0])
        mol2 = Chem.MolFromSmiles(self.qm9.dict['Smiles'][idx][1])
        
        print(Chem.MolToSmiles(mol1) + '; ' + Chem.MolToSmiles(mol2))
        print(self.qm9.dict['Smiles'][idx])
        # print
        self.ui.lineEditSmiles.setText(Chem.MolToSmiles(mol1) + '; ' + Chem.MolToSmiles(mol2))
        self.ui.lineEditInChI.setText(Chem.MolToInchi(mol1) + '; ' + Chem.MolToInchi(mol2))

        # 同分异构体 1
        # with Hs & without Hs
        Draw.MolToFile(Chem.AddHs(mol1), 'mol1-Hs.png', size=(500, 300))
        Draw.MolToFile(mol1, 'mol1.png', size=(500, 300))
        pix1 = QtGui.QPixmap('mol1-Hs.png').scaledToHeight(self.ui.scrollArea1.height() - 30)
        pix2 = QtGui.QPixmap('mol1.png').scaledToHeight(self.ui.scrollArea2.height() - 30)
        self.ui.labelFigure1.setPixmap(pix1)
        self.ui.labelFigure2.setPixmap(pix2)

        # 同分异构体 2
        # with Hs & without Hs
        Draw.MolToFile(Chem.AddHs(mol2), 'mol2-Hs.png', size=(500, 300))
        Draw.MolToFile(mol2, 'mol2.png', size=(500, 300))
        pix1_2 = QtGui.QPixmap('mol2-Hs.png').scaledToHeight(self.ui.scrollArea1_2.height() - 30)
        pix2_2 = QtGui.QPixmap('mol2.png').scaledToHeight(self.ui.scrollArea2_2.height() - 30)
        self.ui.labelFigure2_2.setPixmap(pix2_2)
        self.ui.labelFigure1_2.setPixmap(pix1_2)

        # 3-D figure
        ats = self.qm9.dict['all_atoms'][idx]
        ats.write('ats.eps')
        os.system('magick -density {} {} {}'.format(300, 'ats.eps', 'ats.png'))
        self.ui.labelFigure3.setPixmap(QtGui.QPixmap('ats.png').scaledToHeight(self.ui.scrollArea3.width() - 30))

        # 分组属性信息（QTableWidget）
        print('label size:', self.ui.labelFigure1.width(), self.ui.labelFigure1.height())
        print('scroll size:', self.ui.scrollArea1.width(), self.ui.scrollArea1.height())
        prop = self.qm9.dict['properties'][idx]
        for i in range(15):
            item_value = QtWidgets.QTableWidgetItem(str(round(prop[i], 2)))
            item_unit = QtWidgets.QTableWidgetItem(self.qm9.units_str[i])
            item_value.setTextAlignment(QtCore.Qt.AlignCenter)
            item_unit.setTextAlignment(QtCore.Qt.AlignCenter)
            self.ui.tableWidgetProperties.setItem(i, 0, item_value)
            self.ui.tableWidgetProperties.setItem(i, 1, item_unit)

    @QtCore.pyqtSlot()
    def on_pushButtonPositions_clicked(self):
        # 暂时使用info对话框而不是多窗口table
        idx = self.ui.spinBoxNo.value()
        pos = self.qm9.dict['all_atoms'][idx].positions.round(4)
        txt = ''
        for r in pos:
            txt += str(r).strip('[|]').strip().replace('  ', ' ') + '\n'
        QtWidgets.QMessageBox.about(self, 'Positions of Atoms', txt)

    @QtCore.pyqtSlot()
    def on_pushButtonCharges_clicked(self):
        # 暂时使用info对话框而不是多窗口table
        idx = self.ui.spinBoxNo.value()
        num = self.qm9.dict['all_atoms'][idx].numbers  # Z
        cha = self.qm9.dict['charges'][idx]  # charge
        txt1 = '原子序数：\n'
        txt2 = '所带电荷：\n'
        for i, (Z, C) in enumerate(zip(num, cha)):
            txt1 += str(Z) + '\t'
            txt2 += str(round(C, 4)) + '\t'
            if (i + 1) % 6 == 0:
                txt1 += '\n'
                txt2 += '\n'
        txt = txt1 + '\n' + txt2
        QtWidgets.QMessageBox.about(self, 'Charges of Atoms', txt)

    @QtCore.pyqtSlot()
    def on_actionSave_triggered(self):
        print('保存文件')
        fmt_filter = 'XYZ文件(*.xyz);; MOL文件(*.xyz);; Smile文件(.smile)'
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, '选择分子数据文件', '../', fmt_filter)
        print(fname)

    @QtCore.pyqtSlot()
    def on_actionSaveAs_triggered(self):
        print('要保存分子数据文件啦')
        fmt_filter = 'XYZ文件(*.xyz);; MOL文件(*.xyz);; Smile文件(.smile)'
        fname, fmt = QtWidgets.QFileDialog.getSaveFileName(self, '选择分子数据文件', '../', fmt_filter)
        ##################
        # 另存为什么格式
        fmt = fmt.split('.')[-1][:-1]
        data_process.save_data(fname, fmt)
        ##################

        print(fname)

    @QtCore.pyqtSlot()
    def on_actionAbout_triggered(self):
        txt = '版本:\t {}\n'.format('1.0.0')
        txt += '日期:\t {}\n'.format('2021年2月')
        txt += '作者:\t {}\n'.format('杨朝辉')
        txt += '电邮:\t {}\n'.format('1098998018@qq.com')
        # txt += 'Github:\t {}\n'.format('http://')..............................................................................
        txt += '简介:\t {}\n'.format('一个基于深度学习方法的分子数据集的属性预测系统（USTC 2021届本科毕业设计），欢迎任何个人和团体的使用与交流！')
        QtWidgets.QMessageBox.about(self, 'Software Information', txt)

    @QtCore.pyqtSlot()
    def on_actionHelp_triggered(self):
        txt = '请在”关于“页面联系作者。' + '\n' + 'Please contact the author by referring to the "about" page.'
        QtWidgets.QMessageBox.about(self, 'Help', txt)

    @QtCore.pyqtSlot()
    def on_actionAboutQT_triggered(self):
        QtWidgets.QMessageBox.aboutQt(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MoleculeWindow()
    window.show()
    app.exec()
