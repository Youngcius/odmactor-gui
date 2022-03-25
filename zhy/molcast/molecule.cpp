/********************************************************************************
** Form generated from reading UI file 'molecule.ui'
**
** Created by: Qt User Interface Compiler version 5.9.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MOLECULE_H
#define MOLECULE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QToolBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpenFile;
    QAction *actionOpenFolder;
    QAction *actionClose;
    QAction *actionSave;
    QAction *actionSaveAs;
    QAction *actionTrain;
    QAction *actionPredict;
    QAction *actionHelp;
    QAction *actionAbout;
    QAction *actionAboutQT;
    QAction *actionGraphSAGE;
    QAction *actionGCN;
    QAction *actionXGBoost;
    QWidget *centralwidget;
    QGridLayout *gridLayout_9;
    QSplitter *splitter;
    QToolBox *toolBox;
    QWidget *page;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBoxMolecule;
    QGridLayout *gridLayout_2;
    QPushButton *pushButtonWebView;
    QLabel *labelNo;
    QSpinBox *spinBoxNo;
    QGroupBox *groupBoxDataset;
    QGridLayout *gridLayout;
    QLabel *labelHistType;
    QComboBox *comboBoxHistType;
    QLabel *labelSplit;
    QDoubleSpinBox *doubleSpinBoxSplit;
    QGroupBox *groupBox_4;
    QFormLayout *formLayout;
    QLabel *label_2;
    QWidget *page_2;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_6;
    QLabel *label_3;
    QComboBox *comboBoxModelSelect;
    QLabel *label_7;
    QComboBox *comboBoxPropertyPredict;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_16;
    QCheckBox *checkBoxUseNew;
    QCheckBox *checkBoxUseExisted;
    QGroupBox *groupBox_2;
    QFormLayout *formLayout_3;
    QLabel *label;
    QTabWidget *tabWidget;
    QWidget *tab;
    QGridLayout *gridLayout_13;
    QGroupBox *groupBoxGraph2D;
    QGridLayout *gridLayout_12;
    QScrollArea *scrollArea1;
    QWidget *scrollAreaWidgetContents_2;
    QGridLayout *gridLayout_15;
    QLabel *labelFigure1;
    QScrollArea *scrollArea2;
    QWidget *scrollAreaWidgetContents_3;
    QGridLayout *gridLayout_7;
    QLabel *labelFigure2;
    QScrollArea *scrollArea1_2;
    QWidget *scrollAreaWidgetContents_4;
    QGridLayout *gridLayout_14;
    QLabel *labelFigure1_2;
    QScrollArea *scrollArea2_2;
    QWidget *scrollAreaWidgetContents_5;
    QGridLayout *gridLayout_11;
    QLabel *labelFigure2_2;
    QGroupBox *groupBoxMoleculeInfo;
    QGridLayout *gridLayout_3;
    QLabel *label_4;
    QPushButton *pushButtonPositions;
    QLineEdit *lineEditSmiles;
    QLineEdit *lineEditInChI;
    QLabel *label_5;
    QPushButton *pushButtonCharges;
    QTableWidget *tableWidgetProperties;
    QGroupBox *groupBoxGraph3D;
    QGridLayout *gridLayout_10;
    QScrollArea *scrollArea3;
    QWidget *scrollAreaWidgetContents;
    QGridLayout *gridLayout_8;
    QLabel *labelFigure3;
    QWidget *tab_3;
    QGridLayout *gridLayout_4;
    QGroupBox *groupBoxStatistics;
    QGroupBox *groupBoxTrainTest;
    QGridLayout *gridLayout_17;
    QLabel *label_8;
    QWidget *tab_2;
    QGridLayout *gridLayout_5;
    QGroupBox *groupBoxModelView;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_9;
    QGroupBox *groupBoxResult;
    QMenuBar *menubar;
    QMenu *menu_F;
    QMenu *menu_M;
    QMenu *menu;
    QMenu *menu_A;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1024, 948);
        actionOpenFile = new QAction(MainWindow);
        actionOpenFile->setObjectName(QStringLiteral("actionOpenFile"));
        QIcon icon;
        icon.addFile(QString::fromUtf8("../icons/\346\226\207\346\241\243.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenFile->setIcon(icon);
        actionOpenFolder = new QAction(MainWindow);
        actionOpenFolder->setObjectName(QStringLiteral("actionOpenFolder"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8("../icons/\346\226\207\344\273\266.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionOpenFolder->setIcon(icon1);
        actionClose = new QAction(MainWindow);
        actionClose->setObjectName(QStringLiteral("actionClose"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8("../icons/\351\200\200\345\207\272.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionClose->setIcon(icon2);
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName(QStringLiteral("actionSave"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8("../icons/\344\277\235\345\255\230.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionSave->setIcon(icon3);
        actionSaveAs = new QAction(MainWindow);
        actionSaveAs->setObjectName(QStringLiteral("actionSaveAs"));
        QIcon icon4;
        icon4.addFile(QString::fromUtf8("../icons/\345\217\246\345\255\230\344\270\272.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionSaveAs->setIcon(icon4);
        actionTrain = new QAction(MainWindow);
        actionTrain->setObjectName(QStringLiteral("actionTrain"));
        QIcon icon5;
        icon5.addFile(QString::fromUtf8("../icons/\350\256\255\347\273\203.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionTrain->setIcon(icon5);
        actionPredict = new QAction(MainWindow);
        actionPredict->setObjectName(QStringLiteral("actionPredict"));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8("../icons/\346\265\213\350\257\225.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionPredict->setIcon(icon6);
        actionHelp = new QAction(MainWindow);
        actionHelp->setObjectName(QStringLiteral("actionHelp"));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8("../icons/\345\270\256\345\212\251.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionHelp->setIcon(icon7);
        actionAbout = new QAction(MainWindow);
        actionAbout->setObjectName(QStringLiteral("actionAbout"));
        QIcon icon8;
        icon8.addFile(QString::fromUtf8("../icons/\345\205\263\344\272\216.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionAbout->setIcon(icon8);
        actionAboutQT = new QAction(MainWindow);
        actionAboutQT->setObjectName(QStringLiteral("actionAboutQT"));
        QIcon icon9;
        icon9.addFile(QStringLiteral("../icons/qt.svg"), QSize(), QIcon::Normal, QIcon::Off);
        actionAboutQT->setIcon(icon9);
        actionGraphSAGE = new QAction(MainWindow);
        actionGraphSAGE->setObjectName(QStringLiteral("actionGraphSAGE"));
        actionGCN = new QAction(MainWindow);
        actionGCN->setObjectName(QStringLiteral("actionGCN"));
        actionXGBoost = new QAction(MainWindow);
        actionXGBoost->setObjectName(QStringLiteral("actionXGBoost"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        gridLayout_9 = new QGridLayout(centralwidget);
        gridLayout_9->setObjectName(QStringLiteral("gridLayout_9"));
        splitter = new QSplitter(centralwidget);
        splitter->setObjectName(QStringLiteral("splitter"));
        splitter->setOrientation(Qt::Horizontal);
        toolBox = new QToolBox(splitter);
        toolBox->setObjectName(QStringLiteral("toolBox"));
        page = new QWidget();
        page->setObjectName(QStringLiteral("page"));
        page->setGeometry(QRect(0, 0, 331, 833));
        verticalLayout_2 = new QVBoxLayout(page);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        groupBoxMolecule = new QGroupBox(page);
        groupBoxMolecule->setObjectName(QStringLiteral("groupBoxMolecule"));
        gridLayout_2 = new QGridLayout(groupBoxMolecule);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        pushButtonWebView = new QPushButton(groupBoxMolecule);
        pushButtonWebView->setObjectName(QStringLiteral("pushButtonWebView"));

        gridLayout_2->addWidget(pushButtonWebView, 0, 0, 1, 2);

        labelNo = new QLabel(groupBoxMolecule);
        labelNo->setObjectName(QStringLiteral("labelNo"));

        gridLayout_2->addWidget(labelNo, 1, 0, 1, 1);

        spinBoxNo = new QSpinBox(groupBoxMolecule);
        spinBoxNo->setObjectName(QStringLiteral("spinBoxNo"));

        gridLayout_2->addWidget(spinBoxNo, 1, 1, 1, 1);


        verticalLayout_2->addWidget(groupBoxMolecule);

        groupBoxDataset = new QGroupBox(page);
        groupBoxDataset->setObjectName(QStringLiteral("groupBoxDataset"));
        gridLayout = new QGridLayout(groupBoxDataset);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        labelHistType = new QLabel(groupBoxDataset);
        labelHistType->setObjectName(QStringLiteral("labelHistType"));

        gridLayout->addWidget(labelHistType, 0, 0, 1, 1);

        comboBoxHistType = new QComboBox(groupBoxDataset);
        comboBoxHistType->setObjectName(QStringLiteral("comboBoxHistType"));

        gridLayout->addWidget(comboBoxHistType, 0, 1, 1, 1);

        labelSplit = new QLabel(groupBoxDataset);
        labelSplit->setObjectName(QStringLiteral("labelSplit"));

        gridLayout->addWidget(labelSplit, 1, 0, 1, 1);

        doubleSpinBoxSplit = new QDoubleSpinBox(groupBoxDataset);
        doubleSpinBoxSplit->setObjectName(QStringLiteral("doubleSpinBoxSplit"));

        gridLayout->addWidget(doubleSpinBoxSplit, 1, 1, 1, 1);


        verticalLayout_2->addWidget(groupBoxDataset);

        groupBox_4 = new QGroupBox(page);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        formLayout = new QFormLayout(groupBox_4);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        label_2 = new QLabel(groupBox_4);
        label_2->setObjectName(QStringLiteral("label_2"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_2);


        verticalLayout_2->addWidget(groupBox_4);

        toolBox->addItem(page, QStringLiteral("Molecule"));
        page_2 = new QWidget();
        page_2->setObjectName(QStringLiteral("page_2"));
        page_2->setGeometry(QRect(0, 0, 331, 833));
        verticalLayout = new QVBoxLayout(page_2);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        groupBox = new QGroupBox(page_2);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        gridLayout_6 = new QGridLayout(groupBox);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout_6->addWidget(label_3, 0, 0, 1, 1);

        comboBoxModelSelect = new QComboBox(groupBox);
        comboBoxModelSelect->setObjectName(QStringLiteral("comboBoxModelSelect"));

        gridLayout_6->addWidget(comboBoxModelSelect, 0, 1, 1, 1);

        label_7 = new QLabel(groupBox);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_6->addWidget(label_7, 1, 0, 1, 1);

        comboBoxPropertyPredict = new QComboBox(groupBox);
        comboBoxPropertyPredict->setObjectName(QStringLiteral("comboBoxPropertyPredict"));

        gridLayout_6->addWidget(comboBoxPropertyPredict, 1, 1, 1, 1);


        verticalLayout->addWidget(groupBox);

        groupBox_3 = new QGroupBox(page_2);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        gridLayout_16 = new QGridLayout(groupBox_3);
        gridLayout_16->setObjectName(QStringLiteral("gridLayout_16"));
        checkBoxUseNew = new QCheckBox(groupBox_3);
        checkBoxUseNew->setObjectName(QStringLiteral("checkBoxUseNew"));

        gridLayout_16->addWidget(checkBoxUseNew, 0, 0, 1, 1);

        checkBoxUseExisted = new QCheckBox(groupBox_3);
        checkBoxUseExisted->setObjectName(QStringLiteral("checkBoxUseExisted"));

        gridLayout_16->addWidget(checkBoxUseExisted, 1, 0, 1, 1);


        verticalLayout->addWidget(groupBox_3);

        groupBox_2 = new QGroupBox(page_2);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        formLayout_3 = new QFormLayout(groupBox_2);
        formLayout_3->setObjectName(QStringLiteral("formLayout_3"));
        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));

        formLayout_3->setWidget(0, QFormLayout::LabelRole, label);


        verticalLayout->addWidget(groupBox_2);

        toolBox->addItem(page_2, QStringLiteral("Model"));
        splitter->addWidget(toolBox);
        tabWidget = new QTabWidget(splitter);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        gridLayout_13 = new QGridLayout(tab);
        gridLayout_13->setObjectName(QStringLiteral("gridLayout_13"));
        groupBoxGraph2D = new QGroupBox(tab);
        groupBoxGraph2D->setObjectName(QStringLiteral("groupBoxGraph2D"));
        gridLayout_12 = new QGridLayout(groupBoxGraph2D);
        gridLayout_12->setObjectName(QStringLiteral("gridLayout_12"));
        scrollArea1 = new QScrollArea(groupBoxGraph2D);
        scrollArea1->setObjectName(QStringLiteral("scrollArea1"));
        scrollArea1->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QStringLiteral("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 283, 196));
        gridLayout_15 = new QGridLayout(scrollAreaWidgetContents_2);
        gridLayout_15->setObjectName(QStringLiteral("gridLayout_15"));
        labelFigure1 = new QLabel(scrollAreaWidgetContents_2);
        labelFigure1->setObjectName(QStringLiteral("labelFigure1"));
        labelFigure1->setLayoutDirection(Qt::LeftToRight);
        labelFigure1->setAlignment(Qt::AlignCenter);

        gridLayout_15->addWidget(labelFigure1, 0, 0, 1, 1);

        scrollArea1->setWidget(scrollAreaWidgetContents_2);

        gridLayout_12->addWidget(scrollArea1, 0, 0, 1, 1);

        scrollArea2 = new QScrollArea(groupBoxGraph2D);
        scrollArea2->setObjectName(QStringLiteral("scrollArea2"));
        scrollArea2->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName(QStringLiteral("scrollAreaWidgetContents_3"));
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 283, 196));
        gridLayout_7 = new QGridLayout(scrollAreaWidgetContents_3);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        labelFigure2 = new QLabel(scrollAreaWidgetContents_3);
        labelFigure2->setObjectName(QStringLiteral("labelFigure2"));
        labelFigure2->setAlignment(Qt::AlignCenter);

        gridLayout_7->addWidget(labelFigure2, 0, 0, 1, 1);

        scrollArea2->setWidget(scrollAreaWidgetContents_3);

        gridLayout_12->addWidget(scrollArea2, 1, 0, 1, 1);

        scrollArea1_2 = new QScrollArea(groupBoxGraph2D);
        scrollArea1_2->setObjectName(QStringLiteral("scrollArea1_2"));
        scrollArea1_2->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName(QStringLiteral("scrollAreaWidgetContents_4"));
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 283, 196));
        gridLayout_14 = new QGridLayout(scrollAreaWidgetContents_4);
        gridLayout_14->setObjectName(QStringLiteral("gridLayout_14"));
        labelFigure1_2 = new QLabel(scrollAreaWidgetContents_4);
        labelFigure1_2->setObjectName(QStringLiteral("labelFigure1_2"));
        labelFigure1_2->setAlignment(Qt::AlignCenter);

        gridLayout_14->addWidget(labelFigure1_2, 0, 0, 1, 1);

        scrollArea1_2->setWidget(scrollAreaWidgetContents_4);

        gridLayout_12->addWidget(scrollArea1_2, 2, 0, 1, 1);

        scrollArea2_2 = new QScrollArea(groupBoxGraph2D);
        scrollArea2_2->setObjectName(QStringLiteral("scrollArea2_2"));
        scrollArea2_2->setWidgetResizable(true);
        scrollAreaWidgetContents_5 = new QWidget();
        scrollAreaWidgetContents_5->setObjectName(QStringLiteral("scrollAreaWidgetContents_5"));
        scrollAreaWidgetContents_5->setGeometry(QRect(0, 0, 283, 196));
        gridLayout_11 = new QGridLayout(scrollAreaWidgetContents_5);
        gridLayout_11->setObjectName(QStringLiteral("gridLayout_11"));
        labelFigure2_2 = new QLabel(scrollAreaWidgetContents_5);
        labelFigure2_2->setObjectName(QStringLiteral("labelFigure2_2"));
        labelFigure2_2->setAlignment(Qt::AlignCenter);

        gridLayout_11->addWidget(labelFigure2_2, 0, 0, 1, 1);

        scrollArea2_2->setWidget(scrollAreaWidgetContents_5);

        gridLayout_12->addWidget(scrollArea2_2, 3, 0, 1, 1);


        gridLayout_13->addWidget(groupBoxGraph2D, 0, 0, 2, 1);

        groupBoxMoleculeInfo = new QGroupBox(tab);
        groupBoxMoleculeInfo->setObjectName(QStringLiteral("groupBoxMoleculeInfo"));
        gridLayout_3 = new QGridLayout(groupBoxMoleculeInfo);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        label_4 = new QLabel(groupBoxMoleculeInfo);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout_3->addWidget(label_4, 0, 0, 1, 1);

        pushButtonPositions = new QPushButton(groupBoxMoleculeInfo);
        pushButtonPositions->setObjectName(QStringLiteral("pushButtonPositions"));

        gridLayout_3->addWidget(pushButtonPositions, 0, 2, 1, 1);

        lineEditSmiles = new QLineEdit(groupBoxMoleculeInfo);
        lineEditSmiles->setObjectName(QStringLiteral("lineEditSmiles"));

        gridLayout_3->addWidget(lineEditSmiles, 0, 1, 1, 1);

        lineEditInChI = new QLineEdit(groupBoxMoleculeInfo);
        lineEditInChI->setObjectName(QStringLiteral("lineEditInChI"));

        gridLayout_3->addWidget(lineEditInChI, 1, 1, 1, 1);

        label_5 = new QLabel(groupBoxMoleculeInfo);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 1, 0, 1, 1);

        pushButtonCharges = new QPushButton(groupBoxMoleculeInfo);
        pushButtonCharges->setObjectName(QStringLiteral("pushButtonCharges"));

        gridLayout_3->addWidget(pushButtonCharges, 1, 2, 1, 1);

        tableWidgetProperties = new QTableWidget(groupBoxMoleculeInfo);
        if (tableWidgetProperties->columnCount() < 2)
            tableWidgetProperties->setColumnCount(2);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        tableWidgetProperties->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        tableWidgetProperties->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        if (tableWidgetProperties->rowCount() < 15)
            tableWidgetProperties->setRowCount(15);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(0, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(1, __qtablewidgetitem3);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(2, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(3, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(4, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(5, __qtablewidgetitem7);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(6, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(7, __qtablewidgetitem9);
        QTableWidgetItem *__qtablewidgetitem10 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(8, __qtablewidgetitem10);
        QTableWidgetItem *__qtablewidgetitem11 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(9, __qtablewidgetitem11);
        QTableWidgetItem *__qtablewidgetitem12 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(10, __qtablewidgetitem12);
        QTableWidgetItem *__qtablewidgetitem13 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(11, __qtablewidgetitem13);
        QTableWidgetItem *__qtablewidgetitem14 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(12, __qtablewidgetitem14);
        QTableWidgetItem *__qtablewidgetitem15 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(13, __qtablewidgetitem15);
        QTableWidgetItem *__qtablewidgetitem16 = new QTableWidgetItem();
        tableWidgetProperties->setVerticalHeaderItem(14, __qtablewidgetitem16);
        tableWidgetProperties->setObjectName(QStringLiteral("tableWidgetProperties"));
        tableWidgetProperties->horizontalHeader()->setDefaultSectionSize(90);
        tableWidgetProperties->horizontalHeader()->setMinimumSectionSize(90);
        tableWidgetProperties->verticalHeader()->setDefaultSectionSize(20);
        tableWidgetProperties->verticalHeader()->setMinimumSectionSize(19);

        gridLayout_3->addWidget(tableWidgetProperties, 2, 0, 1, 3);


        gridLayout_13->addWidget(groupBoxMoleculeInfo, 0, 1, 1, 1);

        groupBoxGraph3D = new QGroupBox(tab);
        groupBoxGraph3D->setObjectName(QStringLiteral("groupBoxGraph3D"));
        gridLayout_10 = new QGridLayout(groupBoxGraph3D);
        gridLayout_10->setObjectName(QStringLiteral("gridLayout_10"));
        scrollArea3 = new QScrollArea(groupBoxGraph3D);
        scrollArea3->setObjectName(QStringLiteral("scrollArea3"));
        scrollArea3->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 313, 384));
        gridLayout_8 = new QGridLayout(scrollAreaWidgetContents);
        gridLayout_8->setObjectName(QStringLiteral("gridLayout_8"));
        labelFigure3 = new QLabel(scrollAreaWidgetContents);
        labelFigure3->setObjectName(QStringLiteral("labelFigure3"));
        labelFigure3->setAlignment(Qt::AlignCenter);

        gridLayout_8->addWidget(labelFigure3, 0, 0, 1, 1);

        scrollArea3->setWidget(scrollAreaWidgetContents);

        gridLayout_10->addWidget(scrollArea3, 0, 0, 1, 1);


        gridLayout_13->addWidget(groupBoxGraph3D, 1, 1, 1, 1);

        tabWidget->addTab(tab, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QStringLiteral("tab_3"));
        gridLayout_4 = new QGridLayout(tab_3);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        groupBoxStatistics = new QGroupBox(tab_3);
        groupBoxStatistics->setObjectName(QStringLiteral("groupBoxStatistics"));

        gridLayout_4->addWidget(groupBoxStatistics, 0, 0, 1, 1);

        groupBoxTrainTest = new QGroupBox(tab_3);
        groupBoxTrainTest->setObjectName(QStringLiteral("groupBoxTrainTest"));
        gridLayout_17 = new QGridLayout(groupBoxTrainTest);
        gridLayout_17->setObjectName(QStringLiteral("gridLayout_17"));
        label_8 = new QLabel(groupBoxTrainTest);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_17->addWidget(label_8, 0, 0, 1, 1);


        gridLayout_4->addWidget(groupBoxTrainTest, 1, 0, 1, 1);

        tabWidget->addTab(tab_3, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        gridLayout_5 = new QGridLayout(tab_2);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        groupBoxModelView = new QGroupBox(tab_2);
        groupBoxModelView->setObjectName(QStringLiteral("groupBoxModelView"));
        verticalLayout_3 = new QVBoxLayout(groupBoxModelView);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        label_9 = new QLabel(groupBoxModelView);
        label_9->setObjectName(QStringLiteral("label_9"));

        verticalLayout_3->addWidget(label_9);


        gridLayout_5->addWidget(groupBoxModelView, 0, 0, 1, 1);

        groupBoxResult = new QGroupBox(tab_2);
        groupBoxResult->setObjectName(QStringLiteral("groupBoxResult"));

        gridLayout_5->addWidget(groupBoxResult, 1, 0, 1, 1);

        tabWidget->addTab(tab_2, QString());
        splitter->addWidget(tabWidget);

        gridLayout_9->addWidget(splitter, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 1024, 23));
        menu_F = new QMenu(menubar);
        menu_F->setObjectName(QStringLiteral("menu_F"));
        menu_M = new QMenu(menubar);
        menu_M->setObjectName(QStringLiteral("menu_M"));
        menu = new QMenu(menu_M);
        menu->setObjectName(QStringLiteral("menu"));
        QIcon icon10;
        icon10.addFile(QString::fromUtf8("../icons/\345\233\276\347\245\236\347\273\217\347\275\221\347\273\234.svg"), QSize(), QIcon::Normal, QIcon::Off);
        menu->setIcon(icon10);
        menu_A = new QMenu(menubar);
        menu_A->setObjectName(QStringLiteral("menu_A"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        menubar->addAction(menu_F->menuAction());
        menubar->addAction(menu_M->menuAction());
        menubar->addAction(menu_A->menuAction());
        menu_F->addAction(actionOpenFile);
        menu_F->addAction(actionOpenFolder);
        menu_F->addAction(actionSave);
        menu_F->addAction(actionSaveAs);
        menu_F->addAction(actionClose);
        menu_M->addAction(actionTrain);
        menu_M->addAction(actionPredict);
        menu_M->addAction(menu->menuAction());
        menu->addAction(actionGraphSAGE);
        menu->addAction(actionGCN);
        menu->addAction(actionXGBoost);
        menu_A->addAction(actionAbout);
        menu_A->addAction(actionHelp);
        menu_A->addAction(actionAboutQT);

        retranslateUi(MainWindow);
        QObject::connect(actionClose, SIGNAL(triggered()), MainWindow, SLOT(close()));

        toolBox->setCurrentIndex(1);
        tabWidget->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "\345\210\206\345\255\220\345\261\236\346\200\247\351\242\204\346\265\213\347\263\273\347\273\237", Q_NULLPTR));
        actionOpenFile->setText(QApplication::translate("MainWindow", "\346\211\223\345\274\200\346\226\207\344\273\266", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionOpenFile->setShortcut(QApplication::translate("MainWindow", "Ctrl+O", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionOpenFolder->setText(QApplication::translate("MainWindow", "\346\211\223\345\274\200\347\233\256\345\275\225", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionOpenFolder->setShortcut(QApplication::translate("MainWindow", "Ctrl+Shift+O", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionClose->setText(QApplication::translate("MainWindow", "\351\200\200\345\207\272", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionClose->setShortcut(QApplication::translate("MainWindow", "Ctrl+W", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionSave->setText(QApplication::translate("MainWindow", "\344\277\235\345\255\230", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionSave->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionSaveAs->setText(QApplication::translate("MainWindow", "\345\217\246\345\255\230\344\270\272", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionSaveAs->setShortcut(QApplication::translate("MainWindow", "Ctrl+Shift+S", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionTrain->setText(QApplication::translate("MainWindow", "\350\256\255\347\273\203", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionTrain->setShortcut(QApplication::translate("MainWindow", "Ctrl+T", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionPredict->setText(QApplication::translate("MainWindow", "\351\242\204\346\265\213", Q_NULLPTR));
#ifndef QT_NO_SHORTCUT
        actionPredict->setShortcut(QApplication::translate("MainWindow", "Ctrl+P", Q_NULLPTR));
#endif // QT_NO_SHORTCUT
        actionHelp->setText(QApplication::translate("MainWindow", "\345\270\256\345\212\251", Q_NULLPTR));
        actionAbout->setText(QApplication::translate("MainWindow", "\345\205\263\344\272\216", Q_NULLPTR));
        actionAboutQT->setText(QApplication::translate("MainWindow", "Qt Info", Q_NULLPTR));
        actionGraphSAGE->setText(QApplication::translate("MainWindow", "GraphSAGE", Q_NULLPTR));
        actionGCN->setText(QApplication::translate("MainWindow", "GCN", Q_NULLPTR));
        actionXGBoost->setText(QApplication::translate("MainWindow", "XGBoost", Q_NULLPTR));
        groupBoxMolecule->setTitle(QApplication::translate("MainWindow", "Molecule", Q_NULLPTR));
        pushButtonWebView->setText(QApplication::translate("MainWindow", "\346\265\217\350\247\210\345\231\250\344\270\255\346\211\223\345\274\200\357\274\210\350\247\202\345\257\237\344\272\244\344\272\222\345\233\276\345\275\242\357\274\211", Q_NULLPTR));
        labelNo->setText(QApplication::translate("MainWindow", "\346\211\200\350\247\202\345\257\237\345\210\206\345\255\220\357\274\232NO.", Q_NULLPTR));
        groupBoxDataset->setTitle(QApplication::translate("MainWindow", "Dataset", Q_NULLPTR));
        labelHistType->setText(QApplication::translate("MainWindow", "\347\273\237\350\256\241\344\277\241\346\201\257", Q_NULLPTR));
        comboBoxHistType->clear();
        comboBoxHistType->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\270\270\351\207\217 A", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\270\270\351\207\217 B", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\270\270\351\207\217 C", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\201\266\346\236\201\347\237\251", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\220\204\345\220\221\345\220\214\346\200\247\346\236\201\345\214\226\347\216\207", Q_NULLPTR)
         << QApplication::translate("MainWindow", "HOMO", Q_NULLPTR)
         << QApplication::translate("MainWindow", "LUMO", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\270\246\351\232\231", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\347\251\272\351\227\264\346\234\200\346\246\202\347\204\266\344\275\215\347\275\256", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\351\233\266\347\202\271\346\214\257\345\212\250\350\203\275", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\206\205\350\203\275(0K)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\206\205\350\203\275(RT)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\347\204\223(RT)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "Gibbs\350\207\252\347\224\261\350\203\275(RT)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\347\203\255\345\256\271(RT)", Q_NULLPTR)
        );
        labelSplit->setText(QApplication::translate("MainWindow", "Train:Total", Q_NULLPTR));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "Molecule Manipulation", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", "\345\212\237\350\203\275\346\224\271\350\277\233\344\270\255...", Q_NULLPTR));
        toolBox->setItemText(toolBox->indexOf(page), QApplication::translate("MainWindow", "Molecule", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("MainWindow", "Model Setting", Q_NULLPTR));
        label_3->setText(QApplication::translate("MainWindow", "\346\250\241\345\236\213\351\200\211\346\213\251", Q_NULLPTR));
        comboBoxModelSelect->clear();
        comboBoxModelSelect->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "GraphSAGE", Q_NULLPTR)
         << QApplication::translate("MainWindow", "GCN", Q_NULLPTR)
         << QApplication::translate("MainWindow", "MPNN", Q_NULLPTR)
         << QApplication::translate("MainWindow", "RandomForest", Q_NULLPTR)
         << QApplication::translate("MainWindow", "XGBoost", Q_NULLPTR)
        );
        label_7->setText(QApplication::translate("MainWindow", "\351\242\204\346\265\213\345\261\236\346\200\247", Q_NULLPTR));
        comboBoxPropertyPredict->clear();
        comboBoxPropertyPredict->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "\345\201\266\346\236\201\347\237\251", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\220\204\345\220\221\345\220\214\346\200\247\346\236\201\345\214\226\347\216\207", Q_NULLPTR)
         << QApplication::translate("MainWindow", "HOMO", Q_NULLPTR)
         << QApplication::translate("MainWindow", "LUMO", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\270\246\351\232\231", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\347\251\272\351\227\264\346\234\200\346\246\202\347\204\266\344\275\215\347\275\256", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\351\233\266\347\202\271\346\214\257\345\212\250\350\203\275", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\206\205\350\203\275(0K)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\345\206\205\350\203\275(RT)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\347\204\223(RT)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "Gibbs\350\207\252\347\224\261\350\203\275(RT)", Q_NULLPTR)
         << QApplication::translate("MainWindow", "\347\203\255\345\256\271(RT)", Q_NULLPTR)
        );
        groupBox_3->setTitle(QApplication::translate("MainWindow", "New or Existed", Q_NULLPTR));
        checkBoxUseNew->setText(QApplication::translate("MainWindow", "\345\237\272\344\272\216\346\225\260\346\215\256\351\233\206\351\207\215\346\226\260\350\256\255\347\273\203", Q_NULLPTR));
        checkBoxUseExisted->setText(QApplication::translate("MainWindow", "\345\237\272\344\272\216\345\267\262\346\234\211\346\250\241\345\236\213\351\242\204\346\265\213", Q_NULLPTR));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Parameter Regulation", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "\345\212\237\350\203\275\346\224\271\350\277\233\344\270\255...", Q_NULLPTR));
        toolBox->setItemText(toolBox->indexOf(page_2), QApplication::translate("MainWindow", "Model", Q_NULLPTR));
        groupBoxGraph2D->setTitle(QApplication::translate("MainWindow", "\345\210\206\345\255\220\345\233\276\347\273\223\346\236\204", Q_NULLPTR));
        labelFigure1->setText(QApplication::translate("MainWindow", "Figure1", Q_NULLPTR));
        labelFigure2->setText(QApplication::translate("MainWindow", "Figure2", Q_NULLPTR));
        labelFigure1_2->setText(QApplication::translate("MainWindow", "Figure1", Q_NULLPTR));
        labelFigure2_2->setText(QApplication::translate("MainWindow", "Figure2", Q_NULLPTR));
        groupBoxMoleculeInfo->setTitle(QApplication::translate("MainWindow", "\345\210\206\345\255\220\345\261\236\346\200\247\344\277\241\346\201\257", Q_NULLPTR));
        label_4->setText(QApplication::translate("MainWindow", "Smiles\350\241\250\347\244\272", Q_NULLPTR));
        pushButtonPositions->setText(QApplication::translate("MainWindow", "\345\216\237\345\255\220\347\251\272\351\227\264\344\275\215\347\275\256(...)", Q_NULLPTR));
        label_5->setText(QApplication::translate("MainWindow", "InChI\350\241\250\347\244\272", Q_NULLPTR));
        pushButtonCharges->setText(QApplication::translate("MainWindow", "\345\216\237\345\255\220\346\240\270\347\224\265\350\215\267\346\225\260(...)", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem = tableWidgetProperties->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("MainWindow", "Value", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem1 = tableWidgetProperties->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("MainWindow", "Unit", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem2 = tableWidgetProperties->verticalHeaderItem(0);
        ___qtablewidgetitem2->setText(QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\270\270\351\207\217 A", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem3 = tableWidgetProperties->verticalHeaderItem(1);
        ___qtablewidgetitem3->setText(QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\270\270\351\207\217 B", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem4 = tableWidgetProperties->verticalHeaderItem(2);
        ___qtablewidgetitem4->setText(QApplication::translate("MainWindow", "\346\227\213\350\275\254\345\270\270\351\207\217 C", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem5 = tableWidgetProperties->verticalHeaderItem(3);
        ___qtablewidgetitem5->setText(QApplication::translate("MainWindow", "\345\201\266\346\236\201\347\237\251", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem6 = tableWidgetProperties->verticalHeaderItem(4);
        ___qtablewidgetitem6->setText(QApplication::translate("MainWindow", "\345\220\204\345\220\221\345\220\214\346\200\247\346\236\201\345\214\226\347\216\207", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem7 = tableWidgetProperties->verticalHeaderItem(5);
        ___qtablewidgetitem7->setText(QApplication::translate("MainWindow", "HOMO", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem8 = tableWidgetProperties->verticalHeaderItem(6);
        ___qtablewidgetitem8->setText(QApplication::translate("MainWindow", "LUMO", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem9 = tableWidgetProperties->verticalHeaderItem(7);
        ___qtablewidgetitem9->setText(QApplication::translate("MainWindow", "\345\270\246\351\232\231", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem10 = tableWidgetProperties->verticalHeaderItem(8);
        ___qtablewidgetitem10->setText(QApplication::translate("MainWindow", "\347\251\272\351\227\264\346\234\200\346\246\202\347\204\266\344\275\215\347\275\256", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem11 = tableWidgetProperties->verticalHeaderItem(9);
        ___qtablewidgetitem11->setText(QApplication::translate("MainWindow", "\351\233\266\347\202\271\346\214\257\345\212\250\350\203\275", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem12 = tableWidgetProperties->verticalHeaderItem(10);
        ___qtablewidgetitem12->setText(QApplication::translate("MainWindow", "\345\206\205\350\203\275(0K)", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem13 = tableWidgetProperties->verticalHeaderItem(11);
        ___qtablewidgetitem13->setText(QApplication::translate("MainWindow", "\345\206\205\350\203\275(RT)", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem14 = tableWidgetProperties->verticalHeaderItem(12);
        ___qtablewidgetitem14->setText(QApplication::translate("MainWindow", "\347\204\223(RT)", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem15 = tableWidgetProperties->verticalHeaderItem(13);
        ___qtablewidgetitem15->setText(QApplication::translate("MainWindow", "Gibbs\350\207\252\347\224\261\350\203\275(RT)", Q_NULLPTR));
        QTableWidgetItem *___qtablewidgetitem16 = tableWidgetProperties->verticalHeaderItem(14);
        ___qtablewidgetitem16->setText(QApplication::translate("MainWindow", "\347\203\255\345\256\271(RT)", Q_NULLPTR));
        groupBoxGraph3D->setTitle(QApplication::translate("MainWindow", "\344\270\211\347\273\264\347\273\223\346\236\204", Q_NULLPTR));
        labelFigure3->setText(QApplication::translate("MainWindow", "Figure3", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "\345\210\206\345\255\220\344\277\241\346\201\257", Q_NULLPTR));
        groupBoxStatistics->setTitle(QApplication::translate("MainWindow", "\347\273\237\350\256\241\344\277\241\346\201\257", Q_NULLPTR));
        groupBoxTrainTest->setTitle(QApplication::translate("MainWindow", "\350\256\255\347\273\203/\351\242\204\346\265\213", Q_NULLPTR));
        label_8->setText(QApplication::translate("MainWindow", "\345\212\237\350\203\275\346\224\271\350\277\233\344\270\255...", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("MainWindow", "\346\225\260\346\215\256\351\233\206\344\277\241\346\201\257", Q_NULLPTR));
        groupBoxModelView->setTitle(QApplication::translate("MainWindow", "\346\250\241\345\236\213\350\247\206\345\233\276", Q_NULLPTR));
        label_9->setText(QApplication::translate("MainWindow", "\345\212\237\350\203\275\346\224\271\350\277\233\344\270\255...", Q_NULLPTR));
        groupBoxResult->setTitle(QApplication::translate("MainWindow", "\351\242\204\346\265\213\347\273\223\346\236\234", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "\347\275\221\347\273\234\346\250\241\345\236\213", Q_NULLPTR));
        menu_F->setTitle(QApplication::translate("MainWindow", "\346\226\207\344\273\266(&F)", Q_NULLPTR));
        menu_M->setTitle(QApplication::translate("MainWindow", "\346\250\241\345\236\213(&M)", Q_NULLPTR));
        menu->setTitle(QApplication::translate("MainWindow", "\346\250\241\345\236\213\350\257\264\346\230\216", Q_NULLPTR));
        menu_A->setTitle(QApplication::translate("MainWindow", "\345\205\263\344\272\216(&A)", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MOLECULE_H
