# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'grabcut.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1288, 1008)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.open_btn = QtWidgets.QPushButton(Dialog)
        self.open_btn.setMinimumSize(QtCore.QSize(80, 50))
        self.open_btn.setMaximumSize(QtCore.QSize(80, 50))
        self.open_btn.setObjectName("open_btn")
        self.gridLayout.addWidget(self.open_btn, 2, 0, 1, 1)
        self.run_btn = QtWidgets.QPushButton(Dialog)
        self.run_btn.setMinimumSize(QtCore.QSize(70, 50))
        self.run_btn.setMaximumSize(QtCore.QSize(70, 50))
        self.run_btn.setObjectName("run_btn")
        self.gridLayout.addWidget(self.run_btn, 2, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(372, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 7, 1, 1)
        self.E_btn = QtWidgets.QPushButton(Dialog)
        self.E_btn.setMinimumSize(QtCore.QSize(90, 50))
        self.E_btn.setMaximumSize(QtCore.QSize(90, 50))
        self.E_btn.setObjectName("E_btn")
        self.gridLayout.addWidget(self.E_btn, 2, 2, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 5, 1, 1)
        self.rect_btn = QtWidgets.QPushButton(Dialog)
        self.rect_btn.setMinimumSize(QtCore.QSize(70, 50))
        self.rect_btn.setMaximumSize(QtCore.QSize(70, 50))
        self.rect_btn.setObjectName("rect_btn")
        self.gridLayout.addWidget(self.rect_btn, 2, 3, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 2, 6, 1, 1)
        self.scribble_btn = QtWidgets.QPushButton(Dialog)
        self.scribble_btn.setMinimumSize(QtCore.QSize(200, 50))
        self.scribble_btn.setMaximumSize(QtCore.QSize(200, 50))
        self.scribble_btn.setObjectName("scribble_btn")
        self.gridLayout.addWidget(self.scribble_btn, 2, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/jpg/image.jpg"))
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 8)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "GrabCut"))
        self.open_btn.setText(_translate("Dialog", "open"))
        self.run_btn.setText(_translate("Dialog", "run"))
        self.E_btn.setText(_translate("Dialog", "erase"))
        self.label.setText(_translate("Dialog", "scribble size : "))
        self.rect_btn.setText(_translate("Dialog", "rect"))
        self.scribble_btn.setText(_translate("Dialog", "scribble"))

import image_rc
