#!/usr/lib python3.6
#-*- coding:utf-8 -*-

import sys
from PyQt5.QtCore import pyqtSignal,QObject
from PyQt5.QtWidgets import (QMainWindow,QApplication)

class Communicate(QObject):
    closeApp = pyqtSignal()

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.c = Communicate()
        self.c.closeApp.connect(self.close)

        self.setGeometry(300,300,350,300)
        self.setWindowTitle("signal and slot")
        self.show()

    def mouseDoubleClickEvent(self,event):
        self.c.closeApp.emit()

if __name__ =='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())