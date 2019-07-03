#!/usr/lib python3.6
#-*- coding:utf-8 -*-

# 애는 절대좌표를 하는 것이어서 유동적이지 못함.

import sys,os
from PyQt5.QtWidgets import QWidget,QLabel,QApplication

class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        lbl1 = QLabel('Zetcode',self)
        lbl1.move(15,10)

        lbl2 = QLabel('Tutorials',self)
        lbl2.move(35,40)

        lbl3 = QLabel('for beginners',self)
        lbl3.move(55,70)

        self.setGeometry(300,300,250,150)
        self.setWindowTitle('Application')
        self.show()

if __name__ =='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())