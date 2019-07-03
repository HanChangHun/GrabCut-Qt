#!/usr/lib python3.6
#-*- coding:utf-8 -*-

# 버튼이 눌렸을 때 스테이터스 창에 메세지가 출력되게 만듬.


import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow,QPushButton,QApplication)

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        btn1 = QPushButton("Button1",self)
        btn1.move(30,50)

        btn2 = QPushButton("Button2",self)
        btn2.move(150,50)

        btn1.clicked.connect(self.buttonClicked)
        btn2.clicked.connect(self.buttonClicked)

        self.statusBar()

        self.setGeometry(300,300,350,300)
        self.setWindowTitle("signal and slot")
        self.show()

    def buttonClicked(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + " was pressed.")

if __name__ =='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
    