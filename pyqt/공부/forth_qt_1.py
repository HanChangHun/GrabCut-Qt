#!/usr/lib python3.6
#-*- coding:utf-8 -*-

# 원래는 잘 쌓였었는데, lcd를 움직이면 창에 숫자가 출력되게 만듬
# 마우스가 움직였을 때 스테이터스바에 좌표가 출력되게 만듬

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget,QLCDNumber,QSlider,QMainWindow,
                            QVBoxLayout,QApplication)

class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.statusBar()
        self.setMouseTracking(True)

        lcd = QLCDNumber(self)
        sld = QSlider(Qt.Horizontal,self)

        vbox = QVBoxLayout()
        vbox.addWidget(lcd)
        vbox.addWidget(sld)
        
        sld.valueChanged.connect(lcd.display)

        self.setGeometry(300,300,350,300)
        self.setWindowTitle("signal and slot")
        self.show()

    def keyPressEvent(self,e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def mouseMoveEvent(self,e):
        x = e.x()
        y = e.y()

        text = "x : {0}, Y : {1}".format(x, y)
        self.statusBar().showMessage(text)

if __name__ =='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
    