#!/usr/lib python3.6
#-*- coding:utf-8 -*-

# 스트레치(빈공간만큼 채워주는 것!)에 대해서 배움.

import sys
from PyQt5.QtWidgets import (QWidget,QPushButton,
                            QHBoxLayout,QVBoxLayout,QApplication)

class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        okButton = QPushButton("OK")
        cancleButton = QPushButton("Cancle")

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(okButton)
        hbox.addWidget(cancleButton)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox) # 이 부분이 중요! 이건 공식같이 외워야 한다고 함.

        self.setGeometry(300,300,300,150)
        self.setWindowTitle("Button")
        self.show()

if __name__ =='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())