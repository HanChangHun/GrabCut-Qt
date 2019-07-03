#!/usr/lib python3.6
#-*- coding:utf-8 -*-

# 그리드 레이아웃을 이용한 간단한 계산기를 만듬. 

import sys
from PyQt5.QtWidgets import (QWidget,QPushButton,
                            QGridLayout,QApplication)

class Example(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        grid = QGridLayout()
        self.setLayout(grid)

        names = ['Cls','Bck','','Close',
                '7','8','9','/',
                '4','5','6','*',
                '1','2','3','-',
                '0','.','=','+']

        positions = [(i,j) for i in range(5) for j in range(4)]

        for position,name in zip(positions,names):

            if name == '':
                continue
            button = QPushButton(name)
            grid.addWidget(button, *position)

        
        self.move(300, 150)
        self.setWindowTitle("Calc")
        self.show()

if __name__ =='__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())