# -*- coding: utf-8 -*-


import numpy, sys
import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class MyWidget(QWidget): 
    def __init__(self): 
     super().__init__() 
     self.setGeometry(30,30,600,400) 
     self.begin = QPoint() 
     self.end = QPoint() 
     self.show() 

    def paintEvent(self, event): 
     qp = QPainter(self) 
     br = QBrush(QColor(100, 10, 10, 40)) 
     qp.setBrush(br) 
     qp.drawRect(QRect(self.begin, self.end))  

    def mousePressEvent(self, event): 
     self.begin = event.pos() 
     self.end = event.pos() 
     self.update() 

    def mouseMoveEvent(self, event): 
     self.end = event.pos() 
     self.update() 

    # def mouseReleaseEvent(self, event): 
    #  self.begin = event.pos() 
    #  self.end = event.pos() 
    #  self.update() 

if __name__ == '__main__': 
    app = QApplication(sys.argv) 
    window = MyWidget() 
    window.show() 
    app.aboutToQuit.connect(app.deleteLater) 
    app.exec_()