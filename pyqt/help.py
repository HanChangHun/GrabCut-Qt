import PyQt5
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import Qt

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "PyQt5 Window"
        self.top = 100
        self.left = 100
        self.width = 680
        self.height = 500

        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top,self.left,self.width,self.height)
        self.show()

    def paintEvent(self,e):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 5 , Qt.SolidLine))

        painter.drawRect(100,15,400,200)


App = QApplication(sys.argv)
window = Window()
App.exec_()