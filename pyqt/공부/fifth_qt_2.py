import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        col = QColor(0,0,0)

        self.btn = QPushButton('Dialog',self)
        self.btn.move(20,20)

        self.btn.clicked.connect(self.showDialog)

        self.frm = QFrame(self)
        self.frm.setStyleSheet("QWidget { background-color: %s}"%col.name())
        self.frm.setGeometry(130,22,100,100)

        self.setGeometry(300,300,250,180)
        self.setWindowTitle("color dialog")
        self.show()
        
    def showDialog(self):
        col = QColorDialog.getColor()
        
        if col.isValid :
            self.frm.setStyleSheet("QWidget { background-color: %s}"%col.name())


app = QApplication(sys.argv)
w = Example()
sys.exit(app.exec_())