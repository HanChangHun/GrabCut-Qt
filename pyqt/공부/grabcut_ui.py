import sys
import cv2
import numpy as np

import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic

SourceUI = '../grabcut.ui'
 
class MainDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self, None)
        uic.loadUi(SourceUI, self)
        self.imagepath=''

        self.open_btn.clicked.connect(self.OpenClicked)
        self.palette_map.
        
        

    def OpenClicked(self):
        fname = QFileDialog.getOpenFileName(self)
        self.imagepath = fname[0]
        

app = QApplication(sys.argv)
main_dialog = MainDialog()
main_dialog.show()
app.exec_()