import sys, os, cv2
import numpy as np
import PyQt5
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyQt5 import uic

DIR = os.getcwd()
UIDIR = os.path.join(DIR,"grabcut.ui")

class MainDialog(QDialog):
    def __init__(self):
        QDialog.__init__(self,None)
        uic.loadUi(UIDIR,self)
        self.rect_btn_pushed = 0
        self.scribble_btn_pushed = 0
        self.FG_btn_pushed = 0
        self.BG_btn_pushed = 0
        self.scribble_size = 5
        self.pixmap_image = ''
        self.pixmap_image_c = ''
        
        self.label = QLabel()
        self.bbox_pos1 = [0,0]
        self.bbox_pos2 = [0,0]
        self.scribble_pos = [None,None]
        
        self.open_btn.clicked.connect(lambda state, button=self.open_btn : self.btn_clicked(state, button))
        self.run_btn.clicked.connect(lambda state, button=self.run_btn : self.btn_clicked(state, button))
        self.E_btn.clicked.connect(lambda state, button=self.E_btn : self.btn_clicked(state, button))
        self.rect_btn.clicked.connect(lambda state, button=self.rect_btn : self.btn_clicked(state, button))
        self.scribble_btn.clicked.connect(lambda state, button=self.scribble_btn : self.btn_clicked(state, button))
        self.FG_btn.clicked.connect(lambda state, button=self.FG_btn : self.btn_clicked(state, button))
        self.BG_btn.clicked.connect(lambda state, button=self.BG_btn : self.btn_clicked(state, button))
        self.scribble_size_spin.valueChanged.connect(self.spinBoxChanged)
        

    def btn_clicked(self,state,btn):
        if btn.text() == 'open':
            fname = QFileDialog.getOpenFileName(self)
            self.label.setText(fname[0])
            self.pixmap_image = QPixmap(self.label.text())
            self.pixmap_image_c = QPixmap(self.label.text())
            
            self.palette_map.setPixmap(self.pixmap_image)

            self.image = cv2.imread(self.label.text())
            self.masking_image = np.zeros_like(self.image,dtype=np.float)

        elif btn.text() == 'run' : 
            cv2.imshow('title',self.masking_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif btn.text() == 'erase' : 
            self.pixmap_image = QPixmap(self.label.text())
            self.palette_map.setPixmap(self.pixmap_image)

        elif btn.text() == 'rectangle' : 
            self.rect_btn_pushed = 1
            self.scribble_btn_pushed = 0
            
        elif btn.text() == 'scribble' : 
            self.rect_btn_pushed = 0
            self.scribble_btn_pushed = 1

        elif btn.text() == 'Foreground':
            self.FG_btn_pushed = 1
            self.BG_btn_pushed = 0
            
        elif btn.text() == 'Background':
            self.FG_btn_pushed = 0
            self.BG_btn_pushed = 1

    def spinBoxChanged(self):
        val = self.scribble_size_spin.value()
        self.scribble_size = val
            

    def paintEvent(self, event):
        if self.rect_btn_pushed == 1:
            width = self.bbox_pos2[0]-self.bbox_pos1[0]
            height = self.bbox_pos2[1] - self.bbox_pos1[1]
            
            self.qp1 = QPainter(self.pixmap_image)
            self.qp1.setPen(QPen(Qt.black,3))
            self.qp1.drawRect(self.bbox_pos1[0], self.bbox_pos1[1], width, height)
            self.palette_map.setPixmap(self.pixmap_image)
            self.qp1.end()
            
        elif self.scribble_btn_pushed == 1 :
            if self.FG_btn_pushed == 1 :
                self.qp2 = QPainter(self.pixmap_image)
                self.qp2.setBrush(QColor(200, 0, 0))
                self.qp2.setPen(QColor(200, 0, 0))
                self.qp2.drawRect(self.scribble_pos[0], self.scribble_pos[1],self.scribble_size,self.scribble_size)
                self.palette_map.setPixmap(self.pixmap_image)
                self.qp2.end()

            elif self.BG_btn_pushed == 1 :
                self.qp2 = QPainter(self.pixmap_image)
                self.qp2.setBrush(QColor(200, 100, 0))
                self.qp2.setPen(QColor(200, 100, 0))
                self.qp2.drawRect(self.scribble_pos[0], self.scribble_pos[1],self.scribble_size,self.scribble_size)
                self.palette_map.setPixmap(self.pixmap_image)
                self.qp2.end()


    def mousePressEvent(self, event): 
        if self.rect_btn_pushed == 1:
            self.bbox_pos1[0], self.bbox_pos1[1] = event.pos().x(), event.pos().y()
            self.bbox_pos2[0], self.bbox_pos2[1] = event.pos().x(), event.pos().y()
            self.update() 

        elif self.scribble_btn_pushed == 1 :
            self.scribble_pos[0], self.scribble_pos[1] = event.pos().x() - self.scribble_size// 2 - 2, event.pos().y() - self.scribble_size // 2 - 2
            self.update() 

            if self.FG_btn_pushed == 1:
                self.masking_image[self.scribble_pos[1]-self.scribble_size // 2 : self.scribble_pos[1]+self.scribble_size // 2,
                                    self.scribble_pos[0]-self.scribble_size // 2 : self.scribble_pos[0] + self.scribble_size // 2] = 1.0
            elif self.BG_btn_pushed == 1:
                self.masking_image[self.scribble_pos[1]-self.scribble_size // 2 : self.scribble_pos[1]+self.scribble_size // 2,
                                self.scribble_pos[0]-self.scribble_size // 2 : self.scribble_pos[0] + self.scribble_size // 2] = 0.0

    def mouseMoveEvent(self, event): 

        if self.rect_btn_pushed == 1:
            pass            
            # self.palette_map.setPixmap(QPixmap(self.label.text()))
            # self.bbox_pos2[0], self.bbox_pos2[1] = event.pos().x(), event.pos().y()
            # self.update() 

        elif self.scribble_btn_pushed == 1:
            self.scribble_pos[0], self.scribble_pos[1] = event.pos().x() - self.scribble_size // 2 - 2, event.pos().y() - self.scribble_size // 2 - 2

            if self.FG_btn_pushed == 1:
                self.masking_image[self.scribble_pos[1]-self.scribble_size // 2 : self.scribble_pos[1]+self.scribble_size // 2,
                                    self.scribble_pos[0]-self.scribble_size // 2 : self.scribble_pos[0] + self.scribble_size // 2] = 1.0
            elif self.BG_btn_pushed == 1:
                self.masking_image[self.scribble_pos[1]-self.scribble_size // 2 : self.scribble_pos[1]+self.scribble_size // 2,
                                self.scribble_pos[0]-self.scribble_size // 2 : self.scribble_pos[0] + self.scribble_size // 2] = 0.0

    def mouseReleaseEvent(self, event): 

        if self.rect_btn_pushed == 1:
            self.bbox_pos2[0], self.bbox_pos2[1] = event.pos().x(), event.pos().y()
            self.update()

            if self.bbox_pos1[1] <= self.bbox_pos2[1] :
                self.x_min = self.bbox_pos1[1]
                self.x_max = self.bbox_pos2[1]
            else:
                self.x_min = self.bbox_pos2[1]
                self.x_max = self.bbox_pos1[1]

            if self.bbox_pos1[0] <= self.bbox_pos2[0] :
                self.y_min = self.bbox_pos1[0]
                self.y_max = self.bbox_pos2[0]
            else:
                self.y_min = self.bbox_pos2[0]
                self.y_max = self.bbox_pos1[0]

            self.masking_image[self.x_min:self.x_max,self.y_min:self.y_max] = 1.0

        elif self.scribble_btn_pushed == 1 :
            self.scribble_pos[0], self.scribble_pos[1] = event.pos().x() - self.scribble_size // 2 - 2, event.pos().y() - self.scribble_size // 2 - 2
            self.update()

            if self.FG_btn_pushed == 1:
                self.masking_image[self.scribble_pos[1] - self.scribble_size // 2 : self.scribble_pos[1] + self.scribble_size // 2,
                                    self.scribble_pos[0] - self.scribble_size // 2 : self.scribble_pos[0] + self.scribble_size // 2] = 1.0
            elif self.BG_btn_pushed == 1:
                self.masking_image[self.scribble_pos[1]-self.scribble_size // 2 : self.scribble_pos[1]+self.scribble_size // 2,
                                self.scribble_pos[0]-self.scribble_size // 2 : self.scribble_pos[0] + self.scribble_size // 2] = 0.0

app = QApplication(sys.argv)
main_dialog = MainDialog()
main_dialog.show()

app.exec_()
