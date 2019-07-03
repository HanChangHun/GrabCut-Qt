import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QMessageBox
from PyQt5.QtCore import QCoreApplication

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        btn = QPushButton('hello World',self)
        btn.resize(btn.sizeHint())
        btn.setToolTip('this is tool tip<b>HEllo!<b>')
        btn.move(50,50)
        btn.clicked.connect(QCoreApplication.instance().quit)

        self.resize(500,500)
        self.setWindowTitle('first lecture!')
        self.show()
        
    def closeEvent(self,QCloseEvent):
        ans = QMessageBox.question(self,"check quit","do you quit?",
        QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        
        if ans == QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


app = QApplication(sys.argv)
w = Example()
sys.exit(app.exec_())