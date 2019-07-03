import sys
from PyQt5.QtWidgets import QLineEdit,QInputDialog,QApplication, QWidget, QPushButton,QMessageBox
from PyQt5.QtCore import QCoreApplication

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.btn = QPushButton('Dialog',self)
        self.btn.move(20,20)
        self.btn.clicked.connect(self.showDialog)

        self.le = QLineEdit(self)
        self.le.move(130,22)

        self.resize(500,500)
        self.setWindowTitle('Input Dialog')
        self.show()
        
    def showDialog(self):
        text,ok = QInputDialog.getText(self,'Input Dialog', 'Enter your name : ')

        if ok :
            self.le.setText(text)


app = QApplication(sys.argv)
w = Example()
sys.exit(app.exec_())