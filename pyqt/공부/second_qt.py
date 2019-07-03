# 메뉴, 서브그룹 메뉴 만들기.
# 메뉴에 액션 추가하기.

# 체크메뉴, 컨텍스트 메뉴(오른쪽 클릭하면 나오는 메뉴!) 만들기

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QMenu,qApp

class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.statusBar()
        self.statusBar().showMessage("hello")

        menu = self.menuBar()           # 메뉴바 생성
        menu_file = menu.addMenu('File')# 그룹 생성
        menu_edit = menu.addMenu('Edit')# 그룹 생성
        menu_view = menu.addMenu('View')

        file_exit = QAction('Exit',self)# 메뉴 객체 생성
        file_exit.setShortcut('Ctrl+Q')
        file_exit.setStatusTip("누르면 영원히 빠이빠이")
        file_exit.triggered.connect(qApp.quit)

        new_txt = QAction("text file",self)
        new_py = QAction("py file",self)

        view_stat = QAction("status_bar",self,checkable = True)
        view_stat.setChecked(True)
        view_stat.triggered.connect(self.tglStat)

        file_new = QMenu('New',self)

        file_new.addAction(new_txt)
        file_new.addAction(new_py)

        menu_view.addAction(view_stat)
        menu_file.addMenu(file_new)  # 메뉴 등록
        menu_file.addAction(file_exit)  # 메뉴 등록

        self.resize(450,400)
        self.show()

    def tglStat(self, state):
        if state :
            self.statusBar().show()
        else:
            self.statusBar().hide()

    def contextMenuEvent(self,QContextMenuEvent):
        cm = QMenu(self)

        quit = cm.addAction("Quit")

        action = cm.exec_(self.mapToGlobal(QContextMenuEvent.pos()))
        if action == quit : 
            qApp.quit()


app = QApplication(sys.argv)
w = Example()
sys.exit(app.exec_())
