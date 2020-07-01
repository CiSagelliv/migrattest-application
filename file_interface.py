import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QTextEdit, QHBoxLayout, QLabel, QMainWindow, QFileDialog, QPlainTextEdit

class main_window(QWidget):
	def __init__(self):
		super().__init__()

		self.init_UI()
		
	def init_UI(self):

		self.setGeometry(0,0,1350,695)
		self.setWindowTitle('Migrate files')

		self.setLayout(QHBoxLayout())
		self.layout().addLayout(self.open_and_show_file())

		self.show()

	def open_and_show_file(self):
		btn_open_file = QPushButton("Open file", self)
		btn_open_file.setObjectName("open_file")
		btn_open_file.clicked.connect(self.go_to_open)

		lb_info = QLabel()
		lb_info.setObjectName("lb_info")
		self.txt_file = QPlainTextEdit()

		file_layout = QHBoxLayout()
		file_layout.addWidget(btn_open_file)
		file_layout.addWidget(lb_info)
		file_layout.addWidget(self.txt_file)

		return file_layout

	def go_to_open(self):
		self.txt_file.isReadOnly()
		filename = QFileDialog.getOpenFileName(self, 'Open file', "/Documents")
		if filename[0]:
			file = open(filename[0], 'r', encoding="utf-8")
			with file:
				self.data = file.read()
				self.txt_file.setPlainText(self.data)



if __name__ == '__main__':
	app = QApplication(sys.argv)
	mw = main_window()
	sys.exit(app.exec_())
