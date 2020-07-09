#PyQt5 library
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit, QLabel, QFileDialog, QPlainTextEdit, QTableWidget
#Other libraries 
import sys
import os

class main_window(QDialog):
	def __init__(self):
		super().__init__()

		self.init_UI()

	def init_UI(self):
		self.setGeometry(0,0,1350,695)
		self.setWindowTitle('Migrate files')

		self.setLayout(QVBoxLayout())
		self.layout().addLayout(self.head())
		self.layout().addLayout(self.middle())
		self.layout().addLayout(self.results())

		self.stylesheet = """
		QDialog{
			background-color: #d3e3fc;
		}
		QPushButton{
			background-color: #a8c6fa;
			font-family: "Lucida Console", Courier, monospace;
			font-size: 15px;
		}
		QLabel{
            font-family: "Lucida Console", Courier, monospace;
            font-size: 20px;
		}
		QLabel#lb_1{
            font-family: "Lucida Console", Courier, monospace;
            font-size: 14px;
		}
		QLabel#selected_file{
            font-family: "Lucida Console", Courier, monospace;
            font-size: 10px;
		}
		"""

		self.setStyleSheet(self.stylesheet)
		self.show()

	"""
		Interface basic elements

	"""

	def head(self):
		btn_open_file = QPushButton("Open file", self)
		btn_open_file.clicked.connect(self.go_to_open)

		self.lb_selected_file = QLabel("File path")
		self.lb_selected_file.setObjectName("selected_file")

		lb_n_groups = QLabel("N° groups", self)

		btn_generate_dataframe = QPushButton("Generate dataframe", self)
		btn_generate_dataframe.setObjectName("generate_dataframe")
		btn_generate_dataframe.clicked.connect(self.generate_dataframe)

		head_layout = QHBoxLayout()
		head_layout.addWidget(btn_open_file)
		head_layout.addWidget(self.lb_selected_file)
		head_layout.addWidget(lb_n_groups)
		head_layout.addWidget(btn_generate_dataframe)

		return head_layout

	def middle(self):
		lb_dataframe = QLabel("General" + '\n' + "dataframe")
		lb_dataframe.setObjectName("lb_1")

		dataframe_table = QTableWidget()

		middle_layout = QHBoxLayout()
		middle_layout.addWidget(lb_dataframe)
		middle_layout.addWidget(dataframe_table)

		return middle_layout

	def results(self):
		btn_get_results = QPushButton("Results", self)
		btn_get_results.clicked.connect(self.get_results)

		txt_results = QPlainTextEdit()

		results_layout = QHBoxLayout()
		results_layout.addWidget(btn_get_results)
		results_layout.addWidget(txt_results)

		return results_layout

	"""
		A partir de aquí se codifica la funcionalidad
	"""

	def go_to_open(self):
		filename = QFileDialog.getOpenFileName()
		path = filename[0]
		print(path)

		with open(path, 'r') as origin_file:
			self.lb_selected_file.setText(path)
			#print(origin_file.readline())

	def generate_dataframe(self):
		pass

	def get_results(self):
		pass


if __name__ == '__main__':
	app = QApplication(sys.argv)
	mw = main_window()
	sys.exit(app.exec_())
