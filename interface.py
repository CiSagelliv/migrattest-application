#PyQt5 library
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit, QLabel, QFileDialog, QPlainTextEdit, QTableWidget
#Other libraries 
import sys

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
		"""

		self.setStyleSheet(self.stylesheet)
		self.show()

	def head(self):
		btn_open_file = QPushButton("Open file", self)
		#btn_open_file.clicked.connect(self.go_to_open)

		lb_selected_file = QLabel("File title", self)

		lb_n_groups = QLabel("N° groups", self)

		btn_generate_dataframe = QPushButton("Generate dataframe", self)
		btn_generate_dataframe.setObjectName("generate_dataframe")
		#btn_generate_dataframe.clicked.connect(self.go_to_dataframe)

		head_layout = QHBoxLayout()
		head_layout.addWidget(btn_open_file)
		head_layout.addWidget(lb_selected_file)
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
		#btn_get_results.clicked.connect(self.go_to_results)

		txt_results = QPlainTextEdit()

		results_layout = QHBoxLayout()
		results_layout.addWidget(btn_get_results)
		results_layout.addWidget(txt_results)

		return results_layout


if __name__ == '__main__':
	app = QApplication(sys.argv)
	mw = main_window()
	sys.exit(app.exec_())
