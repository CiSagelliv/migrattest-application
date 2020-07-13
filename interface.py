#PyQt5 library
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit, QLabel, QFileDialog, QPlainTextEdit, QTableWidget
#Other libraries 
import sys
import os
import re
from itertools import islice 
from itertools import permutations
import pandas as pd

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
            font-size: 15px;
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

		lb_title_groups = QLabel("Groups:", self)

		self.lb_n_groups = QLabel("N° groups", self)

		btn_generate_dataframe = QPushButton("Run", self)
		btn_generate_dataframe.setObjectName("generate_dataframe")
		btn_generate_dataframe.clicked.connect(self.generate_dataframe)

		head_layout = QHBoxLayout()
		head_layout.addWidget(btn_open_file)
		head_layout.addWidget(self.lb_selected_file)
		head_layout.addWidget(lb_title_groups)
		head_layout.addWidget(self.lb_n_groups)
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

		btn_to_pdf = QPushButton("Export PDF", self)
		btn_to_pdf.clicked.connect(self.df_to_pdf)

		txt_results = QPlainTextEdit()

		results_layout = QHBoxLayout()
		results_layout.addWidget(btn_get_results)
		results_layout.addWidget(txt_results)
		results_layout.addWidget(btn_to_pdf)

		return results_layout

	"""
		A partir de aquí se codifica la funcionalidad
	"""

	def go_to_open(self):
		filename = QFileDialog.getOpenFileName()
		path = filename[0]
		#print(path)

		self.file_rows = []

		with open(path, 'r') as origin_file:
			self.lb_selected_file.setText(path)
			for rows in origin_file:
				self.file_rows.append(rows)
				#print(self.file_rows)
				#print(origin_file.readline())

		def find_population(substr, infile):
			global population
			with open(path) as origin:
				for line in origin:
					if substr in line:
						population = line
						#print(population)

		find_population("Population", path)

		def find_total_population(substr, infile):
			global total_population
			with open(path) as origin:
				for line in origin:
					if substr in line:
						total_population = line
						#print(total_population)

		find_total_population("Total of", path)


		def find_lower_percentiles(substr, infile):
			global lower
			with open(path) as origin:
				for line in origin:
					if substr in line:
						lower = line
						#print(lower)

		find_lower_percentiles("Lower percentiles", path)


		def find_upper_percentiles(substr, infile):
			global upper
			with open(path) as origin:
				for line in origin:
					if substr in line:
						upper = line
						#print(upper)

		find_upper_percentiles("Upper percentiles", path)

		"""
			Find indexes in file_rows list
		"""

		start_index_group = self.file_rows.index(population)
		end_index_group = self.file_rows.index(total_population)
		index_lower = self.file_rows.index(lower)
		index_upper = self.file_rows.index(upper)

		#print(start_index_group, end_index_group, index_lower, index_upper)

		"""
			This must be executed once (searching how to delete csv file)
			Creates groups csv file
		"""	




		with open(path, 'r') as origin_file:
			for line in islice(origin_file, start_index_group + 4, end_index_group):
				with open('groups.csv','a') as group_file:
					#print(re.sub("\s+",",", line.strip()))
					group_file.write(re.sub("\s+",",", line.strip())+ '\n')

		"""
			Creates groups dataframe with unnecessary data
		"""	

		groups_df = pd.read_csv('groups.csv',header=None)
		#Remove NaN values
		groups_df = groups_df.dropna()
		#Remove 'Locus' columns
		groups_df = groups_df.drop(2,1)
		#Remove 'missing (0)' column
		groups_df = groups_df.drop(4,1)

		"""
			Get total of groups and its n-value
			Change data types str -> int
		"""

		self.total_per_group = list(map(int, (groups_df[3])))
		self.lb_n_groups.setText(str(len(self.total_per_group)))
		#print(self.total_per_group)

		"""
			Gets start and end of lower and upper percentiles
			Use permutations to know how many groups are there
		"""

		range_lower_start = index_lower + 4 + len(self.total_per_group)
		range_lower_end = range_lower_start + len(list(permutations(range(len(self.total_per_group)),2)))
		print(range_lower_start, range_lower_end)

		range_upper_start = index_upper + 4 + len(self.total_per_group)
		range_upper_end = range_upper_start + len(list(permutations(range(len(self.total_per_group)),2)))
		print(range_upper_start, range_upper_end)

		"""
			This must be executed once (searching how to delete csv file)
			Creates lower and upper percentiles csv file
		"""	

		with open(path, 'r') as origin_file:
			for line in islice(origin_file, range_lower_start, range_lower_end):
				with open('lower_percentiles.csv','a') as lower_file:
					#print(re.sub("\s+",",", line.strip()))
					lower_file.write(re.sub("\s+",",", line.strip())+ '\n')

		with open(path, 'r') as origin_file:
			for line in islice(origin_file, range_upper_start, range_upper_end):
				with open('upper_percentiles.csv','a') as upper_file:
					#print(re.sub("\s+",",", line.strip()))
					upper_file.write(re.sub("\s+",",", line.strip())+ '\n')


	def generate_dataframe(self):
		pass

	def get_results(self):
		pass

	def df_to_pdf(self):
		pass


if __name__ == '__main__':
	app = QApplication(sys.argv)
	mw = main_window()
	sys.exit(app.exec_())
