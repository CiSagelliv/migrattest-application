#PyQt5 library
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDialog
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit, QLabel, QFileDialog, QPlainTextEdit, QRadioButton
from PyQt5.QtWidgets import QTableWidget, QScrollArea, QTableWidgetItem
#Other libraries
import sys
import os
import re
from itertools import islice
from itertools import permutations
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind_from_stats
from scipy.special import stdtr


class main_window(QDialog):
	def __init__(self):
		super().__init__()

		self.init_UI()

	def init_UI(self):
		self.setGeometry(0,0,1350,695)
		self.setWindowTitle('Migrate files')

		self.setLayout(QVBoxLayout())
		self.layout().addLayout(self.head())
		self.layout().addLayout(self.percentiles_labels())
		self.layout().addLayout(self.middle())
		self.layout().addLayout(self.choose_percentiles())
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
		QLabel#percentiles_labels{
			font-size: 20px;
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

		lb_title_groups = QLabel("Populations:", self)

		self.lb_n_groups = QLabel("...", self)

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

	def percentiles_labels(self):
		lb_percentiles = QLabel("     Pairs  |   0.005   |   0.025   |   0.050   |   0.250   |   0.750   |   0.950   |   0.975   |  0.995")
		lb_percentiles.setObjectName("percentiles_labels")

		percentiles_labels_layout = QHBoxLayout()
		percentiles_labels_layout.addWidget(lb_percentiles)

		return percentiles_labels_layout

	def middle(self):
		#lb_dataframe = QLabel("General" + '\n' + "dataframe")
		#lb_dataframe.setObjectName("lb_1")

		self.dataframe_table = QTableWidget()

		middle_layout = QHBoxLayout()
		#middle_layout.addWidget(lb_dataframe)
		middle_layout.addWidget(self.dataframe_table)

		return middle_layout

	def choose_percentiles(self):
		lb_choose = QLabel("Select percentiles:")

		self.first_percentiles_pair = QRadioButton(self)
		self.first_percentiles_pair.setText('[0.005 - 0.995]')
		self.second_percentiles_pair = QRadioButton(self)
		self.second_percentiles_pair.setText('[0.025 - 0.975]')
		self.third_percentiles_pair = QRadioButton(self)
		self.third_percentiles_pair.setText('[Default (0.050 - 0.950)]')
		self.fourth_percentiles_pair = QRadioButton(self)
		self.fourth_percentiles_pair.setText('[0. 250 - 0.750]')

		"""self.first_percentiles_pair.toggled.connect(self.first_selected)
		self.second_percentiles_pair.toggled.connect(self.second_selected)
		self.third_percentiles_pair.toggled.connect(self.third_selected)
		self.fourth_percentiles_pair.toggled.connect(self.fourth_selected)"""

		percentiles_layout = QHBoxLayout()
		percentiles_layout.addWidget(lb_choose)
		percentiles_layout.addWidget(self.first_percentiles_pair)
		percentiles_layout.addWidget(self.second_percentiles_pair)
		percentiles_layout.addWidget(self.third_percentiles_pair)
		percentiles_layout.addWidget(self.fourth_percentiles_pair)

		return percentiles_layout

	def results(self):
		btn_get_results = QPushButton("Results", self)
		btn_get_results.clicked.connect(self.get_results)

		btn_to_pdf = QPushButton("Export", self)
		btn_to_pdf.clicked.connect(self.df_to_file)

		self.results_table = QTableWidget()

		results_layout = QHBoxLayout()
		results_layout.addWidget(btn_get_results)
		results_layout.addWidget(self.results_table)
		results_layout.addWidget(btn_to_pdf)

		return results_layout

	"""
	- Opens migrate's txt files
	- Find key words and its index in txt file and append the results into a list
	- Creates different csv files (groups, lower and upper percentiles) for each program's iteration
	"""

	def go_to_open(self):
		filename = QFileDialog.getOpenFileName()
		self.path = filename[0]


		with open(self.path, 'r') as origin_file:
			self.lb_selected_file.setText(self.path)

		line_number = 0
		self.list_of_indexes = []
		list_of_words = ["Population  ","Total of","Lower percentiles","Upper percentiles"]
		with open(self.path,'r') as origin_file:
			for line in origin_file:
				line_number += 1
				for word in list_of_words:
					if word in line:
						self.list_of_indexes.append((line_number))

		for element in self.list_of_indexes:
			print("Line number: ", element)

		"""
			Begins groups file creation
		"""

		groups_filename = 'groups'
		index_1 = 0

		while os.path.exists(f"{groups_filename}{index_1}.csv"):
			index_1 += 1
		with open(self.path, 'r') as origin_file:
			for line in islice(origin_file, (self.list_of_indexes[0]-1) + 4, (self.list_of_indexes[1]-1)):
				with open(f"{groups_filename}{index_1}.csv", 'a') as group_file:
					#print(re.sub("\s+",",", line.strip()))
					group_file.write(re.sub("\s+",",", line.strip()) + '\n')

		"""
			- Creates a dataframe with groups information
			- Deletes all NaN values
			- Deletes columns with unnecessary data (ex. locus column and missing(0))
			- Gets the total of groups and its values (ex. Group 1 - 60) and change its data type (str to int)
		"""

		groups_df = pd.read_csv(f"{groups_filename}{index_1}.csv",header=None)
		#Remove NaN values
		groups_df = groups_df.dropna()
		#Remove 'Locus' columns
		groups_df = groups_df.drop(2,1)
		#Remove 'missing (0)' column
		groups_df = groups_df.drop(4,1)
		#print(groups_df)

		self.total_per_group = list(map(int, (groups_df[3])))
		self.lb_n_groups.setText(str(len(self.total_per_group)))
		#print(self.total_per_group)

		"""
			Begins lower percentiles file creation
		"""

		range_lower_start = (self.list_of_indexes[2]-1) + 4 + len(self.total_per_group)
		range_lower_end = range_lower_start + len(list(permutations(range(len(self.total_per_group)),2)))
		#print(range_lower_start, range_lower_end)

		self.lower_percentiles_filename = 'lower_percentiles'
		self.index_2 = 0

		while os.path.exists(f"{self.lower_percentiles_filename}{self.index_2}.csv"):
			self.index_2 += 1
		with open(self.path, 'r') as origin_file:
			for line in islice(origin_file, range_lower_start, range_lower_end):
				with open(f"{self.lower_percentiles_filename}{self.index_2}.csv",'a') as lower_file:
					#print(re.sub("\s+", ",", line.strip()))
					lower_file.write(re.sub("\s+", ",", line.strip()) + '\n')

		"""
			Begins upper percentiles file creation
		"""

		range_upper_start = (self.list_of_indexes[3]-1) + 4 + len(self.total_per_group)
		range_upper_end = range_upper_start + len(list(permutations(range(len(self.total_per_group)),2)))
		#print(range_upper_start, range_upper_end)

		self.upper_percentiles_filename = 'upper_percentiles'
		self.index_3 = 0

		while os.path.exists(f"{self.upper_percentiles_filename}{self.index_3}.csv"):
			self.index_3 += 1
		with open(self.path, 'r') as origin_file:
			for line in islice(origin_file, range_upper_start, range_upper_end):
				with open(f"{self.upper_percentiles_filename}{self.index_3}.csv", 'a') as upper_file:
					#print(re.sub("\s+", ",", line.strip()))
					upper_file.write(re.sub("\s+", ",", line.strip()) + '\n')

	def generate_dataframe(self):
		lower_df = pd.read_csv(f"{self.lower_percentiles_filename}{self.index_2}.csv", header=None)
		lower_df.columns = ['Pairs','P1','P2','P3','P4','MLE']

		upper_df = pd.read_csv(f"{self.upper_percentiles_filename}{self.index_3}.csv", header=None)
		upper_df.columns = ['Pairs','MLE','P5','P6','P7','P8']

		"""
			- Delete repeated columns
			- Create percentiles dataframe (concat lower and upper dataframe)
			- Delete all * from every row
			- Change dataframe data type (object to numeric)
		"""

		upper_df = upper_df.drop(['MLE'], axis=1)
		upper_df = upper_df.drop(['Pairs'], axis=1)

		self.percentiles_df = pd.concat([lower_df, upper_df], axis=1)


		columns_to_check = ['P1','P2','P3','P4','MLE','P5','P6','P7','P8']
		self.percentiles_df[columns_to_check] = self.percentiles_df[columns_to_check].replace({'\*':''}, regex=True)

		"""
		self.percentiles_df['P1'] = self.percentiles_df['P1'].str.replace('*','')
		self.percentiles_df['P2'] = self.percentiles_df['P2'].str.replace('*','')
		self.percentiles_df['P3'] = self.percentiles_df['P3'].str.replace('*','')
		self.percentiles_df['P4'] = self.percentiles_df['P4'].str.replace('*','')
		#self.percentiles_df['P5'] = self.percentiles_df['P5'].str.replace('*','')
		#self.percentiles_df['P6'] = self.percentiles_df['P6'].str.replace('*','')
		#self.percentiles_df['P7'] = self.percentiles_df['P7'].str.replace('*','')
		#self.percentiles_df['P8'] = self.percentiles_df['P8'].str.replace('*','')"""

		self.percentiles_df['P1'] = pd.to_numeric(self.percentiles_df['P1'], errors='coerce')
		self.percentiles_df['P2'] = pd.to_numeric(self.percentiles_df['P2'], errors='coerce')
		self.percentiles_df['P3'] = pd.to_numeric(self.percentiles_df['P3'], errors='coerce')
		self.percentiles_df['P4'] = pd.to_numeric(self.percentiles_df['P4'], errors='coerce')
		self.percentiles_df['P5'] = pd.to_numeric(self.percentiles_df['P5'], errors='coerce')
		self.percentiles_df['P6'] = pd.to_numeric(self.percentiles_df['P6'], errors='coerce')
		self.percentiles_df['P7'] = pd.to_numeric(self.percentiles_df['P7'], errors='coerce')
		self.percentiles_df['P8'] = pd.to_numeric(self.percentiles_df['P8'], errors='coerce')
		self.percentiles_df['MLE'] = pd.to_numeric(self.percentiles_df['MLE'], errors='coerce')

		"""
			Shows dataframe
		"""

		self.dataframe_table.setColumnCount(len(self.percentiles_df.columns))
		self.dataframe_table.setRowCount(len(self.percentiles_df.index))
		for rows in range(len(self.percentiles_df.index)):
			for columns in range(len(self.percentiles_df.columns)):
				self.dataframe_table.setItem(rows, columns, QTableWidgetItem(str(self.percentiles_df.iloc[rows, columns])))

		print(lower_df,'\n', upper_df,'\n', self.percentiles_df)

	def get_results(self):
		"""
			Get the sum of significant percentiles rows
		"""

		if self.first_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Pairs','P1','P8','MLE']]
			#mixed_percentiles['Sum_of_values'] = mixed_percentiles.sum(axis=1)

			first_element = list(combinations((self.total_per_group), len(self.total_per_group)-1))
			first_element.reverse()

			first_element_list = []

			for tuple_element in first_element:
				for list_element in tuple_element:
					first_element_list.append(list_element)

			second_element_list = [element for element in self.total_per_group for i in range(len(self.total_per_group)-1)]

			mixed_percentiles['First_total'] = first_element_list
			mixed_percentiles['Second_total'] = second_element_list

			print(mixed_percentiles)

			n_permutations = len(list(permutations(range(len(self.total_per_group)),2)))
			steps = len(self.total_per_group) - 1
			start = len(self.total_per_group) - 1
			second_pair = []

			for i in range(0, steps):
				start = i * steps + steps + i
				for j in range(start, n_permutations, steps):
					second_pair.append(j)

			n_permutations_list = list(range(0, n_permutations))

			n_total = n_permutations_list
			n_populations = len(self.total_per_group)
			len_n, n_size = len(n_total), len(n_total)//n_populations
			to_array = []

			for a in range(0, len_n, n_size):
				to_array.append(n_total[a:a+n_size])

			to_matrix = np.array([item for item in to_array])
			secondary_diagonal = (to_matrix[np.triu_indices(len(self.total_per_group)-1)]).tolist()

			half_n_permutations = int(n_permutations/2)
			to_df = []

			for n in range(half_n_permutations):
				to_df.append(mixed_percentiles.iloc[[secondary_diagonal[n],second_pair[n]]])


			#self.result_df = mixed_percentiles[['Pairs','t_value','p_value', 'dof']]
			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Pairs','P1','P8','MLE','First_total','Second_total'])

			self.result_df['P1'] = pd.to_numeric(self.result_df['P1'], errors = 'coerce')
			self.result_df['P8'] = pd.to_numeric(self.result_df['P8'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['Sqrt_n_2'] = np.sqrt(self.result_df['Second_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed_1'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S_1'] = ((self.result_df['P8']-self.result_df['MLE'])/self.result_df['two_tailed_1'])*self.result_df['Sqrt_n']
			self.result_df['S_2'] = ((self.result_df['P8']-self.result_df['MLE'])/self.result_df['two_tailed_2'])*self.result_df['Sqrt_n_2']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var1'] = self.result_df['S_1'] ** 2
			self.result_df['Var2'] = self.result_df['S_2'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var1'])+(self.result_df['dof_2']*self.result_df['Var2']))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			self.results_table.setColumnCount(len(self.result_df.columns))
			self.results_table.setRowCount(len(self.result_df.index))
			for rows_1 in range(len(self.result_df.index)):
				for columns_1 in range(len(self.result_df.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(self.result_df.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

		elif self.second_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Pairs','P2','P7','MLE']]

			first_element = list(combinations((self.total_per_group), len(self.total_per_group)-1))
			first_element.reverse()

			first_element_list = []

			for tuple_element in first_element:
				for list_element in tuple_element:
					first_element_list.append(list_element)

			second_element_list = [element for element in self.total_per_group for i in range(len(self.total_per_group)-1)]

			mixed_percentiles['First_total'] = first_element_list
			mixed_percentiles['Second_total'] = second_element_list

			print(mixed_percentiles)

			n_permutations = len(list(permutations(range(len(self.total_per_group)),2)))
			steps = len(self.total_per_group) - 1
			start = len(self.total_per_group) - 1
			second_pair = []

			for i in range(0, steps):
				start = i * steps + steps + i
				for j in range(start, n_permutations, steps):
					second_pair.append(j)

			n_permutations_list = list(range(0, n_permutations))

			n_total = n_permutations_list
			n_populations = len(self.total_per_group)
			len_n, n_size = len(n_total), len(n_total)//n_populations
			to_array = []

			for a in range(0, len_n, n_size):
				to_array.append(n_total[a:a+n_size])

			to_matrix = np.array([item for item in to_array])
			secondary_diagonal = (to_matrix[np.triu_indices(len(self.total_per_group)-1)]).tolist()

			half_n_permutations = int(n_permutations/2)
			to_df = []

			for n in range(half_n_permutations):
				to_df.append(mixed_percentiles.iloc[[secondary_diagonal[n],second_pair[n]]])


			#self.result_df = mixed_percentiles[['Pairs','t_value','p_value', 'dof']]
			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Pairs','P2','P7','MLE','First_total','Second_total'])

			self.result_df['P2'] = pd.to_numeric(self.result_df['P2'], errors = 'coerce')
			self.result_df['P7'] = pd.to_numeric(self.result_df['P7'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['Sqrt_n_2'] = np.sqrt(self.result_df['Second_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed_1'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S_1'] = ((self.result_df['P7']-self.result_df['MLE'])/self.result_df['two_tailed_1'])*self.result_df['Sqrt_n']
			self.result_df['S_2'] = ((self.result_df['P7']-self.result_df['MLE'])/self.result_df['two_tailed_2'])*self.result_df['Sqrt_n_2']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var1'] = self.result_df['S_1'] ** 2
			self.result_df['Var2'] = self.result_df['S_2'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var1'])+(self.result_df['dof_2']*self.result_df['Var2']))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			self.results_table.setColumnCount(len(self.result_df.columns))
			self.results_table.setRowCount(len(self.result_df.index))
			for rows_1 in range(len(self.result_df.index)):
				for columns_1 in range(len(self.result_df.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(self.result_df.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

		elif self.third_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Pairs','P3','P6','MLE']]

			first_element = list(combinations((self.total_per_group), len(self.total_per_group)-1))
			first_element.reverse()

			first_element_list = []

			for tuple_element in first_element:
				for list_element in tuple_element:
					first_element_list.append(list_element)

			second_element_list = [element for element in self.total_per_group for i in range(len(self.total_per_group)-1)]

			mixed_percentiles['First_total'] = first_element_list
			mixed_percentiles['Second_total'] = second_element_list

			print(mixed_percentiles)

			n_permutations = len(list(permutations(range(len(self.total_per_group)),2)))
			steps = len(self.total_per_group) - 1
			start = len(self.total_per_group) - 1
			second_pair = []

			for i in range(0, steps):
				start = i * steps + steps + i
				for j in range(start, n_permutations, steps):
					second_pair.append(j)

			n_permutations_list = list(range(0, n_permutations))

			n_total = n_permutations_list
			n_populations = len(self.total_per_group)
			len_n, n_size = len(n_total), len(n_total)//n_populations
			to_array = []

			for a in range(0, len_n, n_size):
				to_array.append(n_total[a:a+n_size])

			to_matrix = np.array([item for item in to_array])
			secondary_diagonal = (to_matrix[np.triu_indices(len(self.total_per_group)-1)]).tolist()

			half_n_permutations = int(n_permutations/2)
			to_df = []

			for n in range(half_n_permutations):
				to_df.append(mixed_percentiles.iloc[[secondary_diagonal[n],second_pair[n]]])


			#self.result_df = mixed_percentiles[['Pairs','t_value','p_value', 'dof']]
			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Pairs','P3','P6','MLE','First_total','Second_total'])


			self.result_df['P3'] = pd.to_numeric(self.result_df['P3'], errors = 'coerce')
			self.result_df['P6'] = pd.to_numeric(self.result_df['P6'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['Sqrt_n_2'] = np.sqrt(self.result_df['Second_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed_1'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S_1'] = ((self.result_df['P6']-self.result_df['MLE'])/self.result_df['two_tailed_1'])*self.result_df['Sqrt_n']
			self.result_df['S_2'] = ((self.result_df['P6']-self.result_df['MLE'])/self.result_df['two_tailed_2'])*self.result_df['Sqrt_n_2']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var1'] = self.result_df['S_1'] ** 2
			self.result_df['Var2'] = self.result_df['S_2'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var1'])+(self.result_df['dof_2']*self.result_df['Var2']))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			self.results_table.setColumnCount(len(self.result_df.columns))
			self.results_table.setRowCount(len(self.result_df.index))
			for rows_1 in range(len(self.result_df.index)):
				for columns_1 in range(len(self.result_df.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(self.result_df.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

		elif self.fourth_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Pairs','P4','P5','MLE']]

			first_element = list(combinations((self.total_per_group), len(self.total_per_group)-1))
			first_element.reverse()

			first_element_list = []

			for tuple_element in first_element:
				for list_element in tuple_element:
					first_element_list.append(list_element)

			second_element_list = [element for element in self.total_per_group for i in range(len(self.total_per_group)-1)]

			mixed_percentiles['First_total'] = first_element_list
			mixed_percentiles['Second_total'] = second_element_list

			print(mixed_percentiles)

			n_permutations = len(list(permutations(range(len(self.total_per_group)),2)))
			steps = len(self.total_per_group) - 1
			start = len(self.total_per_group) - 1
			second_pair = []

			for i in range(0, steps):
				start = i * steps + steps + i
				for j in range(start, n_permutations, steps):
					second_pair.append(j)

			n_permutations_list = list(range(0, n_permutations))

			n_total = n_permutations_list
			n_populations = len(self.total_per_group)
			len_n, n_size = len(n_total), len(n_total)//n_populations
			to_array = []

			for a in range(0, len_n, n_size):
				to_array.append(n_total[a:a+n_size])

			to_matrix = np.array([item for item in to_array])
			secondary_diagonal = (to_matrix[np.triu_indices(len(self.total_per_group)-1)]).tolist()

			half_n_permutations = int(n_permutations/2)
			to_df = []

			for n in range(half_n_permutations):
				to_df.append(mixed_percentiles.iloc[[secondary_diagonal[n],second_pair[n]]])


			#self.result_df = mixed_percentiles[['Pairs','t_value','p_value', 'dof']]
			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Pairs','P4','P5','MLE','First_total','Second_total'])

			self.result_df['P4'] = pd.to_numeric(self.result_df['P4'], errors = 'coerce')
			self.result_df['P5'] = pd.to_numeric(self.result_df['P5'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['Sqrt_n_2'] = np.sqrt(self.result_df['Second_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed_1'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S_1'] = ((self.result_df['P5']-self.result_df['MLE'])/self.result_df['two_tailed_1'])*self.result_df['Sqrt_n']
			self.result_df['S_2'] = ((self.result_df['P5']-self.result_df['MLE'])/self.result_df['two_tailed_2'])*self.result_df['Sqrt_n_2']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var1'] = self.result_df['S_1'] ** 2
			self.result_df['Var2'] = self.result_df['S_2'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var1'])+(self.result_df['dof_2']*self.result_df['Var2']))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			self.results_table.setColumnCount(len(self.result_df.columns))
			self.results_table.setRowCount(len(self.result_df.index))
			for rows_1 in range(len(self.result_df.index)):
				for columns_1 in range(len(self.result_df.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(self.result_df.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

		else:
			self.third_percentiles_pair.setChecked(True)
			mixed_percentiles = self.percentiles_df[['Pairs','P3','P6','MLE']]

			first_element = list(combinations((self.total_per_group), len(self.total_per_group)-1))
			first_element.reverse()

			first_element_list = []

			for tuple_element in first_element:
				for list_element in tuple_element:
					first_element_list.append(list_element)

			second_element_list = [element for element in self.total_per_group for i in range(len(self.total_per_group)-1)]

			mixed_percentiles['First_total'] = first_element_list
			mixed_percentiles['Second_total'] = second_element_list

			print(mixed_percentiles)


			n_permutations = len(list(permutations(range(len(self.total_per_group)),2)))
			steps = len(self.total_per_group) - 1
			start = len(self.total_per_group) - 1
			second_pair = []

			for i in range(0, steps):
				start = i * steps + steps + i
				for j in range(start, n_permutations, steps):
					second_pair.append(j)

			n_permutations_list = list(range(0, n_permutations))

			n_total = n_permutations_list
			n_populations = len(self.total_per_group)
			len_n, n_size = len(n_total), len(n_total)//n_populations
			to_array = []

			for a in range(0, len_n, n_size):
				to_array.append(n_total[a:a+n_size])

			to_matrix = np.array([item for item in to_array])
			secondary_diagonal = (to_matrix[np.triu_indices(len(self.total_per_group)-1)]).tolist()

			half_n_permutations = int(n_permutations/2)
			to_df = []

			for n in range(half_n_permutations):
				to_df.append(mixed_percentiles.iloc[[secondary_diagonal[n],second_pair[n]]])


			#self.result_df = mixed_percentiles[['Pairs','t_value','p_value', 'dof']]
			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Pairs','P3','P6','MLE','First_total','Second_total'])

			self.result_df['P3'] = pd.to_numeric(self.result_df['P3'], errors = 'coerce')
			self.result_df['P6'] = pd.to_numeric(self.result_df['P6'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['Sqrt_n_2'] = np.sqrt(self.result_df['Second_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed_1'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S_1'] = ((self.result_df['P6']-self.result_df['MLE'])/self.result_df['two_tailed_1'])*self.result_df['Sqrt_n']
			self.result_df['S_2'] = ((self.result_df['P6']-self.result_df['MLE'])/self.result_df['two_tailed_2'])*self.result_df['Sqrt_n_2']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var1'] = self.result_df['S_1'] ** 2
			self.result_df['Var2'] = self.result_df['S_2'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var1'])+(self.result_df['dof_2']*self.result_df['Var2']))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			self.results_table.setColumnCount(len(self.result_df.columns))
			self.results_table.setRowCount(len(self.result_df.index))
			for rows_1 in range(len(self.result_df.index)):
				for columns_1 in range(len(self.result_df.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(self.result_df.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()


	def df_to_file(self):
		results_filename = 'results'
		index_4 = 0

		while os.path.exists(f"{results_filename}{index_4}.csv"):
			index_4 += 1
		self.result_df.to_csv(f"{results_filename}{index_4}.csv",encoding='utf-8', index=False)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	mw = main_window()
	sys.exit(app.exec_())
