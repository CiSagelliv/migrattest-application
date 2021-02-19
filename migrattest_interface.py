#PyQt5 library
from PyQt5 import QtWidgets, QtGui, QtCore
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
import openpyxl

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
		QLabel#results_labels{
			font-size: 15px;
		}
		"""

		self.setStyleSheet(self.stylesheet)
		self.show()

	"""
		Interface elements
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

	def middle(self):
		self.dataframe_table = QTableWidget()

		middle_layout = QHBoxLayout()
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


	def go_to_open(self):

		"""
		This method open migrate's txt files and return three csv files:
			- groups
			- lower percentiles
			- upper percentiles

		input:
			- Migrate's txt files

		output:
			- three csv files
			- Migrate's txt file path
			- total of populations

		"""

		filename = QFileDialog.getOpenFileName()
		self.path = filename[0]

		print(filename, type(filename))


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

		groups_filename = 'groups'
		index_1 = 0

		while os.path.exists(f"{groups_filename}{index_1}.csv"):
			index_1 += 1
		with open(self.path, 'r') as origin_file:
			print(type(origin_file))
			for line in islice(origin_file, (self.list_of_indexes[0]-1) + 4, (self.list_of_indexes[1]-1)):
				print(type(origin_file))
				with open(f"{groups_filename}{index_1}.csv", 'a') as group_file:
					#print(re.sub("\s+",",", line.strip()))
					group_file.write(re.sub("\s+",",", line.strip()) + '\n')


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

		range_lower_start = (self.list_of_indexes[2]-1) + 4 + len(self.total_per_group)
		range_lower_end = range_lower_start + len(list(permutations(range(len(self.total_per_group)),2)))

		#print(len(list(permutations(range(len(self.total_per_group)),2))))

		self.lower_percentiles_filename = 'lower_percentiles'
		self.index_2 = 0

		while os.path.exists(f"{self.lower_percentiles_filename}{self.index_2}.csv"):
			self.index_2 += 1
		with open(self.path, 'r') as origin_file:
			for line in islice(origin_file, range_lower_start, range_lower_end):
				with open(f"{self.lower_percentiles_filename}{self.index_2}.csv",'a') as lower_file:
					lower_file.write(re.sub("\s+", ",", line.strip()) + '\n')

		range_upper_start = (self.list_of_indexes[3]-1) + 4 + len(self.total_per_group)
		range_upper_end = range_upper_start + len(list(permutations(range(len(self.total_per_group)),2)))

		self.upper_percentiles_filename = 'upper_percentiles'
		self.index_3 = 0

		while os.path.exists(f"{self.upper_percentiles_filename}{self.index_3}.csv"):
			self.index_3 += 1
		with open(self.path, 'r') as origin_file:
			for line in islice(origin_file, range_upper_start, range_upper_end):
				with open(f"{self.upper_percentiles_filename}{self.index_3}.csv", 'a') as upper_file:
					upper_file.write(re.sub("\s+", ",", line.strip()) + '\n')

	def generate_dataframe(self):
		"""
		This method create a table with percentiles and MLE values

		input:
			- groups csv file
			- lower percentiles csv file
			- upper percentiles csv file

		output:
			- general dataframe with all the values
			- column's title painted
			- table with dataframe data

		"""

		lower_df = pd.read_csv(f"{self.lower_percentiles_filename}{self.index_2}.csv", header=None)
		lower_df.columns = ['Parameter','0.005','0.025','0.050','0.250','MLE']

		upper_df = pd.read_csv(f"{self.upper_percentiles_filename}{self.index_3}.csv", header=None)
		upper_df.columns = ['Parameter','MLE','0.750','0.950','0.975','0.995']

		upper_df = upper_df.drop(['MLE'], axis=1)
		upper_df = upper_df.drop(['Parameter'], axis=1)

		self.percentiles_df = pd.concat([lower_df, upper_df], axis=1)


		columns_to_check = ['0.005','0.025','0.050','0.250','MLE','0.750','0.950','0.975','0.995']
		self.percentiles_df[columns_to_check] = self.percentiles_df[columns_to_check].replace({'\*':''}, regex=True)

		self.percentiles_df['0.005'] = pd.to_numeric(self.percentiles_df['0.005'], errors='coerce')
		self.percentiles_df['0.025'] = pd.to_numeric(self.percentiles_df['0.025'], errors='coerce')
		self.percentiles_df['0.050'] = pd.to_numeric(self.percentiles_df['0.050'], errors='coerce')
		self.percentiles_df['0.250'] = pd.to_numeric(self.percentiles_df['0.250'], errors='coerce')
		self.percentiles_df['0.750'] = pd.to_numeric(self.percentiles_df['0.750'], errors='coerce')
		self.percentiles_df['0.950'] = pd.to_numeric(self.percentiles_df['0.950'], errors='coerce')
		self.percentiles_df['0.975'] = pd.to_numeric(self.percentiles_df['0.975'], errors='coerce')
		self.percentiles_df['0.995'] = pd.to_numeric(self.percentiles_df['0.995'], errors='coerce')
		self.percentiles_df['MLE'] = pd.to_numeric(self.percentiles_df['MLE'], errors='coerce')

		df_titles = pd.DataFrame({'Parameter':'Parameter',
                          '0.005':'0.005',
                          '0.025':'0.025',
                          '0.050':'0.050',
                          '0.250':'0.250',
                          'MLE':'MLE',
                          '0.750':'0.750',
                          '0.950':'0.950',
                          '0.975':'0.975',
                          '0.995':'0.995'}, index=[0])

		df_with_titles = pd.concat([df_titles, self.percentiles_df]).reset_index(drop = True)

		self.dataframe_table.setColumnCount(len(df_with_titles.columns))
		self.dataframe_table.setRowCount(len(df_with_titles.index))
		for rows in range(len(df_with_titles.index)):
			for columns in range(len(df_with_titles.columns)):
				self.dataframe_table.setItem(rows, columns, QTableWidgetItem(str(df_with_titles.iloc[rows, columns])))

		for n_cell in range(0,10):
			self.dataframe_table.item(0, n_cell).setBackground(QtGui.QColor(211,227,252))

		print(lower_df,'\n', upper_df,'\n', self.percentiles_df)

	def get_results(self):
		"""
		This method:
			- choose corresponding values (percentiles)
			- take those values and create different dataframes
			- select corresponding pairs (ex. 21 - 12) and do math with them

		input:
			- general dataframe

		output:
			- t test with unequal variance
			- dataframe with math and results

		"""

		if self.first_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Parameter','0.005','0.995','MLE']]

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

			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Parameter','0.005','0.995','MLE','First_total','Second_total'])

			self.result_df['0.005'] = pd.to_numeric(self.result_df['0.005'], errors = 'coerce')
			self.result_df['0.995'] = pd.to_numeric(self.result_df['0.995'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S'] = ((self.result_df['0.995']-self.result_df['MLE'])/self.result_df['two_tailed'])*self.result_df['Sqrt_n']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var'] = self.result_df['S'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var'])+(self.result_df['dof_1']*self.result_df['Var']).rolling(1).sum().shift(-1))/self.result_df['dof']
			self.result_df['calculated_t'] = (+(self.result_df['MLE'] - self.result_df['MLE'].shift(-1)))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			#print(self.result_df)

			cols = ['Parameter','0.005','0.995','MLE','First_total','Second_total','Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']

			self.result_df_2 = self.result_df.loc[:, cols]
			self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']] = self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']].round(3)

			self.result_df_2['result'] = self.result_df_2.two_tailed_both.abs().le(self.result_df_2.calculated_t.abs())
			self.result_df_2['result'] = self.result_df_2['result'].map({True: 'Bidirectional', False: 'Unidirectional'})


			#self.result_df['result'] = np.where((-self.result_df['two_tailed_1']).all() > (self.result_df['calculated_t']).all() > (self.result_df['two_tailed_both']).all(), 'Rejected', 'Accepted')

			percentiles_titles = pd.DataFrame({'Parameter':'Parameter',
			                          		'0.005':'0.005',
			                          		'0.995':'0.995',
			                          		'MLE':'MLE',
			                          		'First_total':'n_1',
			                          		'Second_total':'n_2',
			                          		'Sqrt_n':'n_sqrt',
			                          		'dof_1':'dof_1',
											'dof_2':'dof_2',
											'two_tailed':'two_tailed',
											'S':'S',
											'dof':'dof',
											'Var':'Var',
											'SDp':'SDp',
			                          		'calculated_t':'calculated_t',
											'two_tailed_both':'two_tailed_both',
											'result':'result'}, index=[0])

			percentiles_with_titles = pd.concat([percentiles_titles, self.result_df_2]).reset_index(drop = True)

			self.results_table.setColumnCount(len(percentiles_with_titles.columns))
			self.results_table.setRowCount(len(percentiles_with_titles.index))
			for rows_1 in range(len(percentiles_with_titles.index)):
				for columns_1 in range(len(percentiles_with_titles.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(percentiles_with_titles.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

			# I need to find a way to delete or fill those columns inside one for loop 

			for percentile_cell in range(0,17):
				self.results_table.item(0, percentile_cell).setBackground(QtGui.QColor(211,227,252))

			for indice in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice % 2 == 0:
					self.results_table.item(indice, 11).setBackground(QtGui.QColor(0,0,0))

			for indice_2 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_2 % 2 == 0:
					self.results_table.item(indice_2, 13).setBackground(QtGui.QColor(0,0,0))

			for indice_3 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_3 % 2 == 0:
					self.results_table.item(indice_3, 14).setBackground(QtGui.QColor(0,0,0))

			for indice_4 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_4 % 2 == 0:
					self.results_table.item(indice_4, 15).setBackground(QtGui.QColor(0,0,0))
			
			for indice_5 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_5 % 2 == 0:
					self.results_table.item(indice_5, 16).setBackground(QtGui.QColor(0,0,0))

		elif self.second_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Parameter','0.025','0.975','MLE']]

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

			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Parameter','0.025','0.975','MLE','First_total','Second_total'])

			self.result_df['0.025'] = pd.to_numeric(self.result_df['0.025'], errors = 'coerce')
			self.result_df['0.975'] = pd.to_numeric(self.result_df['0.975'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S'] = ((self.result_df['0.975']-self.result_df['MLE'])/self.result_df['two_tailed'])*self.result_df['Sqrt_n']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var'] = self.result_df['S'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var'])+(self.result_df['dof_1']*self.result_df['Var']).rolling(1).sum().shift(-1))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			cols = ['Parameter','0.025','0.975','MLE','First_total','Second_total','Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']

			self.result_df_2 = self.result_df.loc[:, cols]
			self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']] = self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']].round(3)

			self.result_df_2['result'] = self.result_df_2.two_tailed_both.abs().le(self.result_df_2.calculated_t.abs())
			self.result_df_2['result'] = self.result_df_2['result'].map({True: 'Bidirectional', False: 'Unidirectional'})

			#print(self.result_df)

			percentiles_titles = pd.DataFrame({'Parameter':'Parameter',
			                          		'0.025':'0.025',
			                          		'0.975':'0.975',
			                          		'MLE':'MLE',
			                          		'First_total':'n_1',
			                          		'Second_total':'n_2',
			                          		'Sqrt_n':'n_sqrt',
			                          		'dof_1':'dof_1',
											'dof_2':'dof_2',
											'two_tailed':'two_tailed',
											'S':'S',
											'dof':'dof',
											'Var':'Var',
											'SDp':'SDp',
			                          		'calculated_t':'calculated_t',
											'two_tailed_both':'two_tailed_both',
											'result':'result'}, index=[0])

			percentiles_with_titles = pd.concat([percentiles_titles, self.result_df_2]).reset_index(drop = True)

			self.results_table.setColumnCount(len(percentiles_with_titles.columns))
			self.results_table.setRowCount(len(percentiles_with_titles.index))
			for rows_1 in range(len(percentiles_with_titles.index)):
				for columns_1 in range(len(percentiles_with_titles.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(percentiles_with_titles.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

			for percentile_cell in range(0,17):
				self.results_table.item(0, percentile_cell).setBackground(QtGui.QColor(211,227,252))

			for indice in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice % 2 == 0:
					self.results_table.item(indice, 11).setBackground(QtGui.QColor(0,0,0))

			for indice_2 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_2 % 2 == 0:
					self.results_table.item(indice_2, 13).setBackground(QtGui.QColor(0,0,0))

			for indice_3 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_3 % 2 == 0:
					self.results_table.item(indice_3, 14).setBackground(QtGui.QColor(0,0,0))

			for indice_4 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_4 % 2 == 0:
					self.results_table.item(indice_4, 15).setBackground(QtGui.QColor(0,0,0))

			for indice_5 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_5 % 2 == 0:
					self.results_table.item(indice_5, 16).setBackground(QtGui.QColor(0,0,0))

		elif self.third_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Parameter','0.050','0.950','MLE']]

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

			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Parameter','0.050','0.950','MLE','First_total','Second_total'])


			self.result_df['0.050'] = pd.to_numeric(self.result_df['0.050'], errors = 'coerce')
			self.result_df['0.950'] = pd.to_numeric(self.result_df['0.950'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S'] = ((self.result_df['0.950']-self.result_df['MLE'])/self.result_df['two_tailed'])*self.result_df['Sqrt_n']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var'] = self.result_df['S'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var'])+(self.result_df['dof_1']*self.result_df['Var']).rolling(1).sum().shift(-1))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			cols = ['Parameter','0.050','0.950','MLE','First_total','Second_total','Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']

			self.result_df_2 = self.result_df.loc[:, cols]
			self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']] = self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']].round(3)

			self.result_df_2['result'] = self.result_df_2.two_tailed_both.abs().le(self.result_df_2.calculated_t.abs())
			self.result_df_2['result'] = self.result_df_2['result'].map({True: 'Bidirectional', False: 'Unidirectional'})			        

			percentiles_titles = pd.DataFrame({'Parameter':'Parameter',
			                          		'0.050':'0.050',
			                          		'0.950':'0.950',
			                          		'MLE':'MLE',
			                          		'First_total':'n_1',
			                          		'Second_total':'n_2',
			                          		'Sqrt_n':'n_sqrt',
			                          		'dof_1':'dof_1',
											'dof_2':'dof_2',
											'two_tailed':'two_tailed',
											'S':'S',
											'dof':'dof',
											'Var':'Var',
											'SDp':'SDp',
			                          		'calculated_t':'calculated_t',
											'two_tailed_both':'two_tailed_both',
											'result':'result'}, index=[0])

			percentiles_with_titles = pd.concat([percentiles_titles, self.result_df_2]).reset_index(drop = True)

			self.results_table.setColumnCount(len(percentiles_with_titles.columns))
			self.results_table.setRowCount(len(percentiles_with_titles.index))
			for rows_1 in range(len(percentiles_with_titles.index)):
				for columns_1 in range(len(percentiles_with_titles.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(percentiles_with_titles.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

			for percentile_cell in range(0,17):
				self.results_table.item(0, percentile_cell).setBackground(QtGui.QColor(211,227,252))

			for indice in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice % 2 == 0:
					self.results_table.item(indice, 11).setBackground(QtGui.QColor(0,0,0))

			for indice_2 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_2 % 2 == 0:
					self.results_table.item(indice_2, 13).setBackground(QtGui.QColor(0,0,0))

			for indice_3 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_3 % 2 == 0:
					self.results_table.item(indice_3, 14).setBackground(QtGui.QColor(0,0,0))

			for indice_4 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_4 % 2 == 0:
					self.results_table.item(indice_4, 15).setBackground(QtGui.QColor(0,0,0))

			for indice_5 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_5 % 2 == 0:
					self.results_table.item(indice_5, 16).setBackground(QtGui.QColor(0,0,0))

		elif self.fourth_percentiles_pair.isChecked():
			mixed_percentiles = self.percentiles_df[['Parameter','0.250','0.750','MLE']]

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

			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Parameter','0.250','0.750','MLE','First_total','Second_total'])

			self.result_df['0.250'] = pd.to_numeric(self.result_df['0.250'], errors = 'coerce')
			self.result_df['0.750'] = pd.to_numeric(self.result_df['0.750'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S'] = ((self.result_df['0.750']-self.result_df['MLE'])/self.result_df['two_tailed'])*self.result_df['Sqrt_n']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var'] = self.result_df['S'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var'])+(self.result_df['dof_1']*self.result_df['Var']).rolling(1).sum().shift(-1))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			cols = ['Parameter','0.250','0.750','MLE','First_total','Second_total','Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']

			self.result_df_2 = self.result_df.loc[:, cols]
			self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']] = self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']].round(3)

			self.result_df_2['result'] = self.result_df_2.two_tailed_both.abs().le(self.result_df_2.calculated_t.abs())
			self.result_df_2['result'] = self.result_df_2['result'].map({True: 'Bidirectional', False: 'Unidirectional'})		        

			percentiles_titles = pd.DataFrame({'Parameter':'Parameter',
			                          		'0.250':'0.250',
			                          		'0.750':'0.750',
			                          		'MLE':'MLE',
			                          		'First_total':'n_1',
			                          		'Second_total':'n_2',
			                          		'Sqrt_n':'n_sqrt',
			                          		'dof_1':'dof_1',
											'dof_2':'dof_2',
											'two_tailed':'two_tailed',
											'S':'S',
											'dof':'dof',
											'Var':'Var',
											'SDp':'SDp',
			                          		'calculated_t':'calculated_t',
											'two_tailed_both':'two_tailed_both',
											'result':'result'}, index=[0])

			percentiles_with_titles = pd.concat([percentiles_titles, self.result_df_2]).reset_index(drop = True)

			self.results_table.setColumnCount(len(percentiles_with_titles.columns))
			self.results_table.setRowCount(len(percentiles_with_titles.index))
			for rows_1 in range(len(percentiles_with_titles.index)):
				for columns_1 in range(len(percentiles_with_titles.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(percentiles_with_titles.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

			for percentile_cell in range(0,17):
				self.results_table.item(0, percentile_cell).setBackground(QtGui.QColor(211,227,252))

			for indice in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice % 2 == 0:
					self.results_table.item(indice, 11).setBackground(QtGui.QColor(0,0,0))

			for indice_2 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_2 % 2 == 0:
					self.results_table.item(indice_2, 13).setBackground(QtGui.QColor(0,0,0))

			for indice_3 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_3 % 2 == 0:
					self.results_table.item(indice_3, 14).setBackground(QtGui.QColor(0,0,0))

			for indice_4 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_4 % 2 == 0:
					self.results_table.item(indice_4, 15).setBackground(QtGui.QColor(0,0,0))

			for indice_5 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_5 % 2 == 0:
					self.results_table.item(indice_5, 16).setBackground(QtGui.QColor(0,0,0))

		else:
			self.third_percentiles_pair.setChecked(True)
			mixed_percentiles = self.percentiles_df[['Parameter','0.050','0.950','MLE']]

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

			self.result_df = pd.DataFrame(np.concatenate(to_df), columns=['Parameter','0.050','0.950','MLE','First_total','Second_total'])

			self.result_df['0.050'] = pd.to_numeric(self.result_df['0.050'], errors = 'coerce')
			self.result_df['0.950'] = pd.to_numeric(self.result_df['0.950'], errors = 'coerce')
			self.result_df['MLE'] = pd.to_numeric(self.result_df['MLE'], errors = 'coerce')
			self.result_df['First_total'] = pd.to_numeric(self.result_df['First_total'], errors = 'coerce')
			self.result_df['Second_total'] = pd.to_numeric(self.result_df['Second_total'], errors = 'coerce')

			self.result_df['Sqrt_n'] = np.sqrt(self.result_df['First_total'])
			self.result_df['dof_1'] = self.result_df['First_total'] - 1
			self.result_df['dof_2'] = self.result_df['Second_total'] - 1
			self.result_df['two_tailed'] = t.ppf(1-0.05/2, self.result_df['dof_1'])
			self.result_df['two_tailed_2'] = t.ppf(1-0.05/2, self.result_df['dof_2'])
			self.result_df['S'] = ((self.result_df['0.950']-self.result_df['MLE'])/self.result_df['two_tailed'])*self.result_df['Sqrt_n']
			self.result_df['dof'] = self.result_df['First_total'] + self.result_df['Second_total'] - 2
			self.result_df['Var'] = self.result_df['S'] ** 2
			self.result_df['SDp'] = ((self.result_df['dof_1']*self.result_df['Var'])+(self.result_df['dof_1']*self.result_df['Var']).rolling(1).sum().shift(-1))/self.result_df['dof']
			self.result_df['calculated_t'] = (self.result_df['MLE'] - self.result_df['MLE'].shift(-1))/np.sqrt(self.result_df['SDp']*((1/self.result_df['dof_1'])+(1/self.result_df['dof_2'])))
			self.result_df['two_tailed_both'] = t.ppf(1-0.05/2, self.result_df['dof'])

			cols = ['Parameter','0.050','0.950','MLE','First_total','Second_total','Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']

			self.result_df_2 = self.result_df.loc[:, cols]
			self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']] = self.result_df_2[['Sqrt_n','dof_1','dof_2','two_tailed','S','dof','Var','SDp','calculated_t','two_tailed_both']].round(3)

			self.result_df_2['result'] = self.result_df_2.two_tailed_both.abs().le(self.result_df_2.calculated_t.abs())
			self.result_df_2['result'] = self.result_df_2['result'].map({True: 'Bidirectional', False: 'Unidirectional'})		        

			#print(self.result_df)

			percentiles_titles = pd.DataFrame({'Parameter':'Parameter',
			                          		'0.050':'0.050',
			                          		'0.950':'0.950',
			                          		'MLE':'MLE',
			                          		'First_total':'n_1',
			                          		'Second_total':'n_2',
			                          		'Sqrt_n':'n_sqrt',
			                          		'dof_1':'dof_1',
											'dof_2':'dof_2',
											'two_tailed':'two_tailed',
											'S':'S',
											'dof':'dof',
											'Var':'Var',
											'SDp':'SDp',
			                          		'calculated_t':'calculated_t',
											'two_tailed_both':'two_tailed_both',
											'result':'result'}, index=[0])

			percentiles_with_titles = pd.concat([percentiles_titles, self.result_df_2]).reset_index(drop = True)

			self.results_table.setColumnCount(len(percentiles_with_titles.columns))
			self.results_table.setRowCount(len(percentiles_with_titles.index))
			for rows_1 in range(len(percentiles_with_titles.index)):
				for columns_1 in range(len(percentiles_with_titles.columns)):
					self.results_table.setItem(rows_1, columns_1, QTableWidgetItem(str(percentiles_with_titles.iloc[rows_1, columns_1])))
			self.results_table.resizeColumnsToContents()

			for percentile_cell in range(0,17):
				self.results_table.item(0, percentile_cell).setBackground(QtGui.QColor(211,227,252))

			for indice in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice % 2 == 0:
					self.results_table.item(indice, 11).setBackground(QtGui.QColor(0,0,0))

			for indice_2 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_2 % 2 == 0:
					self.results_table.item(indice_2, 13).setBackground(QtGui.QColor(0,0,0))

			for indice_3 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_3 % 2 == 0:
					self.results_table.item(indice_3, 14).setBackground(QtGui.QColor(0,0,0))

			for indice_4 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_4 % 2 == 0:
					self.results_table.item(indice_4, 15).setBackground(QtGui.QColor(0,0,0))

			for indice_5 in range(1, (len(list(permutations(range(len(self.total_per_group)),2)))+1)):
				if indice_5 % 2 == 0:
					self.results_table.item(indice_5, 16).setBackground(QtGui.QColor(0,0,0))

	def colores(self, data):
		return ['background-color: black' if idx % 2 == 1 else '' for idx in range(len(self.result_df_2['Parameter']))]

	def df_to_file(self):
		"""
		This method export the last dataframe into csv file

		input:
			- dataframe created in get_results method

		output:
			- results csv file
		
		"""

		results_filename = 'results'
		index_4 = 0

		while os.path.exists(f"{results_filename}{index_4}.xlsx"):
			index_4 += 1
		self.result_df_2.style.apply(self.colores, subset=['dof','Var','SDp','calculated_t','two_tailed_both','result']).to_excel(f'{results_filename}{index_4}.xlsx', engine='openpyxl')
		#self.result_df.to_csv(f"{results_filename}{index_4}.csv",encoding='utf-8', index=False)

if __name__ == '__main__':
	app = QApplication(sys.argv)
	mw = main_window()
	sys.exit(app.exec_())
