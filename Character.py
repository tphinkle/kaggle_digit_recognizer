import numpy as np
import matplotlib.pyplot as plt
import cmath
import csv

from matplotlib import rc
from pylab import *
from scipy.optimize import fsolve
from mpl_toolkits.axes_grid.axislines import SubplotZero

import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import sys
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_libs/plot')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_libs/misc/')
import p_plot
import Maze_Solver as ms

#####################################################################################
#####################################################################################

class character:
	matrix_w = 28
	matrix_h = 28
	
	classifications_list = np.fromfunction(lambda i: i, (10,), dtype = int)
	total_classifications = classifications_list.shape[0]
	total_features = 9

	def __init__(self, character_data, type):
		self._classification = None
		self._predicted_classification = None

		self._data_gs = None
		self._data_bw = None

		self._xseries = None
		self._yseries = None

		self._perimeter_path = None

		self._feature_list = None

		self._feature_width = None
		self._feature_height = None
		self._feature_perimeter = None
		self._feature_curvature = None
		self._feature_horizontal_turns = None
		self._feature_vertical_turns = None
		self._feature_fractional_occupancy = None
		self._feature_moment_x = None
		self._feature_moment_y = None



		self._data_gs = np.zeros((self.matrix_h, self.matrix_w))

		if type == 'train':
			self._classification = character_data[0]
			for i in range(0, self.matrix_w):
				for j in range(0, self.matrix_h):
					self._data_gs[j, i] = character_data[1+i+j*self.matrix_w]


		elif type == 'test':
			for i in range(0, self.matrix_w):
				for j in range(0, self.matrix_h):
					self._data_gs[j, i] = character_data[i+j*self.matrix_w]

		self.convert_data_gs_to_bw()

		self._data_bw = self.add_buffer_to_matrix(self._data_bw)

		self._data_gs = self.add_buffer_to_matrix(self._data_gs)

#####################################################################################
#####################################################################################
################# BASIC FUNCTIONS

	def convert_data_gs_to_bw(self):
		self._data_bw = np.zeros((self.matrix_h, self.matrix_w))

		threshold_value = 64
		for i in range(0, self.matrix_w):
			for j in range(0, self.matrix_h):
				if self._data_gs[i,j] < threshold_value:
					val = 0
				else:
					val = 1
				self._data_bw[i,j] = val

		return

	def add_buffer_to_matrix(self, matrix_):
		new_matrix_ = np.zeros((matrix_.shape[0] + 2, matrix_.shape[1] + 2))
		for i in range(0, matrix_.shape[0]):
			for j in range(0, matrix_.shape[1]):
				new_matrix_[i+1, j+1] = matrix_[i,j]

		return new_matrix_



#####################################################################################
#####################################################################################
################# FEATURE CALCULATIONS


	def calculate_char_features(self):
		self._feature_list = np.zeros(self.total_features)
		
		self.calculate_char_width_bw()
		self.calculate_char_height_bw()
		self.calculate_char_perimeter_bw()
		self.calculate_char_curvature_bw()
		self.calculate_char_horizontal_turns()
		self.calculate_char_vertical_turns()
		self.calculate_char_fractional_occupancy()
		self.calculate_char_moment_x()
		self.calculate_char_moment_y()


		self._feature_list[0] = self._feature_width
		self._feature_list[1] = self._feature_height
		self._feature_list[2] = self._feature_perimeter
		self._feature_list[3] = self._feature_curvature
		self._feature_list[4] = self._feature_horizontal_turns
		self._feature_list[5] = self._feature_vertical_turns
		self._feature_list[6] = self._feature_fractional_occupancy
		self._feature_list[7] = self._feature_moment_x
		self._feature_list[8] = self._feature_moment_y

		return

	def calculate_char_width_bw(self):
		data = self._data_bw
		start_column = 0
		end_column = 0
		for j in range(0, data.shape[1]):
			for i in range(0, data.shape[0]):
				if data[i,j] == 1:
					start_column = j
					break
			else:
				continue
			break

		for j in range(0, data.shape[1]):
			for i in range(0, data.shape[0]):
				if data[i, data.shape[1] - 1 - j] == 1:
					end_column = data.shape[1] - 1 - j
					break

			else:
				continue
			break

		width = end_column - start_column

		
		self._feature_width = width
		return


	def calculate_char_height_bw(self):
		data = self._data_bw
		start_row = 0
		end_row = 0
		for i in range(0, data.shape[0]):
			for j in range(0, data.shape[1]):
				if data[i,j] == 1:
					start_row = i
					break
			else:
				continue
			break

		for i in range(0, data.shape[0]):
			for j in range(0, data.shape[1]):
				if data[data.shape[0] - 1 - i, j] == 1:
					end_row = data.shape[0] - 1 - i
					break

			else:
				continue
			break

		height = end_row - start_row
		self._feature_height = height
		return

	def calculate_char_perimeter_bw(self):
		
		if self._perimeter_path == None:
			self._perimeter_path = ms.walk_around_ccw(self._data_bw)
		perimeter_path = self._perimeter_path

		perimeter = 0
		for i in range(len(perimeter_path)):
			perimeter = perimeter +\
			 ((perimeter_path[i][0]-perimeter_path[i-1][0])**2.0+\
			 	(perimeter_path[i][1]-perimeter_path[i-1][1])**2.0)**0.5

		
		self._feature_perimeter = perimeter
		return

	def calculate_char_curvature_bw(self):
		if self._perimeter_path == None:
			self._perimeter_path = ms.walk_around_ccw(self._data_bw)
		perimeter_path = self._perimeter_path


		curvature = 0
		for i in range(len(self._perimeter_path)):
			if ((abs(self._perimeter_path[i-1][0] - self._perimeter_path[i][0]) == 1)\
			and\
			(abs(self._perimeter_path[i-1][1] - self._perimeter_path[i][1]) == 1)):
				curvature = curvature + 1

		self._feature_curvature = curvature

		return

	

	def calculate_char_horizontal_turns(self):
		if self._perimeter_path == None:
			self._perimeter_path = ms.walk_around_ccw(self._data_bw)
		perimeter_path = self._perimeter_path

		turns = 0

		direction = None

		for i in range(len(perimeter_path)):
			if (perimeter_path[i-1][0] - perimeter_path[i][0] == -1):
				new_direction = 'left'
			elif (perimeter_path[i-1][0] - perimeter_path[i][0] == 1):
				new_direction = 'right'
			else:
				new_direction = direction
			if (new_direction != direction and i != 0):
				turns = turns + 1
			direction = new_direction

		self._feature_horizontal_turns = turns

		return

	def calculate_char_vertical_turns(self):
		if self._perimeter_path == None:
			self._perimeter_path = ms.walk_around_ccw(self._data_bw)
		perimeter_path = self._perimeter_path

		turns = 0

		direction = None

		for i in range(len(perimeter_path)):
			if (perimeter_path[i-1][0] - perimeter_path[i][0] == -1):
				new_direction = 'up'
			elif (perimeter_path[i-1][0] - perimeter_path[i][0] == 1):
				new_direction = 'down'
			else:
				new_direction = direction
			if (new_direction != direction and i != 0):
				turns = turns + 1
			direction = new_direction

		self._feature_vertical_turns = turns


		return

	def calculate_char_fractional_occupancy(self):
		white_spaces = 0
		for i in range(self._data_bw.shape[0]):
			for j in range(self._data_bw.shape[1]):
				if self._data_bw[i][j] == 1:
					white_spaces = white_spaces + 1
		
		self._feature_fractional_occupancy = white_spaces/(self._feature_width*self._feature_height)

		return

	def calculate_char_moment_x(self):
		left_column = self.find_left_column()
		
		moment_x = 0

		for i in range(left_column, self._data_bw.shape[1]):
			for j in range(0, self._data_bw.shape[0]):
				if self._data_bw[j][i] == 1:
					moment_x = moment_x + (i - left_column)**2.0/(self.matrix_w**2.0)
		self._feature_moment_x = moment_x

		return



	def calculate_char_moment_y(self):
		bottom_row = self.find_bottom_row()

		moment_y = 0

		for i in range(bottom_row):
			for j in range(self._data_bw.shape[1]):
				if self._data_bw[i][j] == 1:
					moment_y = moment_y + (bottom_row - i)**2.0/(self.matrix_h**2.0)
		self._feature_moment_y = moment_y

		return
	

	def find_bottom_row(self):
		for i in range(self._data_bw.shape[0]):
			for j in range(self._data_bw.shape[1]):
				if self._data_bw[self._data_bw.shape[0] - i - 1][j] == 1:
					bottom_row = self._data_bw.shape[0] - i - 1
					return bottom_row

	def find_left_column(self):
		for i in range(self._data_bw.shape[1]):
			for j in range(self._data_bw.shape[0]):
				if self._data_bw[j][i] == 1:
					left_column = i
					return left_column

	def get_classification_index(self):
		index = self._classification
		return index

	def get_classification_from_index(self, index):
		classification = self.classifications_list[index]
		return classification