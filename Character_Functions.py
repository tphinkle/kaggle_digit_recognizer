import Character as ch

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
sys.path.append('/home/preston/Desktop/Programming/datasci/p_lib/plot/')
sys.path.append('/home/preston/Desktop/Programming/datasci/p_lib/data_transform/')
sys.path.append('/home/preston/Desktop/Programming/datasci/p_lib/dtw/')
sys.path.append('/home/preston/Desktop/Programming/datasci/p_lib/misc/Maze_Solver.py')

import Character as ch
import p_plot
import data_transform
import pDTW



################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################

#DTW Solution

def load_characters_kaggle_training_format(file_name):
	file_contents = np.genfromtxt(file_name, delimiter = ',')
	
	character_list = []

	characters_to_load = file_contents.shape[0]

	for i in range(0, characters_to_load - 1):
		character_list.append(ch.character(file_contents[i+1,:], 'train'))

	return character_list

def load_characters_kaggle_test_format(file_name):
	file_contents = np.genfromtxt(file_name, delimiter = ',')
	
	character_list = []

	characters_to_load = file_contents.shape[0]

	for i in range(0, characters_to_load - 1):
		character_list.append(ch.character(file_contents[i+1,:], 'test'))

	return character_list

def dtw_classify_character(test_char, train_char_list):
	infinity = 2.0**32.0

	score_list = []
	count_list = []
	for i in range(0, 10):
		score_list.append(0)
		count_list.append(0)


	test_data_x = test_char._xseries
	test_data_x = data_transform.shift_1d_data(test_data_x)
	test_data_x = data_transform.normalize_1d_data(test_data_x)
	
	test_data_y = test_char._yseries
	test_data_y = data_transform.shift_1d_data(test_data_y)
	test_data_y = data_transform.normalize_1d_data(test_data_y)

	print('length of train_char_list = ' + str(len(train_char_list)))

	for i in range(0, len(train_char_list)):
		train_char = train_char_list[i]
		train_data_x = train_char._xseries
		train_data_x = data_transform.shift_1d_data(train_data_x)
		train_data_x = data_transform.normalize_1d_data(train_data_x)
	
		train_data_y = train_char._yseries
		train_data_y = data_transform.shift_1d_data(train_data_y)
		train_data_y = data_transform.normalize_1d_data(train_data_y)

		distance_matrix_x = pDTW.get_distance_matrix_DTW(test_data_x, train_data_x)
		distance_matrix_y = pDTW.get_distance_matrix_DTW(test_data_y, train_data_y)

		cost_matrix_x = pDTW.get_cost_matrix(distance_matrix_x)
		cost_matrix_y = pDTW.get_cost_matrix(distance_matrix_y)

		score = cost_matrix_x[-1,-1] + cost_matrix_y[-1,-1]



		if train_char_list[i]._identity == 0:
			score_list[0] = score_list[0] + score
			count_list[0] = count_list[0] + 1

		elif train_char_list[i]._identity == 1:
			score_list[1] = score_list[1] + score
			count_list[1] = count_list[1] + 1

		elif train_char_list[i]._identity == 2:
			score_list[2] = score_list[2] + score
			count_list[2] = count_list[2] + 1

		elif train_char_list[i]._identity == 3:
			score_list[3] = score_list[3] + score
			count_list[3] = count_list[3] + 1

		elif train_char_list[i]._identity == 4:
			score_list[4] = score_list[4] + score
			count_list[4] = count_list[4] + 1

		elif train_char_list[i]._identity == 5:
			score_list[5] = score_list[5] + score
			count_list[5] = count_list[5] + 1

		elif train_char_list[i]._identity == 6:
			score_list[6] = score_list[6] + score
			count_list[6] = count_list[6] + 1

		elif train_char_list[i]._identity == 7:
			score_list[7] = score_list[7] + score
			count_list[7] = count_list[7] + 1

		elif train_char_list[i]._identity == 8:
			score_list[8] = score_list[8] + score
			count_list[8] = count_list[8] + 1

		elif train_char_list[i]._identity == 9:
			score_list[9] = score_list[9] + score
			count_list[9] = count_list[9] + 1

	for i in range(0, 10):
		if count_list[i] != 0:
			score_list[i] = score_list[i] / count_list[i]
		else:
			score_list[i] = infinity

	minim = infinity
	min_i = 11
	for i in range(0, 10):
		print('count ' + str(i) + ' = ' + str(count_list[i]))
		print('score ' + str(i) + ' = ' + str(score_list[i]))
		if score_list[i] < minim:
			#print('i = ' + str(i) + '!!!!!')
			minim = score_list[i]
			min_i = i

	test_char._calculated_identity = min_i
	return




################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################



#Characterization (logistic regression) solution

def calculate_char_width_bw(char):
	data = char._data_bw
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
	char._feature_width = width
	return


def calculate_char_height_bw(char):
	data = char._data_bw
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
	char._feature_height = height
	return

def calculate_char_curvature(char):

