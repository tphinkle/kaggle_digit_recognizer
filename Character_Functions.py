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
sys.path.append('/home/preston/Desktop/Programming/datasci/lib/plot/')
sys.path.append('/home/preston/Desktop/Programming/datasci/lib/data_transform/')
sys.path.append('/home/preston/Desktop/Programming/datasci/lib/dtw/')

import Character as ch
import p_plot
import data_transform
import pDTW

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

	for i in range(0, len(train_char_list)):
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



		if train_char_list[i]._identity == 'zero':
			score_list[0] = score_list[0] + score
			count_list[0] = count_list[0] + 1

		elif train_char_list[i]._identity == 'one':
			score_list[1] = score_list[1] + score
			count_list[1] = count_list[1] + 1

		elif train_char_list[i]._identity == 'two':
			score_list[2] = score_list[2] + score
			count_list[2] = count_list[2] + 1

		elif train_char_list[i]._identity == 'three':
			score_list[3] = score_list[3] + score
			count_list[3] = count_list[3] + 1

		elif train_char_list[i]._identity == 'four':
			score_list[4] = score_list[4] + score
			count_list[4] = count_list[4] + 1

		elif train_char_list[i]._identity == 'five':
			score_list[5] = score_list[5] + score
			count_list[5] = count_list[5] + 1

		elif train_char_list[i]._identity == 'six':
			score_list[6] = score_list[6] + score
			count_list[6] = count_list[6] + 1

		elif train_char_list[i]._identity == 'seven':
			score_list[7] = score_list[7] + score
			count_list[7] = count_list[7] + 1

		elif train_char_list[i]._identity == 'eight':
			score_list[8] = score_list[8] + score
			count_list[8] = count_list[8] + 1

		elif train_char_list[i]._identity == 'nine':
			score_list[9] = score_list[9] + score
			count_list[9] = count_list[9] + 1

	for i in range(0, 10):
		score_list[i] = score_list[i] / count_list[i]

	minim = 10**32.0
	for i in range(0, 10):
		if score_list[i] < minim:
			minim = score_list[i]
			min_i = i

	return i
