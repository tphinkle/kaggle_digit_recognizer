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
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/plot/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/data_transform/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/dtw/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/misc/Maze_Solver.py')

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

def load_characters_kaggle_format(file_name, test_train, load_range = 'default'):
	file_contents = np.genfromtxt(file_name, delimiter = ',')
	
	if(load_range == 'default'):
		load_range = (0, file_contents.shape[0] - 1)

	if(load_range[0] < 0):
		load_range[0] = 0

	if(load_range[1] > file_contents.shape[0] - 1):
		load_range[1] = file_contents.shape[0] - 1
		
	character_list = np.zeros((load_range[1] - load_range[0],), dtype = ch.character)

	for i in range(load_range[0], load_range[1]):
		char = ch.character(file_contents[i+1,:], test_train)
		character_list[i-load_range[0]] = char

	return character_list

def dtw_classify_character(test_char, train_char_list):
	infinity = 2.0**32.0

	score_list = []
	count_list = []
	for i in range(0, 10):
		score_list.append(0)
		count_list.append(0)

	test_data_x = test_char._xseries
	test_data_x = data_transform.xstretch_data(test_data_x)
	test_data_x = data_transform.normalize_data(test_data_x)
	
	test_data_y = test_char._yseries
	test_data_y = data_transform.xstretch_data(test_data_y)
	test_data_y = data_transform.normalize_data(test_data_y)

	for i in range(train_char_list.shape[0]):

		train_char = train_char_list[i]

		train_data_x = train_char._xseries
		train_data_x = data_transform.xstretch_data(train_data_x)
		train_data_x = data_transform.normalize_data(train_data_x)
	
		train_data_y = train_char._yseries
		train_data_y = data_transform.xstretch_data(train_data_y)
		train_data_y = data_transform.normalize_data(train_data_y)
		

		print(train_data_y)
		
		

		distance_matrix_x = pDTW.get_distance_matrix_DTW(test_data_x, train_data_x)
		distance_matrix_y = pDTW.get_distance_matrix_DTW(test_data_y, train_data_y)

		

		cost_matrix_x = pDTW.get_cost_matrix(distance_matrix_x)
		cost_matrix_y = pDTW.get_cost_matrix(distance_matrix_y)


		#path_x = pDTW.get_warp_path(cost_matrix_x)
		#path_y = pDTW.get_warp_path(cost_matrix_y)

		

	


		cost_matrix_x_area = 1.0*cost_matrix_x.shape[0]*cost_matrix_x.shape[1]
		cost_matrix_y_area = 1.0*cost_matrix_y.shape[0]*cost_matrix_y.shape[1]

		score = cost_matrix_x[-1,-1]/cost_matrix_x_area + cost_matrix_y[-1,-1]/cost_matrix_y_area
		#print('identification = ' + str(train_char._classification))
		#print('score = ' + str(score))

		#if i < 3:
			#p_plot.save_plot_matrix(test_char._data_bw)
			#p_plot.save_plot_matrix(train_char._data_bw)
#
			#number_of_lines = path_x.shape[0]
			#nn = 1
			#offset = 2
			#plt.plot(train_data_x[:,0], train_data_x[:,1])
			#plt.plot(test_data_x[:,0], test_data_x[:,1] +offset)
			#for j in range(int(number_of_lines/nn) - 1):
				#plt.plot((test_data_x[path_x[j*nn,0],0],\
					#train_data_x[path_x[j*nn,1],0]),\
					#(test_data_x[path_x[j*nn,0],1] + offset,\
						#train_data_x[path_x[j*nn,1],1]))	
			#plt.show()
#
			#p_plot.save_plot_matrix_line(distance_matrix_x, path_x)
			#p_plot.save_plot_matrix_line(cost_matrix_x, path_x)
#
			#number_of_lines = path_y.shape[0]
			#nn = 1
			#offset = 2
			#plt.plot(train_data_y[:,0], train_data_y[:,1])
			#plt.plot(test_data_y[:,0], test_data_y[:,1] +offset)
			#for j in range(int(number_of_lines/nn) - 1):
				#plt.plot((test_data_y[path_y[j*nn,0],0],\
					#train_data_y[path_y[j*nn,1],0]),\
					#(test_data_y[path_y[j*nn,0],1] + offset,\
						#train_data_y[path_y[j*nn,1],1]))	
			#plt.show()
#
			#p_plot.save_plot_matrix_line(distance_matrix_y, path_y)
			#p_plot.save_plot_matrix_line(cost_matrix_y, path_y)


		if train_char_list[i]._classification == 0:
			score_list[0] = score_list[0] + score
			count_list[0] = count_list[0] + 1

		elif train_char_list[i]._classification == 1:
			score_list[1] = score_list[1] + score
			count_list[1] = count_list[1] + 1

		elif train_char_list[i]._classification == 2:
			score_list[2] = score_list[2] + score
			count_list[2] = count_list[2] + 1

		elif train_char_list[i]._classification == 3:
			score_list[3] = score_list[3] + score
			count_list[3] = count_list[3] + 1

		elif train_char_list[i]._classification == 4:
			score_list[4] = score_list[4] + score
			count_list[4] = count_list[4] + 1

		elif train_char_list[i]._classification == 5:
			score_list[5] = score_list[5] + score
			count_list[5] = count_list[5] + 1

		elif train_char_list[i]._classification == 6:
			score_list[6] = score_list[6] + score
			count_list[6] = count_list[6] + 1

		elif train_char_list[i]._classification == 7:
			score_list[7] = score_list[7] + score
			count_list[7] = count_list[7] + 1

		elif train_char_list[i]._classification == 8:
			score_list[8] = score_list[8] + score
			count_list[8] = count_list[8] + 1

		elif train_char_list[i]._classification == 9:
			score_list[9] = score_list[9] + score
			count_list[9] = count_list[9] + 1

	for i in range(0, 10):
		if count_list[i] != 0:
			score_list[i] = (1.0*score_list[i])/count_list[i]
		else:
			score_list[i] = infinity

	minim = infinity
	min_i = 11
	for i in range(0, 10):
		if score_list[i] < minim:
			minim = score_list[i]
			min_i = i

	test_char._predicted_classification = min_i
	for i in range(len(score_list)):
		print('i = ' + str(i) + '\t\t\t' + str(score_list[i]))

	print("test_char._predicted_classification = " + str(test_char._predicted_classification))
	print("test_char._classification = " + str(test_char._classification))
	return




################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
