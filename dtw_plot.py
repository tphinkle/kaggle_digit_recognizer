#Includes
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
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/plot/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/data_transform/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/dtw/')

import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf
import data_transform
import pDTW

character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'
character_test_file_name = 'test.csv'

train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train')


zero_list = []
one_list = []
two_list = []
three_list = []
four_list = []
five_list = []
six_list = []
seven_list = []
eight_list = []
nine_list = []

fives = 0
i = 0
while fives < 2:
	
	if train_character_list[i]._classification == 5:
		
		

		if fives == 0:
			five_0 = train_character_list[i]
			fives = fives + 1
		

		elif fives == 1:
			five_1 = train_character_list[i]
			if ms.walk_around_ccw(five_1._data_bw).shape[0] == ms.walk_around_ccw(five_0._data_bw).shape[0]:
				fives = fives + 1
			
		
	i = i + 1



		
five_0._perimeter_path = ms.walk_around_ccw(five_0._data_bw)
five_0._xseries = ms.convert_path_to_xseries(five_0._perimeter_path)

five_1._perimeter_path = ms.walk_around_ccw(five_1._data_bw)
five_1._xseries = ms.convert_path_to_xseries(five_1._perimeter_path)



five_0_x = five_0._xseries
five_0_x = data_transform.stretch_data_x(five_0_x)
five_0_x = data_transform.normalize_data(five_0_x)

five_1_x = five_1._xseries
five_1_x = data_transform.stretch_data_x(five_1_x)
five_1_x = data_transform.normalize_data(five_1_x)



distance_matrix_x = pDTW.get_distance_matrix_DTW(five_0_x, five_1_x)



cost_matrix_x = pDTW.get_cost_matrix(distance_matrix_x)
path_x = pDTW.get_warp_path(cost_matrix_x)

p_plot.save_plot_matrix_line(distance_matrix_x, path_x)
p_plot.save_plot_matrix_line(cost_matrix_x, path_x)

offset_1 = 1

for i in range(five_1_x.shape[0]):
	five_1_x[i][1] = five_1_x[i][1] + offset_1

fig = plt.figure()
for i in range(path_x.shape[0]):
	x0 = five_0_x[path_x[i][0]][0]
	x1 = five_1_x[path_x[i][1]][0]
	y0 = five_0_x[path_x[i][0]][1]
	y1 = five_1_x[path_x[i][1]][1]
	plt.plot((x0, x1), (y0, y1), lw = 2)

plt.plot(five_0_x[:,0], five_0_x[:,1], c = (0,0,0), lw = 2)
plt.plot(five_1_x[:,0], five_1_x[:,1], c = (0,0,0), lw = 2)

plt.show()



