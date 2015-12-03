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

#####################################################################################
#####################################################################################

class character:
	identity = ''

	calculated_identity = ''

	matrix_w = 28
	matrix_h = 28


	def __init__(self, character_data):
		self._data_gs = None
		self._data_bw = None


		self._data_gs = np.zeros((self.matrix_h, self.matrix_w))


		identity = character_data[0]
		for i in range(0, self.matrix_w):
			for j in range(0, self.matrix_h):
				self._data_gs[j, i] = character_data[1+i+j*self.matrix_w]

		plot_matrix(self._data_gs, False)

		self.convert_data_gs_to_bw()

		plot_matrix(self._data_bw, False)

		print(get_bottom_left_tile_coords_bw(self))


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

#####################################################################################
#####################################################################################

def load_characters_kaggle_training_format(file_name):
	file_contents = np.genfromtxt(file_name, delimiter = ',')
	
	character_list = []

	characters_to_load = 10

	for i in range(1, characters_to_load):
		character_list.append(character(file_contents[i,:]))

	return character_list

def plot_matrix(data, save):
	plt.imshow(data, origin = 'upper', interpolation = 'none', cmap=plt.cm.gray)

	if save == False:
		plt.show()
	else: 
		plt.save_plot()
	return

#####################################################################################
#####################################################################################

def get_bottom_left_tile_coords_bw(char):
	for i in range(0, character.matrix_h):
		for j in range(0, character.matrix_w):
			if char._data_bw[character.matrix_h-i-1,j] == 1:
				return (character.matrix_h-i-1, j)

def walk_ccw_around_char_bw(char):
	path = []

	matrix_w = char.matrix_w
	matrix_h = char.matrix_h
	data = copy(char._data_bw)
	start = get_bottom_left_tile_coords_bw(tile)
	start[0] = start[0] + 1
	path.append((start[0], start[1]))

	ii = start[0]
	jj = start[1]
	kk = 0

	direction = 'right'

	while (ii != start[0]) and (jj != start[1]) and (kk != 0)and (kk < 1000):
		if direction == 'right'
			if data[ii,jj+1] == 1:            #  010
				direction = 'down'            #  0x1 
				ii = ii                       #
				jj = jj
			elif data[ii-1, jj+1] == 1:       #  001
				direction = 'right'           #  0x
				ii = ii                       #
				jj = jj + 1
				path.append((ii,jj))
			elif data[ii-1, jj+1] == 0:
				direction = 'up'
				ii = ii - 1
				jj = jj + 1
				path.append((ii,jj))

		#elif direction == 'up':

		#elif direction == 'left':

		elif direction == 'down':
			if data[ii+1, jj] == 1:
				direction = 'left'
				ii = ii
				jj = jj
			elif data[ii+1, jj-1] == 1:
				direction = 'down'
				ii = ii + 1
				jj = jj

		kk = kk + 1






#####################################################################################
#####################################################################################



