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
import p_plot

#####################################################################################
#####################################################################################

class character:
	matrix_w = 28
	matrix_h = 28


	def __init__(self, character_data, type):
		self._identity = None
		self._calculated_identity = None

		self._data_gs = None
		self._data_bw = None

		self._xseries = None
		self._yseries = None


		self._data_gs = np.zeros((self.matrix_h, self.matrix_w))

		if type == 'train':
			self._identity = character_data[0]
			for i in range(0, self.matrix_w):
				for j in range(0, self.matrix_h):
					self._data_gs[j, i] = character_data[1+i+j*self.matrix_w]


		elif type == 'test':
			for i in range(0, self.matrix_w):
				for j in range(0, self.matrix_h):
					self._data_gs[j, i] = character_data[i+j*self.matrix_w]

		#p_plot.plot_matrix(self._data_gs)

		self.convert_data_gs_to_bw()

		self._data_bw = self.add_buffer_to_matrix(self._data_bw)

		self._data_gs = self.add_buffer_to_matrix(self._data_gs)


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

