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

import p_plot
import data_transform
import pDTW

def load_characters_kaggle_training_format(file_name):
	file_contents = np.genfromtxt(file_name, delimiter = ',')
	
	character_list = []

	print('file_contents.shape = ')
	print(file_contents.shape)


	characters_to_load = file_contents.shape[0]

	for i in range(0, characters_to_load - 1):
		character_list.append(character(file_contents[i+1,:]))

	print(len(character_list))

	return character_list


def dtw_classify_character(test_char, train_char_list):
	zero_score = ''
	one_score = ''
	two_score = ''
	three_score = ''
	four_score = ''
	five_score = ''
	six_score = ''
	seven_score = ''
	eight_score = ''
	nine_score = ''

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






		if train_char_list[i]._identity == 'zero':
			zero_score.append(score)
		elif train_char_list[i]._identity == 'one':
			one_score.append(score)
		elif train_char_list[i]._identity == 'two':
			two_score.append(score)
		elif train_char_list[i]._identity == 'three':
			three_score.append(score)
		elif train_char_list[i]._identity == 'four':
			four_score.append(score)
		elif train_char_list[i]._identity == 'five':
			five_score.append(score)
		elif train_char_list[i]._identity == 'six':
			six_score.append(score)
		elif train_char_list[i]._identity == 'seven':
			seven_score.append(score)
		elif train_char_list[i]._identity == 'eight':
			eight_score.append(score)
		elif train_char_list[i]._identity == 'nine':
			nine_score.append(score)
