#Includes
import sys
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/plot/')

import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf
import numpy as np

character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'


train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train', 300)

for char in train_character_list:
	if (char._classification == 0) or (char._classification == 1):
		p_plot.save_plot_matrix_temp(char._data_bw)

