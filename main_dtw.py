#Includes
import sys
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/plot/')

import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf

character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'train.csv'
character_test_file_name = 'test.csv'

train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train')
test_character_list = chf.load_characters_kaggle_format(character_file_directory +\
	character_test_file_name, 'test')

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


for i in range(0, 1):#0000):
	train_character_list[i]._perimeter_path = ms.walk_around_ccw(train_character_list[i]._data_bw)

	
	train_character_list[i]._xseries = ms.convert_path_to_xseries(train_character_list[i]._perimeter_path)
	train_character_list[i]._yseries = ms.convert_path_to_yseries(train_character_list[i]._perimeter_path)



	
for i in range(0, 10):
	p_plot.save_plot_matrix(test_character_list[i]._data_gs)
	p_plot.save_plot_matrix(test_character_list[i]._data_bw)
	test_character_list[i]._perimeter_path = ms.walk_around_ccw(test_character_list[i]._data_bw)
	test_character_list[i]._xseries = ms.convert_path_to_xseries(test_character_list[i]._perimeter_path)
	test_character_list[i]._yseries = ms.convert_path_to_yseries(test_character_list[i]._perimeter_path)
	
	chf.dtw_classify_character(test_character_list[i], train_character_list[0:10000])



