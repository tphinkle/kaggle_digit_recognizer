#Includes
import sys
sys.path.append('/home/preston/Desktop/Programming/datasci/lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/datasci/lib/plot/')

import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf

character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'train.csv'
character_test_file_name = 'test.csv'

train_character_list = chf.load_characters_kaggle_training_format(character_file_directory +\
 character_train_file_name)
test_character_list = chf.load_characters_kaggle_test_format(character_file_directory +\
	character_test_file_name)

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

for i in range(0, 1000):
	if train_character_list[i]._identity == 0:
		zero_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 1:
		one_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 2:
		two_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 3:
		three_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 4:
		four_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 5:
		five_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 6:
		six_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 7:
		seven_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 8:
		eight_list.append(train_character_list[i])

	elif train_character_list[i]._identity == 9:
		nine_list.append(train_character_list[i])

	train_character_list[i].path = ms.walk_around_ccw(train_character_list[i]._data_bw)
	train_character_list[i]._xseries = ms.convert_path_to_xseries(train_character_list[i].path)
	train_character_list[i]._yseries = ms.convert_path_to_yseries(train_character_list[i].path)
	
for i in range(0, 1):
	p_plot.plot_matrix(test_character_list[i]._data_bw)
	
	chf.dtw_classify_character(test_character_list[i], train_character_list)
	print('determined type = ' + str(test_character_list[i]._calculated_identity))



