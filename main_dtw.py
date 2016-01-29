###########################
##  INCLUDES
###########################
import sys
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/plot/')

import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf
import numpy as np

###########################
##  GET COMMAND LINE ARGS
###########################
if len(sys.argv) == 5:
	arg_list = sys.argv
	
	train_start_range = int(arg_list[1])
	train_end_range = int(arg_list[2])
	test_start_range = int(arg_list[3])
	test_end_range = int(arg_list[4])

elif len(sys.argv) == 1:
	train_start_range = 0
	train_end_range = 500
	test_start_range = 0
	test_end_range = 100

else:
	sys.exit('cannot understand arguments to main_dtw.py!')


print('beginning DTW calculation in range:')
print('train: ' + str(train_start_range) + ', ' + str(train_end_range))
print('train: ' + str(test_start_range) + ', ' + str(test_end_range))

###########################
##  LOAD DATA
###########################

character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'
character_test_file_name = 'train.csv'

train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train', (train_start_range,train_end_range))
test_character_list = chf.load_characters_kaggle_format(character_file_directory +\
	character_test_file_name, 'train', (test_start_range,test_end_range))



###########################
##  BEGIN DTW ALGORITHM
###########################

correct_list = np.zeros((10,1))
total_list = np.zeros((10,1))

for i in range(train_character_list.shape[0]):
	train_character = train_character_list[i]
	

	train_character_list[i]._perimeter_path = ms.walk_around_ccw(train_character._data_bw)
	train_character._xseries = ms.convert_path_to_xseries(train_character._perimeter_path)
	train_character._yseries = ms.convert_path_to_yseries(train_character._perimeter_path)



	
for i in range(test_character_list.shape[0]):
	test_character = test_character_list[i]

	test_character._perimeter_path = ms.walk_around_ccw(test_character._data_bw)
	test_character._xseries = ms.convert_path_to_xseries(test_character._perimeter_path)
	test_character._yseries = ms.convert_path_to_yseries(test_character._perimeter_path)
	
	chf.dtw_classify_character(test_character, train_character_list)

	if(test_character._classification == test_character._predicted_classification):
		correct_list[int(test_character._classification)] = correct_list[int(test_character_list[i]._classification)] + 1

	total_list[int(test_character._classification)] = total_list[int(test_character._classification)] + 1






for i in range(0, 10):
	print('i = ' + str(i) + ': ' + str(correct_list[i]) + '/' + str(total_list[i]))




###########################
## EXPORT RESULTS
###########################