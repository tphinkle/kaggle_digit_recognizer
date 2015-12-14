#Includes
import sys
sys.path.append('/home/preston/Desktop/Programming/datasci/p_lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/datasci/p_lib/plot/')

import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf

character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'
character_test_file_name = 'test.csv'

train_character_list = chf.load_characters_kaggle_training_format(character_file_directory +\
 character_train_file_name)
test_character_list = chf.load_characters_kaggle_test_format(character_file_directory +\
	character_test_file_name)

for i in range(0,5):
	char = train_character_list[i]
	p_plot.plot_matrix(char._data_bw)
	chf.calculate_char_width_bw(char)
	chf.calculate_char_height_bw(char)