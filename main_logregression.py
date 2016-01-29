#Includes
import sys
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/misc/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/plot/')
sys.path.append('/home/preston/Desktop/Programming/p_lib/python_lib/regression/')
import p_plot
import Maze_Solver as ms
import Character as ch
import Character_Functions as chf
import softmax_regression as sr
import numpy as np


character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'
character_test_file_name = 'train.csv'

train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train', (0,100))
test_character_list = chf.load_characters_kaggle_format(character_file_directory +\
	character_test_file_name, 'train', (0,100))


for i in range(train_character_list.shape[0]):
	char = train_character_list[i]
	char.calculate_char_features()


for i in range(test_character_list.shape[0]):
	if i%1000 == 0:
		print("calculating features for " + str(i) + "th test event...")
	test_character_list[i].calculate_char_features()

print('begin softmax regression...')

Lambda = 0.05
sr.softmax_regression(train_character_list, test_character_list, Lambda)


correct_classifications = np.zeros(ch.character.total_classifications)
total_instances = np.zeros(ch.character.total_classifications)

for i in range(test_character_list.shape[0]):
	char = test_character_list[i]
	
	if char._predicted_classification == char._classification:
		correct_classifications[int(char._classification)] = correct_classifications[int(char._classification)] + 1

	total_instances[int(char._classification)] = total_instances[int(char._classification)] + 1

for i in range(0, ch.character.total_classifications):
	print('i = ' + str(i) + ': ' + str(correct_classifications[i]) + "/" + str(total_instances[i]))
