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


character_file_directory = '/home/preston/Desktop/Programming/datasci/projects/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'
character_test_file_name = 'test.csv'

train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train')
test_character_list = chf.load_characters_kaggle_format(character_file_directory +\
	character_test_file_name, 'test')


print('calculating features...')
#x1 = []
#y1 = []


for i in range(train_character_list.shape[0]):
	char = train_character_list[i]
	char.calculate_char_features()
	#print('char = ' + str(char._classification))
	#print('width = ' + str(char._feature_width))
	#print('height = ' + str(char._feature_height))
	#print('perimeter = ' + str(char._feature_perimeter))
	#print('curvature = ' + str(char._feature_curvature))
	#print('hor turns = ' + str(char._feature_horizontal_turns))
	#print('vert turns = ' + str(char._feature_vertical_turns))
	#print('frac occupancy = ' + str(char._feature_fractional_occupancy))
	#x.append(char._classification)
	#y.append(char._feature_fractional_occupancy)
	#p_plot.plot_matrix_line(char._data_bw, char._perimeter_path)

#p_plot.plot_xy_data(x, y)	
#sys.exit('done')

for i in range(test_character_list.shape[0]):
	if i%1000 == 0:
		print("calculating features for " + str(i) + "th test event...")
	test_character_list[i].calculate_char_features()

print('begin softmax regression...')
sr.softmax_regression(train_character_list, test_character_list)