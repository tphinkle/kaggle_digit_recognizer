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
character_test_file_name = 'train.csv'

train_character_list = chf.load_characters_kaggle_format(character_file_directory +\
 character_train_file_name, 'train')
test_character_list = chf.load_characters_kaggle_format(character_file_directory +\
	character_test_file_name, 'train')


#print('calculating features...')
#x = []
#y_list = []



for i in range(train_character_list.shape[0]):
	char = train_character_list[i]
	char.calculate_char_features()
	#x.append(char._classification)
	#for j in range(ch.character.total_features):
		#y_list[j].append(char._feature_list[j])
	#print('char = ' + str(char._classification))
	#print('width = ' + str(char._feature_width))
	#print('height = ' + str(char._feature_height))
	#print('perimeter = ' + str(char._feature_perimeter))
	#print('curvature = ' + str(char._feature_curvature))
	#print('hor turns = ' + str(char._feature_horizontal_turns))
	#print('vert turns = ' + str(char._feature_vertical_turns))
	#print('frac occupancy = ' + str(char._feature_fractional_occupancy))
	#x.append(char._classification)
	#y.append(char._feature_moment_y)
	#print("x moment = " + str(char._feature_moment_x))
	#print("y moment = " + str(char._feature_moment_y))
	#p_plot.plot_matrix_line(char._data_bw, char._perimeter_path)

	
#for j in range(ch.character.total_features):
	#p_plot.plot_xy_data(x, y_list[j])	
#sys.exit('done')

for i in range(test_character_list.shape[0]):
	if i%1000 == 0:
		print("calculating features for " + str(i) + "th test event...")
	test_character_list[i].calculate_char_features()

print('begin softmax regression...')

sr.softmax_regression(train_character_list, test_character_list)

correct_classifications = []
total_instances = []
for i in range(ch.character.total_classifications):
	#y_list.append([])
	correct_classifications.append(0)
	total_instances.append(0)

for i in range(0, test_character_list.shape[0]):
	char = test_character_list[i]
	
	if char._predicted_classification == char._classification:
		correct_classifications[int(char._classification)] = correct_classifications[int(char._classification)] + 1

	total_instances[int(char._classification)] = total_instances[int(char._classification)] + 1

for i in range(0, ch.character.total_features):
	print('i = ' + str(i) + ': ' + str(correct_classifications[i]) + "/" + str(total_instances[i]))
