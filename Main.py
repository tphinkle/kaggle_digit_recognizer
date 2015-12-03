#Includes
import Character as ch

character_file_directory = '/home/preston/Desktop/Programming/datasci/digit_recognizer/data/'
character_train_file_name = 'short_train.csv'
character_test_file_name = 'test.csv'

character_list = ch.load_characters_kaggle_training_format(character_file_directory +\
 character_train_file_name)


