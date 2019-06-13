
import shutil
import os
import random
import time


def get_file_names(path):
	file_name_list = []
	for filename in os.listdir(path):
		file_name_list.append(filename)
	return file_name_list


def load_list(input_folder):
	file_list_p = []
	file_list_n = []

	for file in os.listdir(input_folder):
		if 'positive' in file:
			print(file)
			pos_file_folder = input_folder + file
			file_list_p = get_file_names(pos_file_folder)

		elif 'negative' in file:
			print(file)
			neg_file_folder = input_folder + file
			file_list_n = get_file_names(neg_file_folder)

	return file_list_p, file_list_n


def split_list(a_list, ratio):
	part_train = int(len(a_list) * ratio[0])
	part_valid = int(len(a_list) * ratio[1])
	t1 = part_train
	t2 = part_train + part_valid
	# random.shuffle(a_list)
	return a_list[:t1], a_list[t1:t2], a_list[t2:]


def check_make_folder(path_train, path_valid, path_test):
	path_train_n = path_train+'/negative/'
	path_train_p = path_train+'/positive/'
	path_valid_n = path_valid+'/negative/'
	path_valid_p = path_valid+'/positive/'
	path_test_n = path_test + '/negative/'
	path_test_p = path_test + '/positive/'

	if os.path.exists(path_train_n):
		shutil.rmtree(path_train_n)
	if os.path.exists(path_train_p):
		shutil.rmtree(path_train_p)
	if os.path.exists(path_valid_n):
		shutil.rmtree(path_valid_n)
	if os.path.exists(path_valid_p):
		shutil.rmtree(path_valid_p)
	if os.path.exists(path_test_n):
		shutil.rmtree(path_test_n)
	if os.path.exists(path_test_p):
		shutil.rmtree(path_test_p)

	os.makedirs(path_train_n)

	os.makedirs(path_train_p)

	os.makedirs(path_valid_n)

	os.makedirs(path_valid_p)

	os.makedirs(path_test_n)

	os.makedirs(path_test_p)
	if os.path.exists('./.mdl_wts.hdf5'):
		os.remove('./.mdl_wts.hdf5')


def split_train_valid_test(path, ratio, labels):

	p_list, n_list = load_list(path)
	train_p, valid_p, test_p = split_list(p_list, ratio)
	train_n, valid_n, test_n = split_list(n_list, ratio)
	train_list = train_p + train_n
	valid_list = valid_p + valid_n
	test_list = test_p + test_n
	print(len(train_list))
	print(len(valid_list))
	print(len(test_list))
	path_train = '../data/train/'
	path_valid = '../data/valid/'
	path_test = '../data/test/'
	check_make_folder(path_train, path_valid, path_test)

	for tmp in os.listdir(path+'/negative/'):
		if tmp in train_n:
			shutil.copy(path+'/negative/'+tmp, path_train+'/negative/'+tmp)
		if tmp in valid_n:
			shutil.copy(path+'/negative/'+tmp, path_valid+'/negative/'+tmp)
		if tmp in test_n:
			shutil.copy(path+'/negative/'+tmp, path_test+'/negative/'+tmp)
		labels[tmp] = 0

	for tmp in os.listdir(path+'/positive/'):

		if tmp in train_p:
			shutil.copy(path+'/positive/'+tmp, path_train+'/positive/'+tmp)
		if tmp in valid_p:
			shutil.copy(path+'/positive/'+tmp, path_valid+'/positive/'+tmp)
		if tmp in test_p:
			shutil.copy(path+'/positive/'+tmp, path_test+'/positive/'+tmp)
		labels[tmp] = 1
	return train_list, valid_list, test_list, path_train, path_valid, path_test



if __name__ == "__main__":

	input_folder = '../data/pre-process/'

	start = time.time()
	labels = {}
	train_list, valid_list, test_list, path_train, path_valid, path_test=\
		split_train_valid_test(input_folder, [0.8, 0.1], labels)

	end = time.time()
	print('vector time elapsed :' + str(end - start))


