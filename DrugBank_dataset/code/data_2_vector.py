import numpy as np
import pickle
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import itertools
import os
import shutil
import os
import pickle


def write_file(df, path):
	name_list = []
	i = 0
	while i < df.shape[0]:
		protein_id = df.iloc[i, 0]
		drug_id = df.iloc[i, 1]
		drug_vec = df.iloc[i, 2:302].values.tolist()
		protein_id_three = df.iloc[i:(i + 3), 0]
		protein_id_three = protein_id_three.values.tolist()
		assert(protein_id_three[0] == protein_id_three[1] and (protein_id_three[0] == protein_id_three[2]))

		protein_vec = df.iloc[i:(i + 3), 302:402]
		protein_vec = protein_vec.values.tolist()

		protein_vec = list(itertools.chain.from_iterable(protein_vec))

		tmp = np.hstack((drug_vec, protein_vec))
		# print(len(tmp))

		name = protein_id + '_' + drug_id
		# if protein_id == "O14949" and drug_id == 'DB04141':
		# 	print(name+'*'*10)
		if name in name_list:
			# print(name)
			print('----')
			# continue
		else:
			name_list.append(name)
		# file = open(path + name, 'w')
		# tmp_str = '\n'.join(str(e) for e in tmp)
		# file.write(tmp_str)
		# file.close()
		pickle_file = open(path + name+'.pkl', 'wb')
		pickle.dump(tmp, pickle_file)
		i = i + 3
		# print(i/3)
	return name_list


def process_data(file, folder, label_flag):

	df = pd.read_csv(file, delimiter=',')
	if label_flag == 1:
		output_path = folder + 'positive/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		print('positive')

	elif label_flag == 0:
		output_path = folder + 'negative/'
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		print('negative')
	name_list = write_file(df, output_path)

	return name_list

def vector_prepare(input_folder, output_folder):
	pos_file = ''
	neg_file = ''

	for file in os.listdir(input_folder):
		if 'positive-' in file:
			print(file)
			pos_file = input_folder + file
		elif 'negative-' in file:
			print(file)
			neg_file = input_folder + file

	p_name_list=process_data(pos_file, output_folder, 1)
	n_name_list=process_data(neg_file, output_folder, 0)
	print('positive' + str(len(p_name_list)))
	print('negative' + str(len(n_name_list)))
	print(list(set(p_name_list).intersection(set(n_name_list))))


if __name__=="__main__":
	input_folder = '../data/'
	output_folder = '../data/pre-process/'

	start = time.time()
	data = vector_prepare(input_folder, output_folder)
	# train(data)
	end = time.time()
	print('vector time elapsed :' + str(end - start))
	print(data)
