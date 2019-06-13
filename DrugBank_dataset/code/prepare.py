import numpy as np
import pickle
import pandas as pd
import time

import matplotlib.pyplot as plt
import itertools
import os
import shutil
import os
import pickle


def process_data(neg_file, ex_neg_file):
	"""

	:param neg_file: 
	:param ex_neg_file: 
	:return:
	O14949,DB04141 -> O14949,DB02210
	O76082,DB00583 -> O76082,DB07506
	O76083,DB00201 -> O76083,DB02974
	O76074,DB00201 -> O76074,DB01643
	O43612,DB03088 -> O43612,DB00928
	O00408,DB00201 -> O00408,DB04751
	"""
	templets=[['O14949','DB04141','DB02210'],
	          ['O76082','DB00583','DB07506'],
	          ['O76083','DB00201','DB02974'],
	          ['O76074','DB00201','DB01643'],
	          ['O43612','DB03088','DB00928'],
	          ['O00408','DB00201','DB04751']]

	df = pd.read_csv(neg_file, delimiter=',')
	df_ex = pd.read_csv(ex_neg_file, delimiter=',')
	# print(df_ex.size)
	# print(df_ex.shape)
	# print(df_ex.values)
	df_value = df.values
	ex_value = df_ex.values
	for i in range(len(templets)):
		protein = templets[i][0]
		drug_id = templets[i][1]
		tmp = df_value[np.where((df_value[:, 0] == protein) * (df_value[:, 1] == drug_id))]
		# print(tmp)
		tmp_tail = tmp[np.where((tmp[:, 0] == templets[i][0]))][:, 302:402]
		ex_head = ex_value[np.where((ex_value[:, 0] == templets[i][0]))]
		ex_head = np.insert(ex_head, 1, ex_head[0], axis=0)
		ex_head = np.insert(ex_head, 2, ex_head[0], axis=0)
		replace = np.hstack((ex_head, tmp_tail))

		for j in range(3):
			mask = np.all(df_value == tmp[j], axis=1)
			df_value[mask] = list(replace[j])
		print(df_value[np.where((df_value[:, 0] == templets[i][0]) * (df_value[:, 1] == templets[i][2]))][0:4])
		# df.drop(df.index[[:]])
	# df = df.iloc[0:0]
	df2 = pd.DataFrame(df_value, columns=df.columns.tolist())
	df2.to_csv(neg_file, sep=',',index=False)

	return True


def prepare(input_folder):
	neg_file = ''
	ex_neg_file = ''
	for file in os.listdir(input_folder):

		if 'negative-data' in file:
			print(file)
			neg_file = input_folder + file
		elif 'Extra_negative' in file:
			print(file)
			ex_neg_file = input_folder + file
	n_name_list = process_data(neg_file, ex_neg_file)


if __name__ == "__main__":
	input_folder = '../data/'

	# output_folder = '../data/pre-process/'

	start = time.time()
	data = prepare(input_folder)

	end = time.time()
	print('vector time elapsed :' + str(end - start))
	print(data)
