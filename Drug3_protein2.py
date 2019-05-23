import numpy as np
import pickle
import pandas as pd
import time

import matplotlib.pyplot as plt


def read_file(filename):
	f = open(filename, 'r')
	list_ = [i.strip("\n").strip(" ") for i in list(f)]
	f.close()
	return list_


def vector_prepare(input_csv):
	df = pd.read_csv(input_csv, delimiter=',')
	list_drug_protein=[]
	# drug_protein = df[["pubchem-id","ensembl-id","confidence score","Drug-Bank-id","uniprot-id"]]
	#
	# # drug_vec_index=[i for i in range(5,df.shape[1]-101)]
	# #
	# # print(drug_vec_index)
	i=0
	while i in range(df.shape[0]):

		drug_id = df.iloc[i, 0]
		drug_vec = df.iloc[i, 5:305].values.tolist()
		# drug_vec = np.asarray(drug_vec)
		# print(drug_vec)
		list_d = [drug_id, drug_vec]
		print(list_d)

		protein_id = df.iloc[i, 1]
		protein_vec = df.iloc[i:(i + 3), 305:405]
		protein_vec = protein_vec.values.tolist()
		protein_vec = np.hstack((np.hstack((protein_vec[0], protein_vec[1])), protein_vec[2]))
		print(protein_vec)

		list_p = [protein_id,protein_vec]
		print(list_p)
		class_label = df.iloc[i, -1]
		list_drug_protein.append([list_d,list_p,class_label])
		i = i + 3

	return list_drug_protein



if __name__=="__main__":
	input_csv = 'training-dataset-rf-mlp.csv'
	output_drug = './output/drug_vec'
	output_protein = './output/protein_vec'

	start = time.time()

	data=vector_prepare(input_csv)
	end = time.time()
	print('vector time elapsed :' + str(end - start))
	print(data)
