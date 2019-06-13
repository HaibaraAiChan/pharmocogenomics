import sys
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

import pickle
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from sklearn import metrics
from data_generator import DataGenerator
from sklearn.metrics import roc_curve, auc, roc_auc_score
import cPickle

def myargs():
	parser = argparse.ArgumentParser()
	parser.add_argument('--protein',
		required=True,
		help='location of the protein pdb file path')
	parser.add_argument('--aux',
		required=True,
		help='location of the auxilary input file')
	parser.add_argument('--r',
		required=False,
		help='radius of the grid to be generated', default=15,
		type=int,
		dest='r')
	parser.add_argument('--N',
		required=False,
		help='number of points long the dimension the generated grid', default=31,
		type=int,
		dest='N')
	args = parser.parse_args()
	return args


def read_file(file):
	myfile = open(file, 'r')
	content = myfile.readlines()
	content = [x.strip().split('\t')[0:-1] for x in content]
	n_len = len(content)
	return content, n_len


def load_pred_data(folder):
	n_len = 0
	n_content = []
	n_label = []
	p_content = []
	p_len = 0
	p_label = []
	for fld in os.listdir(folder):
		if fld=="negative":
			for filenames in os.listdir(os.path.join(folder, fld)):
				tmp_path = os.path.join(folder, fld)
				tmp_file = tmp_path + '/' + filenames
				n_content, n_len = read_file(tmp_file)
				n_label = np.zeros(shape=(n_len,), dtype=int)

		if fld=="positive":
			for filenames in os.listdir(os.path.join(folder, fld)):
				pvalue = float(filenames.split('_')[1])
				if abs(pvalue - p_val) < 1e-10:
					tmp_path = os.path.join(folder, fld)
					tmp_file = tmp_path + '/' + filenames
					p_content, p_len = read_file(tmp_file)
					p_label = np.ones(shape=(p_len,), dtype=int)

	L = p_len + n_len
	print('total length: ', L)
	data = np.vstack((np.array(n_content), np.array(p_content)))
	label = np.hstack((n_label, p_label))
	return data, label


def predict(data_path, model_path):
	pred_list, pred_label = load_pred_data(data_path)
	v_y = np_utils.to_categorical(pred_label, num_classes=2)

	mdl = load_model(model_path)

	score = mdl.predict(pred_list)

	# Compute ROC curve and ROC area for each class
	n_classes = 2

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		y_score = np.array(score[:, i])
		y_test = np.array(v_y[:, i])
		fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=1)
		# fpr_t, tpr_t, _t = metrics.roc_curve(y_test, y_score, pos_label=1)
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(v_y.ravel(), score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	plt.figure()
	lw = 2
	# plt.plot(fpr[0], tpr[0], color='red', lw=lw, label='*0* ROC curve (area = %0.2f)' % roc_auc[0])
	plt.plot(fpr[1], tpr[1], color='red', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('GWAS')
	plt.legend(loc="lower right")
	plt.show()

if __name__=="__main__":

	batch_size = 64
	input_shape = 600
	hidden_units = 300
	num_labels = 2
	epoch = 100
	lr = 0.0001
	model_path = '../model/87/mlp_drug_bank.h5'
	# output = None
	labels = {}
	cnt = 0
	split_ratio = 0.8

	train_list, valid_list, test_list, path_train, path_valid, path_test = split_train_valid(path, split_ratio, labels)
	total_len = len(test_list)
	partition = {"train": train_list, "validation": valid_list, "test": test_list}

	# Parameters
	test_params = {'dim': input_shape,
	                'n_channels': 1,
	                'batch_size': batch_size,
	                'n_classes': 2,
	                'shuffle': True,
	                'path': path_test}


	# loss function for one-hot vector
	# use of sgd optimizer
	# accuracy is good metric for classification tasks

	pre_generator = DataGenerator(partition['train'], labels, **test_params)
	print("the training data is ready")

	# print(labels)

	# valid_voxel, valid_label = load_valid_data(adenines, others, path, 1, 1)
	p_y = np_utils.to_categorical(pre_label_list, num_classes=2)

	model = load_model(model_path)
	score = model.predict_generator(pre_generator,
		steps=np.ceil(total_len / batch_size))
	print(score)
	oname = "./score.pkl"
	cPickle.dump(score, open(oname, "wb"))

	auc = roc_auc_score(y_true=p_y, y_score=score)
	print('auc score from generator', auc)


	path = '../output/'
	p_val = 5e-3
	model_path='./MLP_GWAS.h5'
	# model_path = './result_model/model_2'
	predict(path, p_val,model_path)
