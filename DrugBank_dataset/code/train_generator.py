

import time
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from keras.optimizers import Adam


from keras.models import load_model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import os
import shutil
import os
from data_generator import DataGenerator
from mlp_model import MLP_Builder
from split_train_valid_test import split_train_valid_test


def train_generator(path):
	batch_size = 64
	input_shape = 600
	hidden_units = 300
	num_labels = 2
	epoch = 160
	lr = 0.0001
	output = '../model/'
	# output = None
	labels = {}

	split_ratio = [0.8, 0.1]
	train_list, valid_list, test_list, path_train, path_valid, path_test \
		= split_train_valid_test(path, split_ratio, labels)

	partition = {"train": train_list, "validation": valid_list}

	# Parameters
	train_params = {'dim': input_shape,
	                'n_channels': 1,
	                'batch_size': batch_size,
	                'n_classes': 2,
	                'shuffle': True,
	                'path': path_train}
	valid_params = {'dim': input_shape,
	                'n_channels': 1,
	                'batch_size': batch_size,
	                'n_classes': 2,
	                'shuffle': True,
	                'path': path_valid}

	training_generator = DataGenerator(partition['train'], labels, **train_params)
	print("the training data is ready")
	validation_generator = DataGenerator(partition['validation'], labels, **valid_params)
	print("the validating data is ready")

	model = MLP_Builder.build(input_shape, hidden_units, num_labels)
	# model = multi_gpu_model(model, gpus=2)
	print(model.summary())
	adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	# We add metrics to get more results you want to see
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	earlyStopping = EarlyStopping(monitor='val_loss',
		patience=20,
		verbose=1,
		mode='min')
	mcp_save = ModelCheckpoint('.mdl_wts.hdf5',
		save_best_only=True,
		monitor='val_loss',
		mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
		factor=0.2,
		patience=10,
		verbose=1,
		epsilon=1e-4,
		mode='min')

	print("ready to fit generator")
	# Train model on dataset
	model.fit_generator(generator=training_generator,
		validation_data=validation_generator,
		epochs=epoch,
		verbose=2,
		# workers=1,
		# use_multiprocessing=True,
		# callbacks=[tfCallBack],
		callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
		# validation_split=0.25,

		)

	model_saved = 'mlp_drug_bank.h5'
	if output==None:
		model.save(model_saved)
	else:
		if not os.path.exists(output):
			os.mkdir(output)
		if os.path.exists(model_saved):
			os.remove(model_saved)
		model.save(output + model_saved)
		model.save_weights(output + 'weights.h5')
		mm = load_model(output + model_saved)
		print(mm.summary())

# def train(data):
# 	rtmp = data.values()
#
# 	X = rtmp[:, 0]
# 	Y = rtmp[:, 1]
#
# 	kf = KFold(n_splits=10)
# 	clf = MLPClassifier(solver='lbfgs',
# 		alpha=1e-5,
# 		hidden_layer_sizes=(5, 2),
# 		random_state=1)
#
# 	for train_indices, test_indices in kf.split(X):
# 		print(X[train_indices], Y[train_indices])
# 		clf.fit(X[train_indices], Y[train_indices])
# 		print(clf.score(X[test_indices], Y[test_indices]))


if __name__ == "__main__":

	input_folder = '../data/pre-process/'

	start = time.time()

	train_generator(input_folder)

	end = time.time()
	print('vector time elapsed :' + str(end - start))


