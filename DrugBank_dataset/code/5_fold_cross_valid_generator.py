

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


def fold_cross_valid(train_list, f_num, idx):
	sub = int(len(train_list)/f_num)
	idx = idx+1
	train_list.sort()
	new_valid_list = train_list[sub*(idx-1):sub*idx]
	new_train_list = list(set(train_list) - set(new_valid_list))

	return new_train_list, new_valid_list


def train_generator_cross_valid(path):
	batch_size = 32
	input_shape = 600
	hidden_units = 300
	num_labels = 2
	epoch = 100
	lr = 0.0001
	output = '../model/'
	# output = None
	labels = {}
	fold_num = 5

	split_ratio = [0.9, 0.0]
	train_list, valid_list, test_list, path_train, path_valid, path_test \
		= split_train_valid_test(path, split_ratio, labels)
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
		factor=0.5,
		patience=10,
		verbose=1,
		min_delta=1e-4,
		mode='min')

	for stage in range(fold_num):
		print('times:' + str(stage))
		new_train_list, new_valid_list = fold_cross_valid(train_list, fold_num, stage)
		print(len(new_train_list))
		print(len(new_valid_list))
		partition = {"train": new_train_list, "validation": new_valid_list}

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
		                'path': path_train}

		training_generator = DataGenerator(partition['train'], labels, **train_params)
		print("the training data is ready")
		validation_generator = DataGenerator(partition['validation'], labels, **valid_params)
		print("the validating data is ready")

		print("ready to fit generator")

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

	model_saved = 'mlp_drug_bank_cross.h5'
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





if __name__ == "__main__":

	input_folder = '../data/pre-process/'

	start = time.time()

	# train_generator(input_folder)
	train_generator_cross_valid(input_folder)
	end = time.time()
	print('vector time elapsed :' + str(end - start))


