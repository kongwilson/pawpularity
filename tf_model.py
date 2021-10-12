"""
DESCRIPTION

Copyright (C) Weicong Kong, 13/10/2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns
from tqdm.notebook import trange, tqdm
from collections import Counter

import tensorflow as tf

from utils import data_root

tf.random.set_seed(101)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def Conv2D_NET(input_shape):
	init = tf.keras.initializers.GlorotUniform()
	reg = l2(0.0005)
	chanDim = -1
	model = Sequential()

	model.add(
		Conv2D(
			16, (7, 7), strides=(2, 2), padding="valid",
			kernel_initializer=init, kernel_regularizer=reg,
			input_shape=input_shape))

	model.add(
		Conv2D(
			32, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))

	model.add(Conv2D(
		32, (3, 3), strides=(2, 2), padding="same",
		kernel_initializer=init, kernel_regularizer=reg))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.25))

	model.add(Conv2D(
		64, (3, 3), padding="same",
		kernel_initializer=init, kernel_regularizer=reg))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))

	model.add(Conv2D(
		64, (3, 3), strides=(2, 2), padding="same",
		kernel_initializer=init, kernel_regularizer=reg))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Dropout(0.25))

	model.add(Conv2D(
		kernel_initializer=init, kernel_regularizer=reg))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))

	model.add(Flatten())

	model.add(Dense(512, kernel_initializer=init))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(1))
	model.add(Activation("sigmoid"))

	# Compile
	model.compile(
		loss='mse', optimizer='Adam',
		metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae", "mape"])
	return model


def get_image_file_path_test(image_id):
	return os.path.join(data_root, 'test', f'{image_id}.jpg')


def get_image_file_path_train(image_id):
	return os.path.join(data_root, 'train', f'{image_id}.jpg')


def unison_shuffled_copies(a, b):
	assert a.shape[0] == b.shape[0]
	p = np.random.permutation(len(a))
	return a[p], b[p]


if __name__ == '__main__':

	test = pd.read_csv(os.path.join(data_root, 'test.csv'))
	train = pd.read_csv(os.path.join(data_root, 'train.csv'))

	test['file_path'] = test['Id'].apply(get_image_file_path_test)
	train['file_path'] = train['Id'].apply(get_image_file_path_train)

	kaggle_train_images = []
	i = 0
	for i, img_path in enumerate(tqdm(train['file_path'])):
		if img_path.split("/")[-1] in os.listdir(os.path.join(data_root, 'train')):
			img = cv2.imread(img_path)
			img = cv2.resize(img, (128, 128))
			kaggle_train_images.append(img)
		else:
			train = train.drop(i)
	kaggle_train_images = np.array(kaggle_train_images)

	kaggle_test_images = []
	i = 0
	for i, img_path in enumerate(test['file_path']):
		if img_path.split("/")[-1] in os.listdir(os.path.join(data_root, 'test')):
			img = cv2.imread(img_path)
			img = cv2.resize(img, (128, 128))
			kaggle_test_images.append(img)
		else:
			test = test.drop(i)
	kaggle_test_images = np.array(kaggle_test_images)

	kaggle_pawpularity_train = train["Pawpularity"]
	kaggle_pawpularity_train = kaggle_pawpularity_train / 100
	kaggle_pawpularity_train = kaggle_pawpularity_train.values

	np.random.seed(0)

	model = Conv2D_NET((128, 128, 3))
	EarlyStopper = tf.keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10)
	checkpoint_path_quality = os.path.join(data_root, "paws_c2d_v4.h5")

	checkpoint = ModelCheckpoint(
		checkpoint_path_quality,
		monitor='val_rmse',
		verbose=1,
		save_best_only=True,
		mode='min')
	learning_rate_reduction = ReduceLROnPlateau(
		monitor='rmse',
		patience=2,
		verbose=1,
		factor=0.5,
		min_lr=0.000001)

	aug = ImageDataGenerator(
		rotation_range=20, zoom_range=0.15,
		width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
		horizontal_flip=True, fill_mode="nearest")

	all_images, answers = unison_shuffled_copies(kaggle_train_images, kaggle_pawpularity_train)

	super_test_x = all_images[9000:]
	all_images = all_images[:9000]

	super_test_y = answers[9000:]
	answers = answers[:9000]

	all_images = all_images / 255.0
	super_test_x = super_test_x / 255.0

	history = model.fit(
		x=aug.flow(all_images, answers, batch_size=32),
		validation_data=aug.flow(super_test_x, super_test_y, batch_size=32),
		steps_per_epoch=len(all_images) // 256,
		epochs=60, callbacks=[checkpoint, learning_rate_reduction, EarlyStopper])

	loaded_model = load_model("/kaggle/working/paws_c2d_v4.h5")
	norm_test = np.copy(kaggle_test_images)
	norm_test = norm_test / 255
	test_pred = loaded_model.predict(norm_test)
	test_pred = (test_pred * 100)

	sub = pd.DataFrame()
	sub['Id'] = test['Id']
	sub['Pawpularity'] = test_pred
	sub.to_csv('submission.csv', index=False)
	sub.head(2)
