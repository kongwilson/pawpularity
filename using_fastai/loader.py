"""
DESCRIPTION

Copyright (C) Weicong Kong, 8/01/2022
"""
import numpy as np
import pandas as pd
import timm
from timm import create_model
from fastai.vision.all import set_seed, ImageDataLoaders
from fastai.vision.all import RegressionBlock, Resize, setup_aug_tfms, Brightness, Contrast, Hue, Saturation
from fastai.vision.all import F, Learner, BCEWithLogitsLossFlat
from fastai.vision.all import SaveModelCallback, EarlyStoppingCallback
from fastai.vision.all import *
import gc
import os
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import math

from utilities import RANDOM_SEED, data_root, seed_everything, prediction_validity_check

set_seed(RANDOM_SEED, reproducible=True)
seed_everything()


def get_data(data, fold, timm_model_name='swin_large_patch4_window7_224_in22k'):
	data_fold = data.copy()
	data_fold['is_valid'] = data['fold'] == fold
	if timm_model_name == 'swin_large_patch4_window7_224_in22k':
		img_size = 224
	elif timm_model_name == 'bonky':
		img_size = 224
	else:
		img_size = 384

	# WK: ImageDataLoaders.from_df returns fastai.data.core.DataLoaders
	data_loaders = ImageDataLoaders.from_df(
		data_fold,
		valid_col='is_valid',
		seed=RANDOM_SEED,
		fn_col='path',
		folder=os.path.join(data_root),
		label_col='norm_score',
		y_block=RegressionBlock,
		bs=4,
		num_workers=0,
		item_tfms=Resize(img_size),
		batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation(), FlipItem()])
	)
	return data_loaders


def get_learner(data, fold, timm_model_name='swin_large_patch4_window7_224_in22k'):

	dls = get_data(data, fold, timm_model_name=timm_model_name)
	if timm_model_name == 'swin_large_patch4_window12_384_in22k':
		model = timm.models.swin_large_patch4_window12_384_in22k(pretrained=True, num_classes=dls.c)
	elif timm_model_name == 'bonky':
		model = timm.models.swin_large_patch4_window7_224_in22k(pretrained=True, num_classes=dls.c)
	else:
		model = timm.models.swin_large_patch4_window7_224_in22k(pretrained=True, num_classes=dls.c)
	learner = Learner(dls, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse).to_fp16()
	return learner


def petfinder_rmse(input, target):
	return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))


def get_model_checkpoint_name(timm_model_name, fold):
	return f'{timm_model_name}-fold{fold}'


def get_model_checkpoint_names(timm_model_name, fold, metric_name=None):
	if metric_name is None:
		all_models_checkpoints = glob.glob(os.path.join('models', f'{str(timm_model_name)}-fold{fold}*.pth'))
	else:
		all_models_checkpoints = glob.glob(
			os.path.join('models', f'{str(timm_model_name)}-fold{fold}*-{metric_name}.pth'))
	checkpoint_names = [os.path.basename(p).split('.')[0] for p in all_models_checkpoints]
	return checkpoint_names


def save_best_score(score_df: pd.DataFrame):
	if os.path.exists('best_score.csv'):
		best_scores = pd.read_csv('best_score.csv')
		best_scores = best_scores.append(score_df)
	else:
		best_scores = score_df
	best_scores.to_csv('best_score.csv', index=False)


if __name__ == '__main__':
	model_name = 'swin_large_patch4_window12_384_in22k'
	train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
	train_df['path'] = train_df['Id'].apply(lambda x: os.path.join('train', f'{x}.jpg'))
	train_df = train_df.drop(columns=['Id'])
	train_df = train_df.sample(frac=1).reset_index(drop=True)  # shuffle dataframe
	train_df['norm_score'] = train_df['Pawpularity'] / 100

	# Rice rule
	num_bins = int(np.ceil(2 * ((len(train_df)) ** (1. / 3))))
	train_df['bins'] = pd.cut(train_df['norm_score'], bins=num_bins, labels=False)  # WK: use 'bins' to do stratified kfold

	train_df['fold'] = -1
	N_FOLDS = 10
	strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=RANDOM_SEED, shuffle=True)
	for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
		train_df.loc[train_index, 'fold'] = i

	test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))
	test_df['Pawpularity'] = [1]*len(test_df)
	test_df['path'] = test_df['Id'].apply(lambda x: os.path.join('test', f'{x}.jpg'))

	all_preds = []

	for i in range(N_FOLDS):

		print(f'Fold {i} results')

		learn = get_learner(train_df, fold=i, timm_model_name=model_name)
		lr_find = learn.lr_find(end_lr=3e-2)
		lr = lr_find.valley
		print(f'fold {i} learning rate found is {lr}')

		checkpoint_filename = get_model_checkpoint_name(model_name, i)
		checkpoint_filename_petfinder = f'{checkpoint_filename}-petfinder_rmse'
		checkpoint_filename_valid_loss = f'{checkpoint_filename}-valid_loss'
		learn.fit_one_cycle(
			5, 2e-5, cbs=[
				SaveModelCallback(
					monitor='petfinder_rmse', fname=checkpoint_filename_petfinder, comp=np.less),
				SaveModelCallback(
					monitor='valid_loss', fname=checkpoint_filename_valid_loss, comp=np.less),
				EarlyStoppingCallback(monitor='petfinder_rmse', min_delta=0.0, comp=np.less, patience=2)
			])

		checkpoints = [checkpoint_filename_petfinder, checkpoint_filename_valid_loss]
		remark = 'max_lr_2e-5+randome_seed_365+epochs_5+patience_2'
		for checkpoint_name in checkpoints:
			learn.load(checkpoint_filename_petfinder)
			val_metrics = learn.validate()  # compute the validation loss and metrics

			dls = get_data(train_df, fold=i, timm_model_name=model_name)

			val_df = train_df[train_df['fold'] == i].copy()
			labels = val_df['Pawpularity'].values
			val_dl = dls.test_dl(val_df)
			val_preds_tta, _ = learn.tta(dl=val_dl, n=5, beta=0)
			val_preds, _ = learn.get_preds(dl=val_dl)
			val_preds_tta *= 100
			val_preds *= 100
			petfinder_rmse_tta = mean_squared_error(val_preds_tta, labels, squared=False)
			petfinder_rmse_recal = mean_squared_error(val_preds, labels, squared=False)
			print('val pred tta petfinder_rmse:', petfinder_rmse_tta)
			print('val pred petfinder_rmse:', petfinder_rmse_recal)

			best_score = pd.DataFrame(
				data=[
					[model_name, i] +
					val_metrics.items +
					[petfinder_rmse_tta, petfinder_rmse_recal, datetime.datetime.now(), checkpoint_name, remark]],
				columns=[
					'model_name', 'fold',
					'valid_loss', 'petfinder_rmse', 'petfinder_rmse_tta', 'petfinder_rmse_recal',
					'trained_time', 'checkpoint_name', 'remark'])
			save_best_score(best_score)

		# learn = learn.to_fp32()

		# learn.export(f'model_fold_{i}.pkl')
		# learn.save(f'model_fold_{i}.pkl')

		del learn

		torch.cuda.empty_cache()

		gc.collect()
