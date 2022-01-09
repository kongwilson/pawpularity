"""
DESCRIPTION

Copyright (C) Weicong Kong, 8/01/2022
"""
import timm
from timm import create_model
from fastai.vision.all import set_seed, ImageDataLoaders
from fastai.vision.all import RegressionBlock, Resize, setup_aug_tfms, Brightness, Contrast, Hue, Saturation
from fastai.vision.all import F, Learner, BCEWithLogitsLossFlat
from fastai.vision.all import SaveModelCallback, EarlyStoppingCallback
from fastai.vision.all import *
import gc
import os

from sklearn.model_selection import StratifiedKFold
import math

from utilities import *

set_seed(RANDOM_SEED, reproducible=True)


def get_data(data, fold):
	data_fold = data.copy()
	data_fold['is_valid'] = data['fold'] == fold

	# WK: ImageDataLoaders.from_df returns fastai.data.core.DataLoaders
	data_loaders = ImageDataLoaders.from_df(
		data_fold,
		valid_col='is_valid',
		seed=RANDOM_SEED,
		fn_col='path',
		folder=os.path.join(data_root),
		label_col='norm_score',
		y_block=RegressionBlock,
		bs=8,
		num_workers=0,
		item_tfms=Resize(224),
		batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])
	)
	return data_loaders


def get_learner(data, fold):

	dls = get_data(data, fold)
	model = timm.models.swin_large_patch4_window7_224_in22k(pretrained=True, num_classes=dls.c)
	learn = Learner(dls, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse).to_fp16()
	return learn


def petfinder_rmse(input, target):
	return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))


if __name__ == '__main__':
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
	test_df = test_df.drop(columns=['Id'])

	all_preds = []

	for i in range(N_FOLDS):

		print(f'Fold {i} results')

		learn = get_learner(train_df, fold=i)
		lr_find = learn.lr_find(end_lr=3e-2)
		lr = lr_find.valley
		print(f'fold {i} learning rate found is {lr}')

		learn.fit_one_cycle(
			10, lr, cbs=[
				SaveModelCallback(monitor='petfinder_rmse', fname=f'swin_large_patch4_window7_224_in22k-fold{i}'),
				EarlyStoppingCallback(monitor='petfinder_rmse', min_delta=0.1, comp=np.less, patience=5)
			])

		learn.recorder.plot_loss()

		# learn = learn.to_fp32()

		# learn.export(f'model_fold_{i}.pkl')
		# learn.save(f'model_fold_{i}.pkl')

		dls = get_data(train_df, fold=i)

		test_dl = dls.test_dl(test_df)

		preds, _ = learn.tta(dl=test_dl, n=5, beta=0)

		all_preds.append(preds)

		del learn

		torch.cuda.empty_cache()

		gc.collect()
