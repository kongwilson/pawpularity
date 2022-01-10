"""
DESCRIPTION

Copyright (C) Weicong Kong, 10/01/2022
"""
import pandas as pd

from using_fastai.loader import *


if __name__ == '__main__':
	model_name = 'swin_large_patch4_window7_224_in22k'

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
		checkpoint_names = get_model_checkpoint_names(model_name, i, metric_name=None)
		for cp_name in checkpoint_names:
			learn.load(cp_name)
			val_metrics = learn.validate()  # compute the validation loss and metrics

			dls = get_data(train_df, fold=i)

			test_dl = dls.test_dl(test_df)

			preds, _ = learn.tta(dl=test_dl, n=5, beta=0)

			all_preds.append(preds)

		del learn

		torch.cuda.empty_cache()

		gc.collect()

	sub = pd.DataFrame()
	sub['Id'] = test_df['Id']
	preds = np.mean(np.stack(all_preds), axis=0)
	sub['Pawpularity'] = preds * 100
	sub.to_csv('submission.csv', index=False)
