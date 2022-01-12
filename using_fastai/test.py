"""
DESCRIPTION

Copyright (C) Weicong Kong, 10/01/2022
"""
import optuna
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error

from using_fastai.loader import *
import fastai

FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group',
	'Collage', 'Human', 'Occlusion', 'Info', 'Blur']


def add_tabular_features_with_xgboosting(
		learner: fastai.learner.Learner, train_val_data: pd.DataFrame, fold: int, checkpoint_name: str):

	xgb_model_paths = glob.glob('models', f'*{str(checkpoint_name)}.json')
	if len(xgb_model_paths) > 0:
		xgb_model = xgb.XGBRegressor()
		xgb_model.load_model(xgb_model_paths[0])
		return xgb_model

	mask = train_val_data['fold'] == fold
	train = train_val_data[~mask].copy()
	val = train_val_data[mask].copy()
	timm_model_name = checkpoint_name.split('-')[0]
	fastai_loader = get_data(train_val_data, fold=fold, timm_model_name=timm_model_name)
	train_dl = fastai_loader.test_dl(train)
	print('generate predicted outputs for training samples')
	train_preds, _ = learner.tta(dl=train_dl, n=5, beta=0)
	val_dl = fastai_loader.test_dl(val)
	print('generate predicted outputs for validation samples')
	val_preds, _ = learner.tta(dl=val_dl, n=5, beta=0)

	train_feats = train[FEATURES].values
	val_feats = val[FEATURES].values

	xgb_train_x = np.concatenate((np.array(train_preds), train_feats), axis=1)
	xgb_train_y = train[['norm_score']].values

	xgb_val_x = np.concatenate((np.array(val_preds), val_feats), axis=1)
	xgb_val_y = val[['norm_score']].values

	def loss_func(trial: optuna.trial.Trial):
		params = {
			'n_estimators': trial.suggest_int('n_estimators', 10, 1000),  # default = 100
			'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
			# default = 'gbtree'
			'gamma': trial.suggest_uniform('gamma', 0, 100),  # default = 0
			'max_depth': trial.suggest_int('max_depth', 1, 11),
			# default = 6, the deeper, the easier to overfit
			'learning_rate': trial.suggest_uniform('learning_rate', 0, 1),  # default = 0.3
			'min_child_weight': trial.suggest_uniform('min_child_weight', 0.1, 100),  # default = 1
			'max_delta_step': trial.suggest_int('max_delta_step', 0, 11),  # default = 0
			'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 1),
			'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 1),
		}

		xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, **params)
		xgb_model.fit(xgb_train_x, xgb_train_y)
		xgb_val_preds = xgb_model.predict(xgb_val_x)
		xgb_val_preds = prediction_validity_check(xgb_val_preds, max_val=1)

		rmse_val = round(mean_squared_error(xgb_val_y * 100, xgb_val_preds * 100, squared=False), 5)

		return rmse_val

	study_db_path = os.path.join('models', f'{checkpoint_name}.db')
	study = optuna.create_study(
		direction='minimize', study_name=checkpoint_name,
		storage=f'sqlite:///{study_db_path}', load_if_exists=True)
	study.optimize(loss_func, n_trials=200)
	best_params = study.best_params
	print(f'the best model params are found on Trial #{study.best_trial.number}')
	print(best_params)

	xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, **best_params)
	xgb_model.fit(xgb_train_x, xgb_train_y)
	xgb_train_preds = xgb_model.predict(xgb_train_x)
	xgb_val_preds = xgb_model.predict(xgb_val_x)

	rmse_train = round(mean_squared_error(xgb_train_y * 100, xgb_train_preds * 100, squared=False), 5)
	rmse_val = round(mean_squared_error(xgb_val_y * 100, xgb_val_preds * 100, squared=False), 5)

	print(f'train rmse: {rmse_train}, val rmse: {rmse_val}')

	model_path = os.path.join(
		'models', f"XGB-{rmse_val:.5f}_{checkpoint_name}.json")
	xgb_model.save_model(model_path)

	return xgb_model, rmse_val


def save_test_results(score_df: pd.DataFrame):
	if os.path.exists('test_results.csv'):
		test_results = pd.read_csv('test_results.csv')
		test_results = test_results.append(score_df)
	else:
		test_results = score_df
	test_results.to_csv('test_results.csv', index=False)


if __name__ == '__main__':
	model_name = 'bonky'  # bonky, swin_large_patch4_window12_384_in22k, swin_large_patch4_window7_224_in22k
	include_tabular = True
	metric_name = None  # valid_loss, petfinder_rmse, None

	if model_name == 'bonky':
		N_FOLDS = 10
		train_df = pd.read_csv('train_df.csv')
		train_df['Id'] = train_df['path'].apply(lambda x: os.path.basename(x).split('.')[0])
		train_df['path'] = train_df['Id'].apply(lambda x: os.path.join('train', f'{x}.jpg'))
	else:
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
		checkpoint_names = get_model_checkpoint_names(model_name, i, metric_name=metric_name)
		for cp_name in checkpoint_names:
			print(f'loading {cp_name}')
			learn.load(cp_name)
			val_metrics = learn.validate()  # compute the validation loss and metrics
			print(val_metrics)

			dls = get_data(train_df, fold=i, timm_model_name=model_name)

			test_dl = dls.test_dl(test_df)

			preds, _ = learn.tta(dl=test_dl, n=5, beta=0)

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
			xgb_val_rmse = None

			if include_tabular:
				print('prepare to add tabular features')
				xgb_model, xgb_val_rmse = add_tabular_features_with_xgboosting(learn, train_df, i, cp_name)
				test_feats = test_df[FEATURES].values

				xgb_test_x = np.concatenate((np.array(preds), test_feats), axis=1)

				preds = xgb_model.predict(xgb_test_x)
				preds = prediction_validity_check(preds, max_val=1)

			val_result = pd.DataFrame(
				data=[
					[model_name, i] +
					val_metrics.items +
					[petfinder_rmse_tta, petfinder_rmse_recal, xgb_val_rmse, datetime.datetime.now(), cp_name, dls.bs]],
				columns=[
					'model_name', 'fold',
					'valid_loss', 'petfinder_rmse', 'petfinder_rmse_tta', 'petfinder_rmse_recal', 'xgb_rmse',
					'trained_time', 'checkpoint_name', 'batch_size'])
			save_test_results(val_result)

			all_preds.append(preds)

		del learn

		torch.cuda.empty_cache()

		gc.collect()

	sub = pd.DataFrame()
	sub['Id'] = test_df['Id']
	preds = np.mean(np.stack(all_preds), axis=0)
	sub['Pawpularity'] = preds * 100
	sub.to_csv('submission.csv', index=False)
