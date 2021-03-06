"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import gc
import glob
import re

import torch.optim
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR

import xgboost as xgb
import optuna

from loader import *
from model import *
from utilities import *


def train_epoch(train_loader, model, loss_func, optimizer, epoch, scheduler=None):
	metric_monitor = MetricMonitor()
	model.train()
	stream = tqdm(train_loader)

	device = get_default_device()

	with torch.enable_grad():

		for i, (images, dense, target) in enumerate(stream, start=1):
			images = images.to(device, non_blocking=True)
			dense = dense.to(device, non_blocking=True)
			target = target.to(device, non_blocking=True).float().view(-1, 1)

			output = model(images, dense)

			loss = loss_func(output, target)

			metric_monitor.update('Loss', loss.item())
			metric_monitor.update('RMSE', rmse_from_classifier_output(output, target))
			loss.backward()
			optimizer.step()

			if scheduler is not None:
				scheduler.step()

			optimizer.zero_grad()
			stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

	return


def validate(val_loader, model, loss_func, epoch):
	metric_monitor = MetricMonitor()
	model.eval()
	stream = tqdm(val_loader)
	device = get_default_device()

	final_targets = []
	final_outputs = []
	with torch.no_grad():
		for i, (images, dense, target) in enumerate(stream, start=1):
			images = images.to(device, non_blocking=True)
			dense = dense.to(device, non_blocking=True)
			target = target.to(device, non_blocking=True).float().view(-1, 1)
			output = model(images, dense)
			loss = loss_func(output, target)
			metric_monitor.update('Loss', loss.item())
			metric_monitor.update('RMSE', rmse_from_classifier_output(output, target))
			stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

			targets = (target.detach().cpu().numpy() * 100).tolist()
			# WKNOTE: because we are using class for reg
			outputs = (torch.sigmoid(output).detach().cpu().numpy() * 100).tolist()

			final_targets.extend(targets)
			final_outputs.extend(outputs)
	return final_outputs, final_targets


def extract_intermediate_outputs_and_targets(model, data_loader):
	device = get_default_device()
	new_x = None
	y = None
	preds = None

	model.eval()

	with torch.no_grad():
		for (images, dense, target) in tqdm(data_loader, desc=f'Training with XGB. '):
			images = images.to(device, non_blocking=True)
			dense = dense.to(device, non_blocking=True)
			batch_preds = torch.sigmoid(model(images, dense)).cpu().detach().numpy() * 100
			batch_embed = activation['swin_head']
			batch_x = np.concatenate([batch_embed, dense.cpu().detach().numpy()], axis=1)
			if preds is None:
				preds = batch_preds
				new_x = batch_x
				y = target.view(-1, 1).cpu().detach().numpy()
			else:
				preds = np.vstack((preds, batch_preds))
				new_x = np.vstack((new_x, batch_x))
				y = np.vstack((y, target.view(-1, 1).cpu().detach().numpy()))

	return new_x, y * 100, preds


def xgb_to_the_result(model_type, img_size=384, batch_size=4, embed_size=128, hidden_size=64, n_folds=10):
	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(
		root_dir=data_root, train=True, n_folds=n_folds, model_dir=model_root, image_size=img_size)
	test_preprocessor = PawPreprocessor(root_dir=data_root, train=False, image_size=img_size)

	preds = None
	model = model_type(3, len(preprocessor.features), embed_size, hidden_size)
	all_models_checkpoints = glob.glob(model_root + os.path.sep + f'{str(model)}_*.pth.tar')

	for model_path in all_models_checkpoints:
		fold_info = [f for f in model_path.split('_') if f.startswith('fold')][0]
		pattern = re.compile(r'\d+')
		result = pattern.search(fold_info)
		fold = int(result.group()) - 1

		train_loader = preprocessor.get_dataloader(
			fold=fold, for_validation=False,
			transform=get_albumentation_transform_for_training(img_size), batch_size=batch_size)
		val_loader = preprocessor.get_dataloader(
			fold=fold, for_validation=True,
			transform=get_albumentation_transform_for_validation(img_size), batch_size=batch_size)

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size)
		# WKNOTE: get activation from an intermediate layer
		model.model.head.register_forward_hook(get_activation('swin_head'))
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)

		xgb_train_x, xgb_train_y, train_preds = extract_intermediate_outputs_and_targets(model, train_loader)
		xgb_val_x, xgb_val_y, val_preds = extract_intermediate_outputs_and_targets(model, val_loader)

		def loss_func(trial: optuna.trial.Trial):
			params = {
				'n_estimators': trial.suggest_int('n_estimators', 10, 1000),  # default = 100
				'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),  # default = 'gbtree'
				'gamma': trial.suggest_uniform('gamma', 0, 100),  # default = 0
				'max_depth': trial.suggest_int('max_depth', 1, 11),  # default = 6, the deeper, the easier to overfit
				'learning_rate': trial.suggest_uniform('learning_rate', 0, 1),  # default = 0.3
				'min_child_weight': trial.suggest_uniform('min_child_weight', 0.1, 100),  # default = 1
				'max_delta_step': trial.suggest_int('max_delta_step', 0, 11),  # default = 0
				'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 1),
				'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 1),
			}

			xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, **params)
			xgb_model.fit(xgb_train_x, xgb_train_y)
			xgb_val_preds = xgb_model.predict(xgb_val_x)
			xgb_val_preds = prediction_validity_check(xgb_val_preds)

			rmse_val = round(mean_squared_error(xgb_val_y, xgb_val_preds, squared=False), 5)

			return rmse_val

		model_name = os.path.basename(model_path)

		study = optuna.create_study(direction='minimize', study_name=model_name, storage=f'sqlite:///{model_name}.db')
		study.optimize(loss_func, n_trials=200)
		best_params = study.best_params
		print(f'the best model params are found on Trial #{study.best_trial.number}')
		print(best_params)

		xgb_model = xgb.XGBRegressor(random_state=RANDOM_SEED, **best_params)
		xgb_model.fit(xgb_train_x, xgb_train_y)
		xgb_train_preds = xgb_model.predict(xgb_train_x)
		xgb_val_preds = xgb_model.predict(xgb_val_x)

		rmse_train = round(mean_squared_error(xgb_train_y, xgb_train_preds, squared=False), 5)
		rmse_val = round(mean_squared_error(xgb_val_y, xgb_val_preds, squared=False), 5)

		print(f'train rmse: {rmse_train}, val rmse: {rmse_val}')

		model_name = os.path.basename(model_path)
		model_path = os.path.join(
			model_root, f"XGB-{rmse_val:.5f}_{model_name}.json")
		xgb_model.save_model(model_path)

		test_loader = test_preprocessor.get_dataloader()

		xgb_test_x, xgb_test_y, test_preds = extract_intermediate_outputs_and_targets(model, test_loader)
		xgb_test_preds = xgb_model.predict(xgb_test_x)
		xgb_test_preds = prediction_validity_check(xgb_test_preds)

		if preds is None:
			preds = xgb_test_preds
		else:
			preds += xgb_test_preds

	preds /= (len(all_models_checkpoints))

	return preds


activation = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.cpu().detach().numpy()

	return hook


def train_benchmark(model_type=PawSwinTransformerLarge4Patch12Win22k384, patience=3, fine_tune=False):
	seed_everything()
	gc.enable()
	img_size = 384
	n_folds = 10
	batch_size = 1
	epochs = 30
	embed_size = 128
	hidden_size = 64
	lr = 1e-5
	max_lr = 1e-3
	min_lr = 1e-7
	weight_decay = 1e-6
	preprocessor = PawPreprocessor(
		root_dir=data_root, train=True, n_folds=n_folds, model_dir=model_root, image_size=img_size)
	device = get_default_device()

	for fold in range(n_folds):

		train_loader = preprocessor.get_dataloader(
			fold=fold, for_validation=False, transform=get_albumentation_transform_for_training(img_size))
		val_loader = preprocessor.get_dataloader(
			fold=fold, for_validation=True, transform=get_albumentation_transform_for_validation(img_size))

		# model = PawClassifier(img_size, img_size, 3, len(preprocessor.features), embed_size, hidden_size)
		model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=True, fine_tune=fine_tune)
		model.to(device)
		loss_func = nn.BCEWithLogitsLoss()
		optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
		# scheduler = OneCycleLR(
		#     optimizer,
		#     max_lr=max_lr,
		#     steps_per_epoch=int(len(train_dataset) / batch_size) + 1,
		#     epochs=epochs,
		# )
		scheduler = CosineAnnealingWarmRestarts(
			optimizer,
			T_0=100,
			eta_min=min_lr,
			last_epoch=-1
		)

		epochs_with_no_improvement = 0
		fine_tune_with_no_augmentation = False

		# Training and Validation Loop
		best_rmse = np.inf
		best_epoch = np.inf
		best_model_path = None
		for epoch in range(1, epochs + 1):

			if epochs_with_no_improvement >= patience:
				if fine_tune_with_no_augmentation:
					print(f'No improvement with no augmentation for {patience} epochs, early stop')
					break
				else:
					print(f'No improvement with AUGMENTATION for {patience} epochs, switch to NON-AUG training')
					epochs_with_no_improvement = 0
					fine_tune_with_no_augmentation = True
					train_loader = preprocessor.get_dataloader(
						fold=fold, for_validation=False, transform=get_albumentation_transform_for_validation(img_size))

			train_epoch(train_loader, model, loss_func, optimizer, epoch, scheduler)

			predictions, valid_targets = validate(val_loader, model, loss_func, epoch)
			rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 5)
			if rmse <= best_rmse:
				epochs_with_no_improvement = 0
				best_rmse = rmse
				best_epoch = epoch
				if best_model_path is not None:
					os.remove(best_model_path)
				best_model_path = os.path.join(
					model_root,
					f"{str(model)}_fold{fold + 1}_epoch{epoch}_{rmse}-rmse.pth.tar")
				torch.save(model.state_dict(), best_model_path)
			else:
				epochs_with_no_improvement += 1

		print(f'The best RMSE: {best_rmse} for fold {fold + 1} was achieved on epoch: {best_epoch}.')
		print(f'The Best saved model is: {best_model_path}')
		print(''.join(['#'] * 50))
		del model
		gc.collect()
		torch.cuda.empty_cache()


if __name__ == '__main__':
	train_benchmark(model_type=PawSwinTransformerLarge4Patch12Win22k384, patience=3, fine_tune=True)
	train_benchmark(model_type=PawSwinTransformerLarge4Patch12Win22k384, patience=3, fine_tune=False)
	# preds1 = xgb_to_the_result(PawSwinTransformerLarge4Patch12Win384)
	preds2 = xgb_to_the_result(PawSwinTransformerLarge4Patch12Win22k384)
	# print(preds1)
	print(preds2)