"""
DESCRIPTION

Copyright (C) Weicong Kong, 29/10/2021
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


class Learner(object):

	def __init__(
			self,
			data_root=None, model_root=None,
			model_type=PawSwinTransformerLarge4Patch12Win22k384, patience=3, fine_tune=False,
			img_size=384,
			n_folds=10,
			batch_size=1,
			epochs=30,
			embed_size=128,
			hidden_size=64,
			lr=1e-5,
			max_lr=1e-3,
			min_lr=1e-7,
			weight_decay=1e-6,

	):
		if data_root is None:
			os.makedirs('data', exist_ok=True)
			self.data_root = 'data'
		else:
			self.data_root = data_root

		if model_root is None:
			os.makedirs('models', exist_ok=True)
			self.model_root = 'models'
		else:
			self.model_root = model_root

		self.activation = {}  # to store intermediate layers' activations if needed

		self.model_type = model_type
		self.patience = patience
		self.fine_tune = fine_tune
		self.device = get_default_device()

		self.img_size = img_size
		self.n_folds = n_folds
		self.batch_size = batch_size
		self.epochs = epochs
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.lr = lr 
		self.max_lr = max_lr 
		self.min_lr = min_lr
		self.weight_decay = weight_decay

	def _assert_model_type_is_supported(self):
		assert issubclass(self.model_type, PawVisionTransformerTiny16Patch384), \
			f'{self.model_type.__name__} is not supported'

	def train_one_epoch(self, train_loader, model, loss_func, optimizer, epoch, scheduler=None):
		metric_monitor = MetricMonitor()
		model.train()
		stream = tqdm(train_loader)

		with torch.enable_grad():

			for i, (images, dense, target) in enumerate(stream, start=1):
				images = images.to(self.device, non_blocking=True)
				dense = dense.to(self.device, non_blocking=True)
				target = target.to(self.device, non_blocking=True).float().view(-1, 1)

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

	def validate(self, val_loader, model, loss_func, epoch):
		metric_monitor = MetricMonitor()
		model.eval()
		stream = tqdm(val_loader)
		final_targets = []
		final_outputs = []

		with torch.no_grad():
			for i, (images, dense, target) in enumerate(stream, start=1):
				images = images.to(self.device, non_blocking=True)
				dense = dense.to(self.device, non_blocking=True)
				target = target.to(self.device, non_blocking=True).float().view(-1, 1)
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

	def get_activate_for_model_hook(self, name):
		def hook(model, input, output):
			self.activation[name] = output.detach().cpu().numpy()
		return hook

	def extract_intermediate_outputs_and_targets(self, model, data_loader):
		device = get_default_device()
		new_x = None
		y = None
		preds = None

		model.eval()

		with torch.no_grad():
			for (images, dense, target) in tqdm(data_loader, desc=f'Training with XGB. '):
				images = images.to(device, non_blocking=True)
				dense = dense.to(device, non_blocking=True)
				batch_preds = torch.sigmoid(model(images, dense)).detach().cpu().numpy() * 100
				batch_embed = self.activation['swin_head']
				batch_x = np.concatenate([batch_embed, dense.detach().cpu().numpy()], axis=1)
				if preds is None:
					preds = batch_preds
					new_x = batch_x
					y = target.view(-1, 1).detach().cpu().numpy()
				else:
					preds = np.vstack((preds, batch_preds))
					new_x = np.vstack((new_x, batch_x))
					y = np.vstack((y, target.view(-1, 1).detach().cpu().numpy()))

		return new_x, y * 100, preds

	def perform_training(self):
		preprocessor = PawPreprocessor(root_dir=data_root, train=True, n_folds=self.n_folds, model_dir=model_root)
		for fold in range(self.n_folds):

			train_loader = preprocessor.get_dataloader(
				fold=fold, for_validation=False, transform=get_albumentation_transform_for_training(self.img_size))
			val_loader = preprocessor.get_dataloader(
				fold=fold, for_validation=True, transform=get_albumentation_transform_for_validation(self.img_size))

			# model = PawClassifier(img_size, img_size, 3, len(preprocessor.features), embed_size, hidden_size)
			model = self.model_type(
				3, len(preprocessor.features), self.embed_size, self.hidden_size,
				pretrained=True, fine_tune=self.fine_tune)
			model.to(self.device)
			loss_func = nn.BCEWithLogitsLoss()
			optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
			# scheduler = OneCycleLR(
			#     optimizer,
			#     max_lr=max_lr,
			#     steps_per_epoch=int(len(train_dataset) / batch_size) + 1,
			#     epochs=epochs,
			# )
			scheduler = CosineAnnealingWarmRestarts(
				optimizer,
				T_0=100,
				eta_min=self.min_lr,
				last_epoch=-1
			)

			epochs_with_no_improvement = 0
			fine_tune_with_no_augmentation = False

			# Training and Validation Loop
			best_rmse = np.inf
			best_epoch = np.inf
			best_model_path = None
			for epoch in range(1, self.epochs + 1):

				if epochs_with_no_improvement >= self.patience:
					if fine_tune_with_no_augmentation:
						print(f'No improvement with no augmentation for {self.patience} epochs, early stop')
						break
					else:
						print(f'No improvement with AUGMENTATION for {self.patience} epochs, switch to NON-AUG training')
						epochs_with_no_improvement = 0
						fine_tune_with_no_augmentation = True
						train_loader = preprocessor.get_dataloader(
							fold=fold, for_validation=False,
							transform=get_albumentation_transform_for_validation(self.img_size))

				self.train_one_epoch(train_loader, model, loss_func, optimizer, epoch, scheduler)

				predictions, valid_targets = self.validate(val_loader, model, loss_func, epoch)
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
		return


if __name__ == '__main__':
	learning_params = dict(
		img_size=384,
		n_folds=10,
		batch_size=1,
		patience=3,
		epochs=99,
		embed_size=128,
		hidden_size=64,
		lr=1e-5,
		max_lr=1e-3,
		min_lr=1e-7,
		weight_decay=1e-6,
	)

	learner = Learner(
		data_root=data_root, model_root=model_root, model_type=PawSwinTransformerLarge4Patch12Win22k384,
		fine_tune=True, **learning_params
	)
	learner.perform_training()

