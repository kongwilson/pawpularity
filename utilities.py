"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import os
import random
from collections import defaultdict

import albumentations
import numpy as np
import pandas as pd

import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import mean_squared_error
from torchvision.transforms import transforms

# specifying the root of the data location
# data_root = os.path.join('/kaggle', 'input', 'petfinder-pawpularity-score')
# data_root = r'C:\Users\Myadmin\data\petfinder-pawpularity-score'
data_root = r'C:\Users\wkong\IdeaProjects\kaggle_data\petfinder-pawpularity-score'
model_root = data_root
train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')

# Random Seed Initialize
RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


def build_submission(test_data: pd.DataFrame, pred: np.ndarray, output_folder=None, output_suffix=None):
	sub = pd.DataFrame()
	sub['Id'] = test_data['Id']
	sub['Pawpularity'] = pred
	if output_suffix is None:
		output_name = 'submission.csv'
	else:
		output_name = f'submission_{output_suffix}.csv'
	if output_folder is None:
		sub.to_csv(output_name, index=False)
	else:
		sub.to_csv(os.path.join(output_folder, output_name))
	return sub


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
	print('=> Saving checkpoint')
	torch.save(state, os.path.join(data_root, filename))
	return


def load_checkpoint(checkpoint, model, optimizer):  # , steps):
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer"])


train_transform = transforms.Compose(
	[
		transforms.Resize((384, 384)),
		# transforms.RandomCrop((299, 299)),  # we are doing some sort of data augmentation here
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	]
)

# WKNOTE: Data augmentation, some of them may hurt the model. It's suggested no roration should be made:
#   https://www.kaggle.com/weicongkong/tf-petfinder-vit-cls-tpu-train/edit
val_transform = transforms.Compose(
	[
		transforms.ToPILImage(),  # WKNOTE: transforms only works with PIL images
		transforms.Resize((384, 384)),
		transforms.RandomHorizontalFlip(p=0.5),  # WKNOTE: randomly flip the image with the given prob
		transforms.RandomVerticalFlip(p=0.5),
		transforms.RandomAutocontrast(p=0.3),
		transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
		# transforms.RandomRotation(degrees=180),  # WKNOTE: rotate between -180 to +180
		# transforms.RandomCrop((256, 256)),  # we are doing some sort of data augmentation here
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	]
)


def get_albumentation_transform_for_training(img_size):
	aug_transform = albumentations.Compose(
		[
			albumentations.Resize(img_size, img_size),
			albumentations.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
			albumentations.HorizontalFlip(p=0.5),
			albumentations.VerticalFlip(p=0.5),
			albumentations.Rotate(limit=180, p=0.7),
			albumentations.ShiftScaleRotate(
				shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5
			),
			albumentations.HueSaturationValue(
				hue_shift_limit=0.2, sat_shift_limit=0.2,
				val_shift_limit=0.2, p=0.5
			),
			albumentations.RandomBrightnessContrast(
				brightness_limit=(-0.1, 0.1),
				contrast_limit=(-0.1, 0.1), p=0.5
			),
			ToTensorV2(p=1.0),
		]
	)
	return aug_transform


def get_albumentation_transform_for_validation(img_size):
	aug_transform_val = albumentations.Compose(
		[
			albumentations.Resize(img_size, img_size),
			albumentations.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
			ToTensorV2(p=1.0)
		]
	)
	return aug_transform_val


def return_filpath(name, folder=train_dir):
	path = os.path.join(folder, f'{name}.jpg')
	return path


def get_data(for_test=False):
	if for_test:
		data = pd.read_csv(os.path.join(data_root, 'test.csv'))
		folder = test_dir
	else:
		data = pd.read_csv(os.path.join(data_root, 'train.csv'))
		folder = train_dir

	data['image_path'] = data['Id'].apply(lambda x: return_filpath(x, folder=folder))
	return data


class MetricMonitor:

	def __init__(self, float_precision=3):
		self.float_precision = float_precision
		self.reset()

	def reset(self):
		self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

	def update(self, metric_name, val):
		metric = self.metrics[metric_name]

		metric["val"] += val
		metric["count"] += 1
		metric["avg"] = metric["val"] / metric["count"]

	def __str__(self):
		return " | ".join(
			[
				f"{metric_name}: {metric['avg']:.{self.float_precision}f}"
				for (metric_name, metric) in self.metrics.items()
			]
		)


def rmse_from_classifier_output(output: torch.Tensor, target: torch.Tensor):
	y_pred = torch.sigmoid(output).cpu().detach().numpy() * 100  # WK: move the tensor from gpu to cpu, then detach
	target = target.cpu() * 100

	return mean_squared_error(target, y_pred, squared=False)


def get_default_device():
	# pick GPU if available, else CPU
	if torch.cuda.is_available():
		return torch.device('cuda')
	else:
		return torch.device('cpu')
