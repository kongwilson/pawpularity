"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# specifying the root of the data location
# data_root = os.path.join('/kaggle', 'input', 'petfinder-pawpularity-score')
import torch
from torchvision.transforms import transforms

data_root = r'C:\Users\Myadmin\data\petfinder-pawpularity-score'
train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')


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


transform = transforms.Compose(
	[
		transforms.Resize((356, 356)),
		transforms.RandomCrop((299, 299)),  # we are doing some sort of data augmentation here
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	]
)

# WKNOTE: Data augmentation, some of them may hurt the model. It's suggested no roration should be made:
#   https://www.kaggle.com/weicongkong/tf-petfinder-vit-cls-tpu-train/edit
transform2 = transforms.Compose(
	[
		transforms.ToPILImage(),  # WKNOTE: transforms only works with PIL images
		transforms.Resize((256, 256)),
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
				"{metric_name}: {avg:.{float_precision}f}".format(
					metric_name=metric_name, avg=metric["avg"],
					float_precision=self.float_precision
				)
				for (metric_name, metric) in self.metrics.items()
			]
		)
