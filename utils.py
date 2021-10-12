"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import os

import numpy as np
import pandas as pd

# specifying the root of the data location
# data_root = os.path.join('/kaggle', 'input', 'petfinder-pawpularity-score')
import torch
from torchvision.transforms import transforms

data_root = r'C:\Users\Myadmin\data\petfinder-pawpularity-score'


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
