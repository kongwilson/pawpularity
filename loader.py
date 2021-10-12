"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F


# build image loader
class PawImageDataset(Dataset):

	def __init__(self, root_dir, is_train=True, transform=None):
		self.root_dir = root_dir
		if is_train:
			path = os.path.join(root_dir, 'train.csv')
			self.df = pd.read_csv(path)
			self.image_dir = os.path.join(self.root_dir, 'train')
		else:
			path = os.path.join(root_dir, 'test.csv')
			self.df = pd.read_csv(path)
			self.image_dir = os.path.join(self.root_dir, 'test')

		self.transform = transform

		# Get img, caption columns
		self.imgs = self.df["Id"] + '.jpg'  # image id column
		if 'Pawpularity' in self.df:
			self.pawpularity = self.df['Pawpularity']
		else:
			temp = self.df.copy()
			temp['Blur'] = 0
			self.pawpularity = temp['Blur'].copy()

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):  # to get a single example
		img_id = self.imgs[index]
		img = Image.open(os.path.join(self.image_dir, img_id)).convert("RGB")  # using the PIL library

		if self.transform is not None:
			img = self.transform(img)

		paw = self.pawpularity[index]
		return img, torch.tensor(paw)


class MyCollate:

	def __init__(self):
		pass

	def __call__(self, batch):  # get some batch (list) of
		imgs = [item[0].unsqueeze(0) for item in batch]
		imgs = torch.cat(imgs, dim=0)
		targets = [float(item[1]) for item in batch]

		return imgs, torch.tensor(targets)


def get_loader(
		root_folder,
		is_train,
		transform,
		batch_size=32,
		num_workers=8,
		shuffle=True,  # WKNOTE: you would not want to do this if you are working with timeseries!!!
		pin_memory=True,
):
	"""We want to have some data loaders doing the loading, that can be sent into the model"""
	dataset = PawImageDataset(root_folder, is_train, transform=transform)

	loader = DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		num_workers=num_workers,
		shuffle=shuffle,
		pin_memory=pin_memory,
		collate_fn=MyCollate(),
	)

	return loader, dataset
