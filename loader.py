"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import os

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from kaggle.pawpularity.utils import *


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


class PawImageDatasetPreloaded(Dataset):

	def __init__(self, root_dir, train=True, transform=None, image_size=356, validation=False, num_folds=5):

		self.root_dir = root_dir
		self.train = train
		self.validation = validation
		self.num_folds = num_folds
		self.transform = transform
		self.use_fold = None

		if self.train:
			path = os.path.join(root_dir, 'train.csv')
			self.df = pd.read_csv(path)
			self.image_dir = os.path.join(self.root_dir, 'train')
		else:
			path = os.path.join(root_dir, 'test.csv')
			self.df = pd.read_csv(path)
			self.image_dir = os.path.join(self.root_dir, 'test')

		# prepare all images in the memory (pre-load)
		self.df['image_path'] = self.df['Id'].apply(lambda x: return_filpath(x, folder=self.image_dir))
		images = []
		for idx, img_path in enumerate(tqdm(self.df['image_path'])):
			if os.path.exists(img_path):
				img = cv2.imread(img_path)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				img = cv2.resize(img, (image_size, image_size))
				images.append(img)
			else:
				self.df.drop(idx)

		# set the preloaded images
		self.images = np.array(images)

		# prepare the corresponding tabular features
		self.df.reset_index(drop=True, inplace=True)

		# set the number of folds
		if train:
			skf = StratifiedKFold(self.num_folds)
			self.kfolds = list(skf.split(self.df.index, self.df['Pawpularity'].values))
		else:
			self.kfolds = None

		not_features = ['Id', 'kfold', 'image_path', 'Pawpularity']
		self.features = [feat for feat in self.df.columns if feat not in not_features]

		# prepare the target (label)
		if 'Pawpularity' in self.df:  # for train
			self.pawpularity = self.df['Pawpularity'].values / 100  # normalise the target value
		else:  # for test
			temp = self.df.copy()
			temp['Blur'] = 0
			self.pawpularity = temp['Blur'].copy().values

	def __len__(self):
		if self.train:
			if self.use_fold is None:
				return len(self.df)
			else:
				if self.validation:
					return len(self.kfolds[self.use_fold][1])
				else:
					return len(self.kfolds[self.use_fold][0])
		else:
			return len(self.df)

	def __getitem__(self, index):

		if self.use_fold is None:
			pick = range(len(self))[index]
		else:
			if self.validation:
				pick = self.kfolds[self.use_fold][1][index]
			else:
				pick = self.kfolds[self.use_fold][0][index]

		img = self.images[pick]
		if self.transform is not None:
			img = self.transform(img)

		dense = torch.tensor(self.df.iloc[pick][self.features].astype(int).values)
		paw = torch.tensor(self.pawpularity[pick]).float()
		return img, dense, paw

	def set_fold_to_use(self, fold: int, validation: bool = False) -> None:
		if not self.train:
			print('k folds are unavailable for test')
		else:
			n_folds = len(self.kfolds)
			self.use_fold = fold
			if fold < 0:
				self.use_fold = 0
				print(f'given fold is {fold}, use fold #0 instead')
			if fold > n_folds:
				self.use_fold = n_folds - 1
				print(
					f'given fold is {fold}, larger than the number of folds available, '
					f'use fold #{self.use_fold} instead')
			self.validation = validation
		return

	def reset_not_to_use_fold(self):
		self.use_fold = None
		self.validation = False
		return


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


if __name__ == '__main__':

	test_dataset = PawImageDatasetPreloaded(root_dir=data_root, train=False, transform=None)
	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=32,
		num_workers=1,
		shuffle=False,
		pin_memory=True  # he data loader will copy Tensors into CUDA pinned memory before returning them
	)

