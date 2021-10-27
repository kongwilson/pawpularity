"""
DESCRIPTION

Copyright (C) Weicong Kong, 27/10/2021
"""


from timm import create_model
from fastai.vision.all import set_seed, ImageDataLoaders
from fastai.vision.all import RegressionBlock, Resize, setup_aug_tfms, Brightness, Contrast, Hue, Saturation
from fastai.vision.all import F, Learner, BCEWithLogitsLossFlat

from utilities import *

set_seed(RANDOM_SEED, reproducible=True)

train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
train_df['path'] = train_df['Id'].map(lambda x: os.path.join('train', f'{x}.jpg'))
train_df = train_df.drop(columns=['Id'])
train_df = train_df.sample(frac=1).reset_index(drop=True)   # shuffle dataframe
train_df['norm_score'] = train_df['Pawpularity'] / 100

dls = ImageDataLoaders.from_df(
	train_df,  # pass in train DataFrame
	valid_pct=0.2,  # 80-20 train-validation random split
	seed=RANDOM_SEED,  # seed
	path=data_root,
	fn_col='path',  # filename/path is in the second column of the DataFrame
	label_col='norm_score',  # label is in the first column of the DataFrame
	y_block=RegressionBlock,  # The type of target
	bs=32,  # pass in batch size
	num_workers=8,
	item_tfms=Resize(224),  # pass in item_tfms
	batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()]))  # pass in batch_tfms


model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=dls.c)


def petfinder_rmse(input, target):
	return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))


learn = Learner(dls, model, loss_func=BCEWithLogitsLossFlat(), metrics=petfinder_rmse).to_fp16()
learn.lr_find(end_lr=3e-2)
