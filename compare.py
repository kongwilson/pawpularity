"""
DESCRIPTION

Copyright (C) Weicong Kong, 10/10/2021
"""
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from loader import *
from model import PawpularityNN
from utils import *

import random


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


if __name__ == '__main__':

	seed_everything(RANDOM_SEED)

	train_loader, dataset = get_loader(
		root_folder=data_root,
		is_train=True,
		transform=train_transform,
		num_workers=1,
	)

	test_loader, test_dataset = get_loader(
		root_folder=data_root,
		is_train=False,
		transform=train_transform,
		num_workers=1,
		batch_size=32
	)

	embed_size = 256
	hidden_size = 256
	num_layers = 2
	learning_rate = 3e-4
	num_epochs = 100

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = PawpularityNN(embed_size, hidden_size, num_layers).to(device)

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	step = load_checkpoint(torch.load(os.path.join(data_root, 'my_checkpoint.pth.tar')), model, optimizer)

	model.eval()  #
	torch.no_grad()
	preds = []
	for idx, (imgs, y) in enumerate(test_loader):
		imgs = imgs.to(device)

		outputs = model(imgs)
		preds.append(outputs)

	preds = torch.cat(preds).cpu().detach().numpy()

	train_preds = []
	for imgs, y in tqdm(train_loader):

		imgs = imgs.to(device)

		outputs = model(imgs)
		train_preds.append(outputs)

	train_path = os.path.join(data_root, 'train.csv')
	train = pd.read_csv(train_path)

	# eda test data
	test_path = os.path.join(data_root, 'test.csv')
	test = pd.read_csv(test_path)

	# training for just using the meta data
	features = [c for c in test if c != 'Id']
	y_col = 'Pawpularity'
	train_y = train[y_col].values

	train_preds = torch.cat(train_preds).detach().cpu().numpy()
	print('inceptionNet mse on training data: ', mean_squared_error(train_preds, train_y))
