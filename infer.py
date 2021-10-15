"""
To use the model trained on test dataset

Copyright (C) Weicong Kong, 16/10/2021
"""
import glob

from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import PawPreprocessor, PawDataset
from model import *
from utilities import *


def infer(model_type, img_size=384, batch_size=4):

	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(root_dir=data_root, train=False)
	test_img_paths, test_dense, test_targets = preprocessor.get_data()

	test_dataset = PawDataset(
		images_filepaths=test_img_paths,
		dense_features=test_dense,
		targets=test_targets,
		transform=get_albumentation_transform_for_training(img_size)
	)

	all_models_checkpoints = glob.glob(model_root + os.path.sep + f'{model_type.__name__}_*.pth.tar')
	preds = None
	for model_path in all_models_checkpoints:

		model = model_type()
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)
		model.eval()

		test_loader = DataLoader(
			test_dataset, batch_size=batch_size,
			shuffle=False, num_workers=0,
			pin_memory=True
		)

		temp_preds = None
		with torch.no_grad():
			for (images, dense, target) in tqdm(test_loader, desc=f'Predicting. '):
				images = images.to(device, non_blocking=True)
				dense = dense.to(device, non_blocking=True)
				predictions = torch.sigmoid(model(images, dense)).cpu().detach().numpy() * 100

				if temp_preds is None:
					temp_preds = predictions
				else:
					temp_preds = np.vstack((temp_preds, predictions))

		if preds is None:
			preds = temp_preds
		else:
			preds += temp_preds

	preds /= (len(all_models_checkpoints))
	return preds


if __name__ == '__main__':
	preds = infer(PawVisionTransformerLarge32Patch384)
	print(preds)
