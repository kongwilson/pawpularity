"""
To use the model trained on test dataset

Copyright (C) Weicong Kong, 16/10/2021
"""

from train import *


def infer(model_type, img_size=384, batch_size=4, embed_size=128, hidden_size=64):

	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(root_dir=data_root, train=False)
	test_img_paths, test_dense, test_targets = preprocessor.get_data()

	test_dataset = PawDataset(
		images_filepaths=test_img_paths,
		dense_features=test_dense,
		targets=test_targets,
		transform=get_albumentation_transform_for_validation(img_size)
	)

	all_models_checkpoints = glob.glob(model_root + os.path.sep + f'{model_type.__name__}_*.pth.tar')
	preds = None
	for model_path in all_models_checkpoints:

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size)
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


def infer_with_xgb(model_type, img_size=384, batch_size=4, embed_size=128, hidden_size=64):

	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(root_dir=data_root, train=False)
	test_img_paths, test_dense, test_targets = preprocessor.get_data()

	test_dataset = PawDataset(
		images_filepaths=test_img_paths,
		dense_features=test_dense,
		targets=test_targets,
		transform=get_albumentation_transform_for_validation(img_size)
	)

	all_models_checkpoints = glob.glob(model_root + os.path.sep + f'{model_type.__name__}_*.pth.tar')
	preds = None
	for model_path in all_models_checkpoints:

		model_name = os.path.basename(model_path)
		xgb_path = [p for p in os.listdir(model_root) if model_name in p and '.json' in p][0]

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size)
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)
		model.eval()

		test_loader = DataLoader(
			test_dataset, batch_size=batch_size,
			shuffle=False, num_workers=0,
			pin_memory=True
		)

		xgb_test_x, target, dl_preds = extra_intermediate_outputs_and_targets(model, test_loader)
		xgb_model = xgb.XGBRegressor()
		xgb_model.load_model(xgb_path)
		xgb_preds = xgb_model.predict(xgb_test_x)

		temp_preds = (xgb_preds + dl_preds) / 2

		if preds is None:
			preds = temp_preds
		else:
			preds += temp_preds

	preds /= (len(all_models_checkpoints))
	return preds


def infer_out_of_fold(model_type, img_size=384, batch_size=4, embed_size=128, hidden_size=64):
	fold = 0
	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(root_dir=data_root, train=True, model_dir=model_root)
	valid_img_paths, valid_dense, valid_targets = preprocessor.get_data(fold=fold, for_validation=True)

	test_dataset = PawDataset(
		images_filepaths=valid_img_paths,
		dense_features=valid_dense,
		targets=valid_targets,
		transform=get_albumentation_transform_for_validation(img_size)
	)

	all_models_checkpoints = glob.glob(model_root + os.path.sep + f'{model_type.__name__}_*.pth.tar')
	preds = None
	for model_path in all_models_checkpoints:

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size)
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

	valid_targets *= 100
	rmse = round(mean_squared_error(valid_targets, preds, squared=False), 5)
	print(f'Fold {fold} RMSE: {rmse}')

	return preds


if __name__ == '__main__':
	# preds = infer(PawVisionTransformerLarge32Patch384)
	# print(preds)
	# preds = infer_out_of_fold(PawVisionTransformerLarge32Patch384)
	preds1 = infer_with_xgb(PawSwinTransformerLarge4Patch12Win384)
	preds2 = infer_with_xgb(PawSwinTransformerLarge4Patch12Win22k384)

	preds = (preds1 + preds2) / 2
	print(preds)
