"""
To use the model trained on test dataset

Copyright (C) Weicong Kong, 16/10/2021
"""
from learner import *
from utilities import *
from loader import *
import glob

activation = {}


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.cpu().detach().numpy()

	return hook


def get_all_model_checkpoints(model):

	all_models_checkpoints = glob.glob(model_root + os.path.sep + f'{str(model)}_*.pth.tar')
	return all_models_checkpoints


def extract_intermediate_outputs_and_targets(model, data_loader):
	device = get_default_device()
	new_x = None
	y = None
	preds = None

	model.eval()

	with torch.no_grad():
		for (images, dense, target) in tqdm(data_loader, desc=f'Training with XGB. '):
			images = images.to(device, non_blocking=True)
			dense = dense.to(device, non_blocking=True)
			batch_preds = torch.sigmoid(model(images, dense)).cpu().detach().numpy() * 100
			batch_embed = activation['swin_head']
			batch_x = np.concatenate([batch_embed, dense.cpu().detach().numpy()], axis=1)
			if preds is None:
				preds = batch_preds
				new_x = batch_x
				y = target.view(-1, 1).cpu().detach().numpy()
			else:
				preds = np.vstack((preds, batch_preds))
				new_x = np.vstack((new_x, batch_x))
				y = np.vstack((y, target.view(-1, 1).cpu().detach().numpy()))

	return new_x, y * 100, preds


def infer(model_type, img_size=384, batch_size=4, embed_size=128, hidden_size=64, fine_tune=False):

	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(root_dir=data_root, train=False)
	test_loader = preprocessor.get_dataloader()
	model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=False, fine_tune=fine_tune)

	all_models_checkpoints = get_all_model_checkpoints(model)
	preds = None
	for model_path in all_models_checkpoints:

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=False, fine_tune=fine_tune)
		# WKNOTE: map_location can specify the device to load your model, if your model is trained on other GPU, th
		model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
		model = model.to(device)
		model.eval()

		temp_preds = None
		with torch.no_grad():
			for (images, dense, target) in tqdm(test_loader, desc=f'Predicting. '):
				images = images.to(device, non_blocking=True)
				dense = dense.to(device, non_blocking=True)
				predictions = torch.sigmoid(model(images, dense)).detach().cpu().numpy() * 100

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


def infer_with_xgb(model_type, img_size=384, batch_size=4, embed_size=128, hidden_size=64, xgb_only=False):

	seed_everything()
	device = get_default_device()
	preprocessor = PawPreprocessor(root_dir=data_root, train=False)
	test_loader = preprocessor.get_dataloader()

	model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=False)
	all_models_checkpoints = get_all_model_checkpoints(model)
	preds = None
	for model_path in all_models_checkpoints:

		model_name = os.path.basename(model_path)
		xgb_file = [p for p in os.listdir(model_root) if model_name in p and '.json' in p][0]
		xgb_path = os.path.join(model_root, xgb_file)

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=False)
		model.load_state_dict(torch.load(model_path))
		model.model.head.register_forward_hook(get_activation('swin_head'))
		model = model.to(device)

		xgb_test_x, target, dl_preds = extract_intermediate_outputs_and_targets(model, test_loader)
		xgb_model = xgb.XGBRegressor()
		xgb_model.load_model(xgb_path)
		xgb_preds = xgb_model.predict(xgb_test_x).reshape((-1, 1))
		xgb_preds = prediction_validity_check(xgb_preds)

		if xgb_only:
			temp_preds = xgb_preds
		else:
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

	test_loader = preprocessor.get_dataloader(fold=fold, for_validation=True)
	model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=False)

	all_models_checkpoints = get_all_model_checkpoints(model)
	preds = None
	for model_path in all_models_checkpoints:

		model = model_type(3, len(preprocessor.features), embed_size, hidden_size, pretrained=False)
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)
		model.eval()

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
	# preds1 = infer_with_xgb(PawSwinTransformerLarge4Patch12Win384)
	# preds2 = infer_with_xgb(PawSwinTransformerLarge4Patch12Win22k384)
	#
	# preds = (preds1 + preds2) / 2
	# print(preds)
	preds = infer(PawSwinTransformerLarge4Patch12Win22k384, fine_tune=True)
