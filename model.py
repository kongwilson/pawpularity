"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import timm
import torch.nn as nn
import torchvision.models as models
from utilities import *


class EncoderCNN(nn.Module):

	def __init__(self, embed_size, train_cnn=False):
		super(EncoderCNN, self).__init__()
		self.train_cnn = train_cnn  # false,
		# we will use a pre-trained model, if `pretained` is True, it will download the pretrained weights if there are
		#   not in some paths, if you don't want to download the pretained weight but just load from your saved weights,
		#   this pretrained arg should be set to False
		self.inception = models.inception_v3(pretrained=False, aux_logits=False)
		# access the last linear layer, and replace it with another linear layer with output as the `embed_size`
		self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

	def forward(self, images):
		features = self.inception(images)

		# because we are not going to train the entire network, we are not going to compute the gradients
		# we are just fine-tuning weights, so we set the last layer to be `required_grad`, while the other layers don't
		#   need the gradients
		for name, param in self.inception.named_parameters():
			if 'fc.weight' in name or 'fc.bias' in name:
				param.requires_grad = True
			else:
				param.requires_grad = self.train_cnn  #

		return self.dropout(self.relu(features))


class PawpularityNN(nn.Module):

	def __init__(self, embed_size, hidden_size, num_layers):
		super(PawpularityNN, self).__init__()

		self.embed = EncoderCNN(embed_size)
		self.linear = nn.Linear(embed_size, hidden_size, num_layers)
		self.output = nn.Linear(hidden_size, 1)
		self.dropout = nn.Dropout(0.5)

	def forward(self, images):
		embeddings = self.embed(images)
		hiddens = self.linear(embeddings)
		outputs = self.output(self.dropout(hiddens))
		return outputs


class PawBenchmark(nn.Module):

	def __init__(
			self, in_width, in_height, in_chan, dense_feature_size, embed_size, hidden_size, output_size=1,
			kernel_size=3, stride=1, dilation=1, dropout=0.5):
		super(PawBenchmark, self).__init__()

		self.conv = nn.Sequential(
			# group 1
			nn.Conv2d(
				in_channels=in_chan, out_channels=embed_size, kernel_size=(kernel_size, kernel_size),
				stride=(stride, stride), padding='same', dilation=(dilation, dilation)
			),
			nn.ReLU(),
			nn.Dropout2d(dropout),
			nn.Conv2d(
				in_channels=embed_size, out_channels=embed_size, kernel_size=(kernel_size, kernel_size),
				stride=(stride, stride), padding='same', dilation=(dilation, dilation)
			),
			nn.ReLU(),
			nn.Dropout2d(dropout),
			nn.MaxPool2d(kernel_size=2),

			# group 2
			nn.Conv2d(
				in_channels=embed_size, out_channels=embed_size, kernel_size=(kernel_size, kernel_size),
				stride=(stride, stride), padding='same', dilation=(dilation, dilation)
			),
			nn.ReLU(),
			nn.Dropout2d(dropout),
			nn.Conv2d(
				in_channels=embed_size, out_channels=embed_size, kernel_size=(kernel_size, kernel_size),
				stride=(stride, stride), padding='same', dilation=(dilation, dilation)
			),
			nn.ReLU(),
			nn.Dropout2d(dropout),
			nn.MaxPool2d(kernel_size=2),

			nn.Flatten(start_dim=1, end_dim=-1)
		)

		new_width = int(int(in_width / 2) / 2)
		new_height = int(int(in_height / 2) / 2)

		n_features = new_width * new_height * embed_size

		self.fc = nn.Sequential(
			nn.Linear(n_features + dense_feature_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, output_size)
		)

	def forward(self, image, dense):
		img_embeddings = self.conv(image)
		x = torch.cat([img_embeddings, dense], dim=1)
		output = self.fc(x)
		return output


class PawClassifier(PawBenchmark):

	def __init__(
			self, *args, **kwargs):
		super(PawClassifier, self).__init__(*args, **kwargs)


class PawVisionTransformerTiny16Patch384(nn.Module):

	def __init__(
			self, in_chan, dense_feature_size, embed_size, hidden_size, output_size=1, dropout=0.2,
			pretrained=True, fine_tune=False):

		super().__init__()
		self.pretrained = pretrained
		self.fine_tune = fine_tune
		if pretrained is False:
			self.fine_tune = False  # we can't fine tune the model if the backbone is not pretrained!!
		self.dense_feature_size = dense_feature_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.dropout_rate = dropout
		self.model = self._get_pretrained_model(in_chan)
		self._customise_the_final_layer()
		self.fc = nn.Sequential(
			nn.Linear(embed_size + dense_feature_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, output_size)
		)
		self.dropout = nn.Dropout(dropout)

	def __str__(self):
		fine_tune_flag = 'fine-tuned' if self.fine_tune else 'retrained'
		name_parts = [
			type(self).__name__,
			f'embed-{self.embed_size}',
			f'hidden-{self.hidden_size}',
			f'drop-{self.dropout_rate}',
			fine_tune_flag
		]
		return '_'.join(name_parts)

	def _get_pretrained_model(self, in_chan):
		model = timm.models.vit_tiny_patch16_384(pretrained=self.pretrained, in_chans=in_chan)
		return model

	def _customise_the_final_layer(self):
		n_features = self.model.head.in_features
		self.model.head = nn.Linear(n_features, self.embed_size)
		return

	def forward(self, image, dense):

		self._set_gradient()

		embeddings = self.model(image)
		x = self.dropout(embeddings)
		x = torch.cat([x, dense], dim=1)
		output = self.fc(x)
		return output

	def _set_gradient(self):
		if self.fine_tune:
			for name, param in self.model.named_parameters():
				if 'head.weight' in name or 'head.bias' in name:
					param.requires_grad = True
				else:
					param.requires_grad = False
		return


class PawVisionTransformerLarge32Patch384(PawVisionTransformerTiny16Patch384):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

	def _get_pretrained_model(self, in_chan):
		return timm.models.vit_large_patch32_384(pretrained=self.pretrained, in_chans=in_chan)


class PawSwinTransformerLarge4Patch12Win22k384(PawVisionTransformerTiny16Patch384):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

	def _get_pretrained_model(self, in_chan):
		return timm.models.swin_large_patch4_window12_384_in22k(pretrained=self.pretrained, in_chans=in_chan)


class PawSwinTransformerLarge4Patch7Win22k224(PawVisionTransformerTiny16Patch384):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

	def _get_pretrained_model(self, in_chan):
		return timm.models.swin_large_patch4_window7_224_in22k(
			pretrained=self.pretrained, in_chans=in_chan, num_classes=1)

	def _customise_the_final_layer(self):
		pass


class PawSwinTransformerLarge4Patch12Win384(PawVisionTransformerTiny16Patch384):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

	def _get_pretrained_model(self, in_chan):
		return timm.models.swin_large_patch4_window12_384(pretrained=self.pretrained, in_chans=in_chan)


class PawSwinEfficientNetB6(PawVisionTransformerTiny16Patch384):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

	def _get_pretrained_model(self, in_chan):
		# WKNOTE: efficientNet uses a lot of batch normalisation layers - multiple publications have shown BN
		#   performance degrade for batch_size under 32, and severely for <=8. In a nutshell, batch statistics
		#   "averaged" over a single sample vary greatly sample-to-sample (high variance)
		return timm.models.efficientnet_b6(pretrained=self.pretrained, in_chans=in_chan)

	def _customise_the_final_layer(self):
		n_features = self.model.classifier.in_features
		self.model.classifier = nn.Linear(n_features, self.embed_size)
		return

	def _set_gradient(self):
		if self.fine_tune:
			for name, param in self.model.named_parameters():
				if 'classifier.weight' in name or 'classifier.bias' in name:
					param.requires_grad = True
				else:
					param.requires_grad = False
		return


class PawNFNetF3(PawVisionTransformerTiny16Patch384):

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)

	def _get_pretrained_model(self, in_chan):
		return timm.models.nfnet_f3(pretrained=self.pretrained, in_chans=in_chan)

	def _customise_the_final_layer(self):
		n_features = self.model.head.fc.in_features
		self.model.head.fc = nn.Linear(n_features, self.embed_size)
		return

	def _set_gradient(self):
		if self.fine_tune:
			for name, param in self.model.named_parameters():
				if 'head.fc.weight' in name or 'head.fc.bias' in name:
					param.requires_grad = True
				else:
					param.requires_grad = False
		return

