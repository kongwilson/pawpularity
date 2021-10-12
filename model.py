"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):

	def __init__(self, embed_size, train_cnn=False):
		super(EncoderCNN, self).__init__()
		self.train_cnn = train_cnn  # false,
		# we will use a pre-trained model
		self.inception = models.inception_v3(pretrained=True, aux_logits=False)
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
