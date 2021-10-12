"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from loader import get_loader
from model import PawpularityNN
from utils import *


def train():
    is_train = True

    train_loader, dataset = get_loader(
        root_folder=data_root,
        is_train=is_train,
        transform=transform,
        num_workers=2,
    )

    # Start training
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    # writer = SummaryWriter(input_root)
    step = 0

    # initialise model, loss etc
    model = PawpularityNN(embed_size, hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = F.mse_loss

    if load_model:
        step = load_checkpoint(torch.load(os.path.join(data_root, 'my_checkpoint.pth.tar')), model, optimizer)

    model.train()

    for epoch in tqdm(range(num_epochs)):

        if save_model:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, y) in enumerate(train_loader):

            imgs = imgs.to(device)
            y = y.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, y)

            # writer.add_scalar('Training loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == '__main__':
    train()