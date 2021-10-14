"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from loader import get_loader, PawImageDatasetPreloaded, MyCollate
from model import PawpularityNN, PawBenchmark
from utils import *


def train():
    is_train = True

    train_loader, dataset = get_loader(
        root_folder=data_root,
        is_train=is_train,
        transform=transform,
        num_workers=1,
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


def get_default_device():
    # pick GPU if available, else CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train_epoch(train_loader, model, loss_func, optimizer, epoch, scheduler=None):

    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)

    device = get_default_device()

    with torch.enable_grad():

        for i, (images, dense, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)
            dense = dense.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float().view(-1, 1)

            output = model(images, dense)

            loss = loss_func(output, target)

            metric_monitor.update('Loss', loss.item())
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()
            stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

    return


def validate(val_loader, model, loss_func, epoch):

    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    device = get_default_device()

    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, (images, dense, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)
            dense = dense.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float().view(-1, 1)
            output = model(images, dense)
            loss = loss_func(output, target)
            metric_monitor.update('Loss', loss.item())
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

            targets = (target.detach().cpu().numpy() * 100).tolist()
            outputs = (torch.sigmoid(output).detach().cpu().numpy() * 100).tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets


def train_benchmark():
    n_folds = 5
    batch_size = 16
    epochs = 20
    dataset = PawImageDatasetPreloaded(root_dir=data_root, train=True, transform=transform2, num_folds=n_folds)
    device = get_default_device()
    for fold in range(n_folds):
        dataset.set_fold_to_use(fold, validation=False)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            # collate_fn=MyCollate(),
        )

        embed_size = 64
        hidden_size = 64

        model = PawBenchmark(256, 256, 3, len(dataset.features), embed_size, hidden_size=hidden_size)
        model.to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-4,
            steps_per_epoch=int((n_folds - 1) * len(dataset) / (n_folds * batch_size)) + 1,
            epochs=epochs,
        )

        # Training and Validation Loop
        best_rmse = np.inf
        best_epoch = np.inf
        best_model_path = None
        for epoch in range(1, epochs + 1):
            train_epoch(train_loader, model, loss_func, optimizer, epoch, scheduler)

            dataset.set_fold_to_use(fold, validation=True)
            val_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=False,
                # collate_fn=MyCollate(),
            )
            predictions, valid_targets = validate(val_loader, model, loss_func, epoch)
            rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 3)
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(data_root, f"epoch{epoch}_fold{fold + 1}_{rmse}_rmse.pth.tar")
                torch.save(model.state_dict(), best_model_path)


# if __name__ == '__main__':
#     train_benchmark()
