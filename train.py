"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import gc
import torch.optim
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR

from loader import *
from model import *
from utilities import *


def train():
    is_train = True

    train_loader, dataset = get_loader(
        root_folder=data_root,
        is_train=is_train,
        transform=train_transform,
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
            metric_monitor.update('RMSE', rmse_from_classifier_output(output, target))
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
            metric_monitor.update('RMSE', rmse_from_classifier_output(output, target))
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")

            targets = (target.detach().cpu().numpy() * 100).tolist()
            # WKNOTE: because we are using class for reg
            outputs = (torch.sigmoid(output).detach().cpu().numpy() * 100).tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets


def train_benchmark():
    seed_everything()
    gc.enable()
    img_size = 384
    n_folds = 5
    batch_size = 1
    epochs = 10
    embed_size = 128
    hidden_size = 64
    lr = 1e-5
    max_lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-6
    preprocessor = PawPreprocessor(root_dir=data_root, train=True, n_folds=n_folds)
    device = get_default_device()
    for fold in range(n_folds):

        train_img_paths, train_dense, train_targets = preprocessor.get_data(fold=fold, for_validation=False)
        valid_img_paths, valid_dense, valid_target = preprocessor.get_data(fold=fold, for_validation=True)
        train_dataset = PawDataset(
            images_filepaths=train_img_paths,
            dense_features=train_dense,
            targets=train_targets,
            transform=get_albumentation_transform_for_training(img_size)  # without augmentation, serious overfitting
        )

        valid_dataset = PawDataset(
            images_filepaths=valid_img_paths,
            dense_features=valid_dense,
            targets=valid_target,
            transform=get_albumentation_transform_for_validation(img_size)
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            # collate_fn=MyCollate(),
        )

        val_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            # collate_fn=MyCollate(),
        )

        # model = PawClassifier(img_size, img_size, 3, len(preprocessor.features), embed_size, hidden_size)
        model = PawSwinTransformerLarge4Patch12Win22k384(3, len(preprocessor.features), embed_size, hidden_size)
        model.to(device)
        loss_func = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=max_lr,
        #     steps_per_epoch=int(len(train_dataset) / batch_size) + 1,
        #     epochs=epochs,
        # )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            eta_min=min_lr,
            last_epoch=-1
        )

        # Training and Validation Loop
        best_rmse = np.inf
        best_epoch = np.inf
        best_model_path = None
        for epoch in range(1, epochs + 1):

            train_epoch(train_loader, model, loss_func, optimizer, epoch, scheduler)

            predictions, valid_targets = validate(val_loader, model, loss_func, epoch)
            rmse = round(mean_squared_error(valid_targets, predictions, squared=False), 5)
            if rmse <= best_rmse:
                best_rmse = rmse
                best_epoch = epoch
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = os.path.join(
                    model_root, f"{type(model).__name__}_epoch{epoch}_fold{fold + 1}_{rmse}_rmse.pth.tar")
                torch.save(model.state_dict(), best_model_path)

        print(f'The best RMSE: {best_rmse} for fold {fold + 1} was achieved on epoch: {best_epoch}.')
        print(f'The Best saved model is: {best_model_path}')
        print(''.join(['#'] * 50))
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train_benchmark()
