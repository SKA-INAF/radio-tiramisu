from pathlib import Path
from utils.logging import Logger
import torch
import argparse 
import torch.nn as nn
import numpy as np
import time
from models import tiramisu
import utils.training as train_utils
import wandb
from torch.utils.tensorboard import SummaryWriter
from datasets.rg_masks import SyntheticRGDataset, CLASSES, RGDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--resume", default='', type=str, help="Weights path from which start training")
    parser.add_argument( "--data_dir", default="data/rg-dataset/data", help="Path of data folder")
    parser.add_argument( "--weights_dir", default =".weights", help="Weights dir where to save checkpoints")
    parser.add_argument( "--results_dir", default =".results", help="Weights dir where to store results")
    parser.add_argument( "--log_file", default ="log.txt", help="Log text file path")
    parser.add_argument( "--batch_size", type=int, default=20)
    parser.add_argument( "--lr", type=float, default=1e-4)
    parser.add_argument( "--lr_decay", type=float, default=0.995)
    parser.add_argument( "--weight_decay", type=float, default=1e-4)
    parser.add_argument( "--decay_every_n_epochs", type=int, default=1)
    parser.add_argument( "--n_epochs", type=int, default=300)
    parser.add_argument( "--n_classes", type=int, default=4)
    parser.add_argument( "--device", default="cuda")
    parser.add_argument( "--test", action="store_true")

    return parser

# ## Train

def main(args):
    DATA_PATH = Path(args.data_dir)
    RESULTS_PATH = Path(args.results_dir)
    WEIGHTS_PATH = Path(args.weights_dir)
    RESULTS_PATH.mkdir(exist_ok=True)
    WEIGHTS_PATH.mkdir(exist_ok=True)
    batch_size = args.batch_size
    class_weight = torch.FloatTensor([0.25, 2.85, 0.30, 1.50])

    # Data Loading
    # train_dset = SyntheticRGDataset(DATA_PATH, "data/rg-dataset/synthetic_train.txt")
    # test_dset = SyntheticRGDataset(DATA_PATH, "data/rg-dataset/synthetic_test.txt")
    train_dset = RGDataset(DATA_PATH, "data/rg-dataset/train_mask.txt")
    test_dset = RGDataset(DATA_PATH, "data/rg-dataset/val_mask.txt")

    train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=batch_size, shuffle=False)
    
    # Logging

    writer = SummaryWriter()

    logger = Logger(args.log_file, CLASSES, writer)
    n_samples = {
        'Train' : len(train_loader.dataset),
        'Val' : len(val_loader.dataset),
        # 'Test' : len(test_loader.dataset),
        }

    inputs, targets = next(iter(train_loader))
    logger.log_header(n_samples, inputs.shape, targets.shape)

    torch.cuda.manual_seed(0)

    model = tiramisu.FCDenseNet67(n_classes=4).to(args.device)
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss(weight=class_weight).to(args.device)


    for epoch in range(1, args.n_epochs + 1):
        since = time.time()
        
        ### Train ###
        trn_loss, trn_metrics = train_utils.train(model, train_loader, optimizer, criterion, epoch)

        time_elapsed = time.time() - since
        logger.log_metrics('Train', epoch, trn_loss, trn_metrics, time_elapsed)
        logger.wandb_plot_metrics(trn_metrics, 'train')

        
        ### Val ###
        since = time.time()
        val_loss, val_metrics = train_utils.test(model, val_loader, criterion, epoch)

        time_elapsed = time.time() - since
        logger.log_metrics('Val', epoch, val_loss, val_metrics, time_elapsed)
        logger.wandb_plot_metrics(val_metrics, 'val')

        train_utils.view_sample_predictions(model, val_loader, epoch, n=5, writer=writer)
        
        ## Checkpoint ###    
        val_acc = np.nanmean([val_metrics[class_name]['accuracy'] for class_name in CLASSES])
        if (epoch + 1) % 100 == 0:
            train_utils.save_weights(model, epoch, val_loss, val_acc)

        ### Adjust Lr ###
        train_utils.adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch, args.decay_every_n_epochs)

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)