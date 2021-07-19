
# coding: utf-8

# ## Dependencies

from datasets.rg_data import AstroDataLoaders
from pathlib import Path
from utils.logging import Logger
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter

import argparse
from models import tiramisu
from datasets import rg_data
import utils.training as train_utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--resume", default='latest.pth', type=str, help="Weights path from which start training")
    parser.add_argument( "--data_dir", default="data_reduced", help="Path of data folder")
    parser.add_argument( "--weights_dir", default =".weights", help="Weights dir where to save checkpoints")
    parser.add_argument( "--results_dir", default =".results", help="Weights dir where to store results")
    parser.add_argument( "--log_file", default ="log.txt", help="Log text file path")
    parser.add_argument( "--batch_size", default=20)
    parser.add_argument( "--n_epochs", default=1000)
    parser.add_argument( "--n_classes", default=4)
    parser.add_argument( "--device", default="cuda")

    return parser

def main(args):

    DATA_PATH = Path(args.data_dir)
    RESULTS_PATH = Path(args.results_dir)
    WEIGHTS_PATH = Path(args.weights_dir)
    RESULTS_PATH.mkdir(exist_ok=True)
    WEIGHTS_PATH.mkdir(exist_ok=True)
    batch_size = args.batch_size


    data_loader = AstroDataLoaders(DATA_PATH, batch_size)
    test_loader = data_loader.get_test_loader()
    if args.device == 'cuda':
        torch.cuda.manual_seed(0)

    writer = SummaryWriter()
    logger = Logger(args.log_file, test_loader.dataset.classes, writer)

    model = tiramisu.FCDenseNet67(n_classes=args.n_classes).to(args.device)
    criterion = nn.NLLLoss(weight=data_loader.class_weight.cuda()).cuda()


    train_utils.load_weights(model, str(WEIGHTS_PATH)+'/' + args.resume)
    since = time.time()
    test_loss, test_metrics = train_utils.test(model, test_loader, criterion, epoch=1)  

    time_elapsed = time.time() - since
    logger.log_metrics('Test', 1, test_loss, test_metrics, time_elapsed)
    logger.wandb_plot_metrics(test_metrics, 'test')
    
    train_utils.view_sample_predictions(model, test_loader, 1, 100, None)


if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)