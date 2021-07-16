from pathlib import Path
import torch
import argparse 
import torch.nn as nn
import numpy as np
import time
import torchvision.transforms as transforms
from models import tiramisu
from datasets.rg_data import AstroDataLoaders, LabelTensorToPILImage, LabelToLongTensor
from datasets import joint_transforms
import utils.training as train_utils
import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--resume", default='', type=str, help="Weights path from which start training")
    parser.add_argument( "--data_dir", default="data", help="Path of data folder")
    parser.add_argument( "--weights_dir", default =".weights", help="Weights dir where to save checkpoints")
    parser.add_argument( "--results_dir", default =".results", help="Weights dir where to store results")
    parser.add_argument( "--batch_size", default=20)
    parser.add_argument( "--lr", default=1e-4)
    parser.add_argument( "--lr_decay", default=0.995)
    parser.add_argument( "--weight_decay", default=1e-4)
    parser.add_argument( "--decay_every_n_epochs", default=1)
    parser.add_argument( "--n_epochs", default=1000)
    parser.add_argument( "--n_classes", default=4)

    return parser

# ## Train

# LR = 1e-4
# LR_DECAY = 0.995
# DECAY_EVERY_N_EPOCHS = 1
# N_EPOCHS = 1000

'''


'''


def main(args):
    DATA_PATH = Path(args.data_dir)
    RESULTS_PATH = Path(args.results_dir)
    WEIGHTS_PATH = Path(args.weights_dir)
    RESULTS_PATH.mkdir(exist_ok=True)
    WEIGHTS_PATH.mkdir(exist_ok=True)
    batch_size = 1

    # Data Loading
    data_loader = AstroDataLoaders(DATA_PATH, batch_size)
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()

    print("Train: %d" %len(train_loader.dataset.imgs))
    print("Val: %d" %len(val_loader.dataset.imgs))
    print("Test: %d" %len(test_loader.dataset.imgs))
    print("Classes: %d" % len(train_loader.dataset.classes))

    inputs, targets = next(iter(train_loader))
    print("Inputs: ", inputs.size())
    print("Targets: ", targets.size())

    #utils.imgs.view_image(inputs[0])
    #utils.imgs.view_annotated(targets[0])

    #train_utils.save_mask(train_loader)

    torch.cuda.manual_seed(0)


    model = tiramisu.FCDenseNet67(n_classes=4).cuda()
    model.apply(train_utils.weights_init)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss(weight=data_loader.class_weight.cuda()).cuda()

    # Logging

    # Init WandB logger 
    wandb.init(project='astro_tiramisu')
    config = wandb.config
    config.learning_rate = args.lr

    writer = SummaryWriter()
    output_file = open("log.txt", 'a')
    output_file.write('Date: ' + str(datetime.datetime.now()) + '\n')

    for epoch in range(1, args.n_epochs + 1):
        since = time.time()
        
        ### Train ###
        trn_loss, trn_metrics = train_utils.train(model, train_loader, optimizer, criterion, epoch)    
        writer.add_scalar("Loss/train", trn_loss, epoch)

        for class_name in train_loader.dataset.classes:
            for metric_name in trn_metrics[class_name]:
                writer.add_scalar(metric_name + "/train_" + class_name, trn_metrics[class_name][metric_name], epoch)

        print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss))
        print('Per class metrics: ')
        
        for i, class_name in enumerate(train_loader.dataset.classes):
            print('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}'.format(class_name, trn_metrics[class_name]['accuracy'], trn_metrics[class_name]['iou'], trn_metrics[class_name]['recall'], trn_metrics[class_name]['specificity'], trn_metrics[class_name]['precision'], trn_metrics[class_name]['dice'], trn_metrics[class_name]['obj_precision'], trn_metrics[class_name]['obj_recall']))
        output_file.write('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss)+ '\n')
        output_file.write('Per class metrics: \n')

        for i, class_name in enumerate(train_loader.dataset.classes):
            output_file.write('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}\n'.format(class_name, trn_metrics[class_name]['accuracy'], trn_metrics[class_name]['iou'], trn_metrics[class_name]['recall'], trn_metrics[class_name]['specificity'], trn_metrics[class_name]['precision'], trn_metrics[class_name]['dice'], trn_metrics[class_name]['obj_precision'], trn_metrics[class_name]['obj_recall']))
        time_elapsed = time.time() - since  
        print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        output_file.write('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+ '\n')
        
        ### Val ###
        val_loss, test_metrics = train_utils.test(model, val_loader, criterion, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        for class_name in test_loader.dataset.classes:
            for metric_name in trn_metrics[class_name]:
                writer.add_scalar(metric_name + "/val_" + class_name, trn_metrics[class_name][metric_name], epoch)

        print('Val - Loss: {:.4f}'.format(val_loss))
        print('Per class metrics: ')
        for i, class_name in enumerate(test_loader.dataset.classes):
            print('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}'.format(class_name, test_metrics[class_name]['accuracy'], test_metrics[class_name]['iou'], test_metrics[class_name]['recall'], test_metrics[class_name]['specificity'], test_metrics[class_name]['precision'], test_metrics[class_name]['dice'], test_metrics[class_name]['obj_precision'], test_metrics[class_name]['obj_recall']))
        
        output_file.write('Val - Loss: {:.4f}'.format(val_loss)+ '\n')
        output_file.write('Per class metrics: \n')
        for i, class_name in enumerate(val_loader.dataset.classes):
            output_file.write('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}\n'.format(class_name, test_metrics[class_name]['accuracy'], test_metrics[class_name]['iou'], test_metrics[class_name]['recall'], test_metrics[class_name]['specificity'], test_metrics[class_name]['precision'], test_metrics[class_name]['dice'], test_metrics[class_name]['obj_precision'], test_metrics[class_name]['obj_recall']))
        time_elapsed = time.time() - since  
        print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        output_file.write('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)+ '\n')
        train_utils.view_sample_predictions(model, val_loader, epoch, n=5, writer=writer)
        
        ### Checkpoint ###    
        val_acc = np.nanmean([test_metrics[class_name]['accuracy'] for class_name in val_loader.dataset.classes])
        train_utils.save_weights(model, epoch, val_loss, val_acc)

        ### Adjust Lr ###
        train_utils.adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch, args.decay_every_n_epochs)

    # ## Test

    if args.resume:
        train_utils.load_weights(model, str(WEIGHTS_PATH)+args.resume)
    test_loss, test_metrics = train_utils.test(model, test_loader, criterion, epoch=1)  
    writer.add_scalar("Loss/test", test_loss, 1)

    for class_name in test_loader.dataset.classes:
        for metric_name in test_metrics[class_name]:
            wandb.log({f'test_{class_name}/{metric_name}' : test_metrics[class_name][metric_name]})
            writer.add_scalar(metric_name + "/test_" + class_name, test_metrics[class_name][metric_name], 1)

    train_utils.view_sample_predictions(model, test_loader, 1, 100, None)
    output_file.close()

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)