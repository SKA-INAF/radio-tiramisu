from pathlib import Path
import torch
from tqdm import tqdm
import argparse
from models import tiramisu
import utils.training as train_utils
from datasets.rg_masks import CLASSES, RGDataset
import datetime
from torchmetrics import JaccardIndex as IoU

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--resume", default='latest.th', type=str, help="Weights path from which start training")
    parser.add_argument( "--data_dir", default="data/rg-dataset/data", help="Path of data folder")
    parser.add_argument( "--results_dir", default =".results", help="Weights dir where to store results")
    parser.add_argument( "--log_file", default ="log.txt", help="Log text file path")
    parser.add_argument( "--batch_size", default=20)
    parser.add_argument( "--n_classes", default=4)
    parser.add_argument( "--device", default="cuda")

    return parser

def main(args):

    DATA_PATH = Path(args.data_dir)
    RESULTS_PATH = Path(args.results_dir) / datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    batch_size = args.batch_size

    test_dset = RGDataset(DATA_PATH, "data/rg-dataset/val_mask.txt")

    test_loader = torch.utils.data.DataLoader(
            test_dset, batch_size=batch_size, shuffle=False, num_workers=4)
    if args.device == 'cuda':
        torch.cuda.manual_seed(0)

    model = tiramisu.FCDenseNet67(n_classes=args.n_classes).to(args.device)

    train_utils.load_weights(model, args.resume)
    model.eval()

    iou_all = IoU(task="multiclass", num_classes=4).to(args.device)
    iou_ext = IoU(task="binary").to(args.device)
    iou_comp = IoU(task="binary").to(args.device)
    
    accs_all = []
    accs_ext = []
    accs_comp = []

    for batch in tqdm(test_loader, desc="Testing"):
        data, target = batch
        data = data.to(args.device)
        targets = target.to(args.device)
        with torch.no_grad():
            output = model(data)
        preds = train_utils.get_predictions(output)
        preds = preds.to(args.device)

        # Compute metrics for all classes
        iou_all.update(preds, targets)
        preds[preds == 0] = -1
        batch_acc = (preds == targets).sum() / (targets != 0).sum()
        accs_all.append(batch_acc)

        # Compute metrics for extended class
        preds_ext = preds.clone()
        targets_ext = targets.clone()
        preds_ext[preds_ext != 3] = 0
        targets_ext[targets_ext != 3] = 0
        preds_ext = preds_ext.clip(0, 1)
        targets_ext = targets_ext.clip(0, 1)

        iou_ext.update(preds_ext, targets_ext)
        preds_ext[preds_ext == 0] = -1
        if (targets_ext != 0).sum() != 0:
            batch_acc = (preds_ext == targets_ext).sum() / (targets_ext != 0).sum()
            accs_ext.append(batch_acc)

        # Compute metrics for compact class
        preds_comp = preds.clone()
        targets_comp = targets.clone()
        preds_comp[preds_comp != 2] = 0
        targets_comp[targets_comp != 2] = 0
        preds_comp = preds_comp.clip(0, 1)
        targets_comp = targets_comp.clip(0, 1)

        iou_comp.update(preds_comp, targets_comp)
        preds_comp[preds_comp == 0] = -1
        if (targets_comp != 0).sum() != 0:
            batch_acc = (preds_comp == targets_comp).sum() / (targets_comp != 0).sum()
            accs_comp.append(batch_acc)


    print("All classes")
    print(f'Accuracy: {sum(accs_all) / len(accs_all) * 100:.2f}')
    print(f'IoU: {iou_all.compute() * 100:.2f}')

    print("\nOnly Extended")
    print(f'Accuracy: {sum(accs_ext) / len(accs_ext) * 100:.2f}')
    print(f'IoU: {iou_ext.compute() * 100:.2f}')


    print("\nOnly Compact")
    print(f'Accuracy: {sum(accs_comp) / len(accs_comp) * 100:.2f}')
    print(f'IoU: {iou_comp.compute() * 100:.2f}')


if __name__ == '__main__':
    args = get_args().parse_args()
    args.resume = "weights/real-6600.pth"
    main(args)
