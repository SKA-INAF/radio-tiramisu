from pathlib import Path
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import argparse
from torchsummary import summary
from models import tiramisu
import utils.training as train_utils
from datasets.rg_masks import CLASSES, RGDataset
import datetime
from torchmetrics import IoU, Accuracy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--resume", default='latest.th', type=str, help="Weights path from which start training")
    parser.add_argument( "--data_dir", default="data/rg-dataset/data", help="Path of data folder")
    # parser.add_argument( "--weights_dir", default ="weights", help="Weights dir where to save checkpoints")
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

    summary(model, input_size=(3,132,132))

    train_utils.load_weights(model, args.resume)
    model.eval()

    iou = IoU(num_classes=4).to(args.device)
    acc = 0

    for batch in tqdm(test_loader, desc="Testing"):
        data, target = batch
        data = data.to(args.device)
        targets = target.to(args.device)
        with torch.no_grad():
            output = model(data)
        preds = train_utils.get_predictions(output)
        preds = preds.to(args.device)
        iou.update(preds, targets)

        preds[preds == 0] = -1
        batch_acc = (preds == targets).sum() / (targets != 0).sum()
        acc += batch_acc


    print(f'IoU: {iou.compute()}')
    print(f'Accuracy: {acc / len(test_loader)}')
    
    train_utils.view_sample_predictions(model, test_loader, 1, 100, None)


if __name__ == '__main__':
    args = get_args().parse_args()
    args.resume = "weights/synthetic.pth"
    main(args)