from pathlib import Path
import matplotlib.pyplot as plt
import PIL, cv2
import torch
import json
import numpy as np
import argparse
import torchvision.transforms as T
from scipy import ndimage

from models import tiramisu
from datasets import camvid
import utils.training as train_utils
from utils.imgs import view_image

WEIGHTS_PATH = Path('test/weights/')

classes = ['Void', 'Sidelobe', 'Source', 'Galaxy']

class NpEncoder(json.JSONEncoder):
    # JSON Encoder class to manage output file saving
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main(args):
    info = {'image_id': args.img_path.split('\\')[-1].split('.')[0]}
    normalize = T.Normalize(mean=camvid.mean, std=camvid.std)

    # TODO Add support for FITS images

    img = PIL.Image.open(args.img_path)
    transform=T.Compose([
            T.Resize([132, 132]),
            T.ToTensor(),
            normalize
        ])

    img = transform(img)

    model = tiramisu.FCDenseNet67(n_classes=4).cpu()
    model.eval()
    train_utils.load_weights(model, str(WEIGHTS_PATH)+'/latest.th', device="cpu")
    with torch.no_grad():
        pred = model(img.unsqueeze(0))

    info['objs'] = get_objs(img, pred)

    with open(args.out_path, 'w') as out:
        print(f'Dumping data in file {args.out_path}')
        json.dump(info, out, indent=2, cls=NpEncoder)

def get_objs(img, pred):
    preds = train_utils.get_predictions(pred)
    objs = []
    for pred in preds:
        for i, class_name in enumerate(classes):

            if class_name == "Void":
                # Skip background
                continue
            
            current_class = torch.where(pred == i, 1., 0.) # isolates the class of interest
            pred_objects, nr_pred_objects = ndimage.label(current_class)
            
            for pred_idx in range(nr_pred_objects):
                obj = {}
                current_obj_pred = torch.where(torch.from_numpy(pred_objects == (pred_idx + 1)), 1., 0.)
                obj['class_id'] = i
                obj['class_name'] = class_name

                pixels = (current_obj_pred == 1).nonzero().numpy()
                x_points = pixels[:,0]
                y_points = pixels[:,1]
                obj['pixels'] = pixels
                obj['bbox'] = [np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points)] #[x1,y1,x2,y2]
                obj['area'] = (np.max(x_points) - np.min(x_points)) * (np.max(y_points) - np.min(y_points))
                objs.append(obj)
                
    return objs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File listing all json files that contain mask information
    parser.add_argument("-i", "--img_path", help="Path of image file")

    parser.add_argument("-w", "--weights_path", default="test/weights/latest.th", help="Path of saved weights")

    # Optional argument flag which defaults to False
    parser.add_argument("-o", "--out_path", default="out.json", help="Destination path for extracted data in JSON format")

    args = parser.parse_args()

    main(args)