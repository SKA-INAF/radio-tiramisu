from pathlib import Path
import matplotlib.pyplot as plt
import PIL, cv2
import torch
import json
import numpy as np
from PIL import Image
import argparse
import torchvision.transforms as T
from scipy import ndimage
from utils import imgs as img_utils
from models import tiramisu
from torchvision.transforms.functional import to_pil_image
import utils.training as train_utils
from utils.imgs import view_image
import torchvision.transforms.functional as TF

Background = [0,0,0]
Galaxy = [237,237,12] # Yellow
Sidelobe = [32,207,227] # Light Blue
Source = [250,7,7] # Red

label_colours = np.array([Background, Sidelobe, Source, Galaxy])

classes = ['Background', 'Sidelobe', 'Source', 'Galaxy']

def inv_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return TF.normalize(
        x,
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

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

def get_mask(pred, alpha=255):
    temp = pred.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    
    for l in range(0,4):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgba = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
    rgba[:,:,0] = r
    rgba[:,:,1] = g
    rgba[:,:,2] = b
    
    # img =  to_pil_image(rgba)
    # img.putalpha(alpha)

    return rgba

def main(args):
    info = {'image_id': args.img_path.split('\\')[-1].split('.')[0]}
    mean = [0.28535324, 0.28535324, 0.28535324]
    std = [0.28536762, 0.28536762, 0.28536762]
    normalize = T.Normalize(mean=mean, std=std)

    orig_img = PIL.Image.open(args.img_path)
    transform=T.Compose([
            T.Resize([132, 132]),
            T.ToTensor(),
            normalize
        ])

    img = transform(orig_img)

    model = tiramisu.FCDenseNet67(n_classes=4).cpu()
    model.eval()
    train_utils.load_weights(model, args.weights_dir + '/latest.th', device="cpu")
    with torch.no_grad():
        out = model(img.unsqueeze(0))

    # info['objs'] = get_objs(img, out)

    pred = train_utils.get_predictions(out)
    annotated = get_mask(pred[0], 127)
    orig_arr = np.array(orig_img)
    mask = annotated != 0
    orig_arr[mask] = annotated[mask]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    img_name = Path(args.img_path).stem
    orig_img.save(out_dir / f'{img_name}_input.png')
    Image.fromarray(orig_arr).save(out_dir / f'{img_name}_pred.png')
    print(f'Input image and predictions saved under the {out_dir} folder')

    # with open(args.out_path, 'w') as out:
    #     print(f'Dumping data in file {args.out_path}')
    #     json.dump(info, out, indent=2, cls=NpEncoder)

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
    parser.add_argument("-i", "--img_path", required=True, help="Path of image file")

    parser.add_argument("-w", "--weights_dir", default="weights", help="Path of saved weights")

    # Optional argument flag which defaults to False
    parser.add_argument("-o", "--out_dir", default="output", help="Destination path for extracted data in JSON format")

    args = parser.parse_args()

    main(args)