# Tiramisu
Semantic segmentation with Tiramisu.

In this demo, we show an inference step on the Tiramisu model, pretrained on radioastronomical images, with the objective to perform semantic segmentation, by predicting both the mask and the category of the objects in each image. 

## Usage
To perform inference, run `inference.py`, by passing two arguments in the command line: 
- `-i` specifies which image will be fed as input to the model 
- `-o` determines the name of the folder where the output will be stored (default: "output")


  ```sh
  python inference.py -i sample_input/galaxy1.png -o output
  ```

Sample images can be found under the directory `sample_input`

The script will save two images for each run. One is the image given as input to the model, the other one is the same image with the predicted semantic masks overlayed.

The semantic mask is shown in three different colors, one for each predicted category with the following mapping:

- Red: _Source_
- Yellow: _Galaxy_
- Light Blue: _Sidelobe_
