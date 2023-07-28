import streamlit as st
from PIL import Image
import numpy as np
from datasets.rg_masks import get_transforms
import utils.training as train_utils
from models import tiramisu
from torchvision.transforms.functional import to_pil_image
import torch

# Function to apply color overlay to the input image based on the segmentation mask
def apply_color_overlay(input_image, segmentation_mask, color=[0, 255, 0], alpha=0.5):
    r = (segmentation_mask == 1).float()
    g = (segmentation_mask == 2).float()
    b = (segmentation_mask == 3).float()
    overlay = torch.cat([r, g, b], dim=0)
    overlay = to_pil_image(overlay)
    output = Image.blend(input_image, overlay, alpha=0.5)
    return output

# Streamlit app
def main():
    st.title("Tiramisu for semantic segmentation of radio astronomy images")
    st.write("Upload an image and see the segmentation result!")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    model = tiramisu.FCDenseNet67(n_classes=4).to("cuda")
    train_utils.load_weights(model, "weights/latest.th")
    model.eval()

    st.markdown(
        """
        Category Legend:
        - :blue[Extended]
        - :green[Compact]
        - :red[Spurious]
        """
        )
    if uploaded_image is not None:
        # Load the uploaded image
        input_image = Image.open(uploaded_image)
        input_array = np.array(input_image)
        input_array = input_array.transpose(2,0,1)
        input_array = np.expand_dims(input_array[0], 0)
        print(input_array.shape)
        transforms = get_transforms(input_image.size[0])
        image = transforms(input_array)
        image = image.to("cuda")

        with torch.no_grad():
            output = model(image)
        preds = output.argmax(1)

        # Apply color overlay to the input image
        segmented_image = apply_color_overlay(input_image, preds)

        # Display the input image and the segmented output
        st.image([input_image, segmented_image], caption=["Input Image", "Segmented Output"], width=300)

if __name__ == "__main__":
    main()