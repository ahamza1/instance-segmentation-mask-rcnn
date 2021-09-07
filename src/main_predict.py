import os

import matplotlib
import skimage.io
import skimage.color

from config import InferenceConfig
from main_utils import get_args_inference
from mrcnn import model as model_lib
from mrcnn import utils, visualize

matplotlib.use('TkAgg')


def main():
    # Get input arguments
    image_path, labels_path, weights_path = get_args_inference()
    labels = open(labels_path).read().strip().split("\n")

    # Initialize config & labels
    config = InferenceConfig()
    model_dir = os.path.join(os.path.abspath("./"), "out")

    # Initialize the Mask R-CNN model for inference and then load the weights
    model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model.load_weights(weights_path, by_name=True)

    # Load image and run detection
    image = skimage.io.imread(image_path, plugin="pil")

    result = model.detect([image], verbose=1)

    r = result[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], labels, r['scores'])


if __name__ == "__main__":
    main()
