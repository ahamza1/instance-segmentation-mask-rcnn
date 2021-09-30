import os
import random

import matplotlib

from config import InferenceConfig
from dataset import StreetsDataset
from utils import get_args
from mrcnn import model as model_lib
from mrcnn import visualize

matplotlib.use('TkAgg')


def main():
    # Get input arguments
    data_path, labels_path, weights_path = get_args()
    labels = open(labels_path).read().strip().split("\n")

    dataset = StreetsDataset()
    dataset.load_dataset(data_path, labels_path, "test")
    dataset.prepare()

    # Initialize config & labels
    config = InferenceConfig()
    model_dir = os.path.join(os.path.abspath("./"), "out")

    # Initialize the Mask R-CNN model for inference and then load the weights
    model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model.load_weights(weights_path, by_name=True)

    # Load image and run detection
    image = dataset.load_image(random.choice(dataset.image_ids))

    r = model.detect([image], verbose=1)[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], labels, r['scores'])


if __name__ == "__main__":
    main()
