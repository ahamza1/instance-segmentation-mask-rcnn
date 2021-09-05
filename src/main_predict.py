import os
import matplotlib
import util

from mrcnn import model as model_lib
from mrcnn import visualize

# python main_predict.py -w C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\out\mrcnn_cs_2021_08_11_10_54.h5 -l C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\labels.txt -i C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\images\test1.png
# python main_predict.py -w  -l C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\labels.txt -i C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\images\test1.png


ROOT_DIR = os.path.abspath("./")
MODEL_DIR = os.path.join(ROOT_DIR, "out")


def return_paths():
    weights_path = os.path.join(ROOT_DIR, "out\\cityscapes20210905T1126\\mask_rcnn_cityscapes_0002.h5")
    labels_path = os.path.join(ROOT_DIR, "labels.txt")
    image_path = os.path.join(ROOT_DIR, "images\\test1.png")
    return weights_path, labels_path, image_path


def main():
    # Parse input arguments
    weights_path, labels_path, image_path = return_paths()

    # Initialize config & labels
    config = util.InferenceConfig()
    labels = open(labels_path).read().strip().split("\n")

    # Initialize the Mask R-CNN model for inference and then load the weights
    model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model.load_weights(weights_path, by_name=True)

    # Load image and run detection
    image = util.load_image(image_path, config)

    # Run the model
    results = model.detect([image], verbose=1)
    r = results[0]

    matplotlib.use('TkAgg')
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], labels, r['scores'])


if __name__ == "__main__":
    main()
