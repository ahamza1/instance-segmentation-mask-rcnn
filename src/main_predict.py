import os
import matplotlib
import util

from mrcnn import model as model_lib
from mrcnn import visualize

# python main_predict.py -w C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\out\mrcnn_cs_2021_07_30_11_48.h5 -l C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\labels.txt -i C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\images\IMG_0696.jpeg
# python main_predict.py -w C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\out\mrcnn_cs_2021_07_30_11_48.h5 -l C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\labels.txt -i C:\Users\Almir\Desktop\master\instance-segmentation-mask-rcnn\src\images\IMG_0696.jpeg


def main():
    root_dir = os.path.abspath("./")
    model_dir = os.path.join(root_dir, "out")

    # Parse input arguments
    weights_path, labels_path, image_path = util.get_input_arguments()

    # Initialize config & labels
    config = util.InferenceConfig()
    labels = open(labels_path).read().strip().split("\n")

    # Initialize the Mask R-CNN model for inference and then load the weights
    model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
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
