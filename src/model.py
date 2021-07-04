import os
import skimage.io
import skimage.color
import matplotlib

from mrcnn.config import Config
from mrcnn import model as model_lib
from mrcnn import visualize, utils


class MaskRCNN(object):
    # Override the training configurations with a few changes for inference config
    class InferenceConfig(Config):
        # Give the configuration a recognizable name
        NAME = "coco"

        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 80  # COCO has 80 classes

    def __init__(self, weights_path, labels_path):

        # Initialize paths
        self.root_dir = os.path.abspath("./")
        self.model_dir = os.path.join(self.root_dir, "out")

        # Initialize the configuration
        self.config = self.InferenceConfig()

        # Initialize the Mask R-CNN model for inference and then load the weights
        self.model = model_lib.MaskRCNN(mode="inference", config=self.config, model_dir=self.model_dir)

        self.model.load_weights(weights_path, by_name=True)
        self.labels = open(labels_path).read().strip().split("\n")

        print('Model initialized')

    def detect(self, image):
        image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            min_scale=self.config.IMAGE_MIN_SCALE,
            max_dim=self.config.IMAGE_MAX_DIM,
            mode=self.config.IMAGE_RESIZE_MODE
        )

        # Run the model
        results = self.model.detect([image], verbose=1)
        return image, results[0]

    def visualize(self, image, result):
        matplotlib.use('TkAgg')
        visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'],
                                    self.labels, result['scores'])

    def load_image(self, path):
        # Load image
        image = skimage.io.imread(path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
