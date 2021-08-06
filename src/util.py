import os
import skimage.io
import skimage.color

from mrcnn.config import Config
from mrcnn import utils


class InferenceConfig(Config):
    NAME = "coco_cityscapes"

    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    BACKBONE = 'resnet101'

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes
    NUM_CLASSES = 112 + 1


class TrainConfig(Config):
    NAME = "coco_cityscapes"
    DEBUG = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 112 + 1

    BACKBONE = 'resnet101'

    # IMAGE_MIN_DIM = 384
    # IMAGE_MAX_DIM = 384

    LEARNING_RATE = 0.003

    TRAIN_ROIS_PER_IMAGE = 50
    MAX_GT_INSTANCES = 150
    # DETECTION_MIN_CONFIDENCE = 0.95
    # DETECTION_NMS_THRESHOLD = 0.0

    STEPS_PER_EPOCH = 50
    # VALIDATION_STEPS = 125

    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 30.0,
    #     "rpn_bbox_loss": 0.8,
    #     "mrcnn_class_loss": 6.0,
    #     "mrcnn_bbox_loss": 1.0,
    #     "mrcnn_mask_loss": 1.2
    # }


def get_input_arguments():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("-l", "--labels", required=True)
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    return args["weights"], args["labels"], args["image"]


def get_files(directory, file_extension=None):
    annotations = []
    images = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            if file_extension is None or filename.endswith(file_extension):
                filepath = os.path.join(root, filename)
                annotations.append(filepath)

                images.append(image_name_from_annotation(filename))

    return zip(images, annotations)


def image_name_from_annotation(file):
    return (file.rsplit('.', 1)[0]).replace("_gtFine_polygons", "_leftImg8bit") + ".png"


def load_image(path, config):
    # Load image
    image = skimage.io.imread(path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    image, _, _, _, _ = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE
    )

    return image
