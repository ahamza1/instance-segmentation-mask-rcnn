import os
import skimage.io
import skimage.color

from mrcnn.config import Config
from mrcnn import utils


class TrainConfig(Config):
    NAME = "cityscapes"
    DEBUG = True
    CORRUPT_FILES = ""

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 11+1

    BACKBONE = 'resnet101'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    STEPS_PER_EPOCH = 512
    # VALIDATION_STEPS = 10

    TRAIN_ROIS_PER_IMAGE = 128
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_MAX_INSTANCES = 32

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_NMS_THRESHOLD = 0.7

    LEARNING_RATE = 0.002

    # LOSS_WEIGHTS = {
    #     "rpn_class_loss": 2.,
    #     "rpn_bbox_loss": 0.8,
    #     "mrcnn_class_loss": 1.,
    #     "mrcnn_bbox_loss": 1.0,
    #     "mrcnn_mask_loss": 1.8
    # }


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    mAP_IMAGES_LIMIT = 200


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
    image = skimage.io.imread(path, plugin="pil")
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
