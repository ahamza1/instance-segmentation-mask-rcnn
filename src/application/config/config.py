import os


class DevelopmentConfig:
    RESOURCES_DIR = os.path.abspath("./resources")

    MODEL_DIR_PATH = os.path.abspath("../model/resources")
    LABELS_PATH = os.path.join(MODEL_DIR_PATH, "labels.txt")
    WEIGHTS_PATH = os.path.join(MODEL_DIR_PATH, "mask_rcnn_cityscapes.h5")
    ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")


app_config = {
    'dev': DevelopmentConfig
}
