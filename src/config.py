from mrcnn.config import Config


class TrainConfig(Config):
    NAME = "cityscapes"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 11 + 1

    BACKBONE = 'resnet101'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    STEPS_PER_EPOCH = 512

    TRAIN_ROIS_PER_IMAGE = 128

    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_MAX_INSTANCES = 32

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_NMS_THRESHOLD = 0.7

    LEARNING_RATE = 0.002


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
