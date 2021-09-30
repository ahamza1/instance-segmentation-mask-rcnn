import os

import imgaug.augmenters as iaa
import numpy as np

import utils
from config import TrainConfig
from dataset import StreetsDataset
from mrcnn import model as model_lib


def main():
    # Get input arguments
    data_path, labels_path, weights_path = utils.get_args()

    # Load and prepare train and val dataset
    dataset_train = StreetsDataset()
    dataset_train.load_dataset(data_path, labels_path, "train")
    dataset_train.prepare()

    dataset_val = StreetsDataset()
    dataset_val.load_dataset(data_path, labels_path, "val")
    dataset_val.prepare()

    # Initialize the model for training
    config = TrainConfig()

    model_dir = os.path.join(os.path.abspath("/"), "out")
    model = model_lib.MaskRCNN(mode='training', config=config, model_dir=model_dir)

    weights = weights_path or model.find_last()

    # Exclude the last layers because they require a matching number of classes
    model.load_weights(weights, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"
    ])

    augmentation = iaa.Sometimes(0.4, [
        iaa.Fliplr(0.5),
        iaa.OneOf([
            iaa.Multiply((0.9, 1.1)),
            iaa.ContrastNormalization((0.9, 1.1)),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 0.1)),
            iaa.Sharpen(alpha=(0.0, 0.1)),
        ])
    ])

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads',
                augmentation=None)

    history = model.keras_model.history.history

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 2,
                epochs=6,
                layers='all',
                augmentation=None)

    new_history = model.keras_model.history.history
    for k in new_history: history[k] = history[k] + new_history[k]

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 4,
                epochs=10,
                layers='all',
                augmentation=augmentation)

    new_history = model.keras_model.history.history
    for k in new_history: history[k] = history[k] + new_history[k]

    best_epoch = np.argmin(history["val_loss"])
    score = history["val_loss"][best_epoch]
    print(f'Best Epoch:{best_epoch + 1} val_loss:{score}')


if __name__ == "__main__":
    main()
