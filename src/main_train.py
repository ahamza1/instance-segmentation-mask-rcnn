import os
import sys
import datetime
import numpy as np
import json
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import util
import imgaug
from mrcnn import visualize, utils, model as model_lib


ROOT_DIR = os.path.abspath("./")
DATASET_TRAIN_PATH = os.path.join(ROOT_DIR, "images\\train")
DATASET_VAL_PATH = os.path.join(ROOT_DIR, "images\\val")

ANNOTATIONS_TRAIN_PATH = os.path.join(ROOT_DIR, "images\\annotations\\train")
ANNOTATIONS_VAL_PATH = os.path.join(ROOT_DIR, "images\\annotations\\val")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "out\\mrcnn_cs_2021_08_11_10_09.h5")

LABELS_PATH = os.path.join(ROOT_DIR, "labels.txt")
MODEL_DIR = os.path.join(ROOT_DIR, "out")


class StreetsDataset(utils.Dataset):
    def load_dataset(self, data_path, annotations_path, labels_path):
        labels = open(labels_path).read().strip().split("\n")
        self.class_info = []

        for i in range(len(labels)):
            self.add_class("cityscapes", i, labels[i])

        for file in util.get_files(annotations_path, file_extension=".json"):
            image_name = file[0]
            image_annotations = file[1]
            image_path = os.path.join(data_path, image_name.split("_")[0], image_name)
            annotations = json.load(open(image_annotations))

            self.add_image(
                "cityscapes",
                image_id=image_name,
                path=image_path,
                width=annotations["imgWidth"],
                height=annotations["imgHeight"],
                objects=annotations["objects"]
            )

    def load_mask(self, image_id):
        # Convert polygons to a bitmap mask of shape
        info = self.image_info[image_id]
        annotations = info["objects"]
        count = len(annotations)

        if count == 0:
            mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
            class_ids = list()
            try:
                for i, p in enumerate(annotations):
                    all_points_x, all_points_y = map(list, zip(*p["polygon"]))
                    rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
                    mask[rr-1, cc-1, i] = 1
                    class_ids.append(next(x for x in self.class_info if x["name"] == p["label"])["id"])
            except Exception as e:
                print(e)

        return mask.astype(np.bool), np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


def main():
    # Load and prepare train and val dataset
    dataset_train = StreetsDataset()
    dataset_train.load_dataset(DATASET_TRAIN_PATH, ANNOTATIONS_TRAIN_PATH, LABELS_PATH)
    dataset_train.prepare()

    dataset_val = StreetsDataset()
    dataset_val.load_dataset(DATASET_VAL_PATH, ANNOTATIONS_VAL_PATH, LABELS_PATH)
    dataset_val.prepare()

    # Initialize the model for training
    config = util.TrainConfig()
    model = model_lib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"
    ])

    try:
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE*2,
                    epochs=2,
                    layers='heads',
                    augmentation=None)

        history = model.keras_model.history.history

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=3,
                    layers='all')

        new_history = model.keras_model.history.history
        for k in new_history: history[k] = history[k] + new_history[k]

        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 2,
                    epochs=4,
                    layers='all')

        new_history = model.keras_model.history.history
        for k in new_history: history[k] = history[k] + new_history[k]

        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE/2,
        #             epochs=8,
        #             layers='all')
        #
        # new_history = model.keras_model.history.history
        # for k in new_history: history[k] = history[k] + new_history[k]

        save_trained_model(model)
        plt_results(history)

        best_epoch = np.argmin(history["val_loss"])
        score = history["val_loss"][best_epoch]
        print(f'Best Epoch:{best_epoch + 1} val_loss:{score}')

    except Exception:
        save_trained_model(model)


def save_trained_model(model):
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    trained_model_path = os.path.join(MODEL_DIR, f"mrcnn_cs_{dt}.h5")
    model.keras_model.save_weights(trained_model_path)


def plt_results(history):
    epochs = range(1, len(history['loss']) + 1)
    pd.DataFrame(history, index=epochs)

    plt.figure(figsize=(21, 11))

    plt.subplot(231)
    plt.plot(epochs, history["loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(232)
    plt.plot(epochs, history["rpn_class_loss"], label="Train RPN class ce")
    plt.plot(epochs, history["val_rpn_class_loss"], label="Valid RPN class ce")
    plt.legend()
    plt.subplot(233)
    plt.plot(epochs, history["rpn_bbox_loss"], label="Train RPN box loss")
    plt.plot(epochs, history["val_rpn_bbox_loss"], label="Valid RPN box loss")
    plt.legend()
    plt.subplot(234)
    plt.plot(epochs, history["mrcnn_class_loss"], label="Train MRCNN class ce")
    plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid MRCNN class ce")
    plt.legend()
    plt.subplot(235)
    plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train MRCNN box loss")
    plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid MRCNN box loss")
    plt.legend()
    plt.subplot(236)
    plt.plot(epochs, history["mrcnn_mask_loss"], label="Train Mask loss")
    plt.plot(epochs, history["val_mrcnn_mask_loss"], label="Valid Mask loss")
    plt.legend()

    matplotlib.use('TkAgg')
    plt.show()


if __name__ == "__main__":
    main()
