import os
import sys
import datetime
import numpy as np
import json
import skimage.draw

import util
from mrcnn import visualize, utils, model as model_lib


ROOT_DIR = os.path.abspath("./")
DATASET_TRAIN_PATH = os.path.join(ROOT_DIR, "images\\train")
DATASET_VAL_PATH = os.path.join(ROOT_DIR, "images\\val")

ANNOTATIONS_TRAIN_PATH = os.path.join(ROOT_DIR, "images\\annotations\\train")
ANNOTATIONS_VAL_PATH = os.path.join(ROOT_DIR, "images\\annotations\\val")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

LABELS_PATH = os.path.join(ROOT_DIR, "labels.txt")
MODEL_DIR = os.path.join(ROOT_DIR, "out")


class StreetsDataset(utils.Dataset):
    def load_dataset(self, data_path, annotations_path, labels_path):
        labels = open(labels_path).read().strip().split("\n")

        for i in range(len(labels)):
            self.add_class("cityscapes", i+1, labels[i])

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

    # TODO: Adjust learning rate and weight loss
    # Train heads with higher lr to speedup the learning
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE * 2,
                epochs=2,
                layers='heads',
                augmentation=None)

    history = model.keras_model.history.history

    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    trained_model_path = os.path.join(MODEL_DIR, f"mrcnn_cs_{dt}.h5")
    model.keras_model.save_weights(trained_model_path)

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=4 if config.DEBUG else 14,
    #             layers='all')

    # new_history = model.keras_model.history.history
    # for k in new_history: history[k] = history[k] + new_history[k]
    #
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 2,
    #             epochs=6 if config.DEBUG else 22,
    #             layers='all')
    #
    # new_history = model.keras_model.history.history
    # for k in new_history: history[k] = history[k] + new_history[k]

    best_epoch = np.argmin(history["val_loss"])
    score = history["val_loss"][best_epoch]
    print(f'Best Epoch:{best_epoch + 1} val_loss:{score}')


if __name__ == "__main__":
    main()
