import json
import os
import sys

import numpy as np
import skimage.color
import skimage.draw
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN utils
sys.path.append(ROOT_DIR)  # To find local version of the library

from model.mrcnn import utils


class StreetsDataset(utils.Dataset):

    # Initialize dataset; load class names and annotations
    def load_dataset(self, data_path, labels_path, subset):
        assert subset in ["train", "val", "test"]

        dataset_path = os.path.join(data_path, "dataset", subset)
        annotations_path = os.path.join(data_path, "annotations", subset)

        labels = open(labels_path).read().strip().split("\n")

        self.class_info = []
        for i in range(len(labels)):
            self.add_class("cityscapes", i, labels[i])

        for file in get_files(annotations_path, file_extension=".json"):
            image_name, image_annotations = file

            image_path = os.path.join(dataset_path, image_name.split("_")[0], image_name)
            annotations = json.load(open(image_annotations))
            self.filter_objects(annotations)

            self.add_image(
                "cityscapes",
                image_id=image_name,
                path=image_path,
                width=annotations["imgWidth"],
                height=annotations["imgHeight"],
                objects=annotations["objects"]
            )

    # Convert polygons to a bitmap mask of shape
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info["objects"]
        count = len(annotations)

        if count == 0:
            mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
            class_ids = list()
            for i, p in enumerate(annotations):
                all_points_x, all_points_y = map(list, zip(*p["polygon"]))
                rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
                mask[rr - 1, cc - 1, i] = 1
                class_ids.append(next(x for x in self.class_info if x["name"] == p["label"])["id"])

        return mask.astype(np.bool), np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def filter_objects(self, annotations):
        label_names = [x["name"] for x in self.class_info]
        objects = annotations["objects"]
        annotations["objects"] = [x for x in objects if x["label"] in label_names]


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
    return f"{(file.rsplit('.', 1)[0]).replace('_gtFine_polygons', '_leftImg8bit')}.png"
