import json
import os

import numpy as np
import skimage.color
import skimage.draw
import skimage.io

from mrcnn import utils

exclude = [
    "jena_000117_000019", "dusseldorf_000075_000019", "munster_000005_000019", "munster_000133_000019",
    "lindau_000007_000019", "krefeld_000000_036299", "dusseldorf_000150_000019", "bremen_000092_000019",
    "bremen_000166_000019", "bremen_000081_000019", "bremen_000132_000019", "bremen_000133_000019",
    "bremen_000177_000019", "weimar_000013_000019", "dusseldorf_000157_000019", "dusseldorf_000098_000019",
    "dusseldorf_000177_000019", "bremen_000165_000019", "jena_000059_000019", "dusseldorf_000061_000019",
    "bremen_000130_000019", "weimar_000114_000019", "zurich_000002_000019", "bremen_000124_000019",
    "dusseldorf_000196_000019", "zurich_000051_000019", "weimar_000006_000019", "munster_000045_000019",
    "bremen_000105_000019", "lindau_000020_000019", "bremen_000158_000019", "dusseldorf_000057_000019",
    "zurich_000022_000019", "bremen_000128_000019", "munster_000035_000019", "dusseldorf_000090_000019",
    "bremen_000146_000019", "bremen_000118_000019", "munster_000161_000019", "lindau_000050_000019",
    "bremen_000114_000019", "bremen_000111_000019", "bremen_000121_000019", "jena_000067_000019",
    "bremen_000126_000019", "bremen_000123_000019", "weimar_000119_000019", "munster_000159_000019",
    "dusseldorf_000040_000019", "dusseldorf_000039_000019", "frankfurt_000001_019698",
    "frankfurt_000001_034816", "lindau_000017_000019", "lindau_000018_000019", "lindau_000030_000019",
    "lindau_000031_000019", "lindau_000032_000019", "lindau_000036_000019", "lindau_000040_000019",
    "lindau_000045_000019", "lindau_000046_000019", "lindau_000047_000019", "lindau_000049_000019",
    "lindau_000052_000019", "frankfurt_000000_009291", "frankfurt_000000_009688", "frankfurt_000000_010763",
    "frankfurt_000001_009504", "frankfurt_000001_042098", "frankfurt_000001_051516", "frankfurt_000001_055387",
    "frankfurt_000001_066574", "lindau_000027_000019",
]


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

            if self.filter_image(image_name):
                continue

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

    def filter_image(self, image_name):
        for name in exclude:
            if image_name.startswith(name):
                return True
        return False


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
