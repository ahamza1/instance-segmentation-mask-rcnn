import skimage.color
import skimage.io
import skimage.io
import skimage.transform

from model.config import InferenceConfig
from model.mrcnn import model as model_lib, visualize


class MaskRCNN:
    def __init__(self, model_dir_path, labels_path, weights_path):
        self.model_dir_path = model_dir_path
        self.labels_path = labels_path
        self.weights_path = weights_path

        self.labels = open(self.labels_path).read().strip().split("\n")
        self.model = self.build_model()

    def build_model(self):
        config = InferenceConfig()
        model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=self.model_dir_path)
        model.load_weights(self.weights_path, by_name=True)
        return model

    def load_image(self, image_path):
        image = skimage.io.imread(image_path, plugin="pil")

        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def detect(self, image_path):
        image = self.load_image(image_path)
        r = self.model.detect([image], verbose=1)[0]

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    self.labels, r['scores'], save_fig_path=image_path, min_score=0.5)

        return {
            "objects": self.count_objects(r['class_ids'])
        }

    def count_objects(self, result_class_ids):
        counter = {}
        for class_id in result_class_ids:
            counter[self.labels[class_id]] = counter.get(self.labels[class_id], 0) + 1
        return counter


def register_model(app):
    return MaskRCNN(
        app.config["MODEL_DIR_PATH"],
        app.config["LABELS_PATH"],
        app.config["WEIGHTS_PATH"]
    )
