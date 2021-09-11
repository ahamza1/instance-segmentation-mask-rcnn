import os

import numpy as np

from config import InferenceConfig
from dataset import StreetsDataset
from mrcnn import model as model_lib
from mrcnn.model import load_image_gt, mold_image
from mrcnn.utils import compute_ap_range, compute_ar


def main():
    model_dir = os.path.join(os.path.abspath("./"), "out")
    weights_path = os.path.join(model_dir, "cityscapes20210908T1505", "mask_rcnn_cityscapes_0010.h5")

    data_path = os.path.join(os.path.abspath("./"), "resources")
    labels_path = os.path.join(data_path, "labels.txt")

    config = InferenceConfig()

    # Load and prepare val and test dataset
    dataset_val = StreetsDataset()
    dataset_val.load_dataset(data_path, labels_path, "val")
    dataset_val.prepare()

    # Initialize the Mask R-CNN model for inference and then load the weights
    model = model_lib.MaskRCNN(mode="inference", config=config, model_dir=model_dir)
    model.load_weights(weights_path, by_name=True)

    mAP05, mAR05 = evaluate_model(dataset_val, model, config, [.5])
    print(f"@IoU 0.5: (mAP: {mAP05:.3f}, mAR: {mAR05:.3f})")

    mAP075, mAR075 = evaluate_model(dataset_val, model, config, [.75])
    print(f"@IoU 0.75: (mAP: {mAP075:.3f}, mAR: {mAR075:.3f})")

    mAP05_095, mAR05_095 = evaluate_model(dataset_val, model, config, np.arange(0.5, 1.0, 0.05))
    print(f"@IoU IoU 0.5-0.95: (mAP: {mAP05_095:.3f}, mAR: {mAR05_095:.3f})")


def evaluate_model(dataset, model, cfg, iou_thresholds):
    APs = list()
    ARs = list()

    for image_id in dataset.image_ids:
        # Load image, bounding boxes and masks for the image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False
        )

        # Run detection
        result = model.detect(np.expand_dims(mold_image(image, cfg), 0), verbose=0)
        r = result[0]

        # Calculate AP / AR
        AP = compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"],
                              r["class_ids"], r["scores"], r['masks'], iou_thresholds, verbose=0
                              )
        AR = compute_ar(r['rois'], gt_bbox, np.arange(0.5, 1.0, 0.05))

        APs.append(AP)
        ARs.append(AR)

    # Calculate the mean AP / AR across all images
    mAP = np.mean(APs)
    mAR = np.mean(ARs)

    return mAP, mAR


if __name__ == "__main__":
    main()
