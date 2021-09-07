import argparse
import os


def get_args_train():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True)
    ap.add_argument("-l", "--labels", required=True)
    ap.add_argument("-w", "--weights", required=False)
    args = vars(ap.parse_args())
    return args["data"], args["labels"], args["weights"]


def get_args_inference():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-l", "--labels", required=True)
    ap.add_argument("-w", "--weights", required=True)
    args = vars(ap.parse_args())
    return args["image"], args["labels"], args["weights"]


def save_history(history):
    a_file = open(os.path.join(os.path.abspath("./"), "history.txt"), "a")
    a_file.write("\n")
    a_file.write(str(history))
    a_file.close()


def plt_results(history):
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd

    matplotlib.use('TkAgg')
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

    plt.show()
