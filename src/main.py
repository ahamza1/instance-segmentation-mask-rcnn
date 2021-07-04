from model import MaskRCNN


def get_input_arguments():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weights", required=True)
    ap.add_argument("-l", "--labels", required=True)
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())

    return args["weights"], args["labels"], args["image"]


def main():
    # Parse input arguments
    weights_path, labels_path, image_path = get_input_arguments()

    # Initialize the model for inference on trained weights
    model = MaskRCNN(weights_path, labels_path)

    # Load image and run detection
    image = model.load_image(image_path)
    image, result = model.detect(image)

    # Visualize results
    model.visualize(image, result)


if __name__ == "__main__":
    main()
