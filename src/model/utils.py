import argparse


def get_args():
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
