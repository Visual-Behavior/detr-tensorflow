from data.coco import load_coco



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cocodir",  type=str, required=True, help="/path/to/coco")


def main(args):
    print('args', args)
    #Train dataset
    train_dt = load_coco(args.cocodir, "val")
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    pass

