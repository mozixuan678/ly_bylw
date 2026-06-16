from __future__ import annotations

import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ADdata.npy to a version-stable NPZ cache.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    obj = np.load(args.data, allow_pickle=True).item()
    data = np.asarray(obj["data"], dtype=np.float32)
    label = obj["label"]
    labels = label["Group"].astype(str).to_numpy()
    subjects = label["Subject"].astype(str).to_numpy() if "Subject" in label else np.arange(len(labels)).astype(str)
    image_ids = label["Image Data ID"].astype(str).to_numpy() if "Image Data ID" in label else subjects
    np.savez_compressed(args.out, data=data, labels=labels, subjects=subjects, image_ids=image_ids)
    print("Saved %s with data shape %s" % (args.out, data.shape))


if __name__ == "__main__":
    main()

