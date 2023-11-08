import shutil
import os
from pathlib import Path

if __name__ == "__main__":
    folder = "/cluster/work/haaknes/tdt17/yolov8/data/from_idun/1_train-val_1min_aalesund_from_start/img1"
    output = "/cluster/work/haaknes/tdt17/yolov8/data/keypoint/train"
    labels = "/cluster/work/haaknes/tdt17/yolov8/data/keypoint/labels/train"
    Path(output).mkdir(parents=True, exist_ok=True)
    for file in os.listdir(folder):
        # if int(file[-7:-4]) % 10 != 0:
            # continue
        if file.endswith(".jpg"):
            shutil.copy(os.path.join(folder, file), os.path.join(output, "1_" + file))
    for file in os.listdir(labels):
        # if int(file[-7:-4]) % 10 != 0:
            # continue
        if file.endswith(".txt"):
            shutil.copy(os.path.join(labels, file), os.path.join(output, file))
