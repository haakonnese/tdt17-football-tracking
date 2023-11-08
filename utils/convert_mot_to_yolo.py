import os
from pathlib import Path
import argparse
import shutil


HEIGHT = 1080
WIDTH = 1920
def convert_add_prefix(prefix, folder, output_folder):
    frames = {}
    gt_file = os.path.join(folder, 'gt', 'gt.txt')
    with open(gt_file, 'r') as f:
        "Read file that is in MOT format"
        lines = f.readlines()
    for line in lines:
        frame_num, _, xl, yt, w, h, _, class_id, _ = line.split(',')
        frame_num = int(frame_num)
        xl = float(xl)/WIDTH
        yt = float(yt)/HEIGHT
        w = float(w)/WIDTH
        h = float(h)/HEIGHT
        xc = xl + w/2
        yc = yt + h/2
        class_id = int(class_id) - 1
        if frame_num not in frames:
            frames[frame_num] = []
        frames[frame_num].append([class_id, xc, yc, w, h])
    for frame_num in frames:
        with open(os.path.join(output_folder, prefix + f"{frame_num:06d}" + '.txt'), 'w') as f:
            for box in frames[frame_num]:
                f.write(' '.join([str(x) for x in box]) + '\n')
        shutil.copy(os.path.join(folder, 'img1', f"{frame_num:06d}" + '.jpg'), os.path.join(output_folder, prefix + f"{frame_num:06d}" + '.jpg'))
    

if __name__ == "__main__":
    folders = ["/cluster/work/haaknes/tdt17/yolov8/data/from_idun/1_train-val_1min_aalesund_from_start", 
               "/cluster/work/haaknes/tdt17/yolov8/data/from_idun/2_train-val_1min_after_goal", 
               "/cluster/work/haaknes/tdt17/yolov8/data/from_idun/3_test_1min_hamkam_from_start"]
    output = Path("/cluster/work/haaknes/tdt17/yolov8/data")
    for folder in folders:
        prefix = folder.split("/")[-1][0] + "_"
        if "train" in folder:
            output_folder = Path(output / "train")
        else:
            output_folder = Path(output / "test")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        convert_add_prefix(prefix, folder, output_folder)