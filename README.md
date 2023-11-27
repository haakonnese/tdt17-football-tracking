# Tracking football
## Prerequisites
Use conda, either anaconda or miniconda.

Create a new conda environment
```
conda env create -n yolov8 python=3.11
conda activate yolov8
```
## Requirements
```
conda install pytorch=2.1 torchvision=0.16 torchaudio=2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```
Then run 
```
pip install -r requirements.txt
```
The order is important as ultralytics will install pytorch if it doesn't exist, but without GPU.

## Copy data 
You will need to copy the data from the idun project
Run 
```
cp -r /cluster/projects/vc/data/other/open/RBK_TDT17/1_train-val_1min_aalesund_from_start/ /cluster/work/<username>/<folder-structure>/yolov8/data/from_idun
cp -r /cluster/projects/vc/data/other/open/RBK_TDT17/3_test_1min_hamkam_from_start/ /cluster/work/<username>/<folder-structure>/yolov8/data/from_idun
cp -r /cluster/projects/vc/data/other/open/RBK_TDT17/2_train-val_1min_after_goal/ /cluster/work/<username>/<folder-structure>/yolov8/data/from_idun
```
Change `<username>` and `<folder-structure>` with your actual folder structure

## Training
To train the model, you will first need to create the training and validation sets. This is done by running
```
cd utils
python convert_mot_to_yolo.py
cd ..
python train.py
```

## Trainin pose estimation
To train the model, you will need to create the pose-estimation annotations from the COCO format. The paths are hard-coded, so you will need to change them in the files.
```
cd utils
python cocotoyolopose.py
python copy_train_to_keypoint_training.py
cd ..
python train_pose.py
```

## Testing
If you want to test everything, then you will first need to create pose-estimation annotations from the video, by running pose-estimation inference on it. This is done by running
```
python test_pose.py <CASE>
```
Change `<CASE>` to the case you want to test. The options are `1`, `2`, `3` and `4`, as a shortname for the different datasets.

Then you will need to 
```
python test.py <CASE>
```
The path to the weights file is currently hardcoded, and will need to be changed.