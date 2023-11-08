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