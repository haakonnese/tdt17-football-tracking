from ultralytics import YOLO

model = YOLO('yolov8l-pose.pt')
# model = YOLO('/cluster/work/haaknes/tdt17/yolov8/runs/pose/train18/weights/last.pt')

results = model.train(data='configs/pose.yaml', 
                      epochs=50,
                      batch=10,
                      imgsz=1920,
                      verbose=True,
                      task="pose",
                      mosaic=False,
                      warmup_epochs=3,
                      workers=0,
                      device=[0])
# model(source="/cluster/work/haaknes/tdt17/yolov8/data/train", save=True)