from ultralytics import YOLO

model = YOLO('yolov8l-pose.pt')
# model = YOLO('/cluster/work/haaknes/tdt17/yolov8/runs/pose/train13/weights/last.pt')

results = model.train(data='configs/pose.yaml', 
                      epochs=30,
                      batch=40,
                      imgsz=1920,
                      verbose=True,
                      task="pose",
                    #   close_mosaic=2,
                      mosaic=False,
                      warmup_epochs=3,
                      workers=8,
                      device=[0,1,2,3])
# model(source="/cluster/work/haaknes/tdt17/yolov8/data/train", save=True)