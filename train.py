from ultralytics import YOLO

model = YOLO('yolov8l.pt')

results = model.train(data='configs/data.yaml', 
                      epochs=10,
                      batch=32,
                      imgsz=1920,
                      verbose=True,
                      task="detect",
                      close_mosaic=2,
                      warmup_epochs=3,
                      workers=4,
                      device=[0,1])
model.val()