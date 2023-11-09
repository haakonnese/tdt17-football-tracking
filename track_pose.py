from ultralytics import YOLO
import json
from utils.custom_annotator import Color, KeypointAnnotator, KeypointDetection
import cv2
VIDEO = "1_train-val_1min_aalesund_from_start"
VIDEO = "2_train-val_1min_after_goal"
VIDEO = "3_test_1min_hamkam_from_start"

model = YOLO('/cluster/work/haaknes/tdt17/yolov8/runs/pose/train22/weights/last.pt')
annotator = KeypointAnnotator(colors=[Color(255, 255, 0), 
                                      Color(0, 255, 255),
                                      Color(255, 0, 255),
                                      Color(255, 0, 0),
                                      Color(0, 255, 0),
                                      Color(0, 0, 255),
                                      Color(0, 0, 0),], thickness=10)

results = model.track(f"data/from_idun/{VIDEO}/img1", stream=True, device="cuda", tracker="botsort.yaml", persist=True, conf=0.01)
size = (1920,1080)
out = cv2.VideoWriter(f'outputs/pose_result_{VIDEO}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
keypoints = {}
for i, result in enumerate(results):
    result_dict = json.loads(result.tojson())
    # print(result_dict)
    detections = KeypointDetection.from_results(result_dict, names=model.names)  # length is either 0 or 1.
    for res in detections:
        keypoints[i+1] = res.keypoints.dict()
    out.write(annotator.annotate(image=result.orig_img.copy(), detections=detections))
with open(f"outputs/keypoints_{VIDEO}.json", "w") as f:
    json.dump(keypoints, f)
out.release()

