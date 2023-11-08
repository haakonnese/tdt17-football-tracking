from ultralytics import YOLO
import json
from custom_annotator import BaseAnnotator, Detection, Color
import cv2

model = YOLO("runs/detect/train7/weights/last.pt")
annotator = BaseAnnotator(colors=[Color(255, 255, 0), Color(0, 255, 255)], thickness=2)

results = model.track("data/test", stream=True, device="cuda", tracker="botsort.yaml", persist=True)
size = (1920,1080)
out = cv2.VideoWriter('outputs/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
motoutput = ""
for i, result in enumerate(results):
    result_dict = json.loads(result.tojson())
    for res in result_dict:
        motoutput += f"{i+1},{res['track_id']},{res['box']['x1']},{res['box']['y1']},{res['box']['x2']-res['box']['x1']},{res['box']['y2']-res['box']['y1']},1,{int(res['class'])+1},1.0\n"
    detections = Detection.from_results(result_dict, names=model.names)
    out.write(annotator.annotate(image=result.orig_img.copy(), detections=detections))
out.release()
with open("outputs/mot_result.txt", "w") as f:
    f.write(motoutput)

