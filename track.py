from ultralytics import YOLO
from ultralytics.trackers import BOTSORT, BYTETracker
import json
from utils.custom_annotator import BaseAnnotator, Detection, Color, KeypointAnnotator
import cv2
import numpy as np
from utils.mot_metrics import motMetricsEnhancedCalculator
from utils.case import VIDEO, video
from sahi.predict  import get_sliced_prediction
from sahi import AutoDetectionModel
import os
from PIL import Image
import supervision as sv

abs_positions = np.array([[52.5, 70-25.85, 0], [52.5, 70-44.15, 0], [52.5, 70-70, 0], [43.35, 70-35, 0], [61.65, 70-35, 0]], dtype=np.float32)
model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="runs/detect/train7/weights/last.pt",
    confidence_threshold=0.25,
    device="cuda:0",  # or 'cuda:0'
)
model2 = YOLO("runs/detect/train7/weights/last.pt")
annotator = BaseAnnotator(colors=[Color(255, 255, 0), Color(0, 255, 255)], thickness=2)
keypoint_annotator = KeypointAnnotator(colors=[Color(255, 255, 0), 
                                               Color(0, 255, 255),
                                               Color(255, 0, 255),
                                               Color(255, 0, 0),
                                               Color(0, 255, 0),
                                               Color(0, 0, 255),
                                               Color(0, 0, 0),], thickness=10)

results = model2.track(f"data/from_idun/{VIDEO}/img1", stream=True, device="cuda", tracker="botsort.yaml", persist=True)
size = (1920,1080)
out = cv2.VideoWriter(f'outputs/result_{VIDEO}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
motoutput = ""
run_distance = {}
prev_pos = {}
with open(f"outputs/keypoints_{VIDEO}.json") as f:
    keypoints = dict(json.load(f))

def focalLength_to_camera_matrix(focalLenght, image_size):
    w, h = image_size[0], image_size[1]
    K = np.array([
        [focalLenght, 0, w/2],
        [0, focalLenght, h/2],
        [0, 0, 1],
    ])
    return K
camera_matrix = focalLength_to_camera_matrix(1, (1920, 1080))
path = f"data/from_idun/{VIDEO}/img1"
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Result:
    def __init__(self, result):
        num_pred = len(result.object_prediction_list)
        xyxy = np.zeros((num_pred, 4))
        conf = np.zeros(num_pred)
        cls = np.zeros(num_pred)
        for i, res in enumerate(result.to_coco_annotations()):
            xyxy[i] = self.xywh_toxyxy(res["bbox"])
            conf[i] = res["score"]
            cls[i] = res["category_id"]
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def xywh_toxyxy(self, xywh):
        # print(xywh)
        x = xywh[0]
        y = xywh[1]
        w = xywh[2]
        h = xywh[3]
        x1, y1 = x, y
        x2, y2 = x+w, y+h
        return [x1, y1, x2, y2]

    def print(self):
        return {
            "xyxy": self.xyxy,
            "conf": self.conf,
            "cls": self.cls
        }

def tracker_result_to_json(tracker_result):
    res_list = []
    for res in tracker_result:
        res_list.append({
            "class": res[6],
            "box": {
                "x1": res[0],
                "x2": res[2],
                "y1": res[1],
                "y2": res[3],
            },
            "confidence": res[5],
            "track_id": res[4]
        })
    return res_list

args = dotdict({"track_high_thresh": 0.5,
                  "track_low_thresh": 0.1,
                  "new_track_thresh": 0.6,
                  "track_buffer": 30,
                  "match_thresh": 0.8,
                  "gmc_method": "sparseOptFlow",
                  "proximity_thresh": 0.5,
                  "appearance_thresh": 0.25
                }

                  )

tracker = BOTSORT(args=args)
import time
start = time.perf_counter()
for i, file in enumerate(sorted(os.listdir(path))):
    print(time.perf_counter() - start)
    start = time.perf_counter()
    image = os.path.join(path, file)
    print(image)
    image = np.array(Image.open(image))
    result = get_sliced_prediction(
        image,
        model,
        slice_height = 540,
        slice_width = 960,
        overlap_height_ratio = 0.5,
        overlap_width_ratio = 0.5
    )
    image = image[:, :, ::-1].copy()
    # result2 = next(results)
    # print("YOLO FInished")
    result = Result(result)
    # tracks = tracker.init_track(result.xyx, result.conf, result.cls, image)
    # result = tracker.multi_predict(tracks)
    result = tracker.update(result, image)
    # print(result.print())
    result_dict = tracker_result_to_json(result)
    for res in result_dict:
        motoutput += f"{i+1},{res['track_id']},{res['box']['x1']},{res['box']['y1']},{res['box']['x2']-res['box']['x1']},{res['box']['y2']-res['box']['y1']},1,{int(res['class'])+1},1.0\n"
    detections = Detection.from_results(result_dict, names=["ball", "player"])
    points_to_transform = []
    warp = False
    # calculate position on field for each player if keypoints are available
    if str(i+1) in keypoints:
        # get the x and y coordinates, 
        # point 0: near intersection mid circle, 
        # point 1: far intersection mid circle, 
        # point 2: far intersection midline and outline,
        # point 3: leftmost part of mid circle
        # point 4: rightmost part of mid circle
        x = keypoints[str(i+1)]["x"]
        y = keypoints[str(i+1)]["y"]
        visible = keypoints[str(i+1)]["visible"]
        transform = []
        abs_pos = []
        for j in range(5):
            if visible[j] > 0.95:
                transform.append([x[j], y[j]])
                abs_pos.append(abs_positions[j])
        if len(transform) < 4:
            pass
        else:
            transform = np.array(transform, dtype=np.float32)
            abs_pos = np.array(abs_pos, dtype=np.float32)
            # solve pnp using all the points that are visible and have absolute positions in the field
            # if len(transform) == 3:
                # success, rotation_vector, translation_vector = cv2.solveP3P(abs_pos, transform, camera_matrix, None, flags=cv2.SOLVEPNP_P3P)
            # else:
            success, rotation_vector, translation_vector = cv2.solvePnP(abs_pos, transform, camera_matrix, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
            if success:
                warp = True
                for detection in detections:
                    if detection.class_name == "player":
                        points_to_transform.append([detection.rect.bottom_center.x, detection.rect.bottom_center.y, 1])
                points_to_transform = np.array(points_to_transform, dtype=np.float32).reshape(-1, 3, 1)
                # points = cv2.perspectiveTransform(points_to_transform, M)
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                inv_rotation_matrix = np.linalg.inv(rotation_matrix)
                inv_camera_matrix = np.linalg.inv(camera_matrix)
                right_size = inv_rotation_matrix @ translation_vector
                points = np.zeros((points_to_transform.shape[0], 3, 1))
                for j, point in enumerate(points_to_transform):
                    left_size = inv_rotation_matrix @ inv_camera_matrix @ point
                    s = (right_size[2][0]) / left_size[2][0]
                    points[j] = inv_rotation_matrix @ (s * inv_camera_matrix @ point - translation_vector)
                
                points = points.reshape(-1, 3)
                negative_index = 0
                for j, detection in enumerate(detections):
                    if detection.class_name == "player":
                        index = j - negative_index
                        # if the player has no previous position, set it to the current position
                        if detection.tracker_id not in prev_pos:
                            prev_pos[detection.tracker_id] = points[index]
                        else:
                            # calculate the distance between the previous position and the current position
                            if detection.tracker_id not in run_distance:
                                run_distance[detection.tracker_id] = 0
                            distance_run = np.sqrt((prev_pos[detection.tracker_id][0] - points[index][0])**2 + (prev_pos[detection.tracker_id][1] - points[index][1])**2)
                            if distance_run > 0.4:
                                distance_run = 0.4
                            run_distance[detection.tracker_id] += distance_run
                            prev_pos[detection.tracker_id] = points[index]
                    else:
                        negative_index += 1  # if a ball, remove 1 from index
    else:
        prev_pos = {}  # reset positions

    # add the distance to the detection
    for detection in detections:
        if detection.class_name == "player":
            if detection.tracker_id in run_distance:
                detection.distance = run_distance[detection.tracker_id]
    if str(i+1) in keypoints:
        image = keypoint_annotator.annotate_points(annotator.annotate(image=image.copy(), detections=detections), keypoints=keypoints[str(i+1)])
    else:
        image = annotator.annotate(image=image.copy(), detections=detections)
    out.write(image)

    # show why the distances are not calculated correctly
    # if warp:
    #     out_warp.write(cv2.warpPerspective(image, M_scaled, (105*30, 70*30))) 


out.release()
with open(f"outputs/run_distance_{VIDEO}.json", "w") as f:
    json.dump(run_distance, f)
total_distance = 0
for key in run_distance:
    total_distance += run_distance[key]
print(total_distance)
with open(f"outputs/mot_result_{VIDEO}.txt", "w") as f:
    f.write(motoutput)
if video != 4:   
    motMetricsEnhancedCalculator(f"data/from_idun/{VIDEO}/gt/gt.txt", f"outputs/mot_result_{VIDEO}.txt", 1)
    motMetricsEnhancedCalculator(f"data/from_idun/{VIDEO}/gt/gt.txt", f"outputs/mot_result_{VIDEO}.txt", 2)
    motMetricsEnhancedCalculator(f"data/from_idun/{VIDEO}/gt/gt.txt", f"outputs/mot_result_{VIDEO}.txt")



