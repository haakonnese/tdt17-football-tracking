from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import cv2

import numpy as np


# geometry utilities


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    
    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class KeyPoints:
    x: List[float]
    y: List[float]
    visible: List[float]
    
    @property
    def int_xy_tuple(self, idx) -> Tuple[int, int]:
        return int(self.x[idx]), int(self.y[idx])
    
    @classmethod
    def from_dict(cls, keypoints: Dict[str, Any]) -> KeyPoints:
        return KeyPoints(
            x=keypoints["x"],
            y=keypoints["y"],
            visible=keypoints["visible"]
        )
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.visible[idx]

    def __iter__(self):
        for i in range(len(self.x)):
            yield self[i]
    
    def dict(self):
        return {"x": self.x, "y": self.y, "visible": self.visible}
    
    def __str__(self):
        return str(self.dict())

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x
    
    @property
    def min_y(self) -> float:
        return self.y
    
    @property
    def max_x(self) -> float:
        return self.x + self.width
    
    @property
    def max_y(self) -> float:
        return self.y + self.height
        
    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)
    
    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding, 
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )
    
    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y


# detection utilities


@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None
    distance: Optional[float] = None

    @classmethod
    def from_results(cls, preds: list[dict], names: list[str]) -> List[Detection]:
        result = []
        for pred in preds:
            # print(pred)
            class_id=int(pred['class'])
            x_min = float(pred['box']['x1'])
            y_min = float(pred['box']['y1'])
            x_max = float(pred['box']['x2'])
            y_max = float(pred['box']['y2'])
            confidence = float(pred['confidence'])
            result.append(Detection(
                rect=Rect(
                    x=x_min,
                    y=y_min,
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=confidence,
                tracker_id=int(pred['track_id'])
            ))
        return result


@dataclass
class KeypointDetection:
    name: str
    class_id: int
    confidence: float
    box: Rect
    keypoints: KeyPoints

    @classmethod
    def from_results(cls, preds: list[dict], names: list[str]) -> List[KeypointDetection]:
        result = []
        highest_confidence = 0
        highest_confidence_idx = -1
        for i, pred in enumerate(preds):
            if pred['confidence'] > highest_confidence:
                highest_confidence = pred['confidence']
                highest_confidence_idx = i
        if highest_confidence_idx == -1:
            return result
        pred = preds[highest_confidence_idx]
        class_id=int(pred['class'])
        x_min = float(pred['box']['x1'])
        y_min = float(pred['box']['y1'])
        x_max = float(pred['box']['x2'])
        y_max = float(pred['box']['y2'])
        confidence = float(pred['confidence'])
        result.append(KeypointDetection(
            name=names[class_id],
            class_id=class_id,
            confidence=confidence,
            box=Rect(
                x=x_min,
                y=y_min,
                width=float(x_max - x_min),
                height=float(y_max - y_min)
            ),
            keypoints=KeyPoints.from_dict(pred['keypoints'])
        ))
        return result
    
    @classmethod
    def from_dict(cls, keypoints: Dict[str, Any], names: list[str]) -> List[KeypointDetection]:
        result = []
        result.append(KeypointDetection(
            name=names[keypoints["class_id"]],
            class_id=keypoints["class_id"],
            confidence=keypoints["confidence"],
            box=Rect(
                x=keypoints["box"]["x"],
                y=keypoints["box"]["y"],
                width=keypoints["box"]["width"],
                height=keypoints["box"]["height"]
            ),
            keypoints=KeyPoints.from_dict(keypoints["keypoints"])
        ))
        return result


def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [
        detection
        for detection 
        in detections
        if detection.class_name == class_name
    ]


# draw utilities


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int
        
    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str) -> Color:
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)


def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
    return image


def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image


def draw_filled_polygon(image: np.ndarray, countour: np.ndarray, color: Color) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
    return image


def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.bgr_tuple, thickness, 2, False)
    return image


def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image

def draw_point(image: np.ndarray, point: Point, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.circle(image, point.int_xy_tuple, thickness, color.bgr_tuple, -1)
    return image

# base annotator
  

@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_ellipse(
                image=image,
                rect=detection.rect,
                color=self.colors[detection.class_id],
                thickness=self.thickness
            )
            annotated_image = draw_text(
                image=annotated_image,
                anchor=detection.rect.top_center,
                text=f"{detection.tracker_id}",
                color=self.colors[detection.class_id],
                thickness=self.thickness
            )
            if detection.distance is not None:
                annotated_image = draw_text(
                    image=annotated_image,
                    anchor=detection.rect.bottom_center,
                    text=f"{detection.distance:.2f}m",
                    color=self.colors[detection.class_id],
                    thickness=self.thickness
                )
        return annotated_image
    
@dataclass
class KeypointAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[KeypointDetection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            for i, (x, y, visible) in enumerate(detection.keypoints):
                if visible > 0.5:
                    annotated_image = draw_point(
                        image=annotated_image,
                        point=Point(x=x, y=y),
                        color=self.colors[i],
                        thickness=self.thickness
                    )
        return annotated_image    
    
    def annotate_points(self, image, keypoints) -> np.ndarray:
        keypoints = KeyPoints.from_dict(keypoints)
        anntated_image = image.copy()
        for i, (x, y, visible) in enumerate(keypoints):
            if visible > 0.95:
                anntated_image = draw_point(
                    image=anntated_image,
                    point=Point(x=x, y=y),
                    color=self.colors[i],
                    thickness=self.thickness
                )
        return anntated_image