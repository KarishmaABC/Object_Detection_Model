

# utils.py
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_model(model_path='yolov8n.pt'):
    model = YOLO(model_path)
    return model


def detect_objects(model, image):
    # Perform inference on the image
    results = model.predict(source=image, save=False)

    # Extract detection data
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        detections.append((cls_id, confidence))

    return results, detections  # Return both results and detections


def draw_boxes(results, image, model):
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        cls = result.cls[0]
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image
