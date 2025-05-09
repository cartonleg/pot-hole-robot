from fastapi.responses import JSONResponse
from ultralytics import YOLO
from controllers.BaseController import BaseController
from models.enums import ResponseEnums
from datetime import datetime

class ProcessController(BaseController):
    def __init__(self):
        super().__init__()
        
    def detect_pothole_in_image(self, image, weights, confidence:float = 0.2):
        model = YOLO(weights)

        results = model(image, conf=confidence)

        if len(results[0].boxes) > 0:
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  
                conf = box.conf.item()  
                detections.append({
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": conf
                })
            return {
                "status": "success",
                "message": "Potholes detected",
                "detections": detections,
                "count": len(detections),
                "date-time": datetime.now()
            }
        elif len(results[0].boxes) <= 0:
            return {
                "status": "success",
                "message": "No potholes detected",
                "detections": [],
                "count": 0,
                "date-time": datetime.now()
            }
        return {
            "status": "fail",
            "message": ResponseEnums.IMAGE_PROCESS_FAIL.value}
        
        