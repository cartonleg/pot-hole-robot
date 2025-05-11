from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from controllers.BaseController import BaseController
from models.enums import ResponseEnums
import io
from PIL import Image
import tempfile
import cv2
import os
from datetime import datetime

class ProcessController(BaseController):
    def __init__(self):
        super().__init__()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.current_dir)
        
    def detect_pothole_return_image(self, image, weights, confidence:float = 0.2):
        model = YOLO(weights)

        result = model(image, conf=confidence)

        if len(result[0].boxes) > 0:
            
            result_annotated = result[0].plot()  
            
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, result_annotated) 

            return tmp.name

        elif len(result[0].boxes) <= 0:

            image_path = os.path.join(self.root_dir, 'assets', 'images', 'not_pot_hole_detected.png')

            return image_path
        

        image_path = os.path.join(self.root_dir, 'assets', 'images', 'error_while_detecting.png')
        return image_path
    

    def detect_pothole_return_json(self, image, weights, confidence:float = 0.2):
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