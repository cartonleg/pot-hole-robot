from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from controllers.BaseController import BaseController
from models.enums import ResponseEnums
import io
from PIL import Image
import tempfile
import cv2

class ProcessController(BaseController):
    def __init__(self):
        super().__init__()
        
    def detect_pothole_in_image(self, image, weights, confidence:float = 0.2):
        model = YOLO(weights)

        result = model(image, conf=confidence)

        if len(result[0].boxes) > 0:
            
            result_annotated = result[0].plot()  
            
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, result_annotated) 

            return FileResponse(tmp.name, media_type="image/jpeg")

        elif len(result[0].boxes) <= 0:
            return {
                "status": "success",
                "message": "No potholes detected",
                "detections": [],
                "count": 0
            }
        
        return {
            "status": "fail",
            "message": ResponseEnums.IMAGE_PROCESS_FAIL.value}
        
        