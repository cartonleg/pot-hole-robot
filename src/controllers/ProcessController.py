from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from controllers.BaseController import BaseController
from models.enums import ResponseEnums
import io
from PIL import Image
import tempfile
import cv2
import os

class ProcessController(BaseController):
    def __init__(self):
        super().__init__()
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.current_dir)
        
    def detect_pothole_in_image(self, image, weights, confidence:float = 0.2):
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
        
        