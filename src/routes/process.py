from fastapi import APIRouter, Request, UploadFile, Depends
from helpers.config import Settings, get_settings
from controllers.ProcessController import ProcessController
from typing import List
import os
import shutil
import uuid

process_router = APIRouter(prefix='/process')

@process_router.post('/image')
async def upload_image(uploaded_image: List[UploadFile], 
                       app_settings:Settings = Depends(get_settings)):
    
    output = {}
    
    for image in uploaded_image:
    
        file_ext = os.path.splitext(image.filename)[-1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_filepath = os.path.join("/tmp", temp_filename)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'assets', 'models', 'pothole_segmentation_model_YOLOv8.pt')
        model_path = os.path.abspath(model_path)

        try:
            with open(temp_filepath, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            result = ProcessController().detect_pothole_in_image(
                image=temp_filepath,
                weights=model_path,
                confidence=app_settings.CONFIDENCE
            )
            
            output.update({image.filename: result})

        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    return output
