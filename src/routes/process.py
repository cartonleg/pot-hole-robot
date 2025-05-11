from fastapi import APIRouter, UploadFile, Depends
from fastapi.responses import FileResponse
from helpers.config import Settings, get_settings
from controllers.ProcessController import ProcessController
from typing import List
import os
import shutil
from zipfile import ZipFile
import tempfile
import uuid

process_router = APIRouter(prefix='/process')

@process_router.post('/image')
async def process_image_return_image(uploaded_images: List[UploadFile], 
                       app_settings:Settings = Depends(get_settings)):
    

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'assets', 'models', 'pothole_segmentation_model_YOLOv8.pt')
    model_path = os.path.abspath(model_path)

    result_paths = []

    for uploaded_image in uploaded_images:

        file_ext = os.path.splitext(uploaded_image.filename)[-1]
        temp_filepath = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
        

        try:
            with open(temp_filepath.name, "wb") as buffer:
                shutil.copyfileobj(uploaded_image.file, buffer)

            processed_image_path = ProcessController().detect_pothole_return_image(
                image=temp_filepath.name,
                weights=model_path,
                confidence=app_settings.CONFIDENCE
            )
            
            result_paths.append(processed_image_path)

        finally:
            if os.path.exists(temp_filepath.name):
                os.remove(temp_filepath.name)
    
    zip_tmp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    with ZipFile(zip_tmp.name, 'w') as zipf:
        for path in result_paths:
            arcname = os.path.basename(path)
            zipf.write(filename=path, arcname=arcname)
            if os.path.exists(path):
                os.remove(path)
    
    return FileResponse(path=zip_tmp.name,
                        media_type='application/zip',
                        filename='processed_images.zip')


@process_router.post('/json')
async def process_image_return_json(uploaded_images: List[UploadFile], 
                       app_settings:Settings = Depends(get_settings)):
    
    output = {}
    
    for image in uploaded_images:
    
        file_ext = os.path.splitext(image.filename)[-1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_filepath = os.path.join("/tmp", temp_filename)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'assets', 'models', 'pothole_segmentation_model_YOLOv8.pt')
        model_path = os.path.abspath(model_path)

        try:
            with open(temp_filepath, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            result = ProcessController().detect_pothole_return_json(
                image=temp_filepath,
                weights=model_path,
                confidence=app_settings.CONFIDENCE
            )
            
            output.update({image.filename: result})

        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    
    return output


