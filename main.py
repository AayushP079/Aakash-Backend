import os
from typing import Union, List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from config import GOOGLE_API_KEY
from db import get_db_collection
from datetime import datetime
from io import StringIO
import csv
from PIL import Image, ImageDraw, ImageFont
import joblib
import numpy as np
import cv2
import tempfile
import shutil
from argparse import Namespace

from paddleocr import PaddleOCR
from .ppstructure.kie.predict_kie_token_ser import SerPredictor

# Loading the vectorizer and classifier
vectorizer = joblib.load("models/vectorizer.joblib")
classifier = joblib.load("models/classifier.joblib")

class KIEConfig:
    def __init__(self):
          self.args = Namespace(
            kie_algorithm='LayoutXLM',
            ser_model_dir='models/ser_vi_layoutxlm_xfund_infer',
            ser_dict_path='ppocr/utils/dict/kie_dict/xfund_class_list.txt',
            vis_font_path='fonts/simfang.ttf',
            ocr_order_method='tb-yx',
            use_angle_cls=True,
            det_model_dir=None,
            rec_model_dir=None,
            use_gpu=False,
            use_onnx=False
        )

class KIEProcessor:
    def __init__(self, config: KIEConfig):
        self.predictor = SerPredictor(config.args)
    
    def process_image(self, image: np.ndarray) -> Dict:
        try:
            ser_res, _, _ = self.predictor(image)
            if not ser_res:
                raise ValueError("No results found in the image")
            
            result = ser_res[0]
            extracted_data = {}
            for item in result:
                if 'pred' in item and 'text' in item:
                    key = item['pred']
                    value = item['text']
                    if key and value:
                        if key in extracted_data:
                            if isinstance(extracted_data[key], list):
                                extracted_data[key].append(value)
                            else:
                                extracted_data[key] = [extracted_data[key], value]
                        else:
                            extracted_data[key] = value
            
            return extracted_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Initialize FastAPI app and KIE processor
app = FastAPI()
config = KIEConfig()
kie_processor = KIEProcessor(config)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapeResponse(BaseModel):
    message: str
    extractedText: List[str]

class UploadData(BaseModel):
    name: str
    email: str

def csv_to_dict(csv_text: str) -> List[dict]:
    csv_file = StringIO(csv_text)
    reader = csv.DictReader(csv_file)
    return [row for row in reader]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/api/scrape", response_model=ScrapeResponse)
async def scrape_form(url: str):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch the HTML content")

        soup = BeautifulSoup(response.text, 'html.parser')
        spans = soup.find_all('span', class_='M7eMe')
        span_texts = [span.get_text(strip=True) for span in spans]

        if not span_texts:
            raise HTTPException(status_code=500, detail="No span elements found with class 'M7eMe'")

        return ScrapeResponse(message="Successfully extracted text.", extractedText=span_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), name: str = Form(...), email: str = Form(...)):
    try:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Read and process the image with KIE
            image = cv2.imread(temp_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with KIE
            extracted_data = kie_processor.process_image(image)
            
            # Process with GenAI if needed
            genai_file = genai.upload_file(temp_path, mime_type=file.content_type)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(["Give me a data in csv format with correct value.", genai_file])
            
            csv_as_dict = csv_to_dict(response.text)

            # Combine KIE and GenAI results
            combined_data = {
                "kie_data": extracted_data,
                "genai_data": csv_as_dict
            }

            # Save to MongoDB
            collection_name = email
            data_dict = {
                "name": name,
                "email": email,
                "content": combined_data,
                "uploaded_at": datetime.utcnow()
            }
            
            collection = get_db_collection(collection_name)
            collection.insert_one(data_dict)

            return JSONResponse(content={
                "data": combined_data,
                "status": "Data saved to database."
            })
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.post("/api/fill_form")
async def fill_form_endpoint(file: UploadFile = File(...), name: str = Form(...), email: str = Form(...)):
    upload_dir = './uploaded_files/form'
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)
    
    try:
        # Save file locally
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Query database
        collection = get_db_collection(email)
        data = collection.find_one({"email": email})
        
        if not data:
            raise HTTPException(status_code=404, detail="Data not found for the provided email")

        # Extract fields from database
        form_data = data["content"]["kie_data"] if "kie_data" in data["content"] else data["content"][0]

        # Process with OCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        results = ocr.ocr(file_location, cls=True)
        
        fields = extract_fields(results)
        form_data = predict_and_update_keys(form_data)

        # Parse OCR result
        boxes, txts, scores = parse_result(results[0])

        # Update fields with form data
        for key, value in fields.items():
            if key in form_data:
                fields[key] = form_data[key]

        # Fill form
        fill_form(fields, boxes, file_location)

        return FileResponse('output2.png', media_type='image/png', filename='output2.png')
    
    finally:
        # Cleanup
        if os.path.exists(file_location):
            os.remove(file_location)

def predict_and_update_keys(data: dict) -> dict:
    original_keys = list(data.keys())
    input_tfidf = vectorizer.transform(original_keys)
    predicted_keys = classifier.predict(input_tfidf)
    
    updated_data = {}
    for original_key, predicted_key in zip(original_keys, predicted_keys):
        updated_data[predicted_key] = data[original_key]
    
    return updated_data

def extract_fields(ocr_result):
    fields = {}
    field_names = []
    for line in ocr_result:
        for word_info in line:
            field_name = word_info[1][0]
            field_names.append(field_name)

    if field_names:
        input_tfidf = vectorizer.transform(field_names)
        predicted_keys = classifier.predict(input_tfidf)
        fields = {predicted_key: None for predicted_key in predicted_keys}

    return fields

def parse_result(result):
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return boxes, txts, scores

def fill_form(fields, boxes, image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    fontSize = (boxes[0][2][1] - boxes[0][0][1])
    font = ImageFont.truetype("./fonts/simfang.ttf", fontSize)

    x_shift = 3 * (boxes[0][1][0] - boxes[0][0][0])
    y_shift = 10 * (boxes[0][1][1] - boxes[0][0][1])

    delta_x = boxes[1][1][0] - boxes[0][1][0]
    delta_y = boxes[1][1][1] - boxes[0][1][1]

    for i in range(len(boxes)):
        box_coordinate = boxes[i]
        if delta_x < delta_y:
            shifted_coordinate = [
                [coord[0] + x_shift, coord[1]]
                for coord in box_coordinate
            ]
        else:
            shifted_coordinate = [
                [coord[0], coord[1] + y_shift]
                for coord in box_coordinate
            ]
            
        text = list(fields.values())[i] if i < len(fields) else None
        if text is not None:
            position = (shifted_coordinate[0][0], shifted_coordinate[0][1])
            draw.text(position, text, font=font, fill=(0, 0, 0))

    image.save('output2.png')