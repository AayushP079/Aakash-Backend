import os
from typing import Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import google.generativeai as genai
from config import GOOGLE_API_KEY
from PyPDF2 import PdfReader
from db import get_db_collection
from datetime import datetime
import random
import string
import csv
from io import StringIO
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change to specific URLs in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
class ScrapeResponse(BaseModel):
    message: str
    extractedText: List[str]

# 

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/api/scrape", response_model=ScrapeResponse)
async def scrape_form(url: str):
    try:
        # Step 1: Fetch HTML content using requests
        response = requests.get(url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch the HTML content")

        html = response.text
        
        # Step 2: Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Step 3: Extract text from span elements with class 'M7eMe'
        spans = soup.find_all('span', class_='M7eMe')
        span_texts = [span.get_text(strip=True) for span in spans]

        if not span_texts:
            raise HTTPException(status_code=500, detail="No span elements found with class 'M7eMe'")

        # Step 4: Return the extracted text
        return ScrapeResponse(message="Successfully extracted text.", extractedText=span_texts)

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching the page: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during parsing: {str(e)}")


class UploadData(BaseModel):
    name: str
    email: str

# Function to convert CSV text to a list of dictionaries
def csv_to_dict(csv_text: str) -> List[dict]:
    # Use StringIO to treat the string as a file-like object
    csv_file = StringIO(csv_text)
    reader = csv.DictReader(csv_file)
    # Convert rows into a list of dictionaries
    return [row for row in reader]

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), name: str = Form(...), email: str = Form(...)):
    upload_dir = './uploaded_files'
    os.makedirs(upload_dir, exist_ok=True)  # Create directory if not exists
    file_location = os.path.join(upload_dir, file.filename)
    
    # Save file locally
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process with GenAI
    file = genai.upload_file(file_location, mime_type=file.content_type)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(["Give me a data in csv format with correct value", file])
    
    csv_as_dict = csv_to_dict(response.text)

    # Prepare the data to be saved in MongoDB
    username = name.replace(" ", "_")  # Replace spaces with underscores for the collection name
    unique_number = random.randint(1000, 9999)  # Generate a random number
    collection_name = f"{username}_{unique_number}"

     # Prepare data to be saved in MongoDB
    data_dict = {
        "name": name,
        "email": email,
        "content": csv_as_dict,
        "uploaded_at": datetime.utcnow()
    }


    # Save data into a dynamic collection
    collection = get_db_collection(collection_name)
    collection.insert_one(data_dict)


    # remove current file after extracting data
    os.remove(file_location)

    return JSONResponse(content={"data": csv_as_dict, "status": "Data saved to database."})

@app.post("/api/fill_form")
async def fill_form_endpoint(file: UploadFile = File(...), name: str = Form(...), email: str = Form(...)):
    upload_dir = './uploaded_files/form'
    os.makedirs(upload_dir, exist_ok=True)  # Create directory if not exists
    file_location = os.path.join(upload_dir, file.filename)
    
    # Save file locally
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Query the database to get the data
    username = name.replace(" ", "_")  # Replace spaces with underscores for the collection name
    collection = get_db_collection(username)
    data = collection.find_one({"email": email})
    print(data)
    
    if not data:
        raise HTTPException(status_code=404, detail="Data not found for the provided email")

    # Extract fields from OCR result
    form_data = data["content"]
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    results = ocr.ocr(file_location, cls=True)

    result = results[0]

    fields = extract_fields(results)

    # Parse the OCR result to get bounding boxes
    boxes, txts, scores = parse_result(result)

    # Fill the form with the extracted fields
    fill_form(fields, boxes, file_location)

    # Return the filled form as a response
    return FileResponse('output2.png', media_type='image/png', filename='output2.png')

def extract_fields(ocr_result):
    fields = {}
    for line in ocr_result:
        for word_info in line:
            field_name = word_info[1][0]
            fields[field_name] = None
    return fields

def parse_result(result):
    """
    Returns a tuple of bboxes_list, detected_text_list, confidence_list
    """
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return boxes, txts, scores

def fill_form(fields, boxes, image_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    fontSize = (boxes[0][2][1] - boxes[0][0][1])
    font = ImageFont.truetype("/content/simfang.ttf", fontSize)

    x_shift = 3 * (boxes[0][1][0] - boxes[0][0][0])
    y_shift = 10 * (boxes[0][1][1] - boxes[0][0][1])

    delta_x = boxes[1][1][0] - boxes[0][1][0]
    delta_y = boxes[1][1][1] - boxes[0][1][1]
    # Determine Coordinates of empty space
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
        print(shifted_coordinate)

        text = list(fields.values())[i]
        # Skip if text is None
        if text is not None:
            position = (shifted_coordinate[0][0], shifted_coordinate[0][1])
            draw.text(position, text, font=font, fill=(0, 0, 0))
        else:
            # Handle the case where there are more boxes than fields
            print(f"Warning: No field value for box index {i}")

    image.save('output2.png')