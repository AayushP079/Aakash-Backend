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
import joblib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from time import sleep
import math
from PIL import Image, ImageDraw, ImageFont
import enchant

dictionary = enchant.Dict("en_US")
# Loading the vectorizer and classifer
vectorizer = joblib.load("models/vectorizer.joblib")
classifier = joblib.load("models/classifier.joblib")

app = FastAPI()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
DEBUG = os.getenv('DEBUG', 'False') == 'True'

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
    response = model.generate_content(["Give me a data in csv format with correct value.", file])
    
    print(response.text)
    csv_as_dict = csv_to_dict(response.text)

    # Prepare the data to be saved in MongoDB
    collection_name = email

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
    print(username)
    collection = get_db_collection(email)
    # print(collection)
    print(email)
    data = collection.find_one({"email": email})
    print("collection size")
    col_size = len(data)
    print("collection Size",col_size)

    
    if not data:
        raise HTTPException(status_code=404, detail="Data not found for the provided email")

    # Extract fields from OCR result
    form_data = data["content"][0]



    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    results = ocr.ocr(file_location, cls=True)

    result = results[0]

    fields = extract_fields(results)

    form_data = predict_and_update_keys(form_data)

    print("fields")
    print(fields)

    # Parse the OCR result to get bounding boxes
    boxes, txts, scores = parse_result(result)
    print("formdata:")
    print(form_data)

#  Assign values from fields to form_data
    for key, value in fields.items():
        if key in form_data:
            fields[key] = form_data[key]
    
    print("After filled:")
    print(fields)

    print("len of boxes", len(boxes), boxes)
    print("len of fields", len(fields), fields)
    # Fill the form with the extracted fields
    fill_form(fields, boxes, file_location)

    # Return the filled form as a response
    return FileResponse('output2.png', media_type='image/png', filename='output2.png')

def predict_and_update_keys(data: dict) -> dict:
    """
    Predict and update the keys of the given dictionary using the vectorizer and classifier.
    """
    original_keys = list(data.keys())
    
    # Transform the keys using the vectorizer
    input_tfidf = vectorizer.transform(original_keys)
    
    # Predict the standardized keys
    predicted_keys = classifier.predict(input_tfidf)
    
    # Create a new dictionary with the predicted keys
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

    # Use the vectorizer and classifier to predict standard field keys
    if field_names:
        input_tfidf = vectorizer.transform(field_names)
        predicted_keys = classifier.predict(input_tfidf)

        # Map predicted keys to original field names
        fields = {predicted_key: None for predicted_key in predicted_keys}

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
    font = ImageFont.truetype("./fonts/simfang.ttf", fontSize)

    x_shift = 3 * (boxes[0][1][0] - boxes[0][0][0])
    y_shift = 10 * (boxes[0][1][1] - boxes[0][0][1])

    delta_x = boxes[1][1][0] - boxes[0][1][0]
    delta_y = boxes[1][1][1] - boxes[0][1][1]
    print(len(boxes))
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
        print("fields:")
        print(list(fields.values()))
        text = list(fields.values())[i]

        # Skip if text is None
        if text is not None:
            position = (shifted_coordinate[0][0], shifted_coordinate[0][1])
            draw.text(position, text, font=font, fill=(0, 0, 0))
        else:
            # Handle the case where there are more boxes than fields
            print(f"Warning: No field value for box index {i}")

    image.save('output2.png')

@app.post("/api/fill_box_form")
async def fill_form_box(file: UploadFile = File(...), name: str = Form(...), email: str = Form(...)):
    upload_dir = './uploaded_files/form'
    os.makedirs(upload_dir, exist_ok=True)  # Create directory if not exists
    file_location = os.path.join(upload_dir, file.filename)
    
    # Save file locally
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Query the database to get the data
    username = name.replace(" ", "_")  # Replace spaces with underscores for the collection name
    print(username)
    collection = get_db_collection(email)
    # print(collection)
    print(email)
    data = collection.find_one({"email": email})
    print("collection size")
    col_size = len(data)
    print("collection Size",col_size)

    
    if not data:
        raise HTTPException(status_code=404, detail="Data not found for the provided email")

    # Extract fields from OCR result
    form_data = data["content"][0]



    image = cv2.imread(file_location)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create kernels for horizontal and vertical line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Detect horizontal lines
    detected_horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines
    detected_vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine detected lines
    detected_lines_combined = cv2.add(detected_horizontal_lines, detected_vertical_lines)

    edges = cv2.Canny(detected_lines_combined, 50, 150)

    # plt.imshow(edges)
    # plt.axis('off')
    # plt.show()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    # plt.imshow(edges)
    # plt.axis('off')
    # plt.show()

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_height = 40
    max_height = 200
    min_width = 50
    cntrRect = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is rectangle
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if min_height < h < max_height and w > min_width:
                # cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
                cntrRect.append((x, y, w, h))
    filteredCntrRect = []
    detected_text = []
    skip = False
    # sorted_cntrRect = sort_rectangles(cntrRect)
    for i in range(len(cntrRect)-1):
        if(skip):
            skip = False
            continue
        (x, y, w, h) = cntrRect[i]
        (x1, y1, w1, h1) = cntrRect[i+1]
        if(x-50<x1+w1 and abs(y -y1)<20 and (x-x1)>0 ):
            filteredCntrRect.append((x1,y1,w+x-x1,h))
            skip = True
            print("Combining:")
            print(f"({x},{y}), ({x1} {y1})")
            # print("=======================")
            print("After:", filteredCntrRect[-1])
        else:
            print(f"({x},{y}), ({x1} {y1})")
            filteredCntrRect.append(cntrRect[i])

    cntrRect=[]
    skip = False
    for i in range(len(filteredCntrRect)-1):
        if(skip):
            skip = False
            continue
        (x, y, w, h) = filteredCntrRect[i]
        (x1, y1, w1, h1) = filteredCntrRect[i+1]
        if(x-50<x1+w1 and abs(y -y1)<20 and (x-x1)>0 ):
            cntrRect.append((x1,y1,w+x-x1,h))
            skip = True
            # print("Combining:")
            # print(f"({x},{y}), ({x1} {y1})")
            # print("=======================")
            # print("After:", cntrRect[-1])
        else:
            # print(f"({x},{y}), ({x1} {y1})")
            cntrRect.append(filteredCntrRect[i])


    for i in cntrRect:
        cv2.rectangle(image, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0, 255, 0), 3)
        print(i)
    # sorted_cntrRect = sorted(cntrRect, key=lambda rect: (rect[1] + rect[3] / 2, rect[0] + rect[2] / 2))
    # for (x, y, w, h) in cntrRect:
    #     test = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    #     wm = plt.get_current_fig_manager()
    #     wm.window.state('zoomed')
    #     plt.imshow(test)
    #     plt.axis('off')
    #     plt.show()
    left_offset = 625
    for (x, y, w, h) in cntrRect:
        try:
            # Define regions for text detection
            left_region = gray[y-20:y+20+h, max(x-left_offset, 0):x]  # Left of contour
            top_region = gray[max(y-100, 0):y, x:x+w]  # Above contour
            cut_region = 0

            # Perform OCR on each region with confidence
            left_data = pytesseract.image_to_data(left_region, config='--psm 6', output_type=pytesseract.Output.DICT)
            top_data = pytesseract.image_to_data(top_region, config='--psm 6', output_type=pytesseract.Output.DICT)

            left_confidence = max(left_data['conf'], default=-1)
            top_confidence = max(top_data['conf'], default=-1)
            confidence = max(left_confidence, top_confidence)
            if(confidence>70):
                if left_confidence>80 or (left_confidence > top_confidence and left_confidence != -1):
                    left_text = " ".join([left_data['text'][i] for i in range(len(left_data['text'])) if int(left_data['conf'][i]) > 0]).strip()
                    while '|' in left_text or '[' in left_text or ']' in left_text:
                        left_data = pytesseract.image_to_data(gray[y:y+h, max(x-700+cut_region, 0):x], config='--psm 6', output_type=pytesseract.Output.DICT)
                        left_text = " ".join([left_data['text'][i] for i in range(len(left_data['text'])) if int(left_data['conf'][i]) > 0]).strip()
                        cut_region += 25
                    detected_text.append({"position": "left", "text": left_text, "confidence": left_confidence, "bounding_box": (x, y, w, h)})
                    cv2.rectangle(image, (max(x - left_offset + cut_region, 0), y-20), (x, y + h+20), (255, 0, 0), 3)
                    cut_region = 0
                elif top_confidence != -1:
                    top_text = " ".join([top_data['text'][i] for i in range(len(top_data['text'])) if int(top_data['conf'][i]) > 0]).strip()
                    detected_text.append({"position": "top", "text": top_text, "confidence": top_confidence, "bounding_box": (x, y, w, h)})
                    cv2.rectangle(image, (x, max(y - 100, 0)), (x + w, y), (255, 0, 0), 3)
        except Exception as e:
            print(f"Error processing rectangle {x, y, w, h}: {e}")
    for i in detected_text:
        print(i)
    filteredText = []
    for i in detected_text:
        try:
            if ":" == i['text'][-1]:
                i["text"] = i["text"][:-1]
        except Exception as e:
            print("The error is: ", e)
        if '/' in i["text"]:
            i["text"].split('/')
        else:
            l = i["text"].split()
        factor = False
        for k in l:
            if(dictionary.check(k)):
                factor = True
        if(factor):
            if "other" in i["text"].lower():
                continue
            else:
                filteredText.append(i)
    fields=[]
    for i in filteredText:
        fields.append(i['text'])
    return fields
    # form_data = predict_and_update_keys(form_data)

    # print("fields")
    # print(fields)

    # Parse the OCR result to get bounding boxes
    # boxes, txts, scores = parse_result(result)
    # print("formdata:")
    # print(form_data)

#  Assign values from fields to form_data
    # for key, value in fields.items():
    #     if key in form_data:
    #         fields[key] = form_data[key]
    
    # print("After filled:")
    # print(fields)

    # print("len of boxes", len(boxes), boxes)
    # print("len of fields", len(fields), fields)
    # # Fill the form with the extracted fields
    # fill_form(fields, boxes, file_location)

    # Return the filled form as a response
    # return FileResponse('output2.png', media_type='image/png', filename='output2.png')