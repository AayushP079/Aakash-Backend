import os
from typing import Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
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
    

    # Prepare the data to be saved in MongoDB
    username = name.replace(" ", "_")  # Replace spaces with underscores for the collection name
    unique_number = random.randint(1000, 9999)  # Generate a random number
    collection_name = f"{username}_{unique_number}"

     # Prepare data to be saved in MongoDB
    data_dict = {
        "name": name,
        "email": email,
        "content": response.text,
        "uploaded_at": datetime.utcnow()
    }

    # Save data into a dynamic collection
    collection = get_db_collection(collection_name)
    collection.insert_one(data_dict)


    # remove current file after extracting data
    os.remove(file_location)

    return JSONResponse(content={"data": response.text, "status": "Data saved to database."})