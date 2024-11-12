from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/scrape", response_model=ScrapeResponse)
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