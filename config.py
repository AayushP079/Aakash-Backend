# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DEBUG = os.getenv("DEBUG") == "True"