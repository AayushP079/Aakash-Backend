from pymongo import MongoClient
from config import MONGO_URI

# MongoDB connection setup
client = MongoClient(MONGO_URI)  # Replace with your MongoDB URI if using a remote DB
db = client['aakash_db']

def get_db_collection(collection_name: str):
    return db[collection_name]

