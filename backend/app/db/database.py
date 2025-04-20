from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = AsyncIOMotorClient(settings.MONGODB_URL)
db = client[settings.DATABASE_NAME]

recipe_collection = db.get_collection("recipes")

chat_history_collection = db.get_collection("chat_history")