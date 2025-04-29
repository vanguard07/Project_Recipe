from pydantic import BaseModel, Field
from typing import List, Optional
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, info):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid ObjectId')
        return str(v)

class Recipe(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    title: str
    ingredients: List[str]
    instructions: List[str]
    cuisine: Optional[str] = None
    meal_type: Optional[str] = None
    prep_time_in_mins: Optional[int] = None
    cook_time_in_mins: Optional[int] = None
    tags: Optional[List[str]] = []
    source_url: Optional[str] = None
    estimated_calories: Optional[float] = None
    nutrients_present: Optional[List[str]] = []

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}

class ChatHistory(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    messages: List[dict]